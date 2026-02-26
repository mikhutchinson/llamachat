import SwiftUI
import Foundation
import LlamaInferenceCore
import SwiftPythonRuntime
import ChatStorage
import OSLog

enum ChatResponseMode: String, CaseIterable, Sendable {
    case auto
    case instant
    case thinking

    var title: String {
        switch self {
        case .auto: return "Auto"
        case .instant: return "Instant"
        case .thinking: return "Thinking"
        }
    }

    var subtitle: String {
        switch self {
        case .auto: return "Decides how long to think"
        case .instant: return "Answers right away"
        case .thinking: return "Thinks longer for better answers"
        }
    }
}

enum MentionAssetKind: String, Sendable, CaseIterable {
    case docs
    case img
}

struct ComposerMentionSuggestion: Identifiable, Sendable, Equatable {
    let kind: MentionAssetKind
    let alias: String
    let filename: String

    var id: String { "\(kind.rawValue):\(alias)" }
    var token: String { "@\(kind.rawValue)(\(alias))" }
}

/// Holds composer-specific state (input text, pending attachments) in a
/// separate `ObservableObject` so that keystroke-driven `objectWillChange`
/// notifications only invalidate the composer view — not the entire
/// `ContentView` body (which includes the heavyweight message list).
@MainActor
class ComposerState: ObservableObject {
    @Published var inputText = ""
    @Published var pendingAttachments: [PendingAttachment] = []
    @Published var activeMentionKind: MentionAssetKind?
    @Published var mentionSuggestions: [ComposerMentionSuggestion] = []
    @Published var inlineWarning: String?
}

/// Holds the currently streaming assistant preview row outside persisted history.
@MainActor
class LiveAssistantState: ObservableObject {
    @Published var message: ChatMessage?
}

@MainActor
class ChatViewModel: ObservableObject {
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "ChatViewModel")
    private static let streamPreviewInterval: Duration = .milliseconds(50)
    private static let streamPreviewBoundaryScalars = CharacterSet(charactersIn: ".!?;:\n")
    private static let trailingMentionRegex = try? NSRegularExpression(
        pattern: #"(?:^|\s)@(docs|img)$"#,
        options: [.caseInsensitive]
    )
    private static let mentionTokenRegex = try? NSRegularExpression(
        pattern: #"@(docs|img)\(([^)]+)\)"#,
        options: [.caseInsensitive]
    )
    private static let attachmentRecallRegex = try? NSRegularExpression(
        pattern: #"\b(?:upload(?:ed|ing)?|attach(?:ed|ment)?|image|images|photo|photos|picture|pictures|screenshot|screenshots|csv|xlsx|spreadsheet|document|file|pdf|table|chart|earlier|previous|above)\b"#,
        options: [.caseInsensitive]
    )
    @Published var messages: [ChatMessage] = []
    let composerState = ComposerState()
    let liveAssistantState = LiveAssistantState()
    @Published var isGenerating = false
    @Published var isReady = false
    @Published var isLoading = false
    @Published var isLoadingMessages = false
    @Published var showError = false
    @Published var errorMessage = ""
    @Published var conversations: [Conversation] = []
    @Published var selectedConversationID: String?
    @Published var searchQuery = ""
    @Published var parentConversationTitle: String?
    @Published var discoveredModels: [DiscoveredModel] = []
    @Published var responseMode: ChatResponseMode
    @Published var temporaryChatEnabled = false
    @Published var codeActEnabled: Bool
    @Published private(set) var recentModelPaths: [String] = []
    static let maxAttachmentFileSize: Int64 = 50 * 1024 * 1024
    @Published var config: InferenceConfig {
        didSet {
            let summarizerChanged = config.summarizerModelPath != oldValue.summarizerModelPath
            if summarizerChanged, pool != nil {
                Task { await invalidatePool() }
            }
        }
    }

    /// Database statistics for the Chat & Memory settings panel.
    struct DbStats {
        var sizeBytes: Int64 = 0
        var walMode: String = "—"
        var integrityOk: Bool = true
        var conversationCount: Int = 0
        var messageCount: Int = 0
    }
    @Published var dbStats = DbStats()

    private var pool: InferenceWorkerPool?
    private var scheduler: InferenceScheduler?
    private let vlmMonitor = VLMIdleMonitor()
    /// Dedicated sandbox for executing user code blocks in a Python worker.
    private var pythonSandbox: PythonSandbox?
    /// Persistent session for Llama chat (workers 0..N-1). Cleared when pool invalidates or conversation switches.
    private var chatSessionID: SessionID?
    private var persistence: ChatPersistence?
    private var hasLoadedOnLaunch = false
    /// Tracks message IDs from the last load or save — used as a dirty flag to skip redundant writes.
    private var lastSavedMessageIDs: [UUID] = []
    /// Number of messages to load per page when paginating.
    private static let messagesPageSize = 50
    /// Whether older messages exist beyond what's currently loaded.
    @Published var hasMoreMessages = false

    /// Debounced save: cancels if the user switches again within 300 ms.
    private var pendingSaveTask: Task<Void, Never>?
    /// Snapshot of messages captured for the deferred save.
    private var pendingSaveMessages: [ChatMessage] = []
    private var pendingSaveConversationID: String?
    /// Active streamed decode consumer task. Cancelled by Stop button.
    private var activeStreamTask: Task<StreamCompletion, Error>?
    private var stopRequested = false
    private var warningClearTask: Task<Void, Never>?

    private struct SessionAttachmentReference: Sendable {
        let alias: String
        let kind: MentionAssetKind
        let filename: String
        let mimeType: String
        let extractedText: String?
    }

    private var sessionDocs: [String: SessionAttachmentReference] = [:]
    private var sessionImages: [String: SessionAttachmentReference] = [:]
    private var sessionDocAliasOrder: [String] = []
    private var sessionImageAliasOrder: [String] = []
    private static let recentModelPathLimit = 8

    init() {
        self.config = Self.configFromUserDefaults()
        self.discoveredModels = ModelDiscovery.scan()
        self.responseMode = ChatResponseMode(rawValue: UserDefaults.standard.string(forKey: SettingsKeys.responseMode) ?? "") ?? .auto
        self.codeActEnabled = UserDefaults.standard.object(forKey: SettingsKeys.codeActEnabled) != nil
            ? UserDefaults.standard.bool(forKey: SettingsKeys.codeActEnabled)
            : SettingsDefaults.codeActEnabled
        self.recentModelPaths = Self.loadRecentModelPaths()
        rememberRecentModelPath(config.modelPath)
    }

    private static func loadRecentModelPaths() -> [String] {
        let ud = UserDefaults.standard
        let raw = ud.string(forKey: SettingsKeys.recentModelPaths) ?? ""
        guard !raw.isEmpty,
              let data = raw.data(using: .utf8),
              let decoded = try? JSONDecoder().decode([String].self, from: data) else {
            return []
        }
        return decoded.filter { !$0.isEmpty }
    }

    private func persistRecentModelPaths() {
        guard let data = try? JSONEncoder().encode(recentModelPaths),
              let json = String(data: data, encoding: .utf8) else {
            return
        }
        UserDefaults.standard.set(json, forKey: SettingsKeys.recentModelPaths)
    }

    private func rememberRecentModelPath(_ path: String) {
        guard !path.isEmpty else { return }
        recentModelPaths.removeAll { $0 == path }
        recentModelPaths.insert(path, at: 0)
        if recentModelPaths.count > Self.recentModelPathLimit {
            recentModelPaths = Array(recentModelPaths.prefix(Self.recentModelPathLimit))
        }
        persistRecentModelPaths()
    }

    func setResponseMode(_ mode: ChatResponseMode) {
        guard responseMode != mode else { return }
        responseMode = mode
        UserDefaults.standard.set(mode.rawValue, forKey: SettingsKeys.responseMode)
    }

    func setCodeActEnabled(_ enabled: Bool) {
        guard codeActEnabled != enabled else { return }
        codeActEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: SettingsKeys.codeActEnabled)
        if enabled {
            chatSessionID = nil
        }
    }

    func setTemporaryChatEnabled(_ enabled: Bool) async {
        guard temporaryChatEnabled != enabled else { return }
        temporaryChatEnabled = enabled
        await newChat()
    }

    func resetContextWindow() async {
        guard !isGenerating else { return }
        if let sessionID = chatSessionID {
            do {
                try await scheduler?.evictSession(sessionID)
            } catch {
                logger.warning("Failed to evict context session: \(String(describing: error), privacy: .public)")
            }
        }
        chatSessionID = nil
    }

    var canResetContextWindow: Bool {
        chatSessionID != nil && !isGenerating
    }

    /// Whether there's an active session that can be reset (used to show/hide the Reset Context button)
    var hasSessionToReset: Bool {
        chatSessionID != nil
    }

    func composerInputDidChange(_ text: String) {
        updateMentionTrigger(text: text)
    }

    func showMentionPicker(kind: MentionAssetKind) {
        let suggestions = mentionSuggestions(for: kind)
        guard !suggestions.isEmpty else {
            showInlineWarning(kind == .img ? "No captioned images in this session yet." : "No documents in this session yet.")
            dismissMentionPicker()
            return
        }
        composerState.activeMentionKind = kind
        composerState.mentionSuggestions = suggestions
    }

    func dismissMentionPicker() {
        composerState.activeMentionKind = nil
        composerState.mentionSuggestions = []
    }

    func selectMentionSuggestion(_ suggestion: ComposerMentionSuggestion) {
        insertMentionToken(suggestion.token)
        dismissMentionPicker()
    }

    private func updateMentionTrigger(text: String) {
        guard let kind = trailingMentionKind(in: text) else {
            if composerState.activeMentionKind != nil {
                dismissMentionPicker()
            }
            return
        }
        composerState.activeMentionKind = kind
        composerState.mentionSuggestions = mentionSuggestions(for: kind)
    }

    private func trailingMentionKind(in text: String) -> MentionAssetKind? {
        guard let regex = Self.trailingMentionRegex else { return nil }
        let nsText = text as NSString
        let range = NSRange(location: 0, length: nsText.length)
        guard let match = regex.firstMatch(in: text, options: [], range: range) else { return nil }
        guard match.range.location + match.range.length == nsText.length else { return nil }
        guard let kindRange = Range(match.range(at: 1), in: text) else { return nil }
        return MentionAssetKind(rawValue: text[kindRange].lowercased())
    }

    private func insertMentionToken(_ token: String) {
        var text = composerState.inputText
        if let regex = Self.trailingMentionRegex {
            let nsText = text as NSString
            let range = NSRange(location: 0, length: nsText.length)
            if let match = regex.firstMatch(in: text, options: [], range: range),
               match.range.location + match.range.length == nsText.length,
               let swiftRange = Range(match.range, in: text) {
                let matched = String(text[swiftRange])
                let leadingWhitespace = matched.first.map { String($0).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty } == true ? " " : ""
                text.replaceSubrange(swiftRange, with: "\(leadingWhitespace)\(token) ")
                composerState.inputText = text
                return
            }
        }

        if text.isEmpty || text.hasSuffix(" ") || text.hasSuffix("\n") {
            text += token
        } else {
            text += " \(token)"
        }
        text += " "
        composerState.inputText = text
    }

    private func mentionSuggestions(for kind: MentionAssetKind) -> [ComposerMentionSuggestion] {
        switch kind {
        case .docs:
            return sessionDocAliasOrder.compactMap { alias in
                guard let ref = sessionDocs[alias] else { return nil }
                return ComposerMentionSuggestion(kind: .docs, alias: ref.alias, filename: ref.filename)
            }
        case .img:
            return sessionImageAliasOrder.compactMap { alias in
                guard let ref = sessionImages[alias] else { return nil }
                let caption = ref.extractedText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                guard !caption.isEmpty else { return nil }
                return ComposerMentionSuggestion(kind: .img, alias: ref.alias, filename: ref.filename)
            }
        }
    }

    private func clearSessionAttachmentRegistry() {
        sessionDocs.removeAll()
        sessionImages.removeAll()
        sessionDocAliasOrder.removeAll()
        sessionImageAliasOrder.removeAll()
        dismissMentionPicker()
        warningClearTask?.cancel()
        composerState.inlineWarning = nil
    }

    private func makeAlias(for filename: String, kind: MentionAssetKind) -> String {
        let existing: [SessionAttachmentReference]
        switch kind {
        case .docs:
            existing = Array(sessionDocs.values)
        case .img:
            existing = Array(sessionImages.values)
        }
        let duplicateCount = existing.filter { $0.filename == filename }.count
        if duplicateCount == 0 {
            return filename
        }
        return "\(filename) #\(duplicateCount + 1)"
    }

    private func registerSessionAttachmentReference(
        kind: MentionAssetKind,
        filename: String,
        mimeType: String,
        extractedText: String?
    ) {
        let alias = makeAlias(for: filename, kind: kind)
        let ref = SessionAttachmentReference(
            alias: alias,
            kind: kind,
            filename: filename,
            mimeType: mimeType,
            extractedText: extractedText
        )
        switch kind {
        case .docs:
            sessionDocs[alias] = ref
            sessionDocAliasOrder.append(alias)
        case .img:
            sessionImages[alias] = ref
            sessionImageAliasOrder.append(alias)
        }
        if composerState.activeMentionKind == kind {
            composerState.mentionSuggestions = mentionSuggestions(for: kind)
        }
    }

    private func registerSessionAttachments(_ attachments: [MessageAttachment]) {
        for att in attachments {
            switch att.type {
            case .textFile, .pdf:
                registerSessionAttachmentReference(
                    kind: .docs,
                    filename: att.filename,
                    mimeType: att.mimeType,
                    extractedText: att.extractedText
                )
            case .image:
                registerSessionAttachmentReference(
                    kind: .img,
                    filename: att.filename,
                    mimeType: att.mimeType,
                    extractedText: att.extractedText
                )
            }
        }
    }

    private func repopulateSessionAttachments(from messages: [ChatMessage]) {
        clearSessionAttachmentRegistry()
        for msg in messages {
            registerSessionAttachments(msg.attachments)
        }
    }

    private func repopulateSessionAttachments(for conversationID: String) async {
        clearSessionAttachmentRegistry()
        guard let persistence else { return }
        do {
            let refs = try await persistence.loadAttachmentReferences(for: conversationID)
            for ref in refs {
                switch ref.type {
                case .textFile, .pdf:
                    registerSessionAttachmentReference(
                        kind: .docs,
                        filename: ref.filename,
                        mimeType: ref.mimeType,
                        extractedText: ref.extractedText
                    )
                case .image:
                    registerSessionAttachmentReference(
                        kind: .img,
                        filename: ref.filename,
                        mimeType: ref.mimeType,
                        extractedText: ref.extractedText
                    )
                }
            }
        } catch {
            logger.error("Failed to repopulate session attachments: \(Self.describeError(error), privacy: .public)")
        }
    }

    private func showInlineWarning(_ text: String, clearAfter seconds: Double = 4.0) {
        composerState.inlineWarning = text
        warningClearTask?.cancel()
        warningClearTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard let self, !Task.isCancelled else { return }
            self.composerState.inlineWarning = nil
        }
    }

    // MARK: - Persistence

    func loadOnLaunch() async {
        guard !hasLoadedOnLaunch else { return }
        hasLoadedOnLaunch = true
        refreshDiscoveredModels()

        // Open database (async — runs migrations off main thread)
        let dbPathOverride = UserDefaults.standard.string(forKey: SettingsKeys.chatDbPathOverride)
            .flatMap { $0.isEmpty ? nil : $0 }
        do {
            self.persistence = try await ChatPersistence.open(dbPath: dbPathOverride)
        } catch {
            logger.error("Failed to open chat database: \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to open chat database: \(String(describing: error))")
            return
        }

        guard let persistence else { return }
        do {
            conversations = try await persistence.loadConversations()
            if let savedID = UserDefaults.standard.string(forKey: "selectedConversationID"),
               conversations.contains(where: { $0.id == savedID }) {
                selectedConversationID = savedID
                let totalCount = try await persistence.countMessages(for: savedID)
                messages = try await persistence.loadMessagesPage(for: savedID, limit: Self.messagesPageSize)
                lastSavedMessageIDs = messages.map(\.id)
                hasMoreMessages = totalCount > messages.count
                await repopulateSessionAttachments(for: savedID)
            }
        } catch {
            logger.error("Failed to load conversations: \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to load conversations: \(String(describing: error))")
        }

        await refreshDbStats()

        // Auto-load the saved model on launch
        if !config.modelPath.isEmpty,
           FileManager.default.fileExists(atPath: config.modelPath) {
            await loadModel()
        }
    }

    func refreshDiscoveredModels() {
        discoveredModels = ModelDiscovery.scan()
    }

    func selectPrimaryModel(path: String) async {
        UserDefaults.standard.set(path, forKey: SettingsKeys.modelPath)
        rememberRecentModelPath(path)
        config = Self.configFromUserDefaults()
        await loadModel()
    }

    /// Re-read config from UserDefaults (e.g. after Hub "Use as" writes a new
    /// model path). Triggers model load only when the primary model changed.
    func syncConfigFromDefaults() {
        let fresh = Self.configFromUserDefaults()
        let pathChanged = fresh.modelPath != config.modelPath
        config = fresh
        refreshDiscoveredModels()
        if pathChanged {
            rememberRecentModelPath(fresh.modelPath)
            Task { await loadModel() }
        }
    }

    func newChat() async {
        guard !isGenerating else { return }
        if !messages.isEmpty, let convID = selectedConversationID {
            deferSave(id: convID, messages: messages, savedIDs: lastSavedMessageIDs)
        }
        messages.removeAll()
        lastSavedMessageIDs = []
        chatSessionID = nil
        parentConversationTitle = nil
        clearSessionAttachmentRegistry()
        setSelectedConversation(nil)
        await refreshConversations()
    }

    func selectConversation(id: String) async {
        guard !isGenerating, id != selectedConversationID else { return }
        // Defer save of the previous conversation (runs in parallel with the load below).
        if !messages.isEmpty, let currentID = selectedConversationID {
            deferSave(id: currentID, messages: messages, savedIDs: lastSavedMessageIDs)
        }
        // Optimistic switch: update selection and show loading state immediately.
        chatSessionID = nil
        parentConversationTitle = nil
        clearSessionAttachmentRegistry()
        setSelectedConversation(id)
        messages = []
        lastSavedMessageIDs = []
        isLoadingMessages = true
        defer { isLoadingMessages = false }

        guard let persistence else { return }
        do {
            let totalCount = try await persistence.countMessages(for: id)
            let loaded = try await persistence.loadMessagesPage(for: id, limit: Self.messagesPageSize)
            // Guard against stale results if the user switched again while we were loading.
            guard selectedConversationID == id else { return }
            messages = loaded
            lastSavedMessageIDs = loaded.map(\.id)
            hasMoreMessages = totalCount > loaded.count
            await resolveParentTitle()
            await repopulateSessionAttachments(for: id)
        } catch {
            logger.error("Failed to load messages for \(id, privacy: .public): \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to load messages for \(id): \(String(describing: error))")
            messages = []
            lastSavedMessageIDs = []
            hasMoreMessages = false
            clearSessionAttachmentRegistry()
        }
    }

    /// Load older messages (previous page) and prepend them to the current list.
    func loadMoreMessages() async {
        guard let persistence, let convID = selectedConversationID, hasMoreMessages else { return }
        // Determine the oldest sortOrder we currently have.
        // Messages from DB are loaded in sortOrder ASC, so index 0 is the oldest.
        // We need to know the sortOrder of the first loaded message to paginate.
        // Since we don't store sortOrder in ChatMessage, use the count of total messages
        // minus what we have loaded to compute the offset.
        do {
            let totalCount = try await persistence.countMessages(for: convID)
            let currentCount = messages.count
            let remaining = totalCount - currentCount
            guard remaining > 0 else {
                hasMoreMessages = false
                return
            }
            // Load the next page: we need messages with sortOrder < (totalCount - currentCount)
            // Since sortOrder is 0-based index, the oldest loaded message has sortOrder = totalCount - currentCount
            let oldestSortOrder = totalCount - currentCount
            let page = try await persistence.loadMessagesPage(
                for: convID,
                limit: Self.messagesPageSize,
                beforeSortOrder: oldestSortOrder
            )
            guard selectedConversationID == convID else { return }
            messages.insert(contentsOf: page, at: 0)
            // Update lastSavedMessageIDs to include the newly loaded messages
            lastSavedMessageIDs = Array(page.map(\.id)) + lastSavedMessageIDs
            hasMoreMessages = page.count == Self.messagesPageSize && (oldestSortOrder - page.count) > 0
        } catch {
            logger.error("Failed to load more messages: \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to load more messages: \(String(describing: error))")
        }
    }

    func deleteConversation(id: String) async {
        guard let persistence else { return }
        do {
            try await persistence.deleteConversation(id: id)
            if selectedConversationID == id {
                messages.removeAll()
                lastSavedMessageIDs = []
                chatSessionID = nil
                clearSessionAttachmentRegistry()
                setSelectedConversation(nil)
            }
            await refreshConversations()
        } catch {
            logger.error("Failed to delete conversation: \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to delete conversation: \(String(describing: error))")
        }
    }

    func renameConversation(id: String, newTitle: String) async {
        guard let persistence else { return }
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        do {
            try await persistence.updateConversationTitle(id: id, title: trimmed)
            await refreshConversations()
        } catch {
            logger.error("Failed to rename conversation: \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to rename conversation: \(String(describing: error))")
        }
    }

    func exportConversation(id: String, to url: URL) async throws {
        guard let persistence else { throw ChatPersistenceError.openFailed("No persistence") }
        let messages = try await persistence.loadMessages(for: id)
        let title = (try? await persistence.loadConversation(id: id))?.title ?? "Exported Chat"

        var markdown = "# \(title)\n\n"
        for msg in messages {
            let role = msg.role == .user ? "User" : "Assistant"
            markdown += "## \(role)\n\n"
            markdown += msg.content
            if let thinking = msg.thinking, !thinking.isEmpty {
                markdown += "\n\n<details><summary>Thinking</summary>\n\n\(thinking)\n\n</details>"
            }
            markdown += "\n\n"
        }

        try markdown.write(to: url, atomically: true, encoding: .utf8)
    }

    func refreshConversations() async {
        guard let persistence else { return }
        do {
            let query = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
            if query.isEmpty {
                conversations = try await persistence.loadConversations()
            } else {
                conversations = try await persistence.search(query: query)
            }
        } catch {
            logger.error("Failed to refresh conversations: \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to refresh conversations: \(String(describing: error))")
        }
    }

    private func saveCurrentConversation(id: String) async {
        guard let persistence, !messages.isEmpty else { return }
        // Skip save when messages haven't changed since last load/save (dirty flag).
        let currentIDs = messages.map(\.id)
        guard currentIDs != lastSavedMessageIDs else { return }
        let title = messages.first { $0.role == .user }
            .map { String($0.content.prefix(40)) } ?? "New Chat"
        do {
            let existingSet = Set(lastSavedMessageIDs)
            if existingSet.isEmpty {
                // First save — use full replace (no existing messages to diff against).
                try await persistence.saveConversation(id: id, title: title, messages: messages)
            } else {
                // Incremental save — only insert new / delete removed messages.
                try await persistence.saveConversationIncremental(
                    id: id,
                    title: title,
                    messages: messages,
                    existingMessageIDs: existingSet
                )
            }
            lastSavedMessageIDs = currentIDs
            scheduleSemanticNamingIfNeeded(conversationID: id)
        } catch {
            logger.error("Failed to save conversation: \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to save conversation: \(String(describing: error))")
        }
    }

    /// Schedule a deferred save for the given conversation. Cancels any pending save so only the
    /// latest snapshot is written. The actual write happens after a 300 ms debounce window.
    private func deferSave(id: String, messages snapshot: [ChatMessage], savedIDs: [UUID]) {
        pendingSaveTask?.cancel()
        pendingSaveConversationID = id
        pendingSaveMessages = snapshot
        let currentIDs = snapshot.map(\.id)
        // Skip if unchanged since last save.
        guard currentIDs != savedIDs else {
            pendingSaveConversationID = nil
            pendingSaveMessages = []
            return
        }
        pendingSaveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled, let self else { return }
            await self.executeDeferredSave()
        }
    }

    /// Perform the actual deferred save using the captured snapshot.
    private func executeDeferredSave() async {
        guard let persistence, let id = pendingSaveConversationID, !pendingSaveMessages.isEmpty else { return }
        let title = pendingSaveMessages.first { $0.role == .user }
            .map { String($0.content.prefix(40)) } ?? "New Chat"
        do {
            try await persistence.saveConversation(id: id, title: title, messages: pendingSaveMessages)
            scheduleSemanticNamingIfNeeded(conversationID: id)
        } catch {
            logger.error("Failed to flush deferred save: \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to flush deferred save: \(String(describing: error))")
        }
        pendingSaveConversationID = nil
        pendingSaveMessages = []
    }

    /// Immediately flush any pending deferred save (e.g. before app terminate).
    func flushPendingSave() async {
        pendingSaveTask?.cancel()
        pendingSaveTask = nil
        await executeDeferredSave()
    }

    /// Fire-and-forget: generate semantic title in background, update DB, refresh sidebar.
    /// Never blocks the main thread. Graceful no-op if summarizer unavailable.
    private func scheduleSemanticNamingIfNeeded(conversationID id: String) {
        guard let persistence, let scheduler else { return }

        Task(priority: .utility) { [weak self] in
            guard let self else { return }
            do {
                let messages = try await persistence.loadMessages(for: id)
                guard messages.count >= 2,
                      messages.contains(where: { $0.role == .assistant }) else { return }

                let sessionHistory = messages.map { msg in
                    ["role": msg.role == .user ? "user" : "assistant", "content": msg.content]
                }
                guard let memory = await scheduler.narrativeMemory else { return }
                let title = try await memory.suggestTitle(sessionHistory: sessionHistory)
                guard !title.isEmpty else { return }

                try await persistence.updateConversationTitle(id: id, title: title)
                await self.refreshConversations()
            } catch {
                // Non-fatal — keep truncated title
            }
        }
    }

    private func setSelectedConversation(_ id: String?) {
        selectedConversationID = id
        if let id {
            UserDefaults.standard.set(id, forKey: "selectedConversationID")
        } else {
            UserDefaults.standard.removeObject(forKey: "selectedConversationID")
        }
    }

    /// Refresh database statistics for the settings panel.
    func refreshDbStats() async {
        guard let persistence else { return }
        do {
            let convCount = try await persistence.countConversations()
            let msgCount = try await persistence.countMessages()
            let walMode = try await persistence.journalMode()
            let integrity = try await persistence.checkIntegrity()

            // Compute file sizes on disk (db + wal + shm)
            let dbPathOverride = UserDefaults.standard.string(forKey: SettingsKeys.chatDbPathOverride)
                .flatMap { $0.isEmpty ? nil : $0 }
            let dbPath: String
            if let override = dbPathOverride {
                dbPath = override
            } else {
                let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                dbPath = appSupport.appendingPathComponent("LlamaInferenceDemo/chat.db").path
            }
            let fm = FileManager.default
            var totalBytes: Int64 = 0
            for suffix in ["", "-wal", "-shm"] {
                let path = dbPath + suffix
                if let attrs = try? fm.attributesOfItem(atPath: path),
                   let size = attrs[.size] as? Int64 {
                    totalBytes += size
                }
            }

            dbStats = DbStats(
                sizeBytes: totalBytes,
                walMode: walMode,
                integrityOk: integrity,
                conversationCount: convCount,
                messageCount: msgCount
            )
        } catch {
            logger.error("Failed to refresh DB stats: \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to refresh DB stats: \(String(describing: error))")
        }
    }

    // MARK: - Model Loading

    private var reloadModelTask: Task<Void, Error>?

    func reloadModelDebounced() {
        guard pool != nil else { return }
        reloadModelTask?.cancel()
        reloadModelTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled, let self else { return }

            while self.isGenerating {
                try? await Task.sleep(for: .milliseconds(200))
                guard !Task.isCancelled else { return }
            }

            await self.loadModel()
        }
    }

    func loadModel() async {
        guard !config.modelPath.isEmpty else {
            errorMessage = "Choose a model in Settings (⌘,)"
            showError = true
            return
        }

        isLoading = true
        isReady = false
        defer { isLoading = false }

        do {
            self.scheduler = nil
            self.chatSessionID = nil
            await vlmMonitor.shutdown()
            if let pool = pool {
                await pool.shutdown()
                self.pool = nil
            }

            let newPool = InferenceWorkerPool(config: config)
            try await newPool.startup()

            let newScheduler = InferenceScheduler(
                workerPool: newPool,
                config: config
            )

            self.pool = newPool
            self.scheduler = newScheduler

            // Install narrative memory for context wind management
            do {
                try await newScheduler.installMemoryGraph()
                logger.debug("Narrative memory installed")
            } catch {
                let errorDesc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
                logger.warning("Narrative memory installation failed (non-fatal): \(errorDesc, privacy: .public)")
                await FileLogger.shared.log(level: .warning, category: "ChatViewModel", message: "Narrative memory installation failed (non-fatal): \(errorDesc)")
            }

            // Configure VLM idle monitor (non-fatal)
            await configureVLM(pool: newPool)

            // Configure Python sandbox for code block execution (non-fatal)
            await configureSandbox(pool: newPool)

            // Preserve mention aliases across model/context reloads. If registry was
            // empty (e.g. cold start), rebuild from current conversation data.
            if sessionDocs.isEmpty && sessionImages.isEmpty {
                if let selectedConversationID {
                    await repopulateSessionAttachments(for: selectedConversationID)
                } else if !messages.isEmpty {
                    repopulateSessionAttachments(from: messages)
                }
            }

            self.isReady = true
            await FileLogger.shared.log(level: .info, category: "ChatViewModel", message: "Model loaded: \(config.modelPath)")
        } catch {
            let detailedError = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
            errorMessage = "Failed to load model: \(detailedError)"
            showError = true
            isReady = false
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to load model: \(detailedError)")
        }
    }

    func clearChat() {
        messages.removeAll()
        clearSessionAttachmentRegistry()
    }

    private static func configFromUserDefaults() -> InferenceConfig {
        let ud = UserDefaults.standard
        let workerCount = ud.integer(forKey: SettingsKeys.workerCount).clamped(to: 1...) ?? SettingsDefaults.workerCount
        return InferenceConfig(
            modelPath: ud.string(forKey: SettingsKeys.modelPath) ?? SettingsDefaults.modelPath,
            summarizerModelPath: ud.string(forKey: SettingsKeys.summarizerModelPath).flatMap { $0.isEmpty ? nil : $0 },
            contextSize: ud.integer(forKey: SettingsKeys.contextSize).clamped(to: 512...) ?? SettingsDefaults.contextSize,
            nGpuLayers: ud.object(forKey: SettingsKeys.nGpuLayers) != nil ? ud.integer(forKey: SettingsKeys.nGpuLayers) : SettingsDefaults.nGpuLayers,
            workerCount: workerCount,
            maxSessionsPerWorker: 8,
            maxInFlight: max(1, workerCount) * 4,
            blasThreads: 1,
            useSharedMemory: ud.object(forKey: SettingsKeys.useSharedMemory) != nil ? ud.bool(forKey: SettingsKeys.useSharedMemory) : SettingsDefaults.useSharedMemory,
            venvPath: discoverVenvPath()
        )
    }

    static func discoverVenvPath(sourceFile: String = #filePath) -> String? {
        let fm = FileManager.default

        if let envVenv = ProcessInfo.processInfo.environment["VIRTUAL_ENV"],
           fm.fileExists(atPath: envVenv) {
            return envVenv
        }

        // Walk up from compile-time source path to repo root
        // sourceFile = .../SwiftPython/Demo/LlamaInferenceDemo/UI/ChatViewModel.swift
        var dir = (sourceFile as NSString).deletingLastPathComponent
        for _ in 0..<8 {
            let candidate = (dir as NSString).appendingPathComponent(".venv")
            if fm.fileExists(atPath: candidate) {
                return (candidate as NSString).standardizingPath
            }
            let parent = (dir as NSString).deletingLastPathComponent
            if parent == dir { break }
            dir = parent
        }

        // Fallback: check relative to cwd
        let cwd = fm.currentDirectoryPath
        for suffix in ["/.venv", "/../.venv", "/../../.venv"] {
            let candidate = (cwd + suffix as NSString).standardizingPath
            if fm.fileExists(atPath: candidate) {
                return candidate
            }
        }
        return nil
    }

    // MARK: - Attachments

    func addAttachment(url: URL) {
        let filename = url.lastPathComponent
        let ext = url.pathExtension.lowercased()

        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let fileSize = attrs[.size] as? Int64 else {
            errorMessage = "Could not read file: \(filename)"
            showError = true
            return
        }
        guard fileSize <= Self.maxAttachmentFileSize else {
            let mb = fileSize / (1024 * 1024)
            errorMessage = "File too large (\(mb) MB). Maximum is 50 MB."
            showError = true
            return
        }

        let imageExts = Set(["jpg", "jpeg", "png", "gif", "webp", "heic", "heif", "bmp", "tiff"])
        let documentExts = Set(["pdf", "docx", "pptx", "xlsx"])
        let type: AttachmentType
        if imageExts.contains(ext) {
            type = .image
        } else if documentExts.contains(ext) {
            type = .pdf
        } else {
            type = .textFile
        }

        // Append immediately with no thumbnail so the UI responds instantly.
        let pending = PendingAttachment(
            filename: filename,
            type: type,
            fileURL: url,
            fileSize: fileSize,
            thumbnailImage: nil
        )
        composerState.pendingAttachments.append(pending)

        // Generate thumbnail off the main actor to avoid beachball on large images.
        if type == .image {
            let pendingID = pending.id
            Task {
                let thumbnail = await Self.generateThumbnail(url: url, maxDim: 60)
                // Update the pending attachment in-place if it still exists.
                if let idx = composerState.pendingAttachments.firstIndex(where: { $0.id == pendingID }) {
                    composerState.pendingAttachments[idx] = PendingAttachment(
                        id: pendingID,
                        filename: filename,
                        type: type,
                        fileURL: url,
                        fileSize: fileSize,
                        thumbnailImage: thumbnail
                    )
                }
            }
        }
    }

    /// Generate a thumbnail off the main actor using CGImageSource (thread-safe).
    /// Uses kCGImageSourceCreateThumbnailFromImageAlways for a single-pass
    /// decode+scale — faster than loading the full NSImage and redrawing.
    private nonisolated static func generateThumbnail(url: URL, maxDim: CGFloat) async -> NSImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: Int(maxDim),
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        guard let cgThumb = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        return NSImage(cgImage: cgThumb, size: NSSize(width: cgThumb.width, height: cgThumb.height))
    }

    func removeAttachment(id: UUID) {
        composerState.pendingAttachments.removeAll { $0.id == id }
    }

    private func convertPendingToMessageAttachments(_ pending: [PendingAttachment]) async -> [MessageAttachment] {
        // Compute per-image caption token budget from the attachment context allocation.
        // Images share 30% of context with documents; give images up to 15% of context
        // split across all images, floored at 64 tokens and capped at 1024.
        // Higher ceiling needed for text-heavy images (infographics, screenshots, documents).
        let imageCount = pending.filter { $0.type == .image }.count
        let maxCaptionTokens: Int
        if imageCount > 0 {
            let imageBudgetTokens = Int(Double(config.contextSize) * 0.15) / imageCount
            maxCaptionTokens = min(max(imageBudgetTokens, 64), 1024)
        } else {
            maxCaptionTokens = 512
        }

        var result: [MessageAttachment] = []
        for item in pending {
            // Read file data off the main actor to avoid blocking the UI.
            let data = await Self.readFileData(url: item.fileURL)
            guard let data else { continue }
            let mimeType = Self.mimeType(for: item)

            var extractedText: String?
            if item.type == .textFile {
                extractedText = String(data: data, encoding: .utf8)
            } else if item.type == .pdf {
                extractedText = await extractDocumentText(fileURL: item.fileURL)
            } else if item.type == .image {
                extractedText = await captionImage(data: data, mimeType: mimeType, maxCaptionTokens: maxCaptionTokens)
            }

            var thumbData: Data?
            if let img = item.thumbnailImage,
               let tiff = img.tiffRepresentation,
               let rep = NSBitmapImageRep(data: tiff) {
                thumbData = rep.representation(using: .png, properties: [:])
            }

            result.append(MessageAttachment(
                type: item.type,
                filename: item.filename,
                mimeType: mimeType,
                data: data,
                extractedText: extractedText,
                thumbnailData: thumbData
            ))
        }
        return result
    }

    /// Read file contents off the main actor so large files don't block the UI.
    private nonisolated static func readFileData(url: URL) async -> Data? {
        try? Data(contentsOf: url)
    }

    private func configureVLM(pool: InferenceWorkerPool) async {
        let ud = UserDefaults.standard
        let vlmModel = ud.string(forKey: SettingsKeys.vlmModelPath) ?? ""
        let vlmClip = ud.string(forKey: SettingsKeys.vlmClipPath) ?? ""
        let vlmArchRaw = ud.string(forKey: SettingsKeys.vlmArchitecture)
        let vlmArch = (vlmArchRaw?.isEmpty == false) ? vlmArchRaw : nil
        
        let idleTimeout = ud.object(forKey: SettingsKeys.vlmIdleTimeoutSecs) != nil
            ? TimeInterval(ud.integer(forKey: SettingsKeys.vlmIdleTimeoutSecs))
            : TimeInterval(SettingsDefaults.vlmIdleTimeoutSecs)

        guard !vlmModel.isEmpty, !vlmClip.isEmpty else {
            logger.debug("VLM not configured (no model/clip paths)")
            return
        }

        do {
            let (handle, workerIdx) = try await pool.getVLMKernelHandle()
            let poolRef = try await pool.getPool()
            await vlmMonitor.configure(
                pool: poolRef,
                kernelHandle: handle,
                workerIndex: workerIdx,
                modelPath: vlmModel,
                clipPath: vlmClip,
                vlmArchitecture: vlmArch,
                contextSize: 2048,
                nGpuLayers: config.nGpuLayers,
                idleTimeoutSecs: idleTimeout
            )
            logger.debug("VLM configured: \(vlmModel, privacy: .public)")
        } catch {
            let desc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
            logger.warning("VLM configure failed (non-fatal): \(desc, privacy: .public)")
        }
    }

    private func captionImage(data: Data, mimeType: String, maxCaptionTokens: Int = 1024) async -> String? {
        guard await vlmMonitor.isConfigured else { return nil }

        guard await vlmMonitor.ensureLoaded() else {
            logger.warning("VLM failed to load for image captioning")
            return nil
        }

        // Convert to PNG — llama-cpp only supports JPEG/PNG, not HEIC/TIFF/WebP
        let pngData: Data
        if let nsImage = NSImage(data: data),
           let tiff = nsImage.tiffRepresentation,
           let bitmap = NSBitmapImageRep(data: tiff),
           let png = bitmap.representation(using: .png, properties: [:]) {
            pngData = png
        } else {
            logger.warning("VLM: failed to convert image to PNG")
            return nil
        }

        // Write PNG to temp file to avoid IPC payload limits
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("vlm_\(UUID().uuidString).png")
        do {
            try pngData.write(to: tmpURL)
        } catch {
            logger.warning("VLM: failed to write temp image: \(String(describing: error), privacy: .public)")
            return nil
        }
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        guard let pool else { return nil }
        do {
            let (handle, workerIdx) = try await pool.getVLMKernelHandle()
            let poolRef = try await pool.getPool()
            let result = try await VLMKernel.captionFile(
                pool: poolRef,
                workerIndex: workerIdx,
                kernelHandle: handle,
                filePath: tmpURL.path,
                mimeType: "image/png",
                prompt: "Describe this image in detail.",
                maxTokens: maxCaptionTokens
            )
            await vlmMonitor.markUsed()
            if result.succeeded {
                let formatted = result.formattedCaption()
                let isStructured = formatted != result.caption
                logger.debug("VLM caption: \(formatted.count, privacy: .public) chars, \(result.tokens, privacy: .public) tokens in \(result.durationMs, privacy: .public)ms (structured=\(isStructured, privacy: .public))")
                return formatted
            } else if let error = result.error {
                logger.warning("VLM caption error: \(error, privacy: .public)")
            }
        } catch {
            logger.warning("VLM caption failed: \(String(describing: error), privacy: .public)")
        }
        return nil
    }

    private func extractDocumentText(fileURL: URL) async -> String? {
        guard let pool else { return nil }

        // SHA256 cache check — skip re-extraction for same file content
        let fileData = await Self.readFileData(url: fileURL)
        if let fileData {
            let sha = ExtractionCache.sha256(of: fileData)
            if let cached = await ExtractionCache.shared.lookup(sha256: sha) {
                logger.debug("Cache hit for \(fileURL.lastPathComponent, privacy: .public) (sha256=\(sha.prefix(12), privacy: .public)…)")
                return cached
            }
        }

        do {
            let (handle, workerIdx) = try await pool.getDocExtractorHandle()
            let poolRef = try await pool.getPool()

            let ext = fileURL.pathExtension.lowercased()
            let isXLSX = ext == "xlsx"
            let isPDF = ext == "pdf"
            if isXLSX {
                let xlsxResult = try await DocumentExtractor.extractXlsx(
                    pool: poolRef,
                    workerIndex: workerIdx,
                    kernelHandle: handle,
                    filePath: fileURL.path
                )
                if xlsxResult.succeeded {
                    let sheetNames = xlsxResult.sheets.map(\.name).joined(separator: ", ")
                    logger.debug("Extracted XLSX [\(sheetNames, privacy: .public)] from \(fileURL.lastPathComponent, privacy: .public) in \(xlsxResult.durationMs, privacy: .public)ms")
                    let text = xlsxResult.combinedText
                    if let fileData { await ExtractionCache.shared.store(sha256: ExtractionCache.sha256(of: fileData), text: text) }
                    return text
                } else if let error = xlsxResult.error {
                    logger.warning("DocumentExtractor XLSX error for \(fileURL.lastPathComponent, privacy: .public): \(error, privacy: .public)")
                }
            } else if isPDF {
                let pageResult = try await DocumentExtractor.extractPages(
                    pool: poolRef,
                    workerIndex: workerIdx,
                    kernelHandle: handle,
                    filePath: fileURL.path
                )
                if pageResult.succeeded {
                    let scannedCount = pageResult.scannedPages.count
                    logger.debug("Extracted \(pageResult.totalPages, privacy: .public) pages from \(fileURL.lastPathComponent, privacy: .public) in \(pageResult.durationMs, privacy: .public)ms (\(scannedCount, privacy: .public) scanned)")
                    let text = pageResult.combinedText
                    if let fileData { await ExtractionCache.shared.store(sha256: ExtractionCache.sha256(of: fileData), text: text) }
                    return text
                } else if let error = pageResult.error {
                    logger.warning("DocumentExtractor page error for \(fileURL.lastPathComponent, privacy: .public): \(error, privacy: .public)")
                }
            } else {
                let result = try await DocumentExtractor.extractFile(
                    pool: poolRef,
                    workerIndex: workerIdx,
                    kernelHandle: handle,
                    filePath: fileURL.path
                )
                if result.succeeded {
                    logger.debug("Extracted \(result.chars, privacy: .public) chars from \(fileURL.lastPathComponent, privacy: .public) in \(result.durationMs, privacy: .public)ms")
                    if let fileData { await ExtractionCache.shared.store(sha256: ExtractionCache.sha256(of: fileData), text: result.text) }
                    return result.text
                } else if let error = result.error {
                    logger.warning("DocumentExtractor error for \(fileURL.lastPathComponent, privacy: .public): \(error, privacy: .public)")
                }
            }
        } catch {
            let desc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
            logger.warning("DocumentExtractor unavailable: \(desc, privacy: .public)")
        }
        return nil
    }

    private func buildAttachmentContext(attachments: [MessageAttachment], query: String) async -> String {
        let totalBudget = Int(Double(config.contextSize) * 0.30 * 3.5)
        let docCount = attachments.filter { $0.type == .textFile || $0.type == .pdf }.count
        let perDocBudget = docCount > 0 ? totalBudget / docCount : totalBudget
        var parts: [String] = []
        for att in attachments {
            if att.extractedText == nil || att.extractedText?.isEmpty == true {
                if att.type == .pdf || att.type == .textFile {
                    parts.append("[Attached file: \(att.filename) — content could not be extracted]")
                }
                continue
            }
            let text = att.extractedText!

            if att.type == .image {
                parts.append("[Image: \(att.filename)]\nAuto-caption: \(text)")
                continue
            }
            guard att.type == .textFile || att.type == .pdf else { continue }

            if text.count <= perDocBudget {
                parts.append("[Attached file: \(att.filename)]\n\(text)")
            } else {
                let retrieved = await retrieveRelevantChunks(text: text, query: query, budget: perDocBudget)
                if let retrieved, !retrieved.isEmpty {
                    parts.append("[Attached file: \(att.filename) \u{2014} relevant excerpts]\n\(retrieved)")
                } else {
                    let truncated = String(text.prefix(perDocBudget))
                        + "\n[truncated \u{2014} first \(perDocBudget) of \(text.count) chars]"
                    parts.append("[Attached file: \(att.filename)]\n\(truncated)")
                }
            }
        }
        return parts.joined(separator: "\n\n")
    }

    private func retrieveRelevantChunks(text: String, query: String, budget: Int) async -> String? {
        guard let pool else { return nil }
        do {
            let (handle, workerIdx) = try await pool.getDocIngestionHandle()
            let poolRef = try await pool.getPool()

            // Use priority chunking: heading-based splits + query relevance scoring
            let priorityResult = try await DocumentIngestionKernel.chunkWithPriority(
                pool: poolRef, workerIndex: workerIdx, kernelHandle: handle,
                text: text, query: query, topK: 15
            )

            guard !priorityResult.chunks.isEmpty else {
                // Fallback to old fixed-size chunking if priority chunking returns nothing
                return try await retrieveWithFixedChunks(pool: poolRef, workerIdx: workerIdx, handle: handle, text: text, query: query, budget: budget)
            }

            // Assemble within budget: first chunk always included, then by score
            var assembled = ""
            for chunk in priorityResult.chunks {
                if assembled.count + chunk.text.count > budget { break }
                if !assembled.isEmpty { assembled += "\n\n" }
                assembled += chunk.text
            }
            let method = priorityResult.method
            logger.debug("Priority retrieval (\(method, privacy: .public)): \(priorityResult.chunks.count, privacy: .public) chunks, \(assembled.count, privacy: .public) chars from \(text.count, privacy: .public) char doc")
            return assembled.isEmpty ? nil : assembled
        } catch {
            let desc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
            logger.warning("Semantic retrieval failed, falling back to truncation: \(desc, privacy: .public)")
            return nil
        }
    }

    private func retrieveWithFixedChunks(pool: PythonProcessPool, workerIdx: Int, handle: PyHandle, text: String, query: String, budget: Int) async throws -> String? {
        let chunkResult = try await DocumentIngestionKernel.chunkAndEmbed(
            pool: pool, workerIndex: workerIdx, kernelHandle: handle,
            text: text, chunkSize: 500, overlap: 100
        )
        guard !chunkResult.chunks.isEmpty else { return nil }

        let results = try await DocumentIngestionKernel.search(
            pool: pool, workerIndex: workerIdx, kernelHandle: handle,
            query: query, chunks: chunkResult.chunks, topK: 10
        )

        var assembled = ""
        for r in results {
            if assembled.count + r.text.count > budget { break }
            if !assembled.isEmpty { assembled += "\n\n" }
            assembled += r.text
        }
        logger.debug("Fixed-size retrieval: \(results.count, privacy: .public) chunks, \(assembled.count, privacy: .public) chars assembled from \(text.count, privacy: .public) char doc")
        return assembled.isEmpty ? nil : assembled
    }

    private struct MentionResolution {
        let cleanedPrompt: String
        let resolved: [SessionAttachmentReference]
        let unresolved: [String]
        let hadMentions: Bool
    }

    private func lookupReference(kind: MentionAssetKind, alias: String) -> SessionAttachmentReference? {
        switch kind {
        case .docs: return sessionDocs[alias]
        case .img: return sessionImages[alias]
        }
    }

    private func resolveMentions(in prompt: String) -> MentionResolution {
        guard let regex = Self.mentionTokenRegex else {
            return MentionResolution(cleanedPrompt: prompt, resolved: [], unresolved: [], hadMentions: false)
        }

        let nsPrompt = prompt as NSString
        let fullRange = NSRange(location: 0, length: nsPrompt.length)
        let matches = regex.matches(in: prompt, options: [], range: fullRange)
        guard !matches.isEmpty else {
            return MentionResolution(cleanedPrompt: prompt, resolved: [], unresolved: [], hadMentions: false)
        }

        var resolved: [SessionAttachmentReference] = []
        var unresolved: [String] = []
        for match in matches {
            guard let kindRange = Range(match.range(at: 1), in: prompt),
                  let aliasRange = Range(match.range(at: 2), in: prompt),
                  let kind = MentionAssetKind(rawValue: prompt[kindRange].lowercased()) else {
                continue
            }
            let alias = String(prompt[aliasRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            if let ref = lookupReference(kind: kind, alias: alias) {
                resolved.append(ref)
            } else {
                unresolved.append("@\(kind.rawValue)(\(alias))")
            }
        }

        var cleaned = prompt
        for match in matches.reversed() {
            guard let swiftRange = Range(match.range, in: cleaned) else { continue }
            cleaned.replaceSubrange(swiftRange, with: "")
        }

        let collapsed = cleaned
            .replacingOccurrences(of: #"[ \t]{2,}"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: #"\n{3,}"#, with: "\n\n", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return MentionResolution(
            cleanedPrompt: collapsed,
            resolved: resolved,
            unresolved: unresolved,
            hadMentions: true
        )
    }

    private func buildMentionContext(references: [SessionAttachmentReference], query: String) async -> String {
        let attachments = references.map { ref in
            MessageAttachment(
                type: ref.kind == .img ? .image : .textFile,
                filename: ref.filename,
                mimeType: ref.mimeType,
                data: nil,
                extractedText: ref.extractedText,
                thumbnailData: nil
            )
        }
        return await buildAttachmentContext(attachments: attachments, query: query)
    }

    private func recentSessionAttachmentReferences(maxDocs: Int = 4, maxImages: Int = 4) -> [SessionAttachmentReference] {
        let docs = sessionDocAliasOrder
            .suffix(max(0, maxDocs))
            .compactMap { sessionDocs[$0] }
        let images = sessionImageAliasOrder
            .suffix(max(0, maxImages))
            .compactMap { sessionImages[$0] }
        return images + docs
    }

    private func shouldAutoRecallSessionAttachments(
        prompt: String,
        hadMentions: Bool,
        hasCurrentAttachments: Bool
    ) -> Bool {
        let hasSessionAttachments = !sessionDocs.isEmpty || !sessionImages.isEmpty
        return Self.shouldAutoRecallSessionAttachments(
            prompt: prompt,
            hadMentions: hadMentions,
            hasCurrentAttachments: hasCurrentAttachments,
            hasSessionAttachments: hasSessionAttachments
        )
    }

    #if DEBUG
    func _testResetSessionReferences() {
        clearSessionAttachmentRegistry()
    }

    func _testRegisterSessionReference(
        kind: MentionAssetKind,
        filename: String,
        mimeType: String = "text/plain",
        extractedText: String?
    ) {
        registerSessionAttachmentReference(
            kind: kind,
            filename: filename,
            mimeType: mimeType,
            extractedText: extractedText
        )
    }

    func _testResolveMentions(
        prompt: String
    ) -> (cleanedPrompt: String, resolvedAliases: [String], unresolved: [String], hadMentions: Bool) {
        let result = resolveMentions(in: prompt)
        return (result.cleanedPrompt, result.resolved.map(\.alias), result.unresolved, result.hadMentions)
    }

    func _testMentionSuggestions(for kind: MentionAssetKind) -> [ComposerMentionSuggestion] {
        mentionSuggestions(for: kind)
    }

    static func _testShouldAutoInjectCurrentAttachments(
        prompt: String,
        hadMentions: Bool,
        hasCurrentAttachments: Bool
    ) -> Bool {
        shouldAutoInjectCurrentAttachments(
            originalPrompt: prompt,
            hadMentions: hadMentions,
            hasCurrentAttachments: hasCurrentAttachments
        )
    }

    static func _testShouldAutoRecallSessionAttachments(
        prompt: String,
        hadMentions: Bool,
        hasCurrentAttachments: Bool,
        hasSessionAttachments: Bool
    ) -> Bool {
        shouldAutoRecallSessionAttachments(
            prompt: prompt,
            hadMentions: hadMentions,
            hasCurrentAttachments: hasCurrentAttachments,
            hasSessionAttachments: hasSessionAttachments
        )
    }
    #endif

    // MARK: - Message Sending

    private struct StreamCompletion: Sendable {
        let finishReason: String
        let promptTokens: Int
        let completionTokens: Int
        let decodeMs: Double
        let finalPrefillMs: Double
        let finalText: String
        let finalThinking: String?
    }

    private static func shouldFlushStreamingPreview(for delta: String) -> Bool {
        delta.unicodeScalars.contains { streamPreviewBoundaryScalars.contains($0) }
    }

    private func makeSamplingParams(from ud: UserDefaults) -> SamplingParams {
        let stop = parseStopSequences(ud.string(forKey: SettingsKeys.stopSequences))
        let base = SamplingParams(
            maxTokens: ud.integer(forKey: SettingsKeys.maxTokens).clamped(to: 1...) ?? SettingsDefaults.maxTokens,
            temperature: ud.double(forKey: SettingsKeys.temperature).clamped(to: 0...) ?? SettingsDefaults.temperature,
            topP: ud.double(forKey: SettingsKeys.topP).clamped(to: 0...) ?? SettingsDefaults.topP,
            topK: ud.integer(forKey: SettingsKeys.topK).clamped(to: 1...) ?? SettingsDefaults.topK,
            repeatPenalty: ud.double(forKey: SettingsKeys.repeatPenalty).clamped(to: 1...) ?? SettingsDefaults.repeatPenalty,
            stop: stop
        )
        return Self.applyResponseMode(responseMode, base: base, contextSize: config.contextSize)
    }

    static func applyResponseMode(
        _ mode: ChatResponseMode,
        base: SamplingParams,
        contextSize: Int
    ) -> SamplingParams {
        switch mode {
        case .auto:
            return base
        case .instant:
            let cap = max(96, contextSize / 16)
            return SamplingParams(
                maxTokens: min(base.maxTokens, cap),
                temperature: min(base.temperature, 0.35),
                topP: min(base.topP, 0.90),
                topK: min(base.topK, 30),
                repeatPenalty: base.repeatPenalty,
                stop: base.stop
            )
        case .thinking:
            let floor = min(contextSize, max(512, contextSize / 4))
            return SamplingParams(
                maxTokens: max(base.maxTokens, floor),
                temperature: max(base.temperature, 0.70),
                topP: max(base.topP, 0.95),
                topK: max(base.topK, 40),
                repeatPenalty: base.repeatPenalty,
                stop: base.stop
            )
        }
    }

    /// User requested cancellation of the active streamed generation.
    func stopGeneration() async {
        guard isGenerating else { return }
        stopRequested = true
        activeStreamTask?.cancel()
    }

    func sendMessage() async {
        let hasPendingAttachments = !composerState.pendingAttachments.isEmpty
        guard !isGenerating else { return }
        guard let prompt = ComposerSendPolicy.resolvedPrompt(
            inputText: composerState.inputText,
            hasPendingAttachments: hasPendingAttachments
        ) else {
            return
        }
        guard isReady else {
            errorMessage = "Choose a model in Settings and tap Load Model first"
            showError = true
            return
        }
        dismissMentionPicker()

        // Capture pending attachments and build lightweight placeholders immediately
        // so the UI can respond without waiting for file I/O or extraction.
        let capturedPending = composerState.pendingAttachments
        let placeholderAttachments = capturedPending.map { pending in
            MessageAttachment(
                type: pending.type,
                filename: pending.filename,
                mimeType: Self.mimeType(for: pending),
                data: nil,
                thumbnailData: pending.thumbnailImage.flatMap { img in
                    img.tiffRepresentation.flatMap {
                        NSBitmapImageRep(data: $0)?.representation(using: .png, properties: [:])
                    }
                }
            )
        }

        // Append user message and set generating BEFORE any async work.
        // Otherwise a coalesced render after setSelectedConversation but before append can
        // show emptyState briefly (messages.isEmpty && !isGenerating).
        messages.append(ChatMessage(role: .user, content: prompt, attachments: placeholderAttachments))
        composerState.inputText = ""
        composerState.pendingAttachments = []
        composerState.inlineWarning = nil
        isGenerating = true
        stopRequested = false

        // Ensure we have a conversation ID for persistence (after append so no empty-state flash).
        // Temporary chat keeps the transcript in-memory only.
        if !temporaryChatEnabled && selectedConversationID == nil {
            setSelectedConversation(UUID().uuidString)
        }

        // Heavy work: file I/O, text extraction, VLM captioning, semantic retrieval.
        // Each step yields the main actor via await so the UI stays responsive.
        guard !stopRequested else {
            isGenerating = false
            return
        }
        let messageAttachments = await convertPendingToMessageAttachments(capturedPending)
        registerSessionAttachments(messageAttachments)
        let mentionResolution = resolveMentions(in: prompt)

        let resolvedPrompt: String = {
            if !mentionResolution.cleanedPrompt.isEmpty {
                return mentionResolution.cleanedPrompt
            }
            if mentionResolution.hadMentions || prompt.isEmpty {
                return ComposerSendPolicy.attachmentOnlyModelPrompt
            }
            return prompt
        }()

        let autoInjectCurrentAttachments = Self.shouldAutoInjectCurrentAttachments(
            originalPrompt: prompt,
            hadMentions: mentionResolution.hadMentions,
            hasCurrentAttachments: !messageAttachments.isEmpty
        )
        let autoRecallSessionAttachments = shouldAutoRecallSessionAttachments(
            prompt: resolvedPrompt,
            hadMentions: mentionResolution.hadMentions,
            hasCurrentAttachments: !messageAttachments.isEmpty
        )

        let attachmentContext: String
        if autoInjectCurrentAttachments {
            attachmentContext = await buildAttachmentContext(
                attachments: messageAttachments,
                query: resolvedPrompt
            )
        } else if !mentionResolution.resolved.isEmpty {
            attachmentContext = await buildMentionContext(
                references: mentionResolution.resolved,
                query: resolvedPrompt
            )
        } else if autoRecallSessionAttachments {
            let recalled = recentSessionAttachmentReferences()
            attachmentContext = await buildMentionContext(
                references: recalled,
                query: resolvedPrompt
            )
            logger.debug("Auto-recalled session attachments: \(recalled.count, privacy: .public) refs for prompt")
        } else {
            attachmentContext = ""
        }

        // Update the user message with fully processed attachments (extractedText, data for persistence).
        if !messageAttachments.isEmpty,
           let idx = messages.indices.last, messages[idx].role == .user {
            messages[idx] = ChatMessage(
                id: messages[idx].id,
                role: .user,
                content: prompt,
                attachments: messageAttachments
            )
        }

        if !mentionResolution.unresolved.isEmpty {
            let missing = mentionResolution.unresolved.prefix(3).joined(separator: ", ")
            let suffix = mentionResolution.unresolved.count > 3 ? " (+\(mentionResolution.unresolved.count - 3) more)" : ""
            showInlineWarning("Unresolved references: \(missing)\(suffix)")
        }
        if hasPendingAttachments && !mentionResolution.hadMentions && !autoInjectCurrentAttachments {
            showInlineWarning("Attachments were added but not referenced. Use @docs(...) or @img(...).")
        }

        if stopRequested {
            isGenerating = false
            return
        }

        let docContext: String? = attachmentContext.isEmpty ? nil : attachmentContext
        if codeActEnabled && pythonSandbox != nil {
            await _runCodeActLoop(prompt: resolvedPrompt, documentContext: docContext)
        } else {
            await _sendMessage(prompt: resolvedPrompt, documentContext: docContext)
        }
    }

    private static func mimeType(for pending: PendingAttachment) -> String {
        let ext = pending.fileURL.pathExtension.lowercased()
        switch pending.type {
        case .image:
            return ext == "png" ? "image/png" : ext == "gif" ? "image/gif" : "image/jpeg"
        case .pdf:
            return switch ext {
            case "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            case "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            case "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            default: "application/pdf"
            }
        case .textFile: return "text/plain"
        }
    }

    // MARK: - CodeAct Agent Loop

    /// Execute the CodeAct REPL loop for a single user turn.
    /// System prompt and `<execute>` protocol are verbatim from arxiv 2402.01030v4 Appendix E.
    /// Caller must set `isGenerating = true` and append the user message before calling.
    private func _runCodeActLoop(
        prompt: String,
        documentContext: String? = nil
    ) async {
        defer {
            isGenerating = false
            stopRequested = false
            activeStreamTask = nil
            liveAssistantState.message = nil
        }

        guard !stopRequested else { return }
        guard let scheduler else { return }
        guard let sandbox = pythonSandbox else { return }

        let params = makeSamplingParams(from: UserDefaults.standard)
        let systemPrompt = CodeActAgent.systemPrompt
        var currentPrompt = prompt
        var currentDocumentContext: String? = documentContext

        for iteration in 0..<CodeActAgent.maxIterations {
            guard !stopRequested else { break }

            let prior = messages.dropLast()
            let recentTurns: [(role: String, content: String)] = prior.map {
                (role: $0.role == .user ? "user" : "assistant", content: $0.content)
            }

            do {
                if chatSessionID == nil {
                    let newSession = try await scheduler.createSessionWithHistory(
                        systemPrompt: systemPrompt,
                        recentTurns: recentTurns
                    )
                    chatSessionID = newSession
                    logger.debug("CodeAct: new session \(newSession.description, privacy: .public)")
                }
            } catch {
                errorMessage = "Agent: session failed: \(Self.describeError(error))"
                showError = true
                return
            }

            guard let sessionID = chatSessionID else { return }

            let assistantID = UUID()
            withAnimation(.easeOut(duration: 0.12)) {
                liveAssistantState.message = ChatMessage(
                    id: assistantID, role: .assistant, content: "",
                    metrics: nil, thinking: nil, thinkingDurationSecs: nil
                )
            }

            let turnStart = ContinuousClock.now

            let (stream, resolvedSessionID): (CancellableStream<StreamInferenceChunk>, SessionID)
            do {
                (stream, resolvedSessionID) = try await scheduler.completeStreamWithMemoryManagement(
                    sessionID: sessionID,
                    prompt: currentPrompt,
                    params: params,
                    systemPrompt: systemPrompt,
                    recentTurns: recentTurns,
                    documentContext: currentDocumentContext
                )
            } catch {
                withAnimation(.easeOut(duration: 0.12)) { liveAssistantState.message = nil }
                errorMessage = "Agent: inference failed: \(Self.describeError(error))"
                showError = true
                chatSessionID = nil
                return
            }

            if resolvedSessionID != sessionID {
                chatSessionID = resolvedSessionID
            }

            let streamTask = Task { @MainActor [weak self] () throws -> StreamCompletion in
                guard let self else { throw CancellationError() }
                var rawText = ""
                var finalText = ""
                var finalThinking: String?
                var finalFinishReason = "unknown"
                var finalPromptTokens = 0
                var finalCompletionTokens = 0
                var finalDecodeMs = 0.0
                var finalPrefillMs = 0.0
                var previewDirty = false
                var lastPreviewAt = ContinuousClock.now

                for try await chunk in stream {
                    try Task.checkCancellation()
                    if self.stopRequested { throw CancellationError() }
                    switch chunk.event {
                    case .delta:
                        if chunk.delta.isEmpty { continue }
                        rawText += chunk.delta
                        previewDirty = true
                        let now = ContinuousClock.now
                        if (now - lastPreviewAt) >= Self.streamPreviewInterval
                            || Self.shouldFlushStreamingPreview(for: chunk.delta) {
                            let split = ThinkingTextParser.split(rawText: rawText, streamingInProgress: true)
                            finalText = split.text
                            finalThinking = split.thinking
                            self.updateLiveAssistantMessage(assistantID: assistantID, text: finalText, thinking: finalThinking)
                            previewDirty = false
                            lastPreviewAt = now
                        }
                    case .done:
                        finalFinishReason = chunk.finishReason ?? "stop"
                        finalPromptTokens = chunk.promptTokens ?? 0
                        finalCompletionTokens = chunk.completionTokens ?? 0
                        finalPrefillMs = chunk.prefillMs ?? 0
                        finalDecodeMs = chunk.decodeMs ?? 0
                        let split = ThinkingTextParser.split(
                            rawText: chunk.text ?? rawText,
                            preferredThinking: chunk.thinking
                        )
                        finalText = split.text
                        finalThinking = split.thinking
                        self.updateLiveAssistantMessage(assistantID: assistantID, text: finalText, thinking: finalThinking)
                        previewDirty = false
                    case .error:
                        let detail = [chunk.error, chunk.traceback].compactMap { $0 }.joined(separator: "\n")
                        throw InferenceError.decodeFailed(
                            sessionID: resolvedSessionID,
                            reason: detail.isEmpty ? "decode failed" : detail
                        )
                    }
                }
                if previewDirty {
                    let split = ThinkingTextParser.split(rawText: rawText, streamingInProgress: true)
                    finalText = split.text
                    finalThinking = split.thinking
                    self.updateLiveAssistantMessage(assistantID: assistantID, text: finalText, thinking: finalThinking)
                }
                return StreamCompletion(
                    finishReason: finalFinishReason,
                    promptTokens: finalPromptTokens,
                    completionTokens: finalCompletionTokens,
                    decodeMs: finalDecodeMs,
                    finalPrefillMs: finalPrefillMs,
                    finalText: finalText,
                    finalThinking: finalThinking
                )
            }
            activeStreamTask = streamTask

            let completion: StreamCompletion
            do {
                completion = try await streamTask.value
            } catch is CancellationError {
                await scheduler.finalizeCancelledStream(sessionID: resolvedSessionID)
                if let preview = liveAssistantState.message, preview.id == assistantID {
                    withAnimation(.easeOut(duration: 0.12)) {
                        liveAssistantState.message = nil
                        if LiveAssistantPreviewPolicy.hasVisibleContent(text: preview.content, thinking: preview.thinking) {
                            messages.append(ChatMessage(
                                id: preview.id, role: .assistant, content: preview.content,
                                metrics: "stopped", thinking: preview.thinking,
                                thinkingDurationSecs: preview.thinkingDurationSecs
                            ))
                        }
                    }
                }
                break
            } catch {
                await scheduler.finalizeFailedStream(sessionID: resolvedSessionID, reason: Self.describeError(error))
                withAnimation(.easeOut(duration: 0.12)) { liveAssistantState.message = nil }
                errorMessage = "Agent: \(Self.describeError(error))"
                showError = true
                chatSessionID = nil
                break
            }

            let turnElapsed = ContinuousClock.now - turnStart
            let turnSecs = Double(turnElapsed.components.seconds)
                + Double(turnElapsed.components.attoseconds) / 1e18
            let tps = turnSecs > 0 ? Double(completion.completionTokens) / turnSecs : 0

            await scheduler.finalizeCompletedStream(
                sessionID: resolvedSessionID,
                promptTokens: completion.promptTokens,
                completionTokens: completion.completionTokens,
                decodeMs: completion.decodeMs,
                finishReason: completion.finishReason
            )

            let hasExecute = CodeActAgent.parseExecuteBlock(completion.finalText) != nil
            withAnimation(.easeOut(duration: 0.12)) {
                liveAssistantState.message = nil
                messages.append(ChatMessage(
                    id: assistantID,
                    role: .assistant,
                    content: completion.finalText,
                    metrics: hasExecute ? nil : String(
                        format: "%.1f tok/s  \u{00B7}  %d tokens  \u{00B7}  %.2fs",
                        tps, completion.completionTokens, turnSecs
                    ),
                    thinking: completion.finalThinking,
                    thinkingDurationSecs: turnSecs
                ))
            }

            guard let code = CodeActAgent.parseExecuteBlock(completion.finalText) else { break }
            guard !stopRequested else { break }

            let output = await sandbox.run(code)
            let observation = CodeActAgent.formatObservation(output)
            logger.debug("CodeAct iter=\(iteration, privacy: .public) executed \(code.count, privacy: .public) chars → stdout=\(output.stdout.count, privacy: .public) err=\(output.error != nil, privacy: .public)")

            messages.append(ChatMessage(role: .user, content: observation))
            currentPrompt = observation
            currentDocumentContext = nil
        }

        if !temporaryChatEnabled, let convID = selectedConversationID {
            await saveCurrentConversation(id: convID)
            await refreshConversations()
            stripAttachmentData()
        }
    }

    // MARK: - Branch & Retry

    /// Create a new conversation branching from the current one at the given message index.
    /// Messages up to and including `atMessageIndex` are copied with fresh UUIDs.
    func branchConversation(atMessageIndex index: Int) async {
        guard !isGenerating else { return }
        guard let persistence, let parentID = selectedConversationID else { return }
        guard index >= 0, index < messages.count else { return }

        // Flush any pending save for the parent conversation
        await flushPendingSave()
        await saveCurrentConversation(id: parentID)

        let branchID = UUID().uuidString
        // Copy messages up to and including the fork point, with fresh UUIDs
        let branchMessages = messages.prefix(through: index).map { msg in
            let copiedAttachments = msg.attachments.map { att in
                MessageAttachment(
                    type: att.type, filename: att.filename, mimeType: att.mimeType,
                    data: att.data, extractedText: att.extractedText, thumbnailData: att.thumbnailData
                )
            }
            return ChatMessage(
                id: UUID(),
                role: msg.role,
                content: msg.content,
                metrics: msg.metrics,
                thinking: msg.thinking,
                thinkingDurationSecs: msg.thinkingDurationSecs,
                attachments: copiedAttachments
            )
        }
        let title = branchMessages.first { $0.role == .user }
            .map { String($0.content.prefix(40)) } ?? "Branch"

        do {
            // Save branch to DB immediately with forkNarrative: nil
            try await persistence.saveBranchConversation(
                id: branchID,
                parentID: parentID,
                forkMessageIndex: index,
                forkNarrative: nil,
                title: title,
                messages: branchMessages
            )

            // Switch to the branch
            chatSessionID = nil
            messages = branchMessages
            // Bug #6: Update lastSavedMessageIDs to match branch messages
            lastSavedMessageIDs = branchMessages.map(\.id)
            repopulateSessionAttachments(from: branchMessages)
            setSelectedConversation(branchID)
            await refreshConversations()
            await resolveParentTitle()

            // Bug #4: Fire-and-forget narrative summarization — don't block the branch
            let capturedScheduler = scheduler
            let capturedPersistence = persistence
            let capturedBranchID = branchID
            let sessionHistory: [[String: String]] = messages.map {
                ["role": $0.role == .user ? "user" : "assistant", "content": $0.content]
            }
            Task {
                guard let narrativeMemory = await capturedScheduler?.narrativeMemory else { return }
                do {
                    let narrative = try await narrativeMemory.summarize(
                        sessionID: SessionID(),
                        sessionHistory: sessionHistory
                    )
                    try await capturedPersistence.updateForkNarrative(branchID: capturedBranchID, narrative: narrative)
                } catch {
                    // Non-fatal — branch works without narrative
                }
            }
        } catch {
            logger.error("Failed to create branch: \(String(describing: error), privacy: .public)")
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Failed to create branch: \(String(describing: error))")
            errorMessage = "Failed to create branch: \(String(describing: error))"
            showError = true
        }
    }

    /// Retry the assistant response at the given index. Truncates messages from that point,
    /// resets the session, and re-sends the last user query with an optional retry instruction.
    func retryResponse(atIndex index: Int) async {
        guard !isGenerating else { return }
        guard index > 0, index < messages.count, messages[index].role == .assistant else { return }
        guard isReady else { return }

        // Bug #2: Set isGenerating FIRST to prevent empty-state flash
        isGenerating = true

        // Truncate from the assistant message onward — leaves [...context..., userQuery]
        messages = Array(messages.prefix(index))
        chatSessionID = nil

        // Bug #7: messages.last is the user query (prompt), messages.dropLast() is context
        guard let prompt = messages.last?.content else {
            isGenerating = false
            return
        }

        await _sendMessage(prompt: prompt)
    }

    /// Retry with a more specific instruction injected into the system prompt.
    func retryResponse(atIndex index: Int, instruction: String?) async {
        guard !isGenerating else { return }
        guard index > 0, index < messages.count, messages[index].role == .assistant else { return }
        guard isReady else { return }

        isGenerating = true
        messages = Array(messages.prefix(index))
        chatSessionID = nil

        guard let prompt = messages.last?.content else {
            isGenerating = false
            return
        }

        let systemPrompt: String
        if let instruction, !instruction.isEmpty {
            systemPrompt = "You are a helpful assistant. \(instruction)"
        } else {
            systemPrompt = "You are a helpful assistant."
        }

        await _sendMessage(prompt: prompt, systemPrompt: systemPrompt)
    }

    /// Bug #3: Resolve parent conversation title with staleness guard.
    private func resolveParentTitle() async {
        guard let persistence else { return }
        // Find the current conversation's parentConversationID
        guard let convID = selectedConversationID else { return }
        let targetID = convID  // Capture before await

        do {
            guard let conv = try await persistence.loadConversation(id: convID) else { return }
            guard let parentID = conv.parentConversationID else {
                parentConversationTitle = nil
                return
            }
            guard let parent = try await persistence.loadConversation(id: parentID) else {
                parentConversationTitle = nil
                return
            }
            // Staleness guard: user may have switched conversations during the await
            guard selectedConversationID == targetID else { return }
            parentConversationTitle = parent.title
        } catch {
            logger.error("Failed to resolve parent title: \(String(describing: error), privacy: .public)")
        }
    }

    /// Core message sending logic. Caller must set `isGenerating = true` and append the user
    /// message before calling. This method sets `isGenerating = false` on exit.
    private func _sendMessage(
        prompt: String,
        documentContext: String? = nil,
        systemPrompt: String = "You are a helpful assistant.",
        allowContextRetry: Bool = true
    ) async {
        var activeSessionID: SessionID?
        var activeAssistantID: UUID?
        var recentTurnsForRetry: [(role: String, content: String)] = []
        let clock = ContinuousClock()
        let start = clock.now
        defer {
            isGenerating = false
            stopRequested = false
            activeStreamTask = nil
            liveAssistantState.message = nil
        }

        do {
            guard !stopRequested else {
                throw CancellationError()
            }
            guard let scheduler = scheduler else {
                throw InferenceError.poolNotReady
            }

            let ud = UserDefaults.standard
            let params = makeSamplingParams(from: ud)

            // Bug #7: messages.dropLast() = context, prompt = the user query
            let prior = messages.dropLast()
            let context = prior.isEmpty ? "" : prior.map { "\($0.role == .user ? "User" : "Assistant"): \($0.content)" }.joined(separator: "\n")

            let recentTurns: [(role: String, content: String)] = prior.map {
                (role: $0.role == .user ? "user" : "assistant", content: $0.content)
            }
            recentTurnsForRetry = recentTurns

            let sessionID: SessionID
            if let existing = chatSessionID {
                sessionID = existing
                logger.debug("_sendMessage routing=direct(Llama) session=existing contextChars=\(context.count, privacy: .public)")
            } else {
                sessionID = try await scheduler.createSessionWithHistory(
                    systemPrompt: systemPrompt,
                    recentTurns: recentTurns
                )
                chatSessionID = sessionID
                logger.debug("_sendMessage routing=direct(Llama) session=new withHistory turns=\(recentTurns.count, privacy: .public)")
            }
            activeSessionID = sessionID

            // Pass the raw user prompt; scheduler decides whether to prepend document
            // context for this turn or consume it through rehydration.
            let rawPrompt = prompt

            guard !stopRequested else {
                throw CancellationError()
            }

            // Keep the live stream preview outside `messages` so frequent token
            // updates do not churn the full transcript layout.
            let assistantID = UUID()
            activeAssistantID = assistantID
            withAnimation(.easeOut(duration: 0.12)) {
                liveAssistantState.message = ChatMessage(
                    id: assistantID,
                    role: .assistant,
                    content: "",
                    metrics: nil,
                    thinking: nil,
                    thinkingDurationSecs: nil
                )
            }

            let (stream, newSessionID) = try await scheduler.completeStreamWithMemoryManagement(
                sessionID: sessionID,
                prompt: rawPrompt,
                params: params,
                systemPrompt: systemPrompt,
                recentTurns: recentTurns,
                documentContext: documentContext
            )

            // Session may have changed after memory reset
            if newSessionID != sessionID {
                logger.debug("Session reset by memory management: \(sessionID.description, privacy: .public) → \(newSessionID.description, privacy: .public)")
                chatSessionID = newSessionID
            }
            activeSessionID = newSessionID
            let promptForFallbackAccounting: String = {
                guard newSessionID == sessionID,
                      let docCtx = documentContext,
                      !docCtx.isEmpty else {
                    return rawPrompt
                }
                return "\(docCtx)\n\n\(rawPrompt)"
            }()

            let streamTask = Task { @MainActor [weak self] () throws -> StreamCompletion in
                guard let self else { throw CancellationError() }

                var rawText = ""
                var finalPromptTokens = 0
                var finalCompletionTokens = 0
                var finalDecodeMs = 0.0
                var finalPrefillMs = 0.0
                var finalFinishReason = "unknown"
                var finalText = ""
                var finalThinking: String?
                var previewDirty = false
                var lastPreviewAt = ContinuousClock.now

                for try await chunk in stream {
                    try Task.checkCancellation()
                    if self.stopRequested {
                        throw CancellationError()
                    }

                    switch chunk.event {
                    case .delta:
                        if chunk.delta.isEmpty { continue }
                        rawText += chunk.delta
                        previewDirty = true

                        let now = ContinuousClock.now
                        let shouldFlushByTime = (now - lastPreviewAt) >= Self.streamPreviewInterval
                        let shouldFlushByBoundary = Self.shouldFlushStreamingPreview(for: chunk.delta)
                        if shouldFlushByTime || shouldFlushByBoundary {
                            let split = ThinkingTextParser.split(
                                rawText: rawText,
                                streamingInProgress: true
                            )
                            finalText = split.text
                            finalThinking = split.thinking
                            self.updateLiveAssistantMessage(
                                assistantID: assistantID,
                                text: finalText,
                                thinking: finalThinking
                            )
                            previewDirty = false
                            lastPreviewAt = now
                        }
                    case .done:
                        finalFinishReason = chunk.finishReason ?? "stop"
                        finalPromptTokens = chunk.promptTokens ?? 0
                        finalCompletionTokens = chunk.completionTokens ?? 0
                        finalPrefillMs = chunk.prefillMs ?? 0
                        finalDecodeMs = chunk.decodeMs ?? 0

                        let split = ThinkingTextParser.split(
                            rawText: chunk.text ?? rawText,
                            preferredThinking: chunk.thinking
                        )
                        finalText = split.text
                        finalThinking = split.thinking
                        self.updateLiveAssistantMessage(
                            assistantID: assistantID,
                            text: finalText,
                            thinking: finalThinking
                        )
                        previewDirty = false
                    case .error:
                        let detail = [chunk.error, chunk.traceback]
                            .compactMap { $0 }
                            .joined(separator: "\n")
                        throw InferenceError.decodeFailed(
                            sessionID: newSessionID,
                            reason: detail.isEmpty ? "decode_stream failed" : detail
                        )
                    }
                }

                // Defensive flush: if stream ends without a terminal done chunk,
                // still publish any buffered text once.
                if previewDirty {
                    let split = ThinkingTextParser.split(
                        rawText: rawText,
                        streamingInProgress: true
                    )
                    finalText = split.text
                    finalThinking = split.thinking
                    self.updateLiveAssistantMessage(
                        assistantID: assistantID,
                        text: finalText,
                        thinking: finalThinking
                    )
                }

                return StreamCompletion(
                    finishReason: finalFinishReason,
                    promptTokens: finalPromptTokens,
                    completionTokens: finalCompletionTokens,
                    decodeMs: finalDecodeMs,
                    finalPrefillMs: finalPrefillMs,
                    finalText: finalText,
                    finalThinking: finalThinking
                )
            }
            activeStreamTask = streamTask

            let completion = try await streamTask.value
            let responseChars = (completion.finalText + (completion.finalThinking ?? "")).utf8.count
            let effectiveCompletionTokens: Int = {
                if completion.completionTokens > 0 {
                    return completion.completionTokens
                }
                guard responseChars > 0 else { return 0 }
                return max(1, Int(Double(responseChars) / 3.5))
            }()
            let effectivePromptTokens: Int = {
                if completion.promptTokens > 0 {
                    return completion.promptTokens
                }
                return max(0, Int(Double(promptForFallbackAccounting.utf8.count) / 3.5))
            }()

            await scheduler.finalizeCompletedStream(
                sessionID: newSessionID,
                promptTokens: effectivePromptTokens,
                completionTokens: effectiveCompletionTokens,
                decodeMs: completion.decodeMs,
                finishReason: completion.finishReason
            )

            let elapsed = (clock.now - start)
            let secs = Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            let tokPerSec = secs > 0 ? Double(effectiveCompletionTokens) / secs : 0

            // Calculate actual thinking time from prefillMs (convert ms to seconds)
            let thinkingSecs = completion.finalPrefillMs > 0
                ? completion.finalPrefillMs / 1000.0
                : secs  // Fallback to total time if prefillMs not available
            let generationSecs = max(0, secs - thinkingSecs)

            // Show thinking + generation breakdown when thinking time is significant (> 1s)
            let metrics: String
            if thinkingSecs > 1.0 {
                metrics = String(
                    format: "%.1f tok/s  \u{00B7}  %d tokens  \u{00B7}  thinking %.1fs + gen %.1fs = %.1fs",
                    tokPerSec, effectiveCompletionTokens, thinkingSecs, generationSecs, secs
                )
            } else {
                metrics = String(
                    format: "%.1f tok/s  \u{00B7}  %d tokens  \u{00B7}  %.2fs",
                    tokPerSec, effectiveCompletionTokens, secs
                )
            }

            withAnimation(.easeOut(duration: 0.12)) {
                liveAssistantState.message = nil
                messages.append(ChatMessage(
                    id: assistantID,
                    role: .assistant,
                    content: completion.finalText,
                    metrics: metrics,
                    thinking: completion.finalThinking,
                    thinkingDurationSecs: thinkingSecs  // FIXED: Use actual thinking time
                ))
            }

            // Auto-save conversation after assistant reply
            if !temporaryChatEnabled, let convID = selectedConversationID {
                await saveCurrentConversation(id: convID)
                await refreshConversations()
                // Strip heavy data blobs from in-memory attachments now that
                // they are safely persisted in SQLite. The UI only needs
                // thumbnailData, filename, and extractedText for display.
                stripAttachmentData()
            }

        } catch is CancellationError {
            if let sessionID = activeSessionID {
                await scheduler?.finalizeCancelledStream(sessionID: sessionID)
            }

            if let assistantID = activeAssistantID,
               let preview = liveAssistantState.message,
               preview.id == assistantID {
                let elapsed = (clock.now - start)
                let secs = Double(elapsed.components.seconds)
                    + Double(elapsed.components.attoseconds) / 1e18
                withAnimation(.easeOut(duration: 0.12)) {
                    liveAssistantState.message = nil
                    if LiveAssistantPreviewPolicy.hasVisibleContent(
                        text: preview.content,
                        thinking: preview.thinking
                    ) {
                        messages.append(ChatMessage(
                            id: preview.id,
                            role: .assistant,
                            content: preview.content,
                            metrics: preview.metrics ?? String(format: "stopped  \u{00B7}  %.2fs", secs),
                            thinking: preview.thinking,
                            thinkingDurationSecs: secs,
                            attachments: preview.attachments
                        ))
                    }
                }
            }

            if !temporaryChatEnabled, let convID = selectedConversationID {
                await saveCurrentConversation(id: convID)
                await refreshConversations()
                stripAttachmentData()
            }
        } catch {
            var detailedError = Self.describeError(error)
            let shouldRetry =
                allowContextRetry
                && !stopRequested
                && isRecoverableContextError(error)
                && activeSessionID != nil
                && scheduler != nil

            if let sessionID = activeSessionID {
                await scheduler?.finalizeFailedStream(sessionID: sessionID, reason: detailedError)
            }
            if let assistantID = activeAssistantID,
               let preview = liveAssistantState.message,
               preview.id == assistantID {
                withAnimation(.easeOut(duration: 0.12)) {
                    liveAssistantState.message = nil
                    if !shouldRetry && LiveAssistantPreviewPolicy.hasVisibleContent(
                        text: preview.content,
                        thinking: preview.thinking
                    ) {
                        messages.append(preview)
                    }
                }
            }

            if shouldRetry,
               let scheduler,
               let failedSessionID = activeSessionID {
                do {
                    let newSessionID = try await scheduler.resetAndRehydrate(
                        sessionID: failedSessionID,
                        systemPrompt: systemPrompt,
                        recentTurns: recentTurnsForRetry,
                        narrativeSummary: nil,
                        documentContext: documentContext
                    )
                    chatSessionID = newSessionID
                    await _sendMessage(
                        prompt: prompt,
                        documentContext: nil,
                        systemPrompt: systemPrompt,
                        allowContextRetry: false
                    )
                    return
                } catch {
                    detailedError = Self.describeError(error)
                }
            }

            errorMessage = "Generation failed: \(detailedError)"
            showError = true
            // Clear session so next message creates a fresh one (avoids "Session in phase prefilling/failed" on retry)
            chatSessionID = nil
            await FileLogger.shared.log(level: .error, category: "ChatViewModel", message: "Generation failed: \(detailedError)")
        }
    }

    private static let contextFailureNeedles: [String] = [
        "exceeded context",
        "context window",
        "requested tokens",
        "n_ctx",
        "maximum context",
        "context length"
    ]

    private static func shouldAutoInjectCurrentAttachments(
        originalPrompt: String,
        hadMentions: Bool,
        hasCurrentAttachments: Bool
    ) -> Bool {
        _ = originalPrompt
        return hasCurrentAttachments && !hadMentions
    }

    private static func shouldAutoRecallSessionAttachments(
        prompt: String,
        hadMentions: Bool,
        hasCurrentAttachments: Bool,
        hasSessionAttachments: Bool
    ) -> Bool {
        guard hasSessionAttachments else { return false }
        guard !hasCurrentAttachments else { return false }
        guard !hadMentions else { return false }
        guard !prompt.isEmpty else { return false }
        guard let regex = attachmentRecallRegex else { return false }
        let nsPrompt = prompt as NSString
        let range = NSRange(location: 0, length: nsPrompt.length)
        return regex.firstMatch(in: prompt, options: [], range: range) != nil
    }

    private func isRecoverableContextError(_ error: Error) -> Bool {
        if let inferenceError = error as? InferenceError {
            switch inferenceError {
            case .contextOverflow:
                return true
            case .decodeFailed(_, let reason), .prefillFailed(_, let reason):
                return Self.isContextRelatedReason(reason)
            default:
                break
            }
        }
        return Self.isContextRelatedReason(Self.describeError(error))
    }

    private static func isContextRelatedReason(_ reason: String) -> Bool {
        let haystack = reason.lowercased()
        return contextFailureNeedles.contains { haystack.contains($0) }
    }

    static func describeError(_ error: Error) -> String {
        if let workerError = error as? PythonWorkerError {
            return String(describing: workerError)
        }

        if let diagnostics = PythonError.fetchDetailed() {
            if let traceback = diagnostics.traceback, !traceback.isEmpty {
                return "\(diagnostics.type): \(diagnostics.message)\n\(traceback)"
            }
            return "\(diagnostics.type): \(diagnostics.message)"
        }

        if let pyError = error as? PythonError {
            return String(describing: pyError)
        }

        let described = String(describing: error)
        if !described.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return described
        }

        let localized = (error as NSError).localizedDescription
        if !localized.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return localized
        }

        return "Unknown error"
    }

    private func updateLiveAssistantMessage(
        assistantID: UUID,
        text: String,
        thinking: String?,
        metrics: String? = nil,
        thinkingDurationSecs: Double? = nil
    ) {
        guard let existing = liveAssistantState.message, existing.id == assistantID else { return }
        liveAssistantState.message = ChatMessage(
            id: existing.id,
            role: .assistant,
            content: text,
            metrics: metrics ?? existing.metrics,
            thinking: thinking,
            thinkingDurationSecs: thinkingDurationSecs ?? existing.thinkingDurationSecs,
            attachments: existing.attachments
        )
    }

    /// Replace in-memory attachment data blobs with nil for messages that are
    /// already persisted. The full file data lives in SQLite and is not needed
    /// for UI display. Only strips messages in `lastSavedMessageIDs` so that
    /// un-saved messages (e.g. during branch copy) retain their data.
    private func stripAttachmentData() {
        let savedSet = Set(lastSavedMessageIDs)
        for i in messages.indices {
            guard savedSet.contains(messages[i].id) else { continue }
            let atts = messages[i].attachments
            guard atts.contains(where: { $0.data != nil }) else { continue }
            messages[i].attachments = atts.map { att in
                guard att.data != nil else { return att }
                return MessageAttachment(
                    id: att.id,
                    type: att.type,
                    filename: att.filename,
                    mimeType: att.mimeType,
                    data: nil,
                    extractedText: att.extractedText,
                    thumbnailData: att.thumbnailData
                )
            }
        }
    }

    private func invalidatePool() async {
        await shutdownPoolIfNeeded()
    }

    /// Gracefully shut down the worker pool. Called on app quit and when invalidating the pool.
    func shutdownPoolIfNeeded() async {
        scheduler = nil
        chatSessionID = nil
        pythonSandbox = nil
        await vlmMonitor.shutdown()
        if let pool = pool {
            await pool.shutdown()
            self.pool = nil
        }
        isReady = false
    }

    // MARK: - Python Sandbox

    /// Whether a code sandbox is available for executing code blocks.
    var isSandboxAvailable: Bool {
        pythonSandbox != nil && isReady
    }

    /// Create the sandbox on the pool's dedicated sandbox worker.
    private func configureSandbox(pool: InferenceWorkerPool) async {
        guard let sbIdx = await pool.getSandboxWorkerIndex() else {
            logger.warning("Sandbox worker index not available (non-fatal)")
            return
        }
        do {
            let workerPool = try await pool.getPool()
            pythonSandbox = PythonSandbox(pool: workerPool, workerIdx: sbIdx)
            logger.debug("PythonSandbox configured on worker \(sbIdx)")
            await FileLogger.shared.log(level: .debug, category: "ChatViewModel", message: "PythonSandbox configured on worker \(sbIdx)")
        } catch {
            let desc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
            logger.warning("Failed to configure sandbox (non-fatal): \(desc, privacy: .public)")
        }
    }

    /// Execute a code block string in the sandbox and return the captured output.
    func executeCodeBlock(_ code: String) async -> RunOutput? {
        guard let sandbox = pythonSandbox else { return nil }
        return await sandbox.run(code)
    }

    /// Wipe the sandbox namespace (reset REPL state).
    func resetSandbox() async {
        await pythonSandbox?.clearNamespace()
    }

    // MARK: - Code Follow-Up Actions

    /// Run AST analysis on Python code via the sandbox worker.
    /// Returns nil if sandbox is unavailable or analysis fails.
    func analyzeCodeForReview(_ code: String) async -> CodeAnalysis? {
        guard let sandbox = pythonSandbox else { return nil }
        return await sandbox.analyzeForReview(code)
    }

    /// Compose a prompt for a code follow-up action.
    /// For Python "review" actions, enriches the prompt with AST analysis.
    func composeCodeActionPrompt(
        action: CodeFollowUpAction,
        code: String,
        language: String?,
        analysis: CodeAnalysis? = nil
    ) -> String {
        let lang = language ?? "code"
        let fence = "```\(lang)\n\(code)\n```"

        switch action {
        case .explain:
            return "Explain the following code clearly and concisely:\n\n\(fence)"
        case .review:
            if let analysis {
                return "Review and critique the following Python code. Static analysis found:\n\(analysis.promptSummary)\n\n\(fence)"
            }
            return "Review and critique the following code. Point out bugs, style issues, and suggest improvements:\n\n\(fence)"
        case .improve:
            return "Improve the following code for readability, correctness, and performance:\n\n\(fence)"
        case .writeTests:
            return "Write comprehensive unit tests for the following Python code using pytest:\n\n\(fence)"
        case .translateToPython:
            return "Translate the following \(lang) code to idiomatic Python:\n\n\(fence)"
        }
    }
}

/// Actions available in the code block "Ask AI" menu.
enum CodeFollowUpAction: String, CaseIterable, Sendable {
    case explain
    case review
    case improve
    case writeTests
    case translateToPython

    var label: String {
        switch self {
        case .explain: return "Explain this code"
        case .review: return "Review & critique"
        case .improve: return "Improve this code"
        case .writeTests: return "Write unit tests"
        case .translateToPython: return "Translate to Python"
        }
    }

    var systemImage: String {
        switch self {
        case .explain: return "questionmark.circle"
        case .review: return "magnifyingglass"
        case .improve: return "wand.and.stars"
        case .writeTests: return "checkmark.shield"
        case .translateToPython: return "arrow.left.arrow.right"
        }
    }
}

// MARK: - Helpers

private func parseStopSequences(_ raw: String?) -> [String] {
    guard let raw = raw, !raw.isEmpty else { return [] }
    return StopSequenceStorage.decode(raw)
}

private extension Int {
    func clamped(to range: PartialRangeFrom<Int>) -> Int? {
        self >= range.lowerBound ? self : nil
    }
}

private extension Double {
    func clamped(to range: PartialRangeFrom<Double>) -> Double? {
        self >= range.lowerBound ? self : nil
    }
}
