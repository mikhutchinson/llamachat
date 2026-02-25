import SwiftUI
import AppKit
import ChatStorage
import ChatUIComponents
import LlamaInferenceCore
import UniformTypeIdentifiers

// MARK: - Environment: Chat Font Size

private struct ChatFontSizeKey: EnvironmentKey {
    static let defaultValue: CGFloat = 14
}

extension EnvironmentValues {
    var chatFontSize: CGFloat {
        get { self[ChatFontSizeKey.self] }
        set { self[ChatFontSizeKey.self] = newValue }
    }
}

// MARK: - Theme Colors

struct ThemeColors: Equatable {
    let name: String
    let sidebarBg: Color
    let chatBg: Color
    let inputBg: Color
    let inputBorder: Color
    let userBubbleBg: Color
    let thinkingBg: Color
    let textPrimary: Color
    let textSecondary: Color
    let textTertiary: Color
    let accent: Color
    let divider: Color
    let searchFieldOverlay: Color
    let buttonOverlay: Color
    let selectedRowOverlay: Color
    let hoverRowOverlay: Color
    let avatarOverlay: Color
    let sendButtonEmptyOverlay: Color
    let emptyStateCircleOverlay: Color
    let maxContentWidth: CGFloat = 720

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.name == rhs.name
    }

    static let dark = ThemeColors(
        name: "dark",
        sidebarBg: Color(red: 0.09, green: 0.09, blue: 0.09),
        chatBg: Color(red: 0.13, green: 0.13, blue: 0.13),
        inputBg: Color(red: 0.17, green: 0.17, blue: 0.17),
        inputBorder: Color(red: 0.25, green: 0.25, blue: 0.25),
        userBubbleBg: Color(red: 0.20, green: 0.20, blue: 0.22),
        thinkingBg: Color(red: 0.16, green: 0.16, blue: 0.16),
        textPrimary: Color(red: 0.93, green: 0.93, blue: 0.93),
        textSecondary: Color(red: 0.60, green: 0.60, blue: 0.60),
        textTertiary: Color(red: 0.42, green: 0.42, blue: 0.42),
        accent: Color(red: 0.40, green: 0.75, blue: 0.60),
        divider: Color(red: 0.20, green: 0.20, blue: 0.20),
        searchFieldOverlay: Color.white.opacity(0.04),
        buttonOverlay: Color.white.opacity(0.08),
        selectedRowOverlay: Color.white.opacity(0.12),
        hoverRowOverlay: Color.white.opacity(0.06),
        avatarOverlay: Color.white.opacity(0.08),
        sendButtonEmptyOverlay: Color.white.opacity(0.06),
        emptyStateCircleOverlay: Color.white.opacity(0.06)
    )

    static let light = ThemeColors(
        name: "light",
        sidebarBg: Color(red: 0.96, green: 0.96, blue: 0.96),
        chatBg: Color(red: 0.98, green: 0.98, blue: 0.98),
        inputBg: Color(red: 0.94, green: 0.94, blue: 0.94),
        inputBorder: Color(red: 0.88, green: 0.88, blue: 0.88),
        userBubbleBg: Color(red: 0.88, green: 0.90, blue: 0.95),
        thinkingBg: Color(red: 0.94, green: 0.94, blue: 0.94),
        textPrimary: Color(red: 0.13, green: 0.13, blue: 0.13),
        textSecondary: Color(red: 0.45, green: 0.45, blue: 0.45),
        textTertiary: Color(red: 0.55, green: 0.55, blue: 0.55),
        accent: Color(red: 0.20, green: 0.55, blue: 0.40),
        divider: Color(red: 0.90, green: 0.90, blue: 0.90),
        searchFieldOverlay: Color.black.opacity(0.04),
        buttonOverlay: Color.black.opacity(0.06),
        selectedRowOverlay: Color.black.opacity(0.08),
        hoverRowOverlay: Color.black.opacity(0.04),
        avatarOverlay: Color.black.opacity(0.06),
        sendButtonEmptyOverlay: Color.black.opacity(0.08),
        emptyStateCircleOverlay: Color.black.opacity(0.06)
    )
}

// MARK: - Environment: Theme

private struct ThemeKey: EnvironmentKey {
    static let defaultValue: ThemeColors = .dark
}

extension EnvironmentValues {
    var theme: ThemeColors {
        get { self[ThemeKey.self] }
        set { self[ThemeKey.self] = newValue }
    }
}

struct ContentView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Bindable var downloadManager: ModelDownloadManager
    @Environment(\.openWindow) private var openWindow
    @Environment(\.openSettings) private var openSettings
    @State private var systemAppearanceSeed = 0  // Bump when macOS appearance changes
    @AppStorage(SettingsKeys.appTheme) private var appTheme = SettingsDefaults.appTheme
    @AppStorage(SettingsKeys.chatFontSize) private var chatFontSize = SettingsDefaults.chatFontSize
    @AppStorage(SettingsKeys.sidebarVisibleOnLaunch) private var sidebarVisibleOnLaunch = SettingsDefaults.sidebarVisibleOnLaunch
    @State private var columnVisibility: NavigationSplitViewVisibility = .automatic
    /// Tracks whether the user is scrolled near the bottom of the message list.
    /// Auto-scroll only fires when this is true, so reading history is not disrupted.
    ///
    /// Uses a reference wrapper instead of `@State` because `onScrollGeometryChange`
    /// runs during layout — writing `@State` there causes "onChange(of: Layout) action
    /// tried to update multiple times per frame" faults.  Mutating a property on a
    /// reference-type `@State` does not trigger SwiftUI observation, breaking the cycle.
    private final class _NearBottomFlag { var value = true }
    @State private var _nearBottom = _NearBottomFlag()
    @State private var isModelPopoverPresented = false
    @State private var editingConversationID: String?
    @State private var editingTitle: String = ""

    private var resolvedColorScheme: ColorScheme? {
        switch appTheme {
        case "light": return .light
        case "dark": return .dark
        default: return nil
        }
    }

    private var currentTheme: ThemeColors {
        switch appTheme {
        case "light": return .light
        case "dark": return .dark
        default:
            // Use NSApp.effectiveAppearance for "system" — @Environment(\.colorScheme) can be wrong.
            // systemAppearanceSeed forces re-read when AppleInterfaceThemeChanged fires (after deferred bump).
            _ = systemAppearanceSeed
            let isDark = NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            return isDark ? .dark : .light
        }
    }

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            sidebar
        } detail: {
            chatDetail
        }
        .environment(\.theme, currentTheme)
        .navigationSplitViewStyle(.balanced)
        .toolbar(removing: .sidebarToggle)
        .toolbar(removing: .title)
        .toolbar {
            ToolbarItem(placement: .navigation) {
                toolbarControlsCapsule
            }
            ToolbarItem(placement: .principal) {
                modelSelectorPill
            }
        }
        .frame(minWidth: 700, minHeight: 500)
        .toolbarBackground(currentTheme.chatBg)
        .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
        .preferredColorScheme(resolvedColorScheme)
        .onAppear {
            viewModel.refreshDiscoveredModels()
            if !sidebarVisibleOnLaunch {
                columnVisibility = .detailOnly
            }
        }
        .onReceive(
            DistributedNotificationCenter.default().publisher(for: .init("AppleInterfaceThemeChanged"))
        ) { _ in
            // Defer so NSApp.effectiveAppearance has time to update before we re-render.
            // Otherwise we read stale dark appearance while native SwiftUI has already
            // switched to light → hybrid UI with dark backgrounds and light text/blocks.
            DispatchQueue.main.async {
                systemAppearanceSeed += 1  // Force re-render when user changes macOS appearance
            }
        }
        .alert("Error", isPresented: $viewModel.showError) {
            Button("OK") {}
        } message: {
            Text(viewModel.errorMessage)
        }
    }

    // MARK: - Sidebar

    @State private var searchText = ""
    @State private var searchDebounceTask: Task<Void, Never>?

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 12))
                .foregroundColor(currentTheme.textTertiary)
            TextField("Search", text: $searchText)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .foregroundColor(currentTheme.textPrimary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(currentTheme.searchFieldOverlay)
        )
        .padding(.horizontal, 10)
        .padding(.top, 8)
        .padding(.bottom, 8)
    }

    private var conversationList: some View {
        ScrollView {
            LazyVStack(spacing: 2) {
                if viewModel.conversations.isEmpty {
                    Text(searchText.isEmpty ? "No conversations yet" : "No results")
                        .font(.system(size: 13))
                        .foregroundColor(currentTheme.textTertiary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 20)
                } else {
                    ForEach(viewModel.conversations) { conv in
                        ConversationRow(
                            conversation: conv,
                            isSelected: conv.id == viewModel.selectedConversationID,
                            isEditing: editingConversationID == conv.id,
                            editingTitle: $editingTitle,
                            onSaveEdit: { saveEdit() },
                            onCancelEdit: { cancelEdit() }
                        )
                        .onTapGesture {
                            // Cancel editing when tapping another row
                            if editingConversationID != nil {
                                cancelEdit()
                            }
                            Task { await viewModel.selectConversation(id: conv.id) }
                        }
                        .contextMenu {
                            Button {
                                startEditing(conv)
                            } label: {
                                Label("Rename", systemImage: "pencil")
                            }
                            Button {
                                let panel = NSSavePanel()
                                panel.allowedContentTypes = [.plainText, UTType(filenameExtension: "md") ?? .plainText]
                                panel.nameFieldStringValue = "\(conv.title).md"
                                    .replacingOccurrences(of: "/", with: "-")
                                panel.canCreateDirectories = true
                                if panel.runModal() == .OK, let url = panel.url {
                                    Task {
                                        do {
                                            try await viewModel.exportConversation(id: conv.id, to: url)
                                        } catch {
                                            viewModel.errorMessage = error.localizedDescription
                                            viewModel.showError = true
                                        }
                                    }
                                }
                            } label: {
                                Label("Export…", systemImage: "square.and.arrow.up")
                            }
                            Divider()
                            Button(role: .destructive) {
                                Task { await viewModel.deleteConversation(id: conv.id) }
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - Inline Editing

    private func startEditing(_ conv: Conversation) {
        editingConversationID = conv.id
        editingTitle = conv.title
    }

    private func saveEdit() {
        guard let id = editingConversationID else { return }
        let trimmed = editingTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            cancelEdit()
            return
        }
        Task {
            await viewModel.renameConversation(id: id, newTitle: trimmed)
        }
        clearEditState()
    }

    private func cancelEdit() {
        clearEditState()
    }

    private func clearEditState() {
        editingConversationID = nil
        editingTitle = ""
    }

    private var sidebarBottomBar: some View {
        VStack(spacing: 0) {
            currentTheme.divider.frame(height: 0.5)
            HStack(spacing: 10) {
                statusBadge
                Spacer()
                Button {
                    Task { await viewModel.loadModel() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 14))
                        .foregroundColor(currentTheme.textSecondary)
                }
                .buttonStyle(.plain)
                .disabled(viewModel.config.modelPath.isEmpty || viewModel.isLoading)
                .help("Reload model")
                Button {
                    openWindow(id: AppWindowID.modelHub)
                } label: {
                    Image(systemName: "shippingbox")
                        .font(.system(size: 14))
                        .foregroundColor(currentTheme.textSecondary)
                }
                .buttonStyle(.plain)
                .help("Model Hub")
                Button {
                    openSettings()
                } label: {
                    Image(systemName: "gearshape")
                        .font(.system(size: 14))
                        .foregroundColor(currentTheme.textSecondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
        }
    }

    private var sidebarDownloadStrip: some View {
        VStack(spacing: 0) {
            currentTheme.divider.frame(height: 0.5)
            VStack(spacing: 4) {
                ForEach(Array(downloadManager.activeDownloads.values.filter { task in
                    if case .downloading = task.state { return true }
                    if case .queued = task.state { return true }
                    if case .paused = task.state { return true }
                    if case .verifying = task.state { return true }
                    return false
                }), id: \.id) { task in
                    HStack(spacing: 6) {
                        if case .verifying = task.state {
                            Image(systemName: "checkmark.shield")
                                .font(.system(size: 10))
                                .foregroundColor(currentTheme.accent)
                        } else if case .paused = task.state {
                            Image(systemName: "pause.circle.fill")
                                .font(.system(size: 10))
                                .foregroundColor(currentTheme.textTertiary)
                        } else {
                            Image(systemName: "arrow.down.circle")
                                .font(.system(size: 10))
                                .foregroundColor(currentTheme.accent)
                        }
                        Text(task.filename)
                            .font(.caption2)
                            .foregroundColor(currentTheme.textSecondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer(minLength: 4)
                        if case .verifying = task.state {
                            ProgressView().controlSize(.mini)
                            Text("Verify")
                                .font(.caption2)
                                .foregroundColor(currentTheme.textTertiary)
                        } else if case .paused = task.state {
                            Text("Paused")
                                .font(.caption2)
                                .foregroundColor(currentTheme.textTertiary)
                            Button {
                                downloadManager.resumeDownload(filename: task.id)
                            } label: {
                                Image(systemName: "play.fill")
                                    .font(.system(size: 8))
                                    .foregroundColor(currentTheme.accent)
                            }
                            .buttonStyle(.plain)
                        } else {
                            ProgressView(value: task.progress)
                                .frame(width: 50)
                            Text("\(Int(task.progress * 100))%")
                                .font(.caption2.monospacedDigit())
                                .foregroundColor(currentTheme.textTertiary)
                                .frame(width: 28, alignment: .trailing)
                            Button {
                                downloadManager.pause(filename: task.id)
                            } label: {
                                Image(systemName: "pause.fill")
                                    .font(.system(size: 8))
                                    .foregroundColor(currentTheme.textTertiary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                // Show recently completed downloads briefly
                ForEach(Array(downloadManager.activeDownloads.values.filter { task in
                    if case .completed = task.state { return true }
                    return false
                }), id: \.id) { task in
                    HStack(spacing: 6) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 10))
                            .foregroundColor(.green)
                        Text(task.filename)
                            .font(.caption2)
                            .foregroundColor(currentTheme.textSecondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer(minLength: 4)
                        Button {
                            downloadManager.dismiss(filename: task.id)
                        } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 8))
                                .foregroundColor(currentTheme.textTertiary)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
    }

    private var sidebar: some View {
        VStack(spacing: 0) {
            searchField
            conversationList
            Spacer()
            if !downloadManager.activeDownloads.isEmpty {
                sidebarDownloadStrip
            }
            sidebarBottomBar
        }
        .background(currentTheme.chatBg.ignoresSafeArea(edges: .top))
        .overlay(alignment: .trailing) {
            // Cover the NavigationSplitView column divider (no SwiftUI API to hide it)
            Rectangle()
                .fill(currentTheme.chatBg)
                .frame(width: 4)
                .ignoresSafeArea(edges: .vertical)
        }
        .navigationSplitViewColumnWidth(min: 200, ideal: 260, max: 300)
        .toolbar(removing: .sidebarToggle)
        .onChange(of: searchText) { _, newValue in
            searchDebounceTask?.cancel()
            searchDebounceTask = Task {
                try? await Task.sleep(for: .milliseconds(300))
                guard !Task.isCancelled else { return }
                viewModel.searchQuery = newValue
                await viewModel.refreshConversations()
            }
        }
    }

    private var statusBadge: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(viewModel.isReady ? currentTheme.accent : currentTheme.textTertiary)
                .frame(width: 7, height: 7)
            if viewModel.isLoading {
                ProgressView()
                    .controlSize(.small)
            }
        }
    }

    // MARK: - Chat Detail

    private var chatDetail: some View {
        VStack(spacing: 0) {
            messageList
            ComposerView(
                composerState: viewModel.composerState,
                isGenerating: viewModel.isGenerating,
                isReady: viewModel.isReady,
                onSend: { Task { await viewModel.sendMessage() } },
                onStop: { Task { await viewModel.stopGeneration() } },
                onAddAttachment: { url in viewModel.addAttachment(url: url) },
                onRemoveAttachment: { id in viewModel.removeAttachment(id: id) },
                onComposerTextChanged: { text in viewModel.composerInputDidChange(text) },
                onOpenMentionPicker: { kind in viewModel.showMentionPicker(kind: kind) },
                onSelectMentionSuggestion: { suggestion in viewModel.selectMentionSuggestion(suggestion) },
                onDismissMentionPicker: { viewModel.dismissMentionPicker() },
                codeActEnabled: Binding(
                    get: { viewModel.codeActEnabled },
                    set: { viewModel.setCodeActEnabled($0) }
                ),
                sandboxReady: viewModel.isSandboxAvailable
            )
        }
        .background(currentTheme.chatBg.ignoresSafeArea(edges: .top))
        .overlay(alignment: .leading) {
            // Cover the NavigationSplitView column divider from detail side (belt-and-suspenders)
            Rectangle()
                .fill(currentTheme.chatBg)
                .frame(width: 4)
                .ignoresSafeArea(edges: .vertical)
        }
    }

    private var activeModelShortName: String {
        let path = viewModel.config.modelPath
        guard !path.isEmpty else { return "Choose Model" }
        return ModelShortNameFormatter.shortName(fromModelPath: path)
    }

    private var discoveredMainModels: [DiscoveredModel] {
        viewModel.discoveredModels.filter { !$0.isMMProj }
    }

    private var toolbarModelList: [DiscoveredModel] {
        let discovered = discoveredMainModels
        guard !discovered.isEmpty else { return [] }

        let byPath = Dictionary(uniqueKeysWithValues: discovered.map { ($0.path, $0) })
        var ordered: [DiscoveredModel] = []

        func appendUnique(_ model: DiscoveredModel) {
            if ordered.contains(where: { $0.path == model.path }) { return }
            ordered.append(model)
        }

        if let active = byPath[viewModel.config.modelPath] {
            appendUnique(active)
        }

        for path in viewModel.recentModelPaths {
            if let model = byPath[path] {
                appendUnique(model)
            }
        }

        for model in discovered.sorted(by: { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }) {
            appendUnique(model)
            if ordered.count >= 6 { break }
        }

        return Array(ordered.prefix(6))
    }

    private var temporaryChatBinding: Binding<Bool> {
        Binding(
            get: { viewModel.temporaryChatEnabled },
            set: { enabled in
                Task { await viewModel.setTemporaryChatEnabled(enabled) }
            }
        )
    }

    private var modelSelectorPill: some View {
        ZStack {
            Capsule(style: .continuous)
                .fill(currentTheme == .light ? Color.black.opacity(0.05) : Color.white.opacity(0.08))

            HStack(spacing: 6) {
                Text(activeModelShortName)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(currentTheme.textPrimary)
                    .lineLimit(1)
                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(currentTheme.textSecondary)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 7)
            .background(
                Capsule(style: .continuous)
                    .fill(currentTheme.buttonOverlay)
            )
            .overlay(
                Capsule(style: .continuous)
                    .inset(by: 0.5)
                    .stroke(currentTheme.divider.opacity(0.78), lineWidth: 1)
            )
            .padding(2.5)
        }
        .contentShape(Capsule(style: .continuous))
        .onTapGesture {
            isModelPopoverPresented.toggle()
        }
        .popover(isPresented: $isModelPopoverPresented, arrowEdge: .top) {
            modelPopoverContent
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
        .accessibilityLabel("Model and response settings")
        .fixedSize()
    }

    private var modelPopoverContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Response Mode")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(currentTheme.textTertiary)
                .padding(.bottom, 8)

            VStack(spacing: 1) {
                ForEach(ChatResponseMode.allCases, id: \.rawValue) { mode in
                    Button {
                        viewModel.setResponseMode(mode)
                        isModelPopoverPresented = false
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: viewModel.responseMode == mode ? "checkmark" : "circle")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundColor(viewModel.responseMode == mode ? currentTheme.textPrimary : .clear)
                                .frame(width: 14)
                            Text(mode.title)
                                .font(.system(size: 13, weight: .regular))
                                .foregroundColor(currentTheme.textPrimary)
                            Spacer(minLength: 0)
                        }
                        .padding(.vertical, 6)
                    }
                    .buttonStyle(.plain)
                }
            }

            Divider().padding(.vertical, 10)

            Text("Models")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(currentTheme.textTertiary)
                .padding(.bottom, 8)

            if toolbarModelList.isEmpty {
                Text("No models found")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .padding(.bottom, 8)
            } else {
                VStack(spacing: 1) {
                    ForEach(toolbarModelList) { model in
                        Button {
                            Task { await viewModel.selectPrimaryModel(path: model.path) }
                            isModelPopoverPresented = false
                        } label: {
                            HStack(spacing: 10) {
                                Image(systemName: viewModel.config.modelPath == model.path ? "checkmark" : "circle")
                                    .font(.system(size: 11, weight: .semibold))
                                    .foregroundColor(viewModel.config.modelPath == model.path ? currentTheme.textPrimary : .clear)
                                    .frame(width: 14)
                                Text(ModelShortNameFormatter.shortName(fromFilename: model.name))
                                    .font(.system(size: 13))
                                    .foregroundColor(currentTheme.textPrimary)
                                Spacer(minLength: 0)
                                Text(model.size)
                                    .font(.system(size: 11))
                                    .foregroundStyle(currentTheme.textTertiary)
                            }
                            .padding(.vertical, 6)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            Divider().padding(.vertical, 10)

            Toggle(isOn: temporaryChatBinding) {
                Text("Temporary Chat")
                    .font(.system(size: 13))
            }
            .toggleStyle(.switch)
            .disabled(viewModel.isGenerating)
            .padding(.bottom, 8)

            if viewModel.hasSessionToReset {
                Button {
                    Task { await viewModel.resetContextWindow() }
                    isModelPopoverPresented = false
                } label: {
                    Label("Reset Context", systemImage: "arrow.triangle.2.circlepath")
                        .font(.system(size: 13))
                        .foregroundColor(viewModel.isGenerating ? currentTheme.textTertiary : currentTheme.textPrimary)
                }
                .buttonStyle(.plain)
                .disabled(viewModel.isGenerating)
                .help("Clears the model's memory (KV cache) while preserving conversation history")

                Divider().padding(.vertical, 10)
            }

            Button {
                openWindow(id: AppWindowID.modelHub)
                isModelPopoverPresented = false
            } label: {
                Label("Browse Models\u{2026}", systemImage: "folder")
                    .font(.system(size: 13))
                    .foregroundColor(currentTheme.textPrimary)
            }
            .buttonStyle(.plain)
            .padding(.bottom, 8)

            Button {
                openSettings()
                isModelPopoverPresented = false
            } label: {
                Label("Model Settings\u{2026}", systemImage: "slider.horizontal.3")
                    .font(.system(size: 13))
                    .foregroundColor(currentTheme.textPrimary)
            }
            .buttonStyle(.plain)
        }
        .padding(14)
        .frame(width: 300)
    }

    private var toolbarControlsCapsule: some View {
        HStack(spacing: 10) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    columnVisibility = columnVisibility == .detailOnly ? .all : .detailOnly
                }
            } label: {
                Image(systemName: "sidebar.left")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(currentTheme.textSecondary)
                    .frame(width: 22, height: 22)
            }
            .buttonStyle(.plain)
            .contentShape(Rectangle())
            .help(columnVisibility == .detailOnly ? "Show sidebar" : "Hide sidebar")

            Button {
                Task { await viewModel.newChat() }
            } label: {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(currentTheme.textSecondary)
                    .frame(width: 22, height: 22)
            }
            .buttonStyle(.plain)
            .contentShape(Rectangle())
            .help("New chat")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(
            Capsule(style: .continuous)
                .fill(currentTheme.buttonOverlay)
        )
        .overlay(
            Capsule(style: .continuous)
                .strokeBorder(currentTheme.divider.opacity(0.8), lineWidth: 0.5)
        )
    }

    // MARK: - Message List

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                if viewModel.isLoadingMessages {
                    VStack {
                        Spacer()
                        ProgressView()
                            .controlSize(.regular)
                        Spacer()
                    }
                    .frame(maxWidth: .infinity, minHeight: 200)
                } else if viewModel.messages.isEmpty && !viewModel.isGenerating {
                    emptyState
                } else {
                    VStack(spacing: 0) {
                        if viewModel.hasMoreMessages {
                            Button {
                                Task { await viewModel.loadMoreMessages() }
                            } label: {
                                Text("Load earlier messages")
                                    .font(.system(size: 13))
                                    .foregroundColor(currentTheme.accent)
                                    .padding(.vertical, 8)
                            }
                            .buttonStyle(.plain)
                        }
                        if let parentTitle = viewModel.parentConversationTitle {
                            BranchedFromDivider(parentTitle: parentTitle) {
                                // Navigate to parent conversation
                                if let conv = viewModel.conversations.first(where: { $0.title == parentTitle }) {
                                    Task { await viewModel.selectConversation(id: conv.id) }
                                }
                            }
                        }
                        ForEach(viewModel.messages) { msg in
                            MessageRow(message: msg)
                                .id(msg.id)
                        }
                        LiveAssistantInlineRow(state: viewModel.liveAssistantState)
                    }
                    .environmentObject(viewModel)
                    .environment(\.chatFontSize, CGFloat(chatFontSize))
                    .padding(.top, 12)
                    .padding(.bottom, 20)
                }
                // Sentinel — always present at the end of the scroll content.
                Color.clear.frame(height: 1).id("bottom")
            }
            .defaultScrollAnchor(.bottom)
            .scrollContentBackground(.hidden)
            .overlay(alignment: .bottom) {
                LinearGradient(
                    colors: [Color.clear, currentTheme.chatBg],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .frame(height: 48)
                .allowsHitTesting(false)
            }
            // Track whether the user is near the bottom so we don't yank them
            // away from reading history when new content arrives.
            .onScrollGeometryChange(for: Bool.self, of: { geo in
                let maxY = geo.contentSize.height - geo.containerSize.height
                return maxY <= 0 || geo.contentOffset.y >= maxY - 100
            }) { [_nearBottom] _, newValue in
                _nearBottom.value = newValue
            }
            .onChange(of: viewModel.messages.count) { [_nearBottom] oldCount, _ in
                if _nearBottom.value || oldCount == 0 {
                    DispatchQueue.main.async {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }
            .onChange(of: viewModel.isGenerating) { [_nearBottom] _, generating in
                if generating && _nearBottom.value {
                    DispatchQueue.main.async {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }
            // Bug #9: Scroll to bottom on conversation switch (handles same-count branches)
            .onChange(of: viewModel.selectedConversationID) { _, _ in
                DispatchQueue.main.async {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            ZStack {
                Circle()
                    .fill(currentTheme.emptyStateCircleOverlay)
                    .frame(width: 64, height: 64)
                Text("\u{1F999}")
                    .font(.system(size: 32))
            }
            Text(viewModel.isReady ? "What can I help with?" : "Choose a model to get started")
                .font(.system(size: 22, weight: .semibold))
                .foregroundColor(currentTheme.textPrimary)
            if !viewModel.isReady {
                Text("Open settings with the gear icon")
                    .font(.system(size: 14))
                    .foregroundColor(currentTheme.textTertiary)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding(40)
    }

}

// MARK: - ComposerView (separate ObservableObject observation boundary)

/// Extracted composer bar that observes `ComposerState` directly instead of
/// the full `ChatViewModel`. Keystrokes update `ComposerState.inputText`,
/// firing `ComposerState.objectWillChange` — which only invalidates this
/// view, not the parent `ContentView` or its heavyweight message list.
struct ComposerView: View {
    @ObservedObject var composerState: ComposerState
    let isGenerating: Bool
    let isReady: Bool
    let onSend: () -> Void
    let onStop: () -> Void
    let onAddAttachment: (URL) -> Void
    let onRemoveAttachment: (UUID) -> Void
    let onComposerTextChanged: (String) -> Void
    let onOpenMentionPicker: (MentionAssetKind) -> Void
    let onSelectMentionSuggestion: (ComposerMentionSuggestion) -> Void
    let onDismissMentionPicker: () -> Void
    @Binding var codeActEnabled: Bool
    let sandboxReady: Bool

    @Environment(\.theme) private var theme
    @FocusState private var isInputFocused: Bool
    @State private var inputContentHeight: CGFloat = 22

    private var canSend: Bool {
        ComposerSendPolicy.canSend(
            inputText: composerState.inputText,
            hasPendingAttachments: !composerState.pendingAttachments.isEmpty,
            isReady: isReady,
            isGenerating: isGenerating
        )
    }

    private var isStopMode: Bool {
        isGenerating
    }

    private var mentionPopoverBinding: Binding<Bool> {
        Binding(
            get: {
                composerState.activeMentionKind != nil
                    && !composerState.mentionSuggestions.isEmpty
            },
            set: { isPresented in
                if !isPresented {
                    onDismissMentionPicker()
                }
            }
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                Spacer(minLength: 0)
                VStack(spacing: 0) {
                    // Pending attachment chips (only when attachments exist)
                    if !composerState.pendingAttachments.isEmpty {
                        pendingAttachmentChips
                            .padding(.horizontal, 14)
                            .padding(.top, 10)
                    }

                    // Text input
                    ChatInputTextView(
                        text: $composerState.inputText,
                        placeholder: "Ask anything\u{2026}",
                        font: .systemFont(ofSize: 14),
                        textColor: NSColor(theme.textPrimary),
                        placeholderColor: NSColor(theme.textTertiary),
                        maxLines: 8,
                        isDisabled: isGenerating,
                        onSubmit: { sendAction() },
                        onPasteFiles: { urls in
                            for url in urls { onAddAttachment(url) }
                        },
                        contentHeight: $inputContentHeight
                    )
                    .frame(height: inputContentHeight)
                    .focused($isInputFocused)
                    .padding(.horizontal, 14)
                    .padding(.top, composerState.pendingAttachments.isEmpty ? 10 : 6)
                    .popover(isPresented: mentionPopoverBinding, arrowEdge: .top) {
                        mentionPopoverContent
                    }

                    if let warning = composerState.inlineWarning {
                        HStack(spacing: 6) {
                            Image(systemName: "exclamationmark.triangle")
                                .font(.system(size: 11))
                                .foregroundColor(theme.textTertiary)
                            Text(warning)
                                .font(.system(size: 11))
                                .foregroundColor(theme.textTertiary)
                                .lineLimit(2)
                            Spacer(minLength: 0)
                        }
                        .padding(.horizontal, 14)
                        .padding(.top, 6)
                    }

                    // Icon toolbar row
                    HStack(spacing: 0) {
                        // Left: attach button
                        Button {
                            showAttachmentMenu()
                        } label: {
                            Image(systemName: "plus")
                                .font(.system(size: 15, weight: .medium))
                                .foregroundColor(theme.textSecondary)
                                .frame(width: 28, height: 28)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .help("Attach file")

                        Menu {
                            Button("Documents") {
                                onOpenMentionPicker(.docs)
                            }
                            Button("Images") {
                                onOpenMentionPicker(.img)
                            }
                        } label: {
                            Image(systemName: "at")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(theme.textSecondary)
                                .frame(width: 28, height: 28)
                                .contentShape(Rectangle())
                        }
                        .menuStyle(.borderlessButton)
                        .help("Insert @docs/@img mention")

                        Spacer()

                        // Agent mode toggle — only shown when sandbox is available
                        if sandboxReady {
                            Button {
                                codeActEnabled.toggle()
                            } label: {
                                Image(systemName: "cpu")
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundColor(codeActEnabled ? theme.accent : theme.textTertiary)
                                    .frame(width: 28, height: 28)
                                    .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .help(codeActEnabled ? "Agent mode ON — disable CodeAct loop" : "Enable Agent mode (CodeAct)")
                        }

                        // Right: send / stop dual-mode button.
                        Button {
                            isStopMode ? stopAction() : sendAction()
                        } label: {
                            Image(systemName: isStopMode ? "stop.fill" : "arrow.up")
                                .font(.system(size: 13, weight: .bold))
                                .foregroundColor(
                                    isStopMode
                                        ? .white
                                        : (canSend ? .white : theme.textTertiary)
                                )
                                .frame(width: 28, height: 28)
                                .background(
                                    Circle()
                                        .fill(
                                            isStopMode
                                                ? theme.accent
                                                : (canSend ? theme.accent : theme.sendButtonEmptyOverlay)
                                        )
                                )
                        }
                        .accessibilityLabel(isStopMode ? "Stop generating" : "Send message")
                        .buttonStyle(.plain)
                        .disabled(isStopMode ? false : !canSend)
                    }
                    .padding(.horizontal, 10)
                    .padding(.bottom, 8)
                    .padding(.top, 4)
                }
                .background(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(theme.inputBg)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .stroke(theme.inputBorder, lineWidth: 0.5)
                )
                .frame(maxWidth: theme.maxContentWidth)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
        }
        .background(theme.chatBg)
        .onAppear { isInputFocused = true }
        .onChange(of: isReady) { _, newValue in
            if newValue { isInputFocused = true }
        }
        .onChange(of: isGenerating) { _, generating in
            if !generating { isInputFocused = true }
        }
        .onChange(of: composerState.inputText) { _, newValue in
            onComposerTextChanged(newValue)
        }
    }

    private var pendingAttachmentChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(composerState.pendingAttachments) { att in
                    AttachmentChip(
                        attachment: att,
                        theme: theme,
                        onRemove: { onRemoveAttachment(att.id) }
                    )
                }
            }
        }
    }

    private func showAttachmentMenu() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [
            .plainText, .json, .xml, .html, .pdf, .jpeg, .png, .gif, .webP, .heic, .bmp, .tiff,
            .init(filenameExtension: "md"), .init(filenameExtension: "py"),
            .init(filenameExtension: "swift"), .init(filenameExtension: "csv"),
            .init(filenameExtension: "docx"), .init(filenameExtension: "pptx"),
            .init(filenameExtension: "xlsx"),
        ].compactMap { $0 }
        panel.message = "Select files to attach"
        if panel.runModal() == .OK {
            for url in panel.urls {
                onAddAttachment(url)
            }
        }
    }

    private func sendAction() {
        onSend()
        isInputFocused = true
    }

    private func stopAction() {
        onStop()
        isInputFocused = true
    }

    private var mentionPopoverContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(composerState.activeMentionKind == .img ? "Images" : "Documents")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(theme.textSecondary)

            ForEach(composerState.mentionSuggestions) { suggestion in
                Button {
                    onSelectMentionSuggestion(suggestion)
                    isInputFocused = true
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: suggestion.kind == .img ? "photo" : "doc.text")
                            .font(.system(size: 11))
                            .foregroundColor(theme.textSecondary)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(suggestion.alias)
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(theme.textPrimary)
                                .lineLimit(1)
                            Text(suggestion.filename)
                                .font(.system(size: 10))
                                .foregroundColor(theme.textTertiary)
                                .lineLimit(1)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.vertical, 4)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(12)
        .frame(width: 320)
    }
}

// MARK: - Attachment Chip (pure SwiftUI — no NSViewRepresentable)

struct AttachmentChip: View {
    let attachment: PendingAttachment
    let theme: ThemeColors
    let onRemove: () -> Void

    private var icon: String {
        switch attachment.type {
        case .image: return "photo"
        case .pdf: return "doc.richtext"
        case .textFile: return "doc.text"
        }
    }

    private var sizeLabel: String {
        let kb = attachment.fileSize / 1024
        if kb < 1024 {
            return "\(kb) KB"
        } else {
            let mb = Double(attachment.fileSize) / (1024 * 1024)
            return String(format: "%.1f MB", mb)
        }
    }

    var body: some View {
        HStack(spacing: 6) {
            if let thumb = attachment.thumbnailImage {
                Image(nsImage: thumb)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 32, height: 32)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            } else {
                Image(systemName: icon)
                    .font(.system(size: 14))
                    .foregroundColor(theme.accent)
                    .frame(width: 32, height: 32)
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(attachment.filename)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(theme.textPrimary)
                    .lineLimit(1)
                Text(sizeLabel)
                    .font(.system(size: 9))
                    .foregroundColor(theme.textTertiary)
            }
            Button {
                onRemove()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(theme.textTertiary)
                    .frame(width: 16, height: 16)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(theme.inputBg.opacity(0.8))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(theme.inputBorder, lineWidth: 0.5)
        )
    }
}

// MARK: - Sent Attachment Chip (read-only, no dismiss button)

struct SentAttachmentChip: View {
    let attachment: MessageAttachment
    let theme: ThemeColors

    private var icon: String {
        switch attachment.type {
        case .image: return "photo"
        case .pdf: return "doc.richtext"
        case .textFile: return "doc.text"
        }
    }

    var body: some View {
        HStack(spacing: 6) {
            if attachment.type == .image,
               let thumbData = attachment.thumbnailData,
               let nsImage = NSImage(data: thumbData) {
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 28, height: 28)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            } else {
                Image(systemName: icon)
                    .font(.system(size: 12))
                    .foregroundColor(theme.accent)
                    .frame(width: 28, height: 28)
            }
            Text(attachment.filename)
                .font(.system(size: 11))
                .foregroundColor(theme.textPrimary)
                .lineLimit(1)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(theme.userBubbleBg.opacity(0.7))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(theme.inputBorder, lineWidth: 0.5)
        )
    }
}

// MARK: - Conversation Row

struct ConversationRow: View {
    let conversation: Conversation
    let isSelected: Bool
    let isEditing: Bool
    @Binding var editingTitle: String
    var onSaveEdit: () -> Void
    var onCancelEdit: () -> Void

    @Environment(\.theme) private var theme
    @State private var isHovered = false
    @FocusState private var isTitleFieldFocused: Bool

    private static let relativeDateFormatter: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter
    }()

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: conversation.parentConversationID != nil ? "arrow.triangle.branch" : "bubble.left")
                .font(.system(size: 12))
                .foregroundColor(conversation.parentConversationID != nil ? theme.accent : theme.textSecondary)
            VStack(alignment: .leading, spacing: 2) {
                if isEditing {
                    TextField("Title", text: $editingTitle)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))
                        .foregroundColor(theme.textPrimary)
                        .lineLimit(1)
                        .focused($isTitleFieldFocused)
                        .onSubmit {
                            onSaveEdit()
                        }
                        .task {
                            // Auto-focus when entering edit mode
                            isTitleFieldFocused = true
                        }
                        // Note: Escape key handling would require @available checks
                        // Focus loss (clicking elsewhere) triggers cancel via onChange
                } else {
                    Text(conversation.title)
                        .font(.system(size: 13))
                        .foregroundColor(theme.textPrimary)
                        .lineLimit(1)
                }
                Text(Self.relativeDateFormatter.localizedString(
                    for: conversation.updatedAt, relativeTo: Date()
                ))
                .font(.system(size: 11))
                .foregroundColor(theme.textTertiary)
            }
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(isSelected
                      ? theme.selectedRowOverlay
                      : isHovered ? theme.hoverRowOverlay : Color.clear)
        )
        .padding(.horizontal, 10)
        .onHover { isHovered = $0 }
        .onChange(of: isEditing) { wasEditing, isEditingNow in
            if wasEditing && !isEditingNow {
                // Lost edit mode - ensure focus is cleared
                isTitleFieldFocused = false
            }
        }
    }
}

/// Renders the active streaming assistant preview inline with transcript rows
/// while keeping it outside persisted message history.
struct LiveAssistantInlineRow: View {
    @ObservedObject var state: LiveAssistantState

    var body: some View {
        Group {
            if let message = state.message {
                MessageRow(message: message, showsOverflowActions: false)
                    .id(message.id)
                    .transition(.opacity)
            }
        }
        // Animation is handled by withAnimation at the mutation site in ChatViewModel.
        // Applying .animation here would animate ALL layout changes (including streaming
        // content updates), causing "onChange(of: Layout)" cycling faults.
    }
}

// MARK: - Message Row

struct MessageRow: View {
    let message: ChatMessage
    var showsOverflowActions = true
    @EnvironmentObject private var viewModel: ChatViewModel
    @Environment(\.theme) private var theme
    @Environment(\.chatFontSize) private var chatFontSize
    @State private var copied = false

    var body: some View {
        HStack(spacing: 0) {
            Spacer(minLength: 0)
            VStack(alignment: .leading, spacing: 0) {
                if message.role == .user {
                    userRow
                } else {
                    assistantRow
                }
            }
            .frame(maxWidth: theme.maxContentWidth, alignment: .leading)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 20)
    }

    /// Closure that forwards code to the sandbox for execution.
    private var sandboxRunHandler: (@MainActor (String) async -> RunOutput?)? {
        guard viewModel.isSandboxAvailable else { return nil }
        return { [viewModel] code in
            await viewModel.executeCodeBlock(code)
        }
    }

    /// Closure that populates the composer with a prompt (e.g. "Fix with AI").
    private var composerFillHandler: @MainActor (String) -> Void {
        { [viewModel] prompt in
            viewModel.composerState.inputText = prompt
        }
    }

    /// Async handler for code follow-up actions (Ask AI menu).
    /// For Python "review" actions, runs AST analysis before composing the prompt.
    private var codeActionHandler: @MainActor (CodeFollowUpAction, String, String?) async -> Void {
        { [viewModel] action, code, language in
            let isPython = language == "python" || language == "py"
            var analysis: CodeAnalysis?
            if action == .review && isPython {
                analysis = await viewModel.analyzeCodeForReview(code)
            }
            let prompt = viewModel.composeCodeActionPrompt(
                action: action, code: code, language: language, analysis: analysis
            )
            viewModel.composerState.inputText = prompt
        }
    }

    private var copyButton: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(message.content, forType: .string)
            copied = true
            Task {
                try? await Task.sleep(for: .seconds(1.5))
                copied = false
            }
        } label: {
            Image(systemName: copied ? "checkmark" : "doc.on.doc")
                .font(.system(size: 11))
                .foregroundColor(copied ? theme.accent : theme.textTertiary)
        }
        .buttonStyle(.plain)
        .help("Copy message")
    }

    private var userRow: some View {
        HStack {
            Spacer(minLength: 40)
            VStack(alignment: .trailing, spacing: 4) {
                if !message.attachments.isEmpty {
                    sentAttachmentChips
                }
                EquatableView(content: MessageContentView(
                    messageID: message.id,
                    content: message.content,
                    fontSize: chatFontSize,
                    role: message.role
                ))
                .environment(\.runCodeHandler, sandboxRunHandler)
                .environment(\.isSandboxReady, viewModel.isSandboxAvailable)
                .environment(\.sendCodeToComposer, composerFillHandler)
                .environment(\.codeActionHandler, codeActionHandler)
                .fixedSize(horizontal: false, vertical: true)
                .lineSpacing(4)
                .padding(EdgeInsets(top: 10, leading: 16, bottom: 10, trailing: 16))
                .background(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(theme.userBubbleBg)
                )
                copyButton
            }
        }
        .padding(.vertical, 8)
    }

    private var sentAttachmentChips: some View {
        HStack(spacing: 6) {
            ForEach(message.attachments) { att in
                SentAttachmentChip(attachment: att, theme: theme)
            }
        }
    }

    private var assistantRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                // Avatar
                ZStack {
                    Circle()
                        .fill(theme.avatarOverlay)
                        .frame(width: 28, height: 28)
                    Text("\u{1F999}")
                        .font(.system(size: 14))
                }
                .padding(.top, 2)

                VStack(alignment: .leading, spacing: 6) {
                    // Collapsible thinking disclosure
                    if message.thinking != nil {
                        ThinkingDisclosure(
                            thinking: message.thinking ?? "",
                            durationSecs: message.thinkingDurationSecs
                        )
                    }

                    // Answer text (markdown + LaTeX)
                    EquatableView(content: MessageContentView(
                        messageID: message.id,
                        content: message.content,
                        fontSize: chatFontSize,
                        role: message.role
                    ))
                    .environment(\.runCodeHandler, sandboxRunHandler)
                    .environment(\.isSandboxReady, viewModel.isSandboxAvailable)
                    .environment(\.sendCodeToComposer, composerFillHandler)
                    .environment(\.codeActionHandler, codeActionHandler)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineSpacing(5)

                    // Metrics + Copy + Quote + Overflow
                    HStack(spacing: 8) {
                        if let metrics = message.metrics {
                            Text(metrics)
                                .font(.system(size: 11))
                                .monospacedDigit()
                                .foregroundColor(theme.textTertiary)
                        }
                        copyButton
                        quoteButton
                        if showsOverflowActions {
                            overflowMenu
                        }
                    }
                }
            }
        }
        .padding(.vertical, 12)
    }

    private var quoteButton: some View {
        Button {
            quoteMessageIntoComposer()
        } label: {
            Image(systemName: "text.quote")
                .font(.system(size: 11))
                .foregroundColor(theme.textTertiary)
        }
        .buttonStyle(.plain)
        .help("Quote & follow up (\u{2318}\u{21E7}Q)")
    }

    private func quoteMessageIntoComposer() {
        var textToQuote = message.content

        // Try to get selected text from the active text view
        if let responder = NSApp.keyWindow?.firstResponder as? NSTextView {
            let range = responder.selectedRange()
            if range.length > 0, let str = responder.string as NSString? {
                let selected = str.substring(with: range)
                if !selected.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    textToQuote = selected
                }
            }
        }

        // Truncate very long quotes to keep the composer manageable
        let maxQuoteLength = 2000
        if textToQuote.count > maxQuoteLength {
            textToQuote = String(textToQuote.prefix(maxQuoteLength)) + "\u{2026}"
        }

        // Format as markdown blockquote
        let quoted = textToQuote
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { "> \($0)" }
            .joined(separator: "\n")
        viewModel.composerState.inputText = quoted + "\n\n"
    }

    private var overflowMenu: some View {
        Menu {
            Button {
                quoteMessageIntoComposer()
            } label: {
                Label("Quote & follow up", systemImage: "text.quote")
            }

            Button {
                if let idx = viewModel.messages.firstIndex(where: { $0.id == message.id }) {
                    Task { await viewModel.branchConversation(atMessageIndex: idx) }
                }
            } label: {
                Label("Branch in new chat", systemImage: "arrow.triangle.branch")
            }

            Menu {
                Button {
                    if let idx = viewModel.messages.firstIndex(where: { $0.id == message.id }) {
                        Task { await viewModel.retryResponse(atIndex: idx) }
                    }
                } label: {
                    Label("Try again", systemImage: "arrow.counterclockwise")
                }
                Button {
                    if let idx = viewModel.messages.firstIndex(where: { $0.id == message.id }) {
                        Task { await viewModel.retryResponse(atIndex: idx, instruction: "Please provide more detail in your response.") }
                    }
                } label: {
                    Label("Add details", systemImage: "text.append")
                }
                Button {
                    if let idx = viewModel.messages.firstIndex(where: { $0.id == message.id }) {
                        Task { await viewModel.retryResponse(atIndex: idx, instruction: "Please be more concise.") }
                    }
                } label: {
                    Label("More concise", systemImage: "text.badge.minus")
                }
            } label: {
                Label("Retry", systemImage: "arrow.counterclockwise")
            }
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 11))
                .foregroundColor(theme.textTertiary)
                .frame(width: 20, height: 20)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
    }
}

// MARK: - Thinking Disclosure (ChatGPT-style collapsible)

struct ThinkingDisclosure: View {
    let thinking: String
    let durationSecs: Double?
    @Environment(\.theme) private var theme
    @State private var isExpanded = false

    private var trimmedThinking: String {
        thinking.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var durationLabel: String {
        guard let secs = durationSecs else { return "" }
        let clamped = max(0, secs)
        if clamped < 60 {
            return String(format: "%.0fs", clamped)
        } else {
            let mins = Int(clamped) / 60
            let remainder = Int(clamped) % 60
            return "\(mins)m \(remainder)s"
        }
    }

    private var titleLabel: String {
        if durationSecs == nil {
            return "Thinking"
        }
        return "Thought for \(durationLabel)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header button
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(theme.textTertiary)
                        .frame(width: 12)

                    Text(titleLabel)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(theme.textSecondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            // Expanded thinking content — animation scoped to this block only.
            // compositingGroup prevents stacking/clipping artifacts when ScrollView
            // reflows during the animated height change.
            Group {
                if isExpanded, !trimmedThinking.isEmpty {
                    HStack(alignment: .top, spacing: 0) {
                        // Vertical accent line
                        RoundedRectangle(cornerRadius: 1)
                            .fill(theme.textTertiary.opacity(0.5))
                            .frame(width: 2)
                            .padding(.leading, 5)

                        // Thinking text
                        Text(trimmedThinking)
                            .font(.system(size: 12.5))
                            .foregroundColor(theme.textSecondary)
                            .lineSpacing(4)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.leading, 12)
                            .padding(.vertical, 6)
                    }
                    .padding(.top, 4)
                    .padding(.bottom, 4)
                    .transition(.opacity)
                }
            }
            .compositingGroup()
            .animation(.easeInOut(duration: 0.2), value: isExpanded)
        }
        .padding(.bottom, 4)
    }
}

// MARK: - Branched From Divider

struct BranchedFromDivider: View {
    let parentTitle: String
    let onNavigate: () -> Void
    @Environment(\.theme) private var theme

    var body: some View {
        HStack(spacing: 0) {
            Spacer(minLength: 0)
            HStack(spacing: 10) {
                theme.divider.frame(height: 0.5)
                Button {
                    onNavigate()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.triangle.branch")
                            .font(.system(size: 10))
                        Text("Branched from \(parentTitle)")
                            .font(.system(size: 11))
                            .lineLimit(1)
                    }
                    .foregroundColor(theme.accent)
                }
                .buttonStyle(.plain)
                theme.divider.frame(height: 0.5)
            }
            .frame(maxWidth: theme.maxContentWidth)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
    }
}

// MARK: - Typing Indicator

struct TypingIndicator: View {
    @Environment(\.theme) private var theme
    @State private var phase: CGFloat = 0

    var body: some View {
        HStack(spacing: 0) {
            Spacer(minLength: 0)
            HStack(alignment: .top, spacing: 10) {
                ZStack {
                    Circle()
                        .fill(theme.avatarOverlay)
                        .frame(width: 28, height: 28)
                    Text("\u{1F999}")
                        .font(.system(size: 14))
                }
                .padding(.top, 2)

                HStack(spacing: 5) {
                    ForEach(0..<3, id: \.self) { i in
                        Circle()
                            .fill(theme.textTertiary)
                            .frame(width: 6, height: 6)
                            .offset(y: phase == 1 ? -3 : 3)
                            .animation(
                                .easeInOut(duration: 0.45)
                                .repeatForever(autoreverses: true)
                                .delay(Double(i) * 0.12),
                                value: phase
                            )
                    }
                }
                .frame(height: 12)
                .clipped()
                .padding(.top, 10)

                Spacer()
            }
            .frame(maxWidth: theme.maxContentWidth, alignment: .leading)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .onAppear { phase = 1 }
    }
}
