import Foundation
import OSLog
import LlamaInferenceCore

/// ViewModel for the Model Hub — manages search, filtering, selection, and detail state.
@Observable
@MainActor
final class ModelHubViewModel {
    var searchText: String = "" {
        didSet {
            guard !isRestoringPersistedState else { return }
            persistSearchText()
            onBrowseSearchTextChanged()
        }
    }
    var downloadedSearchText: String = "" {
        didSet {
            guard !isRestoringPersistedState else { return }
            persistDownloadedSearchText()
            onDownloadedSearchTextChanged()
        }
    }
    var results: [HFModelSummary] = []
    var isSearching: Bool = false
    var errorMessage: String?
    var showDownloadedOnly: Bool = false {
        didSet {
            guard !isRestoringPersistedState else { return }
            persistShowDownloadedOnly()
        }
    }
    var sortOrder: HubSortOrder = .downloads {
        didSet {
            guard !isRestoringPersistedState else { return }
            if oldValue != sortOrder {
                persistSortOrder()
                onFilterChanged()
            }
        }
    }
    /// Current selection in the model list.
    var selectedModelId: String? {
        didSet {
            guard oldValue != selectedModelId else { return }
            Task { await selectionDidChange() }
        }
    }
    /// Cached file lists for selected models (keyed by repo id).
    var modelFiles: [String: [GGUFFile]] = [:]
    /// Loading state for file detail fetch.
    var loadingDetailId: String?
    /// Cached README snippets (keyed by repo id). nil = not fetched, empty = no README.
    var readmeSnippets: [String: String] = [:]
    /// Current README fetches.
    var loadingReadmeIds: Set<String> = []
    /// Pagination state.
    var canLoadMore: Bool = false
    var isLoadingMore: Bool = false
    private var nextPageURL: URL?
    private let pageSize: Int = 20

    /// When showDownloadedOnly is true, results come from fetchDownloadedModels (not filtered pagination).
    private var downloadedResults: [HFModelSummary] = []
    private var lastDownloadedFetch: Date?
    private var lastDownloadedDiscoveryBases: Set<String> = []

    private let api = HuggingFaceAPI.shared
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "ModelHub")
    private var searchTask: Task<Void, Never>?
    private var loadMoreTask: Task<Void, Never>?
    private var hasFetchedCurated = false
    private var cachedDiscoveredNames: Set<String>?
    private var lastDiscoveryScan: Date?
    private var remoteGGUFFilenameCache: [String: Set<String>] = [:]
    private var downloadedSourceIndex: [String: String] = [:]
    private var lastDownloadedSourceIndexLoad: Date?
    private var isRestoringPersistedState = false

    private static let downloadedSourceIndexTTL: TimeInterval = 120
    private static let downloadSourceMapFilename = ".modelhub-download-sources.json"
    private static let readmeCacheTTL: TimeInterval = 60 * 60 * 6

    var onModelApplied: (() -> Void)?

    init() {
        restorePersistedState()
    }

    // MARK: - Selection Helpers

    var selectedModel: HFModelSummary? {
        guard let selectedModelId else { return nil }
        return modelForSelection(id: selectedModelId)
    }

    var selectedFiles: [GGUFFile]? {
        guard let selectedModelId else { return nil }
        return modelFiles[selectedModelId]
    }

    var selectedReadmeSnippet: String? {
        guard let selectedModelId else { return nil }
        return readmeSnippets[selectedModelId]
    }

    private func modelForSelection(id: String) -> HFModelSummary? {
        if let match = displayedResults.first(where: { $0.id == id }) {
            return match
        }
        if let match = downloadedResults.first(where: { $0.id == id }) {
            return match
        }
        return results.first { $0.id == id }
    }

    var isLoadingSelectedFiles: Bool {
        selectedModelId == loadingDetailId
    }

    var isLoadingSelectedReadme: Bool {
        guard let selectedModelId else { return false }
        return loadingReadmeIds.contains(selectedModelId)
    }

    // MARK: - Search

    private func onBrowseSearchTextChanged() {
        searchTask?.cancel()
        searchTask = Task {
            do {
                try await Task.sleep(for: .milliseconds(300))
            } catch {
                return
            }
            guard !Task.isCancelled else { return }
            guard !showDownloadedOnly else { return }
            await performSearch()
        }
    }

    private func onDownloadedSearchTextChanged() {
        searchTask?.cancel()
        searchTask = Task {
            do {
                try await Task.sleep(for: .milliseconds(300))
            } catch {
                return
            }
            guard !Task.isCancelled else { return }
            await fetchDownloadedModels()
        }
    }

    private func onFilterChanged() {
        searchTask?.cancel()
        hasFetchedCurated = false
        searchTask = Task {
            if showDownloadedOnly {
                await fetchDownloadedModels()
            } else {
                await performSearch()
            }
        }
    }

    func performSearch() async {
        guard !showDownloadedOnly else {
            await fetchDownloadedModels()
            return
        }

        let parsed = ModelHubSearchParser.parse(searchText)
        let query = parsed.query
        let authorOverride = parsed.authorOverride
        let effectiveAuthor = parsed.authorOverride

        if query.isEmpty, effectiveAuthor == nil, !hasFetchedCurated {
            await fetchCurated()
            return
        }
        if query.isEmpty, effectiveAuthor == nil {
            return
        }

        isSearching = true
        errorMessage = nil
        nextPageURL = nil
        canLoadMore = false
        loadMoreTask?.cancel()
        loadMoreTask = nil

        do {
            let apiSort = sortOrder.apiParam ?? "downloads"
            var page = try await api.search(
                query: query,
                author: effectiveAuthor,
                sort: apiSort,
                limit: pageSize
            )
            guard !Task.isCancelled else { return }

            // Provider-only lookup fallback:
            // If `@author` yields no GGUF results, retry as a plain text search
            // so model-family handles like "@nanbeige" can still discover quantized repos.
            if page.models.isEmpty, query.isEmpty, let authorOverride {
                logger.debug("No results for author=@\(authorOverride, privacy: .public); retrying text search")
                page = try await api.search(
                    query: authorOverride,
                    author: nil,
                    sort: apiSort,
                    limit: pageSize
                )
                guard !Task.isCancelled else { return }
            }

            if sortOrder == .name {
                results = page.models.sorted {
                    $0.modelName.localizedCaseInsensitiveCompare($1.modelName) == .orderedAscending
                }
            } else {
                results = page.models
            }
            nextPageURL = page.nextPageURL
            canLoadMore = page.nextPageURL != nil
            synchronizeSelection()
        } catch {
            guard !Task.isCancelled else { return }
            errorMessage = String(describing: error)
            logger.error("Search failed: \(String(describing: error), privacy: .public)")
        }

        isSearching = false
    }

    /// Force-refresh current Model Hub state (curated/search/downloaded).
    func performRefresh() async {
        searchTask?.cancel()
        loadMoreTask?.cancel()
        loadMoreTask = nil
        nextPageURL = nil
        canLoadMore = false

        if showDownloadedOnly {
            invalidateDownloadedCache()
            await fetchDownloadedModels()
            return
        }

        hasFetchedCurated = false
        await performSearch()
    }

    /// Trigger pagination from the view (stores task for cancellation).
    func triggerLoadMore() {
        guard canLoadMore, !isLoadingMore else { return }
        loadMoreTask = Task { await loadMore() }
    }

    /// Load the next page of results (infinite scroll).
    func loadMore() async {
        guard canLoadMore, !isLoadingMore else { return }

        isLoadingMore = true
        defer { isLoadingMore = false }
        guard let url = nextPageURL else { return }

        do {
            let page = try await api.fetchPage(url: url)
            guard !Task.isCancelled else { return }

            let existingIds = Set(results.map { $0.id })
            let newResults = page.models.filter { !existingIds.contains($0.id) }
            results.append(contentsOf: newResults)
            nextPageURL = page.nextPageURL
            canLoadMore = page.nextPageURL != nil
            synchronizeSelection()
        } catch {
            guard !Task.isCancelled else { return }
            logger.error("Load more failed: \(String(describing: error), privacy: .public)")
        }

    }

    // MARK: - Curated

    func fetchCurated() async {
        guard !hasFetchedCurated else { return }
        isSearching = true
        errorMessage = nil

        do {
            let apiSort = sortOrder.apiParam ?? "downloads"
            let page = try await api.search(
                query: "",
                author: nil,
                sort: apiSort,
                limit: pageSize
            )
            guard !Task.isCancelled else { return }

            results = page.models
            nextPageURL = page.nextPageURL
            canLoadMore = page.nextPageURL != nil
            hasFetchedCurated = true
            synchronizeSelection()
        } catch {
            guard !Task.isCancelled else { return }
            errorMessage = String(describing: error)
        }

        isSearching = false
    }

    func loadOnAppear() async {
        if showDownloadedOnly {
            await fetchDownloadedModels()
            return
        }
        if results.isEmpty {
            await performSearch()
        }
        synchronizeSelection()
    }

    // MARK: - Selection / Detail

    func selectModel(_ modelId: String) {
        selectedModelId = modelId
    }

    private func selectionDidChange() async {
        guard let selectedModelId else { return }
        await loadDetailsIfNeeded(for: selectedModelId)
    }

    private func loadDetailsIfNeeded(for repoId: String) async {
        if modelFiles[repoId] == nil {
            await fetchFiles(for: repoId)
        }
        if readmeSnippets[repoId] == nil {
            await fetchREADME(for: repoId)
        }
    }

    private func fetchFiles(for repoId: String) async {
        guard modelFiles[repoId] == nil else { return }
        loadingDetailId = repoId
        defer { loadingDetailId = nil }
        do {
            let detail = try await api.modelDetail(repoId: repoId)
            let files = await api.ggufFiles(from: detail)
            guard !Task.isCancelled else { return }
            modelFiles[repoId] = files.sorted { file1, file2 in
                // mmproj files last
                if file1.isMMProj != file2.isMMProj { return !file1.isMMProj }
                // Then by quant level order
                let idx1 = QuantLevel.allCases.firstIndex(of: file1.quantLevel ?? .Q4_K_M) ?? 0
                let idx2 = QuantLevel.allCases.firstIndex(of: file2.quantLevel ?? .Q4_K_M) ?? 0
                return idx1 < idx2
            }
        } catch {
            logger.error("Failed to fetch files for \(repoId, privacy: .public): \(String(describing: error), privacy: .public)")
        }
    }

    // MARK: - README

    func fetchREADME(for repoId: String) async {
        guard readmeSnippets[repoId] == nil, !loadingReadmeIds.contains(repoId) else { return }
        loadingReadmeIds.insert(repoId)
        defer { loadingReadmeIds.remove(repoId) }

        if let cached = loadCachedReadme(repoId: repoId) {
            readmeSnippets[repoId] = cached
            return
        }

        guard let raw = await api.fetchREADME(repoId: repoId) else {
            readmeSnippets[repoId] = ""
            return
        }
        // Strip YAML frontmatter (---...---)
        var content = raw
        if content.hasPrefix("---") {
            if let endRange = content.range(of: "---", range: content.index(content.startIndex, offsetBy: 3)..<content.endIndex) {
                content = String(content[endRange.upperBound...])
            }
        }
        // Some HF cards include raw HTML/SVG hero blocks before the first markdown
        // heading; Textual renders those as text. Start rendering at the first heading.
        if let headingRange = content.range(of: #"(?m)^#\s+\S"#, options: .regularExpression) {
            content = String(content[headingRange.lowerBound...])
        }
        content = content.trimmingCharacters(in: .whitespacesAndNewlines)
        readmeSnippets[repoId] = content
        saveCachedReadme(content, repoId: repoId)
    }

    // MARK: - Filtering

    /// Downloaded models (for "sidebar hidden" mode — separate from filter dropdown).
    var downloadedModels: [HFModelSummary] { downloadedResults }

    var displayedResults: [HFModelSummary] {
        let items: [HFModelSummary]
        if showDownloadedOnly {
            items = downloadedResults
        } else {
            items = results
        }
        if sortOrder == .name {
            return items.sorted {
                $0.modelName.localizedCaseInsensitiveCompare($1.modelName) == .orderedAscending
            }
        }
        return items
    }

    /// Cache TTL for downloaded results (HF metadata changes rarely).
    private static let downloadedCacheTTL: TimeInterval = 60

    /// Fetch HF metadata for discovered files. Call when showDownloadedOnly becomes true.
    /// Results are cached for 60s; cache invalidates when discovered set changes.
    func fetchDownloadedModels() async {
        invalidateDiscoveredCache()
        let discovered = ModelDiscovery.scan().filter { !$0.isMMProj }
        let parsed = ModelHubSearchParser.parse(downloadedSearchText)
        let query = parsed.query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let effectiveAuthor = parsed.authorOverride
        let sourceIndex = loadDownloadedSourceIndex()

        var repoIdsByLower: [String: String] = [:]
        var mappingFingerprint: Set<String> = []
        for file in discovered {
            let filename = localFilename(from: file)
            if let mappedRepoId = sourceIndex[filename] ?? inferredRepoId(from: file) {
                let lower = mappedRepoId.lowercased()
                if repoIdsByLower[lower] == nil {
                    repoIdsByLower[lower] = mappedRepoId
                }
                mappingFingerprint.insert("\(filename)|\(lower)")
            } else {
                mappingFingerprint.insert("\(filename)|unmapped")
            }
        }
        var currentCacheKey = mappingFingerprint
        currentCacheKey.insert("__query:\(query)")
        currentCacheKey.insert("__author:\((effectiveAuthor ?? "").lowercased())")

        guard !discovered.isEmpty else {
            downloadedResults = []
            lastDownloadedFetch = nil
            lastDownloadedDiscoveryBases = []
            return
        }

        // Use cache if discovered set unchanged and within TTL.
        if !downloadedResults.isEmpty,
           let last = lastDownloadedFetch,
           Date().timeIntervalSince(last) < Self.downloadedCacheTTL,
           lastDownloadedDiscoveryBases == currentCacheKey {
            return
        }

        isSearching = true
        errorMessage = nil
        defer { isSearching = false }

        var foundById: [String: HFModelSummary] = [:]

        // Strict downloaded-mode source of truth:
        // only mapped repo ids from local source metadata (JSON/log) or explicit repo-path inference.
        for repoId in repoIdsByLower.values.sorted() {
            if let exact = await fetchSummaryForRepoId(repoId, authorOverride: effectiveAuthor) {
                foundById[exact.id.lowercased()] = exact
                logger.debug("Downloaded source match repo=\(repoId, privacy: .public) -> \(exact.id, privacy: .public)")
            } else {
                logger.debug("No HF summary for mapped repo=\(repoId, privacy: .public)")
            }
        }

        var resolved = Array(foundById.values)
        if !query.isEmpty {
            resolved = resolved.filter { model in
                model.id.localizedCaseInsensitiveContains(query) ||
                model.modelName.localizedCaseInsensitiveContains(query) ||
                model.author.localizedCaseInsensitiveContains(query)
            }
        }

        downloadedResults = resolved.sorted { m1, m2 in
            m1.modelName.localizedCaseInsensitiveCompare(m2.modelName) == .orderedAscending
        }
        lastDownloadedFetch = Date()
        lastDownloadedDiscoveryBases = currentCacheKey
        synchronizeSelection(preferred: downloadedResults)
    }

    private func inferredRepoId(from discovered: DiscoveredModel) -> String? {
        let nameParts = discovered.name.split(separator: "/")
        if nameParts.count >= 3 {
            return "\(nameParts[0])/\(nameParts[1])"
        }

        let marker = "/models--"
        guard let markerRange = discovered.path.range(of: marker) else { return nil }
        let tail = discovered.path[markerRange.upperBound...]
        guard let endRange = tail.range(of: "/snapshots/") else { return nil }
        let rawRepo = String(tail[..<endRange.lowerBound])
        let repoId = rawRepo.replacingOccurrences(of: "--", with: "/")
        let components = repoId.split(separator: "/")
        return components.count == 2 ? repoId : nil
    }

    private func fetchSummaryForRepoId(_ repoId: String, authorOverride: String?) async -> HFModelSummary? {
        if let authorOverride,
           !repoId.lowercased().hasPrefix(authorOverride.lowercased() + "/") {
            return nil
        }

        do {
            let directPage = try await api.search(
                query: repoId,
                author: nil,
                sort: sortOrder.apiParam ?? "downloads",
                limit: 50
            )
            if let exact = directPage.models.first(where: { $0.id.caseInsensitiveCompare(repoId) == .orderedSame }) {
                return exact
            }
        } catch {
            logger.debug("Direct repo lookup failed for \(repoId, privacy: .public): \(String(describing: error), privacy: .public)")
        }

        let parts = repoId.split(separator: "/", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return nil }
        let fallbackAuthor = authorOverride ?? parts[0]
        do {
            let fallbackPage = try await api.search(
                query: parts[1],
                author: fallbackAuthor,
                sort: sortOrder.apiParam ?? "downloads",
                limit: 50
            )
            return fallbackPage.models.first { $0.id.caseInsensitiveCompare(repoId) == .orderedSame }
        } catch {
            logger.debug("Fallback repo lookup failed for \(repoId, privacy: .public): \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    private func bestDownloadedMatch(
        files: [DiscoveredModel],
        candidates: [HFModelSummary],
        authorOverride: String?
    ) async -> HFModelSummary? {
        let filtered: [HFModelSummary]
        if let authorOverride {
            filtered = candidates.filter { $0.author.caseInsensitiveCompare(authorOverride) == .orderedSame }
        } else {
            filtered = candidates
        }
        guard !filtered.isEmpty else { return nil }
        let groupBase = files.first.map { GGUFModelInfo.baseName(fromFilename: $0.name).lowercased() } ?? ""

        let hintedRepoIds = Set(files.compactMap(inferredRepoId(from:)).map { $0.lowercased() })
        if let hinted = filtered.first(where: { hintedRepoIds.contains($0.id.lowercased()) }) {
            return hinted
        }

        let localFilenames = Set(files.map { localFilename(from: $0) })
        let normalizedLocalFilenames = Set(localFilenames.map(normalizedGGUFFilename))
        var bestByFilename: (model: HFModelSummary, score: Int)?
        for candidate in filtered {
            let remoteFilenames = await remoteGGUFFilenames(for: candidate.id)
            guard !remoteFilenames.isEmpty else { continue }
            let exactMatches = localFilenames.intersection(remoteFilenames).count
            let normalizedRemote = Set(remoteFilenames.map(normalizedGGUFFilename))
            let normalizedMatches = normalizedLocalFilenames.intersection(normalizedRemote).count
            let score = (exactMatches * 100) + (normalizedMatches * 25)
            guard score > 0 else { continue }
            if let current = bestByFilename {
                if score > current.score ||
                    (score == current.score && candidate.downloads > current.model.downloads) ||
                    (score == current.score &&
                     candidate.downloads == current.model.downloads &&
                     candidate.likes > current.model.likes) {
                    bestByFilename = (candidate, score)
                }
            } else {
                bestByFilename = (candidate, score)
            }
        }
        if let best = bestByFilename?.model {
            return best
        }

        let exactBaseCandidates = filtered.filter {
            GGUFModelInfo.baseNameFromHFModel($0.modelName).lowercased() == groupBase
        }
        if !exactBaseCandidates.isEmpty {
            return exactBaseCandidates.max {
                if $0.downloads != $1.downloads {
                    return $0.downloads < $1.downloads
                }
                return $0.likes < $1.likes
            }
        }

        let normalizedBase = normalizedGGUFFilename(groupBase)
        let normalizedBaseCandidates = filtered.filter {
            normalizedGGUFFilename(GGUFModelInfo.baseNameFromHFModel($0.modelName)) == normalizedBase
        }
        if !normalizedBaseCandidates.isEmpty {
            return normalizedBaseCandidates.max {
                if $0.downloads != $1.downloads {
                    return $0.downloads < $1.downloads
                }
                return $0.likes < $1.likes
            }
        }

        return nil
    }

    private func remoteGGUFFilenames(for repoId: String) async -> Set<String> {
        if let cached = remoteGGUFFilenameCache[repoId] {
            return cached
        }
        do {
            let detail = try await api.modelDetail(repoId: repoId)
            let names = Set(
                detail.siblings.compactMap { sibling in
                    GGUFFile.from(sibling: sibling, repoId: detail.id)?.filename.lowercased()
                }
            )
            remoteGGUFFilenameCache[repoId] = names
            return names
        } catch {
            logger.debug("Filename verification failed for \(repoId, privacy: .public): \(String(describing: error), privacy: .public)")
            remoteGGUFFilenameCache[repoId] = []
            return []
        }
    }

    private func localFilename(from discovered: DiscoveredModel) -> String {
        (discovered.name as NSString).lastPathComponent.lowercased()
    }

    private func loadDownloadedSourceIndex(forceReload: Bool = false) -> [String: String] {
        if !forceReload,
           let last = lastDownloadedSourceIndexLoad,
           Date().timeIntervalSince(last) < Self.downloadedSourceIndexTTL {
            return downloadedSourceIndex
        }

        let directory = resolvedDownloadDirectory()
        let jsonURL = directory.appendingPathComponent(Self.downloadSourceMapFilename)
        var merged = loadDownloadSourceMap(from: jsonURL)
        let logDerived = loadDownloadSourceMapFromLogs(in: directory)
        for (filename, repoId) in logDerived where merged[filename] == nil {
            merged[filename] = repoId
        }

        downloadedSourceIndex = merged
        lastDownloadedSourceIndexLoad = Date()
        return merged
    }

    private func resolvedDownloadDirectory() -> URL {
        let stored = UserDefaults.standard.string(forKey: SettingsKeys.modelDownloadDirectory) ?? ""
        if stored.isEmpty {
            return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Models/gguf")
        }
        return URL(fileURLWithPath: stored)
    }

    private func loadDownloadSourceMap(from url: URL) -> [String: String] {
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([String: String].self, from: data) else {
            return [:]
        }
        var normalized: [String: String] = [:]
        normalized.reserveCapacity(decoded.count)
        for (filename, repoId) in decoded {
            normalized[normalizedFilename(filename)] = repoId
        }
        return normalized
    }

    private func loadDownloadSourceMapFromLogs(in directory: URL) -> [String: String] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
            return [:]
        }

        var map: [String: String] = [:]
        for logFile in entries where logFile.lastPathComponent.lowercased().hasSuffix("-download.log") {
            guard let contents = try? String(contentsOf: logFile, encoding: .utf8) else { continue }
            let parsed = parseDownloadLog(contents)
            for (filename, repoId) in parsed where map[filename] == nil {
                map[filename] = repoId
            }
        }
        return map
    }

    private func parseDownloadLog(_ text: String) -> [String: String] {
        var map: [String: String] = [:]
        var activeRepoId: String?
        var pendingFilename: String?

        for line in text.split(whereSeparator: \.isNewline).map(String.init) {
            if let jsonEntry = decodeDownloadSourceLogEntry(line),
               !jsonEntry.repoId.isEmpty,
               !jsonEntry.filename.isEmpty {
                map[normalizedFilename(jsonEntry.filename)] = jsonEntry.repoId
                pendingFilename = nil
                continue
            }
            if let repo = extractRepoId(fromLogLine: line) {
                activeRepoId = repo
            }
            if let commandRepo = firstRegexCapture("(?:huggingface-cli|hf)\\s+download\\s+([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", in: line),
               let commandFile = firstRegexCapture("(?:huggingface-cli|hf)\\s+download\\s+[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\\s+([^\\s]+\\.gguf)", in: line) {
                map[normalizedFilename(commandFile)] = commandRepo
                pendingFilename = nil
                continue
            }
            if let urlRepo = firstRegexCapture("huggingface\\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/resolve/[^/]+/[^\\s?]+\\.gguf", in: line),
               let urlFile = firstRegexCapture("huggingface\\.co/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/resolve/[^/]+/([^\\s?]+\\.gguf)", in: line) {
                map[normalizedFilename(urlFile)] = urlRepo
                pendingFilename = nil
                continue
            }
            if let downloading = firstRegexCapture("Downloading '([^']+\\.gguf)'", in: line) {
                pendingFilename = downloading
            }
            if let moved = firstRegexCapture("Moving file to\\s+([^\\s]+\\.gguf)", in: line) {
                if let activeRepoId {
                    map[normalizedFilename(moved)] = activeRepoId
                }
                pendingFilename = nil
                continue
            }
            if let bare = firstRegexCapture("^\\s*([^\\s]+\\.gguf)\\s*$", in: line),
               let activeRepoId {
                map[normalizedFilename(bare)] = activeRepoId
                continue
            }
            if line.localizedCaseInsensitiveContains("download complete"),
               let pendingFilename,
               let activeRepoId {
                map[normalizedFilename(pendingFilename)] = activeRepoId
            }
        }
        return map
    }

    private func decodeDownloadSourceLogEntry(_ line: String) -> DownloadSourceLogLine? {
        guard line.first == "{",
              let data = line.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(DownloadSourceLogLine.self, from: data) else {
            return nil
        }
        return decoded
    }

    private func extractRepoId(fromLogLine line: String) -> String? {
        if let fromURL = firstRegexCapture("huggingface\\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", in: line) {
            return fromURL
        }
        if let fromCommand = firstRegexCapture("(?:huggingface-cli|hf)\\s+download\\s+([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", in: line) {
            return fromCommand
        }
        return nil
    }

    private func firstRegexCapture(_ pattern: String, in text: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, range: range),
              match.numberOfRanges > 1,
              let captureRange = Range(match.range(at: 1), in: text) else {
            return nil
        }
        return String(text[captureRange])
    }

    private func normalizedFilename(_ value: String) -> String {
        (value as NSString).lastPathComponent.lowercased()
    }

    private func normalizedGGUFFilename(_ value: String) -> String {
        let stem = (value as NSString).deletingPathExtension.lowercased()
        return stem.replacingOccurrences(
            of: "[^a-z0-9]",
            with: "",
            options: .regularExpression
        )
    }

    private func readmeCacheURL(for repoId: String) -> URL {
        let appSupport = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/LlamaInferenceDemo/ModelHubReadmeCache")
        let safeRepo = repoId.replacingOccurrences(of: "/", with: "__")
        return appSupport.appendingPathComponent("\(safeRepo).md")
    }

    private func loadCachedReadme(repoId: String) -> String? {
        let url = readmeCacheURL(for: repoId)
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return nil }

        if let attrs = try? fm.attributesOfItem(atPath: url.path),
           let modified = attrs[.modificationDate] as? Date,
           Date().timeIntervalSince(modified) > Self.readmeCacheTTL {
            return nil
        }

        guard let data = try? Data(contentsOf: url),
              let content = String(data: data, encoding: .utf8) else {
            return nil
        }
        return content
    }

    private func saveCachedReadme(_ readme: String, repoId: String) {
        let url = readmeCacheURL(for: repoId)
        let fm = FileManager.default
        let dir = url.deletingLastPathComponent()
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        guard let data = readme.data(using: .utf8) else { return }
        try? data.write(to: url, options: .atomic)
    }

    /// Cache ModelDiscovery scan for 10s to avoid repeated filesystem scans.
    private func discoveredModelNames() -> Set<String> {
        if let cached = cachedDiscoveredNames,
           let lastScan = lastDiscoveryScan,
           Date().timeIntervalSince(lastScan) < 10 {
            return cached
        }
        let names = Set(ModelDiscovery.scan().map { $0.name })
        cachedDiscoveredNames = names
        lastDiscoveryScan = Date()
        return names
    }

    /// Invalidate discovered cache so next access rescans (e.g. after filter change or download).
    func invalidateDiscoveredCache() {
        cachedDiscoveredNames = nil
        lastDiscoveryScan = nil
    }

    /// Invalidate downloaded results cache so next fetchDownloadedModels will hit the API.
    func invalidateDownloadedCache() {
        lastDownloadedFetch = nil
        lastDownloadedDiscoveryBases = []
        remoteGGUFFilenameCache.removeAll()
        downloadedSourceIndex.removeAll()
        lastDownloadedSourceIndexLoad = nil
    }

    // MARK: - "Use as" Wiring

    func applyModel(path: String, role: ModelRole, architecture: String? = nil) {
        let key: String
        switch role {
        case .chatModel: key = SettingsKeys.modelPath
        case .vlmModel: key = SettingsKeys.vlmModelPath
        case .vlmProjection: key = SettingsKeys.vlmClipPath
        case .summarizer: key = SettingsKeys.summarizerModelPath
        }
        UserDefaults.standard.set(path, forKey: key)
        
        if role == .vlmModel {
            if let arch = architecture, !arch.isEmpty {
                UserDefaults.standard.set(arch, forKey: SettingsKeys.vlmArchitecture)
                logger.info("Saved VLM architecture: \(arch, privacy: .public)")
            } else {
                UserDefaults.standard.removeObject(forKey: SettingsKeys.vlmArchitecture)
            }
        }
        
        logger.info("Applied model: \(path, privacy: .public) as \(role.rawValue, privacy: .public)")
        onModelApplied?()
    }

    // MARK: - Persistence

    private func restorePersistedState() {
        let ud = UserDefaults.standard
        isRestoringPersistedState = true
        searchText = ud.string(forKey: SettingsKeys.modelHubSearchText) ?? SettingsDefaults.modelHubSearchText
        downloadedSearchText = ud.string(forKey: SettingsKeys.modelHubDownloadedSearchText)
            ?? SettingsDefaults.modelHubDownloadedSearchText
        sortOrder = HubSortOrder.fromStored(
            ud.string(forKey: SettingsKeys.modelHubSortOrder) ?? SettingsDefaults.modelHubSortOrder
        )
        if ud.object(forKey: SettingsKeys.modelHubShowDownloadedOnly) == nil {
            showDownloadedOnly = SettingsDefaults.modelHubShowDownloadedOnly
        } else {
            showDownloadedOnly = ud.bool(forKey: SettingsKeys.modelHubShowDownloadedOnly)
        }
        isRestoringPersistedState = false
    }

    private func persistSearchText() {
        UserDefaults.standard.set(searchText, forKey: SettingsKeys.modelHubSearchText)
    }

    private func persistDownloadedSearchText() {
        UserDefaults.standard.set(downloadedSearchText, forKey: SettingsKeys.modelHubDownloadedSearchText)
    }

    private func persistSortOrder() {
        UserDefaults.standard.set(sortOrder.rawValue, forKey: SettingsKeys.modelHubSortOrder)
    }

    private func persistShowDownloadedOnly() {
        UserDefaults.standard.set(showDownloadedOnly, forKey: SettingsKeys.modelHubShowDownloadedOnly)
    }

    // MARK: - Internal Selection Sync

    private func synchronizeSelection(preferred: [HFModelSummary]? = nil) {
        let available = preferred ?? displayedResults
        let next = ModelHubSelectionPolicy.nextSelection(current: selectedModelId, available: available)
        if next != selectedModelId {
            selectedModelId = next
        }
    }
}

private struct DownloadSourceLogLine: Codable {
    let v: Int?
    let event: String?
    let timestamp: String?
    let repoId: String
    let filename: String
}
