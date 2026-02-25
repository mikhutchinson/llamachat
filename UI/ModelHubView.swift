import SwiftUI
import LlamaInferenceCore
import Textual

/// Main Model Hub view — native split workflow for model browsing and file actions.
struct ModelHubView: View {
    @State private var hubVM = ModelHubViewModel()
    @State private var liveDownloads: [String: DownloadTask] = [:]
    @State private var showingDownloadQueue = false
    @State private var confirmingDeleteId: String?
    @State private var confirmingModelDeleteId: String?
    @AppStorage(SettingsKeys.modelHubSplitVisibility) private var splitVisibilityRaw = SettingsDefaults.modelHubSplitVisibility

    @Bindable var downloadManager: ModelDownloadManager
    var onModelApplied: (() -> Void)? = nil

    @Environment(\.theme) private var theme

    private static let liveDownloadSyncInterval: Duration = .milliseconds(120)

    private var isSidebarHidden: Bool {
        let vis = splitVisibility(from: splitVisibilityRaw)
        return vis == .doubleColumn || vis == .detailOnly
    }

    private var sidebarSearchTextBinding: Binding<String> {
        Binding(
            get: { hubVM.showDownloadedOnly ? hubVM.downloadedSearchText : hubVM.searchText },
            set: { newValue in
                if hubVM.showDownloadedOnly {
                    hubVM.downloadedSearchText = newValue
                } else {
                    hubVM.searchText = newValue
                }
            }
        )
    }

    var body: some View {
        NavigationSplitView(columnVisibility: splitVisibilityBinding) {
            modelColumn
        } content: {
            if isSidebarHidden {
                downloadedModelsColumn
            } else {
                filesColumn
            }
        } detail: {
            if isSidebarHidden {
                combinedDetailColumn
            } else {
                readmeColumn
            }
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 900, idealWidth: 1100, minHeight: 560, idealHeight: 760)
        .background(theme.chatBg)
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Button {
                    Task {
                        if isSidebarHidden {
                            hubVM.invalidateDownloadedCache()
                            await hubVM.fetchDownloadedModels()
                        } else {
                            await hubVM.performRefresh()
                        }
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh results")
            }
            ToolbarItem(placement: .primaryAction) {
                downloadQueueButton
            }
        }
        .task {
            hubVM.onModelApplied = onModelApplied
            hubVM.invalidateDiscoveredCache()
            await hubVM.loadOnAppear()
            if isSidebarHidden {
                await hubVM.fetchDownloadedModels()
            }
        }
        .task {
            await syncLiveDownloadsLoop()
        }
        .onChange(of: liveDownloads.count) {
            hubVM.invalidateDownloadedCache()
            if hubVM.showDownloadedOnly {
                Task { await hubVM.fetchDownloadedModels() }
            }
        }
        .onChange(of: splitVisibilityRaw) {
            if isSidebarHidden {
                Task { await hubVM.fetchDownloadedModels() }
            }
        }
        .onDisappear {
            hubVM.onModelApplied = nil
        }
    }

    // MARK: - Layout Columns

    private var modelColumn: some View {
        VStack(spacing: 0) {
            // Sidebar header: search field + filter icon at top right
            HStack(alignment: .center, spacing: 10) {
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 12))
                        .foregroundColor(theme.textTertiary)
                    TextField(
                        hubVM.showDownloadedOnly ? "Search downloaded models" : "Search",
                        text: sidebarSearchTextBinding
                    )
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))
                        .foregroundColor(theme.textPrimary)
                }
                .padding(.horizontal, 12)
                .frame(height: 34)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(theme.searchFieldOverlay)
                )

                Menu {
                    Picker("Sort", selection: $hubVM.sortOrder) {
                        ForEach(HubSortOrder.allCases, id: \.self) { sort in
                            Text(sort.displayName).tag(sort)
                        }
                    }
                    Toggle("Downloaded only", isOn: Binding(
                        get: { hubVM.showDownloadedOnly },
                        set: { newValue in
                            hubVM.showDownloadedOnly = newValue
                            hubVM.invalidateDiscoveredCache()
                            Task {
                                if newValue {
                                    await hubVM.fetchDownloadedModels()
                                } else {
                                    await hubVM.performSearch()
                                }
                            }
                        }
                    ))
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "line.3.horizontal.decrease")
                            .font(.system(size: 14, weight: .semibold))
                        Image(systemName: "chevron.down")
                            .font(.system(size: 10, weight: .semibold))
                    }
                        .foregroundColor(theme.textTertiary)
                }
                .menuStyle(.borderlessButton)
                .help("Filters")
            }
            .padding(.leading, 24)
            .padding(.trailing, 8)
            .padding(.top, 10)
            .padding(.bottom, 6)

            Group {
            if let error = hubVM.errorMessage, hubVM.displayedResults.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 30))
                        .foregroundColor(theme.textTertiary)
                    Text(error)
                        .font(.caption)
                        .foregroundColor(theme.textSecondary)
                        .multilineTextAlignment(.center)
                    Button("Retry") {
                        Task { await hubVM.performSearch() }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if hubVM.displayedResults.isEmpty && !hubVM.isSearching {
                VStack(spacing: 8) {
                    Image(systemName: "shippingbox")
                        .font(.system(size: 30))
                        .foregroundColor(theme.textTertiary)
                    Text(hubVM.showDownloadedOnly
                         ? "No downloaded models found."
                         : "No results. Try another query.")
                        .font(.caption)
                        .foregroundColor(theme.textSecondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(hubVM.displayedResults) { model in
                        ModelCardRow(
                            model: model,
                            isSelected: hubVM.selectedModelId == model.id
                        )
                        .contentShape(Rectangle())
                        .onTapGesture {
                            hubVM.selectModel(model.id)
                        }
                        .listRowInsets(EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8))
                        .listRowBackground(Color.clear)
                        .listRowSeparator(.hidden)
                        .onAppear {
                            if model.id == hubVM.displayedResults.last?.id {
                                hubVM.triggerLoadMore()
                            }
                        }
                    }
                    if hubVM.isLoadingMore {
                        HStack {
                            Spacer()
                            ProgressView().controlSize(.small)
                            Text("Loading more…")
                                .font(.caption)
                                .foregroundColor(theme.textSecondary)
                            Spacer()
                        }
                        .listRowBackground(Color.clear)
                        .listRowSeparator(.hidden)
                    }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .background(theme.chatBg)
            }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(theme.chatBg)
    }

    private var filesColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let model = hubVM.selectedModel {
                modelHeader(model)
                Divider()
                filesPane(for: model)
            } else {
                placeholderColumn(
                    title: "Select a model",
                    subtitle: "Choose a model from the left to inspect GGUF files and actions."
                )
            }
        }
        .background(theme.chatBg)
    }

    /// Shown when sidebar is hidden — lists downloaded models so user can select one.
    private var downloadedModelsColumn: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Downloaded models")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(theme.textPrimary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 12))
                        .foregroundColor(theme.textTertiary)
                    TextField("Search", text: $hubVM.downloadedSearchText)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))
                        .foregroundColor(theme.textPrimary)
                }
                .padding(.horizontal, 12)
                .frame(height: 34)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(theme.searchFieldOverlay)
                )
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)

            if hubVM.downloadedModels.isEmpty && !hubVM.isSearching {
                placeholderColumn(
                    title: "No downloaded models",
                    subtitle: "Download a model from the Hub to see it here. Show the sidebar to browse."
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if hubVM.downloadedModels.isEmpty && hubVM.isSearching {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading…")
                        .font(.caption)
                        .foregroundColor(theme.textSecondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(hubVM.downloadedModels) { model in
                        ModelCardRow(model: model, isSelected: hubVM.selectedModelId == model.id)
                            .contentShape(Rectangle())
                            .onTapGesture { hubVM.selectModel(model.id) }
                            .listRowInsets(EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8))
                            .listRowBackground(Color.clear)
                            .listRowSeparator(.hidden)
                    }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .background(theme.chatBg)
            }
        }
        .background(theme.chatBg)
    }

    /// Shown when sidebar is hidden — full model detail (header, files, README) in one pane.
    private var combinedDetailColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let model = hubVM.selectedModel {
                modelHeader(model)
                Divider()
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        filesPane(for: model, embedInScrollView: false, filterToLocalOnly: true)
                        Divider()
                        metadataBlock(model)
                        Divider()
                        Text("README Preview")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(theme.textPrimary)
                        if hubVM.isLoadingSelectedReadme && hubVM.selectedReadmeSnippet == nil {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text("Loading README…")
                                    .font(.caption)
                                    .foregroundColor(theme.textSecondary)
                            }
                        } else if let snippet = hubVM.selectedReadmeSnippet, !snippet.isEmpty,
                                  let repoId = hubVM.selectedModelId,
                                  let readmeBaseURL = URL(string: "https://huggingface.co/\(repoId)/") {
                            StructuredText(markdown: snippet, baseURL: readmeBaseURL)
                                .font(.callout)
                                .foregroundStyle(theme.textPrimary)
                                .tint(theme.accent)
                                .textual.structuredTextStyle(.gitHub)
                                .textual.overflowMode(.scroll)
                                .textual.tableStyle(.overflow(relativeWidth: 4))
                                .textual.imageAttachmentLoader(.image(relativeTo: readmeBaseURL))
                                .textual.inlineStyle(
                                    .gitHub.link(.foregroundColor(theme.accent))
                                )
                                .textual.textSelection(.enabled)
                        } else {
                            Text("No README preview available.")
                                .font(.caption)
                                .foregroundColor(theme.textTertiary)
                        }
                    }
                    .padding(14)
                }
            } else {
                placeholderColumn(
                    title: "Select a model",
                    subtitle: "Choose a downloaded model from the list."
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(theme.chatBg)
    }

    private var readmeColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let model = hubVM.selectedModel {
                HStack {
                    Text("Details")
                        .font(.headline)
                        .foregroundColor(theme.textPrimary)
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                Divider()

                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        metadataBlock(model)
                        Divider()
                        Text("README Preview")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(theme.textPrimary)

                        if hubVM.isLoadingSelectedReadme && hubVM.selectedReadmeSnippet == nil {
                            HStack(spacing: 8) {
                                ProgressView().controlSize(.small)
                                Text("Loading README…")
                                    .font(.caption)
                                    .foregroundColor(theme.textSecondary)
                            }
                        } else if let snippet = hubVM.selectedReadmeSnippet,
                                  !snippet.isEmpty,
                                  let repoId = hubVM.selectedModelId,
                                  let readmeBaseURL = URL(string: "https://huggingface.co/\(repoId)/") {
                            StructuredText(markdown: snippet, baseURL: readmeBaseURL)
                                .font(.callout)
                                .foregroundStyle(theme.textPrimary)
                                .tint(theme.accent)
                                .textual.structuredTextStyle(.gitHub)
                                .textual.overflowMode(.scroll)
                                .textual.tableStyle(.overflow(relativeWidth: 4))
                                .textual.imageAttachmentLoader(.image(relativeTo: readmeBaseURL))
                                .textual.inlineStyle(
                                    .gitHub
                                        .link(.foregroundColor(theme.accent))
                                )
                                .textual.textSelection(.enabled)
                        } else {
                            Text("No README preview available.")
                                .font(.caption)
                                .foregroundColor(theme.textTertiary)
                        }
                    }
                    .padding(14)
                }
            } else {
                placeholderColumn(
                    title: "Model details",
                    subtitle: "README and metadata for the selected model appear here."
                )
            }
        }
        .background(theme.chatBg)
    }

    // MARK: - Header + Metadata

    private func modelHeader(_ model: HFModelSummary) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .center, spacing: 8) {
                Text(model.modelName)
                    .font(.title3.weight(.semibold))
                    .foregroundColor(theme.textPrimary)
                    .lineLimit(1)
                if model.isVLM {
                    Text("VLM")
                        .font(.caption2.weight(.medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(theme.accent.opacity(0.15))
                        .foregroundColor(theme.accent)
                        .clipShape(Capsule())
                }
                Spacer()
                Button {
                    if let url = URL(string: "https://huggingface.co/\(model.id)") {
                        NSWorkspace.shared.open(url)
                    }
                } label: {
                    Text("Open on Hugging Face")
                }
                .buttonStyle(.borderless)
                .controlSize(.small)
            }
            Text(model.author)
                .font(.caption)
                .foregroundColor(theme.textTertiary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private func metadataBlock(_ model: HFModelSummary) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Stats")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(theme.textPrimary)
            HStack(spacing: 8) {
                metadataChip(title: "Downloads", value: formatCount(model.downloads))
                metadataChip(title: "Likes", value: formatCount(model.likes))
                if let desc = model.pipelineDescription {
                    metadataChip(title: "Pipeline", value: desc)
                }
            }
        }
    }

    private func metadataChip(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title)
                .font(.caption2)
                .foregroundColor(theme.textTertiary)
            Text(value)
                .font(.caption.weight(.medium))
                .foregroundColor(theme.textPrimary)
                .lineLimit(1)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(theme.inputBg.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func placeholderColumn(title: String, subtitle: String) -> some View {
        VStack(spacing: 8) {
            Text(title)
                .font(.headline)
                .foregroundColor(theme.textPrimary)
            Text(subtitle)
                .font(.caption)
                .foregroundColor(theme.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 260)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Files Pane + Actions

    @ViewBuilder
    private func filesPane(for model: HFModelSummary, embedInScrollView: Bool = true, filterToLocalOnly: Bool = false) -> some View {
        let effectiveShowDownloadedOnly = filterToLocalOnly || hubVM.showDownloadedOnly
        if hubVM.isLoadingSelectedFiles && hubVM.selectedFiles == nil {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Loading files…")
                    .font(.caption)
                    .foregroundColor(theme.textSecondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if let files = hubVM.selectedFiles {
            let allMainFiles = files.filter { !$0.isMMProj }
            let allProjFiles = files.filter { $0.isMMProj }
            let discoveredFallbackFiles = effectiveShowDownloadedOnly ? matchedDiscoveredFiles(for: model) : []
            let discoveredFilenameIndex = Set(discoveredFallbackFiles.map { fileDisplayName($0).lowercased() })
            let visibleMainFiles = effectiveShowDownloadedOnly
                ? allMainFiles.filter { shouldShowInDownloadedOnly($0, discoveredFilenameIndex: discoveredFilenameIndex) }
                : allMainFiles
            let visibleProjFiles = effectiveShowDownloadedOnly
                ? allProjFiles.filter { shouldShowInDownloadedOnly($0, discoveredFilenameIndex: discoveredFilenameIndex) }
                : allProjFiles
            // When the HF repo has no mmproj files, surface locally-available mmproj files
            // from any repo — mmproj adapters are universal vision projections.
            let localProjCandidates: [DiscoveredModel] = effectiveShowDownloadedOnly && visibleProjFiles.isEmpty && model.isVLM
                ? projectionCandidates(
                    for: model,
                    matchedProjectionFiles: discoveredFallbackFiles.filter { $0.isMMProj }
                )
                : []

            let fileList = LazyVStack(alignment: .leading, spacing: 8) {
                    if !effectiveShowDownloadedOnly,
                       model.isVLM,
                       !allMainFiles.isEmpty,
                       !allProjFiles.isEmpty {
                        vlmBundleRow(mainFiles: allMainFiles, projFiles: allProjFiles)
                    }

                    if visibleMainFiles.isEmpty, visibleProjFiles.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(
                                effectiveShowDownloadedOnly && !discoveredFallbackFiles.isEmpty
                                    ? "Local files detected for this model."
                                    : (effectiveShowDownloadedOnly ? "No local files for this model." : "No GGUF files found.")
                            )
                                .font(.caption)
                                .foregroundColor(theme.textTertiary)
                            if effectiveShowDownloadedOnly {
                                modelLifecycleFallbackControls(for: model, localFiles: discoveredFallbackFiles)
                            }
                        }
                        .padding(.top, 8)
                    }

                    ForEach(visibleMainFiles) { file in
                        fileRow(file, model: model)
                    }

                    if !visibleProjFiles.isEmpty {
                        HStack(spacing: 8) {
                            Rectangle().fill(theme.divider).frame(height: 0.5)
                            Text("Vision Projection")
                                .font(.caption2.weight(.medium))
                                .foregroundColor(theme.textTertiary)
                            Rectangle().fill(theme.divider).frame(height: 0.5)
                        }
                        .padding(.vertical, 2)

                        ForEach(visibleProjFiles) { file in
                            fileRow(file, model: model)
                        }
                    } else if !localProjCandidates.isEmpty {
                        HStack(spacing: 8) {
                            Rectangle().fill(theme.divider).frame(height: 0.5)
                            Text("Vision Projection")
                                .font(.caption2.weight(.medium))
                                .foregroundColor(theme.textTertiary)
                            Rectangle().fill(theme.divider).frame(height: 0.5)
                        }
                        .padding(.vertical, 2)

                        ForEach(localProjCandidates) { file in
                            localProjectionRow(file)
                        }
                    }
                }
                .padding(12)

            if embedInScrollView {
                ScrollView { fileList }
            } else {
                fileList
            }
        } else {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Preparing model details…")
                    .font(.caption)
                    .foregroundColor(theme.textSecondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func fileRow(_ file: GGUFFile, model: HFModelSummary) -> some View {
        let downloaded = downloadManager.isDownloaded(filename: file.filename)
        let activeTask = liveDownloads[file.id]
        let displaySize: String? = {
            if let size = file.estimatedSize, size > 0 {
                return ByteCountFormatter.string(fromByteCount: size, countStyle: .file)
            }
            if let path = downloadManager.localPath(for: file.filename),
               let attrs = try? FileManager.default.attributesOfItem(atPath: path),
               let bytes = attrs[.size] as? Int64, bytes > 0 {
                return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            }
            return nil
        }()

        return HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(file.quantLevel?.displayName ?? file.filename)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(theme.textPrimary)
                HStack(spacing: 6) {
                    Text(file.filename)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let displaySize {
                        Text(displaySize)
                    }
                }
                .font(.caption2)
                .foregroundColor(theme.textTertiary)
            }

            Spacer()

            if let task = activeTask {
                downloadProgressView(task, fileId: file.id)
            } else if downloaded {
                HStack(spacing: 6) {
                    useAsMenu(file, model: model)
                    Button {
                        if let path = downloadManager.localPath(for: file.filename) {
                            NSWorkspace.shared.selectFile(path, inFileViewerRootedAtPath: "")
                        }
                    } label: {
                        Image(systemName: "folder")
                            .font(.system(size: 11))
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(theme.textTertiary)
                    .help("Reveal in Finder")
                    deleteButton(for: file)
                }
            } else {
                Button {
                    downloadManager.download(file)
                } label: {
                    Label("Download", systemImage: "arrow.down.circle")
                        .font(.caption)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(theme.inputBg.opacity(0.45))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(theme.divider.opacity(0.7), lineWidth: 0.5)
        }
    }

    private func localProjectionRow(_ file: DiscoveredModel) -> some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(fileDisplayName(file))
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(theme.textPrimary)
                Text(file.size)
                    .font(.caption2)
                    .foregroundColor(theme.textTertiary)
            }

            Spacer()

            HStack(spacing: 6) {
                Button {
                    hubVM.applyModel(path: file.path, role: .vlmProjection)
                } label: {
                    Text("Use")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(theme.accent.opacity(0.15))
                        .foregroundColor(theme.accent)
                        .clipShape(Capsule())
                }
                .buttonStyle(.plain)

                Button {
                    NSWorkspace.shared.selectFile(file.path, inFileViewerRootedAtPath: "")
                } label: {
                    Image(systemName: "folder")
                        .font(.system(size: 11))
                }
                .buttonStyle(.plain)
                .foregroundColor(theme.textTertiary)
                .help("Reveal in Finder")
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(theme.inputBg.opacity(0.45))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(theme.divider.opacity(0.7), lineWidth: 0.5)
        }
    }

    private func vlmBundleRow(mainFiles: [GGUFFile], projFiles: [GGUFFile]) -> some View {
        let preferred = mainFiles.first(where: { $0.quantLevel == .Q4_K_M })
            ?? mainFiles.first(where: { $0.quantLevel == .Q4_K_S })
            ?? mainFiles.first!
        let proj = projFiles.first!

        let mainDownloaded = downloadManager.isDownloaded(filename: preferred.filename)
        let projDownloaded = downloadManager.isDownloaded(filename: proj.filename)
        let mainActive = liveDownloads[preferred.id] != nil
        let projActive = liveDownloads[proj.id] != nil
        let bothDone = mainDownloaded && projDownloaded
        let anyActive = mainActive || projActive

        return HStack(spacing: 8) {
            Image(systemName: "shippingbox.fill")
                .font(.system(size: 12))
                .foregroundColor(theme.accent)
            VStack(alignment: .leading, spacing: 1) {
                Text("VLM Bundle")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(theme.textPrimary)
                Text("\(preferred.quantLevel?.displayName ?? preferred.filename) + projection")
                    .font(.caption2)
                    .foregroundColor(theme.textTertiary)
            }
            Spacer()
            if bothDone {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(theme.accent)
                        .font(.system(size: 13))
                    vlmBundleUseAsMenu(mainFile: preferred, projFile: proj)
                }
            } else if anyActive {
                ProgressView().controlSize(.small)
            } else {
                Button {
                    if !mainDownloaded && !mainActive {
                        downloadManager.download(preferred)
                    }
                    if !projDownloaded && !projActive {
                        downloadManager.download(proj)
                    }
                } label: {
                    Label("Download Both", systemImage: "arrow.down.circle")
                        .font(.caption)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
        .background(theme.accent.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func vlmBundleUseAsMenu(mainFile: GGUFFile, projFile: GGUFFile) -> some View {
        if let mainPath = downloadManager.localPath(for: mainFile.filename),
           let projPath = downloadManager.localPath(for: projFile.filename),
           !mainPath.isEmpty, !projPath.isEmpty {
            Menu {
                Button {
                    hubVM.applyModel(path: mainPath, role: .vlmModel)
                    hubVM.applyModel(path: projPath, role: .vlmProjection)
                } label: {
                    Label("VLM Model + Projection", systemImage: "eye.trianglebadge.exclamationmark")
                }
                Divider()
                Button {
                    hubVM.applyModel(path: mainPath, role: .chatModel)
                } label: {
                    Text("Chat Model only")
                }
            } label: {
                Text("Use as")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(theme.accent.opacity(0.15))
                    .foregroundColor(theme.accent)
                    .clipShape(Capsule())
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
        }
    }

    private func downloadProgressView(_ task: DownloadTask, fileId: String) -> some View {
        HStack(spacing: 6) {
            switch task.state {
            case .queued:
                ProgressView().controlSize(.small)
                Text("Queued")
                    .font(.caption2)
                    .foregroundColor(theme.textTertiary)

            case .downloading:
                VStack(alignment: .trailing, spacing: 2) {
                    ProgressView(value: task.progress)
                        .frame(width: 80)
                    Text("\(Int(task.progress * 100))%")
                        .font(.caption2)
                        .foregroundColor(theme.textTertiary)
                }
                Button {
                    downloadManager.pause(filename: fileId)
                } label: {
                    Image(systemName: "pause.circle")
                        .font(.system(size: 11))
                        .foregroundColor(theme.textTertiary)
                }
                .buttonStyle(.plain)
                .help("Pause")
                Button {
                    downloadManager.cancel(filename: fileId)
                } label: {
                    Image(systemName: "xmark.circle")
                        .font(.system(size: 11))
                        .foregroundColor(theme.textTertiary)
                }
                .buttonStyle(.plain)
                .help("Cancel")

            case .paused:
                VStack(alignment: .trailing, spacing: 2) {
                    ProgressView(value: task.progress)
                        .frame(width: 80)
                    Text("Paused — \(Int(task.progress * 100))%")
                        .font(.caption2)
                        .foregroundColor(theme.textTertiary)
                }
                Button {
                    downloadManager.resumeDownload(filename: fileId)
                } label: {
                    Image(systemName: "play.circle")
                        .font(.system(size: 11))
                        .foregroundColor(theme.accent)
                }
                .buttonStyle(.plain)
                .help("Resume")
                Button {
                    downloadManager.cancel(filename: fileId)
                } label: {
                    Image(systemName: "xmark.circle")
                        .font(.system(size: 11))
                        .foregroundColor(theme.textTertiary)
                }
                .buttonStyle(.plain)
                .help("Cancel")

            case .verifying:
                ProgressView().controlSize(.small)
                Text("Verifying…")
                    .font(.caption2)
                    .foregroundColor(theme.textTertiary)

            case .completed:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(theme.accent)
                    .font(.system(size: 13))

            case .failed(let msg):
                Image(systemName: "exclamationmark.triangle")
                    .foregroundColor(.red)
                    .font(.system(size: 12))
                    .help(msg)
                Button {
                    downloadManager.dismiss(filename: fileId)
                } label: {
                    Text("Retry")
                        .font(.caption2)
                        .foregroundColor(theme.accent)
                }
                .buttonStyle(.plain)

            case .cancelled:
                Text("Cancelled")
                    .font(.caption2)
                    .foregroundColor(theme.textTertiary)
                Button {
                    downloadManager.dismiss(filename: fileId)
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10))
                        .foregroundColor(theme.textTertiary)
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func deleteButton(for file: GGUFFile) -> some View {
        Group {
            if confirmingDeleteId == file.id {
                Button {
                    _ = downloadManager.deleteDownloaded(filename: file.filename)
                    confirmingDeleteId = nil
                } label: {
                    Image(systemName: "trash.fill")
                        .font(.system(size: 11))
                        .foregroundColor(.red)
                }
                .buttonStyle(.plain)
                .help("Confirm delete")
                Button {
                    confirmingDeleteId = nil
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 9))
                        .foregroundColor(theme.textTertiary)
                }
                .buttonStyle(.plain)
                .help("Cancel")
            } else {
                Button {
                    confirmingDeleteId = file.id
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 11))
                        .foregroundColor(theme.textTertiary)
                }
                .buttonStyle(.plain)
                .help("Delete from disk")
            }
        }
    }

    private func useAsMenu(_ file: GGUFFile, model: HFModelSummary) -> some View {
        let suggestedRole = ModelRole.suggest(for: file, isVLMRepo: model.isVLM, pipelineTag: model.pipelineTag)
        let localPath = downloadManager.localPath(for: file.filename) ?? ""

        return Menu {
            ForEach(ModelRole.allCases, id: \.self) { role in
                Button {
                    hubVM.applyModel(path: localPath, role: role)
                } label: {
                    HStack {
                        Text(role.rawValue)
                        if role == suggestedRole {
                            Text("(suggested)")
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            Divider()
            Button {
                NSWorkspace.shared.selectFile(localPath, inFileViewerRootedAtPath: "")
            } label: {
                Label("Reveal in Finder", systemImage: "folder")
            }
        } label: {
            Text("Use as")
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(theme.accent.opacity(0.15))
                .foregroundColor(theme.accent)
                .clipShape(Capsule())
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    // MARK: - Queue Popover

    private var downloadQueueButton: some View {
        Button {
            showingDownloadQueue.toggle()
        } label: {
            ZStack(alignment: .topTrailing) {
                Image(systemName: activeDownloadCount > 0 ? "arrow.down.circle.fill" : "arrow.down.circle")
                if activeDownloadCount > 0 {
                    Text("\(min(activeDownloadCount, 99))")
                        .font(.caption2.weight(.bold))
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(theme.accent)
                        .foregroundColor(.white)
                        .clipShape(Capsule())
                        .offset(x: 10, y: -10)
                }
            }
        }
        .help("Download queue")
        .popover(isPresented: $showingDownloadQueue, arrowEdge: .top) {
            downloadQueuePopover
        }
    }

    private var downloadQueuePopover: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Download Queue")
                    .font(.headline)
                    .foregroundColor(theme.textPrimary)
                Spacer()
                Text("\(queueTasks.count)")
                    .font(.caption)
                    .foregroundColor(theme.textTertiary)
            }

            if queueTasks.isEmpty {
                Text("No active downloads.")
                    .font(.caption)
                    .foregroundColor(theme.textSecondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 6) {
                        ForEach(queueTasks, id: \.id) { task in
                            queueTaskRow(task)
                        }
                    }
                }
            }
        }
        .padding(12)
        .frame(width: 370, height: 280)
        .background(theme.chatBg)
    }

    private func queueTaskRow(_ task: DownloadTask) -> some View {
        HStack(spacing: 8) {
            Text(task.filename)
                .font(.caption)
                .foregroundColor(theme.textSecondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 6)
            switch task.state {
            case .queued:
                Text("Queued")
                    .font(.caption2)
                    .foregroundColor(theme.textTertiary)
                Button {
                    downloadManager.cancel(filename: task.id)
                } label: {
                    Image(systemName: "xmark.circle")
                }
                .buttonStyle(.plain)
                .foregroundColor(theme.textTertiary)
            case .downloading:
                ProgressView(value: task.progress)
                    .frame(width: 70)
                Text("\(Int(task.progress * 100))%")
                    .font(.caption2.monospacedDigit())
                    .foregroundColor(theme.textTertiary)
                    .frame(width: 30, alignment: .trailing)
                Button {
                    downloadManager.pause(filename: task.id)
                } label: {
                    Image(systemName: "pause.circle")
                }
                .buttonStyle(.plain)
                .foregroundColor(theme.textTertiary)
                Button {
                    downloadManager.cancel(filename: task.id)
                } label: {
                    Image(systemName: "xmark.circle")
                }
                .buttonStyle(.plain)
                .foregroundColor(theme.textTertiary)
            case .paused:
                Text("Paused")
                    .font(.caption2)
                    .foregroundColor(theme.textTertiary)
                Button {
                    downloadManager.resumeDownload(filename: task.id)
                } label: {
                    Image(systemName: "play.circle")
                }
                .buttonStyle(.plain)
                .foregroundColor(theme.accent)
                Button {
                    downloadManager.cancel(filename: task.id)
                } label: {
                    Image(systemName: "xmark.circle")
                }
                .buttonStyle(.plain)
                .foregroundColor(theme.textTertiary)
            case .verifying:
                ProgressView().controlSize(.small)
                Text("Verifying")
                    .font(.caption2)
                    .foregroundColor(theme.textTertiary)
            case .completed:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(theme.accent)
                Button {
                    downloadManager.dismiss(filename: task.id)
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
                .foregroundColor(theme.textTertiary)
            case .failed:
                Text("Failed")
                    .font(.caption2)
                    .foregroundColor(.red)
                Button {
                    downloadManager.dismiss(filename: task.id)
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
                .foregroundColor(theme.textTertiary)
            case .cancelled:
                Text("Cancelled")
                    .font(.caption2)
                    .foregroundColor(theme.textTertiary)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(theme.inputBg.opacity(0.55))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Download Snapshot

    private var activeDownloadCount: Int {
        liveDownloads.values.reduce(into: 0) { count, task in
            switch task.state {
            case .queued, .downloading, .paused, .verifying:
                count += 1
            case .completed, .failed, .cancelled:
                break
            }
        }
    }

    private var queueTasks: [DownloadTask] {
        liveDownloads.values.sorted { lhs, rhs in
            lhs.filename.localizedCaseInsensitiveCompare(rhs.filename) == .orderedAscending
        }
    }

    private func syncLiveDownloadsLoop() async {
        while !Task.isCancelled {
            let snapshot = makeLiveDownloadSnapshot(from: downloadManager.activeDownloads)
            if snapshot != liveDownloads {
                liveDownloads = snapshot
            }

            do {
                try await Task.sleep(for: Self.liveDownloadSyncInterval)
            } catch {
                break
            }
        }
    }

    /// Keep high-frequency progress updates transient to avoid relayout churn.
    private func makeLiveDownloadSnapshot(from source: [String: DownloadTask]) -> [String: DownloadTask] {
        var snapshot: [String: DownloadTask] = [:]
        snapshot.reserveCapacity(source.count)

        for (id, task) in source {
            var liveTask = task
            liveTask.resumeData = nil

            if liveTask.totalBytes > 0 {
                let rawPercent = (Double(liveTask.downloadedBytes) / Double(liveTask.totalBytes)) * 100
                let percent = min(100, max(0, Int(rawPercent.rounded(.towardZero))))
                let quantizedBytes = Int64((Double(percent) / 100.0) * Double(liveTask.totalBytes))
                liveTask.downloadedBytes = quantizedBytes
            }

            snapshot[id] = liveTask
        }

        return snapshot
    }

    // MARK: - Misc Helpers

    private func shouldShowInDownloadedOnly(_ file: GGUFFile, discoveredFilenameIndex: Set<String>) -> Bool {
        if downloadManager.isDownloaded(filename: file.filename) {
            return true
        }
        let target = (file.filename as NSString).lastPathComponent.lowercased()
        if discoveredFilenameIndex.contains(target) {
            return true
        }
        guard let task = liveDownloads[file.id] else { return false }
        switch task.state {
        case .queued, .downloading, .paused, .verifying, .completed:
            return true
        case .failed, .cancelled:
            return false
        }
    }

    @ViewBuilder
    private func modelLifecycleFallbackControls(for model: HFModelSummary, localFiles: [DiscoveredModel]) -> some View {
        let mainFiles = localFiles.filter { !$0.isMMProj }
        let matchedProjectionFiles = localFiles.filter { $0.isMMProj }
        let projectionFiles = projectionCandidates(for: model, matchedProjectionFiles: matchedProjectionFiles)

        if localFiles.isEmpty {
            Text("No matching local files detected for this repo.")
                .font(.caption2)
                .foregroundColor(theme.textTertiary)
        } else {
            HStack(alignment: .center, spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(localFiles.count == 1 ? "1 local file available" : "\(localFiles.count) local files available")
                        .font(.caption.weight(.medium))
                        .foregroundColor(theme.textSecondary)
                    Text(localFilesSubtitle(localFiles))
                        .font(.caption2)
                        .foregroundColor(theme.textTertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }

                Spacer(minLength: 12)

                modelLevelUseAsMenu(mainFiles: mainFiles, projectionFiles: projectionFiles)
                revealLocalFilesControl(localFiles)
                copyLocalPathControl(localFiles)
                modelLevelDeleteControls(model: model, files: localFiles)
            }
            .controlSize(.small)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(theme.inputBg.opacity(0.45))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay {
                RoundedRectangle(cornerRadius: 8)
                    .stroke(theme.divider.opacity(0.7), lineWidth: 0.5)
            }
        }
    }

    @ViewBuilder
    private func modelLevelUseAsMenu(mainFiles: [DiscoveredModel], projectionFiles: [DiscoveredModel]) -> some View {
        if !mainFiles.isEmpty || !projectionFiles.isEmpty {
            let suggestedMain = preferredMainFile(from: mainFiles)
            let suggestedProjection = projectionFiles.first
            let extraMainFiles = mainFiles.filter { $0.id != suggestedMain?.id }
            let extraProjectionFiles = projectionFiles.filter { $0.id != suggestedProjection?.id }
            let includeMainFilename = mainFiles.count > 1
            let includeProjectionFilename = projectionFiles.count > 1

            Menu {
                if let suggestedMain {
                    Button(useAsLabel(role: .chatModel, file: suggestedMain, includeFilename: includeMainFilename)) {
                        hubVM.applyModel(path: suggestedMain.path, role: .chatModel)
                    }
                    Button(useAsLabel(role: .summarizer, file: suggestedMain, includeFilename: includeMainFilename)) {
                        hubVM.applyModel(path: suggestedMain.path, role: .summarizer)
                    }
                    Button(useAsLabel(role: .vlmModel, file: suggestedMain, includeFilename: includeMainFilename)) {
                        hubVM.applyModel(path: suggestedMain.path, role: .vlmModel)
                    }
                }
                if let suggestedMain, let suggestedProjection {
                    Button(includeMainFilename || includeProjectionFilename
                           ? "Use as VLM Model + Projection — \(fileDisplayName(suggestedMain)) + \(fileDisplayName(suggestedProjection))"
                           : "Use as VLM Model + Projection") {
                        hubVM.applyModel(path: suggestedMain.path, role: .vlmModel)
                        hubVM.applyModel(path: suggestedProjection.path, role: .vlmProjection)
                    }
                } else if let suggestedProjection {
                    Button(useAsLabel(role: .vlmProjection, file: suggestedProjection, includeFilename: includeProjectionFilename)) {
                        hubVM.applyModel(path: suggestedProjection.path, role: .vlmProjection)
                    }
                }

                if !extraMainFiles.isEmpty || !extraProjectionFiles.isEmpty {
                    Divider()
                }

                ForEach(extraMainFiles) { file in
                    Button(useAsLabel(role: .chatModel, file: file, includeFilename: true)) {
                        hubVM.applyModel(path: file.path, role: .chatModel)
                    }
                    Button(useAsLabel(role: .vlmModel, file: file, includeFilename: true)) {
                        hubVM.applyModel(path: file.path, role: .vlmModel)
                    }
                    Button(useAsLabel(role: .summarizer, file: file, includeFilename: true)) {
                        hubVM.applyModel(path: file.path, role: .summarizer)
                    }
                }

                ForEach(extraProjectionFiles) { file in
                    Button(useAsLabel(role: .vlmProjection, file: file, includeFilename: true)) {
                        hubVM.applyModel(path: file.path, role: .vlmProjection)
                    }
                }
            } label: {
                Text("Use as")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(theme.accent.opacity(0.15))
                    .foregroundColor(theme.accent)
                    .clipShape(Capsule())
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Assign local files to chat, VLM, projection, or summarizer roles")
        }
    }

    @ViewBuilder
    private func revealLocalFilesControl(_ files: [DiscoveredModel]) -> some View {
        if files.count == 1, let first = files.first {
            Button {
                NSWorkspace.shared.selectFile(first.path, inFileViewerRootedAtPath: "")
            } label: {
                Image(systemName: "folder")
                    .font(.system(size: 11))
                    .foregroundColor(theme.textTertiary)
            }
            .buttonStyle(.plain)
            .help("Reveal in Finder")
        } else {
            Menu {
                ForEach(files) { file in
                    Button {
                        NSWorkspace.shared.selectFile(file.path, inFileViewerRootedAtPath: "")
                    } label: {
                        Text(fileDisplayName(file))
                    }
                }
            } label: {
                Image(systemName: "folder")
                    .font(.system(size: 11))
                    .foregroundColor(theme.textTertiary)
            }
            .menuStyle(.borderlessButton)
            .help("Reveal local files in Finder")
        }
    }

    @ViewBuilder
    private func copyLocalPathControl(_ files: [DiscoveredModel]) -> some View {
        if files.count == 1, let first = files.first {
            Button {
                copyToPasteboard(first.path)
            } label: {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 11))
                    .foregroundColor(theme.textTertiary)
            }
            .buttonStyle(.plain)
            .help("Copy local path")
        } else {
            Menu {
                ForEach(files) { file in
                    Button {
                        copyToPasteboard(file.path)
                    } label: {
                        Text(fileDisplayName(file))
                    }
                }
            } label: {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 11))
                    .foregroundColor(theme.textTertiary)
            }
            .menuStyle(.borderlessButton)
            .help("Copy local path")
        }
    }

    @ViewBuilder
    private func modelLevelDeleteControls(model: HFModelSummary, files: [DiscoveredModel]) -> some View {
        if files.count > 1 {
            Menu {
                Button(role: .destructive) {
                    deleteLocalFiles(files)
                } label: {
                    Text("Delete all local files (\(files.count))")
                }
                Divider()
                ForEach(files) { file in
                    Button(role: .destructive) {
                        deleteLocalFiles([file])
                    } label: {
                        Text("Delete \(fileDisplayName(file))")
                    }
                }
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 11))
                    .foregroundColor(theme.textTertiary)
            }
            .menuStyle(.borderlessButton)
            .help("Delete local files for this model")
        } else if let only = files.first {
            if confirmingModelDeleteId == model.id {
                Button {
                    deleteLocalFiles([only])
                } label: {
                    Image(systemName: "trash.fill")
                        .font(.system(size: 11))
                        .foregroundColor(.red)
                }
                .buttonStyle(.plain)
                .help("Confirm delete")

                Button {
                    confirmingModelDeleteId = nil
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 9))
                        .foregroundColor(theme.textTertiary)
                }
                .buttonStyle(.plain)
                .help("Cancel")
            } else {
                Button {
                    confirmingModelDeleteId = model.id
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 11))
                        .foregroundColor(theme.textTertiary)
                }
                .buttonStyle(.plain)
                .help("Delete local file")
            }
        }
    }

    private func matchedDiscoveredFiles(for model: HFModelSummary) -> [DiscoveredModel] {
        let modelBase = GGUFModelInfo.baseNameFromHFModel(model.modelName).lowercased()
        let modelName = model.modelName.lowercased()
        let repoId = model.id.lowercased()

        return ModelDiscovery.scan()
            .filter { file in
                let localName = file.name.lowercased()
                if localName.contains(repoId) || localName.contains(modelName) {
                    return true
                }
                let fileBase = GGUFModelInfo.baseName(fromFilename: file.name).lowercased()
                if fileBase.count >= 5 && fileBase == modelBase {
                    return true
                }
                if file.isMMProj {
                    let projectionBase = normalizedProjectionBase(fileBase)
                    if projectionBase.count >= 5 && projectionBase == modelBase {
                        return true
                    }
                    let filename = fileDisplayName(file).lowercased()
                    return filename.contains(modelBase)
                }
                return false
            }
            .sorted { lhs, rhs in
                lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
    }

    private func projectionCandidates(for model: HFModelSummary, matchedProjectionFiles: [DiscoveredModel]) -> [DiscoveredModel] {
        if !matchedProjectionFiles.isEmpty {
            return matchedProjectionFiles
        }
        let allProjectionFiles = ModelDiscovery.scan().filter { $0.isMMProj }
            .sorted { lhs, rhs in
                lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
        if allProjectionFiles.isEmpty {
            return []
        }
        if allProjectionFiles.count == 1 {
            return allProjectionFiles
        }
        guard model.isVLM else { return [] }

        let modelBase = GGUFModelInfo.baseNameFromHFModel(model.modelName).lowercased()
        let modelName = model.modelName.lowercased()
        let targeted = allProjectionFiles.filter { file in
            let name = fileDisplayName(file).lowercased()
            return name.contains(modelBase) || name.contains(modelName)
        }
        return targeted.isEmpty ? allProjectionFiles : targeted
    }

    private func preferredMainFile(from files: [DiscoveredModel]) -> DiscoveredModel? {
        files.sorted { lhs, rhs in
            let left = filePreferenceRank(lhs)
            let right = filePreferenceRank(rhs)
            if left != right {
                return left < right
            }
            return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
        }.first
    }

    private func filePreferenceRank(_ file: DiscoveredModel) -> Int {
        let name = file.name.lowercased()
        if name.contains("q4_k_m") { return 0 }
        if name.contains("q4_k_s") { return 1 }
        if name.contains("q5_k_m") { return 2 }
        if name.contains("q8_0") { return 3 }
        return 10
    }

    private func fileDisplayName(_ file: DiscoveredModel) -> String {
        (file.name as NSString).lastPathComponent
    }

    private func useAsLabel(role: ModelRole, file: DiscoveredModel, includeFilename: Bool) -> String {
        let base: String
        switch role {
        case .chatModel:
            base = "Use as Chat Model"
        case .vlmModel:
            base = "Use as VLM Model"
        case .vlmProjection:
            base = "Use as VLM Projection"
        case .summarizer:
            base = "Use as Summarizer"
        }
        return includeFilename ? "\(base) — \(fileDisplayName(file))" : base
    }

    private func normalizedProjectionBase(_ base: String) -> String {
        var normalized = base.replacingOccurrences(
            of: #"(^|[-_])mmproj([-_]|$)"#,
            with: "-",
            options: .regularExpression
        )
        normalized = normalized.replacingOccurrences(
            of: #"[-_]{2,}"#,
            with: "-",
            options: .regularExpression
        )
        return normalized.trimmingCharacters(in: CharacterSet(charactersIn: "-_"))
    }

    private func localFilesSubtitle(_ files: [DiscoveredModel]) -> String {
        let names = files.prefix(2).map(fileDisplayName)
        if files.count > 2 {
            return "\(names.joined(separator: " • ")) • +\(files.count - 2) more"
        }
        return names.joined(separator: " • ")
    }

    private func deleteLocalFiles(_ files: [DiscoveredModel]) {
        let fm = FileManager.default
        let deletedPaths = Set(files.map(\.path))
        let deletedFilenames = Set(files.map { fileDisplayName($0).lowercased() })
        for file in files {
            try? fm.removeItem(atPath: file.path)
        }
        removeDeletedSourceRecords(filenames: deletedFilenames)
        clearDeletedRoleAssignments(deletedPaths)
        confirmingModelDeleteId = nil
        hubVM.invalidateDiscoveredCache()
        hubVM.invalidateDownloadedCache()
        Task {
            await hubVM.fetchDownloadedModels()
        }
    }

    private func removeDeletedSourceRecords(filenames: Set<String>) {
        guard !filenames.isEmpty else { return }
        let dir = resolvedDownloadDirectoryForSourceMap()
        let sourceMapURL = dir.appendingPathComponent(".modelhub-download-sources.json")
        guard let data = try? Data(contentsOf: sourceMapURL),
              var map = try? JSONDecoder().decode([String: String].self, from: data) else {
            return
        }
        for filename in filenames {
            map.removeValue(forKey: filename)
        }
        guard let encoded = try? JSONEncoder().encode(map) else { return }
        try? encoded.write(to: sourceMapURL, options: .atomic)
    }

    private func resolvedDownloadDirectoryForSourceMap() -> URL {
        let stored = UserDefaults.standard.string(forKey: SettingsKeys.modelDownloadDirectory) ?? ""
        if stored.isEmpty {
            return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Models/gguf")
        }
        return URL(fileURLWithPath: stored)
    }

    private func clearDeletedRoleAssignments(_ deletedPaths: Set<String>) {
        let roleKeys = [
            SettingsKeys.modelPath,
            SettingsKeys.vlmModelPath,
            SettingsKeys.vlmClipPath,
            SettingsKeys.summarizerModelPath,
        ]
        let ud = UserDefaults.standard
        for key in roleKeys {
            guard let current = ud.string(forKey: key) else { continue }
            if deletedPaths.contains(current) {
                ud.removeObject(forKey: key)
            }
        }
    }

    private func copyToPasteboard(_ value: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(value, forType: .string)
    }

    private func formatCount(_ count: Int) -> String {
        if count >= 1_000_000 {
            return String(format: "%.1fM", Double(count) / 1_000_000)
        } else if count >= 1_000 {
            return String(format: "%.1fK", Double(count) / 1_000)
        }
        return "\(count)"
    }

    private var splitVisibilityBinding: Binding<NavigationSplitViewVisibility> {
        Binding(
            get: { splitVisibility(from: splitVisibilityRaw) },
            set: { splitVisibilityRaw = rawValue(for: $0) }
        )
    }

    private func splitVisibility(from raw: String) -> NavigationSplitViewVisibility {
        switch raw {
        case "automatic":
            return .automatic
        case "doubleColumn":
            return .doubleColumn
        case "detailOnly":
            return .detailOnly
        default:
            return .all
        }
    }

    private func rawValue(for visibility: NavigationSplitViewVisibility) -> String {
        switch visibility {
        case .automatic:
            return "automatic"
        case .doubleColumn:
            return "doubleColumn"
        case .detailOnly:
            return "detailOnly"
        case .all:
            return "all"
        default:
            return "all"
        }
    }
}
