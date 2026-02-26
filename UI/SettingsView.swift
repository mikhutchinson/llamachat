import SwiftUI
import AppKit
import UniformTypeIdentifiers
import LlamaInferenceCore
import ChatStorage

// MARK: - Settings Category

enum SettingsCategory: String, CaseIterable, Identifiable {
    case models
    case generation
    case chatMemory
    case appearance
    case advanced

    var id: String { rawValue }

    var title: String {
        switch self {
        case .models: return "Models"
        case .generation: return "Generation"
        case .chatMemory: return "Chat & Memory"
        case .appearance: return "Appearance"
        case .advanced: return "Advanced"
        }
    }

    var systemImage: String {
        switch self {
        case .models: return "cpu"
        case .generation: return "slider.horizontal.3"
        case .chatMemory: return "bubble.left.and.bubble.right"
        case .appearance: return "paintbrush"
        case .advanced: return "wrench.and.screwdriver"
        }
    }
}

// MARK: - Settings Sheet (fallback for sheet presentation)

struct SettingsSheetView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            SettingsView(viewModel: viewModel)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()
            HStack {
                Spacer()
                Button("Done") {
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(.bar)
        }
        .frame(minWidth: 560, minHeight: 480)
        .presentationDragIndicator(.visible)
    }
}

// MARK: - Settings View

/// Uses TabView per macOS HIG — traditional Preferences pattern with proper title bar and traffic lights.
struct SettingsView: View {
    @ObservedObject var viewModel: ChatViewModel
    @AppStorage(SettingsKeys.appTheme) private var appTheme = SettingsDefaults.appTheme
    @State private var selectedCategory: SettingsCategory = .models

    private var resolvedColorScheme: ColorScheme? {
        switch appTheme {
        case "light": return .light
        case "dark": return .dark
        default: return nil
        }
    }

    var body: some View {
        TabView(selection: $selectedCategory) {
            ModelTab(viewModel: viewModel)
                .tabItem { Label(SettingsCategory.models.title, systemImage: SettingsCategory.models.systemImage) }
                .tag(SettingsCategory.models)

            GenerationTab()
                .tabItem { Label(SettingsCategory.generation.title, systemImage: SettingsCategory.generation.systemImage) }
                .tag(SettingsCategory.generation)

            ChatMemoryTab(viewModel: viewModel)
                .tabItem { Label(SettingsCategory.chatMemory.title, systemImage: SettingsCategory.chatMemory.systemImage) }
                .tag(SettingsCategory.chatMemory)

            AppearanceTab()
                .tabItem { Label(SettingsCategory.appearance.title, systemImage: SettingsCategory.appearance.systemImage) }
                .tag(SettingsCategory.appearance)

            AdvancedTab()
                .tabItem { Label(SettingsCategory.advanced.title, systemImage: SettingsCategory.advanced.systemImage) }
                .tag(SettingsCategory.advanced)
        }
        .tabViewStyle(.automatic)
        .animation(.none, value: selectedCategory)
        .frame(width: 560, height: 500)
        .preferredColorScheme(resolvedColorScheme)
    }
}

// MARK: - Flow Layout (chip container)

private struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        arrange(proposal: proposal, subviews: subviews).size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = arrange(
            proposal: ProposedViewSize(width: bounds.width, height: bounds.height),
            subviews: subviews
        )
        for (index, position) in result.positions.enumerated() {
            subviews[index].place(
                at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y),
                proposal: .unspecified
            )
        }
    }

    private func arrange(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        let maxWidth = proposal.width ?? .infinity
        var positions: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        var maxX: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth, x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            positions.append(CGPoint(x: x, y: y))
            let rightEdge = x + size.width
            rowHeight = max(rowHeight, size.height)
            x = rightEdge + spacing
            maxX = max(maxX, rightEdge)
        }

        return (CGSize(width: max(maxX, 0), height: y + rowHeight), positions)
    }
}

// MARK: - Stop Sequence Chip

private struct StopSequenceChip: View {
    let label: String
    let isSelected: Bool
    var showRemove: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 3) {
                Text(label)
                    .font(.system(.caption, design: .monospaced))
                if showRemove {
                    Image(systemName: "xmark")
                        .font(.system(size: 8, weight: .bold))
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(isSelected ? Color.accentColor.opacity(0.15) : Color.secondary.opacity(0.08))
            .foregroundStyle(isSelected ? Color.accentColor : .secondary)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(isSelected ? Color.accentColor.opacity(0.3) : Color.secondary.opacity(0.15), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

// MARK: - Model Tab

private struct ModelTab: View {
    @ObservedObject var viewModel: ChatViewModel
    @AppStorage("modelPath") private var modelPath = ""
    @AppStorage("summarizerModelPath") private var summarizerModelPath = ""
    @AppStorage("workerCount") private var workerCount = 2
    @AppStorage("contextSize") private var contextSize = 4096
    @AppStorage("nGpuLayers") private var nGpuLayers = -1
    @AppStorage("useSharedMemory") private var useSharedMemory = true
    @AppStorage(SettingsKeys.vlmModelPath) private var vlmModelPath = SettingsDefaults.vlmModelPath
    @AppStorage(SettingsKeys.vlmClipPath) private var vlmClipPath = SettingsDefaults.vlmClipPath
    @AppStorage(SettingsKeys.vlmIdleTimeoutSecs) private var vlmIdleTimeoutSecs = SettingsDefaults.vlmIdleTimeoutSecs
    @AppStorage(SettingsKeys.modelDownloadDirectory) private var downloadDirectory = SettingsDefaults.modelDownloadDirectory

    @State private var discoveredModels: [DiscoveredModel] = []

    var body: some View {
        Form {
            Section("Primary model") {
                LabeledContent("Model") {
                    Menu {
                        if discoveredModels.filter({ !$0.isMMProj }).isEmpty {
                            Text("No models found")
                                .foregroundStyle(.secondary)
                        } else {
                            ForEach(discoveredModels.filter { !$0.isMMProj }) { model in
                                Button {
                                    modelPath = model.path
                                } label: {
                                    Label(
                                        "\(GGUFModelInfo.parse(filename: model.name, sizeBytes: model.sizeBytes).displayName) (\(model.size))",
                                        systemImage: modelPath == model.path ? "checkmark" : ""
                                    )
                                }
                            }
                        }
                        Divider()
                        Button { browseForModel() } label: {
                            Label("Browse\u{2026}", systemImage: "folder")
                        }
                    } label: {
                        Text(modelPath.isEmpty ? "None" : modelDisplayName(for: modelPath))
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .accessibilityLabel("Primary model")
                    .accessibilityHint("Choose the main chat model or browse for a GGUF file")
                }

                if let info = selectedModelInfo {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(info.displayName)
                            .font(.headline)

                        HStack(spacing: 16) {
                            if let quant = info.quantization {
                                HStack(spacing: 4) {
                                    Text("Quant")
                                        .font(.caption)
                                        .foregroundStyle(.tertiary)
                                    Text(quant)
                                        .font(.system(.caption, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                }
                            }

                            HStack(spacing: 4) {
                                Text("Context")
                                    .font(.caption)
                                    .foregroundStyle(.tertiary)
                                Text(contextSizeLabel)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }

                            HStack(spacing: 4) {
                                Text("Size")
                                    .font(.caption)
                                    .foregroundStyle(.tertiary)
                                Text(info.sizeFormatted)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                        }

                        HStack(spacing: 4) {
                            Text("Est. RAM")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                            Text(info.estimatedRAM)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 2)
                }
            }

            Section("Optional models") {
                LabeledContent("Summarizer Model") {
                    Menu {
                        Button {
                            summarizerModelPath = ""
                        } label: {
                            Label("Shared with main", systemImage: summarizerModelPath.isEmpty ? "checkmark" : "")
                        }
                        Divider()
                        ForEach(discoveredModels.filter { !$0.isMMProj }) { model in
                            Button {
                                summarizerModelPath = model.path
                            } label: {
                                Label(
                                    "\(model.name) (\(model.size))",
                                    systemImage: summarizerModelPath == model.path ? "checkmark" : ""
                                )
                            }
                        }
                        Divider()
                        Button { browseForSummarizerModel() } label: {
                            Label("Browse\u{2026}", systemImage: "folder")
                        }
                    } label: {
                        Text(summarizerModelPath.isEmpty ? "Shared with main" : modelDisplayName(for: summarizerModelPath))
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                }
                .accessibilityLabel("Summarizer model")
                .accessibilityHint("Choose a dedicated model for summarization or share the main model")
                Text("Dedicated model for narrative summarization. When set, uses a separate model for context-window summarization. Leave empty to use the main chat model instead.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Vision (VLM)") {
                LabeledContent("VLM Model") {
                    HStack(spacing: 6) {
                        Text(vlmModelPath.isEmpty ? "None" : modelDisplayName(for: vlmModelPath))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(vlmModelPath.isEmpty ? .secondary : .primary)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                        Button { browseForVLMModel() } label: {
                            Image(systemName: "folder")
                        }
                        if !vlmModelPath.isEmpty {
                            Button { vlmModelPath = "" } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .accessibilityLabel("VLM model")
                .accessibilityHint("Choose a VLM GGUF model for image understanding")

                LabeledContent("Vision Projection (mmproj)") {
                    HStack(spacing: 6) {
                        Text(vlmClipPath.isEmpty ? "None" : modelDisplayName(for: vlmClipPath))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(vlmClipPath.isEmpty ? .secondary : .primary)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                        Button { browseForClipProjection() } label: {
                            Image(systemName: "folder")
                        }
                        if !vlmClipPath.isEmpty {
                            Button { vlmClipPath = "" } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .accessibilityLabel("Vision projection")
                .accessibilityHint("Choose the mmproj vision encoder file (CLIP, SigLIP, or ViT) for the VLM")

                LabeledContent("Idle Timeout") {
                    HStack(spacing: 8) {
                        Text("\(vlmIdleTimeoutSecs)s")
                            .monospacedDigit()
                            .frame(minWidth: 40, alignment: .trailing)
                        Stepper("", value: $vlmIdleTimeoutSecs, in: 30...3600, step: 30)
                            .labelsHidden()
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("VLM idle timeout")
                    .accessibilityValue("\(vlmIdleTimeoutSecs) seconds")
                }

                Text("Vision Language Model for image captioning. Set both model and CLIP projection to enable. Loads when needed and frees memory when idle.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Runtime") {
                LabeledContent("Workers") {
                    HStack(spacing: 10) {
                        Text("\(workerCount)")
                            .monospacedDigit()
                            .frame(width: 24, alignment: .trailing)
                        Stepper("", value: $workerCount, in: 1...8)
                            .labelsHidden()
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Workers")
                    .accessibilityValue("\(workerCount) workers")
                }
                Text("Each worker is a separate process that loads the full model. More workers allow parallel conversations but multiply GPU/RAM usage. Use 1 worker for large models (13B+).")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                LabeledContent("Context") {
                    Picker("", selection: $contextSize) {
                        Text("2,048 (2k)").tag(2048)
                        Text("4,096 (4k)").tag(4096)
                        Text("8,192 (8k)").tag(8192)
                        Text("16,384 (16k)").tag(16384)
                        Text("32,768 (32k)").tag(32768)
                        Text("65,536 (64k)").tag(65536)
                        Text("131,072 (128k)").tag(131072)
                    }
                    .labelsHidden()
                    .frame(minWidth: 130)
                    .accessibilityLabel("Context size")
                    .accessibilityValue("\(contextSize) tokens")
                }

                LabeledContent("GPU Layers") {
                    HStack(spacing: 10) {
                        Text(nGpuLayers == -1 ? "All" : "\(nGpuLayers)")
                            .monospacedDigit()
                            .frame(width: 32, alignment: .trailing)
                        Stepper("", value: $nGpuLayers, in: -1...999)
                            .labelsHidden()
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("GPU Layers")
                    .accessibilityValue(nGpuLayers == -1 ? "All layers" : "\(nGpuLayers) layers")
                }
                if nGpuLayers == -1 {
                    Label {
                        Text("All layers on GPU — if the model is too large for your VRAM the worker will crash. Try 4–16 layers for large models (13B+).")
                            .font(.caption)
                    } icon: {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .font(.caption)
                    }
                    .foregroundStyle(.secondary)
                } else {
                    Text("Number of transformer layers to offload to GPU. Set to 0 for CPU-only. Higher values are faster but use more VRAM.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Shared Memory") {
                    Toggle("", isOn: $useSharedMemory)
                        .toggleStyle(.switch)
                        .accessibilityLabel("Shared memory")
                        .accessibilityHint("Use shared memory for model loading")
                }
            }

            Section("Downloads") {
                LabeledContent("Download Directory") {
                    HStack(spacing: 6) {
                        Text(downloadDirDisplay)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(.primary)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                        Button { browseForDownloadDir() } label: {
                            Image(systemName: "folder")
                        }
                        if !downloadDirectory.isEmpty {
                            Button { downloadDirectory = "" } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                            .help("Reset to default ~/Models/gguf")
                        }
                    }
                }
                .accessibilityLabel("Download directory")
                .accessibilityHint("Choose where Model Hub downloads are saved")
                Text("Where new Model Hub downloads are saved. The app also finds models in ~/.cache/huggingface/hub.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .controlSize(.regular)
        .transaction { $0.animation = nil }
        .onAppear { discoveredModels = ModelDiscovery.scan() }
        .onChange(of: modelPath) { rebuildConfig() }
        .onChange(of: summarizerModelPath) { rebuildConfig() }
        .onChange(of: workerCount) { rebuildConfig() }
        .onChange(of: contextSize) { rebuildConfig() }
        .onChange(of: nGpuLayers) { rebuildConfig() }
        .onChange(of: useSharedMemory) { rebuildConfig() }
    }

    /// User-friendly model name (strips .gguf, quantization suffixes) — not raw filename.
    private func modelDisplayName(for path: String) -> String {
        guard !path.isEmpty else { return "" }
        let filename = (path as NSString).lastPathComponent
        let bytes = discoveredModels.first(where: { $0.path == path })?.sizeBytes ?? 0
        return GGUFModelInfo.parse(filename: filename, sizeBytes: bytes).displayName
    }

    private var selectedModelInfo: GGUFModelInfo? {
        guard !modelPath.isEmpty else { return nil }
        let filename = (modelPath as NSString).lastPathComponent
        let bytes = discoveredModels.first(where: { $0.path == modelPath })?.sizeBytes
            ?? (try? FileManager.default.attributesOfItem(atPath: modelPath)[.size] as? Int64) ?? 0
        return GGUFModelInfo.parse(filename: filename, sizeBytes: bytes)
    }

    private var contextSizeLabel: String {
        if contextSize >= 1024 {
            return "\(contextSize / 1024)k"
        }
        return "\(contextSize)"
    }

    private func rebuildConfig() {
        viewModel.config = InferenceConfig(
            modelPath: modelPath,
            summarizerModelPath: summarizerModelPath.isEmpty ? nil : summarizerModelPath,
            contextSize: contextSize,
            nGpuLayers: nGpuLayers,
            workerCount: max(1, workerCount),
            maxSessionsPerWorker: 8,
            maxInFlight: max(1, workerCount) * 4,
            blasThreads: 1,
            useSharedMemory: useSharedMemory,
            venvPath: ChatViewModel.discoverVenvPath()
        )
        viewModel.reloadModelDebounced()
    }

    private func browseForModel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if let gguf = UTType(filenameExtension: "gguf") {
            panel.allowedContentTypes = [gguf]
        }
        // Default to ~/Models/gguf if it exists, else HF cache
        let modelsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Models/gguf")
        let hfCache = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        if FileManager.default.fileExists(atPath: modelsDir.path) {
            panel.directoryURL = modelsDir
        } else if FileManager.default.fileExists(atPath: hfCache.path) {
            panel.directoryURL = hfCache
        }
        if panel.runModal() == .OK, let url = panel.url {
            modelPath = url.path(percentEncoded: false)
            if !discoveredModels.contains(where: { $0.path == modelPath }) {
                discoveredModels = ModelDiscovery.scan()
            }
        }
    }

    private func browseForVLMModel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if let gguf = UTType(filenameExtension: "gguf") {
            panel.allowedContentTypes = [gguf]
        }
        let modelsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Models/gguf")
        let hfCache = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        if FileManager.default.fileExists(atPath: modelsDir.path) {
            panel.directoryURL = modelsDir
        } else if FileManager.default.fileExists(atPath: hfCache.path) {
            panel.directoryURL = hfCache
        }
        if panel.runModal() == .OK, let url = panel.url {
            vlmModelPath = url.path(percentEncoded: false)
        }
    }

    private func browseForClipProjection() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if let gguf = UTType(filenameExtension: "gguf") {
            panel.allowedContentTypes = [gguf]
        }
        let modelsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Models/gguf")
        let hfCache = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        if FileManager.default.fileExists(atPath: modelsDir.path) {
            panel.directoryURL = modelsDir
        } else if FileManager.default.fileExists(atPath: hfCache.path) {
            panel.directoryURL = hfCache
        }
        if panel.runModal() == .OK, let url = panel.url {
            vlmClipPath = url.path(percentEncoded: false)
        }
    }

    private func browseForSummarizerModel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        if let gguf = UTType(filenameExtension: "gguf") {
            panel.allowedContentTypes = [gguf]
        }
        let modelsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Models/gguf")
        let hfCache = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        if FileManager.default.fileExists(atPath: modelsDir.path) {
            panel.directoryURL = modelsDir
        } else if FileManager.default.fileExists(atPath: hfCache.path) {
            panel.directoryURL = hfCache
        }
        if panel.runModal() == .OK, let url = panel.url {
            summarizerModelPath = url.path(percentEncoded: false)
            if !discoveredModels.contains(where: { $0.path == summarizerModelPath }) {
                discoveredModels = ModelDiscovery.scan()
            }
        }
    }

    private var downloadDirDisplay: String {
        if downloadDirectory.isEmpty {
            return "~/Models/gguf (default)"
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        if downloadDirectory.hasPrefix(home) {
            return "~" + downloadDirectory.dropFirst(home.count)
        }
        return downloadDirectory
    }

    private func browseForDownloadDir() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        let modelsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Models/gguf")
        let hfCache = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        if FileManager.default.fileExists(atPath: modelsDir.path) {
            panel.directoryURL = modelsDir
        } else if FileManager.default.fileExists(atPath: hfCache.path) {
            panel.directoryURL = hfCache
        }
        if panel.runModal() == .OK, let url = panel.url {
            downloadDirectory = url.path(percentEncoded: false)
        }
    }
}

// MARK: - Generation Tab

private struct GenerationTab: View {
    @AppStorage(SettingsKeys.maxTokens) private var maxTokens = SettingsDefaults.maxTokens
    @AppStorage(SettingsKeys.temperature) private var temperature = SettingsDefaults.temperature
    @AppStorage(SettingsKeys.topP) private var topP = SettingsDefaults.topP
    @AppStorage(SettingsKeys.topK) private var topK = SettingsDefaults.topK
    @AppStorage(SettingsKeys.repeatPenalty) private var repeatPenalty = SettingsDefaults.repeatPenalty
    @AppStorage(SettingsKeys.generationPreset) private var generationPreset = SettingsDefaults.generationPreset
    @AppStorage(SettingsKeys.stopSequences) private var stopSequences = SettingsDefaults.stopSequences
    @State private var showAddCustomPopover = false
    @State private var customStopInput = ""

    var body: some View {
        Form {
            Section("Presets") {
                Picker("Style", selection: $generationPreset) {
                    ForEach([GenerationPreset.creative, .precise, .balanced], id: \.rawValue) { preset in
                        Text(preset.displayName).tag(preset.rawValue)
                    }
                    Text("Custom").tag(GenerationPreset.custom.rawValue)
                }
                .accessibilityLabel("Generation style")
                .onChange(of: generationPreset) { _, newValue in
                    if let preset = GenerationPreset(rawValue: newValue), preset != .custom {
                        temperature = preset.temperature
                        topP = preset.topP
                        topK = preset.topK
                        repeatPenalty = preset.repeatPenalty
                    }
                }
            }

            Section("Output") {
                LabeledContent("Max Tokens") {
                    HStack(spacing: 8) {
                        Text("\(maxTokens)")
                            .monospacedDigit()
                            .frame(minWidth: 40, alignment: .trailing)
                        Stepper("", value: $maxTokens, in: 16...16384, step: 64)
                            .labelsHidden()
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Max Tokens")
                    .accessibilityValue("\(maxTokens) tokens")
                }
            }

            Section("Sampling") {
                LabeledContent("Temperature") {
                    HStack(spacing: 8) {
                        Text("0")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Slider(value: $temperature, in: 0...2, step: 0.05)
                            .accessibilityLabel("Temperature")
                            .accessibilityValue(String(format: "%.2f", temperature))
                        Text("2")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Text(String(format: "%.2f", temperature))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                            .foregroundStyle(.secondary)
                    }
                }

                LabeledContent("Top P") {
                    HStack(spacing: 8) {
                        Text("0")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Slider(value: $topP, in: 0...1, step: 0.05)
                            .accessibilityLabel("Top P")
                            .accessibilityValue(String(format: "%.2f", topP))
                        Text("1")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Text(String(format: "%.2f", topP))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                            .foregroundStyle(.secondary)
                    }
                }

                LabeledContent("Top K") {
                    HStack(spacing: 8) {
                        Text("\(topK)")
                            .monospacedDigit()
                            .frame(minWidth: 28, alignment: .trailing)
                        Stepper("", value: $topK, in: 1...200)
                            .labelsHidden()
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Top K")
                    .accessibilityValue("\(topK)")
                }

                LabeledContent("Repeat Penalty") {
                    HStack(spacing: 8) {
                        Text("1.0")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Slider(value: $repeatPenalty, in: 1.0...2.0, step: 0.05)
                            .accessibilityLabel("Repeat Penalty")
                            .accessibilityValue(String(format: "%.2f", repeatPenalty))
                        Text("2.0")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Text(String(format: "%.2f", repeatPenalty))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section("Stop Sequences") {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(StopSequencePreset.categories, id: \.self) { category in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(category)
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                            FlowLayout(spacing: 6) {
                                ForEach(StopSequencePreset.all.filter { $0.category == category }) { preset in
                                    StopSequenceChip(
                                        label: preset.label,
                                        isSelected: activeSequences.contains(preset.value)
                                    ) {
                                        togglePreset(preset)
                                    }
                                }
                            }
                        }
                    }

                    if !customEntries.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Custom")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                            FlowLayout(spacing: 6) {
                                ForEach(customEntries, id: \.self) { entry in
                                    StopSequenceChip(
                                        label: displayLabel(for: entry),
                                        isSelected: true,
                                        showRemove: true
                                    ) {
                                        removeSequence(entry)
                                    }
                                }
                            }
                        }
                    }

                    Button {
                        showAddCustomPopover = true
                    } label: {
                        Label("Add custom", systemImage: "plus.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .popover(isPresented: $showAddCustomPopover) {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Custom Stop Sequence")
                                .font(.headline)
                            TextField("Enter sequence", text: $customStopInput)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 200)
                                .onSubmit {
                                    let trimmed = customStopInput.trimmingCharacters(in: .whitespaces)
                                    guard !trimmed.isEmpty else { return }
                                    addCustomSequence(trimmed)
                                    customStopInput = ""
                                    showAddCustomPopover = false
                                }
                            HStack {
                                Button("Cancel") {
                                    customStopInput = ""
                                    showAddCustomPopover = false
                                }
                                Spacer()
                                Button("Add") {
                                    addCustomSequence(customStopInput.trimmingCharacters(in: .whitespaces))
                                    customStopInput = ""
                                    showAddCustomPopover = false
                                }
                                .disabled(customStopInput.trimmingCharacters(in: .whitespaces).isEmpty)
                                .keyboardShortcut(.defaultAction)
                            }
                        }
                        .padding()
                        .frame(width: 280)
                    }
                }
                .accessibilityLabel("Stop sequences")
            }
        }
        .formStyle(.grouped)
        .controlSize(.regular)
        .transaction { $0.animation = nil }
        .onAppear { syncPresetToSliders() }
        .onChange(of: temperature) { _, _ in generationPreset = GenerationPreset.custom.rawValue }
        .onChange(of: topP) { _, _ in generationPreset = GenerationPreset.custom.rawValue }
        .onChange(of: topK) { _, _ in generationPreset = GenerationPreset.custom.rawValue }
        .onChange(of: repeatPenalty) { _, _ in generationPreset = GenerationPreset.custom.rawValue }
    }

    private func syncPresetToSliders() {
        guard let preset = GenerationPreset(rawValue: generationPreset), preset != .custom else { return }
        temperature = preset.temperature
        topP = preset.topP
        topK = preset.topK
        repeatPenalty = preset.repeatPenalty
    }

    // MARK: Stop Sequence Helpers

    private var activeSequences: [String] {
        StopSequenceStorage.decode(stopSequences)
    }

    private var customEntries: [String] {
        activeSequences.filter { StopSequencePreset.preset(forValue: $0) == nil }
    }

    private func togglePreset(_ preset: StopSequencePreset) {
        var current = activeSequences
        if let index = current.firstIndex(of: preset.value) {
            current.remove(at: index)
        } else {
            current.append(preset.value)
        }
        stopSequences = StopSequenceStorage.encode(current)
    }

    private func addCustomSequence(_ value: String) {
        var current = activeSequences
        guard !value.isEmpty, !current.contains(value) else { return }
        current.append(value)
        stopSequences = StopSequenceStorage.encode(current)
    }

    private func removeSequence(_ value: String) {
        var current = activeSequences
        current.removeAll { $0 == value }
        stopSequences = StopSequenceStorage.encode(current)
    }

    private func displayLabel(for value: String) -> String {
        if let preset = StopSequencePreset.preset(forValue: value) {
            return preset.label
        }
        return value
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
    }
}

// MARK: - Chat & Memory Tab

private struct ChatMemoryTab: View {
    @ObservedObject var viewModel: ChatViewModel
    @AppStorage(SettingsKeys.chatDbPathOverride) private var chatDbPathOverride = SettingsDefaults.chatDbPathOverride
    @State private var showClearAlert = false
    @State private var clearAction: ClearHistoryAction?

    private enum ClearHistoryAction {
        case thisConversation
        case all
    }

    private var defaultDbPath: String {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("LlamaInferenceDemo/chat.db").path
    }

    private var effectiveDbPath: String {
        chatDbPathOverride.isEmpty ? defaultDbPath : chatDbPathOverride
    }

    private var effectiveDataDir: String {
        (effectiveDbPath as NSString).deletingLastPathComponent
    }

    private var clearAlertTitle: String {
        clearAction == .all ? "Delete All Conversations?" : "Delete This Conversation?"
    }

    private var clearAlertMessage: String {
        if clearAction == .all {
            let stats = viewModel.dbStats
            return "This will permanently delete \(stats.conversationCount) conversation\(stats.conversationCount == 1 ? "" : "s") and \(stats.messageCount) message\(stats.messageCount == 1 ? "" : "s"). This cannot be undone."
        }
        return "This will permanently delete the current conversation. This cannot be undone."
    }

    private var dbSizeFormatted: String {
        ByteCountFormatter.string(fromByteCount: viewModel.dbStats.sizeBytes, countStyle: .file)
    }

    var body: some View {
        Form {
            Section("Database") {
                LabeledContent("File") {
                    HStack(spacing: 6) {
                        Text((effectiveDbPath as NSString).lastPathComponent)
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(.primary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(effectiveDbPath, forType: .string)
                        } label: {
                            Image(systemName: "doc.on.doc")
                                .font(.caption)
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel("Copy database path")
                        .help("Copy full path to clipboard")
                    }
                }

                LabeledContent("Size") {
                    Text(dbSizeFormatted)
                        .foregroundStyle(.secondary)
                }

                LabeledContent("WAL") {
                    Text(viewModel.dbStats.walMode.isEmpty ? "—" : viewModel.dbStats.walMode)
                        .foregroundStyle(.secondary)
                }

                LabeledContent("Integrity") {
                    HStack(spacing: 4) {
                        Image(systemName: viewModel.dbStats.integrityOk ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                            .foregroundStyle(viewModel.dbStats.integrityOk ? .green : .red)
                            .font(.caption)
                        Text(viewModel.dbStats.integrityOk ? "OK" : "Error")
                            .foregroundStyle(.secondary)
                    }
                }

                LabeledContent("Path") {
                    Text(effectiveDbPath)
                        .font(.system(.caption, design: .monospaced))
                        .lineLimit(2)
                        .truncationMode(.middle)
                        .foregroundStyle(.tertiary)
                        .textSelection(.enabled)
                }

                HStack(spacing: 12) {
                    Button("Reveal in Finder") {
                        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: effectiveDataDir)
                    }
                    .accessibilityLabel("Reveal data folder in Finder")
                }
            }

            Section("Custom Location") {
                HStack(alignment: .center, spacing: 6) {
                    StopSequenceChip(
                        label: "Default",
                        isSelected: chatDbPathOverride.isEmpty
                    ) {
                        chatDbPathOverride = ""
                    }

                    if !chatDbPathOverride.isEmpty {
                        let customFolder = ((chatDbPathOverride as NSString).deletingLastPathComponent as NSString).lastPathComponent
                        StopSequenceChip(
                            label: customFolder,
                            isSelected: true,
                            showRemove: true
                        ) {
                            chatDbPathOverride = ""
                        }
                    }

                    StopSequenceChip(
                        label: "Browse…",
                        isSelected: false
                    ) {
                        let panel = NSOpenPanel()
                        panel.canChooseDirectories = true
                        panel.canChooseFiles = false
                        panel.message = "Choose a folder for chat.db"
                        if panel.runModal() == .OK, let url = panel.url {
                            chatDbPathOverride = url.appendingPathComponent("chat.db").path
                        }
                    }
                }
                Text("Override database location. Restart required after changing.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("History") {
                HStack {
                    Text("\(viewModel.dbStats.conversationCount) conversation\(viewModel.dbStats.conversationCount == 1 ? "" : "s"), \(viewModel.dbStats.messageCount) message\(viewModel.dbStats.messageCount == 1 ? "" : "s")")
                        .foregroundStyle(.secondary)
                }

                HStack(spacing: 12) {
                    Button(role: .destructive) {
                        clearAction = .all
                        showClearAlert = true
                    } label: {
                        Label("Delete all", systemImage: "trash")
                    }
                    .disabled(viewModel.conversations.isEmpty)
                    .accessibilityLabel("Delete all conversations")
                    .accessibilityHint("Permanently deletes all conversations; cannot be undone")

                    if let selectedID = viewModel.selectedConversationID,
                       let conv = viewModel.conversations.first(where: { $0.id == selectedID }) {
                        let displayTitle = conv.title.count > 30 ? String(conv.title.prefix(27)) + "…" : conv.title
                        Button(role: .destructive) {
                            clearAction = .thisConversation
                            showClearAlert = true
                        } label: {
                            Label("Delete \"\(displayTitle)\"", systemImage: "bubble.left")
                        }
                        .accessibilityLabel("Delete conversation: \(conv.title)")
                        .accessibilityHint("Permanently deletes this conversation")
                    } else {
                        Button(role: .destructive) {
                            clearAction = .thisConversation
                            showClearAlert = true
                        } label: {
                            Label("Delete current", systemImage: "bubble.left")
                        }
                        .disabled(true)
                        .help("Select a conversation in the sidebar first")
                        .accessibilityLabel("Delete current conversation")
                        .accessibilityHint("Disabled. Select a conversation in the chat sidebar first.")
                    }
                }
            }
        }
        .formStyle(.grouped)
        .controlSize(.regular)
        .transaction { $0.animation = nil }
        .onAppear {
            Task { await viewModel.refreshDbStats() }
        }
        .alert(clearAlertTitle, isPresented: $showClearAlert) {
            Button("Cancel", role: .cancel) { clearAction = nil }
            Button("Delete", role: .destructive) {
                if let action = clearAction {
                    Task {
                        await performClear(action)
                        await viewModel.refreshDbStats()
                    }
                }
                clearAction = nil
            }
        } message: {
            Text(clearAlertMessage)
        }
    }

    private func performClear(_ action: ClearHistoryAction) async {
        switch action {
        case .thisConversation:
            if let id = viewModel.selectedConversationID {
                await viewModel.deleteConversation(id: id)
            }
        case .all:
            let ids = viewModel.conversations.map(\.id)
            for id in ids {
                await viewModel.deleteConversation(id: id)
            }
            await viewModel.newChat()
        }
    }
}

// MARK: - Appearance Tab

private struct AppearanceTab: View {
    @AppStorage(SettingsKeys.appTheme) private var appTheme = SettingsDefaults.appTheme
    @AppStorage(SettingsKeys.chatFontSize) private var chatFontSize = SettingsDefaults.chatFontSize
    @AppStorage(SettingsKeys.sidebarVisibleOnLaunch) private var sidebarVisibleOnLaunch = SettingsDefaults.sidebarVisibleOnLaunch

    var body: some View {
        Form {
            Section("Theme") {
                Picker("Color scheme", selection: $appTheme) {
                    Text("Light").tag("light")
                    Text("Dark").tag("dark")
                    Text("System").tag("system")
                }
                .accessibilityLabel("Color scheme")
                .accessibilityValue(appTheme)
            }

            Section("Reading") {
                LabeledContent("Chat font size") {
                    HStack(spacing: 8) {
                        Text("\(chatFontSize)")
                            .monospacedDigit()
                            .frame(minWidth: 24, alignment: .trailing)
                        Stepper("", value: $chatFontSize, in: 12...18)
                            .labelsHidden()
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Chat font size")
                    .accessibilityValue("\(chatFontSize) points")
                }
            }

            Section("Sidebar") {
                LabeledContent("Show sidebar on launch") {
                    Toggle("", isOn: $sidebarVisibleOnLaunch)
                        .toggleStyle(.switch)
                        .accessibilityLabel("Show sidebar on launch")
                        .accessibilityHint("Show the conversation list when the app starts")
                }
            }
        }
        .formStyle(.grouped)
        .controlSize(.regular)
        .transaction { $0.animation = nil }
    }
}

// MARK: - Advanced Tab

private struct AdvancedTab: View {
    @AppStorage(SettingsKeys.logLevel) private var logLevel = SettingsDefaults.logLevel
    @AppStorage(SettingsKeys.logRotationSizeMB) private var logRotationSizeMB = SettingsDefaults.logRotationSizeMB
    @AppStorage(SettingsKeys.logJsonFormat) private var logJsonFormat = SettingsDefaults.logJsonFormat
    @State private var resetScope: ResetScope?
    @State private var showResetAlert = false

    private static var logDirectory: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("LlamaInferenceDemo/Logs")
    }

    private static var logFilePath: String {
        logDirectory.appendingPathComponent("llama-inference.log").path
    }

    var body: some View {
        Form {
            Section("Logging") {
                Picker("Log level", selection: $logLevel) {
                    Text("Debug").tag("debug")
                    Text("Info").tag("info")
                    Text("Warning").tag("warning")
                    Text("Error").tag("error")
                }
                .accessibilityLabel("Log level")
                .accessibilityValue(logLevel)

                Text("Logs include app events and errors. Debug adds technical details. Conversations are never logged.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                LabeledContent("Log file") {
                    HStack(spacing: 6) {
                        Text(Self.logFilePath)
                            .font(.system(.caption, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(Self.logFilePath, forType: .string)
                        } label: {
                            Image(systemName: "doc.on.doc")
                                .font(.caption)
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel("Copy log path")
                    }
                }

                Button {
                    let fm = FileManager.default
                    let dir = Self.logDirectory
                    if !fm.fileExists(atPath: dir.path) {
                        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
                    }
                    NSWorkspace.shared.open(dir)
                } label: {
                    Label("Open log folder", systemImage: "folder")
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Open log folder in Finder")

                LabeledContent("Rotation size") {
                    Picker("", selection: $logRotationSizeMB) {
                        Text("5 MB").tag(5)
                        Text("10 MB").tag(10)
                        Text("50 MB").tag(50)
                        Text("100 MB").tag(100)
                    }
                    .labelsHidden()
                    .frame(width: 88)
                    .accessibilityLabel("Log rotation size")
                    .accessibilityValue("\(logRotationSizeMB) megabytes")
                }

                LabeledContent("Structured JSON logs") {
                    Toggle("", isOn: $logJsonFormat)
                        .toggleStyle(.switch)
                        .accessibilityLabel("Structured JSON logs")
                        .accessibilityHint("Write log entries as JSON objects for machine parsing")
                }
            }

            Section("Reset") {
                Button(role: .destructive) {
                    resetScope = .generation
                    showResetAlert = true
                } label: {
                    Label("Reset generation settings", systemImage: "slider.horizontal.3")
                }
                .accessibilityLabel("Reset generation settings")
                .accessibilityHint("Resets sampling, tokens, and stop sequences to defaults")

                Button(role: .destructive) {
                    resetScope = .models
                    showResetAlert = true
                } label: {
                    Label("Reset model settings", systemImage: "cpu")
                }
                .accessibilityLabel("Reset model settings")
                .accessibilityHint("Resets model paths, workers, context, and GPU layers to defaults")

                Button(role: .destructive) {
                    resetScope = .ui
                    showResetAlert = true
                } label: {
                    Label("Reset UI settings", systemImage: "paintbrush")
                }
                .accessibilityLabel("Reset UI settings")
                .accessibilityHint("Resets theme, font size, and sidebar visibility to defaults")

                Button(role: .destructive) {
                    resetScope = .all
                    showResetAlert = true
                } label: {
                    Label("Reset all settings to defaults", systemImage: "arrow.counterclockwise")
                }
                .accessibilityLabel("Reset all settings")
                .accessibilityHint("Restores all preferences to default values")
            }

            Section("About") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Llama Chat")
                        .font(.headline)
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text("Version 0.2.0 (Alpha)")
                            .foregroundStyle(.secondary)
                        Text("•")
                            .foregroundStyle(.tertiary)
                        Text("Build 2026.02.13")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Text("Developer: Dr. Mikholae Hutchinson")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack(alignment: .firstTextBaseline, spacing: 4) {
                        Text("Made using")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Link("SwiftPython", destination: URL(string: "https://swiftpython.dev")!)
                            .font(.caption)
                        Text("•")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                        Text("All rights reserved.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 2)
            }
        }
        .formStyle(.grouped)
        .controlSize(.regular)
        .transaction { $0.animation = nil }
        .alert(resetAlertTitle, isPresented: $showResetAlert) {
            Button("Cancel", role: .cancel) { resetScope = nil }
            Button("Reset", role: .destructive) {
                resetScope?.reset()
                resetScope = nil
            }
        } message: {
            Text(resetAlertMessage)
        }
    }

    private var resetAlertTitle: String {
        guard let scope = resetScope else { return "Reset?" }
        return "Reset \(scope.displayName)?"
    }

    private var resetAlertMessage: String {
        guard let scope = resetScope else { return "" }
        switch scope {
        case .generation:
            return "Sampling parameters, token limits, and stop sequences will be restored to defaults."
        case .models:
            return "Model paths, worker count, context size, and GPU layer settings will be restored to defaults."
        case .ui:
            return "Theme, font size, and sidebar settings will be restored to defaults."
        case .all:
            return "All preferences will be restored to their default values."
        }
    }
}
