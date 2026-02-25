import Foundation

/// Centralized UserDefaults keys and default values for settings and reset logic.
enum SettingsKeys {
    static let modelPath = "modelPath"
    static let summarizerModelPath = "summarizerModelPath"
    static let workerCount = "workerCount"
    static let contextSize = "contextSize"
    static let nGpuLayers = "nGpuLayers"
    static let useSharedMemory = "useSharedMemory"

    static let maxTokens = "maxTokens"
    static let temperature = "temperature"
    static let topP = "topP"
    static let topK = "topK"
    static let repeatPenalty = "repeatPenalty"
    static let generationPreset = "generationPreset"
    static let stopSequences = "stopSequences"
    static let responseMode = "responseMode"

    static let vlmModelPath = "vlmModelPath"
    static let vlmClipPath = "vlmClipPath"
    static let vlmIdleTimeoutSecs = "vlmIdleTimeoutSecs"
    static let recentModelPaths = "recentModelPaths"

    static let chatDbPathOverride = "chatDbPathOverride"
    static let codeActEnabled = "codeActEnabled"

    static let appTheme = "appTheme"
    static let chatFontSize = "chatFontSize"
    static let sidebarVisibleOnLaunch = "sidebarVisibleOnLaunch"
    static let modelHubSearchText = "modelHubSearchText"
    static let modelHubDownloadedSearchText = "modelHubDownloadedSearchText"
    static let modelHubSortOrder = "modelHubSortOrder"
    static let modelHubShowDownloadedOnly = "modelHubShowDownloadedOnly"
    static let modelHubSplitVisibility = "modelHubSplitVisibility"

    static let logLevel = "logLevel"
    static let logRotationSizeMB = "logRotationSizeMB"
    static let logJsonFormat = "logJsonFormat"

    static let modelDownloadDirectory = "modelDownloadDirectory"
}

/// Default values for reset logic.
enum SettingsDefaults {
    static let modelPath = ""
    static let summarizerModelPath = ""
    static let workerCount = 2
    static let contextSize = 4096
    static let nGpuLayers = -1
    static let useSharedMemory = true

    static let maxTokens = 512
    static let temperature = 0.7
    static let topP = 0.95
    static let topK = 40
    static let repeatPenalty = 1.1
    static let generationPreset = "balanced"
    static let stopSequences = ""
    static let responseMode = "auto"

    static let vlmModelPath = ""
    static let vlmClipPath = ""
    static let vlmIdleTimeoutSecs = 300
    static let recentModelPaths = ""

    static let chatDbPathOverride = ""
    static let codeActEnabled = false

    static let appTheme = "dark"
    static let chatFontSize = 14
    static let sidebarVisibleOnLaunch = true
    static let modelHubSearchText = ""
    static let modelHubDownloadedSearchText = ""
    static let modelHubSortOrder = "downloads"
    static let modelHubShowDownloadedOnly = false
    static let modelHubSplitVisibility = "all"

    static let logLevel = "info"
    static let logRotationSizeMB = 10
    static let logJsonFormat = false

    static let modelDownloadDirectory = ""  // empty = ~/Models/gguf/
}

/// Generation preset definitions (temp, topP, topK, repeatPenalty).
enum GenerationPreset: String, CaseIterable {
    case creative
    case precise
    case balanced
    case custom

    var displayName: String {
        switch self {
        case .creative: return "Creative"
        case .precise: return "Precise"
        case .balanced: return "Balanced"
        case .custom: return "Custom"
        }
    }

    var temperature: Double {
        switch self {
        case .creative: return 0.9
        case .precise: return 0.2
        case .balanced: return 0.7
        case .custom: return SettingsDefaults.temperature
        }
    }

    var topP: Double {
        switch self {
        case .creative: return 0.98
        case .precise: return 0.9
        case .balanced: return 0.95
        case .custom: return SettingsDefaults.topP
        }
    }

    var topK: Int {
        switch self {
        case .creative: return 60
        case .precise: return 20
        case .balanced: return 40
        case .custom: return SettingsDefaults.topK
        }
    }

    var repeatPenalty: Double {
        switch self {
        case .creative: return 1.05
        case .precise: return 1.15
        case .balanced: return 1.1
        case .custom: return SettingsDefaults.repeatPenalty
        }
    }
}

// MARK: - Stop Sequence Presets

/// Preset stop sequences for UI-driven chip selection.
struct StopSequencePreset: Identifiable, Hashable, Sendable {
    let id: String
    let label: String
    let value: String
    let category: String

    static let categories: [String] = ["Newline", "Model Tokens", "Generic"]

    static let all: [StopSequencePreset] = [
        StopSequencePreset(id: "newline", label: "\\n", value: "\n", category: "Newline"),
        StopSequencePreset(id: "double-newline", label: "\\n\\n", value: "\n\n", category: "Newline"),
        StopSequencePreset(id: "crlf", label: "\\r\\n", value: "\r\n", category: "Newline"),
        StopSequencePreset(id: "endoftext", label: "<|endoftext|>", value: "<|endoftext|>", category: "Model Tokens"),
        StopSequencePreset(id: "eos", label: "</s>", value: "</s>", category: "Model Tokens"),
        StopSequencePreset(id: "im-end", label: "<|im_end|>", value: "<|im_end|>", category: "Model Tokens"),
        StopSequencePreset(id: "end", label: "END", value: "END", category: "Generic"),
        StopSequencePreset(id: "eof", label: "EOF", value: "EOF", category: "Generic"),
        StopSequencePreset(id: "separator", label: "---", value: "---", category: "Generic"),
    ]

    /// Find the preset matching a raw stop sequence value, if any.
    static func preset(forValue value: String) -> StopSequencePreset? {
        all.first { $0.value == value }
    }
}

// MARK: - Stop Sequence Storage

/// JSON-based stop sequence storage — reads JSON array, falls back to legacy comma-separated.
enum StopSequenceStorage {
    static func decode(_ raw: String) -> [String] {
        guard !raw.isEmpty else { return [] }
        // New format: JSON array
        if let data = raw.data(using: .utf8),
           let array = try? JSONDecoder().decode([String].self, from: data) {
            return array
        }
        // Legacy comma-separated migration: map known preset labels to actual values
        return raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\"")) }
            .filter { !$0.isEmpty }
            .map { legacyValue in
                if let preset = StopSequencePreset.all.first(where: { $0.label == legacyValue }) {
                    return preset.value
                }
                return legacyValue
            }
    }

    static func encode(_ sequences: [String]) -> String {
        guard !sequences.isEmpty else { return "" }
        guard let data = try? JSONEncoder().encode(sequences),
              let json = String(data: data, encoding: .utf8) else { return "" }
        return json
    }
}

// MARK: - Reset Scope

/// Scoped reset categories for granular settings reset.
enum ResetScope: String, CaseIterable {
    case generation
    case models
    case ui
    case all

    var displayName: String {
        switch self {
        case .generation: return "Generation"
        case .models: return "Models"
        case .ui: return "UI"
        case .all: return "All Settings"
        }
    }

    func reset() {
        let ud = UserDefaults.standard
        switch self {
        case .generation:
            ud.set(SettingsDefaults.maxTokens, forKey: SettingsKeys.maxTokens)
            ud.set(SettingsDefaults.temperature, forKey: SettingsKeys.temperature)
            ud.set(SettingsDefaults.topP, forKey: SettingsKeys.topP)
            ud.set(SettingsDefaults.topK, forKey: SettingsKeys.topK)
            ud.set(SettingsDefaults.repeatPenalty, forKey: SettingsKeys.repeatPenalty)
            ud.set(SettingsDefaults.generationPreset, forKey: SettingsKeys.generationPreset)
            ud.set(SettingsDefaults.stopSequences, forKey: SettingsKeys.stopSequences)
            ud.set(SettingsDefaults.responseMode, forKey: SettingsKeys.responseMode)
        case .models:
            ud.set(SettingsDefaults.modelPath, forKey: SettingsKeys.modelPath)
            ud.set(SettingsDefaults.summarizerModelPath, forKey: SettingsKeys.summarizerModelPath)
            ud.set(SettingsDefaults.workerCount, forKey: SettingsKeys.workerCount)
            ud.set(SettingsDefaults.contextSize, forKey: SettingsKeys.contextSize)
            ud.set(SettingsDefaults.nGpuLayers, forKey: SettingsKeys.nGpuLayers)
            ud.set(SettingsDefaults.useSharedMemory, forKey: SettingsKeys.useSharedMemory)
            ud.set(SettingsDefaults.vlmModelPath, forKey: SettingsKeys.vlmModelPath)
            ud.set(SettingsDefaults.vlmClipPath, forKey: SettingsKeys.vlmClipPath)
            ud.set(SettingsDefaults.vlmIdleTimeoutSecs, forKey: SettingsKeys.vlmIdleTimeoutSecs)
            ud.set(SettingsDefaults.recentModelPaths, forKey: SettingsKeys.recentModelPaths)
        case .ui:
            ud.set(SettingsDefaults.appTheme, forKey: SettingsKeys.appTheme)
            ud.set(SettingsDefaults.chatFontSize, forKey: SettingsKeys.chatFontSize)
            ud.set(SettingsDefaults.sidebarVisibleOnLaunch, forKey: SettingsKeys.sidebarVisibleOnLaunch)
            ud.set(SettingsDefaults.modelHubSearchText, forKey: SettingsKeys.modelHubSearchText)
            ud.set(SettingsDefaults.modelHubDownloadedSearchText, forKey: SettingsKeys.modelHubDownloadedSearchText)
            ud.set(SettingsDefaults.modelHubSortOrder, forKey: SettingsKeys.modelHubSortOrder)
            ud.set(SettingsDefaults.modelHubShowDownloadedOnly, forKey: SettingsKeys.modelHubShowDownloadedOnly)
            ud.set(SettingsDefaults.modelHubSplitVisibility, forKey: SettingsKeys.modelHubSplitVisibility)
        case .all:
            ResetScope.generation.reset()
            ResetScope.models.reset()
            ResetScope.ui.reset()
            ud.set(SettingsDefaults.chatDbPathOverride, forKey: SettingsKeys.chatDbPathOverride)
            ud.set(SettingsDefaults.logLevel, forKey: SettingsKeys.logLevel)
            ud.set(SettingsDefaults.logRotationSizeMB, forKey: SettingsKeys.logRotationSizeMB)
            ud.set(SettingsDefaults.logJsonFormat, forKey: SettingsKeys.logJsonFormat)
        }
    }
}
