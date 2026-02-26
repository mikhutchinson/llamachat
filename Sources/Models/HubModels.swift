import Foundation

// MARK: - HF API Response Models

public struct HFModelSummary: Codable, Identifiable, Sendable {
    public let id: String
    public let likes: Int
    public let downloads: Int
    public let tags: [String]
    public let pipelineTag: String?
    public let createdAt: String?

    public var author: String { id.components(separatedBy: "/").first ?? "" }
    public var modelName: String { id.components(separatedBy: "/").last ?? id }
    public var isVLM: Bool { pipelineTag == "image-text-to-text" || tags.contains("vision") }
    public var vlmArchitecture: String? { nil }

    /// Human-friendly description derived from the HF pipeline_tag.
    public var pipelineDescription: String? {
        pipelineTag.flatMap { Self.descriptionForTag($0) }
    }

    static func descriptionForTag(_ tag: String) -> String {
        switch tag {
        case "text-generation": return "Text Generation"
        case "image-text-to-text": return "Vision-Language"
        case "summarization": return "Summarization"
        case "text2text-generation": return "Text-to-Text"
        case "text-classification": return "Text Classification"
        case "question-answering": return "Question Answering"
        case "translation": return "Translation"
        case "conversational": return "Conversational"
        default: return tag.replacingOccurrences(of: "-", with: " ").capitalized
        }
    }
}

public struct HFModelDetail: Codable, Sendable {
    public let id: String
    public let siblings: [HFSibling]
    public let gguf: HFGGUFInfo?
    public let cardData: HFCardData?
    public let likes: Int
    public let downloads: Int
    public let tags: [String]
    public let pipelineTag: String?

    public var author: String { id.components(separatedBy: "/").first ?? "" }
    public var modelName: String { id.components(separatedBy: "/").last ?? id }
    public var isVLM: Bool { pipelineTag == "image-text-to-text" || tags.contains("vision") }
    public var vlmArchitecture: String? { gguf?.architecture }

    private enum CodingKeys: String, CodingKey {
        case id
        case siblings
        case gguf
        case cardData
        case likes
        case downloads
        case tags
        case pipelineTag
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        siblings = (try? container.decode([HFSibling].self, forKey: .siblings)) ?? []
        gguf = try? container.decode(HFGGUFInfo.self, forKey: .gguf)
        cardData = try? container.decode(HFCardData.self, forKey: .cardData)
        likes = (try? container.decode(Int.self, forKey: .likes)) ?? 0
        downloads = (try? container.decode(Int.self, forKey: .downloads)) ?? 0
        tags = (try? container.decode([String].self, forKey: .tags)) ?? []
        pipelineTag = try? container.decode(String.self, forKey: .pipelineTag)
    }
}

public struct HFSibling: Codable, Sendable {
    public let rfilename: String
    public let lfs: HFLfsInfo?
}

public struct HFLfsInfo: Codable, Sendable {
    public let sha256: String?
    public let size: Int64?
}

public struct HFGGUFInfo: Codable, Sendable {
    public let total: Int64?
    public let architecture: String?
    public let contextLength: Int?
}

public struct HFCardData: Codable, Sendable {
    public let pipelineTag: String?
    public let baseModel: String?
}

// MARK: - Quantization

public enum QuantLevel: String, CaseIterable, Sendable {
    case Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q4_K_L
    case Q5_K_S, Q5_K_M, Q5_K_L, Q6_K, Q8_0, f16, bf16

    public var displayName: String { rawValue.replacingOccurrences(of: "_", with: " ") }

    public var tier: String {
        switch self {
        case .Q2_K: return "Tiny"
        case .Q3_K_S, .Q3_K_M, .Q3_K_L: return "Small"
        case .Q4_0, .Q4_K_S, .Q4_K_M, .Q4_K_L: return "Medium"
        case .Q5_K_S, .Q5_K_M, .Q5_K_L: return "Good"
        case .Q6_K, .Q8_0: return "High"
        case .f16, .bf16: return "Full"
        }
    }

    /// Parse quantization level from a GGUF filename.
    public static func detect(from filename: String) -> QuantLevel? {
        let base = filename
            .replacingOccurrences(of: ".gguf", with: "")
            .uppercased()
        // Try longest matches first to avoid Q4_K matching before Q4_K_M
        let sorted = QuantLevel.allCases.sorted { $0.rawValue.count > $1.rawValue.count }
        for level in sorted {
            let pattern = level.rawValue.uppercased()
            if base.hasSuffix(pattern) || base.contains("-\(pattern)") || base.contains("_\(pattern)") {
                return level
            }
        }
        return nil
    }
}

// MARK: - GGUF File

public struct GGUFFile: Identifiable, Sendable {
    public let id: String
    public let filename: String
    public let repoId: String
    public let quantLevel: QuantLevel?
    public let isMMProj: Bool
    public let estimatedSize: Int64?
    public let expectedSHA256: String?

    public var downloadURL: URL {
        URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filename)")!
    }

    /// Parse a sibling entry into a typed GGUFFile.
    public static func from(sibling: HFSibling, repoId: String) -> GGUFFile? {
        let name = sibling.rfilename
        guard name.hasSuffix(".gguf") else { return nil }
        let isProj = isProjectionFilename(name)
        return GGUFFile(
            id: name,
            filename: name,
            repoId: repoId,
            quantLevel: QuantLevel.detect(from: name),
            isMMProj: isProj,
            estimatedSize: sibling.lfs?.size,
            expectedSHA256: sibling.lfs?.sha256
        )
    }

    private static func isProjectionFilename(_ filename: String) -> Bool {
        let stem = (filename as NSString).deletingPathExtension.lowercased()
        return stem.contains("mmproj")
    }
}

// MARK: - Model Role

public enum ModelRole: String, CaseIterable, Sendable {
    case chatModel = "Chat Model"
    case vlmModel = "VLM Model"
    case vlmProjection = "VLM Projection"
    case summarizer = "Summarizer"

    /// Suggest the most likely role for a file based on its properties and pipeline tag.
    public static func suggest(for file: GGUFFile, isVLMRepo: Bool, pipelineTag: String? = nil) -> ModelRole {
        if file.isMMProj { return .vlmProjection }
        if isVLMRepo { return .vlmModel }
        if let tag = pipelineTag {
            switch tag {
            case "summarization": return .summarizer
            case "image-text-to-text": return .vlmModel
            default: break
            }
        }
        return .chatModel
    }
}

// MARK: - Download State

public enum DownloadState: Sendable, Equatable {
    case queued
    case downloading
    case paused
    case verifying
    case completed(localPath: String)
    case failed(String)
    case cancelled
}

public struct DownloadTask: Identifiable, Sendable, Equatable {
    public let id: String
    public let filename: String
    public let repoId: String
    public var totalBytes: Int64
    public var downloadedBytes: Int64
    public var state: DownloadState
    public var resumeData: Data?

    public var progress: Double {
        totalBytes > 0 ? Double(downloadedBytes) / Double(totalBytes) : 0
    }
}
