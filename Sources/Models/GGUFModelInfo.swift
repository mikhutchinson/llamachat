import Foundation

/// Parsed metadata from a GGUF filename — user-friendly display names (strips .gguf, quantization).
public struct GGUFModelInfo {
    public let displayName: String
    public let quantization: String?
    public let sizeBytes: Int64

    public var sizeFormatted: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }

    /// Rough RAM estimate: file size + ~20% overhead for runtime buffers.
    public var estimatedRAM: String {
        let estimate = Double(sizeBytes) * 1.2
        return ByteCountFormatter.string(fromByteCount: Int64(estimate), countStyle: .memory)
    }

    /// Base model name from a GGUF filename (strips path, .gguf, quant). For matching downloaded files.
    /// Example: "Qwen3-4B-Q4_K_M.gguf" → "Qwen3-4B"
    public static func baseName(fromFilename filename: String) -> String {
        var stem = filename
        if stem.hasSuffix(".gguf") { stem = String(stem.dropLast(5)) }
        if let slashIndex = stem.lastIndex(of: "/") { stem = String(stem[stem.index(after: slashIndex)...]) }
        return stemByRemovingQuant(stem)
    }

    /// Base model name from HF repo name (strips -GGUF). For matching Model Hub repo to local files.
    /// Example: "Qwen3-4B-GGUF" → "Qwen3-4B"
    public static func baseNameFromHFModel(_ modelName: String) -> String {
        for suffix in ["-GGUF", "-gguf", "-Gguf"] {
            if modelName.hasSuffix(suffix) { return String(modelName.dropLast(suffix.count)) }
        }
        return modelName
    }

    /// Parse a GGUF filename into structured model info.
    /// Examples:
    ///   "Qwen3-4B-Q4_K_M.gguf" → name: "Qwen3 4B", quant: "Q4_K_M"
    ///   "Llama-3.2-1B-Instruct-Q4_K_M.gguf" → name: "Llama 3.2 1B Instruct", quant: "Q4_K_M"
    public static func parse(filename: String, sizeBytes: Int64) -> GGUFModelInfo {
        var stem = filename
        if stem.hasSuffix(".gguf") { stem = String(stem.dropLast(5)) }
        if let slashIndex = stem.lastIndex(of: "/") { stem = String(stem[stem.index(after: slashIndex)...]) }

        let (namePart, quantization) = stemByRemovingQuantWithValue(stem)

        let displayName = namePart
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
            .components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
            .joined(separator: " ")

        return GGUFModelInfo(
            displayName: displayName.isEmpty ? filename : displayName,
            quantization: quantization,
            sizeBytes: sizeBytes
        )
    }

    private static func stemByRemovingQuant(_ stem: String) -> String {
        stemByRemovingQuantWithValue(stem).base
    }

    private static func stemByRemovingQuantWithValue(_ stem: String) -> (base: String, quant: String?) {
        let quantPatterns = [
            "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
            "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
            "IQ4_XS", "IQ4_NL",
            "Q2_K_S", "Q2_K",
            "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K",
            "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q4_K", "Q4_0", "Q4_1",
            "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q5_K", "Q5_0", "Q5_1",
            "Q6_K", "Q6_0",
            "Q8_0", "Q8_1",
            "F16", "F32", "BF16",
        ]

        var quantization: String?
        var namePart = stem
        for pattern in quantPatterns {
            let dashVariant = "-\(pattern)"
            let underscoreVariant = "_\(pattern)"
            if namePart.hasSuffix(dashVariant) {
                quantization = pattern
                namePart = String(namePart.dropLast(dashVariant.count))
                break
            } else if namePart.hasSuffix(underscoreVariant) {
                quantization = pattern
                namePart = String(namePart.dropLast(underscoreVariant.count))
                break
            }
        }
        return (namePart, quantization)
    }
}
