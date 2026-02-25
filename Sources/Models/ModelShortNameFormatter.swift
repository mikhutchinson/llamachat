import Foundation

/// Produces compact model-family labels suitable for toolbar pills.
public enum ModelShortNameFormatter {

    public static func shortName(fromModelPath path: String) -> String {
        guard !path.isEmpty else { return "No Model" }
        let filename = (path as NSString).lastPathComponent
        return shortName(fromFilename: filename)
    }

    public static func shortName(fromFilename filename: String) -> String {
        let parsed = GGUFModelInfo.parse(filename: filename, sizeBytes: 0).displayName
        let tokens = parsed
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
        guard let first = tokens.first, !first.isEmpty else {
            return fallbackStem(filename)
        }

        let lowerFirst = first.lowercased()
        if lowerFirst == "llama" {
            if tokens.count > 1, isVersionToken(tokens[1]) {
                return "Llama \(tokens[1])"
            }
            return "Llama"
        }

        if lowerFirst == "gemma" {
            if tokens.count > 1, isVersionToken(tokens[1]) {
                return "Gemma \(tokens[1])"
            }
            return "Gemma"
        }

        if lowerFirst == "qwen", tokens.count > 1, isVersionToken(tokens[1]) {
            return "Qwen \(tokens[1])"
        }

        if lowerFirst.hasPrefix("qwen") {
            return canonicalizeFamilyToken(first)
        }

        return canonicalizeFamilyToken(first)
    }

    private static func isVersionToken(_ token: String) -> Bool {
        token.range(of: #"^[0-9]+(?:\.[0-9]+)?$"#, options: .regularExpression) != nil
    }

    private static func canonicalizeFamilyToken(_ token: String) -> String {
        if token.lowercased().hasPrefix("qwen") {
            let suffix = token.dropFirst(4)
            return suffix.isEmpty ? "Qwen" : "Qwen\(suffix)"
        }
        if token.isEmpty { return token }
        let lower = token.lowercased()
        return lower.prefix(1).uppercased() + lower.dropFirst()
    }

    private static func fallbackStem(_ filename: String) -> String {
        var stem = (filename as NSString).lastPathComponent
        if stem.hasSuffix(".gguf") { stem.removeLast(5) }
        if stem.hasSuffix(".bin") { stem.removeLast(4) }
        if stem.isEmpty { return "No Model" }
        return stem
    }
}
