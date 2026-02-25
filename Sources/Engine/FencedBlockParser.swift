import Foundation

// MARK: - Fenced Code Block Extraction

/// Represents a single fenced code block parsed from markdown source.
public struct FencedBlock: Sendable {
    public let language: String?
    public let code: String

    public init(language: String?, code: String) {
        self.language = language
        self.code = code
    }
}

/// Parses fenced code blocks from raw markdown using a single regex pass.
public enum FencedBlockParser {
    /// Matches ``` or ~~~ fenced code blocks with optional language hint.
    private static let pattern = #"(?:^|\n)(`{3,}|~{3,})[ \t]*(\w+)?[ \t]*\n([\s\S]*?)\n\1"#

    /// Languages that Textual renders as special blocks (not code blocks).
    /// These must be excluded so tracker indices align with `CodeBlockStyle.makeBody` calls.
    private static let excludedLanguages: Set<String> = ["math"]

    public static func extract(from markdown: String) -> [FencedBlock] {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return []
        }
        let range = NSRange(markdown.startIndex..., in: markdown)
        let matches = regex.matches(in: markdown, range: range)
        return matches.compactMap { match in
            guard match.numberOfRanges >= 4,
                  let codeRange = Range(match.range(at: 3), in: markdown) else {
                return nil
            }
            let lang: String?
            if let langRange = Range(match.range(at: 2), in: markdown) {
                lang = String(markdown[langRange])
            } else {
                lang = nil
            }
            // Skip fenced blocks that Textual handles as non-code (e.g. ```math)
            if let lang, excludedLanguages.contains(lang.lowercased()) {
                return nil
            }
            return FencedBlock(language: lang, code: String(markdown[codeRange]))
        }
    }
}
