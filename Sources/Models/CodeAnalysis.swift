import Foundation

/// Lightweight static analysis result from Python's `ast` module.
/// Used to enrich LLM review prompts with structural context.
public struct CodeAnalysis: Sendable, Equatable {
    public let functions: [String]
    public let classes: [String]
    public let imports: [String]
    public let issues: [String]
    public let lineCount: Int

    public init(functions: [String], classes: [String], imports: [String], issues: [String], lineCount: Int) {
        self.functions = functions
        self.classes = classes
        self.imports = imports
        self.issues = issues
        self.lineCount = lineCount
    }

    /// Parse from the JSON dict returned by the AST analysis script.
    public static func from(json: String) -> CodeAnalysis? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return CodeAnalysis(
            functions: obj["functions"] as? [String] ?? [],
            classes: obj["classes"] as? [String] ?? [],
            imports: obj["imports"] as? [String] ?? [],
            issues: obj["issues"] as? [String] ?? [],
            lineCount: obj["line_count"] as? Int ?? 0
        )
    }

    /// Format as a concise summary string for inclusion in an LLM prompt.
    public var promptSummary: String {
        var parts: [String] = []
        if !functions.isEmpty {
            parts.append("- Functions: \(functions.joined(separator: ", "))")
        }
        if !classes.isEmpty {
            parts.append("- Classes: \(classes.joined(separator: ", "))")
        }
        if !imports.isEmpty {
            parts.append("- Imports: \(imports.joined(separator: ", "))")
        }
        if !issues.isEmpty {
            for issue in issues {
                parts.append("- Potential issue: \(issue)")
            }
        }
        parts.append("- Lines: \(lineCount)")
        return parts.joined(separator: "\n")
    }
}
