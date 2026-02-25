import Foundation

/// Parses Model Hub search text and extracts an optional provider filter.
public enum ModelHubSearchParser {
    /// Parse search text into a free-text query and optional `@provider` override.
    ///
    /// Examples:
    /// - `"nanbeige @bartowski"` -> `("nanbeige", "bartowski")`
    /// - `"@bartowski nanbeige"` -> `("nanbeige", "bartowski")`
    public static func parse(_ text: String) -> (query: String, authorOverride: String?) {
        let tokens = text.split(whereSeparator: \.isWhitespace).map(String.init)
        var queryParts: [String] = []
        var authorOverride: String?

        for token in tokens {
            if token == "@" {
                // Ignore a bare "@" while the user is still typing.
                continue
            }
            if token.hasPrefix("@"), token.count > 1 {
                let provider = String(token.dropFirst())
                if !provider.isEmpty, authorOverride == nil {
                    authorOverride = provider
                }
            } else {
                queryParts.append(token)
            }
        }

        return (
            query: queryParts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines),
            authorOverride: authorOverride
        )
    }
}
