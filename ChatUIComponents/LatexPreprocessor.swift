import Foundation

// MARK: - LaTeX Pre-processing (runs off main actor)

/// Convert LaTeX delimiters in raw LLM output into markdown-compatible representations
/// so Textual's markdown math extension can parse them consistently.
public enum LatexPreprocessor {
    // MARK: - Main Preprocessing

    /// Pre-process content: normalize LaTeX delimiters for Textual markdown parsing.
    /// This is intentionally a pure function suitable for calling from `Task.detached`.
    ///
    /// Pattern order matters — more specific delimiters are matched first:
    ///   0. De-indent accidental indented math lines so markdown does not treat them as code blocks.
    ///   1. Heal truncated math: model cut mid-token (e.g. `$g_{tt} = -(1 + 2\`) → append `$` to close span.
    ///   2. Escape likely currency `$` so it won't be interpreted as math delimiters.
    ///   3. `$$...$$`  -> fenced ```math block
    ///   4. `\[...\]`  -> fenced ```math block
    ///   5. `\(...\)`  -> `$...$`
    ///   6. `$`...`$`  -> `$...$`
    ///   7. Normalize markdown-escaped subscripts (for example `g\_0`) to TeX (`g_0`).
    ///   8. Escape markdown control chars (except `_`) inside inline math spans.
    public static func preprocess(_ content: String) -> String {
        // 0a. Convert CodeAct <execute>…</execute> blocks to fenced ```python blocks
        // so the markdown renderer shows them as syntax-highlighted code instead of
        // raw tags and misinterpreted # comment lines.
        var result = replaceExecuteTags(content)

        // 0b. De-indent accidental indented math lines (markdown would otherwise
        // render them as code blocks and skip math parsing).
        result = deindentAccidentalMathCodeBlocks(result)

        // 1. Heal truncated math (model cut mid-command); must run before any $ processing.
        result = healTruncatedMath(result)

        // 2. Escape likely currency values like "$10" or "$1,000.50" so
        // markdown math parsing doesn't treat them as inline math delimiters.
        result = escapeLikelyCurrencyDollars(in: result)

        // 3. Block math: $$...$$ -> fenced math block.
        result = replacePattern(
            in: result,
            pattern: #"\$\$([\s\S]*?)\$\$"#,
            replacement: { match in
                let inner = match.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !inner.isEmpty else { return "$$$$" }
                let normalized = normalizeBlockMathContent(inner)
                return "\n```math\n\(normalized)\n```\n"
            }
        )

        // 4. Display math: \[...\] -> fenced math block
        result = replacePattern(
            in: result,
            pattern: #"\\\[([\s\S]*?)\\\]"#,
            replacement: { match in
                let inner = match.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !inner.isEmpty else { return "\\[\\]" }
                let normalized = normalizeBlockMathContent(inner)
                return "\n```math\n\(normalized)\n```\n"
            }
        )

        // 5. Inline math: \(...\) -> $...$
        result = replacePattern(
            in: result,
            pattern: #"\\\((.*?)\\\)"#,
            replacement: { match in
                let trimmed = match.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { return "\\(\\)" }
                return "$\(trimmed)$"
            }
        )

        // 6. Inline math: $`...`$ (GitHub style) -> $...$
        result = replacePattern(
            in: result,
            pattern: #"\$`([^`]+)`\$"#,
            replacement: { match in
                let trimmed = match.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { return "$``$" }
                return "$\(trimmed)$"
            }
        )

        // 7. Normalize markdown-escaped underscore to TeX subscript underscore inside
        // inline math. Model outputs often include `\_` to satisfy markdown renderers.
        result = normalizeInlineMathEscapedUnderscores(result)

        // 8. Escape markdown control chars inside inline math spans so markdown parsing
        // doesn't split runs before Textual's `.math` syntax extension executes.
        result = protectMarkdownInsideInlineMath(result)

        // 9. Leave existing $...$ unchanged.
        return result
    }

    /// Convert `<execute>…</execute>` blocks emitted by the CodeAct agent into
    /// fenced ` ```python ` code blocks so the markdown renderer displays them
    /// correctly — preventing raw tag display and `#` comment lines becoming headings.
    ///
    /// Handles multiline content and normalises whitespace around the open/close tags.
    /// Leaves content already inside an existing fenced code block untouched.
    private static func replaceExecuteTags(_ input: String) -> String {
        guard input.contains("<execute>") else { return input }
        return replacePattern(
            in: input,
            pattern: #"<execute>([\s\S]*?)</execute>"#,
            replacement: { inner in
                let code = inner.trimmingCharacters(in: .newlines)
                return "\n```python\n\(code)\n```\n"
            }
        )
    }

    /// De-indent likely-accidental indented math lines.
    /// Markdown treats 4+ leading spaces as code blocks, which disables math parsing.
    /// This pass is conservative and only touches lines that look like LaTeX math content.
    private static func deindentAccidentalMathCodeBlocks(_ content: String) -> String {
        var lines = content.components(separatedBy: "\n")
        var inFence = false
        var activeFence: String?

        for idx in lines.indices {
            let line = lines[idx]
            if let fence = markdownFenceMarker(in: line) {
                if inFence {
                    if fence == activeFence {
                        inFence = false
                        activeFence = nil
                    }
                } else {
                    inFence = true
                    activeFence = fence
                }
                continue
            }
            guard !inFence else { continue }

            let leadingSpaces = line.prefix { $0 == " " }.count
            guard leadingSpaces >= 4 else { continue }

            let dollarCount = countUnescapedDollars(in: line)
            let likelyTruncatedInline = dollarCount == 1 && hasSingleTrailingBackslash(line)
            let hasCompleteInlineMath = dollarCount >= 2
            guard likelyTruncatedInline || hasCompleteInlineMath else { continue }

            // Keep this strict to avoid altering legitimate code blocks.
            let hasLikelyLatexMarkers = line.contains("\\") || line.contains("_") || line.contains("^")
            guard hasLikelyLatexMarkers else { continue }

            // Reduce indentation to <= 3 spaces so markdown stops treating it as code.
            let spacesToRemove = leadingSpaces - 3
            lines[idx] = String(line.dropFirst(spacesToRemove))
        }
        return lines.joined(separator: "\n")
    }

    /// Heal truncated inline math when model output is cut mid-token
    /// (e.g. `$g_{tt} = -(1 + 2\` at end of a paragraph/line).
    /// Appends `$` only on candidate lines with odd unescaped `$` count and single trailing `\`.
    private static func healTruncatedMath(_ content: String) -> String {
        var lines = content.components(separatedBy: "\n")
        var inFence = false
        var activeFence: String?

        for idx in lines.indices {
            let line = lines[idx]
            if let fence = markdownFenceMarker(in: line) {
                if inFence {
                    if fence == activeFence {
                        inFence = false
                        activeFence = nil
                    }
                } else {
                    inFence = true
                    activeFence = fence
                }
                continue
            }
            guard !inFence else { continue }

            let (core, trailing) = splitTrailingSpacesAndTabs(in: line)
            guard hasSingleTrailingBackslash(core) else { continue }
            let dollarCount = countUnescapedDollars(in: core)
            guard dollarCount % 2 == 1 else { continue }
            lines[idx] = core + "$" + trailing
        }
        return lines.joined(separator: "\n")
    }

    private static func markdownFenceMarker(in line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("```") { return "```" }
        if trimmed.hasPrefix("~~~") { return "~~~" }
        return nil
    }

    private static func splitTrailingSpacesAndTabs(in line: String) -> (core: String, trailing: String) {
        var end = line.endIndex
        while end > line.startIndex {
            let prev = line.index(before: end)
            let ch = line[prev]
            guard ch == " " || ch == "\t" else { break }
            end = prev
        }
        return (String(line[..<end]), String(line[end...]))
    }

    private static func hasSingleTrailingBackslash(_ s: String) -> Bool {
        guard s.hasSuffix("\\") else { return false }
        return !s.hasSuffix("\\\\")
    }

    private static func countUnescapedDollars(in s: String) -> Int {
        var count = 0
        var i = s.startIndex
        while i < s.endIndex {
            if s[i] == "\\" {
                i = s.index(after: i)
                if i < s.endIndex { i = s.index(after: i) }
                continue
            }
            if s[i] == "$" {
                count += 1
            }
            i = s.index(after: i)
        }
        return count
    }

    private static func escapeLikelyCurrencyDollars(in input: String) -> String {
        replacePattern(
            in: input,
            pattern: #"(?<!\\)\$(\d[\d,]*(?:\.\d+)?)(?=(?:\s|$|[.,;:!?)]))"#,
            replacement: { amount in
                #"\$\#(amount)"#
            }
        )
    }

    /// Normalize block math content for renderers that lack full amsmath support.
    /// - Unwraps \boxed{...} to its inner content — SwiftUIMath/iosMath do not implement \boxed.
    /// - Strips \displaystyle / \textstyle which can cause parse failures.
    /// - Rewrites \accent{\mathbf{X}} → \mathbf{\accent{X}} for \hat, \vec to fix mid-expression parse bailout.
    /// - Replaces \Box (d'Alembertian) with □; SwiftUIMath/iosMath do not implement it.
    private static func normalizeBlockMathContent(_ latex: String) -> String {
        var result = latex
            .replacingOccurrences(of: "\\displaystyle", with: "")
            .replacingOccurrences(of: "\\textstyle", with: "")
            .replacingOccurrences(of: "\\Box", with: "\u{25A1}")  // d'Alembertian
        result = rewriteAccentOverBold(result, accent: "hat")
        result = rewriteAccentOverBold(result, accent: "vec")
        result = stripBraceAnnotation(result, command: "underbrace", trailingMarker: "_")
        result = stripBraceAnnotation(result, command: "overbrace", trailingMarker: "^")
        var searchStart = result.startIndex
        while let cmdRange = result.range(of: "\\boxed", options: .literal, range: searchStart..<result.endIndex) {
            var pos = cmdRange.upperBound
            while pos < result.endIndex && result[pos] == " " {
                pos = result.index(after: pos)
            }
            guard pos < result.endIndex, result[pos] == "{",
                  let (content, afterBrace) = findBraceGroup(in: result, at: pos) else {
                searchStart = cmdRange.upperBound
                continue
            }
            let charOffset = result.distance(from: result.startIndex, to: cmdRange.lowerBound)
            result.replaceSubrange(cmdRange.lowerBound..<afterBrace, with: content)
            searchStart = result.index(result.startIndex, offsetBy: charOffset)
        }
        return result
    }

    // MARK: - Inline LaTeX Simplification

    /// Simplify common LaTeX commands to Unicode for readable inline display.
    /// Retained for compatibility and unit-test coverage.
    public static func simplifyForInline(_ latex: String) -> String {
        var s = latex

        // 1. Structural commands (process before symbol replacement)
        s = replaceFractions(in: s)
        s = replaceSqrt(in: s)

        // Text commands (longer variants first to avoid prefix collision)
        for cmd in ["\\textbf", "\\textit", "\\textrm", "\\texttt", "\\text"] {
            s = replaceCommandWithOneArg(cmd, in: s) { $0 }
        }

        // Style wrappers (strip command, keep content)
        for cmd in ["\\boldsymbol", "\\operatorname", "\\mathcal", "\\mathsf",
                     "\\mathit", "\\mathbf", "\\mathrm"] {
            s = replaceCommandWithOneArg(cmd, in: s) { $0 }
        }

        // Color commands: \color{red}{text} -> text
        for cmd in ["\\textcolor", "\\color"] {
            s = replaceCommandWithTwoArgs(cmd, in: s) { _, content in content }
        }

        // Accent commands with Unicode combining characters for single chars
        for (cmd, combining) in accentCombining {
            s = replaceCommandWithOneArg(cmd, in: s) { content in
                content.count == 1 ? "\(content)\(combining)" : content
            }
        }

        // Strip remaining accent/decoration commands
        for cmd in ["\\overline", "\\underline", "\\overbrace", "\\underbrace",
                     "\\overrightarrow", "\\overleftarrow", "\\widehat", "\\widetilde"] {
            s = replaceCommandWithOneArg(cmd, in: s) { $0 }
        }

        // 2. Symbol replacements via regex (longest-first, with word boundary)
        for (regex, template) in Self.symbolRegexPairs {
            s = regex.stringByReplacingMatches(
                in: s, range: NSRange(s.startIndex..., in: s), withTemplate: template
            )
        }

        // 3. Function names: \lim -> lim, \sin -> sin, etc.
        for (regex, name) in Self.functionRegexPairs {
            s = regex.stringByReplacingMatches(
                in: s, range: NSRange(s.startIndex..., in: s), withTemplate: name
            )
        }

        // 4. Delimiter sizing commands: \left, \right, \big, etc.
        for regex in Self.delimiterRegexes {
            s = regex.stringByReplacingMatches(
                in: s, range: NSRange(s.startIndex..., in: s), withTemplate: ""
            )
        }

        // 5. Spacing and style commands
        s = s.replacingOccurrences(of: "\\,", with: " ")
        s = s.replacingOccurrences(of: "\\;", with: " ")
        s = s.replacingOccurrences(of: "\\:", with: " ")
        s = s.replacingOccurrences(of: "\\!", with: "")
        s = s.replacingOccurrences(of: "\\quad", with: "  ")
        s = s.replacingOccurrences(of: "\\qquad", with: "   ")
        s = s.replacingOccurrences(of: "\\ ", with: " ")
        s = s.replacingOccurrences(of: "\\displaystyle", with: "")
        s = s.replacingOccurrences(of: "\\textstyle", with: "")
        s = s.replacingOccurrences(of: "\\limits", with: "")
        s = s.replacingOccurrences(of: "\\nolimits", with: "")

        // 6. Strip single-level braces: {x} -> x (repeat until stable)
        var prev = ""
        while prev != s {
            prev = s
            s = replacePattern(in: s, pattern: #"\{([^{}]*)\}"#, replacement: { $0 })
        }

        return s
    }

    // MARK: - Math span protection

    private static func normalizeInlineMathEscapedUnderscores(_ input: String) -> String {
        replacePattern(
            in: input,
            pattern: #"(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)"#,
            replacement: { inner in
                "$\(normalizeInlineMathContent(inner))$"
            }
        )
    }

    /// Normalize inline math content: fix common issues that cause renderers to fail and show raw LaTeX.
    /// - Fixes markdown-escaped subscripts (`\_` → `_`).
    /// - Strips `\displaystyle` and `\textstyle`; unsupported in many inline renderers and cause fallback to raw display.
    /// - Rewrites `\accent{\mathbf{X}}` → `\mathbf{\accent{X}}` for \hat, \vec; SwiftUIMath fails on outer-accent form.
    /// - Replaces `\Box` (d'Alembertian) with □; SwiftUIMath/iosMath do not implement it.
    private static func normalizeInlineMathContent(_ latex: String) -> String {
        var s = latex.replacingOccurrences(of: "\\_", with: "_")
        s = s.replacingOccurrences(of: "\\displaystyle", with: "")
        s = s.replacingOccurrences(of: "\\textstyle", with: "")
        s = s.replacingOccurrences(of: "\\Box", with: "\u{25A1}")  // d'Alembertian
        s = rewriteAccentOverBold(s, accent: "hat")
        s = rewriteAccentOverBold(s, accent: "vec")
        s = stripBraceAnnotation(s, command: "underbrace", trailingMarker: "_")
        s = stripBraceAnnotation(s, command: "overbrace", trailingMarker: "^")
        return s
    }

    private static func protectMarkdownInsideInlineMath(_ input: String) -> String {
        replacePattern(
            in: input,
            pattern: #"(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)"#,
            replacement: { inner in
                "$\(escapeMarkdownControlCharacters(in: inner))$"
            }
        )
    }

    private static func escapeMarkdownControlCharacters(in text: String) -> String {
        var escaped = ""
        escaped.reserveCapacity(text.count)

        // `_` is intentionally excluded to preserve TeX subscript semantics.
        let controlChars: Set<Character> = ["*", "[", "]", "`"]
        var previous: Character?
        for ch in text {
            if controlChars.contains(ch), previous != "\\" {
                escaped.append("\\")
            }
            escaped.append(ch)
            previous = ch
        }
        return escaped
    }

    // MARK: - Structural Command Helpers

    /// Replace \frac{a}{b}, \dfrac, \tfrac, \cfrac → a/b (with parens for complex content)
    private static func replaceFractions(in s: String) -> String {
        var result = s
        // Process longer variants first to avoid prefix collision
        for cmd in ["\\cfrac", "\\dfrac", "\\tfrac", "\\frac"] {
            result = replaceCommandWithTwoArgs(cmd, in: result) { num, den in
                let n = num.count > 1 ? "(\(num))" : num
                let d = den.count > 1 ? "(\(den))" : den
                return "\(n)/\(d)"
            }
        }
        return result
    }

    /// Replace \sqrt{x} → √(x) or √x for single chars
    private static func replaceSqrt(in s: String) -> String {
        replaceCommandWithOneArg("\\sqrt", in: s) { content in
            content.count == 1 ? "√\(content)" : "√(\(content))"
        }
    }

    /// Replace a LaTeX command that takes one brace-delimited argument.
    /// Properly skips occurrences that don't have a following brace group
    /// (e.g. \text inside \textbf won't be matched).
    private static func replaceCommandWithOneArg(
        _ cmd: String, in s: String, transform: (String) -> String
    ) -> String {
        var result = s
        var searchStart = result.startIndex
        while let cmdRange = result.range(of: cmd, options: .literal, range: searchStart..<result.endIndex) {
            var pos = cmdRange.upperBound
            while pos < result.endIndex && result[pos] == " " {
                pos = result.index(after: pos)
            }
            guard pos < result.endIndex, result[pos] == "{",
                  let (content, afterBrace) = findBraceGroup(in: result, at: pos) else {
                searchStart = cmdRange.upperBound
                continue
            }
            let replacement = transform(content)
            // Save character offset before mutation — String indices are
            // invalidated by replaceSubrange and must not be reused.
            let charOffset = result.distance(from: result.startIndex, to: cmdRange.lowerBound)
            result.replaceSubrange(cmdRange.lowerBound..<afterBrace, with: replacement)
            // Re-scan from replacement start so nested commands are processed
            searchStart = result.index(result.startIndex, offsetBy: charOffset)
        }
        return result
    }

    /// Replace a LaTeX command that takes two brace-delimited arguments.
    private static func replaceCommandWithTwoArgs(
        _ cmd: String, in s: String, transform: (String, String) -> String
    ) -> String {
        var result = s
        var searchStart = result.startIndex
        while let cmdRange = result.range(of: cmd, options: .literal, range: searchStart..<result.endIndex) {
            var pos = cmdRange.upperBound
            while pos < result.endIndex && result[pos] == " " {
                pos = result.index(after: pos)
            }
            guard pos < result.endIndex, result[pos] == "{",
                  let (arg1, afterArg1) = findBraceGroup(in: result, at: pos) else {
                searchStart = cmdRange.upperBound
                continue
            }
            var pos2 = afterArg1
            while pos2 < result.endIndex && result[pos2] == " " {
                pos2 = result.index(after: pos2)
            }
            guard pos2 < result.endIndex, result[pos2] == "{",
                  let (arg2, afterArg2) = findBraceGroup(in: result, at: pos2) else {
                searchStart = cmdRange.upperBound
                continue
            }
            let replacement = transform(arg1, arg2)
            // Save character offset before mutation — String indices are
            // invalidated by replaceSubrange and must not be reused.
            let charOffset = result.distance(from: result.startIndex, to: cmdRange.lowerBound)
            result.replaceSubrange(cmdRange.lowerBound..<afterArg2, with: replacement)
            // Re-scan from replacement start so nested commands are processed
            searchStart = result.index(result.startIndex, offsetBy: charOffset)
        }
        return result
    }

    /// Strip `\underbrace{content}_{label}` → `content` (or `\overbrace{content}^{label}`).
    /// SwiftUIMath does not support these amsmath annotations; leaving them causes the entire
    /// math expression to fail rendering (blank box).  The trailing label (`_{...}` / `^{...}`)
    /// is consumed when present so it doesn't turn into a spurious subscript/superscript.
    private static func stripBraceAnnotation(_ latex: String, command: String, trailingMarker: String) -> String {
        let marker = "\\\(command)"
        var result = latex
        var searchStart = result.startIndex
        while let cmdRange = result.range(of: marker, options: .literal, range: searchStart..<result.endIndex) {
            // Ensure full command match (not a prefix of a longer command)
            if cmdRange.upperBound < result.endIndex {
                let next = result[cmdRange.upperBound]
                if next.isLetter { searchStart = cmdRange.upperBound; continue }
            }
            // Find the brace group: \underbrace{...}
            var pos = cmdRange.upperBound
            while pos < result.endIndex && result[pos] == " " { pos = result.index(after: pos) }
            guard pos < result.endIndex, result[pos] == "{",
                  let (content, afterBrace) = findBraceGroup(in: result, at: pos) else {
                searchStart = cmdRange.upperBound
                continue
            }
            // Optionally consume trailing _{...} or ^{...} label
            var endPos = afterBrace
            if endPos < result.endIndex && String(result[endPos]) == trailingMarker {
                let afterMarker = result.index(after: endPos)
                if afterMarker < result.endIndex, result[afterMarker] == "{",
                   let (_, afterLabel) = findBraceGroup(in: result, at: afterMarker) {
                    endPos = afterLabel
                }
            }
            let charOffset = result.distance(from: result.startIndex, to: cmdRange.lowerBound)
            result.replaceSubrange(cmdRange.lowerBound..<endPos, with: content)
            searchStart = result.index(result.startIndex, offsetBy: charOffset)
        }
        return result
    }

    /// Rewrite \accent{\mathbf{X}} → \mathbf{\accent{X}}. SwiftUIMath parses the outer-accent form
    /// incorrectly and treats everything after as raw text (visible \mathbf{r^}, \mathbf{g}, etc.).
    private static func rewriteAccentOverBold(_ latex: String, accent: String) -> String {
        let marker = "\\\(accent){"
        var result = latex
        var searchStart = result.startIndex
        while let headRange = result.range(of: marker, options: .literal, range: searchStart..<result.endIndex) {
            let braceIndex = result.index(before: headRange.upperBound)
            guard result[braceIndex] == "{",
                  let (content, afterBrace) = findBraceGroup(in: result, at: braceIndex),
                  content.hasPrefix("\\mathbf{"),
                  let bfRange = content.range(of: "\\mathbf{", options: .literal),
                  bfRange.lowerBound == content.startIndex,
                  let (inner, _) = findBraceGroup(in: content, at: content.index(before: bfRange.upperBound)) else {
                searchStart = headRange.upperBound
                continue
            }
            let replacement = "\\mathbf{\\\(accent){\(inner)}}"
            let charOffset = result.distance(from: result.startIndex, to: headRange.lowerBound)
            result.replaceSubrange(headRange.lowerBound..<afterBrace, with: replacement)
            searchStart = result.index(result.startIndex, offsetBy: charOffset)
        }
        return result
    }

    /// Find matching closing brace for an opening brace at the given index.
    /// Handles nesting and escaped braces (\{ \}).
    private static func findBraceGroup(
        in s: String, at index: String.Index
    ) -> (content: String, end: String.Index)? {
        guard index < s.endIndex, s[index] == "{" else { return nil }
        let start = s.index(after: index)
        var depth = 1
        var current = start
        while current < s.endIndex {
            let ch = s[current]
            if ch == "\\" {
                // Skip escaped character (\{ \} \\ etc.)
                let next = s.index(after: current)
                if next < s.endIndex { current = next }
            } else if ch == "{" {
                depth += 1
            } else if ch == "}" {
                depth -= 1
                if depth == 0 {
                    return (String(s[start..<current]), s.index(after: current))
                }
            }
            current = s.index(after: current)
        }
        return nil
    }

    // MARK: - Accent Combining Characters

    private static let accentCombining: [(String, String)] = [
        ("\\bar", "\u{0304}"),     // combining macron
        ("\\hat", "\u{0302}"),     // combining circumflex
        ("\\tilde", "\u{0303}"),   // combining tilde
        ("\\vec", "\u{20D7}"),     // combining right arrow
        ("\\dot", "\u{0307}"),     // combining dot above
        ("\\ddot", "\u{0308}"),    // combining diaeresis
    ]

    // MARK: - Cached Regex Tables

    /// Symbol replacements sorted by descending key length, compiled to regexes
    /// with a non-letter lookahead to prevent partial command matches.
    private static let symbolRegexPairs: [(NSRegularExpression, String)] = {
        let replacements: [(String, String)] = [
            // Relations
            ("\\approx", "≈"), ("\\equiv", "≡"), ("\\propto", "∝"),
            ("\\neq", "≠"), ("\\leq", "≤"), ("\\geq", "≥"),
            ("\\sim", "∼"), ("\\ll", "≪"), ("\\gg", "≫"),
            // Arrows (longest first)
            ("\\leftrightarrow", "↔"), ("\\Leftrightarrow", "⇔"),
            ("\\rightarrow", "→"), ("\\Rightarrow", "⇒"),
            ("\\leftarrow", "←"), ("\\Leftarrow", "⇐"),
            ("\\mapsto", "↦"), ("\\to", "→"),
            // Binary operators
            ("\\times", "×"), ("\\cdot", "·"), ("\\div", "÷"),
            ("\\pm", "±"), ("\\mp", "∓"),
            ("\\circ", "∘"), ("\\bullet", "•"),
            // Set / logic (longest first)
            ("\\varnothing", "∅"), ("\\emptyset", "∅"),
            ("\\subseteq", "⊆"), ("\\supseteq", "⊇"),
            ("\\setminus", "∖"),
            ("\\subset", "⊂"), ("\\supset", "⊃"),
            ("\\notin", "∉"), ("\\in", "∈"),
            ("\\cup", "∪"), ("\\cap", "∩"),
            ("\\forall", "∀"), ("\\exists", "∃"), ("\\neg", "¬"),
            ("\\land", "∧"), ("\\lor", "∨"),
            // Big operators (inline)
            ("\\iiint", "∭"), ("\\iint", "∬"), ("\\int", "∫"),
            ("\\sum", "∑"), ("\\prod", "∏"),
            // Misc (longest first)
            ("\\partial", "∂"), ("\\nabla", "∇"),
            ("\\infty", "∞"), ("\\prime", "′"),
            ("\\ldots", "…"), ("\\cdots", "⋯"), ("\\dots", "…"),
            ("\\star", "⋆"), ("\\dagger", "†"),
            // Greek lowercase (var- variants first)
            ("\\varepsilon", "ε"), ("\\vartheta", "ϑ"),
            ("\\varsigma", "ς"), ("\\varphi", "φ"),
            ("\\alpha", "α"), ("\\beta", "β"), ("\\gamma", "γ"), ("\\delta", "δ"),
            ("\\epsilon", "ε"), ("\\zeta", "ζ"), ("\\eta", "η"), ("\\theta", "θ"),
            ("\\iota", "ι"), ("\\kappa", "κ"), ("\\lambda", "λ"), ("\\mu", "μ"),
            ("\\nu", "ν"), ("\\xi", "ξ"), ("\\pi", "π"),
            ("\\rho", "ρ"), ("\\sigma", "σ"), ("\\tau", "τ"),
            ("\\upsilon", "υ"), ("\\phi", "φ"), ("\\chi", "χ"),
            ("\\psi", "ψ"), ("\\omega", "ω"),
            // Greek uppercase
            ("\\Gamma", "Γ"), ("\\Delta", "Δ"), ("\\Theta", "Θ"),
            ("\\Lambda", "Λ"), ("\\Xi", "Ξ"), ("\\Pi", "Π"),
            ("\\Sigma", "Σ"), ("\\Phi", "Φ"), ("\\Psi", "Ψ"), ("\\Omega", "Ω"),
        ]
        // Sort by descending key length to prevent partial matches
        let sorted = replacements.sorted { $0.0.count > $1.0.count }
        return sorted.compactMap { (cmd, unicode) in
            let escaped = NSRegularExpression.escapedPattern(for: cmd)
            guard let regex = try? NSRegularExpression(
                pattern: escaped + "(?![a-zA-Z])"
            ) else { return nil }
            return (regex, NSRegularExpression.escapedTemplate(for: unicode))
        }
    }()

    /// Function name regexes with non-letter lookahead.
    private static let functionRegexPairs: [(NSRegularExpression, String)] = {
        let names = [
            "arcsin", "arccos", "arctan",
            "sinh", "cosh", "tanh",
            "sin", "cos", "tan", "cot", "sec", "csc",
            "log", "exp", "lim", "sup", "inf",
            "max", "min", "det", "dim", "ker", "deg", "gcd",
            "arg", "hom", "ln",
        ]
        // Sort by descending length
        let sorted = names.sorted { $0.count > $1.count }
        return sorted.compactMap { fn in
            guard let regex = try? NSRegularExpression(
                pattern: "\\\\\(fn)(?![a-zA-Z])"
            ) else { return nil }
            return (regex, fn)
        }
    }()

    /// Delimiter sizing commands to strip: \left, \right, \big, \Big, \bigg, \Bigg
    private static let delimiterRegexes: [NSRegularExpression] = {
        let commands = ["\\left", "\\right", "\\bigg", "\\Bigg", "\\big", "\\Big"]
        let sorted = commands.sorted { $0.count > $1.count }
        return sorted.compactMap { cmd in
            let escaped = NSRegularExpression.escapedPattern(for: cmd)
            return try? NSRegularExpression(pattern: escaped + "(?![a-zA-Z])")
        }
    }()

    // MARK: - Internal Regex Helpers

    static func replacePattern(
        in input: String,
        pattern: String,
        replacement: (String) -> String
    ) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return input
        }
        let nsInput = input as NSString
        let matches = regex.matches(in: input, range: NSRange(location: 0, length: nsInput.length))

        guard !matches.isEmpty else { return input }

        var result = ""
        var lastEnd = 0

        for match in matches {
            // Append text before this match
            let beforeRange = NSRange(location: lastEnd, length: match.range.location - lastEnd)
            result += nsInput.substring(with: beforeRange)

            // Get the capture group (group 1)
            if match.numberOfRanges > 1 {
                let captureRange = match.range(at: 1)
                if captureRange.location != NSNotFound {
                    let captured = nsInput.substring(with: captureRange)
                    result += replacement(captured)
                } else {
                    // No capture group match — emit the full match unchanged
                    result += nsInput.substring(with: match.range)
                }
            } else {
                result += nsInput.substring(with: match.range)
            }

            lastEnd = match.range.location + match.range.length
        }

        // Append remaining text
        if lastEnd < nsInput.length {
            result += nsInput.substring(from: lastEnd)
        }

        return result
    }
}
