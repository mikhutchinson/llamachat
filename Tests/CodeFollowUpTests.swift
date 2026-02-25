import XCTest
@testable import LlamaChatUI
@testable import LlamaInferenceCore

// MARK: - CodeAnalysis Tests

final class CodeAnalysisTests: XCTestCase {

    func testParseValidJSON() {
        let json = """
        {"functions":["foo","bar"],"classes":["MyClass"],"imports":["os","sys"],"issues":["bare except clause (line 5)"],"line_count":20}
        """
        let analysis = CodeAnalysis.from(json: json)
        XCTAssertNotNil(analysis)
        XCTAssertEqual(analysis?.functions, ["foo", "bar"])
        XCTAssertEqual(analysis?.classes, ["MyClass"])
        XCTAssertEqual(analysis?.imports, ["os", "sys"])
        XCTAssertEqual(analysis?.issues, ["bare except clause (line 5)"])
        XCTAssertEqual(analysis?.lineCount, 20)
    }

    func testParseEmptyAnalysis() {
        let json = """
        {"functions":[],"classes":[],"imports":[],"issues":[],"line_count":1}
        """
        let analysis = CodeAnalysis.from(json: json)
        XCTAssertNotNil(analysis)
        XCTAssertTrue(analysis!.functions.isEmpty)
        XCTAssertTrue(analysis!.classes.isEmpty)
        XCTAssertTrue(analysis!.imports.isEmpty)
        XCTAssertTrue(analysis!.issues.isEmpty)
        XCTAssertEqual(analysis!.lineCount, 1)
    }

    func testParseMalformedJSON() {
        let analysis = CodeAnalysis.from(json: "not json at all")
        XCTAssertNil(analysis)
    }

    func testParsePartialJSON() {
        let json = """
        {"functions":["x"],"line_count":5}
        """
        let analysis = CodeAnalysis.from(json: json)
        XCTAssertNotNil(analysis)
        XCTAssertEqual(analysis?.functions, ["x"])
        XCTAssertTrue(analysis!.classes.isEmpty)
        XCTAssertTrue(analysis!.imports.isEmpty)
        XCTAssertTrue(analysis!.issues.isEmpty)
        XCTAssertEqual(analysis!.lineCount, 5)
    }

    func testPromptSummary() {
        let analysis = CodeAnalysis(
            functions: ["process", "validate"],
            classes: ["Handler"],
            imports: ["numpy", "os"],
            issues: ["bare except clause (line 14)", "mutable default arg in process() (line 3)"],
            lineCount: 45
        )
        let summary = analysis.promptSummary
        XCTAssertTrue(summary.contains("Functions: process, validate"))
        XCTAssertTrue(summary.contains("Classes: Handler"))
        XCTAssertTrue(summary.contains("Imports: numpy, os"))
        XCTAssertTrue(summary.contains("bare except clause"))
        XCTAssertTrue(summary.contains("mutable default arg"))
        XCTAssertTrue(summary.contains("Lines: 45"))
    }

    func testPromptSummaryMinimal() {
        let analysis = CodeAnalysis(
            functions: [], classes: [], imports: [], issues: [], lineCount: 3
        )
        let summary = analysis.promptSummary
        XCTAssertEqual(summary, "- Lines: 3")
    }
}

// MARK: - Code Follow-Up Action Prompt Tests

@MainActor
final class CodeBlockActionTests: XCTestCase {

    func testExplainPrompt() {
        let vm = ChatViewModel()
        let prompt = vm.composeCodeActionPrompt(
            action: .explain,
            code: "print('hello')",
            language: "python"
        )
        XCTAssertTrue(prompt.contains("Explain the following code"))
        XCTAssertTrue(prompt.contains("```python"))
        XCTAssertTrue(prompt.contains("print('hello')"))
    }

    func testReviewPromptWithoutAnalysis() {
        let vm = ChatViewModel()
        let prompt = vm.composeCodeActionPrompt(
            action: .review,
            code: "x = 1",
            language: "swift"
        )
        XCTAssertTrue(prompt.contains("Review and critique"))
        XCTAssertTrue(prompt.contains("```swift"))
        XCTAssertFalse(prompt.contains("Static analysis"))
    }

    func testReviewPromptWithAnalysis() {
        let vm = ChatViewModel()
        let analysis = CodeAnalysis(
            functions: ["foo"],
            classes: [],
            imports: ["os"],
            issues: ["bare except clause (line 3)"],
            lineCount: 10
        )
        let prompt = vm.composeCodeActionPrompt(
            action: .review,
            code: "def foo():\n    try:\n        pass\n    except:\n        pass",
            language: "python",
            analysis: analysis
        )
        XCTAssertTrue(prompt.contains("Static analysis found:"))
        XCTAssertTrue(prompt.contains("Functions: foo"))
        XCTAssertTrue(prompt.contains("bare except clause"))
        XCTAssertTrue(prompt.contains("```python"))
    }

    func testImprovePrompt() {
        let vm = ChatViewModel()
        let prompt = vm.composeCodeActionPrompt(
            action: .improve,
            code: "for i in range(10): print(i)",
            language: "python"
        )
        XCTAssertTrue(prompt.contains("Improve the following code"))
        XCTAssertTrue(prompt.contains("```python"))
    }

    func testWriteTestsPrompt() {
        let vm = ChatViewModel()
        let prompt = vm.composeCodeActionPrompt(
            action: .writeTests,
            code: "def add(a, b): return a + b",
            language: "python"
        )
        XCTAssertTrue(prompt.contains("Write comprehensive unit tests"))
        XCTAssertTrue(prompt.contains("pytest"))
        XCTAssertTrue(prompt.contains("```python"))
    }

    func testTranslateToPythonPrompt() {
        let vm = ChatViewModel()
        let prompt = vm.composeCodeActionPrompt(
            action: .translateToPython,
            code: "let x = 42",
            language: "swift"
        )
        XCTAssertTrue(prompt.contains("Translate the following swift code"))
        XCTAssertTrue(prompt.contains("```swift"))
    }

    func testNilLanguageFallback() {
        let vm = ChatViewModel()
        let prompt = vm.composeCodeActionPrompt(
            action: .explain,
            code: "x = 1",
            language: nil
        )
        XCTAssertTrue(prompt.contains("```code"))
    }
}

// MARK: - CodeFollowUpAction Tests

final class CodeFollowUpActionEnumTests: XCTestCase {

    func testAllActionsHaveLabels() {
        for action in CodeFollowUpAction.allCases {
            XCTAssertFalse(action.label.isEmpty, "\(action) has empty label")
            XCTAssertFalse(action.systemImage.isEmpty, "\(action) has empty systemImage")
        }
    }

    func testActionCount() {
        XCTAssertEqual(CodeFollowUpAction.allCases.count, 5)
    }
}

// MARK: - Quote Reply Tests

final class QuoteReplyTests: XCTestCase {

    func testQuoteFormatsAsBlockquote() {
        let text = "Hello world\nSecond line"
        let quoted = formatQuote(text)
        XCTAssertEqual(quoted, "> Hello world\n> Second line\n\n")
    }

    func testQuoteSingleLine() {
        let quoted = formatQuote("Single line")
        XCTAssertEqual(quoted, "> Single line\n\n")
    }

    func testQuoteTruncation() {
        let longText = String(repeating: "a", count: 3000)
        let quoted = formatQuote(longText)
        XCTAssertTrue(quoted.count < 3000)
        XCTAssertTrue(quoted.contains("\u{2026}"))
    }

    func testQuoteEmptyLines() {
        let text = "Line 1\n\nLine 3"
        let quoted = formatQuote(text)
        XCTAssertEqual(quoted, "> Line 1\n> \n> Line 3\n\n")
    }

    /// Helper that mirrors the quote formatting logic from MessageRow.
    private func formatQuote(_ text: String) -> String {
        var textToQuote = text
        let maxQuoteLength = 2000
        if textToQuote.count > maxQuoteLength {
            textToQuote = String(textToQuote.prefix(maxQuoteLength)) + "\u{2026}"
        }
        let quoted = textToQuote
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { "> \($0)" }
            .joined(separator: "\n")
        return quoted + "\n\n"
    }
}
