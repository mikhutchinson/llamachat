import XCTest
@testable import LlamaInferenceCore

final class PythonSandboxTests: XCTestCase {

    // MARK: - RunOutput JSON Decoding (via PythonSandbox.decodeJSON)

    func testDecodeJSON_validComplete() {
        let json = """
        {"out":"hello\\n","err":"","figs":[],"exc":null}
        """
        let result = PythonSandbox.decodeJSON(json, elapsedMs: 42)
        XCTAssertEqual(result.stdout, "hello\n")
        XCTAssertTrue(result.stderr.isEmpty)
        XCTAssertNil(result.error)
        XCTAssertTrue(result.figures.isEmpty)
        XCTAssertEqual(result.elapsedMs, 42)
    }

    func testDecodeJSON_withError() {
        let json = """
        {"out":"","err":"","figs":[],"exc":"Traceback (most recent call last):\\n  NameError: name 'x' is not defined\\n"}
        """
        let result = PythonSandbox.decodeJSON(json, elapsedMs: 10)
        XCTAssertTrue(result.stdout.isEmpty)
        XCTAssertNotNil(result.error)
        XCTAssertTrue(result.error!.contains("NameError"))
    }

    func testDecodeJSON_withStderr() {
        let json = """
        {"out":"done","err":"warning: deprecated","figs":[],"exc":null}
        """
        let result = PythonSandbox.decodeJSON(json, elapsedMs: 5)
        XCTAssertEqual(result.stdout, "done")
        XCTAssertEqual(result.stderr, "warning: deprecated")
        XCTAssertNil(result.error)
    }

    func testDecodeJSON_withFigureData() {
        // Small 1x1 red PNG in base64
        let pngB64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        let json = """
        {"out":"","err":"","figs":["\(pngB64)"],"exc":null}
        """
        let result = PythonSandbox.decodeJSON(json, elapsedMs: 100)
        XCTAssertEqual(result.figures.count, 1)
        XCTAssertFalse(result.figures[0].isEmpty)
    }

    func testDecodeJSON_invalidJSON() {
        let result = PythonSandbox.decodeJSON("not json", elapsedMs: 1)
        XCTAssertNotNil(result.error)
        XCTAssertTrue(result.error!.contains("Failed to decode"))
    }

    func testDecodeJSON_emptyString() {
        let result = PythonSandbox.decodeJSON("", elapsedMs: 0)
        XCTAssertNotNil(result.error)
    }

    func testDecodeJSON_missingFields() {
        // Only "out" present — missing "err", "figs", "exc"
        let json = """
        {"out":"hello"}
        """
        let result = PythonSandbox.decodeJSON(json, elapsedMs: 7)
        XCTAssertEqual(result.stdout, "hello")
        XCTAssertTrue(result.stderr.isEmpty)
        XCTAssertNil(result.error)
        XCTAssertTrue(result.figures.isEmpty)
    }

    // MARK: - RunOutput Model

    func testRunOutput_initAndFields() {
        let output = RunOutput(
            stdout: "abc",
            stderr: "def",
            figures: [Data([0xFF])],
            error: "boom",
            elapsedMs: 123
        )
        XCTAssertEqual(output.stdout, "abc")
        XCTAssertEqual(output.stderr, "def")
        XCTAssertEqual(output.figures.count, 1)
        XCTAssertEqual(output.error, "boom")
        XCTAssertEqual(output.elapsedMs, 123)
    }

    // MARK: - Fenced Block Extraction

    func testExtract_singlePythonBlock() {
        let md = """
        Some text

        ```python
        print("hello")
        ```

        More text
        """
        let blocks = FencedBlockParser.extract(from: md)
        XCTAssertEqual(blocks.count, 1)
        XCTAssertEqual(blocks[0].language, "python")
        XCTAssertEqual(blocks[0].code, "print(\"hello\")")
    }

    func testExtract_multipleMixedBlocks() {
        let md = """
        Start

        ```python
        x = 1
        ```

        Middle

        ```javascript
        let y = 2;
        ```

        ```python
        z = x + 1
        ```

        End
        """
        let blocks = FencedBlockParser.extract(from: md)
        XCTAssertEqual(blocks.count, 3)
        XCTAssertEqual(blocks[0].language, "python")
        XCTAssertEqual(blocks[0].code, "x = 1")
        XCTAssertEqual(blocks[1].language, "javascript")
        XCTAssertEqual(blocks[1].code, "let y = 2;")
        XCTAssertEqual(blocks[2].language, "python")
        XCTAssertEqual(blocks[2].code, "z = x + 1")
    }

    func testExtract_noLanguageHint() {
        let md = """
        Text

        ```
        raw code
        ```
        """
        let blocks = FencedBlockParser.extract(from: md)
        XCTAssertEqual(blocks.count, 1)
        XCTAssertNil(blocks[0].language)
        XCTAssertEqual(blocks[0].code, "raw code")
    }

    func testExtract_tildeFences() {
        let md = """
        ~~~python
        x = 42
        ~~~
        """
        let blocks = FencedBlockParser.extract(from: md)
        XCTAssertEqual(blocks.count, 1)
        XCTAssertEqual(blocks[0].language, "python")
        XCTAssertEqual(blocks[0].code, "x = 42")
    }

    func testExtract_multilineCode() {
        let md = """
        ```python
        import os
        for f in os.listdir('.'):
            print(f)
        ```
        """
        let blocks = FencedBlockParser.extract(from: md)
        XCTAssertEqual(blocks.count, 1)
        XCTAssertTrue(blocks[0].code.contains("import os"))
        XCTAssertTrue(blocks[0].code.contains("print(f)"))
    }

    func testExtract_noCodeBlocks() {
        let md = "Just plain text with no code blocks."
        let blocks = FencedBlockParser.extract(from: md)
        XCTAssertTrue(blocks.isEmpty)
    }

    func testExtract_emptyMarkdown() {
        let blocks = FencedBlockParser.extract(from: "")
        XCTAssertTrue(blocks.isEmpty)
    }

    func testExtract_skipsMathBlocks() {
        let md = """
        Some text

        ```math
        E = mc^2
        ```

        ```python
        print("hello")
        ```

        ```math
        a^2 + b^2 = c^2
        ```
        """
        let blocks = FencedBlockParser.extract(from: md)
        // math blocks must be excluded — only the python block should appear
        XCTAssertEqual(blocks.count, 1)
        XCTAssertEqual(blocks[0].language, "python")
        XCTAssertEqual(blocks[0].code, "print(\"hello\")")
    }
}
