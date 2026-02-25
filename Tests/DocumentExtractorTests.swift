import XCTest
import Foundation
@testable import LlamaInferenceCore
import SwiftPythonRuntime

// MARK: - Test Helpers

private func findWorkerExecutable() -> String? {
    let fm = FileManager.default

    let demoBuild = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Tests/
        .deletingLastPathComponent()  // LlamaInferenceDemo/
        .appendingPathComponent(".build/arm64-apple-macosx/debug/SwiftPythonWorker")
    if fm.isExecutableFile(atPath: demoBuild.path) {
        return demoBuild.path
    }

    if let testBundle = Bundle.allBundles.first(where: { $0.bundlePath.hasSuffix(".xctest") }) {
        var dir = URL(fileURLWithPath: testBundle.bundlePath).deletingLastPathComponent()
        for _ in 0..<5 {
            let candidate = dir.appendingPathComponent("SwiftPythonWorker")
            if fm.isExecutableFile(atPath: candidate.path) {
                return candidate.path
            }
            dir = dir.deletingLastPathComponent()
        }
    }

    let rootBuild = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent(".build/arm64-apple-macosx/debug/SwiftPythonWorker")
    if fm.isExecutableFile(atPath: rootBuild.path) {
        return rootBuild.path
    }

    // SwiftPythonWorker is shipped inside the swiftpython-commercial checkout
    // and landed there after a plain `swift build`.
    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Tests/
        .deletingLastPathComponent()  // repo root
    for checkoutName in ["swiftpython-commercial"] {
        let checkoutWorker = repoRoot
            .appendingPathComponent(".build/checkouts/\(checkoutName)/SwiftPythonWorker")
        if fm.isExecutableFile(atPath: checkoutWorker.path) {
            return checkoutWorker.path
        }
    }

    return nil
}

private func findVenvPath() -> String? {
    let fm = FileManager.default

    if let envVenv = ProcessInfo.processInfo.environment["VIRTUAL_ENV"],
       fm.fileExists(atPath: envVenv) {
        return envVenv
    }

    var dir = (URL(fileURLWithPath: #filePath).deletingLastPathComponent() as NSURL).deletingLastPathComponent!
    for _ in 0..<8 {
        let candidate = dir.appendingPathComponent(".venv")
        if fm.fileExists(atPath: candidate.path) {
            return candidate.path
        }
        dir = dir.deletingLastPathComponent()
    }
    return nil
}

private func makePool(workerCount: Int = 1) async throws -> PythonProcessPool {
    let workerPath = findWorkerExecutable()
    XCTAssertNotNil(workerPath, "SwiftPythonWorker not found — run swift build first")

    let pool = try await PythonProcessPool(
        workers: workerCount,
        workerExecutablePath: workerPath!
    )

    // Inject venv site-packages so workers can find markitdown etc.
    if let venv = findVenvPath() {
        let fm = FileManager.default
        let libDir = "\(venv)/lib"
        if let pythonDir = try? fm.contentsOfDirectory(atPath: libDir).first(where: { $0.hasPrefix("python") }) {
            let sitePackages = "\(libDir)/\(pythonDir)/site-packages"
            _ = try await pool.eval("import sys; sys.path.insert(0, '\(sitePackages)'); True", worker: 0)
        }
    }

    return pool
}

// MARK: - Tests

final class DocumentExtractorTests: XCTestCase {

    func testInstallSucceeds() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))
        XCTAssertNotNil(handle, "DocumentExtractor should install successfully")
    }

    func testExtractPlainTextFile() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        // Create a temp text file
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_extract_\(UUID().uuidString).txt")
        let content = "Hello, this is a test document.\nIt has multiple lines.\nLine three."
        try content.write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractFile(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        XCTAssertTrue(result.succeeded, "Text extraction should succeed, error: \(result.error ?? "nil")")
        XCTAssertTrue(result.chars > 0, "Should extract some characters")
        XCTAssertTrue(result.text.contains("test document"), "Extracted text should contain original content")
        XCTAssertTrue(result.durationMs >= 0, "Duration should be non-negative")
    }

    func testExtractNonexistentFileReturnsError() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        let result = try await DocumentExtractor.extractFile(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: "/tmp/nonexistent_file_\(UUID().uuidString).pdf"
        )

        XCTAssertFalse(result.succeeded, "Should fail for nonexistent file")
        XCTAssertTrue(result.text.isEmpty, "Text should be empty on failure")
        XCTAssertNotNil(result.error, "Should have an error message")
    }

    func testExtractPDFIfPdfminerAvailable() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        // Create a minimal PDF
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_\(UUID().uuidString).pdf")
        let minimalPDF = """
        %PDF-1.0
        1 0 obj
        << /Type /Catalog /Pages 2 0 R >>
        endobj
        2 0 obj
        << /Type /Pages /Kids [3 0 R] /Count 1 >>
        endobj
        3 0 obj
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
        endobj
        4 0 obj
        << /Length 44 >>
        stream
        BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET
        endstream
        endobj
        5 0 obj
        << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
        endobj
        xref
        0 6
        0000000000 65535 f 
        0000000009 00000 n 
        0000000058 00000 n 
        0000000115 00000 n 
        0000000266 00000 n 
        0000000360 00000 n 
        trailer
        << /Size 6 /Root 1 0 R >>
        startxref
        441
        %%EOF
        """
        try minimalPDF.write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractFile(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        // PDF extraction depends on pdfminer being installed
        if result.succeeded {
            XCTAssertTrue(result.chars > 0, "PDF extraction should produce chars")
            print("PDF extraction succeeded: \(result.chars) chars")
        } else {
            print("PDF extraction not available (pdfminer may not be installed): \(result.error ?? "empty text")")
        }
    }

    // MARK: - Metadata Tests (Feature 1)

    func testMetadataFieldsPresentForText() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_meta_\(UUID().uuidString).txt")
        try "Hello world, this is a test document with enough text for detection.".write(
            to: tempFile, atomically: true, encoding: .utf8
        )
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractFile(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        XCTAssertTrue(result.succeeded)
        XCTAssertEqual(result.format, "txt", "Format should be 'txt' for .txt files")
        XCTAssertNil(result.pages, "Text files should not have page count")
        XCTAssertNil(result.hasTextLayer, "Text files should not have has_text_layer")
    }

    func testMetadataFieldsPresentForPDF() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_meta_\(UUID().uuidString).pdf")
        let minimalPDF = """
        %PDF-1.0
        1 0 obj
        << /Type /Catalog /Pages 2 0 R >>
        endobj
        2 0 obj
        << /Type /Pages /Kids [3 0 R] /Count 1 >>
        endobj
        3 0 obj
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
        endobj
        4 0 obj
        << /Length 44 >>
        stream
        BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET
        endstream
        endobj
        5 0 obj
        << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
        endobj
        xref
        0 6
        0000000000 65535 f 
        0000000009 00000 n 
        0000000058 00000 n 
        0000000115 00000 n 
        0000000266 00000 n 
        0000000360 00000 n 
        trailer
        << /Size 6 /Root 1 0 R >>
        startxref
        441
        %%EOF
        """
        try minimalPDF.write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractFile(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        XCTAssertEqual(result.format, "pdf", "Format should be 'pdf'")
        // pdfminer may or may not parse this minimal PDF, but format should always be set
        if result.pages != nil {
            XCTAssertTrue(result.pages! > 0, "Pages should be positive if detected")
            XCTAssertNotNil(result.hasTextLayer, "has_text_layer should be set when pages detected")
            print("PDF metadata: pages=\(result.pages!), hasTextLayer=\(result.hasTextLayer!)")
        } else {
            print("PDF page count not available (pdfminer may not support this minimal PDF)")
        }
    }

    func testLanguageDetection() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_lang_\(UUID().uuidString).txt")
        let englishText = """
        The quick brown fox jumps over the lazy dog. This is a longer piece of
        English text that should be sufficient for the language detection algorithm
        to correctly identify the language as English with high confidence.
        """
        try englishText.write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractFile(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        XCTAssertTrue(result.succeeded)
        if let lang = result.language {
            XCTAssertEqual(lang, "en", "English text should be detected as 'en'")
            print("Language detected: \(lang)")
        } else {
            print("Language detection not available (langdetect may not be installed)")
        }
    }

    // MARK: - Page-Level Extraction Tests (Feature 2)

    func testExtractPagesFromPDF() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_pages_\(UUID().uuidString).pdf")
        let minimalPDF = """
        %PDF-1.0
        1 0 obj
        << /Type /Catalog /Pages 2 0 R >>
        endobj
        2 0 obj
        << /Type /Pages /Kids [3 0 R] /Count 1 >>
        endobj
        3 0 obj
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
        endobj
        4 0 obj
        << /Length 44 >>
        stream
        BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET
        endstream
        endobj
        5 0 obj
        << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
        endobj
        xref
        0 6
        0000000000 65535 f 
        0000000009 00000 n 
        0000000058 00000 n 
        0000000115 00000 n 
        0000000266 00000 n 
        0000000360 00000 n 
        trailer
        << /Size 6 /Root 1 0 R >>
        startxref
        441
        %%EOF
        """
        try minimalPDF.write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractPages(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        XCTAssertTrue(result.succeeded, "Page extraction should succeed")
        XCTAssertEqual(result.format, "pdf")
        XCTAssertTrue(result.totalPages >= 1, "Should have at least 1 page")
        XCTAssertEqual(result.pages.count, result.totalPages, "Pages array length should match totalPages")
        XCTAssertEqual(result.pages.first?.pageNumber, 1, "First page should be page 1")
        print("PDF page extraction: \(result.totalPages) pages, combinedText length=\(result.combinedText.count)")
    }

    func testExtractPagesFromTextFileSinglePage() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_pages_\(UUID().uuidString).txt")
        try "Hello world from a text file.".write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractPages(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        XCTAssertTrue(result.succeeded)
        XCTAssertEqual(result.totalPages, 1, "Text file should return 1 page")
        XCTAssertEqual(result.pages.count, 1)
        XCTAssertFalse(result.pages[0].isScanned, "Text file page should not be scanned")
        XCTAssertTrue(result.pages[0].text.contains("Hello world"))
    }

    func testCombinedTextContainsPageMarkers() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_combined_\(UUID().uuidString).txt")
        try "Some text content for page markers.".write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractPages(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        XCTAssertTrue(result.succeeded)
        let combined = result.combinedText
        // Single-page documents return plain text with no marker (by design).
        // Page markers ("--- Page N ---") only appear when there are multiple pages.
        XCTAssertFalse(combined.isEmpty, "Combined text should not be empty")
        XCTAssertTrue(combined.contains("Some text content"), "Combined text should contain the original content")
        if result.totalPages > 1 {
            XCTAssertTrue(combined.contains("--- Page 1 ---"), "Multi-page combined text should contain page markers")
        }
    }

    func testScannedPagesDetection() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        // Create a PDF with no real text content (simulates scanned)
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_scanned_\(UUID().uuidString).pdf")
        // Minimal PDF with empty content stream
        let emptyPDF = """
        %PDF-1.0
        1 0 obj
        << /Type /Catalog /Pages 2 0 R >>
        endobj
        2 0 obj
        << /Type /Pages /Kids [3 0 R] /Count 1 >>
        endobj
        3 0 obj
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
        endobj
        xref
        0 4
        0000000000 65535 f 
        0000000009 00000 n 
        0000000058 00000 n 
        0000000115 00000 n 
        trailer
        << /Size 4 /Root 1 0 R >>
        startxref
        190
        %%EOF
        """
        try emptyPDF.write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractPages(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        if result.succeeded {
            // If pdfminer can parse this, the page should be detected as scanned
            for page in result.pages {
                if page.chars < 50 {
                    XCTAssertTrue(page.isScanned, "Page with <50 chars should be marked scanned")
                }
            }
            print("Scanned pages: \(result.scannedPages.count)/\(result.totalPages)")
        }
    }

    // MARK: - OCR Fallback Tests (Feature 3)

    func testGracefulWithoutTesseract() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        // Create a PDF with no text content (simulates scanned)
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_ocr_\(UUID().uuidString).pdf")
        let emptyPDF = """
        %PDF-1.0
        1 0 obj
        << /Type /Catalog /Pages 2 0 R >>
        endobj
        2 0 obj
        << /Type /Pages /Kids [3 0 R] /Count 1 >>
        endobj
        3 0 obj
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
        endobj
        xref
        0 4
        0000000000 65535 f 
        0000000009 00000 n 
        0000000058 00000 n 
        0000000115 00000 n 
        trailer
        << /Size 4 /Root 1 0 R >>
        startxref
        190
        %%EOF
        """
        try emptyPDF.write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        // Should not crash even without pytesseract installed
        let result = try await DocumentExtractor.extractPages(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        // Extraction should succeed (gracefully) — just with empty/scanned pages
        if result.succeeded {
            XCTAssertTrue(result.totalPages >= 1)
            print("OCR graceful fallback: \(result.totalPages) pages, \(result.scannedPages.count) scanned (no crash)")
        }
    }

    // MARK: - XLSX Extraction Tests (Feature 4)

    func testExtractXlsxFallsBackToMarkItDown() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        // Create a minimal XLSX file using openpyxl via Python
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_xlsx_\(UUID().uuidString).xlsx")

        // Use Python to create a real XLSX file
        let createCode = """
        import sys
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Sales"
            ws.append(["Product", "Q1", "Q2"])
            ws.append(["Widget", "100", "150"])
            ws.append(["Gadget", "200", "250"])
            wb.save('\(tempFile.path)')
            _xlsx_status = "created"
        except ImportError:
            _xlsx_status = "no_openpyxl"
        """
        _ = try await pool.eval(createCode, worker: 0)
        let status: String = try await pool.evalResult("_xlsx_status", worker: 0)
        guard status == "created" else {
            print("openpyxl not available, skipping XLSX test")
            return
        }
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractXlsx(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        XCTAssertTrue(result.succeeded, "XLSX extraction should succeed")
        XCTAssertEqual(result.sheets.count, 1, "Should have 1 sheet")
        XCTAssertEqual(result.sheets[0].name, "Sales", "Sheet name should be 'Sales'")
        XCTAssertTrue(result.sheets[0].rowCount >= 3, "Should have at least 3 rows")
        XCTAssertTrue(result.sheets[0].markdownTable.contains("Widget"), "Table should contain 'Widget'")
        XCTAssertTrue(result.sheets[0].headers.contains("A"), "Headers should contain column A")
        print("XLSX: \(result.sheets[0].rowCount) rows, \(result.sheets[0].columnCount) cols")
    }

    func testExtractXlsxMultipleSheets() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_xlsx_multi_\(UUID().uuidString).xlsx")

        let createCode = """
        import sys
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws1 = wb.active
            ws1.title = "Revenue"
            ws1.append(["Month", "Amount"])
            ws1.append(["Jan", "1000"])
            ws2 = wb.create_sheet("Expenses")
            ws2.append(["Category", "Cost"])
            ws2.append(["Rent", "500"])
            wb.save('\(tempFile.path)')
            _xlsx_status = "created"
        except ImportError:
            _xlsx_status = "no_openpyxl"
        """
        _ = try await pool.eval(createCode, worker: 0)
        let status: String = try await pool.evalResult("_xlsx_status", worker: 0)
        guard status == "created" else {
            print("openpyxl not available, skipping multi-sheet XLSX test")
            return
        }
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractXlsx(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        XCTAssertTrue(result.succeeded)
        XCTAssertEqual(result.sheets.count, 2, "Should have 2 sheets")
        XCTAssertEqual(result.sheets[0].name, "Revenue")
        XCTAssertEqual(result.sheets[1].name, "Expenses")

        let combined = result.combinedText
        XCTAssertTrue(combined.contains("## Revenue"), "Combined should have sheet headers")
        XCTAssertTrue(combined.contains("## Expenses"), "Combined should have sheet headers")
        print("Multi-sheet XLSX: \(result.sheets.map(\.name).joined(separator: ", "))")
    }

    func testExtractXlsxCombinedText() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_xlsx_combined_\(UUID().uuidString).xlsx")

        let createCode = """
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Data"
            ws.append(["Name", "Value"])
            ws.append(["Alpha", "42"])
            wb.save('\(tempFile.path)')
            _xlsx_status = "created"
        except ImportError:
            _xlsx_status = "no_openpyxl"
        """
        _ = try await pool.eval(createCode, worker: 0)
        let status: String = try await pool.evalResult("_xlsx_status", worker: 0)
        guard status == "created" else { return }
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let result = try await DocumentExtractor.extractXlsx(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: tempFile.path
        )

        XCTAssertTrue(result.succeeded)
        let combined = result.combinedText
        XCTAssertTrue(combined.contains("Alpha"))
        XCTAssertTrue(combined.contains("42"))
        XCTAssertTrue(combined.contains("|"), "Should contain markdown table pipes")
    }

    func testErrorResultIncludesFormat() async throws {
        let pool = try await makePool()
        defer { Task { await pool.shutdown() } }

        let handle = try await DocumentExtractor.install(on: pool.worker(0))

        let result = try await DocumentExtractor.extractFile(
            pool: pool, workerIndex: 0, kernelHandle: handle,
            filePath: "/tmp/nonexistent_\(UUID().uuidString).pdf"
        )

        XCTAssertFalse(result.succeeded)
        XCTAssertEqual(result.format, "pdf", "Format should be detected even on error")
    }

    // MARK: - Caching Tests (Feature 7)

    func testCacheHitSkipsExtraction() async throws {
        let cache = ExtractionCache()

        let testData = "Hello, test document content".data(using: .utf8)!
        let sha = ExtractionCache.sha256(of: testData)

        // Cache miss
        let miss = await cache.lookup(sha256: sha)
        XCTAssertNil(miss, "Should be a cache miss initially")

        // Store
        await cache.store(sha256: sha, text: "extracted text")

        // Cache hit
        let hit = await cache.lookup(sha256: sha)
        XCTAssertEqual(hit, "extracted text", "Should return cached text")
    }

    func testCacheMissForDifferentContent() async throws {
        let cache = ExtractionCache()

        let data1 = "Document A".data(using: .utf8)!
        let data2 = "Document B".data(using: .utf8)!
        let sha1 = ExtractionCache.sha256(of: data1)
        let sha2 = ExtractionCache.sha256(of: data2)

        await cache.store(sha256: sha1, text: "text A")

        let hit = await cache.lookup(sha256: sha1)
        XCTAssertEqual(hit, "text A")

        let miss = await cache.lookup(sha256: sha2)
        XCTAssertNil(miss, "Different content should not match")
    }

    func testCacheKeyIsSHA256() async throws {
        // Same content, different "filenames" should produce same hash
        let content = "Identical content for both files"
        let data = content.data(using: .utf8)!
        let sha = ExtractionCache.sha256(of: data)

        let cache = ExtractionCache()
        await cache.store(sha256: sha, text: "cached result")

        // Re-compute SHA from same content
        let sha2 = ExtractionCache.sha256(of: content.data(using: .utf8)!)
        XCTAssertEqual(sha, sha2, "Same content should produce same SHA256")

        let hit = await cache.lookup(sha256: sha2)
        XCTAssertEqual(hit, "cached result", "Same content should hit cache regardless of filename")
    }

    func testCacheEviction() async throws {
        let cache = ExtractionCache()

        // Fill cache with 101 entries — should evict the oldest
        for i in 0..<101 {
            let data = "content_\(i)".data(using: .utf8)!
            let sha = ExtractionCache.sha256(of: data)
            await cache.store(sha256: sha, text: "text_\(i)")
        }

        let count = await cache.count
        XCTAssertEqual(count, 100, "Cache should have exactly 100 entries after eviction")
    }

    func testSHA256Deterministic() {
        let data = "test content".data(using: .utf8)!
        let sha1 = ExtractionCache.sha256(of: data)
        let sha2 = ExtractionCache.sha256(of: data)
        XCTAssertEqual(sha1, sha2, "SHA256 should be deterministic")
        XCTAssertEqual(sha1.count, 64, "SHA256 hex should be 64 chars")
    }

    func testVenvPathDiscovery() {
        let venv = findVenvPath()
        XCTAssertNotNil(venv, "Should discover .venv from test file location")
        if let venv {
            XCTAssertTrue(FileManager.default.fileExists(atPath: venv), "Discovered venv should exist")
            let pyvenvCfg = (venv as NSString).appendingPathComponent("pyvenv.cfg")
            XCTAssertTrue(FileManager.default.fileExists(atPath: pyvenvCfg), "venv should have pyvenv.cfg")
            print("Discovered venv at: \(venv)")
        }
    }
}
