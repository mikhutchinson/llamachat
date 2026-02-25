import Foundation
import SwiftPythonRuntime
import CryptoKit

/// MarkItDown-based document text extraction kernel.
///
/// Converts PDF, DOCX, PPTX, XLSX, HTML, and other document formats into
/// Markdown text. Lightweight — no LLM needed. Installed on worker 0.
public enum DocumentExtractor {
    public static let kernelSource = #"""
    import json, sys, os, tempfile, time

    class DocumentExtractor:
        """Extract text from documents using MarkItDown.

        Supports: PDF, DOCX, PPTX, XLSX, HTML, CSV, XML, JSON, and more.
        Returns Markdown-formatted text with metadata.
        """

        def __init__(self):
            self._md = None
            self._log("DocumentExtractor init (markitdown loaded lazily)")

        def _log(self, msg):
            print(f"[DocExtract] {msg}", file=sys.stderr, flush=True)

        def _ensure_md(self):
            if self._md is None:
                self._log(f"Loading markitdown... sys.path={sys.path[:5]}")
                from markitdown import MarkItDown
                self._md = MarkItDown()
                self._log("MarkItDown loaded")
            return self._md

        def _detect_format(self, file_path):
            """Detect document format from extension."""
            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            format_map = {
                'pdf': 'pdf', 'docx': 'docx', 'doc': 'doc',
                'pptx': 'pptx', 'ppt': 'ppt', 'xlsx': 'xlsx',
                'xls': 'xls', 'html': 'html', 'htm': 'html',
                'csv': 'csv', 'xml': 'xml', 'json': 'json',
                'txt': 'txt', 'md': 'markdown', 'py': 'python',
                'swift': 'swift', 'rs': 'rust', 'js': 'javascript',
            }
            return format_map.get(ext, ext if ext else 'unknown')

        def _get_pdf_metadata(self, file_path):
            """Extract PDF-specific metadata: page count, text layer, title, author."""
            meta = {'pages': None, 'has_text_layer': None, 'title': None, 'author': None}
            try:
                from pdfminer.high_level import extract_text
                from pdfminer.pdfparser import PDFParser
                from pdfminer.pdfdocument import PDFDocument
                from pdfminer.pdfpage import PDFPage

                with open(file_path, 'rb') as f:
                    parser = PDFParser(f)
                    doc = PDFDocument(parser)
                    pages = list(PDFPage.create_pages(doc))
                    meta['pages'] = len(pages)

                    # Document info (title, author)
                    if doc.info:
                        info = doc.info[0] if isinstance(doc.info, list) else doc.info
                        for key in ('Title', 'title'):
                            val = info.get(key)
                            if val:
                                meta['title'] = val.decode('utf-8', errors='ignore') if isinstance(val, bytes) else str(val)
                                break
                        for key in ('Author', 'author'):
                            val = info.get(key)
                            if val:
                                meta['author'] = val.decode('utf-8', errors='ignore') if isinstance(val, bytes) else str(val)
                                break

                    # Check text layer: extract first page text
                    if meta['pages'] and meta['pages'] > 0:
                        try:
                            first_page_text = extract_text(file_path, page_numbers=[0])
                            chars_first = len(first_page_text.strip()) if first_page_text else 0
                            meta['has_text_layer'] = chars_first > 20
                        except Exception:
                            meta['has_text_layer'] = None
            except ImportError:
                self._log("pdfminer not available for PDF metadata")
            except Exception as e:
                self._log(f"PDF metadata extraction failed: {e}")
            return meta

        def _detect_language(self, text):
            """Detect language of text. Returns ISO 639-1 code or None."""
            if not text or len(text.strip()) < 20:
                return None
            try:
                from langdetect import detect
                sample = text[:5000]
                return detect(sample)
            except ImportError:
                self._log("langdetect not available for language detection")
                return None
            except Exception:
                return None

        def extract_file(self, file_path):
            """Extract text from a file on disk.

            Args:
                file_path: Absolute path to the file.

            Returns:
                JSON string with text, metadata, and timing:
                {"text": "...", "chars": N, "duration_ms": N,
                 "format": "pdf", "pages": N, "has_text_layer": bool,
                 "title": "...", "author": "...", "language": "en"}
                or {"error": "...", "text": ""} on failure.
            """
            t0 = time.perf_counter()
            try:
                md = self._ensure_md()
                result = md.convert(file_path)
                text = result.text_content if result.text_content else ""
                fmt = self._detect_format(file_path)

                # PDF-specific metadata
                pdf_meta = self._get_pdf_metadata(file_path) if fmt == 'pdf' else {}
                pages = pdf_meta.get('pages')
                has_text_layer = pdf_meta.get('has_text_layer')
                title = pdf_meta.get('title')
                author = pdf_meta.get('author')

                # If we have pages and text, refine has_text_layer with full char count
                if pages and pages > 0 and has_text_layer is None:
                    has_text_layer = len(text.strip()) > pages * 50

                language = self._detect_language(text)

                t1 = time.perf_counter()
                self._log(f"Extracted {len(text)} chars from {os.path.basename(file_path)} in {(t1-t0)*1000:.0f}ms")
                return json.dumps({
                    "text": text,
                    "chars": len(text),
                    "duration_ms": round((t1 - t0) * 1000, 2),
                    "format": fmt,
                    "pages": pages,
                    "has_text_layer": has_text_layer,
                    "title": title,
                    "author": author,
                    "language": language,
                })
            except Exception as e:
                t1 = time.perf_counter()
                self._log(f"Extraction failed for {os.path.basename(file_path)}: {e}")
                return json.dumps({
                    "error": str(e),
                    "text": "",
                    "chars": 0,
                    "duration_ms": round((t1 - t0) * 1000, 2),
                    "format": self._detect_format(file_path),
                    "pages": None,
                    "has_text_layer": None,
                    "title": None,
                    "author": None,
                    "language": None,
                })

        def extract_pages(self, file_path):
            """Extract text per-page from a document.

            For PDFs, uses pdfminer to extract each page individually.
            For non-PDF files, returns a single page with the full text.

            Args:
                file_path: Absolute path to the file.

            Returns:
                JSON string: {"pages": [{"page": 1, "text": "...", "chars": N,
                              "is_scanned": bool}, ...], "total_pages": N,
                              "format": "pdf", "duration_ms": N}
                or {"error": "...", "pages": []} on failure.
            """
            t0 = time.perf_counter()
            fmt = self._detect_format(file_path)
            try:
                if fmt == 'pdf':
                    return self._extract_pdf_pages(file_path, t0)
                else:
                    # Non-PDF: extract as single page
                    md = self._ensure_md()
                    result = md.convert(file_path)
                    text = result.text_content if result.text_content else ""
                    t1 = time.perf_counter()
                    return json.dumps({
                        "pages": [{"page": 1, "text": text, "chars": len(text), "is_scanned": False}],
                        "total_pages": 1,
                        "format": fmt,
                        "duration_ms": round((t1 - t0) * 1000, 2),
                    })
            except Exception as e:
                t1 = time.perf_counter()
                self._log(f"Page extraction failed for {os.path.basename(file_path)}: {e}")
                return json.dumps({
                    "error": str(e),
                    "pages": [],
                    "total_pages": 0,
                    "format": fmt,
                    "duration_ms": round((t1 - t0) * 1000, 2),
                })

        def _has_ocr(self):
            """Check if pytesseract and pdf2image are available for OCR."""
            try:
                import pytesseract
                from pdf2image import convert_from_path
                return True
            except ImportError:
                return False

        def _ocr_page(self, file_path, page_number):
            """OCR a single PDF page. Returns extracted text or empty string.

            Args:
                file_path: Path to PDF file.
                page_number: 1-indexed page number.

            Returns:
                Extracted text string, or "" on failure.
            """
            try:
                import pytesseract
                from pdf2image import convert_from_path
                images = convert_from_path(file_path, first_page=page_number, last_page=page_number, dpi=300)
                if not images:
                    return ""
                text = pytesseract.image_to_string(images[0])
                self._log(f"OCR page {page_number}: {len(text.strip())} chars")
                return text
            except ImportError:
                return ""
            except Exception as e:
                self._log(f"OCR failed for page {page_number}: {e}")
                return ""

        def _extract_pdf_pages(self, file_path, t0):
            """Extract text from each PDF page via pdfminer, with OCR fallback for scanned pages."""
            try:
                from pdfminer.high_level import extract_text
                from pdfminer.pdfparser import PDFParser
                from pdfminer.pdfdocument import PDFDocument
                from pdfminer.pdfpage import PDFPage
            except ImportError:
                # Fallback: extract as single page via MarkItDown
                self._log("pdfminer not available, falling back to single-page extraction")
                md = self._ensure_md()
                result = md.convert(file_path)
                text = result.text_content if result.text_content else ""
                t1 = time.perf_counter()
                return json.dumps({
                    "pages": [{"page": 1, "text": text, "chars": len(text), "is_scanned": False}],
                    "total_pages": 1,
                    "format": "pdf",
                    "duration_ms": round((t1 - t0) * 1000, 2),
                })

            with open(file_path, 'rb') as f:
                parser = PDFParser(f)
                doc = PDFDocument(parser)
                page_list = list(PDFPage.create_pages(doc))
                total_pages = len(page_list)

            ocr_available = self._has_ocr()
            pages = []
            for i in range(total_pages):
                try:
                    page_text = extract_text(file_path, page_numbers=[i])
                    page_text = page_text if page_text else ""
                    chars = len(page_text.strip())
                    is_scanned = chars < 50

                    # OCR fallback for scanned pages
                    if is_scanned and ocr_available:
                        ocr_text = self._ocr_page(file_path, i + 1)
                        if len(ocr_text.strip()) > chars:
                            page_text = ocr_text
                            chars = len(ocr_text.strip())
                            # Still marked scanned (original had no text layer)

                    pages.append({
                        "page": i + 1,
                        "text": page_text,
                        "chars": chars,
                        "is_scanned": is_scanned,
                    })
                except Exception as e:
                    self._log(f"Page {i+1} extraction failed: {e}")
                    pages.append({
                        "page": i + 1,
                        "text": "",
                        "chars": 0,
                        "is_scanned": True,
                    })

            total_chars = sum(p["chars"] for p in pages)
            scanned_count = sum(1 for p in pages if p["is_scanned"])

            # If pdfminer returned nothing, try MarkItDown whole-document fallback
            # (handles unusual encodings and some CID-font PDFs)
            if total_chars == 0:
                try:
                    md = self._ensure_md()
                    fallback_result = md.convert(file_path)
                    fallback_text = fallback_result.text_content if fallback_result.text_content else ""
                    if fallback_text.strip():
                        self._log(f"pdfminer returned no text; MarkItDown fallback: {len(fallback_text)} chars")
                        t1 = time.perf_counter()
                        return json.dumps({
                            "pages": [{"page": 1, "text": fallback_text, "chars": len(fallback_text), "is_scanned": False}],
                            "total_pages": 1,
                            "format": "pdf",
                            "duration_ms": round((t1 - t0) * 1000, 2),
                        })
                except Exception as fe:
                    self._log(f"MarkItDown fallback also failed: {fe}")

            t1 = time.perf_counter()
            self._log(f"Extracted {total_pages} pages ({total_chars} chars, {scanned_count} scanned) from {os.path.basename(file_path)} in {(t1-t0)*1000:.0f}ms")
            return json.dumps({
                "pages": pages,
                "total_pages": total_pages,
                "format": "pdf",
                "duration_ms": round((t1 - t0) * 1000, 2),
            })

        def extract_xlsx(self, file_path):
            """Extract spreadsheet data with sheet names and cell references.

            Uses openpyxl for richer extraction than MarkItDown. Falls back
            to MarkItDown if openpyxl is unavailable.

            Args:
                file_path: Absolute path to XLSX file.

            Returns:
                JSON string: {"sheets": [{"name": "Sheet1",
                  "headers": ["A", "B", ...], "rows": [[v1, v2, ...], ...],
                  "row_count": N, "col_count": N,
                  "markdown_table": "| A | B |\\n..."}, ...],
                  "duration_ms": N}
            """
            t0 = time.perf_counter()
            try:
                import openpyxl
            except ImportError:
                self._log("openpyxl not available, falling back to MarkItDown")
                md = self._ensure_md()
                result = md.convert(file_path)
                text = result.text_content if result.text_content else ""
                t1 = time.perf_counter()
                return json.dumps({
                    "sheets": [{"name": "Sheet1", "headers": [], "rows": [],
                                "row_count": 0, "col_count": 0,
                                "markdown_table": text}],
                    "format": "xlsx",
                    "duration_ms": round((t1 - t0) * 1000, 2),
                })

            try:
                wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                sheets = []
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    rows_data = []
                    max_col = 0
                    for row in ws.iter_rows(values_only=False):
                        row_vals = []
                        for cell in row:
                            val = cell.value
                            if val is None:
                                row_vals.append("")
                            else:
                                row_vals.append(str(val))
                        rows_data.append(row_vals)
                        max_col = max(max_col, len(row_vals))

                    # Generate column headers (A, B, C, ...)
                    def col_letter(n):
                        s = ""
                        while n > 0:
                            n -= 1
                            s = chr(65 + n % 26) + s
                            n //= 26
                        return s

                    headers = [col_letter(i + 1) for i in range(max_col)]

                    # Build markdown table
                    md_lines = []
                    if rows_data:
                        first = rows_data[0]
                        md_lines.append("| " + " | ".join(first) + " |")
                        md_lines.append("| " + " | ".join(["---"] * len(first)) + " |")
                        for row in rows_data[1:]:
                            # Pad to same length
                            padded = row + [""] * (len(first) - len(row))
                            md_lines.append("| " + " | ".join(padded) + " |")

                    sheets.append({
                        "name": sheet_name,
                        "headers": headers,
                        "rows": rows_data,
                        "row_count": len(rows_data),
                        "col_count": max_col,
                        "markdown_table": "\n".join(md_lines),
                    })

                wb.close()
                t1 = time.perf_counter()
                total_rows = sum(s["row_count"] for s in sheets)
                self._log(f"Extracted {len(sheets)} sheets ({total_rows} rows) from {os.path.basename(file_path)} in {(t1-t0)*1000:.0f}ms")
                return json.dumps({
                    "sheets": sheets,
                    "format": "xlsx",
                    "duration_ms": round((t1 - t0) * 1000, 2),
                })
            except Exception as e:
                t1 = time.perf_counter()
                self._log(f"XLSX extraction failed: {e}")
                return json.dumps({
                    "error": str(e),
                    "sheets": [],
                    "format": "xlsx",
                    "duration_ms": round((t1 - t0) * 1000, 2),
                })

    """#

    /// Install on a worker. No model or GPU needed.
    public static func install(
        on worker: PythonProcessPool.WorkerContext
    ) async throws -> PyHandle {
        // Step 1: Define the class in the worker namespace
        _ = try await worker.eval(kernelSource + "\nTrue")
        // Step 2: Instantiate via a separate eval
        return try await worker.eval("DocumentExtractor()")
    }

    /// Extract text from a file path using the installed kernel.
    public static func extractFile(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        filePath: String,
        timeout: TimeInterval = 60
    ) async throws -> DocumentExtractionResult {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "extract_file",
            args: [.python(filePath)],
            worker: workerIndex,
            timeout: timeout
        )
        let parsed = try LlamaSessionKernel.parseJSON(json)
        let text = (parsed["text"] as? String) ?? ""
        let chars = (parsed["chars"] as? Int) ?? 0
        let durationMs = (parsed["duration_ms"] as? Double) ?? 0
        let error = parsed["error"] as? String
        let format = parsed["format"] as? String
        let pages = parsed["pages"] as? Int
        let hasTextLayer = parsed["has_text_layer"] as? Bool
        let title = parsed["title"] as? String
        let author = parsed["author"] as? String
        let language = parsed["language"] as? String
        return DocumentExtractionResult(
            text: text,
            chars: chars,
            durationMs: durationMs,
            error: error,
            format: format,
            pages: pages,
            hasTextLayer: hasTextLayer,
            title: title,
            author: author,
            language: language
        )
    }

    /// Extract text per-page from a document using the installed kernel.
    public static func extractPages(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        filePath: String,
        timeout: TimeInterval = 120
    ) async throws -> PageExtractionResult {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "extract_pages",
            args: [.python(filePath)],
            worker: workerIndex,
            timeout: timeout
        )
        let parsed = try LlamaSessionKernel.parseJSON(json)
        let error = parsed["error"] as? String
        let totalPages = (parsed["total_pages"] as? Int) ?? 0
        let format = parsed["format"] as? String
        let durationMs = (parsed["duration_ms"] as? Double) ?? 0
        let pagesRaw = (parsed["pages"] as? [[String: Any]]) ?? []

        let pages = pagesRaw.map { p in
            PageContent(
                pageNumber: (p["page"] as? Int) ?? 0,
                text: (p["text"] as? String) ?? "",
                chars: (p["chars"] as? Int) ?? 0,
                isScanned: (p["is_scanned"] as? Bool) ?? false
            )
        }
        return PageExtractionResult(
            pages: pages,
            totalPages: totalPages,
            format: format,
            durationMs: durationMs,
            error: error
        )
    }

    /// Extract XLSX spreadsheet with sheet names and cell references.
    public static func extractXlsx(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        filePath: String,
        timeout: TimeInterval = 60
    ) async throws -> XlsxExtractionResult {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "extract_xlsx",
            args: [.python(filePath)],
            worker: workerIndex,
            timeout: timeout
        )
        let parsed = try LlamaSessionKernel.parseJSON(json)
        let error = parsed["error"] as? String
        let durationMs = (parsed["duration_ms"] as? Double) ?? 0
        let sheetsRaw = (parsed["sheets"] as? [[String: Any]]) ?? []

        let sheets = sheetsRaw.map { s in
            SheetContent(
                name: (s["name"] as? String) ?? "Sheet",
                headers: (s["headers"] as? [String]) ?? [],
                rowCount: (s["row_count"] as? Int) ?? 0,
                columnCount: (s["col_count"] as? Int) ?? 0,
                markdownTable: (s["markdown_table"] as? String) ?? ""
            )
        }
        return XlsxExtractionResult(
            sheets: sheets,
            durationMs: durationMs,
            error: error
        )
    }
}

// MARK: - Extraction Cache

/// Thread-safe SHA256-keyed cache for extraction results. LRU eviction at 100 entries.
public actor ExtractionCache {
    public static let shared = ExtractionCache()

    private struct CachedEntry {
        let text: String
        var lastAccessed: Date
    }

    private var cache: [String: CachedEntry] = [:]
    private let maxEntries = 100

    /// Compute SHA256 hex digest for file data.
    public nonisolated static func sha256(of data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Look up cached extraction text by SHA256 hash.
    public func lookup(sha256: String) -> String? {
        guard var entry = cache[sha256] else { return nil }
        entry.lastAccessed = Date()
        cache[sha256] = entry
        return entry.text
    }

    /// Store extraction text keyed by SHA256 hash. Evicts oldest if over limit.
    public func store(sha256: String, text: String) {
        cache[sha256] = CachedEntry(text: text, lastAccessed: Date())
        if cache.count > maxEntries {
            evictOldest()
        }
    }

    /// Number of cached entries (for testing).
    public var count: Int { cache.count }

    /// Clear all cached entries.
    public func clear() {
        cache.removeAll()
    }

    private func evictOldest() {
        guard let oldest = cache.min(by: { $0.value.lastAccessed < $1.value.lastAccessed }) else { return }
        cache.removeValue(forKey: oldest.key)
    }
}

// MARK: - Result Types

public struct DocumentExtractionResult: Sendable {
    public let text: String
    public let chars: Int
    public let durationMs: Double
    public let error: String?
    public let format: String?
    public let pages: Int?
    public let hasTextLayer: Bool?
    public let title: String?
    public let author: String?
    public let language: String?

    public var succeeded: Bool { error == nil && !text.isEmpty }
}

public struct PageContent: Sendable {
    public let pageNumber: Int
    public let text: String
    public let chars: Int
    public let isScanned: Bool
}

public struct PageExtractionResult: Sendable {
    public let pages: [PageContent]
    public let totalPages: Int
    public let format: String?
    public let durationMs: Double
    public let error: String?

    public var succeeded: Bool { error == nil && !pages.isEmpty }

    /// Combine all page texts, skipping empty pages.
    /// Single-page results are returned without a page marker.
    public var combinedText: String {
        let nonEmpty = pages.filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard !nonEmpty.isEmpty else { return "" }
        if nonEmpty.count == 1 { return nonEmpty[0].text }
        return nonEmpty.map { "--- Page \($0.pageNumber) ---\n\($0.text)" }.joined(separator: "\n\n")
    }

    /// Pages detected as scanned (no text layer).
    public var scannedPages: [PageContent] {
        pages.filter(\.isScanned)
    }
}

public struct SheetContent: Sendable {
    public let name: String
    public let headers: [String]
    public let rowCount: Int
    public let columnCount: Int
    public let markdownTable: String
}

public struct XlsxExtractionResult: Sendable {
    public let sheets: [SheetContent]
    public let durationMs: Double
    public let error: String?

    public var succeeded: Bool { error == nil && !sheets.isEmpty }

    /// Combined markdown tables for all sheets with sheet name headers.
    public var combinedText: String {
        sheets.map { "## \($0.name)\n\($0.markdownTable)" }.joined(separator: "\n\n")
    }
}
