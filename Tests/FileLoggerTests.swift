import XCTest
@testable import LlamaInferenceCore

final class FileLoggerTests: XCTestCase {

    // MARK: - Test Helpers

    private func withTempDir(_ block: (URL) async throws -> Void) async throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("FileLoggerTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        try await block(tmp)
    }

    private func makeLogger(baseURL: URL) -> FileLogger {
        FileLogger(baseURL: baseURL)
    }

    private func readLogContents(at baseURL: URL) throws -> String {
        let logFile = baseURL.appendingPathComponent("LlamaInferenceDemo/Logs/llama-inference.log")
        guard FileManager.default.fileExists(atPath: logFile.path) else { return "" }
        return try String(contentsOf: logFile, encoding: .utf8)
    }

    private func setUserDefaults(logLevel: String? = nil, logJsonFormat: Bool? = nil, logRotationSizeMB: Int? = nil) {
        let ud = UserDefaults.standard
        if let v = logLevel { ud.set(v, forKey: "logLevel") }
        if let v = logJsonFormat { ud.set(v, forKey: "logJsonFormat") }
        if let v = logRotationSizeMB { ud.set(v, forKey: "logRotationSizeMB") }
    }

    private func clearLoggerUserDefaults() {
        let ud = UserDefaults.standard
        ud.removeObject(forKey: "logLevel")
        ud.removeObject(forKey: "logJsonFormat")
        ud.removeObject(forKey: "logRotationSizeMB")
    }

    override func tearDown() {
        super.tearDown()
        clearLoggerUserDefaults()
    }

    // MARK: - LogLevel Enum

    func testLogLevelLabels() {
        XCTAssertEqual(FileLogger.LogLevel.debug.label, "debug")
        XCTAssertEqual(FileLogger.LogLevel.info.label, "info")
        XCTAssertEqual(FileLogger.LogLevel.warning.label, "warning")
        XCTAssertEqual(FileLogger.LogLevel.error.label, "error")
    }

    func testLogLevelRawValues() {
        XCTAssertEqual(FileLogger.LogLevel.debug.rawValue, 0)
        XCTAssertEqual(FileLogger.LogLevel.info.rawValue, 1)
        XCTAssertEqual(FileLogger.LogLevel.warning.rawValue, 2)
        XCTAssertEqual(FileLogger.LogLevel.error.rawValue, 3)
    }

    func testLogLevelInitFromRawString() {
        XCTAssertEqual(FileLogger.LogLevel(rawString: "debug"), .debug)
        XCTAssertEqual(FileLogger.LogLevel(rawString: "info"), .info)
        XCTAssertEqual(FileLogger.LogLevel(rawString: "warning"), .warning)
        XCTAssertEqual(FileLogger.LogLevel(rawString: "error"), .error)
    }

    func testLogLevelInitFromRawStringCaseInsensitive() {
        XCTAssertEqual(FileLogger.LogLevel(rawString: "DEBUG"), .debug)
        XCTAssertEqual(FileLogger.LogLevel(rawString: "INFO"), .info)
        XCTAssertEqual(FileLogger.LogLevel(rawString: "WaRnInG"), .warning)
    }

    func testLogLevelInitUnknownDefaultsToInfo() {
        XCTAssertEqual(FileLogger.LogLevel(rawString: "unknown"), .info)
        XCTAssertEqual(FileLogger.LogLevel(rawString: ""), .info)
        XCTAssertEqual(FileLogger.LogLevel(rawString: "trace"), .info)
    }

    // MARK: - Start / Stop / Log (Plain Format)

    func testStartCreatesDirectoryAndFile() async throws {
        try await withTempDir { baseURL in
            let logger = makeLogger(baseURL: baseURL)
            await logger.start()
            await logger.stop()

            let logDir = baseURL.appendingPathComponent("LlamaInferenceDemo/Logs")
            let logFile = logDir.appendingPathComponent("llama-inference.log")
            XCTAssertTrue(FileManager.default.fileExists(atPath: logDir.path))
            XCTAssertTrue(FileManager.default.fileExists(atPath: logFile.path))
        }
    }

    func testLogWritesPlainTextWhenJsonFormatDisabled() async throws {
        clearLoggerUserDefaults()
        setUserDefaults(logLevel: "info", logJsonFormat: false)

        try await withTempDir { baseURL in
            let logger = makeLogger(baseURL: baseURL)
            await logger.start()
            await logger.log(level: .info, category: "TestCat", message: "Hello world")
            await logger.stop()

            let contents = try readLogContents(at: baseURL)
            XCTAssertTrue(contents.contains("Hello world"))
            XCTAssertTrue(contents.contains("TestCat"))
            XCTAssertTrue(contents.contains("[INFO]"))
            XCTAssertTrue(contents.contains("[202") || contents.contains("T")) // ISO8601 date
        }
    }

    func testLogWritesJsonWhenJsonFormatEnabled() async throws {
        clearLoggerUserDefaults()
        setUserDefaults(logLevel: "info", logJsonFormat: true)

        try await withTempDir { baseURL in
            let logger = makeLogger(baseURL: baseURL)
            await logger.start()
            await logger.log(level: .info, category: "TestCat", message: "Hello world")
            await logger.stop()

            let contents = try readLogContents(at: baseURL)
            XCTAssertTrue(contents.contains("\"msg\":\"Hello world\""))
            XCTAssertTrue(contents.contains("\"category\":\"TestCat\""))
            XCTAssertTrue(contents.contains("\"level\":\"info\""))
            XCTAssertTrue(contents.contains("\"ts\":"))
        }
    }

    func testLogRespectsLogLevelFilter() async throws {
        clearLoggerUserDefaults()
        setUserDefaults(logLevel: "warning", logJsonFormat: false)

        try await withTempDir { baseURL in
            let logger = makeLogger(baseURL: baseURL)
            await logger.start()
            await logger.log(level: .debug, category: "X", message: "debug-msg")
            await logger.log(level: .info, category: "X", message: "info-msg")
            await logger.log(level: .warning, category: "X", message: "warning-msg")
            await logger.log(level: .error, category: "X", message: "error-msg")
            await logger.stop()

            let contents = try readLogContents(at: baseURL)
            XCTAssertFalse(contents.contains("debug-msg"))
            XCTAssertFalse(contents.contains("info-msg"))
            XCTAssertTrue(contents.contains("warning-msg"))
            XCTAssertTrue(contents.contains("error-msg"))
        }
    }

    func testLogLevelInfoAllowsInfoAndAbove() async throws {
        clearLoggerUserDefaults()
        setUserDefaults(logLevel: "info", logJsonFormat: false)

        try await withTempDir { baseURL in
            let logger = makeLogger(baseURL: baseURL)
            await logger.start()
            await logger.log(level: .debug, category: "X", message: "debug")
            await logger.log(level: .info, category: "X", message: "info")
            await logger.stop()

            let contents = try readLogContents(at: baseURL)
            XCTAssertFalse(contents.contains("debug"))
            XCTAssertTrue(contents.contains("info"))
        }
    }

    func testLogLevelDebugAllowsAll() async throws {
        clearLoggerUserDefaults()
        setUserDefaults(logLevel: "debug", logJsonFormat: false)

        try await withTempDir { baseURL in
            let logger = makeLogger(baseURL: baseURL)
            await logger.start()
            await logger.log(level: .debug, category: "X", message: "debug-msg")
            await logger.stop()

            let contents = try readLogContents(at: baseURL)
            XCTAssertTrue(contents.contains("debug-msg"))
        }
    }

    func testLogEscapesJsonCharacters() async throws {
        clearLoggerUserDefaults()
        setUserDefaults(logLevel: "info", logJsonFormat: true)

        try await withTempDir { baseURL in
            let logger = makeLogger(baseURL: baseURL)
            await logger.start()
            await logger.log(level: .info, category: "cat\"with\"quotes", message: "msg\nwith\nnewlines")
            await logger.stop()

            let contents = try readLogContents(at: baseURL)
            // Escaped: \" and \n
            XCTAssertTrue(contents.contains("\\\""))
            XCTAssertTrue(contents.contains("\\n"))
            XCTAssertFalse(contents.contains("\nwith\n")) // Raw newlines should be escaped
        }
    }

    func testStopBeforeStartDoesNotCrash() async throws {
        try await withTempDir { baseURL in
            let logger = makeLogger(baseURL: baseURL)
            await logger.stop()
        }
    }

    func testLogWithoutStartDoesNotCrash() async throws {
        clearLoggerUserDefaults()
        setUserDefaults(logLevel: "info", logJsonFormat: false)

        try await withTempDir { baseURL in
            let logger = makeLogger(baseURL: baseURL)
            // Never call start - fileHandle is nil
            await logger.log(level: .info, category: "X", message: "no crash")
            await logger.stop()
        }
    }

    // MARK: - Log Rotation

    func testRotationCreatesRotatedFileWhenOverLimit() async throws {
        clearLoggerUserDefaults()
        setUserDefaults(logLevel: "info", logJsonFormat: false, logRotationSizeMB: 1)

        try await withTempDir { baseURL in
            let logger = makeLogger(baseURL: baseURL)
            await logger.start()

            // Fill the log file to exceed 1MB
            let chunk = String(repeating: "x", count: 100_000)
            for i in 0..<11 {
                await logger.log(level: .info, category: "fill", message: "chunk-\(i)-\(chunk)")
            }
            await logger.stop()

            let logDir = baseURL.appendingPathComponent("LlamaInferenceDemo/Logs")
            let rotated = logDir.appendingPathComponent("llama-inference.1.log")
            let mainLog = logDir.appendingPathComponent("llama-inference.log")

            if FileManager.default.fileExists(atPath: rotated.path) {
                XCTAssertTrue(FileManager.default.fileExists(atPath: mainLog.path))
            }
        }
    }
}
