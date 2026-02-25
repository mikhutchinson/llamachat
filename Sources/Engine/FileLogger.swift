import Foundation
import OSLog

/// Lightweight file-based logger that complements OSLog.
///
/// Writes structured or plain-text log entries to a rotating file.
/// All writes are serialized on a dedicated dispatch queue.
/// Reads `logLevel`, `logRotationSizeMB`, and `logJsonFormat` from UserDefaults.
public actor FileLogger {
    public static let shared = FileLogger()

    private let logDir: URL
    private let logFile: URL
    private var fileHandle: FileHandle?
    private let dateFormatter: ISO8601DateFormatter = {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fmt
    }()
    private let osLogger = Logger(subsystem: "com.llama-inference-demo", category: "FileLogger")

    private init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        self.init(baseURL: appSupport)
    }

    /// For testing: use a custom base URL instead of Application Support.
    internal init(baseURL: URL) {
        logDir = baseURL.appendingPathComponent("LlamaInferenceDemo/Logs")
        logFile = logDir.appendingPathComponent("llama-inference.log")
    }

    /// Ensure the log directory and file handle are ready.
    public func start() {
        let fm = FileManager.default
        if !fm.fileExists(atPath: logDir.path) {
            try? fm.createDirectory(at: logDir, withIntermediateDirectories: true)
        }
        if !fm.fileExists(atPath: logFile.path) {
            fm.createFile(atPath: logFile.path, contents: nil)
        }
        fileHandle = FileHandle(forWritingAtPath: logFile.path)
        fileHandle?.seekToEndOfFile()
        osLogger.info("FileLogger started at \(self.logFile.path, privacy: .public)")
        // Write effective log level so user can verify Settings → Advanced is applied (only when INFO would pass)
        let level = currentLogLevel
        guard level.rawValue <= LogLevel.info.rawValue else { return }
        let ts = dateFormatter.string(from: Date())
        let bootstrap = "[\(ts)] [INFO] [FileLogger] Started, effective level: \(level.label)\n"
        if let data = bootstrap.data(using: .utf8) {
            fileHandle?.write(data)
        }
    }

    /// Write a log entry if the level passes the configured filter.
    public func log(level: LogLevel, category: String, message: String) {
        guard level.rawValue >= currentLogLevel.rawValue else { return }
        rotateIfNeeded()

        let timestamp = dateFormatter.string(from: Date())
        let useJson = UserDefaults.standard.bool(forKey: "logJsonFormat")

        let line: String
        if useJson {
            // Escape JSON strings
            let escapedMsg = message
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
                .replacingOccurrences(of: "\n", with: "\\n")
            let escapedCat = category
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            line = "{\"ts\":\"\(timestamp)\",\"level\":\"\(level.label)\",\"category\":\"\(escapedCat)\",\"msg\":\"\(escapedMsg)\"}\n"
        } else {
            line = "[\(timestamp)] [\(level.label.uppercased())] [\(category)] \(message)\n"
        }

        if let data = line.data(using: .utf8) {
            fileHandle?.write(data)
        }
    }

    /// Flush and close the file handle.
    public func stop() {
        fileHandle?.closeFile()
        fileHandle = nil
    }

    // MARK: - Private

    private var currentLogLevel: LogLevel {
        let raw = UserDefaults.standard.string(forKey: "logLevel") ?? "info"
        return LogLevel(rawString: raw)
    }

    private func rotateIfNeeded() {
        let maxMB = UserDefaults.standard.integer(forKey: "logRotationSizeMB")
        let maxBytes = (maxMB > 0 ? maxMB : 10) * 1_048_576

        guard let attrs = try? FileManager.default.attributesOfItem(atPath: logFile.path),
              let size = attrs[.size] as? Int64,
              size > Int64(maxBytes) else { return }

        fileHandle?.closeFile()
        fileHandle = nil

        let rotated = logDir.appendingPathComponent("llama-inference.1.log")
        let fm = FileManager.default
        try? fm.removeItem(at: rotated)
        try? fm.moveItem(at: logFile, to: rotated)
        fm.createFile(atPath: logFile.path, contents: nil)
        fileHandle = FileHandle(forWritingAtPath: logFile.path)
    }

    // MARK: - Log Level

    public enum LogLevel: Int, Sendable {
        case debug = 0
        case info = 1
        case warning = 2
        case error = 3

        var label: String {
            switch self {
            case .debug: return "debug"
            case .info: return "info"
            case .warning: return "warning"
            case .error: return "error"
            }
        }

        init(rawString: String) {
            switch rawString.lowercased() {
            case "debug": self = .debug
            case "info": self = .info
            case "warning": self = .warning
            case "error": self = .error
            default: self = .info
            }
        }
    }
}
