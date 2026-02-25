import Foundation
import OSLog
import CryptoKit

/// Manages GGUF model downloads with real-time progress tracking.
/// Uses URLSessionDownloadTask with a delegate for efficient chunked downloads.
@Observable
@MainActor
public final class ModelDownloadManager {
    public var activeDownloads: [String: DownloadTask] = [:]
    public var downloadDirectory: URL

    private var downloadDelegate: DownloadDelegate?
    private var delegateSession: URLSession?
    private var sessionTasks: [String: URLSessionDownloadTask] = [:]
    private var downloadMeta: [String: DownloadMeta] = [:]
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "DownloadManager")
    private static let sourceMapFilename = ".modelhub-download-sources.json"
    private static let sourceLogFilename = "modelhub-download.log"
    private static let sourceLogVersion = 1

    private struct DownloadMeta {
        let destURL: URL
        let expectedSHA256: String?
    }

    public var hasActiveDownloads: Bool {
        activeDownloads.values.contains { task in
            if case .downloading = task.state { return true }
            if case .queued = task.state { return true }
            return false
        }
    }

    public init() {
        let defaultDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Models/gguf")
        let stored = UserDefaults.standard.string(forKey: "modelDownloadDirectory")
        if let stored, !stored.isEmpty {
            self.downloadDirectory = URL(fileURLWithPath: stored)
        } else {
            self.downloadDirectory = defaultDir
        }
    }

    private func ensureSession() {
        guard downloadDelegate == nil else { return }
        let delegate = DownloadDelegate()
        downloadDelegate = delegate
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 86400  // 24h for large models
        delegateSession = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
    }

    // MARK: - Download

    /// Start downloading a GGUF file. Progress tracked via `activeDownloads`.
    public func download(_ file: GGUFFile) {
        guard sessionTasks[file.id] == nil else { return }
        ensureSession()
        persistDownloadSource(filename: file.filename, repoId: file.repoId)

        let fm = FileManager.default
        if !fm.fileExists(atPath: downloadDirectory.path) {
            do {
                try fm.createDirectory(at: downloadDirectory, withIntermediateDirectories: true)
            } catch {
                activeDownloads[file.id] = DownloadTask(
                    id: file.id, filename: file.filename, repoId: file.repoId,
                    totalBytes: 0, downloadedBytes: 0,
                    state: .failed("Cannot create directory: \(String(describing: error))")
                )
                return
            }
        }

        let destURL = downloadDirectory.appendingPathComponent(file.filename)

        // Already downloaded
        if fm.fileExists(atPath: destURL.path) {
            let attrs = try? fm.attributesOfItem(atPath: destURL.path)
            let size = attrs?[.size] as? Int64 ?? 0
            activeDownloads[file.id] = DownloadTask(
                id: file.id, filename: file.filename, repoId: file.repoId,
                totalBytes: size, downloadedBytes: size,
                state: .completed(localPath: destURL.path)
            )
            return
        }

        activeDownloads[file.id] = DownloadTask(
            id: file.id, filename: file.filename, repoId: file.repoId,
            totalBytes: file.estimatedSize ?? 0, downloadedBytes: 0,
            state: .downloading
        )

        var request = URLRequest(url: file.downloadURL)
        request.setValue("SwiftPython/ModelHub", forHTTPHeaderField: "User-Agent")

        let task = delegateSession!.downloadTask(with: request)
        sessionTasks[file.id] = task

        let expectedHash = file.expectedSHA256
        downloadMeta[file.id] = DownloadMeta(destURL: destURL, expectedSHA256: expectedHash)

        // Register callbacks with the delegate
        downloadDelegate?.register(
            taskId: task.taskIdentifier,
            fileId: file.id,
            destURL: destURL,
            onProgress: { [weak self] fileId, written, total in
                Task { @MainActor [weak self] in
                    self?.activeDownloads[fileId]?.downloadedBytes = written
                    self?.activeDownloads[fileId]?.totalBytes = total
                }
            },
            onComplete: { [weak self] fileId, result in
                Task { @MainActor [weak self] in
                    switch result {
                    case .success(let path):
                        if let expectedHash, !expectedHash.isEmpty {
                            self?.activeDownloads[fileId]?.state = .verifying
                            self?.logger.info("Verifying SHA256: \(fileId, privacy: .public)")
                            self?.verifyAndComplete(fileId: fileId, path: path, expectedHash: expectedHash)
                        } else {
                            self?.activeDownloads[fileId]?.state = .completed(localPath: path)
                            self?.logger.info("Download complete (no checksum): \(fileId, privacy: .public)")
                        }
                    case .failure(let error):
                        if (error as NSError).code == NSURLErrorCancelled {
                            if case .paused = self?.activeDownloads[fileId]?.state { return }
                            self?.activeDownloads[fileId]?.state = .cancelled
                        } else {
                            self?.activeDownloads[fileId]?.state = .failed(String(describing: error))
                            self?.logger.error("Download failed: \(fileId, privacy: .public) — \(String(describing: error), privacy: .public)")
                        }
                    }
                    self?.sessionTasks.removeValue(forKey: fileId)
                }
            }
        )

        logger.info("Starting download: \(file.filename, privacy: .public) from \(file.repoId, privacy: .public)")
        task.resume()
    }

    /// Cancel an in-progress download (discards partial data).
    public func cancel(filename: String) {
        sessionTasks[filename]?.cancel()
        sessionTasks.removeValue(forKey: filename)
        downloadMeta.removeValue(forKey: filename)
        activeDownloads[filename]?.state = .cancelled
        activeDownloads[filename]?.resumeData = nil
    }

    /// Pause an in-progress download (preserves partial data for resume).
    public func pause(filename: String) {
        guard let task = sessionTasks[filename] else { return }
        activeDownloads[filename]?.state = .paused
        task.cancel(byProducingResumeData: { [weak self] data in
            Task { @MainActor [weak self] in
                self?.activeDownloads[filename]?.resumeData = data
                self?.sessionTasks.removeValue(forKey: filename)
                if data == nil {
                    self?.logger.warning("No resume data for \(filename, privacy: .public) — server may not support Range")
                } else {
                    self?.logger.info("Paused: \(filename, privacy: .public)")
                }
            }
        })
    }

    /// Resume a paused download from saved resume data.
    public func resumeDownload(filename: String) {
        guard let resumeData = activeDownloads[filename]?.resumeData,
              let meta = downloadMeta[filename] else {
            logger.error("Cannot resume \(filename, privacy: .public) — no resume data or metadata")
            return
        }
        ensureSession()

        activeDownloads[filename]?.state = .downloading
        activeDownloads[filename]?.resumeData = nil

        let task = delegateSession!.downloadTask(withResumeData: resumeData)
        sessionTasks[filename] = task

        let expectedHash = meta.expectedSHA256

        downloadDelegate?.register(
            taskId: task.taskIdentifier,
            fileId: filename,
            destURL: meta.destURL,
            onProgress: { [weak self] fileId, written, total in
                Task { @MainActor [weak self] in
                    self?.activeDownloads[fileId]?.downloadedBytes = written
                    self?.activeDownloads[fileId]?.totalBytes = total
                }
            },
            onComplete: { [weak self] fileId, result in
                Task { @MainActor [weak self] in
                    if case .paused = self?.activeDownloads[fileId]?.state { return }
                    switch result {
                    case .success(let path):
                        if let expectedHash, !expectedHash.isEmpty {
                            self?.activeDownloads[fileId]?.state = .verifying
                            self?.verifyAndComplete(fileId: fileId, path: path, expectedHash: expectedHash)
                        } else {
                            self?.activeDownloads[fileId]?.state = .completed(localPath: path)
                        }
                    case .failure(let error):
                        if (error as NSError).code == NSURLErrorCancelled {
                            if case .paused = self?.activeDownloads[fileId]?.state { return }
                            self?.activeDownloads[fileId]?.state = .cancelled
                        } else {
                            self?.activeDownloads[fileId]?.state = .failed(String(describing: error))
                        }
                    }
                    self?.sessionTasks.removeValue(forKey: fileId)
                }
            }
        )

        logger.info("Resuming: \(filename, privacy: .public)")
        task.resume()
    }

    /// Remove a completed/failed/cancelled task from the active list.
    public func dismiss(filename: String) {
        activeDownloads.removeValue(forKey: filename)
        sessionTasks.removeValue(forKey: filename)
    }

    /// Delete a downloaded file from disk.
    /// Returns true if the file was successfully removed.
    @discardableResult
    public func deleteDownloaded(filename: String) -> Bool {
        guard let path = localPath(for: filename) else { return false }
        do {
            try FileManager.default.removeItem(atPath: path)
            removeDownloadSource(filename: filename)
            activeDownloads.removeValue(forKey: filename)
            logger.info("Deleted model: \(filename, privacy: .public)")
            return true
        } catch {
            logger.error("Failed to delete \(filename, privacy: .public): \(String(describing: error), privacy: .public)")
            return false
        }
    }

    /// Dispatch SHA256 verification off main actor, then update download state.
    private func verifyAndComplete(fileId: String, path: String, expectedHash: String) {
        Task {
            let verified = await Task.detached {
                ModelDownloadManager.verifySHA256(path: path, expected: expectedHash)
            }.value
            if verified {
                self.activeDownloads[fileId]?.state = .completed(localPath: path)
                self.logger.info("SHA256 verified: \(fileId, privacy: .public)")
            } else {
                self.activeDownloads[fileId]?.state = .failed("SHA256 mismatch — file may be corrupt")
                self.logger.error("SHA256 mismatch: \(fileId, privacy: .public)")
            }
        }
    }

    /// Verify SHA256 of a file against expected hash. Streams in 1MB chunks to avoid loading entire file.
    nonisolated static func verifySHA256(path: String, expected: String) -> Bool {
        guard let handle = FileHandle(forReadingAtPath: path) else { return false }
        defer { handle.closeFile() }
        var hasher = SHA256()
        let chunkSize = 1_048_576 // 1MB
        while true {
            let data = handle.readData(ofLength: chunkSize)
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        let computed = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return computed.lowercased() == expected.lowercased()
    }

    /// Check if a file is already downloaded locally.
    public func isDownloaded(filename: String) -> Bool {
        localPath(for: filename) != nil
    }

    /// Get the local path for a downloaded file.
    public func localPath(for filename: String) -> String? {
        let dest = downloadDirectory.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: dest.path) {
            return dest.path
        }
        // Also check HF cache
        let hfHub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        if let repos = try? FileManager.default.contentsOfDirectory(at: hfHub, includingPropertiesForKeys: nil) {
            for repo in repos where repo.lastPathComponent.hasPrefix("models--") {
                let snapshots = repo.appendingPathComponent("snapshots")
                if let snaps = try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil) {
                    for snap in snaps {
                        let candidate = snap.appendingPathComponent(filename)
                        if FileManager.default.fileExists(atPath: candidate.path) {
                            return candidate.path
                        }
                    }
                }
            }
        }
        return nil
    }

    private func sourceMapURL() -> URL {
        downloadDirectory.appendingPathComponent(Self.sourceMapFilename)
    }

    private func persistDownloadSource(filename: String, repoId: String) {
        guard !filename.isEmpty, !repoId.isEmpty else { return }
        let key = (filename as NSString).lastPathComponent.lowercased()
        var map = loadDownloadSourceMap()
        map[key] = repoId
        saveDownloadSourceMap(map)
        appendDownloadSourceLog(filename: filename, repoId: repoId)
    }

    private func removeDownloadSource(filename: String) {
        let key = (filename as NSString).lastPathComponent.lowercased()
        var map = loadDownloadSourceMap()
        map.removeValue(forKey: key)
        saveDownloadSourceMap(map)
    }

    private func loadDownloadSourceMap() -> [String: String] {
        let url = sourceMapURL()
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([String: String].self, from: data) else {
            return [:]
        }
        return decoded
    }

    private func saveDownloadSourceMap(_ map: [String: String]) {
        let fm = FileManager.default
        if !fm.fileExists(atPath: downloadDirectory.path) {
            try? fm.createDirectory(at: downloadDirectory, withIntermediateDirectories: true)
        }
        let url = sourceMapURL()
        guard let data = try? JSONEncoder().encode(map) else { return }
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            logger.error("Failed to save source map: \(String(describing: error), privacy: .public)")
        }
    }

    private func appendDownloadSourceLog(filename: String, repoId: String) {
        let fm = FileManager.default
        if !fm.fileExists(atPath: downloadDirectory.path) {
            try? fm.createDirectory(at: downloadDirectory, withIntermediateDirectories: true)
        }
        let url = downloadDirectory.appendingPathComponent(Self.sourceLogFilename)
        let entry = DownloadSourceLogEntry(
            v: Self.sourceLogVersion,
            event: "download_requested",
            timestamp: ISO8601DateFormatter().string(from: Date()),
            repoId: repoId,
            filename: filename
        )
        guard let payload = try? JSONEncoder().encode(entry),
              let newline = "\n".data(using: .utf8) else { return }
        var data = Data()
        data.append(payload)
        data.append(newline)

        if !fm.fileExists(atPath: url.path) {
            try? data.write(to: url, options: .atomic)
            return
        }

        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: data)
    }
}

private struct DownloadSourceLogEntry: Codable {
    let v: Int
    let event: String
    let timestamp: String
    let repoId: String
    let filename: String
}

// MARK: - URLSession Download Delegate

private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, Sendable {
    private struct Registration: Sendable {
        let fileId: String
        let destURL: URL
        let onProgress: @Sendable (String, Int64, Int64) -> Void
        let onComplete: @Sendable (String, Result<String, Error>) -> Void
    }

    private let registrations = OSAllocatedUnfairLock(initialState: [Int: Registration]())

    func register(
        taskId: Int,
        fileId: String,
        destURL: URL,
        onProgress: @escaping @Sendable (String, Int64, Int64) -> Void,
        onComplete: @escaping @Sendable (String, Result<String, Error>) -> Void
    ) {
        registrations.withLock {
            $0[taskId] = Registration(
                fileId: fileId, destURL: destURL,
                onProgress: onProgress, onComplete: onComplete
            )
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard let reg = registrations.withLock({ $0[downloadTask.taskIdentifier] }) else { return }
        let total = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : 0
        reg.onProgress(reg.fileId, totalBytesWritten, total)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let reg = registrations.withLock({ $0[downloadTask.taskIdentifier] }) else { return }

        do {
            let fm = FileManager.default
            if fm.fileExists(atPath: reg.destURL.path) {
                try fm.removeItem(at: reg.destURL)
            }
            try fm.moveItem(at: location, to: reg.destURL)
            reg.onComplete(reg.fileId, .success(reg.destURL.path))
        } catch {
            reg.onComplete(reg.fileId, .failure(error))
        }

        _ = registrations.withLock { $0.removeValue(forKey: downloadTask.taskIdentifier) }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let error,
              let reg = registrations.withLock({ $0[task.taskIdentifier] }) else { return }
        reg.onComplete(reg.fileId, .failure(error))
        _ = registrations.withLock { $0.removeValue(forKey: task.taskIdentifier) }
    }
}
