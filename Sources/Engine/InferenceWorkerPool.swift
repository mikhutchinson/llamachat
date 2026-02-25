import Foundation
import SwiftPythonRuntime
import OSLog

public actor InferenceWorkerPool {
    private let config: InferenceConfig
    private var pool: PythonProcessPool?
    private var kernelHandles: [Int: PyHandle] = [:]
    private var summarizerHandle: PyHandle?
    private var summarizerWorkerIndex: Int?
    private var docExtractorHandles: [Int: PyHandle] = [:]
    private var nextDocExtractorWorker: Int = 0
    private var docIngestionHandle: PyHandle?
    private var vlmKernelHandle: PyHandle?
    private var vlmWorkerIndex: Int?
    private var sandboxWorkerIdx: Int?
    private var shmResultBuffers: [String: PyHandle] = [:]
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "pool")

    public enum PoolState: String, Sendable {
        case idle
        case starting
        case ready
        case shuttingDown
        case shutdown
    }

    public private(set) var state: PoolState = .idle

    private var useDedicatedSummarizer: Bool {
        (config.summarizerModelPath ?? "").isEmpty == false
    }

    private var workerExecutablePath: String {
        if let explicit = config.workerExecutablePath {
            return explicit
        }
        if let envPath = ProcessInfo.processInfo.environment["SWIFTPYTHON_WORKER_PATH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !envPath.isEmpty {
            return envPath
        }
        if let bundledPath = WorkerRuntimeEnvironment.bundledWorkerPath() {
            return bundledPath
        }
        let mainPath = CommandLine.arguments[0]
        let mainDir = (mainPath as NSString).deletingLastPathComponent
        return (mainDir as NSString).appendingPathComponent("SwiftPythonWorker")
    }

    public init(config: InferenceConfig) {
        self.config = config
    }

    // MARK: - Lifecycle

    public func startup() async throws {
        guard state == .idle else { return }
        state = .starting
        WorkerRuntimeEnvironment.configureForBundledWorker()

        let resourceLimits: WorkerResourceLimits
        if let maxMem = config.maxMemoryBytesPerWorker {
            resourceLimits = WorkerResourceLimits(maxMemoryBytes: maxMem)
        } else {
            resourceLimits = .default
        }

        let backpressure: BackpressurePolicy
        if let maxInFlight = config.maxInFlight {
            backpressure = .suspend(maxInFlight: maxInFlight)
        } else {
            backpressure = .unbounded
        }

        let totalWorkers = config.workerCount
            + (useDedicatedSummarizer ? 1 : 0)
            + 1 // dedicated VLM worker (model loaded lazily)
            + 1 // dedicated sandbox worker for code execution
        let created = try await PythonProcessPool(
            workers: totalWorkers,
            workerExecutablePath: self.workerExecutablePath,
            blasThreads: config.blasThreads,
            resourceLimits: resourceLimits,
            backpressure: backpressure
        )
        pool = created

        var warmupParts: [String] = []
        let venvToUse = config.venvPath ?? ProcessInfo.processInfo.environment["VIRTUAL_ENV"]
        if let venv = venvToUse {
            let fm = FileManager.default
            let libDir = "\(venv)/lib"
            var candidateSitePackages: [String] = []
            if let entries = try? fm.contentsOfDirectory(atPath: libDir) {
                for entry in entries where entry.hasPrefix("python3.") {
                    let sp = "\(libDir)/\(entry)/site-packages"
                    var isDir: ObjCBool = false
                    if fm.fileExists(atPath: sp, isDirectory: &isDir), isDir.boolValue {
                        candidateSitePackages.append(sp)
                    }
                }
            }
            candidateSitePackages.sort()
            if candidateSitePackages.isEmpty {
                let fallback = "\(venv)/lib/python3/site-packages"
                candidateSitePackages.append(fallback)
            }
            let pythonList = "[" + candidateSitePackages.map { "'\($0)'" }.joined(separator: ",") + "]"
            warmupParts.append("import sys; __sps__ = \(pythonList); [sys.path.insert(0, sp) for sp in __sps__ if sp not in sys.path]")
            logger.debug("Injecting venv site-packages: \(candidateSitePackages.joined(separator: ","), privacy: .public)")
        }

        warmupParts.append(#"""
import sys
import os
# Suppress noisy llama.cpp Metal initialization warnings
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = os.dup(2)
        os.close(2)
        os.open(os.devnull, os.O_WRONLY)
        return self
    def __exit__(self, *args):
        os.close(2)
        os.dup(self.original_stderr)
        os.close(self.original_stderr)

try:
    with SuppressStderr():
        import llama_cpp as __llama_cpp__
except Exception as __e__:
    raise ImportError(
        "llama_cpp import failed in worker. "
        + f"python={sys.version} "
        + f"sys.path={sys.path} "
        + f"error={repr(__e__)}"
    )
"""#)
        // Suppress noisy llama.cpp Metal initialization warnings
        let originalStderr = dup(STDERR_FILENO)
        let devNull = open("/dev/null", O_WRONLY)
        if devNull >= 0 {
            dup2(devNull, STDERR_FILENO)
            close(devNull)
        }
        defer {
            dup2(originalStderr, STDERR_FILENO)
            close(originalStderr)
        }
        
        try await created.warmup(warmupParts.joined(separator: "\n"))

        // Also suppress during kernel installation (model loading)
        try await installKernels()

        state = .ready
        logger.debug("InferenceWorkerPool ready: \(self.config.workerCount, privacy: .public) workers, model=\(self.config.modelPath, privacy: .public)")
        await FileLogger.shared.log(level: .debug, category: "Pool", message: "Ready: \(self.config.workerCount) workers, model=\(self.config.modelPath)")
    }

    public func shutdown() async {
        guard state == .ready || state == .starting else { return }
        state = .shuttingDown

        for (_, handle) in shmResultBuffers {
            try? await pool?.release(handle)
        }
        shmResultBuffers.removeAll()

        for (_, handle) in kernelHandles {
            try? await pool?.release(handle)
        }
        kernelHandles.removeAll()

        if let sumHandle = summarizerHandle {
            try? await pool?.release(sumHandle)
            summarizerHandle = nil
            summarizerWorkerIndex = nil
        }

        for (_, handle) in docExtractorHandles {
            try? await pool?.release(handle)
        }
        docExtractorHandles.removeAll()

        if let ingestHandle = docIngestionHandle {
            try? await pool?.release(ingestHandle)
            docIngestionHandle = nil
        }

        if let vlmHandle = vlmKernelHandle {
            try? await pool?.release(vlmHandle)
            vlmKernelHandle = nil
        }

        await pool?.shutdown()
        pool = nil
        state = .shutdown
        logger.debug("InferenceWorkerPool shut down")
        await FileLogger.shared.log(level: .debug, category: "Pool", message: "Shut down")
    }

    // MARK: - Access

    public func getPool() throws -> PythonProcessPool {
        guard let pool, state == .ready else {
            throw InferenceError.poolNotReady
        }
        return pool
    }

    public func kernelHandle(for workerIndex: Int) throws -> PyHandle {
        guard let handle = kernelHandles[workerIndex] else {
            throw InferenceError.poolNotReady
        }
        return handle
    }

    public var workerCount: Int { config.workerCount }

    public func healthCheck() async throws -> [Bool] {
        guard let pool else { return [] }
        return try await pool.healthCheck()
    }

    /// Worker process PID for diagnostics (e.g. proving generator vs reviewer use different workers)
    public func workerPID(for workerIndex: Int) async -> pid_t? {
        guard let pool else { return nil }
        return await pool.workerPID(for: workerIndex)
    }

    // MARK: - Shared Memory Result Buffers

    public func getOrCreateResultBuffer(for sessionID: String) async throws -> PyHandle {
        if let existing = shmResultBuffers[sessionID] {
            return existing
        }
        guard let pool else { throw InferenceError.poolNotReady }
        let handle = try await pool.createSharedTensor(
            shape: [config.sharedMemorySlotSize],
            dtype: .uint8
        )
        shmResultBuffers[sessionID] = handle
        logger.debug("Created shared result buffer for \(sessionID, privacy: .public)")
        return handle
    }

    public func releaseResultBuffer(for sessionID: String) async {
        guard let handle = shmResultBuffers.removeValue(forKey: sessionID) else { return }
        try? await pool?.release(handle)
    }

    public var useSharedMemory: Bool { config.useSharedMemory }

    // MARK: - Private

    private func installKernels() async throws {
        guard let pool else { return }

        try await withThrowingTaskGroup(of: (Int, PyHandle).self) { group in
            for i in 0..<config.workerCount {
                let worker = pool.worker(i)
                group.addTask {
                    let handle = try await LlamaSessionKernel.install(
                        on: worker,
                        config: self.config,
                        workerIndex: i
                    )
                    return (i, handle)
                }
            }

            for try await (workerIdx, handle) in group {
                kernelHandles[workerIdx] = handle
            }
        }

        logger.debug("Installed \(self.kernelHandles.count, privacy: .public) session kernels")

        // Install summarizer: dedicated worker when summarizerModelPath set, else shared on worker 0
        let useDedSum = useDedicatedSummarizer
        if useDedSum {
            let sumIdx = config.workerCount
            let sumHandle = try await SummarizationKernel.installDedicated(
                on: pool.worker(sumIdx),
                config: config,
                workerIndex: sumIdx
            )
            summarizerHandle = sumHandle
            summarizerWorkerIndex = sumIdx
            logger.debug("Installed SummarizationKernel on dedicated worker \(sumIdx, privacy: .public)")
        } else if let kernelHandle0 = kernelHandles[0] {
            let sumHandle = try await SummarizationKernel.installShared(
                on: pool.worker(0),
                llamaKernelHandle: kernelHandle0,
                config: config
            )
            summarizerHandle = sumHandle
            summarizerWorkerIndex = 0
            logger.debug("Installed SummarizationKernel on worker 0 (shared model)")
        }

        // Install document extractor on ALL inference workers (stateless, pure CPU)
        await withTaskGroup(of: (Int, PyHandle?).self) { group in
            for i in 0..<config.workerCount {
                let worker = pool.worker(i)
                group.addTask {
                    do {
                        let handle = try await DocumentExtractor.install(on: worker)
                        return (i, handle)
                    } catch {
                        return (i, nil)
                    }
                }
            }
            for await (workerIdx, handle) in group {
                if let handle {
                    docExtractorHandles[workerIdx] = handle
                } else {
                    logger.warning("DocumentExtractor install failed on worker \(workerIdx, privacy: .public) (non-fatal)")
                }
            }
        }
        if !docExtractorHandles.isEmpty {
            logger.debug("Installed DocumentExtractor on \(self.docExtractorHandles.count, privacy: .public) workers")
        }

        // Install document ingestion kernel on worker 0 (MiniLM loaded lazily on first embed)
        do {
            let ingestHandle = try await DocumentIngestionKernel.install(on: pool.worker(0))
            docIngestionHandle = ingestHandle
            logger.debug("Installed DocumentIngestionKernel on worker 0")
        } catch {
            let desc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
            logger.warning("DocumentIngestionKernel install failed (non-fatal): \(desc, privacy: .public)")
        }

        // Install VLM kernel on dedicated worker (model NOT loaded — lazy load on first image)
        let vlmIdx = config.workerCount
            + (useDedSum ? 1 : 0)
        do {
            let vlmHandle = try await VLMKernel.install(on: pool.worker(vlmIdx))
            vlmKernelHandle = vlmHandle
            vlmWorkerIndex = vlmIdx
            logger.debug("Installed VLMKernel on dedicated worker \(vlmIdx, privacy: .public) (model not loaded yet)")
        } catch {
            let desc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
            logger.warning("VLMKernel install failed (non-fatal): \(desc, privacy: .public)")
        }

        // Record sandbox worker index (no kernel to install — PythonSandbox handles init lazily)
        let sbIdx = vlmIdx + 1
        sandboxWorkerIdx = sbIdx
        logger.debug("Sandbox worker reserved at index \(sbIdx, privacy: .public)")
    }

    public func getSummarizerHandle() throws -> (handle: PyHandle, workerIndex: Int) {
        guard let handle = summarizerHandle,
              let idx = summarizerWorkerIndex,
              pool != nil,
              state == .ready else {
            throw InferenceError.poolNotReady
        }
        return (handle, idx)
    }

    public func getDocIngestionHandle() throws -> (handle: PyHandle, workerIndex: Int) {
        guard let handle = docIngestionHandle,
              pool != nil,
              state == .ready else {
            throw InferenceError.poolNotReady
        }
        return (handle, 0)
    }

    public func getDocExtractorHandle() throws -> (handle: PyHandle, workerIndex: Int) {
        guard !docExtractorHandles.isEmpty,
              pool != nil,
              state == .ready else {
            throw InferenceError.poolNotReady
        }
        // Round-robin across workers with installed extractors
        let workerIndices = docExtractorHandles.keys.sorted()
        let idx = nextDocExtractorWorker % workerIndices.count
        let workerIndex = workerIndices[idx]
        nextDocExtractorWorker = idx + 1
        return (docExtractorHandles[workerIndex]!, workerIndex)
    }

    /// Count tokens using the model's actual tokenizer (worker 0).
    public func countTokens(_ text: String) async throws -> Int {
        guard let pool, state == .ready else {
            throw InferenceError.poolNotReady
        }
        let kernel = try kernelHandle(for: 0)
        return try await LlamaSessionKernel.countTokens(
            pool: pool,
            workerIndex: 0,
            kernelHandle: kernel,
            text: text
        )
    }

    public func getVLMKernelHandle() throws -> (handle: PyHandle, workerIndex: Int) {
        guard let handle = vlmKernelHandle,
              let idx = vlmWorkerIndex,
              pool != nil,
              state == .ready else {
            throw InferenceError.poolNotReady
        }
        return (handle, idx)
    }

    /// Returns the sandbox worker index for code execution, or nil when pool is not ready.
    public func getSandboxWorkerIndex() -> Int? {
        guard state == .ready, pool != nil else { return nil }
        return sandboxWorkerIdx
    }

}
