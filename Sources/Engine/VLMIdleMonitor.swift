import Foundation
import SwiftPythonRuntime
import OSLog

/// Manages VLM lifecycle: lazy loading on first image, idle timeout to free VRAM.
///
/// The VLM model is only loaded when `ensureLoaded()` is called (first image caption).
/// After `idleTimeoutSecs` of no use, the model is automatically unloaded to free VRAM.
public actor VLMIdleMonitor {
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "vlm-idle")

    private var pool: PythonProcessPool?
    private var kernelHandle: PyHandle?
    private var workerIndex: Int = 0
    private var modelPath: String = ""
    private var clipPath: String = ""
    private var contextSize: Int = 2048
    private var nGpuLayers: Int = -1
    private var idleTimeoutSecs: TimeInterval = 300

    public enum State: String, Sendable {
        case idle
        case loading
        case ready
        case unloading
        case failed
    }

    public private(set) var state: State = .idle
    private var lastUsed: Date = .distantPast
    private var idleCheckTask: Task<Void, Never>?

    public init() {}

    // MARK: - Lifecycle

    /// Configure the monitor with pool and settings. Does NOT load the model.
    public func configure(
        pool: PythonProcessPool,
        kernelHandle: PyHandle,
        workerIndex: Int,
        modelPath: String,
        clipPath: String,
        contextSize: Int = 2048,
        nGpuLayers: Int = -1,
        idleTimeoutSecs: TimeInterval = 300
    ) {
        self.pool = pool
        self.kernelHandle = kernelHandle
        self.workerIndex = workerIndex
        self.modelPath = modelPath
        self.clipPath = clipPath
        self.contextSize = contextSize
        self.nGpuLayers = nGpuLayers
        self.idleTimeoutSecs = idleTimeoutSecs
        logger.debug("VLMIdleMonitor configured: model=\(modelPath, privacy: .public) timeout=\(idleTimeoutSecs, privacy: .public)s")
    }

    /// True if both model and clip paths are set.
    public var isConfigured: Bool {
        !modelPath.isEmpty && !clipPath.isEmpty && pool != nil && kernelHandle != nil
    }

    /// Ensure VLM is loaded. Lazy-loads on first call. Returns true if ready.
    public func ensureLoaded() async -> Bool {
        if state == .ready {
            lastUsed = Date()
            return true
        }

        // Allow retry after previous failure
        if state == .failed {
            logger.debug("VLM was in failed state, resetting for retry")
            state = .idle
        }

        guard self.state == .idle || self.state == .loading else {
            logger.warning("VLM in unexpected state \(self.state.rawValue, privacy: .public), cannot load")
            return false
        }

        guard isConfigured, let pool, let handle = kernelHandle else {
            logger.warning("VLM not configured — cannot load")
            return false
        }

        state = .loading
        logger.debug("Loading VLM model...")

        do {
            let result = try await VLMKernel.load(
                pool: pool, workerIndex: workerIndex, kernelHandle: handle,
                modelPath: modelPath, clipPath: clipPath,
                contextSize: contextSize, nGpuLayers: nGpuLayers
            )

            if let error = result["error"] as? String {
                logger.error("VLM load failed: \(error, privacy: .public)")
                state = .failed
                return false
            }

            let durationMs = (result["duration_ms"] as? Double) ?? 0
            logger.debug("VLM loaded in \(durationMs, privacy: .public)ms")

            // Verify worker is still alive after load
            do {
                let status = try await VLMKernel.status(
                    pool: pool, workerIndex: workerIndex, kernelHandle: handle
                )
                let loaded = (status["loaded"] as? Bool) ?? false
                logger.debug("VLM post-load verify: loaded=\(loaded, privacy: .public) worker=\(self.workerIndex, privacy: .public)")
                guard loaded else {
                    logger.error("VLM post-load verify: model not loaded despite successful load() call")
                    state = .failed
                    return false
                }
            } catch {
                logger.error("VLM worker died after load: \(String(describing: error), privacy: .public)")
                state = .failed
                return false
            }

            state = .ready
            lastUsed = Date()
            startIdleCheck()
            return true
        } catch {
            logger.error("VLM load error: \(String(describing: error), privacy: .public)")
            state = .failed
            return false
        }
    }

    /// Unload VLM model to free VRAM.
    public func unload() async {
        guard state == .ready, let pool, let handle = kernelHandle else { return }
        state = .unloading
        idleCheckTask?.cancel()
        idleCheckTask = nil

        do {
            _ = try await VLMKernel.unload(
                pool: pool, workerIndex: workerIndex, kernelHandle: handle
            )
            logger.debug("VLM unloaded (idle timeout)")
        } catch {
            logger.warning("VLM unload error: \(String(describing: error), privacy: .public)")
        }
        state = .idle
    }

    /// Record that VLM was just used (resets idle timer).
    public func markUsed() {
        lastUsed = Date()
    }

    /// Shutdown — cancel idle check and unload.
    public func shutdown() async {
        idleCheckTask?.cancel()
        idleCheckTask = nil
        await unload()
    }

    // MARK: - Idle Check

    private func startIdleCheck() {
        idleCheckTask?.cancel()
        idleCheckTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 30_000_000_000)
                guard !Task.isCancelled else { return }
                guard let self else { return }
                let idle = await self.checkIdle()
                if idle { return }
            }
        }
    }

    private func checkIdle() async -> Bool {
        guard state == .ready else { return false }
        let elapsed = Date().timeIntervalSince(lastUsed)
        if elapsed >= idleTimeoutSecs {
            logger.debug("VLM idle for \(elapsed, privacy: .public)s, unloading...")
            await unload()
            return true
        }
        return false
    }
}
