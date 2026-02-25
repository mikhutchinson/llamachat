import Foundation
import SwiftPythonRuntime
import OSLog

public actor LlamaSessionManager {
    private let workerPool: InferenceWorkerPool
    private let config: InferenceConfig
    private var sessions: [SessionID: SessionState] = [:]
    private var workerSessionCounts: [Int: Int] = [:]
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "session-mgr")

    // MARK: - Metrics

    public private(set) var totalRequests: Int = 0
    public private(set) var totalPrefillMs: Double = 0
    public private(set) var totalDecodeMs: Double = 0
    public private(set) var totalCompletionTokens: Int = 0

    public init(workerPool: InferenceWorkerPool, config: InferenceConfig) {
        self.workerPool = workerPool
        self.config = config
        for i in 0..<config.workerCount {
            workerSessionCounts[i] = 0
        }
    }

    // MARK: - Session Lifecycle

    public func createSession(
        systemPrompt: String? = nil
    ) async throws -> SessionID {
        let workerIndex = try selectWorker()
        let sessionID = SessionID()

        // Reserve the slot BEFORE the async call so concurrent tasks see the updated count
        workerSessionCounts[workerIndex, default: 0] += 1

        do {
            let pool = try await workerPool.getPool()
            let kernel = try await workerPool.kernelHandle(for: workerIndex)

            let result = try await LlamaSessionKernel.createSession(
                pool: pool,
                workerIndex: workerIndex,
                kernelHandle: kernel,
                sessionID: sessionID.description,
                systemPrompt: systemPrompt
            )

            guard (result["status"] as? String) == "created" || (result["status"] as? String) == "exists" else {
                workerSessionCounts[workerIndex, default: 1] -= 1
                throw InferenceError.modelLoadFailed("Failed to create session: \(result)")
            }

            var state = SessionState(id: sessionID, workerIndex: workerIndex)
            state.transitionTo(.idle)
            sessions[sessionID] = state

            logger.debug("Session \(sessionID.description, privacy: .public) created on worker \(workerIndex, privacy: .public)")
            return sessionID
        } catch {
            workerSessionCounts[workerIndex, default: 1] -= 1
            throw error
        }
    }

    public func evictSession(_ sessionID: SessionID) async throws {
        guard var state = sessions[sessionID] else {
            throw InferenceError.sessionNotFound(sessionID)
        }

        let pool = try await workerPool.getPool()
        let kernel = try await workerPool.kernelHandle(for: state.workerIndex)

        _ = try await LlamaSessionKernel.evict(
            pool: pool,
            workerIndex: state.workerIndex,
            kernelHandle: kernel,
            sessionID: sessionID.description
        )

        state.transitionTo(.evicted)
        sessions[sessionID] = state
        workerSessionCounts[state.workerIndex, default: 1] -= 1

        logger.debug("Session \(sessionID.description, privacy: .public) evicted from worker \(state.workerIndex, privacy: .public)")
    }

    // MARK: - Inference

    public func complete(
        sessionID: SessionID,
        prompt: String,
        params: SamplingParams = .default
    ) async throws -> InferenceResult {
        guard var state = sessions[sessionID] else {
            throw InferenceError.sessionNotFound(sessionID)
        }
        guard state.phase == .idle || state.phase == .completed else {
            throw InferenceError.decodeFailed(
                sessionID: sessionID,
                reason: "Session in phase \(state.phase.rawValue), expected idle or completed"
            )
        }

        let pool = try await workerPool.getPool()
        let kernel = try await workerPool.kernelHandle(for: state.workerIndex)

        // Prefill phase
        state.transitionTo(.prefilling)
        sessions[sessionID] = state

        let prefillStart = ContinuousClock.now

        let prefillResult = try await LlamaSessionKernel.prefill(
            pool: pool,
            workerIndex: state.workerIndex,
            kernelHandle: kernel,
            sessionID: sessionID.description,
            prompt: prompt,
            params: params
        )

        if let error = prefillResult["error"] as? String {
            state.transitionTo(.failed)
            sessions[sessionID] = state
            throw InferenceError.prefillFailed(sessionID: sessionID, reason: error)
        }

        let prefillEnd = ContinuousClock.now
        let prefillDuration = prefillEnd - prefillStart

        // Decode phase
        state.transitionTo(.decoding)
        sessions[sessionID] = state

        let decodeStart = ContinuousClock.now

        let decodeResult = try await LlamaSessionKernel.decode(
            pool: pool,
            workerIndex: state.workerIndex,
            kernelHandle: kernel,
            sessionID: sessionID.description,
            params: params
        )

        if let error = decodeResult["error"] as? String {
            state.transitionTo(.failed)
            sessions[sessionID] = state
            throw InferenceError.decodeFailed(sessionID: sessionID, reason: error)
        }

        let decodeEnd = ContinuousClock.now
        let decodeDuration = decodeEnd - decodeStart
        let totalDuration = decodeEnd - prefillStart

        let promptTokens = (decodeResult["prompt_tokens"] as? Int) ?? (prefillResult["prompt_tokens"] as? Int) ?? 0
        state.recordPrefill(promptTokens: promptTokens)

        let text = (decodeResult["text"] as? String) ?? ""
        let completionTokens = (decodeResult["completion_tokens"] as? Int) ?? 0
        let finishReason = (decodeResult["finish_reason"] as? String) ?? "unknown"

        state.recordDecodeStep(newText: text, tokens: completionTokens, finishReason: finishReason)
        state.transitionTo(.completed)
        sessions[sessionID] = state

        // Update metrics (JSON numbers come back as NSNumber; use .doubleValue so prefill_ms/decode_ms are read correctly)
        totalRequests += 1
        let pythonPrefillMs = (decodeResult["prefill_ms"] as? NSNumber)?.doubleValue ?? (prefillResult["prefill_ms"] as? NSNumber)?.doubleValue ?? 0
        let prefillMsSwift = Double(prefillDuration.components.seconds) * 1000 + Double(prefillDuration.components.attoseconds) / 1e15
        totalPrefillMs += pythonPrefillMs > 0 ? pythonPrefillMs : prefillMsSwift
        totalDecodeMs += (decodeResult["decode_ms"] as? NSNumber)?.doubleValue ?? (decodeResult["decode_ms"] as? Double) ?? 0
        totalCompletionTokens += completionTokens

        logger.debug("Session \(sessionID.description, privacy: .public) complete: \(promptTokens, privacy: .public) prompt + \(completionTokens, privacy: .public) completion, finish=\(finishReason, privacy: .public)")

        return InferenceResult(
            sessionID: sessionID,
            text: text,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            finishReason: finishReason,
            workerIndex: state.workerIndex,
            prefillDuration: prefillDuration,
            decodeDuration: decodeDuration,
            totalDuration: totalDuration
        )
    }

    public func completeOneShot(
        prompt: String,
        params: SamplingParams = .default,
        systemPrompt: String? = nil
    ) async throws -> InferenceResult {
        let sessionID = try await createSession(systemPrompt: systemPrompt)
        defer {
            Task { [sessionID] in
                try? await self.evictSession(sessionID)
            }
        }
        return try await complete(sessionID: sessionID, prompt: prompt, params: params)
    }

    // MARK: - Query

    public func sessionState(for id: SessionID) -> SessionState? {
        sessions[id]
    }

    public var activeSessions: [SessionState] {
        sessions.values.filter { $0.phase != .evicted && $0.phase != .failed }
    }

    public var workerLoad: [Int: Int] {
        workerSessionCounts
    }

    public var aggregateStats: AggregateStats {
        AggregateStats(
            totalRequests: totalRequests,
            totalPrefillMs: totalPrefillMs,
            totalDecodeMs: totalDecodeMs,
            totalCompletionTokens: totalCompletionTokens,
            activeSessions: activeSessions.count,
            workerLoad: workerSessionCounts
        )
    }

    // MARK: - Worker Selection

    private func selectWorker() throws -> Int {
        var bestWorker = 0
        var bestCount = Int.max

        for i in 0..<config.workerCount {
            let count = workerSessionCounts[i, default: 0]
            if count < bestCount {
                bestCount = count
                bestWorker = i
            }
        }

        if bestCount >= config.maxSessionsPerWorker {
            throw InferenceError.workerFull(workerIndex: bestWorker)
        }

        return bestWorker
    }
}

// MARK: - Aggregate Stats

public struct AggregateStats: Sendable {
    public let totalRequests: Int
    public let totalPrefillMs: Double
    public let totalDecodeMs: Double
    public let totalCompletionTokens: Int
    public let activeSessions: Int
    public let workerLoad: [Int: Int]

    public var avgPrefillMs: Double {
        totalRequests > 0 ? totalPrefillMs / Double(totalRequests) : 0
    }

    public var avgDecodeMs: Double {
        totalRequests > 0 ? totalDecodeMs / Double(totalRequests) : 0
    }

    public var avgTokensPerRequest: Double {
        totalRequests > 0 ? Double(totalCompletionTokens) / Double(totalRequests) : 0
    }
}
