import Foundation
import SwiftPythonRuntime
import OSLog

/// DAG-based inference scheduler with dual-queue prioritization.
///
/// Collects incoming inference requests and batches them into `ProcessPoolDAG`
/// executions with proper prefill→decode dependency chains, worker affinity,
/// and token budget enforcement.
public actor InferenceScheduler {
    private let workerPool: InferenceWorkerPool
    private let config: InferenceConfig
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "scheduler")
    public let contextMonitor: ContextWindMonitor
    public private(set) var narrativeMemory: NarrativeMemoryWorker?

    // MARK: - Session Registry

    private var sessions: [SessionID: ScheduledSession] = [:]
    private var workerSessionCounts: [Int: Int] = [:]

    // MARK: - Queues

    private var prefillQueue: [PendingRequest] = []
    private var decodeQueue: [PendingRequest] = []

    // MARK: - Metrics

    public private(set) var totalScheduled: Int = 0
    public private(set) var totalCompleted: Int = 0
    public private(set) var totalFailed: Int = 0
    public private(set) var totalTokensGenerated: Int = 0
    public private(set) var totalPrefillMs: Double = 0
    public private(set) var totalDecodeMs: Double = 0

    // MARK: - Types

    public struct ScheduledSession: Sendable {
        public let id: SessionID
        public let workerIndex: Int
        public var tokenBudgetUsed: Int = 0
        public var phase: SessionPhase = .idle
        public var createdAt: Date = Date()
        public var lastActivity: Date = Date()
    }

    struct PendingRequest: Sendable {
        let sessionID: SessionID
        let workerIndex: Int
        let kernelHandle: PyHandle
        let prompt: String?
        let params: SamplingParams
        let stage: RequestStage
        let submittedAt: ContinuousClock.Instant

        enum RequestStage: Sendable {
            case prefill
            case decode
            case complete
        }
    }

    // MARK: - Init

    public init(workerPool: InferenceWorkerPool, config: InferenceConfig) {
        self.workerPool = workerPool
        self.config = config
        self.contextMonitor = ContextWindMonitor(contextSize: config.contextSize)
        for i in 0..<config.workerCount {
            workerSessionCounts[i] = 0
        }
    }

    // MARK: - Session Lifecycle

    public func createSession(systemPrompt: String? = nil) async throws -> SessionID {
        let workerIndex = try selectWorker()
        let sessionID = SessionID()

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

            sessions[sessionID] = ScheduledSession(id: sessionID, workerIndex: workerIndex)
            await contextMonitor.registerSession(sessionID)
            logger.debug("Session \(sessionID.description, privacy: .public) created on W\(workerIndex, privacy: .public)")
            await FileLogger.shared.log(level: .debug, category: "Scheduler", message: "Session \(sessionID.description) created on worker \(workerIndex)")
            return sessionID
        } catch {
            workerSessionCounts[workerIndex, default: 1] -= 1
            throw error
        }
    }

    public func evictSession(_ sessionID: SessionID) async throws {
        guard let session = sessions[sessionID] else {
            throw InferenceError.sessionNotFound(sessionID)
        }

        let pool = try await workerPool.getPool()
        let kernel = try await workerPool.kernelHandle(for: session.workerIndex)

        _ = try await LlamaSessionKernel.evict(
            pool: pool,
            workerIndex: session.workerIndex,
            kernelHandle: kernel,
            sessionID: sessionID.description
        )

        await workerPool.releaseResultBuffer(for: sessionID.description)
        await contextMonitor.unregisterSession(sessionID)

        sessions[sessionID] = nil
        workerSessionCounts[session.workerIndex, default: 1] -= 1
        logger.debug("Session \(sessionID.description, privacy: .public) evicted from W\(session.workerIndex, privacy: .public)")
        await FileLogger.shared.log(level: .debug, category: "Scheduler", message: "Session \(sessionID.description) evicted from worker \(session.workerIndex)")
    }

    // MARK: - Single Request (uses DAG internally)

    public func complete(
        sessionID: SessionID,
        prompt: String,
        params: SamplingParams = .default
    ) async throws -> InferenceResult {
        guard var session = sessions[sessionID] else {
            throw InferenceError.sessionNotFound(sessionID)
        }
        guard session.phase == .idle || session.phase == .completed else {
            throw InferenceError.decodeFailed(
                sessionID: sessionID,
                reason: "Session in phase \(session.phase.rawValue), expected idle or completed"
            )
        }

        let tokenBudgetRemaining = config.contextSize - session.tokenBudgetUsed
        if session.tokenBudgetUsed >= config.contextSize {
            throw InferenceError.contextOverflow(
                sessionID: sessionID,
                used: session.tokenBudgetUsed,
                max: config.contextSize
            )
        }
        // Only reject when prompt alone wouldn't fit. Do NOT reject on tokenBudgetRemaining < maxTokens:
        // maxTokens is a ceiling; kernel will stop at context limit. Rejecting when maxTokens > contextSize
        // would block valid requests (e.g. contextSize=2048, maxTokens=4096).
        let promptTokens = await tokenCountForBudget(prompt)
        if promptTokens > tokenBudgetRemaining {
            throw InferenceError.contextOverflow(
                sessionID: sessionID,
                used: session.tokenBudgetUsed,
                max: config.contextSize
            )
        }

        if config.useSharedMemory {
            return try await completeViaShm(sessionID: sessionID, session: &session, prompt: prompt, params: params)
        } else {
            return try await completeViaPickle(sessionID: sessionID, session: &session, prompt: prompt, params: params)
        }
    }

    /// Start a streamed decode for an existing session.
    ///
    /// This path performs prefill first, then returns a `CancellableStream` of
    /// decode events (`delta`, `done`, `error`) emitted by `decode_stream`.
    /// The caller must invoke one of the `finalize*Stream` methods when stream
    /// consumption ends so scheduler metrics/session phase stay consistent.
    public func completeStream(
        sessionID: SessionID,
        prompt: String,
        params: SamplingParams = .default
    ) async throws -> CancellableStream<StreamInferenceChunk> {
        guard var session = sessions[sessionID] else {
            throw InferenceError.sessionNotFound(sessionID)
        }
        guard session.phase == .idle || session.phase == .completed else {
            throw InferenceError.decodeFailed(
                sessionID: sessionID,
                reason: "Session in phase \(session.phase.rawValue), expected idle or completed"
            )
        }

        let tokenBudgetRemaining = config.contextSize - session.tokenBudgetUsed
        if session.tokenBudgetUsed >= config.contextSize {
            throw InferenceError.contextOverflow(
                sessionID: sessionID,
                used: session.tokenBudgetUsed,
                max: config.contextSize
            )
        }
        let promptTokens = await tokenCountForBudget(prompt)
        if promptTokens > tokenBudgetRemaining {
            throw InferenceError.contextOverflow(
                sessionID: sessionID,
                used: session.tokenBudgetUsed,
                max: config.contextSize
            )
        }

        let pool = try await workerPool.getPool()
        let workerIdx = session.workerIndex
        let kernel = try await workerPool.kernelHandle(for: workerIdx)

        session.phase = .prefilling
        session.lastActivity = Date()
        sessions[sessionID] = session
        totalScheduled += 1

        let prefillResult: [String: Any]
        do {
            prefillResult = try await LlamaSessionKernel.prefill(
                pool: pool,
                workerIndex: workerIdx,
                kernelHandle: kernel,
                sessionID: sessionID.description,
                prompt: prompt,
                params: params
            )
        } catch {
            session.phase = .failed
            sessions[sessionID] = session
            totalFailed += 1
            throw error
        }

        if let error = prefillResult["error"] as? String {
            session.phase = .failed
            sessions[sessionID] = session
            totalFailed += 1
            throw InferenceError.prefillFailed(sessionID: sessionID, reason: error)
        }

        let prefillMs = (prefillResult["prefill_ms"] as? NSNumber)?.doubleValue
            ?? (prefillResult["prefill_ms"] as? Double)
            ?? 0
        totalPrefillMs += prefillMs

        session.phase = .decoding
        session.lastActivity = Date()
        sessions[sessionID] = session

        do {
            let kernelStream = try await LlamaSessionKernel.decodeStream(
                pool: pool,
                workerIndex: workerIdx,
                kernelHandle: kernel,
                sessionID: sessionID.description,
                params: params
            )
            return Self.mapKernelStreamToPublic(kernelStream)
        } catch {
            session.phase = .failed
            sessions[sessionID] = session
            totalFailed += 1
            throw error
        }
    }

    /// Finalize scheduler accounting after a streamed decode completed normally.
    public func finalizeCompletedStream(
        sessionID: SessionID,
        promptTokens: Int,
        completionTokens: Int,
        decodeMs: Double,
        finishReason: String
    ) async {
        guard var session = sessions[sessionID] else { return }
        session.tokenBudgetUsed = min(config.contextSize, max(0, promptTokens + completionTokens))
        session.phase = .completed
        session.lastActivity = Date()
        sessions[sessionID] = session

        totalCompleted += 1
        totalTokensGenerated += completionTokens
        totalDecodeMs += decodeMs

        await contextMonitor.recordTokenUsage(
            sessionID: sessionID,
            promptTokens: promptTokens,
            completionTokens: completionTokens
        )

        logger.debug(
            "Session \(sessionID.description, privacy: .public) stream complete: finish=\(finishReason, privacy: .public) \(promptTokens, privacy: .public)p+\(completionTokens, privacy: .public)c"
        )
    }

    /// Finalize scheduler state after a user-cancelled stream.
    public func finalizeCancelledStream(sessionID: SessionID) {
        guard var session = sessions[sessionID] else { return }
        session.phase = .completed
        session.lastActivity = Date()
        sessions[sessionID] = session
        logger.debug("Session \(sessionID.description, privacy: .public) stream cancelled")
    }

    /// Finalize scheduler state after a failed stream.
    public func finalizeFailedStream(sessionID: SessionID, reason: String) async {
        guard var session = sessions[sessionID] else { return }
        session.phase = .failed
        session.lastActivity = Date()
        sessions[sessionID] = session
        totalFailed += 1
        logger.error("Session \(sessionID.description, privacy: .public) stream failed: \(reason, privacy: .public)")
        await FileLogger.shared.log(level: .error, category: "Scheduler", message: "Stream failed \(sessionID.description): \(reason)")
    }

    // MARK: - Pickle Path (original)

    private func completeViaPickle(
        sessionID: SessionID,
        session: inout ScheduledSession,
        prompt: String,
        params: SamplingParams
    ) async throws -> InferenceResult {
        let pool = try await workerPool.getPool()
        let kernel = try await workerPool.kernelHandle(for: session.workerIndex)

        session.phase = .prefilling
        session.lastActivity = Date()
        sessions[sessionID] = session

        totalScheduled += 1
        let overallStart = ContinuousClock.now

        typealias DAG = ProcessPoolDAG<String, String>

        let prefillNodeID = "prefill-\(sessionID.description)"
        let decodeNodeID = "decode-\(sessionID.description)"
        let workerIdx = session.workerIndex

        let dag = DAG(nodes: [
            DAG.Node(
                id: prefillNodeID,
                preferredWorker: workerIdx
            ) { ctx in
                let json: String = try await ctx.pool.methodResult(
                    handle: kernel,
                    name: "prefill",
                    args: [
                        .python(sessionID.description),
                        .python(prompt),
                        .python("{}"),
                    ],
                    worker: ctx.workerIndex,
                    timeout: 300
                )
                return json
            },
            DAG.Node(
                id: decodeNodeID,
                dependencies: [prefillNodeID],
                preferredWorker: workerIdx
            ) { ctx in
                let json: String = try await ctx.pool.methodResult(
                    handle: kernel,
                    name: "decode",
                    args: [.python(sessionID.description)],
                    kwargs: [
                        "max_tokens": .python(params.maxTokens),
                        "temperature": .python(params.temperature),
                        "top_p": .python(params.topP),
                        "top_k": .python(params.topK),
                        "repeat_penalty": .python(params.repeatPenalty),
                    ],
                    worker: ctx.workerIndex,
                    timeout: 300
                )
                return json
            },
        ])

        let results: [String: String]
        do {
            results = try await pool.run(dag)
        } catch {
            session.phase = .failed
            sessions[sessionID] = session
            totalFailed += 1
            throw error
        }
        let overallEnd = ContinuousClock.now

        guard let prefillJSON = results[prefillNodeID],
              let decodeJSON = results[decodeNodeID] else {
            session.phase = .failed
            sessions[sessionID] = session
            totalFailed += 1
            throw InferenceError.decodeFailed(sessionID: sessionID, reason: "DAG returned incomplete results")
        }

        let prefillResult = try LlamaSessionKernel.parseJSON(prefillJSON)
        let decodeResult = try LlamaSessionKernel.parseJSON(decodeJSON)

        if let error = prefillResult["error"] as? String {
            session.phase = .failed
            sessions[sessionID] = session
            totalFailed += 1
            throw InferenceError.prefillFailed(sessionID: sessionID, reason: error)
        }
        if let error = decodeResult["error"] as? String {
            session.phase = .failed
            sessions[sessionID] = session
            totalFailed += 1
            throw InferenceError.decodeFailed(sessionID: sessionID, reason: error)
        }

        let text = (decodeResult["text"] as? String) ?? ""
        let thinking = decodeResult["thinking"] as? String
        let promptTokens = (decodeResult["prompt_tokens"] as? Int) ?? 0
        let completionTokens = (decodeResult["completion_tokens"] as? Int) ?? 0
        let finishReason = (decodeResult["finish_reason"] as? String) ?? "unknown"
        let prefillMs = (prefillResult["prefill_ms"] as? Double) ?? 0
        let decodeMs = (decodeResult["decode_ms"] as? Double) ?? 0

        session.tokenBudgetUsed = min(config.contextSize, max(0, promptTokens + completionTokens))
        session.phase = .completed
        session.lastActivity = Date()
        sessions[sessionID] = session

        totalCompleted += 1
        totalTokensGenerated += completionTokens
        totalPrefillMs += prefillMs
        totalDecodeMs += decodeMs

        await contextMonitor.recordTokenUsage(
            sessionID: sessionID,
            promptTokens: promptTokens,
            completionTokens: completionTokens
        )

        let totalDuration = overallEnd - overallStart

        logger.debug("Session \(sessionID.description, privacy: .public) DAG complete: \(promptTokens, privacy: .public)p+\(completionTokens, privacy: .public)c on W\(workerIdx)")
        await FileLogger.shared.log(level: .debug, category: "Scheduler", message: "Session \(sessionID.description) complete: \(promptTokens)p+\(completionTokens)c on W\(workerIdx)")

        return InferenceResult(
            sessionID: sessionID,
            text: text,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            finishReason: finishReason,
            workerIndex: workerIdx,
            prefillDuration: .milliseconds(Int64(prefillMs)),
            decodeDuration: .milliseconds(Int64(decodeMs)),
            totalDuration: totalDuration,
            thinking: thinking
        )
    }

    // MARK: - Shared Memory Path (Phase 4)

    private func completeViaShm(
        sessionID: SessionID,
        session: inout ScheduledSession,
        prompt: String,
        params: SamplingParams
    ) async throws -> InferenceResult {
        let pool = try await workerPool.getPool()
        let kernel = try await workerPool.kernelHandle(for: session.workerIndex)
        let resultBuffer = try await workerPool.getOrCreateResultBuffer(for: sessionID.description)

        session.phase = .prefilling
        session.lastActivity = Date()
        sessions[sessionID] = session

        totalScheduled += 1
        let overallStart = ContinuousClock.now
        let workerIdx = session.workerIndex

        let completeResult: [String: Any]
        do {
            completeResult = try await LlamaSessionKernel.completeToShm(
                pool: pool,
                workerIndex: workerIdx,
                kernelHandle: kernel,
                sessionID: sessionID.description,
                prompt: prompt,
                resultBuffer: resultBuffer,
                params: params
            )
        } catch {
            session.phase = .failed
            sessions[sessionID] = session
            totalFailed += 1
            throw error
        }
        let overallEnd = ContinuousClock.now

        if let error = completeResult["error"] as? String {
            session.phase = .failed
            sessions[sessionID] = session
            totalFailed += 1
            throw InferenceError.decodeFailed(sessionID: sessionID, reason: error)
        }

        let text = (completeResult["text"] as? String) ?? ""
        let thinking = completeResult["thinking"] as? String
        let promptTokens = (completeResult["prompt_tokens"] as? Int) ?? 0
        let completionTokens = (completeResult["completion_tokens"] as? Int) ?? 0
        let finishReason = (completeResult["finish_reason"] as? String) ?? "unknown"
        let prefillMs = (completeResult["prefill_ms"] as? Double) ?? 0
        let decodeMs = (completeResult["decode_ms"] as? Double) ?? 0

        session.tokenBudgetUsed = min(config.contextSize, max(0, promptTokens + completionTokens))
        session.phase = .completed
        session.lastActivity = Date()
        sessions[sessionID] = session

        totalCompleted += 1
        totalTokensGenerated += completionTokens
        totalPrefillMs += prefillMs
        totalDecodeMs += decodeMs

        await contextMonitor.recordTokenUsage(
            sessionID: sessionID,
            promptTokens: promptTokens,
            completionTokens: completionTokens
        )

        let totalDuration = overallEnd - overallStart

        logger.debug("Session \(sessionID.description, privacy: .public) SHM complete: \(promptTokens, privacy: .public)p+\(completionTokens, privacy: .public)c on W\(workerIdx)")
        await FileLogger.shared.log(level: .debug, category: "Scheduler", message: "Session \(sessionID.description) SHM complete: \(promptTokens)p+\(completionTokens)c on W\(workerIdx)")

        return InferenceResult(
            sessionID: sessionID,
            text: text,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            finishReason: finishReason,
            workerIndex: workerIdx,
            prefillDuration: .milliseconds(Int64(prefillMs)),
            decodeDuration: .milliseconds(Int64(decodeMs)),
            totalDuration: totalDuration,
            thinking: thinking
        )
    }

    // MARK: - Batch Execution (multi-session DAG)

    public func completeBatch(
        requests: [(sessionID: SessionID, prompt: String, params: SamplingParams)]
    ) async throws -> [SessionID: Result<InferenceResult, Error>] {
        if config.useSharedMemory {
            return try await completeBatchViaShm(requests: requests)
        } else {
            return try await completeBatchViaPickle(requests: requests)
        }
    }

    private func completeBatchViaPickle(
        requests: [(sessionID: SessionID, prompt: String, params: SamplingParams)]
    ) async throws -> [SessionID: Result<InferenceResult, Error>] {
        let pool = try await workerPool.getPool()
        let overallStart = ContinuousClock.now

        typealias DAG = ProcessPoolDAG<String, String>
        var nodes: [DAG.Node] = []
        var sessionMap: [String: SessionID] = [:]
        var outcomes: [SessionID: Result<InferenceResult, Error>] = [:]

        for req in requests {
            guard var session = sessions[req.sessionID] else { continue }
            guard session.phase == .idle || session.phase == .completed else { continue }

            let tokenBudgetRemaining = config.contextSize - session.tokenBudgetUsed
            let promptTokens = await tokenCountForBudget(req.prompt)
            if session.tokenBudgetUsed >= config.contextSize
                || promptTokens > tokenBudgetRemaining
            {
                outcomes[req.sessionID] = .failure(InferenceError.contextOverflow(
                    sessionID: req.sessionID,
                    used: session.tokenBudgetUsed,
                    max: config.contextSize
                ))
                continue
            }

            let kernel = try await workerPool.kernelHandle(for: session.workerIndex)
            let sid = req.sessionID.description
            let prefillID = "prefill-\(sid)"
            let decodeID = "decode-\(sid)"
            let workerIdx = session.workerIndex

            sessionMap[decodeID] = req.sessionID

            session.phase = .prefilling
            session.lastActivity = Date()
            sessions[req.sessionID] = session

            nodes.append(DAG.Node(
                id: prefillID,
                preferredWorker: workerIdx
            ) { ctx in
                let json: String = try await ctx.pool.methodResult(
                    handle: kernel,
                    name: "prefill",
                    args: [
                        .python(sid),
                        .python(req.prompt),
                        .python("{}"),
                    ],
                    worker: ctx.workerIndex
                )
                return json
            })

            nodes.append(DAG.Node(
                id: decodeID,
                dependencies: [prefillID],
                preferredWorker: workerIdx
            ) { ctx in
                let json: String = try await ctx.pool.methodResult(
                    handle: kernel,
                    name: "decode",
                    args: [.python(sid)],
                    kwargs: [
                        "max_tokens": .python(req.params.maxTokens),
                        "temperature": .python(req.params.temperature),
                        "top_p": .python(req.params.topP),
                        "top_k": .python(req.params.topK),
                        "repeat_penalty": .python(req.params.repeatPenalty),
                    ],
                    worker: ctx.workerIndex
                )
                return json
            })
        }

        totalScheduled += requests.count

        let dag = DAG(nodes: nodes)
        let dagResult = try await pool.run(dag, failurePolicy: .continueIndependent)
        let overallEnd = ContinuousClock.now

        for (decodeID, sessionID) in sessionMap {
            guard var session = sessions[sessionID] else { continue }
            let prefillID = "prefill-\(sessionID.description)"

            if dagResult.failed[decodeID] != nil || dagResult.skipped.contains(decodeID) {
                session.phase = .failed
                sessions[sessionID] = session
                totalFailed += 1
                let err = dagResult.failed[decodeID] ?? dagResult.failed[prefillID]
                outcomes[sessionID] = .failure(err ?? InferenceError.decodeFailed(
                    sessionID: sessionID, reason: "DAG node failed or skipped"
                ))
                continue
            }

            guard let prefillJSON = dagResult.completed[prefillID],
                  let decodeJSON = dagResult.completed[decodeID] else {
                session.phase = .failed
                sessions[sessionID] = session
                totalFailed += 1
                outcomes[sessionID] = .failure(InferenceError.decodeFailed(
                    sessionID: sessionID, reason: "Missing DAG results"
                ))
                continue
            }

            do {
                let prefillResult = try LlamaSessionKernel.parseJSON(prefillJSON)
                let decodeResult = try LlamaSessionKernel.parseJSON(decodeJSON)

                if let error = prefillResult["error"] as? String {
                    throw InferenceError.prefillFailed(sessionID: sessionID, reason: error)
                }
                if let error = decodeResult["error"] as? String {
                    throw InferenceError.decodeFailed(sessionID: sessionID, reason: error)
                }

                let text = (decodeResult["text"] as? String) ?? ""
                let thinking = decodeResult["thinking"] as? String
                let promptTokens = (decodeResult["prompt_tokens"] as? Int) ?? 0
                let completionTokens = (decodeResult["completion_tokens"] as? Int) ?? 0
                let finishReason = (decodeResult["finish_reason"] as? String) ?? "unknown"
                let prefillMs = (prefillResult["prefill_ms"] as? Double) ?? 0
                let decodeMs = (decodeResult["decode_ms"] as? Double) ?? 0

                session.tokenBudgetUsed = min(config.contextSize, max(0, promptTokens + completionTokens))
                session.phase = .completed
                session.lastActivity = Date()
                sessions[sessionID] = session

                totalCompleted += 1
                totalTokensGenerated += completionTokens
                totalPrefillMs += prefillMs
                totalDecodeMs += decodeMs

                outcomes[sessionID] = .success(InferenceResult(
                    sessionID: sessionID,
                    text: text,
                    promptTokens: promptTokens,
                    completionTokens: completionTokens,
                    finishReason: finishReason,
                    workerIndex: session.workerIndex,
                    prefillDuration: .milliseconds(Int64(prefillMs)),
                    decodeDuration: .milliseconds(Int64(decodeMs)),
                    totalDuration: overallEnd - overallStart,
                    thinking: thinking
                ))
            } catch {
                session.phase = .failed
                sessions[sessionID] = session
                totalFailed += 1
                outcomes[sessionID] = .failure(error)
            }
        }

        logger.debug("Batch DAG completed: \(dagResult.completed.count / 2, privacy: .public) ok, \(dagResult.failed.count, privacy: .public) failed, \(dagResult.skipped.count, privacy: .public) skipped")
        return outcomes
    }

    private func completeBatchViaShm(
        requests: [(sessionID: SessionID, prompt: String, params: SamplingParams)]
    ) async throws -> [SessionID: Result<InferenceResult, Error>] {
        let pool = try await workerPool.getPool()
        let overallStart = ContinuousClock.now

        var outcomes: [SessionID: Result<InferenceResult, Error>] = [:]

        try await withThrowingTaskGroup(of: (SessionID, Result<InferenceResult, Error>).self) { group in
            for req in requests {
                guard var session = sessions[req.sessionID] else { continue }
                guard session.phase == .idle || session.phase == .completed else { continue }
                let tokenBudgetRemaining = config.contextSize - session.tokenBudgetUsed
                let promptTokens = await tokenCountForBudget(req.prompt)
                if session.tokenBudgetUsed >= config.contextSize
                    || promptTokens > tokenBudgetRemaining
                {
                    outcomes[req.sessionID] = .failure(InferenceError.contextOverflow(
                        sessionID: req.sessionID,
                        used: session.tokenBudgetUsed,
                        max: config.contextSize
                    ))
                    continue
                }

                let kernel = try await workerPool.kernelHandle(for: session.workerIndex)
                let resultBuffer = try await workerPool.getOrCreateResultBuffer(for: req.sessionID.description)
                let workerIdx = session.workerIndex

                session.phase = .prefilling
                session.lastActivity = Date()
                sessions[req.sessionID] = session

                totalScheduled += 1

                group.addTask { [req] in
                    do {
                        let completeResult = try await LlamaSessionKernel.completeToShm(
                            pool: pool,
                            workerIndex: workerIdx,
                            kernelHandle: kernel,
                            sessionID: req.sessionID.description,
                            prompt: req.prompt,
                            resultBuffer: resultBuffer,
                            params: req.params
                        )

                        if let error = completeResult["error"] as? String {
                            throw InferenceError.decodeFailed(sessionID: req.sessionID, reason: error)
                        }

                        let text = (completeResult["text"] as? String) ?? ""
                        let thinking = completeResult["thinking"] as? String
                        let promptTokens = (completeResult["prompt_tokens"] as? Int) ?? 0
                        let completionTokens = (completeResult["completion_tokens"] as? Int) ?? 0
                        let finishReason = (completeResult["finish_reason"] as? String) ?? "unknown"
                        let prefillMs = (completeResult["prefill_ms"] as? Double) ?? 0
                        let decodeMs = (completeResult["decode_ms"] as? Double) ?? 0

                        let result = InferenceResult(
                            sessionID: req.sessionID,
                            text: text,
                            promptTokens: promptTokens,
                            completionTokens: completionTokens,
                            finishReason: finishReason,
                            workerIndex: workerIdx,
                            prefillDuration: .milliseconds(Int64(prefillMs)),
                            decodeDuration: .milliseconds(Int64(decodeMs)),
                            totalDuration: ContinuousClock.now - overallStart,
                            thinking: thinking
                        )
                        return (req.sessionID, .success(result))
                    } catch {
                        return (req.sessionID, .failure(error))
                    }
                }
            }

            for try await (sessionID, result) in group {
                guard var session = sessions[sessionID] else { continue }
                switch result {
                case .success(let inferenceResult):
                    session.tokenBudgetUsed = min(
                        config.contextSize,
                        max(0, inferenceResult.promptTokens + inferenceResult.completionTokens)
                    )
                    session.phase = .completed
                    session.lastActivity = Date()
                    sessions[sessionID] = session
                    totalCompleted += 1
                    totalTokensGenerated += inferenceResult.completionTokens
                    let decodeMs = Double(inferenceResult.decodeDuration.components.seconds) * 1000
                        + Double(inferenceResult.decodeDuration.components.attoseconds) / 1e15
                    let prefillMs = Double(inferenceResult.prefillDuration.components.seconds) * 1000
                        + Double(inferenceResult.prefillDuration.components.attoseconds) / 1e15
                    totalDecodeMs += decodeMs
                    totalPrefillMs += prefillMs
                case .failure:
                    session.phase = .failed
                    sessions[sessionID] = session
                    totalFailed += 1
                }
                outcomes[sessionID] = result
            }
        }

        let succeeded = outcomes.values.filter { if case .success = $0 { return true }; return false }.count
        let failed = outcomes.count - succeeded
        logger.debug("Batch SHM completed: \(succeeded, privacy: .public) ok, \(failed, privacy: .public) failed")
        return outcomes
    }

    // MARK: - Session Reset + Rehydration (Memory Management)

    /// Evict old session, create fresh one on same worker, prefill with rehydrated context.
    ///
    /// This flushes the KV cache and rebuilds context from:
    /// 1. System prompt (canonical)
    /// 2. Retrieved facts from memory graph (structured)
    /// 3. Last N conversational turns (recent context)
    ///
    /// Returns the new session ID (same worker affinity).
    /// Target utilization after rehydration, as a fraction of contextSize.
    /// Leaves headroom for the new user prompt + a full assistant response.
    private static let rehydrationBudgetRatio: Double = 0.40

    /// Approximate characters per token for budget estimation.
    private static let charsPerToken: Double = 3.5

    /// Do not treat `maxTokens` as guaranteed output; reserve a bounded
    /// completion headroom slice for proactive pre-decode rollover checks.
    private static let projectedHeadroomRatio: Double = 0.25
    private static let projectedHeadroomMinTokens: Int = 256

    private func projectedCompletionHeadroom(maxTokens: Int) -> Int {
        let boundedMax = min(config.contextSize, max(0, maxTokens))
        let ratioCap = Int(Double(config.contextSize) * Self.projectedHeadroomRatio)
        let reserveCap = min(
            config.contextSize,
            max(Self.projectedHeadroomMinTokens, ratioCap)
        )
        return min(boundedMax, reserveCap)
    }

    private func tokenCountForBudget(_ text: String) async -> Int {
        guard !text.isEmpty else { return 0 }
        if let count = try? await workerPool.countTokens(text) {
            return max(0, count)
        }
        return max(1, Int(Double(text.utf8.count) / Self.charsPerToken))
    }

    private nonisolated static func promptWithDocumentContext(
        prompt: String,
        documentContext: String?
    ) -> String {
        guard let documentContext, !documentContext.isEmpty else { return prompt }
        return "<current_attachment_context>\n\(documentContext)\n</current_attachment_context>\n\n\(prompt)"
    }

    /// Compute budgeted system prompt and turns for session creation (cold-start or rehydration).
    /// Returns (rehydratedSystemPrompt, turnsToReplay, estimatedTokens).
    /// Maximum fraction of the rehydration budget allocated to document context.
    private static let documentContextBudgetRatio: Double = 0.30

    private func computeBudgetedRehydration(
        systemPrompt: String,
        recentTurns: [(role: String, content: String)],
        narrativeSummary: String? = nil,
        documentContext: String? = nil
    ) async -> (rehydratedPrompt: String, turnsToReplay: [(role: String, content: String)], estimatedTokens: Int) {
        let budgetChars = Int(Double(config.contextSize) * Self.rehydrationBudgetRatio * Self.charsPerToken)
        var remainingChars = budgetChars

        let systemChars = systemPrompt.utf8.count
        remainingChars -= systemChars

        // 1. Last 2 turns get half of remaining budget (highest priority after system prompt)
        let lastPair = Array(recentTurns.suffix(2))
        let lastPairBudget = max(remainingChars / 2, 0)
        var pairCharsUsed = 0
        let truncatedLastPair: [(role: String, content: String)] = lastPair.map { turn in
            let overhead = 20
            let available = lastPairBudget - pairCharsUsed - overhead
            guard available > 4 else { return (role: turn.role, content: "...") }
            let c = turn.content.utf8.count
            if c <= available {
                pairCharsUsed += c + overhead
                return turn
            }
            let truncated = String(turn.content.prefix(available - 3)) + "..."
            pairCharsUsed += truncated.utf8.count + overhead
            return (role: turn.role, content: truncated)
        }
        remainingChars -= pairCharsUsed

        // 2. Document context gets up to 30% of total budget (capped by remaining)
        var truncatedDocContext: String? = nil
        if let docCtx = documentContext, !docCtx.isEmpty {
            let docBudget = min(
                Int(Double(budgetChars) * Self.documentContextBudgetRatio),
                max(remainingChars / 2, 0)
            )
            if docCtx.utf8.count <= docBudget {
                truncatedDocContext = docCtx
                remainingChars -= docCtx.utf8.count
            } else if docBudget > 60 {
                let truncated = String(docCtx.prefix(docBudget - 40))
                    + "\n[truncated \u{2014} first \(docBudget - 40) of \(docCtx.utf8.count) chars]"
                truncatedDocContext = truncated
                remainingChars -= truncated.utf8.count
            }
        }

        // 3. Narrative summary gets 2/3 of what remains
        var truncatedNarrative: String? = nil
        if let narrative = narrativeSummary, !narrative.isEmpty {
            let narrativeBudget = max(remainingChars * 2 / 3, 0)
            if narrative.utf8.count <= narrativeBudget {
                truncatedNarrative = narrative
                remainingChars -= narrative.utf8.count
            } else if narrativeBudget > 40 {
                let tail = String(narrative.suffix(narrativeBudget - 4))
                truncatedNarrative = "... " + tail
                remainingChars -= truncatedNarrative!.utf8.count
            }
        }

        // 4. Older turns fill remaining space
        let olderTurns = recentTurns.dropLast(min(2, recentTurns.count))
        var includedOlderTurns: [(role: String, content: String)] = []
        for turn in olderTurns.reversed() {
            let turnChars = turn.content.utf8.count + 20
            if turnChars <= remainingChars {
                includedOlderTurns.insert(turn, at: 0)
                remainingChars -= turnChars
            } else {
                break
            }
        }

        let turnsToReplay = includedOlderTurns + truncatedLastPair

        var promptParts = [systemPrompt]
        if let narrative = truncatedNarrative, !narrative.isEmpty {
            promptParts.append("\n[CONVERSATION SUMMARY]:\n\(narrative)")
        }
        if let docCtx = truncatedDocContext, !docCtx.isEmpty {
            promptParts.append("\n[DOCUMENT CONTEXT]:\n\(docCtx)")
        }
        let rehydratedPrompt = promptParts.joined(separator: "\n")

        // Use real tokenizer for the final count when available, fall back to chars/3.5.
        let fullText = rehydratedPrompt + turnsToReplay.map(\.content).joined(separator: "\n")
        let estimatedTokens: Int
        if let count = try? await workerPool.countTokens(fullText) {
            estimatedTokens = count
        } else {
            let totalChars = rehydratedPrompt.utf8.count + turnsToReplay.reduce(0) { $0 + $1.content.utf8.count }
            estimatedTokens = Int(Double(totalChars) / Self.charsPerToken)
        }
        return (rehydratedPrompt, turnsToReplay, min(config.contextSize, max(0, estimatedTokens)))
    }

    /// Create a session with prior conversation history replayed (cold-start for returned conversations).
    /// Use when switching to a conversation that has existing messages — replays budgeted turns so the
    /// model has context. When `recentTurns` is empty, behaves like `createSession`.
    public func createSessionWithHistory(
        systemPrompt: String = "You are a helpful assistant.",
        recentTurns: [(role: String, content: String)] = []
    ) async throws -> SessionID {
        if recentTurns.isEmpty {
            return try await createSession(systemPrompt: systemPrompt)
        }

        let (rehydratedPrompt, turnsToReplay, estimatedTokens) = await computeBudgetedRehydration(
            systemPrompt: systemPrompt,
            recentTurns: recentTurns,
            narrativeSummary: nil
        )

        let workerIndex = try selectWorker()
        let sessionID = SessionID()

        workerSessionCounts[workerIndex, default: 0] += 1

        do {
            let pool = try await workerPool.getPool()
            let kernel = try await workerPool.kernelHandle(for: workerIndex)

            let result = try await LlamaSessionKernel.createSession(
                pool: pool,
                workerIndex: workerIndex,
                kernelHandle: kernel,
                sessionID: sessionID.description,
                systemPrompt: rehydratedPrompt
            )

            guard (result["status"] as? String) == "created" else {
                workerSessionCounts[workerIndex, default: 1] -= 1
                throw InferenceError.modelLoadFailed("Failed to create session with history: \(result)")
            }

            var sess = ScheduledSession(id: sessionID, workerIndex: workerIndex)
            sess.tokenBudgetUsed = min(config.contextSize, max(0, estimatedTokens))
            sessions[sessionID] = sess
            await contextMonitor.registerSession(sessionID)

            try await replayTurnsIntoSession(
                pool: pool,
                workerIndex: workerIndex,
                kernelHandle: kernel,
                sessionID: sessionID.description,
                turns: turnsToReplay
            )

            await contextMonitor.resetSession(sessionID, newPromptTokens: estimatedTokens)

            logger.debug(
                "Session \(sessionID.description, privacy: .public) created with history on W\(workerIndex, privacy: .public) turns=\(turnsToReplay.count, privacy: .public) estTokens=\(estimatedTokens, privacy: .public)"
            )
            return sessionID
        } catch {
            workerSessionCounts[workerIndex, default: 1] -= 1
            throw error
        }
    }

    public func resetAndRehydrate(
        sessionID: SessionID,
        systemPrompt: String,
        recentTurns: [(role: String, content: String)],
        narrativeSummary: String? = nil,
        documentContext: String? = nil
    ) async throws -> SessionID {
        guard let session = sessions[sessionID] else {
            throw InferenceError.sessionNotFound(sessionID)
        }
        let workerIndex = session.workerIndex

        let (rehydratedPrompt, turnsToReplay, estimatedTokens) = await computeBudgetedRehydration(
            systemPrompt: systemPrompt,
            recentTurns: recentTurns,
            narrativeSummary: narrativeSummary,
            documentContext: documentContext
        )

        let estimatedUtilization = Double(estimatedTokens) / Double(config.contextSize)
        logger.debug(
            "Rehydration: \(estimatedTokens, privacy: .public) tokens (\(String(format: "%.0f%%", estimatedUtilization * 100), privacy: .public) of \(self.config.contextSize, privacy: .public)) narrative=\(narrativeSummary?.isEmpty == false, privacy: .public) turns=\(turnsToReplay.count, privacy: .public)"
        )

        let pool = try await workerPool.getPool()
        let kernel = try await workerPool.kernelHandle(for: workerIndex)
        let newSessionID = SessionID()

        let result = try await LlamaSessionKernel.createSession(
            pool: pool,
            workerIndex: workerIndex,
            kernelHandle: kernel,
            sessionID: newSessionID.description,
            systemPrompt: rehydratedPrompt
        )

        guard (result["status"] as? String) == "created" else {
            throw InferenceError.modelLoadFailed("Rehydration session creation failed: \(result)")
        }

        do {
            try await replayTurnsIntoSession(
                pool: pool,
                workerIndex: workerIndex,
                kernelHandle: kernel,
                sessionID: newSessionID.description,
                turns: turnsToReplay
            )
        } catch {
            _ = try? await LlamaSessionKernel.evict(
                pool: pool,
                workerIndex: workerIndex,
                kernelHandle: kernel,
                sessionID: newSessionID.description
            )
            throw error
        }

        do {
            try await evictSession(sessionID)
        } catch {
            _ = try? await LlamaSessionKernel.evict(
                pool: pool,
                workerIndex: workerIndex,
                kernelHandle: kernel,
                sessionID: newSessionID.description
            )
            throw error
        }

        var sess = ScheduledSession(id: newSessionID, workerIndex: workerIndex)
        sess.tokenBudgetUsed = min(config.contextSize, max(0, estimatedTokens))
        sessions[newSessionID] = sess
        workerSessionCounts[workerIndex, default: 0] += 1
        await contextMonitor.registerSession(newSessionID)
        await contextMonitor.resetSession(newSessionID, newPromptTokens: estimatedTokens)

        logger.debug(
            "Rehydrated \(sessionID.description, privacy: .public) → \(newSessionID.description, privacy: .public) on W\(workerIndex, privacy: .public)"
        )

        return newSessionID
    }

    /// Replay user+assistant turns into a Python session's message history.
    private func replayTurnsIntoSession(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionID: String,
        turns: [(role: String, content: String)]
    ) async throws {
        guard !turns.isEmpty else { return }

        let messages = turns.map { ["role": $0.role, "content": $0.content] }
        guard let data = try? JSONSerialization.data(withJSONObject: messages),
              let messagesJSON = String(data: data, encoding: .utf8) else {
            return
        }

        let injectCode = """
        import json as _json
        _sid = \(LlamaSessionKernel.pythonStringLiteral(sessionID))
        _msgs = _json.loads(\(LlamaSessionKernel.pythonStringLiteral(messagesJSON)))
        _sess = _kernel._sessions.get(_sid)
        if _sess is not None:
            _sess['messages'].extend(_msgs)
        len(_msgs)
        """
        _ = try await pool.eval(injectCode, worker: workerIndex)
    }

    // MARK: - Memory Management Integration

    /// Install the narrative memory worker on the pool.
    /// Call after pool startup.
    public func installMemoryGraph() async throws {
        let pool = try await workerPool.getPool()
        let worker = NarrativeMemoryWorker()
        let (sumHandle, sumWorkerIdx) = try await workerPool.getSummarizerHandle()

        await worker.install(
            pool: pool,
            summarizerWorkerIndex: sumWorkerIdx,
            summarizerHandle: sumHandle,
            contextSize: config.contextSize
        )
        self.narrativeMemory = worker
        logger.debug("NarrativeMemoryWorker installed")
        await FileLogger.shared.log(level: .debug, category: "Memory", message: "NarrativeMemoryWorker installed")
    }

    /// Resolve the session that should be used after context-wind checks.
    private func sessionAfterMemoryManagement(
        sessionID: SessionID,
        prompt: String,
        systemPrompt: String = "You are a helpful assistant.",
        recentTurns: [(role: String, content: String)] = [],
        documentContext: String? = nil,
        projectedCompletionTokens: Int? = nil
    ) async throws -> SessionID {
        var currentSessionID = sessionID

        if let projectedCompletionTokens {
            guard let session = sessions[currentSessionID] else {
                throw InferenceError.sessionNotFound(currentSessionID)
            }
            let projectedPrompt = Self.promptWithDocumentContext(
                prompt: prompt,
                documentContext: documentContext
            )
            let estimatedPromptTokens = await tokenCountForBudget(projectedPrompt)
            let completionHeadroom = projectedCompletionHeadroom(maxTokens: projectedCompletionTokens)
            let projectedTotal = session.tokenBudgetUsed + estimatedPromptTokens + completionHeadroom
            if projectedTotal >= config.contextSize {
                logger.debug(
                    "Projected context headroom breach for \(currentSessionID.description, privacy: .public): used=\(session.tokenBudgetUsed, privacy: .public) + estPrompt=\(estimatedPromptTokens, privacy: .public) + projectedCompletion=\(completionHeadroom, privacy: .public) (from maxTokens=\(projectedCompletionTokens, privacy: .public)) >= \(self.config.contextSize, privacy: .public)"
                )
                do {
                    currentSessionID = try await resetAndRehydrate(
                        sessionID: currentSessionID,
                        systemPrompt: systemPrompt,
                        recentTurns: recentTurns,
                        narrativeSummary: nil,
                        documentContext: documentContext
                    )
                } catch {
                    let desc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
                    logger.error("Projected rehydration failed: \(desc, privacy: .public)")
                    await FileLogger.shared.log(level: .error, category: "Memory", message: "Projected rehydration failed: \(desc)")
                    currentSessionID = try await resetAndRehydrate(
                        sessionID: currentSessionID,
                        systemPrompt: systemPrompt,
                        recentTurns: Array(recentTurns.suffix(2)),
                        narrativeSummary: nil
                    )
                }
            }
        }

        // Check context wind
        let utilization = await contextMonitor.utilization(for: currentSessionID)

        if utilization >= ContextThreshold.commit.rawValue, let memory = narrativeMemory {
            logger.debug(
                "Context wind commit triggered: u=\(String(format: "%.3f", utilization), privacy: .public) for \(currentSessionID.description, privacy: .public)"
            )
            await FileLogger.shared.log(level: .debug, category: "Memory", message: "Context wind commit: u=\(String(format: "%.3f", utilization)) session=\(currentSessionID.description)")

            // Strip document context blocks from session history so the summarizer
            // focuses on the conversation, not static document text.
            let cleanedTurns = Self.stripDocumentBlocks(from: recentTurns)
            let sessionHistory = cleanedTurns.map { ["role": $0.role, "content": $0.content] }
            var narrative: String?

            do {
                narrative = try await memory.summarize(
                    sessionID: currentSessionID,
                    sessionHistory: sessionHistory
                )
            } catch {
                let desc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
                logger.error("Summarization failed: \(desc, privacy: .public)")
                await FileLogger.shared.log(level: .error, category: "Memory", message: "Summarization failed: \(desc)")
            }

            // Reset and rehydrate with narrative + recent turns + document context
            do {
                currentSessionID = try await resetAndRehydrate(
                    sessionID: currentSessionID,
                    systemPrompt: systemPrompt,
                    recentTurns: recentTurns,
                    narrativeSummary: narrative,
                    documentContext: documentContext
                )
            } catch {
                let desc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
                logger.error("Rehydration failed: \(desc, privacy: .public)")
                await FileLogger.shared.log(level: .error, category: "Memory", message: "Rehydration failed: \(desc)")
                // Last resort: bare reset so inference can proceed
                currentSessionID = try await resetAndRehydrate(
                    sessionID: currentSessionID,
                    systemPrompt: systemPrompt,
                    recentTurns: Array(recentTurns.suffix(2)),
                    narrativeSummary: nil
                )
            }
        } else if utilization >= ContextThreshold.prepare.rawValue, narrativeMemory != nil {
            logger.debug(
                "Context wind prepare: u=\(String(format: "%.3f", utilization), privacy: .public) for \(currentSessionID.description, privacy: .public) — summarization will trigger at 0.70"
            )
        }

        return currentSessionID
    }

    /// Complete with automatic context wind management.
    ///
    /// Monitors context utilization and triggers summarize→rehydrate
    /// when thresholds are crossed. Transparent to callers — returns the same
    /// `InferenceResult` as `complete()`.
    ///
    /// - Parameters:
    ///   - sessionID: Mutable — may change if session is reset.
    ///   - prompt: User message (without document context — pass that separately).
    ///   - params: Sampling parameters.
    ///   - systemPrompt: Original system prompt (needed for rehydration).
    ///   - recentTurns: Last N turns for rehydration context.
    ///   - documentContext: Extracted document text, budgeted separately from conversation turns.
    /// - Returns: Tuple of (result, newSessionID) — sessionID changes on reset.
    public func completeWithMemoryManagement(
        sessionID: SessionID,
        prompt: String,
        params: SamplingParams = .default,
        systemPrompt: String = "You are a helpful assistant.",
        recentTurns: [(role: String, content: String)] = [],
        documentContext: String? = nil
    ) async throws -> (result: InferenceResult, sessionID: SessionID) {
        let resolvedSessionID = try await sessionAfterMemoryManagement(
            sessionID: sessionID,
            prompt: prompt,
            systemPrompt: systemPrompt,
            recentTurns: recentTurns,
            documentContext: documentContext
        )
        let completionPrompt = resolvedSessionID == sessionID
            ? Self.promptWithDocumentContext(prompt: prompt, documentContext: documentContext)
            : prompt
        let result = try await complete(sessionID: resolvedSessionID, prompt: completionPrompt, params: params)
        return (result, resolvedSessionID)
    }

    /// Streamed variant of `completeWithMemoryManagement`.
    ///
    /// This resolves context-wind resets exactly like the non-streaming path,
    /// then returns a streamed decode sequence from the resolved session.
    public func completeStreamWithMemoryManagement(
        sessionID: SessionID,
        prompt: String,
        params: SamplingParams = .default,
        systemPrompt: String = "You are a helpful assistant.",
        recentTurns: [(role: String, content: String)] = [],
        documentContext: String? = nil
    ) async throws -> (stream: CancellableStream<StreamInferenceChunk>, sessionID: SessionID) {
        let resolvedSessionID = try await sessionAfterMemoryManagement(
            sessionID: sessionID,
            prompt: prompt,
            systemPrompt: systemPrompt,
            recentTurns: recentTurns,
            documentContext: documentContext,
            projectedCompletionTokens: params.maxTokens
        )
        let completionPrompt = resolvedSessionID == sessionID
            ? Self.promptWithDocumentContext(prompt: prompt, documentContext: documentContext)
            : prompt
        let stream = try await completeStream(
            sessionID: resolvedSessionID,
            prompt: completionPrompt,
            params: params
        )
        return (stream, resolvedSessionID)
    }

    private nonisolated static func mapKernelStreamToPublic(
        _ kernelStream: CancellableStream<LlamaSessionKernel.DecodeStreamChunk>
    ) -> CancellableStream<StreamInferenceChunk> {
        let (stream, continuation) = AsyncThrowingStream<StreamInferenceChunk, Error>.makeStream()
        let forwardTask = Task {
            do {
                for try await chunk in kernelStream {
                    let event = StreamEventKind(rawValue: chunk.event.rawValue) ?? .error
                    continuation.yield(
                        StreamInferenceChunk(
                            event: event,
                            delta: chunk.delta,
                            finishReason: chunk.finishReason,
                            promptTokens: chunk.promptTokens,
                            completionTokens: chunk.completionTokens,
                            decodeMs: chunk.decodeMs,
                            prefillMs: chunk.prefillMs,
                            text: chunk.text,
                            thinking: chunk.thinking,
                            error: chunk.error,
                            traceback: chunk.traceback
                        )
                    )
                }
                continuation.finish()
            } catch is CancellationError {
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return CancellableStream(stream) {
            forwardTask.cancel()
        }
    }

    // MARK: - Document Block Stripping

    /// Remove `[Attached file: ...]` and `[Image: ...]` blocks from turn content
    /// so the summarizer focuses on conversation, not static document text.
    static func stripDocumentBlocks(
        from turns: [(role: String, content: String)]
    ) -> [(role: String, content: String)] {
        let pattern = #"(?m)^\[(?:Attached file|Image): [^\]]*\]\n(?:[\s\S]*?)(?=\n\[(?:Attached file|Image): |\n\n[A-Z]|\z)"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return turns
        }
        return turns.map { turn in
            let nsContent = turn.content as NSString
            let cleaned = regex.stringByReplacingMatches(
                in: turn.content,
                range: NSRange(location: 0, length: nsContent.length),
                withTemplate: ""
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            return (role: turn.role, content: cleaned.isEmpty ? turn.content : cleaned)
        }
    }

    // MARK: - Tokenizer

    /// Count tokens using the model's actual tokenizer.
    /// Falls back to `chars / 3.5` if the pool is unavailable.
    public func countTokens(_ text: String) async -> Int {
        do {
            return try await workerPool.countTokens(text)
        } catch {
            return Int(Double(text.utf8.count) / Self.charsPerToken)
        }
    }

    // MARK: - One-Shot Convenience

    public func completeOneShot(
        prompt: String,
        params: SamplingParams = .default,
        systemPrompt: String? = nil
    ) async throws -> InferenceResult {
        let sessionID = try await createSession(systemPrompt: systemPrompt)
        let result = try await complete(sessionID: sessionID, prompt: prompt, params: params)
        try? await evictSession(sessionID)
        return result
    }

    // MARK: - LRU Eviction

    public func evictLRU(keepMax: Int? = nil) async throws -> [SessionID] {
        let maxSessions = keepMax ?? config.maxTotalSessions
        guard sessions.count > maxSessions else { return [] }

        let sorted = sessions.values
            .filter { $0.phase != .prefilling && $0.phase != .decoding }
            .sorted { $0.lastActivity < $1.lastActivity }

        var evicted: [SessionID] = []
        for session in sorted {
            guard sessions.count - evicted.count > maxSessions else { break }
            try? await evictSession(session.id)
            evicted.append(session.id)
        }

        return evicted
    }

    // MARK: - Query

    public var activeSessions: [ScheduledSession] {
        sessions.values.filter { $0.phase != .evicted && $0.phase != .failed }
            .sorted { $0.createdAt < $1.createdAt }
    }

    public func sessionInfo(_ id: SessionID) -> ScheduledSession? {
        sessions[id]
    }

    public var workerLoad: [Int: Int] {
        workerSessionCounts
    }

    public var schedulerStats: SchedulerStats {
        SchedulerStats(
            totalScheduled: totalScheduled,
            totalCompleted: totalCompleted,
            totalFailed: totalFailed,
            totalTokensGenerated: totalTokensGenerated,
            totalPrefillMs: totalPrefillMs,
            totalDecodeMs: totalDecodeMs,
            activeSessions: activeSessions.count,
            workerLoad: workerSessionCounts,
            pendingPrefills: prefillQueue.count,
            pendingDecodes: decodeQueue.count
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

// MARK: - Scheduler Stats

public struct SchedulerStats: Sendable {
    public let totalScheduled: Int
    public let totalCompleted: Int
    public let totalFailed: Int
    public let totalTokensGenerated: Int
    public let totalPrefillMs: Double
    public let totalDecodeMs: Double
    public let activeSessions: Int
    public let workerLoad: [Int: Int]
    public let pendingPrefills: Int
    public let pendingDecodes: Int

    public var avgPrefillMs: Double {
        totalCompleted > 0 ? totalPrefillMs / Double(totalCompleted) : 0
    }

    public var avgDecodeMs: Double {
        totalCompleted > 0 ? totalDecodeMs / Double(totalCompleted) : 0
    }

    public var avgTokensPerRequest: Double {
        totalCompleted > 0 ? Double(totalTokensGenerated) / Double(totalCompleted) : 0
    }
}
