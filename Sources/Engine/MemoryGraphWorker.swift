import Foundation
import SwiftPythonRuntime
import OSLog

/// Narrative memory worker for context wind management.
///
/// Coordinates the `SummarizationKernel` to produce narrative summaries
/// when context utilization crosses the commit threshold. The narrative
/// is injected into the rehydrated session's system prompt so the LLM
/// retains conversational context across session resets.
public actor NarrativeMemoryWorker {
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "memory")

    private var pool: PythonProcessPool?
    private var summarizerHandle: PyHandle?
    private var summarizerWorkerIndex: Int?
    private var contextSize: Int = 4096

    public enum State: String, Sendable {
        case idle
        case ready
        case failed
    }

    public private(set) var state: State = .idle

    // MARK: - Lifecycle

    /// Install the summarizer on a worker.
    public func install(
        pool: PythonProcessPool,
        summarizerWorkerIndex: Int,
        summarizerHandle: PyHandle,
        contextSize: Int = 4096
    ) async {
        self.pool = pool
        self.summarizerWorkerIndex = summarizerWorkerIndex
        self.summarizerHandle = summarizerHandle
        self.contextSize = contextSize
        self.state = .ready
        logger.debug("NarrativeMemoryWorker ready")
    }

    // MARK: - Summarize

    /// Generate a narrative summary of the conversation history.
    ///
    /// Called when context utilization hits the commit threshold (u >= 0.7).
    /// Returns the narrative text for injection into the rehydrated session.
    public func summarize(
        sessionID: SessionID,
        sessionHistory: [[String: String]]
    ) async throws -> String {
        guard let pool, let sumHandle = summarizerHandle, let sumWorkerIdx = summarizerWorkerIndex else {
            throw InferenceError.poolNotReady
        }

        logger.debug("summarize: session=\(sessionID.description, privacy: .public) messages=\(sessionHistory.count, privacy: .public)")

        // Cap narrative tokens to ~15% of context so the summary doesn't eat the
        // budget on small context windows (e.g. 2048 -> 307 tokens max).
        let maxTokens = max(128, contextSize * 15 / 100)

        let result = try await SummarizationKernel.summarize(
            pool: pool,
            workerIndex: sumWorkerIdx,
            kernelHandle: sumHandle,
            sessionHistory: sessionHistory,
            maxTokens: maxTokens
        )

        let narrative = (result["narrative_summary"] as? String) ?? ""
        let metadata = result["metadata"] as? [String: Any] ?? [:]
        let promptTokens = (metadata["prompt_tokens"] as? Int) ?? 0
        let completionTokens = (metadata["completion_tokens"] as? Int) ?? 0

        logger.debug("Summarized: \(narrative.count, privacy: .public) chars, \(promptTokens, privacy: .public)p+\(completionTokens, privacy: .public)c")
        return narrative
    }

    /// Generate a short semantic title (3–8 words) for the conversation.
    ///
    /// Used for automatic sidebar naming. Returns empty string if unavailable.
    public func suggestTitle(sessionHistory: [[String: String]]) async throws -> String {
        guard let pool, let sumHandle = summarizerHandle, let sumWorkerIdx = summarizerWorkerIndex else {
            throw InferenceError.poolNotReady
        }

        logger.debug("suggestTitle: messages=\(sessionHistory.count, privacy: .public)")

        let title = try await SummarizationKernel.suggestTitle(
            pool: pool,
            workerIndex: sumWorkerIdx,
            kernelHandle: sumHandle,
            sessionHistory: sessionHistory,
            maxTokens: 24
        )

        logger.debug("suggestTitle done: '\(title, privacy: .public)'")
        return title
    }
}
