import Foundation
import OSLog

/// Deterministic context window utilization monitor.
///
/// Tracks token usage per session and emits `ContextWindEvent` when
/// utilization crosses thresholds (prepare=0.6, commit=0.7, reset=0.8).
/// No heuristics — always calculated from actual `prompt_tokens` reported
/// by the inference kernel.
public actor ContextWindMonitor {
    private let contextSize: Int
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "context-wind")

    /// Per-session tracking state.
    private var sessionStates: [SessionID: SessionWindState] = [:]

    /// Callback invoked when a threshold is crossed.
    public var onThresholdCrossed: (@Sendable (ContextWindEvent) -> Void)?

    // MARK: - Types

    private struct SessionWindState {
        var lastPromptTokens: Int = 0
        var lastCompletionTokens: Int = 0
        var highestThresholdCrossed: ContextThreshold?
        var crossingHistory: [ContextWindEvent] = []
    }

    // MARK: - Init

    public init(contextSize: Int) {
        self.contextSize = contextSize
    }

    public func setOnThresholdCrossed(_ callback: (@Sendable (ContextWindEvent) -> Void)?) {
        self.onThresholdCrossed = callback
    }

    // MARK: - Session Lifecycle

    public func registerSession(_ sessionID: SessionID) {
        sessionStates[sessionID] = SessionWindState()
        logger.debug("Registered session \(sessionID.description, privacy: .public) for context wind monitoring")
    }

    public func unregisterSession(_ sessionID: SessionID) {
        sessionStates.removeValue(forKey: sessionID)
    }

    // MARK: - Token Reporting

    /// Record token usage after a decode step and check thresholds.
    ///
    /// - Parameters:
    ///   - sessionID: The session that just completed a decode.
    ///   - promptTokens: Prompt tokens reported by the kernel for the finished decode.
    ///   - completionTokens: Completion tokens from the finished decode.
    /// - Returns: The crossed threshold if a new threshold was crossed, nil otherwise.
    @discardableResult
    public func recordTokenUsage(
        sessionID: SessionID,
        promptTokens: Int,
        completionTokens: Int
    ) -> ContextThreshold? {
        guard var state = sessionStates[sessionID] else {
            logger.warning("recordTokenUsage for unregistered session \(sessionID.description, privacy: .public)")
            return nil
        }

        state.lastPromptTokens = max(0, promptTokens)
        state.lastCompletionTokens = max(0, completionTokens)

        let totalTokens = state.lastPromptTokens + state.lastCompletionTokens
        let u = Double(totalTokens) / Double(contextSize)

        logger.debug(
            "Session \(sessionID.description, privacy: .public) context wind: u=\(String(format: "%.3f", u), privacy: .public) (\(totalTokens, privacy: .public)/\(self.contextSize, privacy: .public))"
        )

        let crossedThreshold = checkThresholds(
            sessionID: sessionID,
            state: &state,
            utilization: u,
            promptTokens: promptTokens
        )

        sessionStates[sessionID] = state
        return crossedThreshold
    }

    // MARK: - Query

    /// Current utilization ratio for a session (0.0 – 1.0+).
    public func utilization(for sessionID: SessionID) -> Double {
        guard let state = sessionStates[sessionID] else { return 0 }
        let total = state.lastPromptTokens + state.lastCompletionTokens
        return Double(total) / Double(contextSize)
    }

    /// Total tokens tracked for a session.
    public func tokenCount(for sessionID: SessionID) -> Int {
        guard let state = sessionStates[sessionID] else { return 0 }
        return state.lastPromptTokens + state.lastCompletionTokens
    }

    /// Highest threshold crossed so far for a session.
    public func highestThreshold(for sessionID: SessionID) -> ContextThreshold? {
        sessionStates[sessionID]?.highestThresholdCrossed
    }

    /// Full crossing history for a session.
    public func crossingHistory(for sessionID: SessionID) -> [ContextWindEvent] {
        sessionStates[sessionID]?.crossingHistory ?? []
    }

    /// Reset tracking state for a session (e.g., after worker respawn + rehydration).
    public func resetSession(_ sessionID: SessionID, newPromptTokens: Int = 0) {
        sessionStates[sessionID] = SessionWindState(lastPromptTokens: newPromptTokens)
        logger.debug("Reset context wind for \(sessionID.description, privacy: .public) to \(newPromptTokens, privacy: .public) tokens")
    }

    // MARK: - Private

    private func checkThresholds(
        sessionID: SessionID,
        state: inout SessionWindState,
        utilization: Double,
        promptTokens: Int
    ) -> ContextThreshold? {
        var newlyCrossed: ContextThreshold?

        for threshold in ContextThreshold.allCases.sorted() {
            guard utilization >= threshold.rawValue else { continue }

            let alreadyCrossed: Bool
            if let highest = state.highestThresholdCrossed {
                alreadyCrossed = threshold <= highest
            } else {
                alreadyCrossed = false
            }

            guard !alreadyCrossed else { continue }

            let event = ContextWindEvent(
                sessionID: sessionID,
                threshold: threshold,
                utilization: utilization,
                promptTokens: promptTokens,
                contextSize: contextSize
            )

            state.highestThresholdCrossed = threshold
            state.crossingHistory.append(event)
            newlyCrossed = threshold

            logger.debug(
                "Session \(sessionID.description, privacy: .public) crossed \(threshold.rawValue, privacy: .public) threshold: u=\(String(format: "%.3f", utilization), privacy: .public)"
            )

            onThresholdCrossed?(event)
        }

        return newlyCrossed
    }
}
