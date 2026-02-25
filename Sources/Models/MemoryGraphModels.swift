import Foundation

// MARK: - Context Snapshot

/// Snapshot of session state passed to the summarizer.
public struct ContextSnapshot: Sendable {
    public let sessionID: SessionID
    public let sessionHistory: [[String: String]]
    public let systemPrompt: String?

    public init(
        sessionID: SessionID,
        sessionHistory: [[String: String]],
        systemPrompt: String? = nil
    ) {
        self.sessionID = sessionID
        self.sessionHistory = sessionHistory
        self.systemPrompt = systemPrompt
    }
}
