import Foundation

/// Rules for whether the composer can submit a message.
public enum ComposerSendPolicy {
    /// Display text stored in the message bubble when the user sends attachments without typing.
    /// Empty so only the attachment thumbnail is shown — no redundant text bubble.
    public static let attachmentOnlyDefaultPrompt = ""

    /// Prompt sent to the chat model (not displayed) when the user sends attachments without typing.
    /// Informs the model it is downstream of a vision model and should synthesize the analysis.
    public static let attachmentOnlyModelPrompt = "You are part of a multi-model pipeline. A vision model has already processed the image attached to this message — its analysis is in <current_attachment_context> above. This is a different image from any previously discussed in this conversation. Base your response only on the current image's context. If the user wants to discuss an earlier image, they will reference it by name."

    /// Returns true when sending is allowed.
    ///
    /// Sending is allowed when the model is ready, generation is idle, and the
    /// user provided either message text or at least one attachment.
    public static func canSend(
        inputText: String,
        hasPendingAttachments: Bool,
        isReady: Bool,
        isGenerating: Bool
    ) -> Bool {
        let hasText = !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return (hasText || hasPendingAttachments) && isReady && !isGenerating
    }

    /// Resolves the prompt text that should be sent to the model.
    ///
    /// - Returns: Trimmed user text, the attachment-only default prompt, or nil
    /// when there is no text and no attachments.
    public static func resolvedPrompt(
        inputText: String,
        hasPendingAttachments: Bool,
        attachmentOnlyPrompt: String = attachmentOnlyDefaultPrompt
    ) -> String? {
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            return trimmed
        }
        return hasPendingAttachments ? attachmentOnlyPrompt : nil
    }
}

/// Rules for deciding whether a transient streamed assistant preview should be
/// committed into transcript history.
public enum LiveAssistantPreviewPolicy {
    /// Returns true when the preview contains user-visible answer or thinking.
    public static func hasVisibleContent(text: String, thinking: String?) -> Bool {
        if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return true
        }
        guard let thinking else { return false }
        return !thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

/// Parsed split between assistant visible text and `<think>` disclosure text.
public struct ThinkingTextSplit: Sendable, Equatable {
    public let text: String
    public let thinking: String?

    public init(text: String, thinking: String?) {
        self.text = text
        self.thinking = thinking
    }
}

/// Extracts `<think>...</think>` blocks from model output.
public enum ThinkingTextParser {
    private static let thinkPattern = try? NSRegularExpression(
        pattern: "<think>(.*?)</think>",
        options: .dotMatchesLineSeparators
    )
    private static let unclosedThinkPattern = try? NSRegularExpression(
        pattern: "<think>(.*)",
        options: .dotMatchesLineSeparators
    )

    /// Splits raw model text into disclosure thinking and visible answer text.
    ///
    /// If `preferredThinking` is provided and non-empty, it is used directly.
    /// Otherwise `<think>` tags are parsed from `rawText`, including unclosed
    /// trailing think blocks.
    ///
    /// When `streamingInProgress` is `true` and `rawText` contains no think
    /// tags at all, the entire text is treated as in-progress thinking (for
    /// models like Qwen3 that emit thinking tokens before any `<think>` tag).
    public static func split(
        rawText: String,
        preferredThinking: String? = nil,
        streamingInProgress: Bool = false
    ) -> ThinkingTextSplit {
        var answerText = rawText
        var thinking = normalized(preferredThinking)

        if thinking == nil {
            let nsText = answerText as NSString
            if let matches = thinkPattern?.matches(
                in: answerText,
                range: NSRange(location: 0, length: nsText.length)
            ), !matches.isEmpty {
                let parts = matches.compactMap { match -> String? in
                    guard match.numberOfRanges > 1 else { return nil }
                    return nsText.substring(with: match.range(at: 1))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                }
                let joined = parts.filter { !$0.isEmpty }.joined(separator: "\n\n")
                thinking = normalized(joined)
                answerText = thinkPattern?
                    .stringByReplacingMatches(
                        in: answerText,
                        range: NSRange(location: 0, length: nsText.length),
                        withTemplate: ""
                    )
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? answerText
            }

            let nsAnswer = answerText as NSString
            if let unclosedMatch = unclosedThinkPattern?.firstMatch(
                in: answerText,
                range: NSRange(location: 0, length: nsAnswer.length)
            ) {
                let trailing = nsAnswer.substring(with: unclosedMatch.range(at: 1))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !trailing.isEmpty {
                    let pieces = [thinking, trailing]
                        .compactMap { $0 }
                        .filter { !$0.isEmpty }
                    thinking = normalized(pieces.joined(separator: "\n\n"))
                }
                answerText = nsAnswer
                    .substring(to: unclosedMatch.range.location)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }

            // Handle models that omit the opening <think> tag (e.g. Qwen3):
            // an orphan </think> means everything before it is thinking content.
            // When streaming is still in progress and no tags exist yet, the
            // entire accumulated text is in-progress thinking.
            if thinking == nil && streamingInProgress
                && !answerText.contains("<think>")
                && !answerText.contains("</think>") {
                let candidate = answerText.trimmingCharacters(in: .whitespacesAndNewlines)
                return ThinkingTextSplit(text: "", thinking: candidate.isEmpty ? nil : candidate)
            }

            if thinking == nil, let closeRange = answerText.range(of: "</think>") {
                let before = String(answerText[answerText.startIndex..<closeRange.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let after = String(answerText[closeRange.upperBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !before.isEmpty {
                    thinking = normalized(before)
                }
                answerText = after
            }
        }

        return ThinkingTextSplit(
            text: answerText.trimmingCharacters(in: .whitespacesAndNewlines),
            thinking: normalized(thinking)
        )
    }

    private static func normalized(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
