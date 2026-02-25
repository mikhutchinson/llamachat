import XCTest
@testable import LlamaInferenceCore

final class SessionTypesTests: XCTestCase {

    // MARK: - SessionID

    func testSessionIDUniqueness() {
        let a = SessionID()
        let b = SessionID()
        XCTAssertNotEqual(a, b)
    }

    func testSessionIDDescription() {
        let id = SessionID()
        XCTAssertTrue(id.description.hasPrefix("session-"))
        XCTAssertEqual(id.description.count, "session-".count + 8)
    }

    func testSessionIDHashable() {
        let id = SessionID()
        var set: Set<SessionID> = []
        set.insert(id)
        set.insert(id)
        XCTAssertEqual(set.count, 1)
    }

    // MARK: - SessionState

    func testSessionStateInitialValues() {
        let id = SessionID()
        let state = SessionState(id: id, workerIndex: 2)
        XCTAssertEqual(state.id, id)
        XCTAssertEqual(state.workerIndex, 2)
        XCTAssertEqual(state.phase, .idle)
        XCTAssertEqual(state.promptTokenCount, 0)
        XCTAssertEqual(state.completionTokenCount, 0)
        XCTAssertEqual(state.totalTokenCount, 0)
        XCTAssertTrue(state.generatedText.isEmpty)
        XCTAssertNil(state.finishReason)
    }

    func testSessionStateTransition() {
        let id = SessionID()
        var state = SessionState(id: id, workerIndex: 0)
        let beforeTransition = state.lastActivityAt

        state.transitionTo(.prefilling)
        XCTAssertEqual(state.phase, .prefilling)
        XCTAssertGreaterThanOrEqual(state.lastActivityAt, beforeTransition)

        state.transitionTo(.decoding)
        XCTAssertEqual(state.phase, .decoding)

        state.transitionTo(.completed)
        XCTAssertEqual(state.phase, .completed)
    }

    func testSessionStateRecordPrefill() {
        let id = SessionID()
        var state = SessionState(id: id, workerIndex: 0)
        state.recordPrefill(promptTokens: 42)
        XCTAssertEqual(state.promptTokenCount, 42)
    }

    func testSessionStateRecordDecodeStep() {
        let id = SessionID()
        var state = SessionState(id: id, workerIndex: 0)
        state.recordDecodeStep(newText: "hello", tokens: 3, finishReason: nil)
        XCTAssertEqual(state.generatedText, "hello")
        XCTAssertEqual(state.completionTokenCount, 3)
        XCTAssertNil(state.finishReason)

        state.recordDecodeStep(newText: " world", tokens: 2, finishReason: "stop")
        XCTAssertEqual(state.generatedText, "hello world")
        XCTAssertEqual(state.completionTokenCount, 5)
        XCTAssertEqual(state.finishReason, "stop")
    }

    func testSessionStateTotalTokenCount() {
        let id = SessionID()
        var state = SessionState(id: id, workerIndex: 0)
        state.recordPrefill(promptTokens: 10)
        state.recordDecodeStep(newText: "out", tokens: 5, finishReason: nil)
        XCTAssertEqual(state.totalTokenCount, 15)
    }

    // MARK: - SamplingParams

    func testSamplingParamsDefaults() {
        let params = SamplingParams.default
        XCTAssertEqual(params.maxTokens, 256)
        XCTAssertEqual(params.temperature, 0.7)
        XCTAssertEqual(params.topP, 0.95)
        XCTAssertEqual(params.topK, 40)
        XCTAssertEqual(params.repeatPenalty, 1.1)
        XCTAssertTrue(params.stop.isEmpty)
    }

    func testSamplingParamsGreedy() {
        let params = SamplingParams.greedy
        XCTAssertEqual(params.temperature, 0.0)
        XCTAssertEqual(params.topK, 1)
        XCTAssertEqual(params.topP, 1.0)
        XCTAssertEqual(params.repeatPenalty, 1.0)
    }

    func testSamplingParamsCustom() {
        let params = SamplingParams(
            maxTokens: 512,
            temperature: 0.9,
            topP: 0.8,
            topK: 50,
            repeatPenalty: 1.2,
            stop: ["\\n", "END"]
        )
        XCTAssertEqual(params.maxTokens, 512)
        XCTAssertEqual(params.temperature, 0.9)
        XCTAssertEqual(params.stop.count, 2)
    }

    // MARK: - InferenceConfig

    func testInferenceConfigDefaults() {
        let config = InferenceConfig(modelPath: "/tmp/test.gguf")
        XCTAssertEqual(config.modelPath, "/tmp/test.gguf")
        XCTAssertEqual(config.contextSize, 4096)
        XCTAssertEqual(config.nGpuLayers, -1)
        XCTAssertEqual(config.workerCount, 2)
        XCTAssertEqual(config.maxSessionsPerWorker, 8)
        XCTAssertEqual(config.maxTotalSessions, 16)
        XCTAssertNil(config.maxMemoryBytesPerWorker)
        XCTAssertEqual(config.maxInFlight, 16)
        XCTAssertEqual(config.blasThreads, 1)
    }

    func testInferenceConfigCustom() {
        let config = InferenceConfig(
            modelPath: "/models/llama.gguf",
            contextSize: 8192,
            nGpuLayers: 32,
            workerCount: 4,
            maxSessionsPerWorker: 4,
            maxMemoryBytesPerWorker: 8 * 1024 * 1024 * 1024,
            maxInFlight: 32,
            blasThreads: 2
        )
        XCTAssertEqual(config.contextSize, 8192)
        XCTAssertEqual(config.workerCount, 4)
        XCTAssertEqual(config.maxTotalSessions, 16)
        XCTAssertEqual(config.maxMemoryBytesPerWorker, 8 * 1024 * 1024 * 1024)
    }

    // MARK: - InferenceRequest

    func testInferenceRequestDefaults() {
        let req = InferenceRequest(prompt: "Hello")
        XCTAssertEqual(req.prompt, "Hello")
        XCTAssertEqual(req.params.maxTokens, 256)
        XCTAssertNil(req.chatMessages)
    }

    func testInferenceRequestCustom() {
        let id = SessionID()
        let req = InferenceRequest(
            sessionID: id,
            prompt: "test",
            params: SamplingParams(maxTokens: 64),
            chatMessages: [["role": "user", "content": "test"]]
        )
        XCTAssertEqual(req.sessionID, id)
        XCTAssertEqual(req.params.maxTokens, 64)
        XCTAssertNotNil(req.chatMessages)
    }

    // MARK: - InferenceResult

    func testInferenceResultTokensPerSecond() {
        let result = InferenceResult(
            sessionID: SessionID(),
            text: "output",
            promptTokens: 10,
            completionTokens: 50,
            finishReason: "stop",
            workerIndex: 0,
            prefillDuration: .milliseconds(100),
            decodeDuration: .seconds(2),
            totalDuration: .milliseconds(2100)
        )
        XCTAssertEqual(result.tokensPerSecond, 25.0, accuracy: 0.1)
        XCTAssertEqual(result.finishReason, "stop")
    }

    func testInferenceResultZeroDuration() {
        let result = InferenceResult(
            sessionID: SessionID(),
            text: "",
            promptTokens: 0,
            completionTokens: 0,
            finishReason: "stop",
            workerIndex: 0,
            prefillDuration: .zero,
            decodeDuration: .zero,
            totalDuration: .zero
        )
        XCTAssertEqual(result.tokensPerSecond, 0)
    }

    // MARK: - InferenceError

    func testInferenceErrorDescriptions() {
        let id = SessionID()
        let errors: [InferenceError] = [
            .poolNotReady,
            .modelLoadFailed("bad path"),
            .sessionNotFound(id),
            .workerFull(workerIndex: 3),
            .contextOverflow(sessionID: id, used: 5000, max: 4096),
            .prefillFailed(sessionID: id, reason: "oom"),
            .decodeFailed(sessionID: id, reason: "timeout"),
            .evicted(sessionID: id),
            .timeout(sessionID: id),
        ]
        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }

    func testContextOverflowErrorContainsUsedAndMax() {
        let id = SessionID()
        let err = InferenceError.contextOverflow(sessionID: id, used: 5000, max: 4096)
        let desc = err.errorDescription ?? ""
        XCTAssertTrue(desc.contains("5000") || desc.contains("4096"), "Description should include used/max: \(desc)")
    }

    // MARK: - SessionPhase

    func testSessionPhaseAllCases() {
        XCTAssertEqual(SessionPhase.allCases.count, 6)
        XCTAssertEqual(SessionPhase.idle.rawValue, "idle")
        XCTAssertEqual(SessionPhase.prefilling.rawValue, "prefilling")
        XCTAssertEqual(SessionPhase.decoding.rawValue, "decoding")
        XCTAssertEqual(SessionPhase.completed.rawValue, "completed")
        XCTAssertEqual(SessionPhase.failed.rawValue, "failed")
        XCTAssertEqual(SessionPhase.evicted.rawValue, "evicted")
    }
}

// MARK: - Kernel JSON Parsing Regression Tests

final class KernelJSONParsingTests: XCTestCase {

    func testParseJSONValidDict() throws {
        let json = #"{"status": "created", "session_id": "test-123"}"#
        let result = try LlamaSessionKernel.parseJSON(json)
        XCTAssertEqual(result["status"] as? String, "created")
        XCTAssertEqual(result["session_id"] as? String, "test-123")
    }

    func testParseJSONMixedValueTypes() throws {
        let json = #"{"session_id": "s1", "prompt_tokens": 42, "prefill_ms": 1.23}"#
        let result = try LlamaSessionKernel.parseJSON(json)
        XCTAssertEqual(result["session_id"] as? String, "s1")
        XCTAssertEqual(result["prompt_tokens"] as? Int, 42)
        XCTAssertEqual(result["prefill_ms"] as! Double, 1.23, accuracy: 0.001)
    }

    func testParseJSONErrorField() throws {
        let json = #"{"error": "session not found"}"#
        let result = try LlamaSessionKernel.parseJSON(json)
        XCTAssertNotNil(result["error"] as? String)
    }

    func testParseJSONInvalidStringThrows() {
        XCTAssertThrowsError(try LlamaSessionKernel.parseJSON("not json"))
    }

    func testParseJSONArrayThrows() {
        XCTAssertThrowsError(try LlamaSessionKernel.parseJSON("[1, 2, 3]"))
    }

    func testParseJSONDecodeResultShape() throws {
        let json = #"{"session_id":"s1","text":"Hello","finish_reason":"stop","prompt_tokens":10,"completion_tokens":5,"decode_ms":123.45}"#
        let r = try LlamaSessionKernel.parseJSON(json)
        XCTAssertEqual(r["text"] as? String, "Hello")
        XCTAssertEqual(r["finish_reason"] as? String, "stop")
        XCTAssertEqual(r["prompt_tokens"] as? Int, 10)
        XCTAssertEqual(r["completion_tokens"] as? Int, 5)
        XCTAssertEqual(r["decode_ms"] as! Double, 123.45, accuracy: 0.01)
    }

    func testSuggestTitleResultParsing() throws {
        let json = #"{"suggested_title": "L'Hospital's Rule Explanation", "metadata": {"prompt_tokens": 50, "completion_tokens": 5, "duration_ms": 120.5}}"#
        let result = try LlamaSessionKernel.parseJSON(json)
        XCTAssertEqual(result["suggested_title"] as? String, "L'Hospital's Rule Explanation")
        let meta = result["metadata"] as? [String: Any]
        XCTAssertEqual(meta?["prompt_tokens"] as? Int, 50)
        XCTAssertEqual(meta?["completion_tokens"] as? Int, 5)
    }
}

// MARK: - Composer Send Policy Tests

final class ComposerSendPolicyTests: XCTestCase {
    func testCanSendWhenTextPresent() {
        let canSend = ComposerSendPolicy.canSend(
            inputText: "hello",
            hasPendingAttachments: false,
            isReady: true,
            isGenerating: false
        )
        XCTAssertTrue(canSend)
    }

    func testCanSendWhenOnlyAttachmentsPresent() {
        let canSend = ComposerSendPolicy.canSend(
            inputText: "   ",
            hasPendingAttachments: true,
            isReady: true,
            isGenerating: false
        )
        XCTAssertTrue(canSend)
    }

    func testCannotSendWhenNotReadyOrGenerating() {
        XCTAssertFalse(
            ComposerSendPolicy.canSend(
                inputText: "hello",
                hasPendingAttachments: true,
                isReady: false,
                isGenerating: false
            )
        )
        XCTAssertFalse(
            ComposerSendPolicy.canSend(
                inputText: "hello",
                hasPendingAttachments: true,
                isReady: true,
                isGenerating: true
            )
        )
    }

    func testResolvedPromptUsesDefaultForAttachmentOnly() {
        let resolved = ComposerSendPolicy.resolvedPrompt(
            inputText: "  ",
            hasPendingAttachments: true
        )
        XCTAssertEqual(resolved, ComposerSendPolicy.attachmentOnlyDefaultPrompt)
    }
}

// MARK: - Live Assistant Preview Policy Tests

final class LiveAssistantPreviewPolicyTests: XCTestCase {
    func testHasVisibleContentForText() {
        XCTAssertTrue(
            LiveAssistantPreviewPolicy.hasVisibleContent(
                text: "Hello",
                thinking: nil
            )
        )
    }

    func testHasVisibleContentForThinkingOnly() {
        XCTAssertTrue(
            LiveAssistantPreviewPolicy.hasVisibleContent(
                text: "   ",
                thinking: "step-by-step"
            )
        )
    }

    func testHasVisibleContentFalseWhenEmpty() {
        XCTAssertFalse(
            LiveAssistantPreviewPolicy.hasVisibleContent(
                text: " \n ",
                thinking: "   "
            )
        )
    }
}

// MARK: - Thinking Text Parser Tests

final class ThinkingTextParserTests: XCTestCase {
    func testSplitClosedThinkBlock() {
        let split = ThinkingTextParser.split(rawText: "<think>reasoning</think>answer")
        XCTAssertEqual(split.thinking, "reasoning")
        XCTAssertEqual(split.text, "answer")
    }

    func testSplitUnclosedThinkBlock() {
        let split = ThinkingTextParser.split(rawText: "<think>unfinished thoughts")
        XCTAssertEqual(split.thinking, "unfinished thoughts")
        XCTAssertEqual(split.text, "")
    }

    func testSplitTextOnly() {
        let split = ThinkingTextParser.split(rawText: "plain response")
        XCTAssertNil(split.thinking)
        XCTAssertEqual(split.text, "plain response")
    }

    func testOrphanCloseTag_Qwen3Style() {
        let raw = "Okay, let me think step by step.\nFirst A, then B.</think>\nThe answer is 42."
        let split = ThinkingTextParser.split(rawText: raw)
        XCTAssertEqual(split.text, "The answer is 42.")
        XCTAssertEqual(split.thinking, "Okay, let me think step by step.\nFirst A, then B.")
    }

    func testOrphanCloseTag_MultilineThinking() {
        let raw = "Let me reason.\nConsider the chain rule.\n</think>\nThe **chain rule** applies."
        let split = ThinkingTextParser.split(rawText: raw)
        XCTAssertEqual(split.text, "The **chain rule** applies.")
        XCTAssertNotNil(split.thinking)
        XCTAssertTrue(split.thinking!.contains("chain rule"))
        XCTAssertFalse(split.text.contains("Let me reason"))
    }

    func testOrphanCloseTag_EmptyThinkingBeforeClose() {
        let raw = "</think>The answer is 42."
        let split = ThinkingTextParser.split(rawText: raw)
        XCTAssertEqual(split.text, "The answer is 42.")
    }

    func testPreferredThinkingBypassesOrphanDetection() {
        let raw = "thinking tokens</think>answer"
        let split = ThinkingTextParser.split(rawText: raw, preferredThinking: "preferred")
        XCTAssertEqual(split.thinking, "preferred")
        XCTAssertEqual(split.text, "thinking tokens</think>answer")
    }

    func testStreamingInProgress_NoTagsYet_AllTextIsThinking() {
        let raw = "Okay, let me think about this carefully.\nFirst consider the problem."
        let split = ThinkingTextParser.split(rawText: raw, streamingInProgress: true)
        XCTAssertEqual(split.text, "")
        XCTAssertEqual(split.thinking, "Okay, let me think about this carefully.\nFirst consider the problem.")
    }

    func testStreamingInProgress_CloseTagArrived_SplitsCorrectly() {
        let raw = "Let me reason through this.\n</think>\nThe answer is 42."
        let split = ThinkingTextParser.split(rawText: raw, streamingInProgress: true)
        XCTAssertEqual(split.text, "The answer is 42.")
        XCTAssertEqual(split.thinking, "Let me reason through this.")
    }

    func testStreamingInProgress_OpenTagPresent_NotOrphanPath() {
        let raw = "<think>Some reasoning..."
        let split = ThinkingTextParser.split(rawText: raw, streamingInProgress: true)
        XCTAssertEqual(split.text, "")
        XCTAssertEqual(split.thinking, "Some reasoning...")
    }

    func testStreamingNotInProgress_NoTagsYet_TextPassesThrough() {
        let raw = "Just a plain answer with no thinking tags."
        let split = ThinkingTextParser.split(rawText: raw, streamingInProgress: false)
        XCTAssertEqual(split.text, "Just a plain answer with no thinking tags.")
        XCTAssertNil(split.thinking)
    }
}

// MARK: - Decode Stream Chunk Tests

final class DecodeStreamChunkTests: XCTestCase {
    func testDecodeStreamChunkDeltaEvent() throws {
        let chunk = try LlamaSessionKernel.DecodeStreamChunk(
            jsonString: #"{"event":"delta","delta":"Hello"}"#
        )
        XCTAssertEqual(chunk.event, .delta)
        XCTAssertEqual(chunk.delta, "Hello")
        XCTAssertFalse(chunk.isTerminal)
    }

    func testDecodeStreamChunkDoneEvent() throws {
        let chunk = try LlamaSessionKernel.DecodeStreamChunk(
            jsonString: #"{"event":"done","finish_reason":"stop","prompt_tokens":10,"completion_tokens":5,"decode_ms":12.5,"text":"hi","thinking":"plan"}"#
        )
        XCTAssertEqual(chunk.event, .done)
        XCTAssertEqual(chunk.finishReason, "stop")
        XCTAssertEqual(chunk.promptTokens, 10)
        XCTAssertEqual(chunk.completionTokens, 5)
        XCTAssertEqual(chunk.decodeMs ?? 0, 12.5, accuracy: 0.001)
        XCTAssertEqual(chunk.text, "hi")
        XCTAssertEqual(chunk.thinking, "plan")
        XCTAssertTrue(chunk.isTerminal)
    }

    func testDecodeStreamChunkDoneEventWithPrefillMs() throws {
        let chunk = try LlamaSessionKernel.DecodeStreamChunk(
            jsonString: #"{"event":"done","finish_reason":"stop","prompt_tokens":10,"completion_tokens":5,"prefill_ms":102.61,"decode_ms":50.0,"text":"hi","thinking":"plan"}"#
        )
        XCTAssertEqual(chunk.event, .done)
        XCTAssertEqual(chunk.finishReason, "stop")
        XCTAssertEqual(chunk.promptTokens, 10)
        XCTAssertEqual(chunk.completionTokens, 5)
        XCTAssertEqual(chunk.prefillMs ?? 0, 102.61, accuracy: 0.001)
        XCTAssertEqual(chunk.decodeMs ?? 0, 50.0, accuracy: 0.001)
        XCTAssertEqual(chunk.text, "hi")
        XCTAssertEqual(chunk.thinking, "plan")
        XCTAssertTrue(chunk.isTerminal)
    }

    func testDecodeStreamChunkErrorEvent() throws {
        let chunk = try LlamaSessionKernel.DecodeStreamChunk(
            jsonString: #"{"event":"error","error":"boom","traceback":"trace"}"#
        )
        XCTAssertEqual(chunk.event, .error)
        XCTAssertEqual(chunk.error, "boom")
        XCTAssertEqual(chunk.traceback, "trace")
        XCTAssertTrue(chunk.isTerminal)
    }

    func testDecodeStreamChunkInvalidEventThrows() {
        XCTAssertThrowsError(
            try LlamaSessionKernel.DecodeStreamChunk(
                jsonString: #"{"event":"unknown"}"#
            )
        )
    }
}

// MARK: - Scheduler Stats Tests

final class SchedulerStatsTests: XCTestCase {

    func testSchedulerStatsZero() {
        let stats = SchedulerStats(
            totalScheduled: 0,
            totalCompleted: 0,
            totalFailed: 0,
            totalTokensGenerated: 0,
            totalPrefillMs: 0,
            totalDecodeMs: 0,
            activeSessions: 0,
            workerLoad: [:],
            pendingPrefills: 0,
            pendingDecodes: 0
        )
        XCTAssertEqual(stats.avgPrefillMs, 0)
        XCTAssertEqual(stats.avgDecodeMs, 0)
        XCTAssertEqual(stats.avgTokensPerRequest, 0)
    }

    func testSchedulerStatsComputed() {
        let stats = SchedulerStats(
            totalScheduled: 12,
            totalCompleted: 10,
            totalFailed: 2,
            totalTokensGenerated: 500,
            totalPrefillMs: 100,
            totalDecodeMs: 2000,
            activeSessions: 5,
            workerLoad: [0: 3, 1: 2],
            pendingPrefills: 1,
            pendingDecodes: 3
        )
        XCTAssertEqual(stats.avgPrefillMs, 10, accuracy: 0.01)
        XCTAssertEqual(stats.avgDecodeMs, 200, accuracy: 0.01)
        XCTAssertEqual(stats.avgTokensPerRequest, 50, accuracy: 0.01)
        XCTAssertEqual(stats.pendingPrefills, 1)
        XCTAssertEqual(stats.pendingDecodes, 3)
        XCTAssertEqual(stats.totalFailed, 2)
    }
}

// MARK: - Phase 4: Shared Memory Config Tests

final class SharedMemoryConfigTests: XCTestCase {

    func testInferenceConfigShmDefaults() {
        let config = InferenceConfig(modelPath: "/tmp/test.gguf")
        XCTAssertFalse(config.useSharedMemory)
        XCTAssertEqual(config.sharedMemorySlotSize, 65536)
    }

    func testInferenceConfigShmEnabled() {
        let config = InferenceConfig(
            modelPath: "/tmp/test.gguf",
            useSharedMemory: true,
            sharedMemorySlotSize: 131072
        )
        XCTAssertTrue(config.useSharedMemory)
        XCTAssertEqual(config.sharedMemorySlotSize, 131072)
    }

    func testInferenceConfigShmPreservesOtherDefaults() {
        let config = InferenceConfig(
            modelPath: "/tmp/test.gguf",
            useSharedMemory: true
        )
        XCTAssertEqual(config.contextSize, 4096)
        XCTAssertEqual(config.workerCount, 2)
        XCTAssertEqual(config.maxSessionsPerWorker, 8)
        XCTAssertEqual(config.blasThreads, 1)
    }
}

// MARK: - Phase 4: Shared Memory Result Buffer Tests

final class ShmResultBufferTests: XCTestCase {

    func testPythonStringLiteralEscaping() {
        // Test basic string
        let basic = LlamaSessionKernel.pythonStringLiteral("hello")
        XCTAssertEqual(basic, "\"hello\"")

        // Test string with single quotes (apostrophes pass through unescaped)
        let quotes = LlamaSessionKernel.pythonStringLiteral("it's a test")
        XCTAssertEqual(quotes, "\"it's a test\"")

        // Test string with double quotes
        let dblQuotes = LlamaSessionKernel.pythonStringLiteral("he said \"hello\"")
        XCTAssertEqual(dblQuotes, "\"he said \\\"hello\\\"\"")

        // Test string with backslashes
        let backslash = LlamaSessionKernel.pythonStringLiteral("path\\to\\file")
        XCTAssertEqual(backslash, "\"path\\\\to\\\\file\"")

        // Test string with newlines
        let newline = LlamaSessionKernel.pythonStringLiteral("line1\nline2")
        XCTAssertEqual(newline, "\"line1\\nline2\"")
    }
}

final class AggregateStatsTests: XCTestCase {

    func testAggregateStatsZero() {
        let stats = AggregateStats(
            totalRequests: 0,
            totalPrefillMs: 0,
            totalDecodeMs: 0,
            totalCompletionTokens: 0,
            activeSessions: 0,
            workerLoad: [:]
        )
        XCTAssertEqual(stats.avgPrefillMs, 0)
        XCTAssertEqual(stats.avgDecodeMs, 0)
        XCTAssertEqual(stats.avgTokensPerRequest, 0)
    }

    func testAggregateStatsComputed() {
        let stats = AggregateStats(
            totalRequests: 10,
            totalPrefillMs: 500,
            totalDecodeMs: 2000,
            totalCompletionTokens: 1000,
            activeSessions: 3,
            workerLoad: [0: 2, 1: 1]
        )
        XCTAssertEqual(stats.avgPrefillMs, 50, accuracy: 0.01)
        XCTAssertEqual(stats.avgDecodeMs, 200, accuracy: 0.01)
        XCTAssertEqual(stats.avgTokensPerRequest, 100, accuracy: 0.01)
    }
}

final class VLMCaptionResultTests: XCTestCase {

    private func makeResult(caption: String) -> VLMCaptionResult {
        VLMCaptionResult(caption: caption, tokens: 10, durationMs: 100, error: nil)
    }

    func testFormattedCaptionFullStructuredJSON() {
        let json = """
        {"objects":["apple","table"],"colors":["red","brown"],"description":"A red apple sits on a wooden table."}
        """
        let result = makeResult(caption: json)
        let formatted = result.formattedCaption()
        XCTAssertTrue(formatted.contains("Objects: apple, table."), "Expected objects in output, got: \(formatted)")
        XCTAssertTrue(formatted.contains("Colors: red, brown."), "Expected colors in output, got: \(formatted)")
        XCTAssertTrue(formatted.contains("Description: A red apple sits on a wooden table."), "Expected description in output, got: \(formatted)")
    }

    func testFormattedCaptionObjectsAndDescriptionOnly() {
        let json = """
        {"objects":["cat"],"colors":[],"description":"A cat on a mat."}
        """
        let result = makeResult(caption: json)
        let formatted = result.formattedCaption()
        XCTAssertTrue(formatted.contains("Objects: cat."))
        XCTAssertFalse(formatted.contains("Colors:"), "Empty colors array should be omitted")
        XCTAssertTrue(formatted.contains("Description: A cat on a mat."))
    }

    func testFormattedCaptionFallsBackToRawStringForNonJSON() {
        let raw = "A detailed free-form description of the image."
        let result = makeResult(caption: raw)
        XCTAssertEqual(result.formattedCaption(), raw)
    }

    func testFormattedCaptionFallsBackForEmptyCaption() {
        let result = makeResult(caption: "")
        XCTAssertEqual(result.formattedCaption(), "")
    }

    func testFormattedCaptionFallsBackWhenJSONMissingExpectedKeys() {
        let json = "{\"foo\":\"bar\"}"
        let result = makeResult(caption: json)
        XCTAssertEqual(result.formattedCaption(), json, "Unknown JSON should be returned as-is")
    }

    func testFormattedCaptionHandlesJSONWithExtraWhitespace() {
        let json = """
        {
          "objects": ["book"],
          "colors": ["blue"],
          "description": "A blue book on a shelf."
        }
        """
        let result = makeResult(caption: json)
        let formatted = result.formattedCaption()
        XCTAssertTrue(formatted.contains("Objects: book."))
        XCTAssertTrue(formatted.contains("Colors: blue."))
        XCTAssertTrue(formatted.contains("Description: A blue book on a shelf."))
    }
}

final class ModelShortNameFormatterTests: XCTestCase {

    func testLlamaShortNameKeepsVersionOnly() {
        let short = ModelShortNameFormatter.shortName(fromFilename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        XCTAssertEqual(short, "Llama 3.2")
    }

    func testQwenShortNameUsesFamilyToken() {
        let short = ModelShortNameFormatter.shortName(fromFilename: "Qwen3-4B-Instruct-Q4_K_M.gguf")
        XCTAssertEqual(short, "Qwen3")
    }

    func testGemmaShortNameKeepsMajorVersion() {
        let short = ModelShortNameFormatter.shortName(fromFilename: "gemma-3-4b-it-Q4_K_M.gguf")
        XCTAssertEqual(short, "Gemma 3")
    }

    func testFallbackWhenNoKnownFamily() {
        let short = ModelShortNameFormatter.shortName(fromFilename: "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")
        XCTAssertEqual(short, "Deepseek")
    }
}

