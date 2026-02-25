import XCTest
@testable import LlamaChatUI
@testable import LlamaInferenceCore
@testable import ChatStorage

@MainActor
final class ToolbarMenuBehaviorTests: XCTestCase {

    func testResponseModeAutoKeepsBaseSampling() {
        let base = SamplingParams(
            maxTokens: 768,
            temperature: 0.62,
            topP: 0.91,
            topK: 37,
            repeatPenalty: 1.12,
            stop: ["END"]
        )

        let result = ChatViewModel.applyResponseMode(.auto, base: base, contextSize: 4096)

        XCTAssertEqual(result.maxTokens, base.maxTokens)
        XCTAssertEqual(result.temperature, base.temperature, accuracy: 0.0001)
        XCTAssertEqual(result.topP, base.topP, accuracy: 0.0001)
        XCTAssertEqual(result.topK, base.topK)
        XCTAssertEqual(result.repeatPenalty, base.repeatPenalty, accuracy: 0.0001)
        XCTAssertEqual(result.stop, base.stop)
    }

    func testResponseModeInstantClampsSampling() {
        let base = SamplingParams(
            maxTokens: 1024,
            temperature: 0.90,
            topP: 0.98,
            topK: 60,
            repeatPenalty: 1.10
        )

        let result = ChatViewModel.applyResponseMode(.instant, base: base, contextSize: 4096)

        XCTAssertEqual(result.maxTokens, 256) // context / 16
        XCTAssertEqual(result.temperature, 0.35, accuracy: 0.0001)
        XCTAssertEqual(result.topP, 0.90, accuracy: 0.0001)
        XCTAssertEqual(result.topK, 30)
        XCTAssertEqual(result.repeatPenalty, 1.10, accuracy: 0.0001)
    }

    func testResponseModeThinkingExpandsSampling() {
        let base = SamplingParams(
            maxTokens: 128,
            temperature: 0.20,
            topP: 0.80,
            topK: 20,
            repeatPenalty: 1.10
        )

        let result = ChatViewModel.applyResponseMode(.thinking, base: base, contextSize: 4096)

        XCTAssertEqual(result.maxTokens, 1024) // max(512, context / 4)
        XCTAssertEqual(result.temperature, 0.70, accuracy: 0.0001)
        XCTAssertEqual(result.topP, 0.95, accuracy: 0.0001)
        XCTAssertEqual(result.topK, 40)
        XCTAssertEqual(result.repeatPenalty, 1.10, accuracy: 0.0001)
    }

    func testTemporaryChatToggleClearsMessages() async {
        let viewModel = ChatViewModel()
        viewModel.messages = [ChatMessage(role: .user, content: "hello")]
        XCTAssertFalse(viewModel.temporaryChatEnabled)

        await viewModel.setTemporaryChatEnabled(true)

        XCTAssertTrue(viewModel.temporaryChatEnabled)
        XCTAssertTrue(viewModel.messages.isEmpty)
    }
}
