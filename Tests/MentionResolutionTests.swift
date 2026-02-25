import XCTest
@testable import LlamaChatUI
@testable import LlamaInferenceCore

@MainActor
final class MentionResolutionTests: XCTestCase {

    func testResolveMixedDocsAndImageMentions() {
        let viewModel = ChatViewModel()
        viewModel._testResetSessionReferences()
        viewModel._testRegisterSessionReference(
            kind: .docs,
            filename: "report.pdf",
            mimeType: "application/pdf",
            extractedText: "report text"
        )
        viewModel._testRegisterSessionReference(
            kind: .img,
            filename: "photo.png",
            mimeType: "image/png",
            extractedText: "a caption"
        )

        let resolved = viewModel._testResolveMentions(
            prompt: "Summarize @docs(report.pdf) and @img(photo.png)"
        )

        XCTAssertEqual(resolved.cleanedPrompt, "Summarize and")
        XCTAssertEqual(resolved.resolvedAliases, ["report.pdf", "photo.png"])
        XCTAssertTrue(resolved.unresolved.isEmpty)
        XCTAssertTrue(resolved.hadMentions)
    }

    func testDuplicateFilenameGetsDeterministicAlias() {
        let viewModel = ChatViewModel()
        viewModel._testResetSessionReferences()
        viewModel._testRegisterSessionReference(
            kind: .docs,
            filename: "shared.pdf",
            mimeType: "application/pdf",
            extractedText: "first"
        )
        viewModel._testRegisterSessionReference(
            kind: .docs,
            filename: "shared.pdf",
            mimeType: "application/pdf",
            extractedText: "second"
        )

        let suggestions = viewModel._testMentionSuggestions(for: .docs)
        XCTAssertEqual(suggestions.map(\.alias), ["shared.pdf", "shared.pdf #2"])
    }

    func testUnresolvedMentionIsReported() {
        let viewModel = ChatViewModel()
        viewModel._testResetSessionReferences()
        viewModel._testRegisterSessionReference(
            kind: .docs,
            filename: "known.pdf",
            mimeType: "application/pdf",
            extractedText: "known"
        )

        let resolved = viewModel._testResolveMentions(prompt: "Check @docs(missing.pdf)")
        XCTAssertEqual(resolved.cleanedPrompt, "Check")
        XCTAssertEqual(resolved.resolvedAliases, [])
        XCTAssertEqual(resolved.unresolved, ["@docs(missing.pdf)"])
        XCTAssertTrue(resolved.hadMentions)
    }

    func testMentionOnlyPromptFallsBackToAttachmentDefault() {
        let viewModel = ChatViewModel()
        viewModel._testResetSessionReferences()
        viewModel._testRegisterSessionReference(
            kind: .docs,
            filename: "notes.txt",
            mimeType: "text/plain",
            extractedText: "notes"
        )

        let resolved = viewModel._testResolveMentions(prompt: "@docs(notes.txt)")
        let effectivePrompt = resolved.cleanedPrompt.isEmpty && resolved.hadMentions
            ? ComposerSendPolicy.attachmentOnlyDefaultPrompt
            : resolved.cleanedPrompt
        XCTAssertEqual(effectivePrompt, ComposerSendPolicy.attachmentOnlyDefaultPrompt)
    }

    func testTrailingDocsTriggerPopulatesSuggestionList() {
        let viewModel = ChatViewModel()
        viewModel._testResetSessionReferences()
        viewModel._testRegisterSessionReference(
            kind: .docs,
            filename: "report.pdf",
            mimeType: "application/pdf",
            extractedText: "body"
        )

        viewModel.composerState.inputText = "Use @docs"
        viewModel.composerInputDidChange(viewModel.composerState.inputText)

        XCTAssertEqual(viewModel.composerState.activeMentionKind, .docs)
        XCTAssertEqual(viewModel.composerState.mentionSuggestions.map(\.alias), ["report.pdf"])
    }

    func testSelectSuggestionInsertsMentionToken() {
        let viewModel = ChatViewModel()
        viewModel._testResetSessionReferences()
        viewModel._testRegisterSessionReference(
            kind: .docs,
            filename: "report.pdf",
            mimeType: "application/pdf",
            extractedText: "body"
        )

        viewModel.composerState.inputText = "Use @docs"
        viewModel.composerInputDidChange(viewModel.composerState.inputText)
        guard let suggestion = viewModel.composerState.mentionSuggestions.first else {
            XCTFail("Expected at least one mention suggestion")
            return
        }

        viewModel.selectMentionSuggestion(suggestion)

        XCTAssertEqual(viewModel.composerState.inputText, "Use @docs(report.pdf) ")
        XCTAssertNil(viewModel.composerState.activeMentionKind)
        XCTAssertTrue(viewModel.composerState.mentionSuggestions.isEmpty)
    }

    func testMentionPickerWithoutReferencesShowsInlineWarning() {
        let viewModel = ChatViewModel()
        viewModel._testResetSessionReferences()

        viewModel.showMentionPicker(kind: .docs)

        XCTAssertNil(viewModel.composerState.activeMentionKind)
        XCTAssertTrue(viewModel.composerState.mentionSuggestions.isEmpty)
        XCTAssertEqual(viewModel.composerState.inlineWarning, "No documents in this session yet.")
    }

    func testAttachmentOnlyFallbackCanAutoInjectCurrentAttachments() {
        XCTAssertTrue(
            ChatViewModel._testShouldAutoInjectCurrentAttachments(
                prompt: ComposerSendPolicy.attachmentOnlyDefaultPrompt,
                hadMentions: false,
                hasCurrentAttachments: true
            )
        )
        XCTAssertTrue(
            ChatViewModel._testShouldAutoInjectCurrentAttachments(
                prompt: "what is this document about?",
                hadMentions: false,
                hasCurrentAttachments: true
            )
        )
        XCTAssertFalse(
            ChatViewModel._testShouldAutoInjectCurrentAttachments(
                prompt: "Describe this.",
                hadMentions: true,
                hasCurrentAttachments: true
            )
        )
        XCTAssertFalse(
            ChatViewModel._testShouldAutoInjectCurrentAttachments(
                prompt: ComposerSendPolicy.attachmentOnlyDefaultPrompt,
                hadMentions: false,
                hasCurrentAttachments: false
            )
        )
    }
}
