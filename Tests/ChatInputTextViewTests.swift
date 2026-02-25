import XCTest
@testable import ChatUIComponents
import AppKit

@MainActor
final class ChatInputTextViewTests: XCTestCase {

    // MARK: - Return Sends (calls onSubmit)

    func testReturnKeyCallsOnSubmit() {
        let textView = SubmitTextView()
        textView.string = "hello"

        var onSubmitCalled = false
        textView.onSubmit = { onSubmitCalled = true }

        let event = NSEvent.keyEvent(
            with: .keyDown,
            location: .zero,
            modifierFlags: [],
            timestamp: 0,
            windowNumber: 0,
            context: nil,
            characters: "\r",
            charactersIgnoringModifiers: "\r",
            isARepeat: false,
            keyCode: 36
        )!
        textView.keyDown(with: event)

        XCTAssertTrue(onSubmitCalled)
        XCTAssertEqual(textView.string, "hello")
    }

    func testReturnKeyWithEmptyTextCallsOnSubmit() {
        let textView = SubmitTextView()
        textView.string = ""

        var onSubmitCalled = false
        textView.onSubmit = { onSubmitCalled = true }

        let event = NSEvent.keyEvent(
            with: .keyDown,
            location: .zero,
            modifierFlags: [],
            timestamp: 0,
            windowNumber: 0,
            context: nil,
            characters: "\r",
            charactersIgnoringModifiers: "\r",
            isARepeat: false,
            keyCode: 36
        )!
        textView.keyDown(with: event)

        XCTAssertTrue(onSubmitCalled)
    }

    // MARK: - Shift+Return Inserts Newline

    func testShiftReturnInsertsNewline() {
        let textView = SubmitTextView()
        textView.string = "line1"
        textView.selectedRange = NSRange(location: 5, length: 0) // cursor at end

        var onSubmitCalled = false
        textView.onSubmit = { onSubmitCalled = true }

        let event = NSEvent.keyEvent(
            with: .keyDown,
            location: .zero,
            modifierFlags: .shift,
            timestamp: 0,
            windowNumber: 0,
            context: nil,
            characters: "\r",
            charactersIgnoringModifiers: "\r",
            isARepeat: false,
            keyCode: 36
        )!
        textView.keyDown(with: event)

        XCTAssertFalse(onSubmitCalled)
        XCTAssertEqual(textView.string, "line1\n")
    }

    func testShiftReturnInsertsNewlineInMiddleOfText() {
        let textView = SubmitTextView()
        textView.string = "ab"
        textView.selectedRange = NSRange(location: 1, length: 0) // cursor between a and b

        var onSubmitCalled = false
        textView.onSubmit = { onSubmitCalled = true }

        let event = NSEvent.keyEvent(
            with: .keyDown,
            location: .zero,
            modifierFlags: .shift,
            timestamp: 0,
            windowNumber: 0,
            context: nil,
            characters: "\r",
            charactersIgnoringModifiers: "\r",
            isARepeat: false,
            keyCode: 36
        )!
        textView.keyDown(with: event)

        XCTAssertFalse(onSubmitCalled)
        XCTAssertEqual(textView.string, "a\nb")
    }

    // MARK: - Placeholder Visibility

    func testPlaceholderVisibilityRequiresCoordinator() {
        // Placeholder visibility is managed by the Coordinator which needs a full NSView hierarchy.
        // We test the underlying logic: empty string → placeholder visible; non-empty → hidden.
        // The Coordinator's updatePlaceholder sets placeholderView?.isHidden = !textView.string.isEmpty.
        // For a unit test without instantiating the full ChatInputTextView, we verify SubmitTextView
        // key behavior (done above) and that the text binding contract holds.
        // A full integration test would require hosting the SwiftUI view in a window.
        // For now we document the expected behavior.
        let textView = SubmitTextView()
        XCTAssertTrue(textView.string.isEmpty)
    }
}
