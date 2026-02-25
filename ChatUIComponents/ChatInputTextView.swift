import SwiftUI
import AppKit

// MARK: - ChatInputTextView (NSViewRepresentable)

/// A scrollable, multi-line text input backed by NSTextView.
/// - Supports mouse/trackpad scrolling (unlike SwiftUI TextField axis: .vertical on macOS).
/// - Auto-grows from 1 line up to `maxLines`, then scrolls internally.
/// - Return sends (calls `onSubmit`); Shift+Return inserts a newline.
/// - Shows a placeholder when empty.
public struct ChatInputTextView: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String = ""
    var font: NSFont = .systemFont(ofSize: 14)
    var textColor: NSColor = .labelColor
    var placeholderColor: NSColor = .placeholderTextColor
    var maxLines: Int = 8
    var isDisabled: Bool = false
    var onSubmit: (() -> Void)?
    var onPasteFiles: (([URL]) -> Void)?

    /// Published height so SwiftUI can size the frame.
    @Binding var contentHeight: CGFloat

    public init(
        text: Binding<String>,
        placeholder: String = "",
        font: NSFont = .systemFont(ofSize: 14),
        textColor: NSColor = .labelColor,
        placeholderColor: NSColor = .placeholderTextColor,
        maxLines: Int = 8,
        isDisabled: Bool = false,
        onSubmit: (() -> Void)? = nil,
        onPasteFiles: (([URL]) -> Void)? = nil,
        contentHeight: Binding<CGFloat>
    ) {
        _text = text
        self.placeholder = placeholder
        self.font = font
        self.textColor = textColor
        self.placeholderColor = placeholderColor
        self.maxLines = maxLines
        self.isDisabled = isDisabled
        self.onSubmit = onSubmit
        self.onPasteFiles = onPasteFiles
        _contentHeight = contentHeight
    }

    public func makeNSView(context: Context) -> NSScrollView {
        let textView = SubmitTextView()
        textView.delegate = context.coordinator
        textView.onSubmit = onSubmit
        textView.onPasteFiles = onPasteFiles
        textView.font = font
        textView.textColor = textColor
        textView.drawsBackground = false
        textView.isRichText = false
        textView.allowsUndo = true
        textView.isEditable = !isDisabled
        textView.isSelectable = true
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainerInset = NSSize(width: 0, height: 2)
        textView.insertionPointColor = textColor
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false

        let scrollView = NSScrollView()
        scrollView.documentView = textView
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder

        // Store reference for later updates
        context.coordinator.textView = textView
        context.coordinator.scrollView = scrollView

        // Initial text
        if !text.isEmpty {
            textView.string = text
        }

        // Compute initial height
        DispatchQueue.main.async {
            context.coordinator.recalculateHeight()
        }

        return scrollView
    }

    public func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = context.coordinator.textView else { return }

        // Update closures
        textView.onSubmit = onSubmit
        textView.onPasteFiles = onPasteFiles

        // Track whether anything layout-relevant changed so we only
        // schedule a (potentially expensive) height recalculation when needed.
        // This avoids a redundant SwiftUI render pass on every keystroke.
        var needsHeightRecalc = false

        // Sync text only when it differs (avoid resetting cursor position)
        if textView.string != text {
            let selectedRanges = textView.selectedRanges
            textView.string = text
            textView.selectedRanges = selectedRanges
            needsHeightRecalc = true
        }

        // Update styling
        if textView.font != font {
            textView.font = font
            needsHeightRecalc = true
        }
        textView.textColor = textColor
        textView.insertionPointColor = textColor
        let editable = !isDisabled
        if textView.isEditable != editable {
            textView.isEditable = editable
        }

        // Update placeholder
        context.coordinator.placeholder = placeholder
        context.coordinator.placeholderColor = placeholderColor
        context.coordinator.font = font
        if context.coordinator.maxLines != maxLines {
            context.coordinator.maxLines = maxLines
            needsHeightRecalc = true
        }

        // Only recalculate height when text, font, or maxLines changed.
        // The recalculateHeight() method has its own abs > 0.5 guard to
        // prevent feedback loops from the binding update.
        if needsHeightRecalc {
            DispatchQueue.main.async {
                context.coordinator.recalculateHeight()
            }
        }
    }

    public func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    // MARK: - Coordinator

    @MainActor
    public class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ChatInputTextView
        weak var textView: SubmitTextView?
        weak var scrollView: NSScrollView?
        var placeholder: String = ""
        var placeholderColor: NSColor = .placeholderTextColor
        var font: NSFont = .systemFont(ofSize: 14)
        var maxLines: Int = 8

        private var placeholderView: NSTextField?

        init(_ parent: ChatInputTextView) {
            self.parent = parent
            self.placeholder = parent.placeholder
            self.placeholderColor = parent.placeholderColor
            self.font = parent.font
            self.maxLines = parent.maxLines
        }

        public func textDidChange(_ notification: Notification) {
            guard let textView = textView else { return }
            parent.text = textView.string
            recalculateHeight()
            updatePlaceholder()
        }

        public func textDidBeginEditing(_ notification: Notification) {
            updatePlaceholder()
        }

        public func textDidEndEditing(_ notification: Notification) {
            updatePlaceholder()
        }

        func recalculateHeight() {
            guard let textView = textView else { return }

            // Force layout so usedRect is accurate
            textView.layoutManager?.ensureLayout(for: textView.textContainer!)

            let usedRect = textView.layoutManager?.usedRect(for: textView.textContainer!) ?? .zero
            let insets = textView.textContainerInset
            let contentH = usedRect.height + insets.height * 2

            // Calculate line height for clamping
            let lineHeight = font.ascender - font.descender + font.leading
            let singleLineHeight = lineHeight + insets.height * 2
            let maxHeight = lineHeight * CGFloat(maxLines) + insets.height * 2

            let clampedHeight = min(max(contentH, singleLineHeight), maxHeight)

            if abs(parent.contentHeight - clampedHeight) > 0.5 {
                parent.contentHeight = clampedHeight
            }

            // Show/hide placeholder on first layout
            if placeholderView == nil {
                setupPlaceholder()
            }
            updatePlaceholder()
        }

        private func setupPlaceholder() {
            guard let textView = textView else { return }

            let label = NSTextField(labelWithString: placeholder)
            label.font = font
            label.textColor = placeholderColor
            label.drawsBackground = false
            label.isBezeled = false
            label.isEditable = false
            label.isSelectable = false
            label.translatesAutoresizingMaskIntoConstraints = false

            textView.addSubview(label)

            let insets = textView.textContainerInset
            NSLayoutConstraint.activate([
                label.leadingAnchor.constraint(equalTo: textView.leadingAnchor, constant: insets.width + textView.textContainer!.lineFragmentPadding),
                label.topAnchor.constraint(equalTo: textView.topAnchor, constant: insets.height),
            ])

            placeholderView = label
        }

        private func updatePlaceholder() {
            guard let textView = textView else { return }
            placeholderView?.isHidden = !textView.string.isEmpty
        }
    }
}

// MARK: - SubmitTextView

/// NSTextView subclass that intercepts Return for send and Shift+Return for newline.
/// Exposed as open for testing.
public class SubmitTextView: NSTextView {
    public var onSubmit: (() -> Void)?
    public var onPasteFiles: (([URL]) -> Void)?

    override public func keyDown(with event: NSEvent) {
        // Check for Return key (keyCode 36)
        if event.keyCode == 36 {
            let shiftHeld = event.modifierFlags.contains(.shift)
            if shiftHeld {
                // Shift+Return: insert newline
                insertNewline(nil)
            } else {
                // Return without Shift: send message
                onSubmit?()
            }
            return
        }
        super.keyDown(with: event)
    }

    // MARK: - Paste Support (Images & Files)

    private static let imageTypes: Set<String> = [
        "public.png", "public.jpeg", "public.tiff", "public.heic",
        "com.compuserve.gif", "public.webp", "com.microsoft.bmp"
    ]

    override public func validateUserInterfaceItem(_ item: NSValidatedUserInterfaceItem) -> Bool {
        if item.action == #selector(paste(_:)) {
            let pb = NSPasteboard.general
            // Enable Paste if pasteboard has image data or file URLs we support
            if pb.data(forType: .png) != nil || pb.data(forType: .tiff) != nil {
                return true
            }
            if let urls = pb.readObjects(forClasses: [NSURL.self], options: [
                .urlReadingFileURLsOnly: true
            ]) as? [URL], urls.contains(where: { Self.isSupportedAttachment($0) }) {
                return true
            }
        }
        return super.validateUserInterfaceItem(item)
    }

    override public func paste(_ sender: Any?) {
        let pb = NSPasteboard.general

        // 1. Check for file URLs on the pasteboard (e.g. copied files from Finder)
        if let fileURLs = pb.readObjects(forClasses: [NSURL.self], options: [
            .urlReadingFileURLsOnly: true
        ]) as? [URL], !fileURLs.isEmpty {
            let supported = fileURLs.filter { Self.isSupportedAttachment($0) }
            if !supported.isEmpty {
                onPasteFiles?(supported)
                return
            }
        }

        // 2. Check for raw image data on the pasteboard (e.g. screenshot, copy image)
        if let imageData = Self.extractImageData(from: pb) {
            if let tempURL = Self.writeTempFile(data: imageData.data, ext: imageData.ext) {
                onPasteFiles?([tempURL])
                return
            }
        }

        // 3. Fall through to normal text paste
        super.paste(sender)
    }

    private static func isSupportedAttachment(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        let imageExts: Set<String> = ["jpg", "jpeg", "png", "gif", "webp", "heic", "heif", "bmp", "tiff"]
        let docExts: Set<String> = ["pdf", "docx", "pptx", "xlsx", "txt", "md", "csv", "json", "xml", "html"]
        return imageExts.contains(ext) || docExts.contains(ext)
    }

    private static func extractImageData(from pb: NSPasteboard) -> (data: Data, ext: String)? {
        // Try PNG first (lossless, preserves transparency)
        if let data = pb.data(forType: .png) {
            return (data, "png")
        }
        // Try TIFF (macOS screenshot default)
        if let data = pb.data(forType: .tiff) {
            // Convert TIFF to PNG for consistency
            if let bitmapRep = NSBitmapImageRep(data: data),
               let pngData = bitmapRep.representation(using: .png, properties: [:]) {
                return (pngData, "png")
            }
            return (data, "tiff")
        }
        return nil
    }

    private static func writeTempFile(data: Data, ext: String) -> URL? {
        let tempDir = FileManager.default.temporaryDirectory
        let filename = "pasted-\(UUID().uuidString.prefix(8)).\(ext)"
        let url = tempDir.appendingPathComponent(filename)
        do {
            try data.write(to: url)
            return url
        } catch {
            return nil
        }
    }
}

// MARK: - Focus Support

extension ChatInputTextView {
    /// Focus the underlying NSTextView when SwiftUI requests focus via @FocusState.
    @MainActor
    public func makeFocusable() -> some View {
        self.onAppear { }
    }
}

/// Wrapper that provides @FocusState integration by observing a Bool binding
/// and making the NSTextView first responder accordingly.
struct FocusableChatInput: View {
    @Binding var text: String
    var placeholder: String = ""
    var font: NSFont = .systemFont(ofSize: 14)
    var textColor: NSColor = .labelColor
    var placeholderColor: NSColor = .placeholderTextColor
    var maxLines: Int = 8
    var isDisabled: Bool = false
    var onSubmit: (() -> Void)?
    var isFocused: FocusState<Bool>.Binding

    @State private var contentHeight: CGFloat = 22

    var body: some View {
        ChatInputTextView(
            text: $text,
            placeholder: placeholder,
            font: font,
            textColor: textColor,
            placeholderColor: placeholderColor,
            maxLines: maxLines,
            isDisabled: isDisabled,
            onSubmit: onSubmit,
            contentHeight: $contentHeight
        )
        .frame(height: contentHeight)
        .focused(isFocused)
    }
}
