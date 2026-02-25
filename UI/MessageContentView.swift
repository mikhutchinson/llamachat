import SwiftUI
import AppKit
import Textual
import ChatStorage
import ChatUIComponents
import LlamaInferenceCore

// MARK: - Code Block Tracker

/// Sequential tracker used to assign each rendered code block its parsed-source index.
/// Created per-message-render; each `CodeBlockWithCopyView` claims the next index on appear.
@MainActor
final class CodeBlockTracker: @unchecked Sendable {
    let blocks: [FencedBlock]
    private var cursor: Int = 0

    init(blocks: [FencedBlock]) {
        self.blocks = blocks
    }

    /// Claim the next sequential index.
    func claimNext() -> Int {
        let idx = cursor
        cursor += 1
        return idx
    }
}

// MARK: - Environment Keys

private struct CodeBlockTrackerKey: EnvironmentKey {
    static let defaultValue: CodeBlockTracker? = nil
}

private struct RunCodeHandlerKey: EnvironmentKey {
    static let defaultValue: (@MainActor (String) async -> RunOutput?)? = nil
}

private struct SandboxReadyKey: EnvironmentKey {
    static let defaultValue: Bool = false
}

private struct SendCodeToComposerKey: EnvironmentKey {
    static let defaultValue: (@MainActor (String) -> Void)? = nil
}

/// Async handler for code follow-up actions that may need AST analysis.
private struct CodeActionHandlerKey: EnvironmentKey {
    static let defaultValue: (@MainActor (CodeFollowUpAction, String, String?) async -> Void)? = nil
}

extension EnvironmentValues {
    var codeBlockTracker: CodeBlockTracker? {
        get { self[CodeBlockTrackerKey.self] }
        set { self[CodeBlockTrackerKey.self] = newValue }
    }
    var runCodeHandler: (@MainActor (String) async -> RunOutput?)? {
        get { self[RunCodeHandlerKey.self] }
        set { self[RunCodeHandlerKey.self] = newValue }
    }
    var isSandboxReady: Bool {
        get { self[SandboxReadyKey.self] }
        set { self[SandboxReadyKey.self] = newValue }
    }
    /// Closure that populates the composer with a prompt (e.g. code + error for "Fix with AI").
    var sendCodeToComposer: (@MainActor (String) -> Void)? {
        get { self[SendCodeToComposerKey.self] }
        set { self[SendCodeToComposerKey.self] = newValue }
    }
    /// Async handler for code follow-up actions (Explain, Review, Improve, etc.).
    var codeActionHandler: (@MainActor (CodeFollowUpAction, String, String?) async -> Void)? {
        get { self[CodeActionHandlerKey.self] }
        set { self[CodeActionHandlerKey.self] = newValue }
    }
}

// MARK: - Code Block Style (GitHub-style with per-block copy + run buttons)

extension StructuredText {
    /// Code block style with copy and optional run buttons in the top-right (GitHub-style).
    struct CodeBlockWithCopyStyle: CodeBlockStyle {
        @Environment(\.theme) private var theme: ThemeColors

        func makeBody(configuration: Configuration) -> some View {
            CodeBlockWithCopyView(configuration: configuration, theme: theme)
        }
    }
}

private struct CodeBlockWithCopyView: View {
    let configuration: StructuredText.CodeBlockStyleConfiguration
    let theme: ThemeColors
    @State private var copied = false
    @State private var blockIndex: Int?
    @State private var runOutput: RunOutput?
    @State private var isRunning = false

    @Environment(\.codeBlockTracker) private var tracker
    @Environment(\.runCodeHandler) private var runHandler
    @Environment(\.isSandboxReady) private var sandboxReady
    @Environment(\.sendCodeToComposer) private var sendToComposer
    @Environment(\.codeActionHandler) private var codeActionHandler

    /// Whether this code block is a Python block eligible for sandbox execution.
    private var isPythonBlock: Bool {
        guard let hint = configuration.languageHint?.lowercased() else { return false }
        return hint == "python" || hint == "py"
    }

    /// Whether to show the Run button: must be Python, sandbox must be ready.
    private var showRunButton: Bool {
        isPythonBlock && sandboxReady && runHandler != nil
    }

    /// Resolve the code text for this block from Textual's rendered proxy first
    /// (authoritative for what the user sees), then fall back to tracker-parsed
    /// fenced-block source when proxy extraction fails.
    private var resolvedCode: String? {
        if let proxyCode = codeFromProxy(), !proxyCode.isEmpty {
            return proxyCode
        }
        if let idx = blockIndex, let tracker, idx < tracker.blocks.count {
            return tracker.blocks[idx].code
        }
        return nil
    }

    /// Fallback: extract the plain-text code from `CodeBlockProxy`'s private
    /// `content: AttributedSubstring` via `Mirror`.  This covers cases where
    /// `FencedBlockParser` and Textual's CommonMark parser disagree on the
    /// number or order of code blocks.
    private func codeFromProxy() -> String? {
        let mirror = Mirror(reflecting: configuration.codeBlock)
        guard let child = mirror.children.first(where: { $0.label == "content" }) else { return nil }
        let value = child.value
        // AttributedSubstring → String via its `characters` view
        if let attrSub = value as? AttributedSubstring {
            return String(attrSub.characters)
        }
        return nil
    }

    /// Whether the last execution produced an error worth sending to the LLM.
    private var hasExecutionError: Bool {
        guard let output = runOutput else { return false }
        return output.error != nil || !output.stderr.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ZStack(alignment: .topTrailing) {
                Overflow {
                    configuration.label
                        .textual.lineSpacing(.fontScaled(0.225))
                        .textual.fontScale(0.85)
                        .fixedSize(horizontal: false, vertical: true)
                        .monospaced()
                        .padding(16)
                }

                // Button cluster — copy always, run conditionally, fix on error
                HStack(spacing: 2) {
                    // Fix with AI — appears after a failed run
                    if hasExecutionError, let code = resolvedCode, let sendToComposer {
                        Button {
                            sendToComposer(composeFixPrompt(code: code, output: runOutput!))
                        } label: {
                            Image(systemName: "sparkles")
                                .font(.system(size: 11))
                                .foregroundColor(.orange)
                        }
                        .buttonStyle(.plain)
                        .help("Fix with AI")
                        .padding(8)
                        .zIndex(2)
                    }

                    if showRunButton {
                        Button {
                            guard !isRunning else { return }
                            guard let code = resolvedCode else {
                                runOutput = RunOutput(
                                    stdout: "", stderr: "", figures: [],
                                    error: "Could not resolve code for this block (tracker: \(blockIndex.map(String.init) ?? "nil"), blocks: \(tracker?.blocks.count ?? -1))",
                                    elapsedMs: 0
                                )
                                return
                            }
                            isRunning = true
                            Task {
                                runOutput = await runHandler?(code)
                                isRunning = false
                            }
                        } label: {
                            if isRunning {
                                ProgressView()
                                    .controlSize(.mini)
                                    .frame(width: 11, height: 11)
                            } else {
                                Image(systemName: runOutput != nil ? "arrow.clockwise" : "play.fill")
                                    .font(.system(size: 11))
                                    .foregroundColor(theme.accent)
                            }
                        }
                        .buttonStyle(.plain)
                        .help(runOutput != nil ? "Re-run code" : "Run code")
                        .padding(8)
                        .zIndex(2)
                    }

                    // Ask AI — overflow menu for code follow-up actions
                    if let code = resolvedCode, codeActionHandler != nil {
                        askAIMenu(code: code)
                    }

                    Button {
                        configuration.codeBlock.copyToPasteboard()
                        copied = true
                        Task {
                            try? await Task.sleep(for: .seconds(1.5))
                            copied = false
                        }
                    } label: {
                        Image(systemName: copied ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 11))
                            .foregroundColor(copied ? theme.accent : theme.textTertiary)
                    }
                    .buttonStyle(.plain)
                    .help("Copy code")
                    .padding(8)
                    .zIndex(2)
                }
                .zIndex(2)
            }
            .background(theme.thinkingBg)
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .stroke(theme.divider, lineWidth: 0.5)
            }

            // Inline output panel (shown after execution)
            if let output = runOutput {
                SandboxOutputPanel(output: output, theme: theme)
            }
        }
        .textual.blockSpacing(.init(top: 0, bottom: 16))
        .onAppear {
            if blockIndex == nil {
                blockIndex = tracker?.claimNext()
            }
        }
    }

    private func askAIMenu(code: String) -> some View {
        let lang = configuration.languageHint?.lowercased()
        return Menu {
            Button {
                Task { await codeActionHandler?(.explain, code, lang) }
            } label: {
                Label(CodeFollowUpAction.explain.label, systemImage: CodeFollowUpAction.explain.systemImage)
            }
            Button {
                Task { await codeActionHandler?(.review, code, lang) }
            } label: {
                Label(CodeFollowUpAction.review.label, systemImage: CodeFollowUpAction.review.systemImage)
            }
            Button {
                Task { await codeActionHandler?(.improve, code, lang) }
            } label: {
                Label(CodeFollowUpAction.improve.label, systemImage: CodeFollowUpAction.improve.systemImage)
            }
            if isPythonBlock {
                Button {
                    Task { await codeActionHandler?(.writeTests, code, lang) }
                } label: {
                    Label(CodeFollowUpAction.writeTests.label, systemImage: CodeFollowUpAction.writeTests.systemImage)
                }
            }
            if !isPythonBlock {
                Button {
                    Task { await codeActionHandler?(.translateToPython, code, lang) }
                } label: {
                    Label(CodeFollowUpAction.translateToPython.label, systemImage: CodeFollowUpAction.translateToPython.systemImage)
                }
            }
        } label: {
            Image(systemName: "ellipsis.bubble")
                .font(.system(size: 11))
                .foregroundColor(theme.textTertiary)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Ask AI")
        .padding(8)
        .zIndex(2)
    }

    private func composeFixPrompt(code: String, output: RunOutput) -> String {
        var parts: [String] = ["The following Python code produced an error. Please fix it:"]
        parts.append("\n```python\n\(code)\n```")
        if let error = output.error {
            parts.append("\nError:\n```\n\(error)\n```")
        }
        if !output.stderr.isEmpty {
            parts.append("\nStderr:\n```\n\(output.stderr)\n```")
        }
        if !output.stdout.isEmpty {
            parts.append("\nStdout (partial):\n```\n\(output.stdout.prefix(500))\n```")
        }
        return parts.joined(separator: "\n")
    }
}

// MARK: - Sandbox Output Panel

/// Compact inline output rendered below an executed code block.
private struct SandboxOutputPanel: View {
    let output: RunOutput
    let theme: ThemeColors

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !output.stdout.isEmpty {
                outputSection(label: "stdout", text: output.stdout, color: theme.textPrimary)
            }
            if !output.stderr.isEmpty {
                outputSection(label: "stderr", text: output.stderr, color: .orange)
            }
            if let error = output.error {
                outputSection(label: "error", text: error, color: .red)
            }
            ForEach(Array(output.figures.enumerated()), id: \.offset) { _, pngData in
                if let nsImage = NSImage(data: pngData) {
                    Image(nsImage: nsImage)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: 480)
                        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                }
            }
            Text("\(output.elapsedMs)ms")
                .font(.system(size: 10))
                .monospacedDigit()
                .foregroundColor(theme.textTertiary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(theme.thinkingBg.opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(theme.divider.opacity(0.5), lineWidth: 0.5)
        }
        .padding(.top, 4)
    }

    private func outputSection(label: String, text: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundColor(theme.textTertiary)
                .textCase(.uppercase)
            Text(text)
                .font(.system(size: 12))
                .monospaced()
                .foregroundColor(color)
                .textSelection(.enabled)
        }
    }
}

extension StructuredText.CodeBlockStyle where Self == StructuredText.CodeBlockWithCopyStyle {
    static var withCopyButton: Self { .init() }
}

// MARK: - Render Cache

/// Thread-safe LRU cache for normalized markdown strings keyed by message UUID.
/// Bounded to avoid unbounded memory growth on long conversation histories.
final class MarkdownRenderCache: @unchecked Sendable {
    static let shared = MarkdownRenderCache()

    private let cache = NSCache<NSUUID, NSString>()

    private init() {
        cache.countLimit = 500
    }

    func get(_ id: UUID) -> String? {
        cache.object(forKey: id as NSUUID) as String?
    }

    func set(_ id: UUID, value: String) {
        cache.setObject(value as NSString, forKey: id as NSUUID)
    }
}

// MARK: - MessageContentView

/// Renders message content as structured markdown with Textual.
/// Preprocessing remains synchronous (regex-only) so each message renders in a
/// single layout pass without a plain-text → rich-text swap.
struct MessageContentView: View, Equatable {
    let messageID: UUID
    let content: String
    let fontSize: CGFloat
    let role: MessageRole

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.messageID == rhs.messageID
            && lhs.content == rhs.content
            && lhs.fontSize == rhs.fontSize
            && lhs.role == rhs.role
    }

    @Environment(\.theme) private var theme
    @State private var processedContent: String

    init(messageID: UUID, content: String, fontSize: CGFloat, role: MessageRole) {
        self.messageID = messageID
        self.content = content
        self.fontSize = fontSize
        self.role = role

        let cached = MarkdownRenderCache.shared.get(messageID)
        _processedContent = State(initialValue: cached ?? LatexPreprocessor.preprocess(content))
        if cached == nil {
            MarkdownRenderCache.shared.set(messageID, value: _processedContent.wrappedValue)
        }
    }

    var body: some View {
        let tracker = CodeBlockTracker(blocks: FencedBlockParser.extract(from: processedContent))

        StructuredText(
            markdown: processedContent,
            syntaxExtensions: [.math]
        )
        .font(.system(size: fontSize))
        .foregroundStyle(theme.textPrimary)
        .tint(theme.accent)
        // Set custom code block style before and after the bundled style to make
        // precedence explicit regardless of environment resolution order.
        .textual.codeBlockStyle(.withCopyButton)
        .textual.structuredTextStyle(.gitHub)
        .textual.codeBlockStyle(.withCopyButton)
        .textual.inlineStyle(
            .gitHub
                .code(.monospaced, .fontScale(0.88), .foregroundColor(theme.accent))
                .link(.foregroundColor(theme.accent))
        )
        .textual.mathProperties(
            MathProperties(
                fontName: .latinModern,
                fontScale: 1.15,
                textAlignment: .leading
            )
        )
        .textual.textSelection(.enabled)
        .environment(\.codeBlockTracker, tracker)
        .onChange(of: content) { _, newContent in
            let processed = LatexPreprocessor.preprocess(newContent)
            MarkdownRenderCache.shared.set(messageID, value: processed)
            processedContent = processed
        }
    }
}
