import Foundation

/// CodeAct agent — system prompt and protocol utilities.
///
/// Specification: "Executable Code Actions Elicit Better LLM Agents" (arxiv 2402.01030v4)
/// System prompt is verbatim from Appendix E (zero-shot, chatML format).
/// Action protocol: model emits `<execute>…</execute>` blocks; caller runs them in a
/// Python REPL and feeds back `Observation:\n<stdout>` as the next user turn.
public enum CodeActAgent {

    /// Verbatim zero-shot system prompt from CodeAct paper Appendix E.
    public static let systemPrompt = """
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    The assistant can interact with an interactive Python (Jupyter Notebook) environment and receive the corresponding output when needed. The code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>.
    The assistant should attempt fewer things at a time instead of putting too much code in one <execute> block. The assistant can install packages through PIP by <execute> !pip install [package needed] </execute> and should always import packages and define variables before starting to use them.
    The assistant should stop <execute> and provide an answer when they have already obtained the answer from the execution result. Whenever possible, execute the code for the user using <execute> instead of providing it.
    The assistant's response should be concise, but do express their thoughts.
    """

    /// Maximum agent loop iterations before forcing termination (safety guard).
    public static let maxIterations = 10

    /// Extract the first `<execute>…</execute>` block from model output.
    /// Returns trimmed Python code, or nil if no block is present.
    public static func parseExecuteBlock(_ text: String) -> String? {
        guard let openRange = text.range(of: "<execute>"),
              let closeRange = text.range(of: "</execute>",
                                          range: openRange.upperBound..<text.endIndex) else {
            return nil
        }
        let code = String(text[openRange.upperBound..<closeRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return code.isEmpty ? nil : code
    }

    /// Format a `RunOutput` as a CodeAct `Observation:` turn injected back to the model.
    public static func formatObservation(_ output: RunOutput) -> String {
        var parts: [String] = []
        let stdout = output.stdout.trimmingCharacters(in: .newlines)
        let stderr = output.stderr.trimmingCharacters(in: .newlines)
        if !stdout.isEmpty { parts.append(stdout) }
        if !stderr.isEmpty { parts.append("[stderr]\n\(stderr)") }
        if let error = output.error?.trimmingCharacters(in: .newlines), !error.isEmpty {
            parts.append("[error]\n\(error)")
        }
        let body = parts.isEmpty ? "(no output)" : parts.joined(separator: "\n")
        return "Observation:\n\(body)"
    }
}
