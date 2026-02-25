import Foundation
import SwiftPythonRuntime

/// Narrative summarization kernel for context wind management.
///
/// Installed alongside `LlamaSessionKernel` on the same worker, sharing the
/// model instance. Produces a concise narrative summary of conversation history
/// for rehydration into a fresh session. Stateless — does not retain any session state.
enum SummarizationKernel {
    static let kernelSource = #"""
    import json, time, re, sys

    class SummarizationKernel:
        """Narrative conversation summarizer.

        Shares the Llama model instance with LlamaSessionKernel on the same
        worker. Stateless — each call is independent.
        """

        _CHARS_PER_TOKEN = 3.5

        NARRATIVE_SYSTEM = (
            "You are a conversation summarizer. Given a conversation history, "
            "produce a concise narrative summary that captures:\n"
            "1. The user's intent and goals\n"
            "2. Key decisions made\n"
            "3. Open questions or unresolved topics\n"
            "4. Any constraints or preferences the user stated\n\n"
            "Be factual and concise. Do not add interpretation beyond what was discussed."
        )

        def __init__(self, llm, n_ctx=4096):
            self._llm = llm
            self._n_ctx = n_ctx

        def _log(self, msg):
            print(f"[Summarize] {msg}", file=sys.stderr, flush=True)

        def _strip_think_blocks(self, text):
            if not text:
                return text
            cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            unclosed = re.search(r'<think>(.*)', cleaned, re.DOTALL)
            if unclosed:
                cleaned = cleaned[:unclosed.start()].strip()
            return cleaned

        def _llm_call(self, messages, max_tokens=512, temperature=0.2):
            try:
                r = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=int(max_tokens),
                    temperature=float(max(0.01, temperature)),
                    top_p=0.95,
                    top_k=40,
                    repeat_penalty=1.0,
                )
                raw = r['choices'][0]['message']['content']
                usage = r.get('usage', {})
                return self._strip_think_blocks(raw), usage
            except Exception as e:
                self._log(f"LLM call failed: {e}")
                return f"[error: {e}]", {}

        def _format_history(self, session_history):
            parts = []
            for msg in session_history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if role == 'system':
                    continue
                parts.append(f"{role.capitalize()}: {content}")
            return '\n'.join(parts)

        def _truncate_history(self, history_text, budget_ratio=0.5):
            budget_chars = int(self._n_ctx * self._CHARS_PER_TOKEN * budget_ratio)
            if len(history_text) <= budget_chars:
                return history_text
            return history_text[-budget_chars:]

        def summarize(self, session_history_json, max_tokens=512):
            """Produce a narrative summary of the conversation.

            Args:
                session_history_json: JSON string of [{"role": "...", "content": "..."}]
                max_tokens: Max tokens for the summary

            Returns:
                JSON string with narrative_summary and metadata.
            """
            t0 = time.perf_counter()
            session_history = json.loads(session_history_json) if isinstance(session_history_json, str) else session_history_json

            history_text = self._format_history(session_history)
            history_text = self._truncate_history(history_text)

            self._log(f"summarize: {len(session_history)} messages, {len(history_text)} chars")

            messages = [
                {'role': 'system', 'content': self.NARRATIVE_SYSTEM},
                {'role': 'user', 'content': f"Conversation:\n{history_text}"},
            ]
            narrative, usage = self._llm_call(messages, max_tokens=int(max_tokens), temperature=0.2)

            t1 = time.perf_counter()
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)

            self._log(f"summarize done: {prompt_tokens}p+{completion_tokens}c in {(t1-t0)*1000:.0f}ms, narrative={len(narrative)}c")

            return json.dumps({
                'narrative_summary': narrative,
                'metadata': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'duration_ms': round((t1 - t0) * 1000, 2),
                    'message_count': len(session_history),
                    'history_chars': len(history_text),
                }
            })

        TITLE_SYSTEM = (
            "Output only a short title (3-8 words) that describes this conversation. "
            "No punctuation at end. Title only."
        )

        def suggest_title(self, session_history_json, max_tokens=24):
            """Produce a short semantic title for the conversation.

            Args:
                session_history_json: JSON string of [{"role": "...", "content": "..."}]
                max_tokens: Max tokens for the title (default 24)

            Returns:
                JSON string with suggested_title and metadata.
            """
            t0 = time.perf_counter()
            session_history = json.loads(session_history_json) if isinstance(session_history_json, str) else session_history_json

            history_text = self._format_history(session_history)
            history_text = self._truncate_history(history_text, budget_ratio=0.3)

            self._log(f"suggest_title: {len(session_history)} messages")

            messages = [
                {'role': 'system', 'content': self.TITLE_SYSTEM},
                {'role': 'user', 'content': f"Conversation:\n{history_text}"},
            ]
            title, usage = self._llm_call(messages, max_tokens=int(max_tokens), temperature=0.1)

            t1 = time.perf_counter()
            title = title.strip().strip('."\'')
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)

            self._log(f"suggest_title done: '{title}' in {(t1-t0)*1000:.0f}ms")

            return json.dumps({
                'suggested_title': title,
                'metadata': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'duration_ms': round((t1 - t0) * 1000, 2),
                }
            })

    """#

    /// Install on a worker that already has a LlamaSessionKernel (`_kernel`).
    /// Shares the existing `_kernel._llm` instance — no extra model loading.
    static func installShared(
        on worker: PythonProcessPool.WorkerContext,
        llamaKernelHandle: PyHandle,
        config: InferenceConfig
    ) async throws -> PyHandle {
        let installCode = kernelSource + "\n_summarizer = SummarizationKernel(_kernel._llm, n_ctx=\(config.contextSize))\n_summarizer"
        return try await worker.eval(installCode)
    }

    /// Install on a dedicated worker with its own model.
    /// Use when `summarizerModelPath` is set in config.
    static func installDedicated(
        on worker: PythonProcessPool.WorkerContext,
        config: InferenceConfig,
        workerIndex: Int
    ) async throws -> PyHandle {
        guard let path = config.summarizerModelPath, !path.isEmpty else {
            throw InferenceError.modelLoadFailed("summarizerModelPath required for installDedicated")
        }
        let pathLiteral = Self.jsonEncodedPathLiteral(path)
        let dedicatedSource = """
        import os
        _sum_llm = None
        try:
            with SuppressStderr():
                from llama_cpp import Llama
                _sum_llm = Llama(
                    model_path=\(pathLiteral),
                    n_ctx=\(config.contextSize),
                    n_gpu_layers=\(config.nGpuLayers),
                    seed=\(42 + workerIndex),
                    verbose=False,
                )
        except Exception as _e:
            print(f"[Summarize] Dedicated model load failed: {_e}", file=__import__('sys').stderr)
            raise
        _summarizer = SummarizationKernel(_sum_llm, n_ctx=\(config.contextSize))
        _summarizer
        """
        let installCode = kernelSource + "\n" + dedicatedSource
        return try await worker.eval(installCode)
    }

    static func summarize(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionHistory: [[String: String]],
        maxTokens: Int = 512,
        timeout: TimeInterval = 120
    ) async throws -> [String: Any] {
        let historyJSON = try encodeJSON(sessionHistory)

        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "summarize",
            args: [.python(historyJSON)],
            kwargs: ["max_tokens": .python(maxTokens)],
            worker: workerIndex,
            timeout: timeout
        )
        return try LlamaSessionKernel.parseJSON(json)
    }

    /// Suggest a short semantic title (3–8 words) for a conversation.
    static func suggestTitle(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionHistory: [[String: String]],
        maxTokens: Int = 24,
        timeout: TimeInterval = 60
    ) async throws -> String {
        let historyJSON = try encodeJSON(sessionHistory)

        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "suggest_title",
            args: [.python(historyJSON)],
            kwargs: ["max_tokens": .python(maxTokens)],
            worker: workerIndex,
            timeout: timeout
        )
        let result = try LlamaSessionKernel.parseJSON(json)
        return (result["suggested_title"] as? String) ?? ""
    }

    // MARK: - Helpers

    private static func jsonEncodedPathLiteral(_ path: String) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .withoutEscapingSlashes
        guard let data = try? encoder.encode(path),
              let jsonStr = String(data: data, encoding: .utf8) else {
            return "None"
        }
        return jsonStr
    }

    private static func encodeJSON<T: Encodable>(_ value: T) throws -> String {
        let data = try JSONEncoder().encode(value)
        guard let str = String(data: data, encoding: .utf8) else {
            throw InferenceError.decodeFailed(sessionID: SessionID(), reason: "Failed to encode JSON")
        }
        return str
    }
}
