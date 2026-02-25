import Foundation
import SwiftPythonRuntime

enum LlamaSessionKernel {
    static let kernelSource = #"""
    import json, time, os, re, sys

    # Suppress noisy llama.cpp Metal initialization warnings
    class SuppressStderr:
        def __enter__(self):
            self.original_stderr = os.dup(2)
            os.close(2)
            os.open(os.devnull, os.O_WRONLY)
            return self
        def __exit__(self, *args):
            os.close(2)
            os.dup(self.original_stderr)
            os.close(self.original_stderr)

    class LlamaSessionKernel:
        """Per-worker session-affinity llama.cpp context manager.

        Each worker process hosts one model and manages multiple sessions.
        KV cache stays in-place (never serialized across processes).
        """

        def __init__(self, model_path, n_ctx=4096, n_gpu_layers=-1, seed=42, verbose=False):
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                seed=seed,
                verbose=verbose,
            )
            self._n_ctx = n_ctx
            self._sessions = {}
            self._model_path = model_path
            self._chars_per_token = 3.5

        @property
        def n_ctx(self):
            return self._n_ctx

        @property
        def session_count(self):
            return len(self._sessions)

        @property
        def session_ids(self):
            return list(self._sessions.keys())

        def _log(self, msg):
            print(f"[LlamaSession] {msg}", file=sys.stderr, flush=True)

        def _extract_think_blocks(self, text):
            """Extract <think> content and return (thinking, cleaned_text)."""
            if not text:
                return '', text
            parts = []
            for m in re.finditer(r'<think>(.*?)</think>', text, re.DOTALL):
                parts.append(m.group(1).strip())
            cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            unclosed = re.search(r'<think>(.*)', cleaned, re.DOTALL)
            if unclosed:
                trailing = unclosed.group(1).strip()
                if trailing:
                    parts.append(trailing)
                cleaned = cleaned[:unclosed.start()].strip()
            # Handle models that omit the opening <think> tag (e.g. Qwen3):
            # an orphan </think> means everything before it is thinking content.
            if not parts and '</think>' in cleaned:
                idx = cleaned.index('</think>')
                before = cleaned[:idx].strip()
                after = cleaned[idx + len('</think>'):].strip()
                if before:
                    parts.append(before)
                cleaned = after
            return '\n\n'.join(parts), cleaned

        def _parse_context_to_messages(self, context):
            """Parse 'User: .../Assistant: ...' context string into chat message array."""
            if not context or not context.strip():
                return []
            messages = []
            current_role = None
            current_lines = []
            for line in context.split('\n'):
                if line.startswith('User: '):
                    if current_role and current_lines:
                        messages.append({'role': current_role, 'content': '\n'.join(current_lines)})
                    current_role = 'user'
                    current_lines = [line[6:]]
                elif line.startswith('Assistant: '):
                    if current_role and current_lines:
                        messages.append({'role': current_role, 'content': '\n'.join(current_lines)})
                    current_role = 'assistant'
                    current_lines = [line[11:]]
                else:
                    current_lines.append(line)
            if current_role and current_lines:
                messages.append({'role': current_role, 'content': '\n'.join(current_lines)})
            return messages

        def _truncate_messages_for_budget(self, messages, max_tokens, budget_ratio=0.75):
            """Drop oldest user/assistant turns to fit within context window budget."""
            budget_tokens = int(self._n_ctx * budget_ratio)
            available_tokens = budget_tokens - int(max_tokens)
            if available_tokens <= 0:
                return messages
            available_chars = int(available_tokens * self._chars_per_token)
            total_chars = sum(len(m.get('content', '')) for m in messages)
            if total_chars <= available_chars:
                return messages
            pruned = list(messages)
            # Preserve system prompt if present at index 0
            start_index = 1 if pruned and pruned[0].get('role') == 'system' else 0
            while total_chars > available_chars and len(pruned) - start_index > 2:
                removed = pruned.pop(start_index)
                total_chars -= len(removed.get('content', ''))
            self._log(f"truncated messages: total_chars={total_chars} budget_chars={available_chars} count={len(pruned)}")
            return pruned

        def count_tokens(self, text):
            """Count the number of tokens in a text string using the model's tokenizer.

            Args:
                text: The text to tokenize.

            Returns:
                JSON string: {"token_count": N}
            """
            try:
                tokens = self._llm.tokenize(text.encode('utf-8') if isinstance(text, str) else text)
                return json.dumps({"token_count": len(tokens)})
            except Exception as e:
                self._log(f"count_tokens failed: {e}")
                return json.dumps({"token_count": int(len(text) / self._chars_per_token), "fallback": True})

        def create_session(self, session_id, system_prompt=None):
            """Register a new session with optional system prompt."""
            if session_id in self._sessions:
                return json.dumps({'status': 'exists', 'session_id': session_id})
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            self._sessions[session_id] = {
                'messages': messages,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'created_at': time.time(),
                'last_activity': time.time(),
            }
            return json.dumps({'status': 'created', 'session_id': session_id})

        def prefill(self, session_id, prompt, params_json='{}'):
            """Add prompt to session message history.

            Actual tokenization happens in decode() via create_chat_completion.
            Returns JSON string with timing for the message append operation.
            """
            params = json.loads(params_json) if isinstance(params_json, str) else params_json
            sess = self._sessions.get(session_id)
            if sess is None:
                return json.dumps({'error': f'session {session_id} not found'})

            t0 = time.perf_counter()
            sess['messages'].append({'role': 'user', 'content': prompt})
            sess['last_activity'] = time.time()
            t1 = time.perf_counter()

            return json.dumps({
                'session_id': session_id,
                'prompt_tokens': 0,
                'prefill_ms': round((t1 - t0) * 1000, 2),
            })

        def decode(self, session_id, max_tokens=256, temperature=0.7, top_p=0.95, top_k=40, repeat_penalty=1.1):
            """Run decode (generation) for a session using accumulated context.

            Uses create_chat_completion for the full message history.
            Returns generated text, token counts, finish reason, and timing.
            """
            sess = self._sessions.get(session_id)
            if sess is None:
                return json.dumps({'error': f'session {session_id} not found'})

            try:
                t0 = time.perf_counter()
                messages = self._truncate_messages_for_budget(sess['messages'], int(max_tokens))
                sess['messages'] = messages
                self._log(f"decode session={session_id} messages={len(messages)}")
                result = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=int(max_tokens),
                    temperature=float(max(temperature, 0.01)),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repeat_penalty=float(repeat_penalty),
                )
                t1 = time.perf_counter()
            except Exception as e:
                import traceback
                return json.dumps({
                    'error': f'decode failed: {type(e).__name__}: {str(e)}',
                    'traceback': traceback.format_exc()
                })

            choice = result['choices'][0]
            raw_text = choice['message']['content']
            thinking, text = self._extract_think_blocks(raw_text)
            finish_reason = choice.get('finish_reason', 'unknown')
            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)

            sess['messages'].append({'role': 'assistant', 'content': text})
            sess['completion_tokens'] += completion_tokens
            sess['prompt_tokens'] = prompt_tokens
            sess['last_activity'] = time.time()

            return json.dumps({
                'session_id': session_id,
                'text': text,
                'thinking': thinking,
                'finish_reason': finish_reason,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'decode_ms': round((t1 - t0) * 1000, 2),
            })

        def decode_stream(self, session_id, max_tokens=256, temperature=0.7, top_p=0.95, top_k=40, repeat_penalty=1.1, stop=None):
            """Run decode as a streaming generator.

            Yields JSON strings with one of three event shapes:
            - {"event":"delta","delta":"..."}
            - {"event":"done","finish_reason":"...","prompt_tokens":N,"completion_tokens":N,"prefill_ms":N,"decode_ms":N,"text":"...","thinking":"..."}
            - {"event":"error","error":"...","traceback":"..."}
            """
            sess = self._sessions.get(session_id)
            if sess is None:
                yield json.dumps({
                    "event": "error",
                    "error": f"session {session_id} not found",
                })
                return

            t0 = time.perf_counter()
            messages = self._truncate_messages_for_budget(sess['messages'], int(max_tokens))
            sess['messages'] = messages
            self._log(f"decode_stream session={session_id} messages={len(messages)}")

            chunks = []
            usage = {}
            finish_reason = "unknown"
            first_token_time = None

            try:
                stream = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=int(max_tokens),
                    temperature=float(max(temperature, 0.01)),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repeat_penalty=float(repeat_penalty),
                    stop=stop,
                    stream=True,
                )

                for event in stream:
                    choices = event.get('choices', [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get('delta', {}).get('content') or ''
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        chunks.append(delta)
                        yield json.dumps({
                            "event": "delta",
                            "delta": delta,
                        })
                    if choice.get('finish_reason') is not None:
                        finish_reason = choice.get('finish_reason') or finish_reason

                    maybe_usage = event.get('usage')
                    if isinstance(maybe_usage, dict) and maybe_usage:
                        usage = maybe_usage
            except Exception as e:
                import traceback
                yield json.dumps({
                    "event": "error",
                    "error": f"decode_stream failed: {type(e).__name__}: {str(e)}",
                    "traceback": traceback.format_exc(),
                })
                return

            t1 = time.perf_counter()
            raw_text = ''.join(chunks)
            thinking, text = self._extract_think_blocks(raw_text)
            prompt_tokens = int(usage.get('prompt_tokens', 0))
            completion_tokens = int(usage.get('completion_tokens', 0))

            # Some llama-cpp-python stream handlers omit usage in streamed chunks.
            # Fall back to tokenizer/heuristics so UI/accounting metrics remain useful.
            if completion_tokens <= 0 and raw_text:
                try:
                    completion_tokens = len(self._llm.tokenize(raw_text.encode('utf-8')))
                except Exception:
                    completion_tokens = max(0, int(len(raw_text) / self._chars_per_token))
            if prompt_tokens <= 0:
                prompt_chars = sum(len((m.get('content') or '')) for m in messages)
                prompt_tokens = max(0, int(prompt_chars / self._chars_per_token))

            sess['messages'].append({'role': 'assistant', 'content': text})
            sess['completion_tokens'] += completion_tokens
            sess['prompt_tokens'] = prompt_tokens
            sess['last_activity'] = time.time()

            # Calculate timing: prefill = time to first token, decode = generation time
            if first_token_time is not None:
                prefill_ms = round((first_token_time - t0) * 1000, 2)
                decode_ms = round((t1 - first_token_time) * 1000, 2)
            else:
                prefill_ms = 0
                decode_ms = round((t1 - t0) * 1000, 2)

            yield json.dumps({
                "event": "done",
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "text": text,
                "thinking": thinking,
            })

        def complete(self, session_id, prompt, params_json='{}'):
            """One-shot: prefill + decode in a single call.

            Convenience for simple request/response patterns.
            Returns JSON string with combined result.
            """
            prefill_json = self.prefill(session_id, prompt, params_json)
            prefill_result = json.loads(prefill_json)
            if 'error' in prefill_result:
                return prefill_json
            params = json.loads(params_json) if isinstance(params_json, str) else params_json
            decode_json = self.decode(
                session_id,
                max_tokens=params.get('max_tokens', 256),
                temperature=params.get('temperature', 0.7),
                top_p=params.get('top_p', 0.95),
                top_k=params.get('top_k', 40),
                repeat_penalty=params.get('repeat_penalty', 1.1),
            )
            decode_result = json.loads(decode_json)
            if 'error' in decode_result:
                return decode_json
            decode_result['prefill_ms'] = prefill_result.get('prefill_ms', 0)
            return json.dumps(decode_result)

        def decode_to_shm(self, session_id, result_buf, max_tokens=256, temperature=0.7, top_p=0.95, top_k=40, repeat_penalty=1.1):
            """Run decode and write JSON result to shared memory buffer.

            Instead of returning a JSON string via pickle, writes the result
            bytes directly into the provided numpy uint8 shared memory array.
            Returns the byte count written (int) so Swift knows how much to read.
            Layout: [4 bytes little-endian length][N bytes UTF-8 JSON]
            """
            import numpy as np
            import struct

            decode_json = self.decode(
                session_id,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repeat_penalty=float(repeat_penalty),
            )
            result_bytes = decode_json.encode('utf-8')
            n = len(result_bytes)
            header = struct.pack('<I', n)
            result_buf[:4] = np.frombuffer(header, dtype=np.uint8)
            result_buf[4:4+n] = np.frombuffer(result_bytes, dtype=np.uint8)
            return 4 + n

        def complete_to_shm(self, session_id, prompt, result_buf, params_json='{}'):
            """One-shot prefill+decode writing result to shared memory buffer.

            Same layout as decode_to_shm. Returns byte count written.
            """
            import numpy as np
            import struct

            complete_json = self.complete(session_id, prompt, params_json)
            result_bytes = complete_json.encode('utf-8')
            n = len(result_bytes)
            header = struct.pack('<I', n)
            result_buf[:4] = np.frombuffer(header, dtype=np.uint8)
            result_buf[4:4+n] = np.frombuffer(result_bytes, dtype=np.uint8)
            return 4 + n

        def evict(self, session_id):
            """Remove a session and free its resources."""
            if session_id in self._sessions:
                del self._sessions[session_id]
                return json.dumps({'status': 'evicted', 'session_id': session_id})
            return json.dumps({'status': 'not_found', 'session_id': session_id})

        def session_info(self, session_id):
            """Return metadata for a session."""
            sess = self._sessions.get(session_id)
            if sess is None:
                return json.dumps({'error': f'session {session_id} not found'})
            return json.dumps({
                'session_id': session_id,
                'message_count': len(sess['messages']),
                'prompt_tokens': sess['prompt_tokens'],
                'completion_tokens': sess['completion_tokens'],
                'total_tokens': sess['prompt_tokens'] + sess['completion_tokens'],
                'created_at': sess['created_at'],
                'last_activity': sess['last_activity'],
            })

        def worker_stats(self):
            """Return aggregate stats for this worker."""
            total_prompt = sum(s['prompt_tokens'] for s in self._sessions.values())
            total_completion = sum(s['completion_tokens'] for s in self._sessions.values())
            return json.dumps({
                'model_path': self._model_path,
                'n_ctx': self._n_ctx,
                'session_count': len(self._sessions),
                'session_ids': list(self._sessions.keys()),
                'total_prompt_tokens': total_prompt,
                'total_completion_tokens': total_completion,
            })

        def evict_lru(self, max_sessions):
            """Evict least-recently-used sessions until count <= max_sessions."""
            evicted = []
            while len(self._sessions) > max_sessions:
                lru_id = min(
                    self._sessions,
                    key=lambda sid: self._sessions[sid]['last_activity']
                )
                del self._sessions[lru_id]
                evicted.append(lru_id)
            return json.dumps(evicted)

    _kernel = None
    try:
        with SuppressStderr():
            _kernel = LlamaSessionKernel(
                model_path=__MODEL_PATH__,
                n_ctx=__N_CTX__,
                n_gpu_layers=__N_GPU_LAYERS__,
                seed=__SEED__,
                verbose=False,
            )
    except Exception as _e:
        import sys
        print(f"Failed to load model: {_e}", file=sys.stderr)
        raise
    _kernel
    """#

    enum DecodeStreamEvent: String, Sendable {
        case delta
        case done
        case error
    }

    /// One streamed decode event yielded from Python `decode_stream`.
    struct DecodeStreamChunk: Sendable {
        let event: DecodeStreamEvent
        let delta: String
        let finishReason: String?
        let promptTokens: Int?
        let completionTokens: Int?
        let decodeMs: Double?
        let prefillMs: Double?
        let text: String?
        let thinking: String?
        let error: String?
        let traceback: String?

        var isTerminal: Bool {
            event == .done || event == .error
        }

        init(jsonString: String) throws {
            let parsed = try parseJSON(jsonString)
            guard let eventRaw = parsed["event"] as? String,
                  let parsedEvent = DecodeStreamEvent(rawValue: eventRaw) else {
                throw InferenceError.decodeFailed(
                    sessionID: SessionID(),
                    reason: "decode_stream event missing or invalid: \(jsonString.prefix(160))"
                )
            }

            self.event = parsedEvent
            self.delta = (parsed["delta"] as? String) ?? ""
            self.finishReason = parsed["finish_reason"] as? String
            self.promptTokens = intValue(parsed["prompt_tokens"])
            self.completionTokens = intValue(parsed["completion_tokens"])
            self.decodeMs = doubleValue(parsed["decode_ms"])
            self.prefillMs = doubleValue(parsed["prefill_ms"])
            self.text = parsed["text"] as? String
            self.thinking = parsed["thinking"] as? String
            self.error = parsed["error"] as? String
            self.traceback = parsed["traceback"] as? String
        }
    }

    static func install(
        on worker: PythonProcessPool.WorkerContext,
        config: InferenceConfig,
        workerIndex: Int
    ) async throws -> PyHandle {
        let pathLiteral = jsonEncodedPathLiteral(config.modelPath)
        let code = kernelSource
            .replacingOccurrences(of: "__MODEL_PATH__", with: pathLiteral)
            .replacingOccurrences(of: "__N_CTX__", with: String(config.contextSize))
            .replacingOccurrences(of: "__N_GPU_LAYERS__", with: String(config.nGpuLayers))
            .replacingOccurrences(of: "__SEED__", with: String(42 + workerIndex))
        return try await worker.eval(code)
    }

    static func createSession(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionID: String,
        systemPrompt: String? = nil,
        timeout: TimeInterval = 30
    ) async throws -> [String: Any] {
        var kwargs: [String: RemotePythonValue] = [
            "session_id": .python(sessionID),
        ]
        if let systemPrompt {
            kwargs["system_prompt"] = .python(systemPrompt)
        }
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "create_session",
            kwargs: kwargs,
            worker: workerIndex,
            timeout: timeout
        )
        return try parseJSON(json)
    }

    static func complete(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionID: String,
        prompt: String,
        params: SamplingParams,
        timeout: TimeInterval = 300
    ) async throws -> [String: Any] {
        let paramsJSON = encodeParams(params)
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "complete",
            args: [.python(sessionID), .python(prompt), .python(paramsJSON)],
            worker: workerIndex,
            timeout: timeout
        )
        return try parseJSON(json)
    }

    static func prefill(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionID: String,
        prompt: String,
        params: SamplingParams,
        timeout: TimeInterval = 60
    ) async throws -> [String: Any] {
        let paramsJSON = encodeParams(params)
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "prefill",
            args: [.python(sessionID), .python(prompt), .python(paramsJSON)],
            worker: workerIndex,
            timeout: timeout
        )
        return try parseJSON(json)
    }

    static func decode(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionID: String,
        params: SamplingParams,
        timeout: TimeInterval = 120
    ) async throws -> [String: Any] {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "decode",
            args: [.python(sessionID)],
            kwargs: [
                "max_tokens": .python(params.maxTokens),
                "temperature": .python(params.temperature),
                "top_p": .python(params.topP),
                "top_k": .python(params.topK),
                "repeat_penalty": .python(params.repeatPenalty),
            ],
            worker: workerIndex,
            timeout: timeout
        )
        return try parseJSON(json)
    }

    /// Run decode as a streamed sequence of delta/done/error events.
    static func decodeStream(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionID: String,
        params: SamplingParams,
        timeout: TimeInterval = 300
    ) async throws -> CancellableStream<DecodeStreamChunk> {
        var kwargs: [String: RemotePythonValue] = [
            "max_tokens": .python(params.maxTokens),
            "temperature": .python(params.temperature),
            "top_p": .python(params.topP),
            "top_k": .python(params.topK),
            "repeat_penalty": .python(params.repeatPenalty),
        ]
        if !params.stop.isEmpty {
            kwargs["stop"] = .python(params.stop)
        }

        return try await pool.methodStream(
            handle: kernelHandle,
            name: "decode_stream",
            args: [.python(sessionID)],
            kwargs: kwargs,
            worker: workerIndex,
            timeout: timeout,
            decode: { data in
                let chunkJSON: String = try await PythonExecutor.shared.run {
                    let pickle = try Python.import("pickle")
                    let pyBytes = try data.toPythonObject()
                    let result = try pickle.loads(pyBytes)
                    return try String(pythonObject: result)
                }
                return try DecodeStreamChunk(jsonString: chunkJSON)
            }
        )
    }

    static func evict(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionID: String,
        timeout: TimeInterval = 10
    ) async throws -> [String: Any] {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "evict",
            args: [.python(sessionID)],
            worker: workerIndex,
            timeout: timeout
        )
        return try parseJSON(json)
    }

    /// Count tokens in a text string using the model's actual tokenizer.
    ///
    /// Falls back to `chars / 3.5` estimation if the tokenizer call fails.
    static func countTokens(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        text: String,
        timeout: TimeInterval = 10
    ) async throws -> Int {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "count_tokens",
            args: [.python(text)],
            worker: workerIndex,
            timeout: timeout
        )
        let parsed = try parseJSON(json)
        return (parsed["token_count"] as? Int) ?? Int(Double(text.utf8.count) / 3.5)
    }

    static func workerStats(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        timeout: TimeInterval = 10
    ) async throws -> [String: Any] {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "worker_stats",
            worker: workerIndex,
            timeout: timeout
        )
        return try parseJSON(json)
    }

    // MARK: - Shared Memory Methods

    static func decodeToShm(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionID: String,
        resultBuffer: PyHandle,
        params: SamplingParams,
        timeout: TimeInterval = 120
    ) async throws -> [String: Any] {
        let byteCount: Int = try await pool.methodResult(
            handle: kernelHandle,
            name: "decode_to_shm",
            args: [
                .python(sessionID),
                .handle(resultBuffer),
            ],
            kwargs: [
                "max_tokens": .python(params.maxTokens),
                "temperature": .python(params.temperature),
                "top_p": .python(params.topP),
                "top_k": .python(params.topK),
                "repeat_penalty": .python(params.repeatPenalty),
            ],
            worker: workerIndex,
            timeout: timeout
        )
        return try await readResultFromShm(pool: pool, resultBuffer: resultBuffer, byteCount: byteCount)
    }

    static func completeToShm(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        sessionID: String,
        prompt: String,
        resultBuffer: PyHandle,
        params: SamplingParams,
        timeout: TimeInterval = 300
    ) async throws -> [String: Any] {
        let paramsJSON = encodeParams(params)
        let byteCount: Int = try await pool.methodResult(
            handle: kernelHandle,
            name: "complete_to_shm",
            args: [
                .python(sessionID),
                .python(prompt),
                .handle(resultBuffer),
            ],
            kwargs: [
                "params_json": .python(paramsJSON),
            ],
            worker: workerIndex,
            timeout: timeout
        )
        return try await readResultFromShm(pool: pool, resultBuffer: resultBuffer, byteCount: byteCount)
    }

    static func readResultFromShm(
        pool: PythonProcessPool,
        resultBuffer: PyHandle,
        byteCount: Int
    ) async throws -> [String: Any] {
        let jsonString: String = try await pool.withSharedBuffer(resultBuffer, as: UInt8.self) { buffer in
            guard byteCount >= 4, byteCount <= buffer.count else {
                throw InferenceError.decodeFailed(
                    sessionID: SessionID(),
                    reason: "Shared memory result size invalid: \(byteCount) (buffer: \(buffer.count))"
                )
            }
            let jsonLen = Int(buffer[0]) | (Int(buffer[1]) << 8) | (Int(buffer[2]) << 16) | (Int(buffer[3]) << 24)
            guard 4 + jsonLen <= byteCount else {
                throw InferenceError.decodeFailed(
                    sessionID: SessionID(),
                    reason: "JSON length \(jsonLen) exceeds written bytes: \(byteCount)"
                )
            }
            let jsonBytes = Array(buffer[4..<(4 + jsonLen)])
            guard let str = String(bytes: jsonBytes, encoding: .utf8) else {
                throw InferenceError.decodeFailed(
                    sessionID: SessionID(),
                    reason: "Invalid UTF-8 in shared memory result"
                )
            }
            return str
        }
        return try parseJSON(jsonString)
    }

    static func pythonStringLiteral(_ s: String) -> String {
        let escaped = s
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
            .replacingOccurrences(of: "\0", with: "\\0")
        return "\"\(escaped)\""
    }

    // MARK: - Helpers

    static func parseJSON(_ jsonString: String) throws -> [String: Any] {
        guard let data = jsonString.data(using: .utf8),
              let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw InferenceError.decodeFailed(sessionID: SessionID(), reason: "Failed to parse kernel JSON: \(jsonString.prefix(200))")
        }
        return obj
    }

    private static func intValue(_ any: Any?) -> Int? {
        if let int = any as? Int {
            return int
        }
        if let number = any as? NSNumber {
            return number.intValue
        }
        return nil
    }

    private static func doubleValue(_ any: Any?) -> Double? {
        if let double = any as? Double {
            return double
        }
        if let number = any as? NSNumber {
            return number.doubleValue
        }
        return nil
    }

    private static func jsonEncodedPathLiteral(_ path: String) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .withoutEscapingSlashes
        guard let data = try? encoder.encode(path),
              let jsonStr = String(data: data, encoding: .utf8) else {
            return "None"
        }
        return jsonStr
    }

    private static func encodeParams(_ params: SamplingParams) -> String {
        var dict: [String: Any] = [
            "max_tokens": params.maxTokens,
            "temperature": params.temperature,
            "top_p": params.topP,
            "top_k": params.topK,
            "repeat_penalty": params.repeatPenalty,
        ]
        if !params.stop.isEmpty {
            dict["stop"] = params.stop
        }
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let str = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return str
    }
}
