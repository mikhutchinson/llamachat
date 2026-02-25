import Foundation
import SwiftPythonRuntime

/// Vision Language Model kernel for image captioning.
///
/// Wraps llama-cpp-python's Llava15ChatHandler to provide image understanding.
/// Loaded lazily — model is only instantiated when first needed. Supports
/// unloading to free VRAM after an idle timeout.
public enum VLMKernel {
    public static let kernelSource = #"""
    import json, sys, os, time, base64

    class VLMKernel:
        """Vision Language Model for image captioning via llava."""

        def __init__(self):
            self._llm = None
            self._model_path = None
            self._clip_path = None
            self._loaded = False
            self._last_used = 0
            self._log("VLMKernel init (not loaded)")

        def _log(self, msg):
            print(f"[VLM] {msg}", file=sys.stderr, flush=True)

        @property
        def is_loaded(self):
            return self._loaded

        @property
        def last_used(self):
            return self._last_used

        def load(self, model_path, clip_path, n_ctx=2048, n_gpu_layers=-1):
            """Load VLM model + CLIP projection.

            Args:
                model_path: Path to the VLM GGUF model
                clip_path: Path to the mmproj CLIP file
                n_ctx: Context size
                n_gpu_layers: GPU layers (-1 = all)
            """
            if self._loaded:
                if self._model_path == model_path and self._clip_path == clip_path:
                    self._log("Already loaded, skipping")
                    return json.dumps({"status": "already_loaded"})
                self.unload()

            t0 = time.perf_counter()
            try:
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import Llava15ChatHandler

                handler = Llava15ChatHandler(clip_model_path=clip_path, verbose=False)

                class SuppressStderr:
                    def __enter__(self):
                        self.fd = os.dup(2)
                        os.close(2)
                        os.open(os.devnull, os.O_WRONLY)
                        return self
                    def __exit__(self, *a):
                        os.close(2)
                        os.dup(self.fd)
                        os.close(self.fd)

                with SuppressStderr():
                    self._llm = Llama(
                        model_path=model_path,
                        chat_handler=handler,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        verbose=False,
                    )
                self._model_path = model_path
                self._clip_path = clip_path
                self._loaded = True
                self._last_used = time.time()
                t1 = time.perf_counter()
                self._log(f"Loaded in {(t1-t0)*1000:.0f}ms: {os.path.basename(model_path)}")
                return json.dumps({"status": "loaded", "duration_ms": round((t1-t0)*1000, 2)})
            except Exception as e:
                t1 = time.perf_counter()
                self._log(f"Load failed: {e}")
                return json.dumps({"status": "error", "error": str(e), "duration_ms": round((t1-t0)*1000, 2)})

        def unload(self):
            """Unload VLM model to free VRAM."""
            if not self._loaded:
                return json.dumps({"status": "not_loaded"})
            del self._llm
            self._llm = None
            self._loaded = False
            self._model_path = None
            self._clip_path = None
            self._log("Unloaded VLM model")
            return json.dumps({"status": "unloaded"})

        _STRUCTURED_PROMPT = (
            "Analyze this image completely. Do the following:"
            "1. List every distinct object, person (with clothing details), or UI element — no repeats. "
            "2. List every distinct color visible. "
            "3. Transcribe ALL text you can read: headings, labels, numbers, captions, buttons, legends — verbatim. "
            "4. Write a thorough multi-sentence description of the scene or content. "
            "Respond ONLY with JSON in this exact format, no commentary before or after: "
            "{\"objects\": [\"adult man in gray sweatpants and Nike sneakers\", \"girl in pink pants\", ...], "
            "\"colors\": [\"gray\", \"pink\", \"cyan\", ...], "
            "\"text\": [\"PIPELINE FULLY OPTIMIZED\", \"97.6% idle\", \"60 fps\", ...], "
            "\"description\": \"Full description of the scene or diagram.\"}"
        )

        _CAPTION_SCHEMA = {
            "type": "object",
            "properties": {
                "objects": {"type": "array", "items": {"type": "string"}},
                "colors": {"type": "array", "items": {"type": "string"}},
                "text": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
            },
            "required": ["objects", "colors", "description"],
            "additionalProperties": False,
        }

        def _run_caption(self, data_uri, prompt, max_tokens):
            """Internal: run captioning given a data URI."""
            use_prompt = prompt if prompt != "Describe this image in detail." else self._STRUCTURED_PROMPT
            messages = [
                {"role": "system", "content": "You are a precise image analysis assistant. Describe everything you see accurately and completely."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": use_prompt},
                ]},
            ]
            r = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=0.1,
            )
            caption = r['choices'][0]['message']['content']
            usage = r.get('usage', {})
            tokens = usage.get('completion_tokens', 0)
            return caption, tokens

        def caption(self, image_data_uri, prompt="Describe this image in detail.", max_tokens=1024):
            """Generate a caption for an image.

            Args:
                image_data_uri: Base64 data URI (data:image/...;base64,...)
                prompt: Caption prompt
                max_tokens: Max tokens for caption

            Returns:
                JSON: {"caption": "...", "tokens": N, "duration_ms": N}
            """
            if not self._loaded:
                return json.dumps({"error": "VLM not loaded", "caption": ""})

            t0 = time.perf_counter()
            try:
                caption, tokens = self._run_caption(image_data_uri, prompt, max_tokens)
                self._last_used = time.time()
                t1 = time.perf_counter()
                self._log(f"Caption: {len(caption)} chars, {tokens} tokens in {(t1-t0)*1000:.0f}ms")
                return json.dumps({
                    "caption": caption,
                    "tokens": tokens,
                    "duration_ms": round((t1-t0)*1000, 2),
                })
            except Exception as e:
                t1 = time.perf_counter()
                self._log(f"Caption failed: {e}")
                return json.dumps({
                    "error": str(e),
                    "caption": "",
                    "duration_ms": round((t1-t0)*1000, 2),
                })

        def caption_file(self, file_path, mime_type="image/png", prompt="Describe this image in detail.", max_tokens=1024):
            """Generate a caption for an image file on disk.

            Reads the file locally to avoid IPC payload limits.

            Args:
                file_path: Path to image file
                mime_type: MIME type of the image
                prompt: Caption prompt
                max_tokens: Max tokens for caption

            Returns:
                JSON: {"caption": "...", "tokens": N, "duration_ms": N}
            """
            if not self._loaded:
                return json.dumps({"error": "VLM not loaded", "caption": ""})

            t0 = time.perf_counter()
            try:
                with open(file_path, 'rb') as f:
                    raw = f.read()
                data_uri = f"data:{mime_type};base64,{base64.b64encode(raw).decode()}"
                caption, tokens = self._run_caption(data_uri, prompt, max_tokens)
                self._last_used = time.time()
                t1 = time.perf_counter()
                self._log(f"Caption (file): {len(caption)} chars, {tokens} tokens in {(t1-t0)*1000:.0f}ms")
                return json.dumps({
                    "caption": caption,
                    "tokens": tokens,
                    "duration_ms": round((t1-t0)*1000, 2),
                })
            except Exception as e:
                t1 = time.perf_counter()
                self._log(f"Caption file failed: {e}")
                return json.dumps({
                    "error": str(e),
                    "caption": "",
                    "duration_ms": round((t1-t0)*1000, 2),
                })

        def status(self):
            """Return current VLM status."""
            return json.dumps({
                "loaded": self._loaded,
                "model": os.path.basename(self._model_path) if self._model_path else None,
                "last_used": self._last_used,
                "idle_secs": time.time() - self._last_used if self._last_used > 0 else -1,
            })

    """#

    /// Install VLM kernel on a worker. Does NOT load the model — that happens lazily.
    public static func install(
        on worker: PythonProcessPool.WorkerContext
    ) async throws -> PyHandle {
        let installCode = kernelSource + "\n_vlm = VLMKernel()\n_vlm"
        return try await worker.eval(installCode)
    }

    /// Load the VLM model + CLIP projection.
    public static func load(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        modelPath: String,
        clipPath: String,
        contextSize: Int = 2048,
        nGpuLayers: Int = -1,
        timeout: TimeInterval = 120
    ) async throws -> [String: Any] {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "load",
            args: [.python(modelPath), .python(clipPath)],
            kwargs: [
                "n_ctx": .python(contextSize),
                "n_gpu_layers": .python(nGpuLayers),
            ],
            worker: workerIndex,
            timeout: timeout
        )
        return try LlamaSessionKernel.parseJSON(json)
    }

    /// Unload the VLM model to free VRAM.
    public static func unload(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        timeout: TimeInterval = 30
    ) async throws -> [String: Any] {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "unload",
            worker: workerIndex,
            timeout: timeout
        )
        return try LlamaSessionKernel.parseJSON(json)
    }

    /// Caption an image from a file on disk (avoids IPC payload limits).
    public static func captionFile(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        filePath: String,
        mimeType: String = "image/png",
        prompt: String = "Describe this image in detail.",
        maxTokens: Int = 1024,
        timeout: TimeInterval = 120
    ) async throws -> VLMCaptionResult {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "caption_file",
            args: [.python(filePath)],
            kwargs: [
                "mime_type": .python(mimeType),
                "prompt": .python(prompt),
                "max_tokens": .python(maxTokens),
            ],
            worker: workerIndex,
            timeout: timeout
        )
        let parsed = try LlamaSessionKernel.parseJSON(json)
        return VLMCaptionResult(
            caption: (parsed["caption"] as? String) ?? "",
            tokens: (parsed["tokens"] as? Int) ?? 0,
            durationMs: (parsed["duration_ms"] as? Double) ?? 0,
            error: parsed["error"] as? String
        )
    }

    /// Get VLM status.
    public static func status(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        timeout: TimeInterval = 10
    ) async throws -> [String: Any] {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "status",
            worker: workerIndex,
            timeout: timeout
        )
        return try LlamaSessionKernel.parseJSON(json)
    }
}

public struct VLMCaptionResult: Sendable {
    public let caption: String
    public let tokens: Int
    public let durationMs: Double
    public let error: String?

    public var succeeded: Bool { error == nil && !caption.isEmpty }

    /// Parse the VLM caption as structured JSON and return a human-readable string.
    ///
    /// Expects `{"objects": [...], "colors": [...], "description": "..."}`.
    /// Returns `"Objects: X, Y. Colors: A, B. Description: ..."` if parseable,
    /// or the raw `caption` string for legacy free-form output.
    public func formattedCaption() -> String {
        guard !caption.isEmpty else { return caption }
        guard let data = caption.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return caption
        }
        var parts: [String] = []
        if let objects = obj["objects"] as? [String], !objects.isEmpty {
            parts.append("Objects: \(objects.joined(separator: ", ")).")
        }
        if let colors = obj["colors"] as? [String], !colors.isEmpty {
            parts.append("Colors: \(colors.joined(separator: ", ")).")
        }
        if let texts = obj["text"] as? [String], !texts.isEmpty {
            parts.append("Text: \(texts.joined(separator: " | ")).")
        }
        if let description = obj["description"] as? String, !description.isEmpty {
            parts.append("Description: \(description)")
        }
        guard !parts.isEmpty else { return caption }
        return parts.joined(separator: " ")
    }
}
