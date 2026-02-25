import Foundation
import SwiftPythonRuntime

/// Python kernel for document chunking and MiniLM embedding.
///
/// Chunks text into overlapping segments, generates 384-dim embeddings
/// via `all-MiniLM-L6-v2`, and provides cosine similarity search.
/// Installed on worker 0 alongside DocumentExtractor.
public enum DocumentIngestionKernel {
    public static let kernelSource = #"""
    import json, sys, time, math

    class DocumentIngestionKernel:
        """Chunk documents and generate MiniLM embeddings."""

        CHUNK_SIZE = 500
        CHUNK_OVERLAP = 100
        EMBEDDING_DIM = 384

        def __init__(self):
            self._model = None
            self._log("DocumentIngestionKernel init (model loaded lazily)")

        def _log(self, msg):
            print(f"[DocIngest] {msg}", file=sys.stderr, flush=True)

        def _ensure_model(self):
            if self._model is None:
                t0 = time.perf_counter()
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                t1 = time.perf_counter()
                self._log(f"MiniLM loaded in {(t1-t0)*1000:.0f}ms")

        def chunk_text(self, text, chunk_size=None, overlap=None):
            """Split text into overlapping chunks.

            Returns JSON: {"chunks": [{"index": 0, "text": "...", "start": 0, "end": 500}, ...]}
            """
            cs = chunk_size or self.CHUNK_SIZE
            ov = overlap or self.CHUNK_OVERLAP
            chunks = []
            start = 0
            idx = 0
            while start < len(text):
                end = min(start + cs, len(text))
                chunk_text = text[start:end]
                if chunk_text.strip():
                    chunks.append({
                        "index": idx,
                        "text": chunk_text,
                        "start": start,
                        "end": end,
                    })
                    idx += 1
                start += cs - ov
            return json.dumps({"chunks": chunks, "count": len(chunks)})

        def chunk_by_headings(self, text, fallback_chunk_size=None):
            """Split text by Markdown headings. Falls back to fixed-size if no headings.

            Returns JSON: {"chunks": [{"index": 0, "text": "...", "heading": "# Intro",
                           "start": 0, "end": 500}, ...], "method": "heading"|"fixed"}
            """
            import re
            cs = fallback_chunk_size or self.CHUNK_SIZE

            # Split on Markdown headings (# through ####)
            heading_pattern = re.compile(r'^(#{1,4}\s+.+)$', re.MULTILINE)
            matches = list(heading_pattern.finditer(text))

            if len(matches) < 2:
                # No meaningful heading structure — fall back to fixed-size
                chunk_result = json.loads(self.chunk_text(text, cs, self.CHUNK_OVERLAP))
                for c in chunk_result["chunks"]:
                    c["heading"] = None
                chunk_result["method"] = "fixed"
                return json.dumps(chunk_result)

            chunks = []
            idx = 0
            for i, match in enumerate(matches):
                heading = match.group(1).strip()
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "index": idx,
                        "text": chunk_text,
                        "heading": heading,
                        "start": start,
                        "end": end,
                    })
                    idx += 1

            # Handle text before first heading
            if matches and matches[0].start() > 0:
                preamble = text[:matches[0].start()].strip()
                if preamble:
                    chunks.insert(0, {
                        "index": 0,
                        "text": preamble,
                        "heading": None,
                        "start": 0,
                        "end": matches[0].start(),
                    })
                    for i, c in enumerate(chunks):
                        c["index"] = i

            return json.dumps({"chunks": chunks, "count": len(chunks), "method": "heading"})

        def chunk_with_priority(self, text, query, top_k=10):
            """Chunk by headings and rank by relevance to query.

            Priority: (1) first chunk/page, (2) heading match to query, (3) embedding similarity.

            Returns JSON: {"chunks": [{"index": N, "text": "...", "score": 0.85,
                           "heading": "...", "priority": "first"|"heading_match"|"semantic"}, ...]}
            """
            self._ensure_model()
            import numpy as np

            chunk_result = json.loads(self.chunk_by_headings(text))
            chunks = chunk_result["chunks"]
            if not chunks:
                return json.dumps({"chunks": [], "count": 0})

            # Embed all chunks
            chunk_texts = [c["text"] for c in chunks]
            embeddings = self._model.encode(chunk_texts, normalize_embeddings=True)
            query_emb = self._model.encode([query], normalize_embeddings=True)[0]
            sim_scores = embeddings @ query_emb

            scored = []
            query_lower = query.lower()
            for i, c in enumerate(chunks):
                priority = "semantic"
                score = float(sim_scores[i])

                # Boost first chunk (preamble / first page)
                if i == 0:
                    priority = "first"
                    score += 0.3

                # Boost heading match
                heading = c.get("heading") or ""
                if heading and query_lower:
                    heading_words = set(heading.lower().split())
                    query_words = set(query_lower.split())
                    overlap = heading_words & query_words
                    if overlap:
                        priority = "heading_match"
                        score += 0.2 * len(overlap)

                scored.append({
                    "index": c["index"],
                    "text": c["text"],
                    "heading": c.get("heading"),
                    "score": round(score, 4),
                    "priority": priority,
                    "start": c["start"],
                    "end": c["end"],
                })

            scored.sort(key=lambda x: x["score"], reverse=True)
            top = scored[:top_k]
            self._log(f"Priority chunking: {len(chunks)} chunks, top score={top[0]['score']:.3f}, method={chunk_result['method']}")
            return json.dumps({"chunks": top, "count": len(top), "method": chunk_result["method"]})

        def embed_texts(self, texts_json):
            """Generate embeddings for a list of texts.

            Args:
                texts_json: JSON string of ["text1", "text2", ...]

            Returns:
                JSON: {"embeddings": [[f1, f2, ...], ...], "dim": 384}
            """
            self._ensure_model()
            texts = json.loads(texts_json) if isinstance(texts_json, str) else texts_json
            t0 = time.perf_counter()
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            t1 = time.perf_counter()
            self._log(f"Embedded {len(texts)} texts in {(t1-t0)*1000:.0f}ms")
            return json.dumps({
                "embeddings": embeddings.tolist(),
                "dim": int(embeddings.shape[1]) if len(embeddings.shape) > 1 else self.EMBEDDING_DIM,
                "count": len(texts),
                "duration_ms": round((t1 - t0) * 1000, 2),
            })

        def chunk_and_embed(self, text, chunk_size=None, overlap=None):
            """Chunk text and embed all chunks in one call.

            Returns JSON: {"chunks": [...], "embeddings": [[...], ...], "dim": 384}
            """
            cs = chunk_size or self.CHUNK_SIZE
            ov = overlap or self.CHUNK_OVERLAP
            chunk_result = json.loads(self.chunk_text(text, cs, ov))
            chunks = chunk_result["chunks"]
            if not chunks:
                return json.dumps({"chunks": [], "embeddings": [], "dim": self.EMBEDDING_DIM})

            texts = [c["text"] for c in chunks]
            embed_result = json.loads(self.embed_texts(json.dumps(texts)))

            return json.dumps({
                "chunks": chunks,
                "embeddings": embed_result["embeddings"],
                "dim": embed_result["dim"],
                "duration_ms": embed_result.get("duration_ms", 0),
            })

        def search(self, query, chunks_json, embeddings_json, top_k=5):
            """Find top-k most similar chunks to a query.

            Args:
                query: Search query string
                chunks_json: JSON array of chunk objects
                embeddings_json: JSON array of embedding arrays
                top_k: Number of results

            Returns:
                JSON: {"results": [{"index": 0, "text": "...", "score": 0.85}, ...]}
            """
            self._ensure_model()
            chunks = json.loads(chunks_json) if isinstance(chunks_json, str) else chunks_json
            embeddings = json.loads(embeddings_json) if isinstance(embeddings_json, str) else embeddings_json

            import numpy as np
            t0 = time.perf_counter()
            query_emb = self._model.encode([query], normalize_embeddings=True)[0]
            emb_matrix = np.array(embeddings)
            scores = emb_matrix @ query_emb
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for i in top_indices:
                i = int(i)
                if i < len(chunks):
                    results.append({
                        "index": chunks[i].get("index", i),
                        "text": chunks[i]["text"],
                        "score": float(scores[i]),
                    })
            t1 = time.perf_counter()
            self._log(f"Search '{query[:50]}' over {len(chunks)} chunks: top score={results[0]['score']:.3f} in {(t1-t0)*1000:.0f}ms")

            return json.dumps({"results": results})

    """#

    /// Install on a worker. Loads MiniLM lazily on first embed call.
    public static func install(
        on worker: PythonProcessPool.WorkerContext
    ) async throws -> PyHandle {
        let installCode = kernelSource + "\n_doc_ingest = DocumentIngestionKernel()\n_doc_ingest"
        return try await worker.eval(installCode)
    }

    /// Chunk text by Markdown headings. Falls back to fixed-size if no headings found.
    public static func chunkByHeadings(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        text: String,
        fallbackChunkSize: Int? = nil,
        timeout: TimeInterval = 30
    ) async throws -> HeadingChunkResult {
        var kwargs: [String: RemotePythonValue] = [:]
        if let fcs = fallbackChunkSize {
            kwargs["fallback_chunk_size"] = .python(fcs)
        }
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "chunk_by_headings",
            args: [.python(text)],
            kwargs: kwargs,
            worker: workerIndex,
            timeout: timeout
        )
        let parsed = try LlamaSessionKernel.parseJSON(json)
        let chunksRaw = (parsed["chunks"] as? [[String: Any]]) ?? []
        let method = (parsed["method"] as? String) ?? "fixed"
        let chunks = chunksRaw.map { c in
            HeadingChunk(
                index: (c["index"] as? Int) ?? 0,
                text: (c["text"] as? String) ?? "",
                heading: c["heading"] as? String,
                startOffset: (c["start"] as? Int) ?? 0,
                endOffset: (c["end"] as? Int) ?? 0
            )
        }
        return HeadingChunkResult(chunks: chunks, method: method)
    }

    /// Chunk by headings and rank by relevance to query with priority scoring.
    public static func chunkWithPriority(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        text: String,
        query: String,
        topK: Int = 10,
        timeout: TimeInterval = 120
    ) async throws -> PriorityChunkResult {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "chunk_with_priority",
            args: [.python(text), .python(query)],
            kwargs: ["top_k": .python(topK)],
            worker: workerIndex,
            timeout: timeout
        )
        let parsed = try LlamaSessionKernel.parseJSON(json)
        let chunksRaw = (parsed["chunks"] as? [[String: Any]]) ?? []
        let method = (parsed["method"] as? String) ?? "fixed"
        let chunks = chunksRaw.map { c in
            PriorityChunk(
                index: (c["index"] as? Int) ?? 0,
                text: (c["text"] as? String) ?? "",
                heading: c["heading"] as? String,
                score: (c["score"] as? Double) ?? 0,
                priority: (c["priority"] as? String) ?? "semantic"
            )
        }
        return PriorityChunkResult(chunks: chunks, method: method)
    }

    /// Chunk text and generate embeddings in one call.
    public static func chunkAndEmbed(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        text: String,
        chunkSize: Int = 500,
        overlap: Int = 100,
        timeout: TimeInterval = 120
    ) async throws -> ChunkAndEmbedResult {
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "chunk_and_embed",
            args: [.python(text)],
            kwargs: [
                "chunk_size": .python(chunkSize),
                "overlap": .python(overlap),
            ],
            worker: workerIndex,
            timeout: timeout
        )
        let parsed = try LlamaSessionKernel.parseJSON(json)
        let chunksRaw = (parsed["chunks"] as? [[String: Any]]) ?? []
        let embeddingsRaw = (parsed["embeddings"] as? [[Double]]) ?? []
        let dim = (parsed["dim"] as? Int) ?? 384

        let chunks = chunksRaw.enumerated().map { i, c in
            DocumentChunk(
                index: (c["index"] as? Int) ?? i,
                text: (c["text"] as? String) ?? "",
                startOffset: (c["start"] as? Int) ?? 0,
                endOffset: (c["end"] as? Int) ?? 0,
                embedding: i < embeddingsRaw.count ? embeddingsRaw[i] : []
            )
        }
        return ChunkAndEmbedResult(chunks: chunks, dim: dim)
    }

    /// Search over pre-computed chunks + embeddings.
    public static func search(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        query: String,
        chunks: [DocumentChunk],
        topK: Int = 5,
        timeout: TimeInterval = 30
    ) async throws -> [SearchResult] {
        let chunkDicts: [[String: Any]] = chunks.map { c in
            ["index": c.index, "text": c.text, "start": c.startOffset, "end": c.endOffset]
        }
        let embeddings: [[Double]] = chunks.map(\.embedding)
        let chunksJSON = try serializeJSON(chunkDicts)
        let embeddingsJSON = try serializeJSON(embeddings)

        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "search",
            args: [
                .python(query),
                .python(chunksJSON),
                .python(embeddingsJSON),
            ],
            kwargs: ["top_k": .python(topK)],
            worker: workerIndex,
            timeout: timeout
        )
        let parsed = try LlamaSessionKernel.parseJSON(json)
        let results = (parsed["results"] as? [[String: Any]]) ?? []
        return results.map { r in
            SearchResult(
                index: (r["index"] as? Int) ?? 0,
                text: (r["text"] as? String) ?? "",
                score: (r["score"] as? Double) ?? 0
            )
        }
    }

    /// Embed a single query text for retrieval.
    public static func embedQuery(
        pool: PythonProcessPool,
        workerIndex: Int,
        kernelHandle: PyHandle,
        query: String,
        timeout: TimeInterval = 30
    ) async throws -> [Double] {
        let textsJSON = try encodeJSON([query])
        let json: String = try await pool.methodResult(
            handle: kernelHandle,
            name: "embed_texts",
            args: [.python(textsJSON)],
            worker: workerIndex,
            timeout: timeout
        )
        let parsed = try LlamaSessionKernel.parseJSON(json)
        let embeddings = (parsed["embeddings"] as? [[Double]]) ?? []
        return embeddings.first ?? []
    }

    private static func encodeJSON<T: Encodable>(_ value: T) throws -> String {
        let data = try JSONEncoder().encode(value)
        guard let str = String(data: data, encoding: .utf8) else {
            throw InferenceError.decodeFailed(sessionID: SessionID(), reason: "Failed to encode JSON")
        }
        return str
    }

    private static func serializeJSON(_ value: Any) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: value)
        guard let str = String(data: data, encoding: .utf8) else {
            throw InferenceError.decodeFailed(sessionID: SessionID(), reason: "Failed to serialize JSON")
        }
        return str
    }
}

// MARK: - Result Types

public struct DocumentChunk: Sendable {
    public let index: Int
    public let text: String
    public let startOffset: Int
    public let endOffset: Int
    public let embedding: [Double]
}

public struct ChunkAndEmbedResult: Sendable {
    public let chunks: [DocumentChunk]
    public let dim: Int
}

public struct SearchResult: Sendable {
    public let index: Int
    public let text: String
    public let score: Double
}

public struct HeadingChunk: Sendable {
    public let index: Int
    public let text: String
    public let heading: String?
    public let startOffset: Int
    public let endOffset: Int
}

public struct HeadingChunkResult: Sendable {
    public let chunks: [HeadingChunk]
    public let method: String
}

public struct PriorityChunk: Sendable {
    public let index: Int
    public let text: String
    public let heading: String?
    public let score: Double
    public let priority: String
}

public struct PriorityChunkResult: Sendable {
    public let chunks: [PriorityChunk]
    public let method: String
}
