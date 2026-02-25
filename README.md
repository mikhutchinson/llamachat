# LlamaInferenceDemo

Session-affinity inference pipeline for llama.cpp using SwiftPython's multi-process worker pool.

## Architecture

**Option A: Session Affinity + Scheduler Specialization**

Instead of moving KV cache between processes (expensive, fragile), each session is pinned to a single worker. The Swift-side scheduler distributes sessions across workers using least-loaded assignment.

```
┌─────────────────────────────────────────────┐
│            LlamaSessionManager              │
│  (session routing, queue, load balancing)   │
├──────────┬──────────┬──────────┬────────────┤
│  Worker 0│  Worker 1│  Worker 2│  Worker N  │
│  ┌──────┐│  ┌──────┐│  ┌──────┐│  ┌──────┐  │
│  │Llama ││  │Llama ││  │Llama ││  │Llama │  │
│  │ ctx  ││  │ ctx  ││  │ ctx  ││  │ ctx  │  │
│  ├──────┤│  ├──────┤│  ├──────┤│  ├──────┤  │
│  │Sess A││  │Sess C││  │Sess E││  │Sess G│  │
│  │Sess B││  │Sess D││  │Sess F││  │Sess H│  │
│  └──────┘│  └──────┘│  └──────┘│  └──────┘  │
└──────────┴──────────┴──────────┴────────────┘
```

**Why this wins over true prefill/decode split:**
- KV cache stays in-place (zero copy, zero serialization)
- Works with CPU and GPU backends (Metal/CUDA)
- No cross-process GPU memory headaches
- `llama_copy_state_data` overhead (100ms+) makes split uneconomical

## Prerequisites

- **Python virtual environment** — Llama Chat uses the project's `.venv` (at `SwiftPython/.venv`, discovered via `discoverVenvPath()`). Install Python dependencies **into this venv** from the repo root:

  ```bash
  cd /path/to/SwiftPython
  .venv/bin/pip install -r python-requirements.txt
  ```

  Using system `pip` will not put packages where the app's worker processes can find them. Document extraction (PDF, DOCX, PPTX, XLSX) will fail with "content could not be extracted" if markitdown and its extras are not in the project venv.

- **llama-cpp-python** (included in `python-requirements.txt`)
- A GGUF model file (e.g. from [Hugging Face](https://huggingface.co/models?search=gguf))

## Running

```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo --model /path/to/model.gguf
```

**LlamaChatUI** (macOS chat interface with persistence): Conversations persist across restarts. The previously selected model auto-loads on launch. Switch models from the toolbar dropdown (loads immediately) or via Model Hub (search, download, and activate GGUF models from Hugging Face). Model Hub search supports inline provider filtering with `@author` (for example `nanbeige @bartowski` or `@bartowski nanbeige`) and pull-to-refresh in the results list. Model names are displayed in user-friendly form (e.g. "gemma 3 4b it" not raw filenames); mmproj projection files are excluded from chat model selectors since they can't be used for chat. Settings → Models offers context sizes from 2k to 128k tokens (2,048–131,072). Sidebar shows a green dot when ready, with ↻ reload and ⚙ settings buttons. Search uses FTS5 full-text indexing. For proper focus and Dock integration, build as an app bundle and launch with `open`:

```bash
cd Demo/LlamaInferenceDemo
./scripts/build-app-bundle.sh
open build/Llama\ Chat.app
```

### Scenarios

**Single-shot completions** (default):
```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo \
  --model /path/to/model.gguf
```

**Multi-turn conversation** (session reuse, KV cache accumulation):
```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo \
  --model /path/to/model.gguf --multi-turn
```

**Burst load** (concurrent requests across workers):
```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo \
  --model /path/to/model.gguf --burst --workers 4
```

**Latency benchmark** (sequential, measures prefill/decode split):
```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo \
  --model /path/to/model.gguf --benchmark
```

**DAG burst** (Phase 3 — batch DAG with prefill→decode chains, worker affinity, failure isolation):
```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo \
  --model /path/to/model.gguf --dag-burst --workers 4
```

**DAG multi-turn** (Phase 3 — DAG pipeline with token budget tracking):
```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo \
  --model /path/to/model.gguf --dag-multi-turn
```

**Script review** (generate Python script on W0, review on W1 — cross-worker, requires 2 workers):
```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo \
  --model /path/to/model.gguf --script-review --workers 2
```

**Script review parallel** (two scripts, cross-reviewed in parallel):
```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo \
  --model /path/to/model.gguf --script-review-parallel --workers 2
```

**Script review adversarial** (contracts + intersection + Judge, pre-execution spec linter, parallelism proof):
```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo \
  --model /path/to/model.gguf --script-review-adversarial --workers 2
```
Output includes `[Parallelism] Phase 2 wall-clock` and `[Parallelism] Worker PIDs` to verify concurrent execution across distinct processes.

**Script review fail** (kill one worker mid-review, verify the other completes):
```bash
swift run --package-path Demo/LlamaInferenceDemo LlamaInferenceDemo \
  --model /path/to/model.gguf --script-review-fail --workers 2
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to GGUF model file |
| `--workers` | 2 | Number of worker processes |
| `--ctx` | 4096 | Context size in tokens (Settings UI: 2k–128k) |
| `--gpu-layers` | -1 (all) | GPU layers to offload |
| `--summarizer-model` | (none) | Dedicated model for narrative summarization. When set, a separate worker loads this model for context-wind summarization instead of sharing the chat model on worker 0. |

## Dedicated Summarizer Model

When `--summarizer-model` is set (or "Summarizer Model" in LlamaChatUI Settings), a dedicated worker loads the specified model for narrative summarization. This unmarries summarization from the chat model, reducing worker contention when context utilization crosses the commit threshold (70%).

Without a summarizer model, the summarizer shares the chat model on worker 0 (original behavior).

## Components

| File | Role |
|------|------|
| `Sources/Models/SessionTypes.swift` | Session state, sampling params, config, error types |
| `Sources/Python/LlamaSessionKernel.swift` | Python kernel installed per-worker (llama.cpp session manager) |
| `Sources/Python/SummarizationKernel.swift` | Narrative summarization for context-wind rehydration |
| `Sources/Engine/InferenceWorkerPool.swift` | Pool lifecycle, kernel installation, venv injection |
| `Sources/Engine/MemoryGraphWorker.swift` | Narrative memory worker — coordinates summarization when context crosses 70% |
| `Storage/ChatPersistence.swift` | SQLite3 + FTS5 persistence; incremental save, paginated load (`loadMessagesPage`) |
| `Storage/Models.swift` | `Conversation`, `ChatMessage`, `MessageRole` for chat UI |
| `Sources/Engine/LlamaSessionManager.swift` | Session routing, prefill→decode pipeline, metrics |
| `Sources/Engine/InferenceScheduler.swift` | DAG scheduler — batch execution, token budgets, context-wind management, LRU eviction |
| `ChatUIComponents/ChatInputTextView.swift` | NSTextView-backed multi-line composer; Return=send, Shift+Return=newline; fixes SwiftUI scroll bug on macOS |
| `UI/MessageContentView.swift` | Textual-based markdown + LaTeX rendering (`.math` syntax extension) with NSCache-backed, malformed-output-tolerant preprocessing |
| `UI/GGUFModelInfo.swift` | Parses GGUF filenames into user-friendly display names (strips .gguf, quantization); shared by Settings and capsule |
| `UI/ModelDiscovery.swift` | Scans ~/Models/gguf, HF cache, user-configured download dir; `DiscoveredModel.isMMProj` excludes projection files from chat selectors |
| `App/LlamaInferenceDemoApp.swift` | CLI entry point with demo scenarios (single-shot, multi-turn, burst, DAG, script-review, etc.) |
| `UI/AppDelegate.swift` | Graceful shutdown — `applicationShouldTerminate` + terminateLater; flushes pending save and shuts down worker pool before quit so SwiftPythonWorker processes exit cleanly |

## Context Wind Management

LlamaChatUI uses automatic context management for long conversations. When utilization crosses 70%, the system:

1. Summarizes the conversation into a narrative
2. Evicts the current session (flushes KV cache)
3. Creates a fresh session with system prompt + narrative + recent turns
4. Continues inference transparently (session ID updates in the UI)

Budget-aware rehydration keeps the new context under ~35% to leave headroom for the new prompt and response.

## Tests

All tests live in `Tests/` under the `LlamaInferenceDemoTests` target:

```bash
swift test --package-path Demo/LlamaInferenceDemo
swift test --package-path Demo/LlamaInferenceDemo --filter ChatInputTextViewTests
swift test --package-path Demo/LlamaInferenceDemo --filter ChatPersistenceTests
swift test --package-path Demo/LlamaInferenceDemo --filter FileLoggerTests
```

Covers session types, config, state transitions, context wind monitoring, narrative memory, multi-turn conversation, chat persistence (15 tests), ChatInputTextView (Return/Shift+Return, placeholder), FileLogger, and full pipeline.

## SwiftPython Capabilities Demonstrated

- **`PythonProcessPool`** — multi-process worker management with true GIL parallelism
- **`ProcessPoolDAG`** — declarative prefill→decode dependency chains with worker affinity and failure isolation
- **Hot handle reuse** — model loaded once per worker, reused across all sessions
- **`BackpressurePolicy.suspend`** — prevents unbounded session creation
- **`WorkerResourceLimits`** — per-worker memory ceiling
- **Worker affinity** — sessions pinned to specific workers via least-loaded assignment
- **Structured `methodResult`** — typed Swift↔Python method calls without eval strings
- **Batch DAG execution** — multiple sessions scheduled as a single DAG with `continueIndependent` failure policy
- **`pool.warmup()`** — pre-import llama_cpp in all workers
- **venv site-packages injection** — automatic VIRTUAL_ENV detection for worker processes
