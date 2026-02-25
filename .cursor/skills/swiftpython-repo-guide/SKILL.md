---
name: swiftpython-repo-guide
description: Guide for the SwiftPython repo - a modern Swift-Python bridge with zero-copy NumPy interop, generated type-safe bindings, Swift 6 concurrency, ProcessPool with ResourceMonitor, streaming generators, and DAG orchestration. Use when editing bindings, stubs, generator code, runtime bridge, CPython interop, or ProcessPool features.
---

# SwiftPython Repo Guide

## What is SwiftPython?

A modern Swift-Python bridge that improves on PythonKit with:
- **Type-safe generated bindings** from `.pyi` stubs (vs PythonKit's untyped `PythonObject`)
- **Zero-copy NumPy ↔ Accelerate** buffer sharing with validation
- **Full Python slicing** including negative indices and steps
- **Swift 6 concurrency** with actor-isolated GIL management
- **Ergonomic escape hatch** via `@dynamicMemberLookup` and `@dynamicCallable` on `PyObjectRef`
- **ProcessPool** for multi-worker Python execution with shared memory IPC
- **ResourceMonitor** for thermal/memory-aware task scheduling (Darwin)
- **Streaming generators** via `CancellableStream<T>` for IPC generator iteration
- **DAG orchestration** with `ProcessPoolDAG` for dependency-aware execution

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Generated Modules (NumPy, Pandas, Sklearn, SciPy, PyTorch) │
│  - Type-safe Swift structs from .pyi stubs                  │
│  - @PythonClass, @PythonMethod, @PythonProperty macros      │
├─────────────────────────────────────────────────────────────┤
│  SwiftPythonRuntime                                         │
│  - Python.run / PythonExecutor (GIL management)               │
│  - PyObjectRef (RAII reference counting + dynamic features)   │
│  - PythonBuffer (zero-copy buffer protocol)                 │
│  - Type conversions (Int, Double, String, Array, Dict)      │
│  - PythonProcessPool (multi-worker IPC execution)           │
│  - ResourceMonitor (thermal/memory pressure management)       │
├─────────────────────────────────────────────────────────────┤
│  Python C API (libpython3.13)                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Source Locations

| Component | Path | Purpose |
|-----------|------|---------|
| **Runtime Core** | `Sources/SwiftPythonRuntime/PyObject.swift` | `PyObjectRef`, `Python` struct, dynamic features |
| **GIL Management** | `Sources/SwiftPythonRuntime/GILManager.swift` | `PythonExecutor`, `PythonObjectRef`, `withGIL` |
| **Type Conversion** | `Sources/SwiftPythonRuntime/TypeConversion.swift` | `PythonConvertible` protocol |
| **C API Bindings** | `Sources/SwiftPythonRuntime/PythonCAPI.swift` | Direct `Py*` function wrappers |
| **Buffer Protocol** | `Sources/SwiftPythonRuntime/PythonBuffer.swift` | Zero-copy array access |
| **ProcessPool** | `Sources/SwiftPythonRuntime/ProcessPool/` | Worker processes, IPC, shared memory |
| **ResourceMonitor** | `Sources/SwiftPythonRuntime/ProcessPool/ResourceMonitor.swift` | Thermal/memory pressure sampling |
| **Generator** | `Sources/SwiftPythonGen/` | `StubParser`, `StubResolver`, `SwiftGenerator` |
| **Macros** | `Sources/SwiftPythonMacros/` | `@PythonClass`, `@PythonMethod`, `@PythonProperty` |
| **Stubs** | `stubs/` | Python `.pyi` interface definitions |
| **Generated Bindings** | `Sources/NumPy/`, `Sources/Pandas/`, etc. | DO NOT EDIT - regenerate from stubs |
| **Worker Binary** | `Sources/SwiftPythonWorker/` | `SwiftPythonWorker` executable for ProcessPool |

## Ergonomic API Patterns

### Python.run (Preferred for escape-hatch code)
```swift
let result = try await Python.run {
    let np = try Python.import("numpy")
    let arr = try np.array([1, 2, 3])  // Dynamic member lookup + callable
    return try np.sum(arr)
}
```

### Generated Bindings (Preferred for type safety)
```swift
let arr = try await NumPy.linspace(start: 0.0, stop: 10.0, num: 100)
let mean = try await arr.mean()
```

### Convenience Imports
```swift
// Available: Python.sys, .os, .builtins, .json, .io, .re, .math, .datetime, .pathlib, .collections, .asyncio
let sys = try Python.sys
let version = try sys.version
```

### ProcessPool with ResourceMonitor (Production hardened)
```swift
let pool = try await PythonProcessPool(
    workers: 4,
    resourceLimits: WorkerResourceLimits(maxMemoryBytes: 4 * 1024 * 1024 * 1024),
    backpressure: .suspend(maxInFlight: 16)
)

// Resource-aware execution
let result = try await pool.evalResult("numpy.arange(100)")

// Check thermal/memory pressure
let snapshot = await pool.resourceSnapshot()
print("Thermal: \(snapshot.thermalLevel), Memory: \(snapshot.systemMemoryPressure)")
```

### Streaming Generators (CancellableStream)
```swift
// Finite generator — natural exhaustion
let stream: CancellableStream<Int> = try await pool.evalStream("range(10)")
for try await value in stream { print(value) }

// Infinite generator — break triggers cooperative abort
let stream: CancellableStream<Int> = try await pool.evalStream(
    "import itertools; itertools.count()"
)
for try await value in stream {
    if value >= 5 { break }  // Cleanup fires immediately → SIGUSR1 → worker stops
}
```

### DAG Orchestration
```swift
let dag = ProcessPoolDAG(nodes: [
    .init(id: "load") { ctx in try await ctx.worker.evalResult("load_data()") },
    .init(id: "process", dependencies: ["load"]) { ctx in
        let data = try ctx.result(for: "load")
        return try await ctx.worker.evalResult("process(\(data))")
    }
])
let results = try await pool.run(dag)
```

## Generator Pipeline

The generator transforms `.pyi` stubs into Swift bindings:

```
stubs/numpy/__init__.pyi  →  swift-python-gen  →  Sources/NumPy/NumPy.swift
```

### Key Generator Features
- **Module name casing preserved**: `NumPy` → `NumPy`, not `Numpy`
- **Positional-only parameters**: Detects `/` marker, generates positional args
- **Optional parameters**: `hasDefault` → Swift optional with `nil` default
- **Variadic support**: `*args` and `**kwargs` passthrough
- **Re-export resolution**: Follows `from X import Y` to find actual definitions
- **Remote generation**: `--remote` flag generates ProcessPool wrappers (`rNumPy`)
- **Value types**: `# swiftpython: value-type` directive emits Swift structs

### Generator Commands
```bash
# Single module
swift run swift-python-gen --module NumPy --stub-dir stubs/numpy --output Sources/NumPy

# With custom Python module name
swift run swift-python-gen --module PyTorch --python-module torch --stub-dir stubs/torch --output Sources/PyTorch

# Remote (ProcessPool) wrappers
swift run swift-python-gen --module NumPy --stub-dir stubs/numpy --output Sources/NumPy/Remote --remote

# Regenerate all (CI does this)
# See .github/workflows/ci.yml for the full list
```

## Testing Strategy

- **400+ tests** across all modules (ProcessPool tests: 305+)
- Run all: `swift test`
- Run specific: `swift test --filter NumPyTests`
- Run examples: `swift run examples`

### Test Locations
| Test Suite | Coverage |
|------------|----------|
| `SwiftPythonTests` | Core runtime, type conversion, Python.run, dynamic features |
| `GILManagerTests` | GIL stress tests, concurrent access |
| `PythonCAPITests` | Direct C API bindings |
| `NumPyTests` | NumPy bindings, buffer protocol |
| `PandasTests`, `SklearnTests`, etc. | Generated binding integration |
| `ProcessPoolTests` | Worker processes, IPC, shared memory, streaming, DAG |
| `ResourceMonitorTests` | Thermal/memory pressure management |

## Workflow: Adding/Updating Bindings

1. **Edit stubs** under `stubs/<package>/`
2. **Regenerate**: `swift run swift-python-gen --module X --stub-dir stubs/x --output Sources/X`
3. **Build**: `swift build`
4. **Test**: `swift test`
5. **Update callers** if API changed (examples, tests, demo)

⚠️ **Never hand-edit generated bindings** - change the generator or stubs instead.

## Workflow: Runtime Changes

When modifying `SwiftPythonRuntime`:

1. **Understand GIL implications** - all Python ops need GIL
2. **Check for Sendable boundaries** - `PyObjectRef` is `@unchecked Sendable`
3. **Run full test suite** - intermittent crashes may indicate GIL issues
4. **Update docs** if API changes (README.md, docs/API.md, CHANGELOG.md)

## ProcessPool Production Hardening

### Resource Limits (per worker)
```swift
WorkerResourceLimits(maxMemoryBytes: 4 * 1024 * 1024 * 1024)  // 4 GB
```

### Backpressure Policies
```swift
.unbounded                    // No limit (default)
.reject(maxInFlight: 16)      // Throw .backpressure when full
.suspend(maxInFlight: 16)     // FIFO cooperative throttling
```

### ResourceMonitor Configuration (Darwin)
```swift
ResourceMonitorConfig(
    sampleInterval: 2.0,               // seconds
    memoryPressureThrottle: 0.85,       // force suspend ≥ 85%
    memoryPressureReject: 0.95,          // throw .backpressure ≥ 95%
    thermalThrottleLevel: .fair,       // force suspend ≥ .fair
    workerCPUThrottlePercent: 90.0,    // skip workers > 90% CPU
    enabled: true                       // default true on Darwin
)
```

### Lifecycle Management
```swift
// Warmup: run setup on all workers
await pool.warmup("import numpy as np")

// Drain: wait for in-flight work, block new submissions
try await pool.drain(timeout: .seconds(30))

// Resume: re-enable submissions
pool.resume()

// Shutdown: clean teardown
await pool.shutdown()
```

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| Segfault in tests | GIL not held | Wrap in `Python.run` or `withGIL` |
| `Numpy` instead of `NumPy` | Old generator casing | Regenerate bindings |
| `parameters` kwarg error | Positional-only param | Add `/` to stub, regenerate |
| Intermittent test crashes | Known XCTest+Python issue | Re-run tests, clean build |
| Worker not found | SwiftPythonWorker not built | `swift build --product SwiftPythonWorker` |
| Orphan worker at 100% CPU | Parent died before shutdown | Use `shutdownSync()` in app lifecycle |

## Key Concepts

### PyObjectRef
RAII wrapper with `@dynamicMemberLookup` and `@dynamicCallable`. Auto-decrefs on deinit.

### PyHandle
Sendable handle for cross-actor/cross-process Python object references. Use with `withObject` APIs.

### PythonObjectRef
Thread-safe `Sendable` wrapper around `PyObjectRef`. Used for storing refs across async boundaries.

### PythonConvertible
Protocol for Swift↔Python conversion. Conformers: `Int`, `Double`, `String`, `Bool`, `Array`, `Dict`, `Optional`, all `@PythonClass` types.

### PythonObjectWrapper
Protocol for generated class wrappers (`ndarray`, `DataFrame`, etc.). Provides `pyRef`, subscripts, `getAttribute`.

### CancellableStream
Custom `AsyncSequence` that reliably triggers cleanup when iteration stops. Solves `AsyncThrowingStream.onTermination` not firing on `break`.

### ProcessPoolDAG
Dependency-aware execution graph for ProcessPool. Supports cycle detection, worker affinity, and failure policy (`failFast` or `continueIndependent`).

## Documentation

| Doc | Purpose |
|-----|---------|
| `README.md` | User-facing overview, quick start |
| `docs/API.md` | API reference, type mappings, examples |
| `CHANGELOG.md` | Version history |
| `STUB_GENERATION_GUIDE.md` | How to write stubs |
| `IMPLEMENTATION.md` | Design decisions, internals |
| `docs/wiki/Home.md` | Wiki root with current highlights |
| `docs/wiki/API-Core-Runtime.md` | Core runtime API details |
| `docs/wiki/API-Concurrency-and-Handles.md` | ProcessPool, streaming, DAG |
| `docs/wiki/API-Generated-Modules.md` | Generated binding conventions |
| `docs/wiki/API-Data-Interop.md` | Type conversion, slicing, buffers |
| `docs/wiki/Runbook-Commercial-XCFramework-Runtime.md` | Commercial distribution |

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
1. Regenerates all bindings and fails if diff
2. Builds the package
3. Runs full test suite
4. Runs smoke example (matplotlib headless)

Ensure CI passes before merging.

## Recent Major Features (Current)

### ResourceMonitor (Darwin)
- Thermal state monitoring via `ProcessInfo.thermalState`
- Memory pressure via `host_statistics64`
- Per-worker CPU/RSS via `proc_pidinfo`
- Adaptive sampling (0.5s under pressure, 2s idle)
- Automatic backpressure when thresholds exceeded

### Streaming API
- `CancellableStream<T>` for reliable generator cleanup
- V2 cooperative abort via `SIGUSR1`
- Three-tier API: Raw pickle, custom decoder, `PythonConvertible`
- Full coverage for `evalStream`, `methodStream`, `invokeStream`

### Production Hardening
- `WorkerResourceLimits` with `RLIMIT_AS` enforcement
- `BackpressurePolicy.reject` and `.suspend`
- Formal lifecycle: `warmup`, `drain`, `resume`
- Worker quarantine after repeated failures
- `DAGFailurePolicy.continueIndependent`

### Commercial XCFramework Runtime
- Binary artifact distribution track
- URL-safe package manifest template
- Worker bundling as separate app artifact
- Dual-mode demo dependency wiring

### Generated Modules
- **LlamaCpp**: llama-cpp-python bindings with 8 value-type structs
- **Transformers**: In-place generated against `swiftpython_transformers` shim
- **Remote wrappers**: `rNumPy`, `rPandas` for ProcessPool structured invocation
