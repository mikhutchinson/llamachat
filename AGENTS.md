# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

SwiftPython is a Swift-Python bridge library. See `README.md` for full details. The codebase is a single Swift package (not a monorepo) with generated bindings for 13+ Python libraries. No Docker or external services required.

### Environment requirements

- **Swift 6.0.3+** — `Package.swift` requires swift-tools-version 6.0.
- **Python 3.10 + dev headers** — `Package.swift` hardcodes `-lpython3.10` and `-I/usr/include/python3.10` on Linux. Python 3.12 (system default on Ubuntu 24.04) will NOT work.
- **Python packages** — Core: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `Pillow`, `opencv-python`, `networkx`. Install with `python3.10 -m pip install`.
- **System packages** — `build-essential`, `cmake`, `ninja-build`, `libncurses6`.

### Environment variables (persisted in `~/.bashrc`)

These must be set for `swift build` / `swift test` to find `libpython3.10`:

```
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib
C_INCLUDE_PATH=/usr/include/python3.10
MPLBACKEND=Agg
PYTHONFAULTHANDLER=1
```

### Build and test

```bash
PYTHON_LIB=$(find /usr/lib /usr/lib/x86_64-linux-gnu -name 'libpython3.10.so*' 2>/dev/null | head -1)

# Build
swift build --explicit-target-dependency-import-check=none -Xlinker "$PYTHON_LIB"

# Test (run each module separately per CI pattern to avoid teardown crashes)
swift test --explicit-target-dependency-import-check=none -Xlinker "$PYTHON_LIB" --filter "^SwiftPythonTests\."

# Run examples (use stdbuf to flush output before _Exit)
stdbuf -oL swift run --explicit-target-dependency-import-check=none -Xlinker "$PYTHON_LIB" examples --numpy --networkx
```

### Gotchas

- **`-Xlinker "$PYTHON_LIB"`** — Always pass the explicit `libpython3.10.so` path to the linker. Without it, the linker may fail to resolve `-lpython3.10`.
- **`_Exit(0)` in examples** — The examples binary calls `_Exit(0)` to avoid Python teardown crashes. This does NOT flush stdio buffers, so pipe through `stdbuf -oL` when capturing output.
- **`SWIFTPYTHON_EXAMPLES_REENTRY_GUARD`** — The examples binary sets this env var to prevent re-entry from Python subprocesses. It should not leak to the parent shell, but if you see empty output from examples, ensure this var is unset.
- **ProcessPool performance tests** — `ProcessPoolPerformanceTests` may fail in cloud VMs due to tight latency budgets (600μs / 750μs). These are not functional failures.
- **Test modules should be run individually** — The CI runs each test module in its own `swift test` invocation to avoid cross-module teardown crashes. See `.github/workflows/ci.yml` for the full list.
- **Optional test modules** — `PyTorchTests`, `TorchvisionTests`, `MLXTests`, `TransformersTests`, `LlamaCppTests` require optional Python packages. They skip gracefully if the packages aren't installed.
