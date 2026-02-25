# Llama Chat — Bug Fixes & Pitfalls

Known issues encountered while shipping Llama Chat as a macOS `.app` bundle built with SPM against the SwiftPython commercial runtime.

---

## BUG-001: Finder/Dock launch crash (Worker exit code -1)

**Symptom:** App works from `swift run` in terminal but crashes immediately when launched from Finder or Dock. Worker subprocess exits with code -1.

**Root cause:** Finder/Dock launches don't inherit shell environment variables. The worker subprocess can't find Python site-packages without `PYTHONHOME`.

**Fix:** Use a bash wrapper script as `CFBundleExecutable` that sets `PYTHONHOME` and `PATH` before exec'ing the real binary. `LSEnvironment` in Info.plist was tried first but proved unreliable.

```
Contents/MacOS/
├── LlamaChatUI        ← bash wrapper (CFBundleExecutable)
├── LlamaChatUI.bin    ← actual Swift binary
└── SwiftPythonWorker
```

**Commit:** `build-app-bundle.sh` — wrapper launcher pattern

---

## BUG-002: Bundle.module crash in SPM dependencies (SwiftUIMath, Textual)

**Symptom:** `EXC_BREAKPOINT` on launch. Crash in `Math.FontRegistry.graphicsFont` → `CGFontRef.named` → `Bundle.module` → `fatalError("could not load resource bundle")`.

**Root cause:** SPM's generated `resource_bundle_accessor.swift` uses `Bundle.main.bundleURL` to locate `.bundle` resources. For a macOS `.app` bundle, CoreFoundation resolves `bundleURL` to the `.app` root directory — NOT `Contents/Resources/` or `Contents/MacOS/`.

**Fix:** Copy all `.bundle` directories from the SPM build output to the `.app` root:

```bash
find "$BUILD_DIR" -maxdepth 1 -name "*.bundle" -type d | while read -r bundle; do
    cp -r "$bundle" "$APP_DIR/"
done
```

Resulting layout:
```
Llama Chat.app/
├── swiftui-math_SwiftUIMath.bundle    ← .app root, NOT Contents/Resources/
├── textual_Textual.bundle
└── Contents/
    ├── MacOS/...
    └── Info.plist
```

**Commit:** `build-app-bundle.sh` — resource bundle placement fix

---

## BUG-003: Bundle.module crash in SwiftPythonRuntime (bootstrapSwiftPythonInternalShims)

**Symptom:** `EXC_BREAKPOINT` when sending first message. Crash in `bootstrapSwiftPythonInternalShims()` → `Bundle.module` → `fatalError`. Happens in binary distribution only (xcframework), not source builds.

**Root cause:** The xcframework binary at v0.1.3 and v0.1.4 still contained the old `bootstrapSwiftPythonInternalShims()` code that called `Bundle.module`. The source fix (removing `Bundle.module` lookup) was never actually rebuilt into the xcframework.

**Fix:** Rebuilt xcframework from current SwiftPython source at **v0.1.6**. The function now uses a CWD fallback for source builds and gracefully no-ops for binary distribution.

**Additional pitfall:** v0.1.4 and v0.1.5 were also broken because SPM incremental builds cached the old `.o` files. The xcframework was "rebuilt" but reused stale object files containing the old `Bundle.module` call. Only `rm -rf .build` before `swift build` guarantees a fresh compile. See the xcframework build runbook in SwiftPython for the full procedure.

**Action:** Always use `SWIFTPYTHON_COMMERCIAL_PACKAGE_VERSION=0.1.6` or later.

---

## BUG-004: Python 3.13 not found (linker error on consumer machine)

**Symptom:** Build fails with `ld: library 'python3.13' not found` even though the preflight check passes.

**Root cause:** The preflight check only tested `[ -d "/opt/homebrew/opt/python@3.13" ]` which can pass for partial/broken installs (empty directory, non-Framework build). The linker needs the full Framework path.

**Fix:** Changed preflight to check `[ -d "$PYTHON_HOME/Frameworks" ]` instead of just the prefix directory.

**Action:** Consumer must install via Homebrew: `brew install python@3.13`

---

## BUG-005: PEP 668 — externally-managed-environment (pip install blocked)

**Symptom:** `pip install llama-cpp-python` fails with `error: externally-managed-environment` on newer Homebrew Python.

**Root cause:** PEP 668 marks Homebrew Python as externally managed, blocking system-wide pip installs.

**Fix:** Install with `--break-system-packages --user`:

```bash
/opt/homebrew/bin/python3.13 -m pip install --break-system-packages --user llama-cpp-python
```

The `--user` flag installs into `~/Library/Python/3.13/lib/python/site-packages/` which is on `sys.path` when `PYTHONHOME` is set. Safe with `--user` since it doesn't touch Homebrew-managed directories.

---

## BUG-006: Wrong Python version for pip install

**Symptom:** `Worker 0 crashed with exit code -1` despite `llama-cpp-python` being installed.

**Root cause:** User ran `python3 -m pip install` which installs for the default `python3` (e.g. 3.12), not the Homebrew Python 3.13 that the app uses via `PYTHONHOME`.

**Fix:** Always use the explicit Python 3.13 binary:

```bash
/opt/homebrew/bin/python3.13 -m pip install --break-system-packages --user llama-cpp-python
```

---

## Quick Reference: Full Consumer Setup

```bash
# Prerequisites
brew install python@3.13
/opt/homebrew/bin/python3.13 -m pip install --break-system-packages --user llama-cpp-python

# Build
git clone https://github.com/mikhutchinson/llamachat.git
cd llamachat
export SWIFTPYTHON_COMMERCIAL_PACKAGE_URL=https://github.com/mikhutchinson/swiftpython-commercial.git
export SWIFTPYTHON_COMMERCIAL_PACKAGE_VERSION=0.1.6
./scripts/build-app-bundle.sh
cp -R "build/Llama Chat.app" /Applications/
```
