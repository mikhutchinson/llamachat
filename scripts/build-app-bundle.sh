#!/usr/bin/env bash
# Build LlamaChatUI as a proper macOS .app bundle for `open` and Dock activation.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
# Build first, then locate output
APP_NAME="Llama Chat"
APP_DIR="$PKG_DIR/build/Llama Chat.app"

# ── Preflight checks ──────────────────────────────────────────────
echo "Checking prerequisites..."

# 1. Xcode (not just CLT)
XCODE_PATH="$(xcode-select -p 2>/dev/null || true)"
if [ -z "$XCODE_PATH" ]; then
    echo "error: No developer tools found. Install Xcode from the App Store."
    exit 1
fi
if [[ "$XCODE_PATH" == */CommandLineTools* ]]; then
    echo "error: Xcode Command Line Tools alone are not sufficient."
    echo "       Install Xcode from the App Store, then run:"
    echo "       sudo xcode-select -s /Applications/Xcode.app"
    exit 1
fi

# 2. Swift compiler
if ! command -v swift &>/dev/null; then
    echo "error: Swift compiler not found. Install Xcode from the App Store."
    exit 1
fi
SWIFT_VER="$(swift --version 2>&1 | head -1)"
echo "  Swift: $SWIFT_VER"

# 3. Python 3.13
PYTHON_HOME="/opt/homebrew/opt/python@3.13"
if [ ! -d "$PYTHON_HOME/Frameworks" ]; then
    PYTHON_HOME="/usr/local/opt/python@3.13"
fi
if [ ! -d "$PYTHON_HOME/Frameworks" ]; then
    echo "error: Homebrew Python 3.13 Framework not found. Install with: brew install python@3.13"
    exit 1
fi
echo "  Python: $PYTHON_HOME"

# 4. llama-cpp-python
if ! python3 -c "import llama_cpp" &>/dev/null; then
    echo "warning: llama-cpp-python not installed. Install with: pip3 install llama-cpp-python"
    echo "         The app will build but model loading will fail at runtime."
fi

echo "Prerequisites OK."
echo ""

# ── Build ─────────────────────────────────────────────────────────
echo "Building LlamaChatUI..."
cd "$PKG_DIR"
swift build --product LlamaChatUI

BUILD_DIR="$(dirname "$(find "$PKG_DIR/.build" -name "LlamaChatUI" -type f -not -path '*dSYM*' 2>/dev/null | head -1)")"
[ -z "$BUILD_DIR" ] && { echo "Build failed or LlamaChatUI not found"; exit 1; }

echo "Creating app bundle..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"

cp "$BUILD_DIR/LlamaChatUI" "$APP_DIR/Contents/MacOS/LlamaChatUI.bin"

WORKER_SRC="${SWIFTPYTHON_WORKER_PATH:-}"
if [ -z "$WORKER_SRC" ] || [ ! -f "$WORKER_SRC" ]; then
    WORKER_SRC="$(find "$PKG_DIR/.build/checkouts" -name "SwiftPythonWorker" -type f -not -path '*dSYM*' 2>/dev/null | head -1)"
fi
if [ -z "$WORKER_SRC" ] || [ ! -f "$WORKER_SRC" ]; then
    WORKER_SRC="$BUILD_DIR/SwiftPythonWorker"
fi
if [ -f "$WORKER_SRC" ]; then
    cp "$WORKER_SRC" "$APP_DIR/Contents/MacOS/SwiftPythonWorker"
else
    echo "warning: SwiftPythonWorker not found — app will not be able to run Python"
fi

# Create launcher wrapper that sets up Python environment before exec.
# Required because Finder/Dock launches don't inherit shell env, so
# the worker subprocess can't find Python site-packages without this.
# PYTHON_HOME was already resolved by the preflight checks above.
PYTHON_FW="$PYTHON_HOME/Frameworks/Python.framework/Versions/3.13"

cat > "$APP_DIR/Contents/MacOS/LlamaChatUI" << LAUNCHER
#!/bin/bash
export PYTHONHOME="${PYTHON_FW}"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:\$PATH"
DIR="\$(cd "\$(dirname "\$0")" && pwd)"
exec "\$DIR/LlamaChatUI.bin" "\$@"
LAUNCHER
chmod +x "$APP_DIR/Contents/MacOS/LlamaChatUI"

# Copy SPM resource bundles into the .app root directory. SPM's generated
# resource_bundle_accessor.swift uses Bundle.main.bundleURL (the .app dir)
# NOT Bundle.main.resourceURL (Contents/Resources/). For a macOS .app,
# CoreFoundation detects the bundle structure and sets bundleURL to the
# .app root, so the accessor looks for:
#   /path/to/Foo.app/packagename_TargetName.bundle
# Without this the app crashes with EXC_BREAKPOINT in Bundle.module.
find "$BUILD_DIR" -maxdepth 1 -name "*.bundle" -type d | while read -r bundle; do
    cp -r "$bundle" "$APP_DIR/"
done

if [ -f "$PKG_DIR/assets/AppIcon.icns" ]; then
    mkdir -p "$APP_DIR/Contents/Resources"
    cp "$PKG_DIR/assets/AppIcon.icns" "$APP_DIR/Contents/Resources/AppIcon.icns"
fi

# Info.plist for proper .app behavior (Dock, activation, etc.)
cat > "$APP_DIR/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>LlamaChatUI</string>
    <key>CFBundleIdentifier</key>
    <string>com.swiftpython.LlamaChatUI</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleName</key>
    <string>Llama Chat</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
</dict>
</plist>
PLIST

echo "Created: $APP_DIR"
echo "Run with: open \"$APP_DIR\""
