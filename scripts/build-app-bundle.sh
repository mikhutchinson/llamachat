#!/usr/bin/env bash
# Build LlamaChatUI as a proper macOS .app bundle for `open` and Dock activation.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
# Build first, then locate output
APP_NAME="Llama Chat"
APP_DIR="$PKG_DIR/build/Llama Chat.app"

echo "Building LlamaChatUI and SwiftPythonWorker..."
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
PYTHON_HOME="/opt/homebrew/opt/python@3.13"
if [ ! -d "$PYTHON_HOME" ] && [ -d "/usr/local/opt/python@3.13" ]; then
    PYTHON_HOME="/usr/local/opt/python@3.13"
fi
PYTHON_FW="$PYTHON_HOME/Frameworks/Python.framework/Versions/3.13"

cat > "$APP_DIR/Contents/MacOS/LlamaChatUI" << LAUNCHER
#!/bin/bash
export PYTHONHOME="${PYTHON_FW}"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:\$PATH"
DIR="\$(cd "\$(dirname "\$0")" && pwd)"
exec "\$DIR/LlamaChatUI.bin" "\$@"
LAUNCHER
chmod +x "$APP_DIR/Contents/MacOS/LlamaChatUI"

# Copy SPM resource bundles into Contents/Resources so Bundle.module
# can locate them at runtime. Required by any package that embeds
# resources (e.g. Textual/iosMath math fonts). Without this step the
# app crashes with EXC_BREAKPOINT inside Math.FontRegistry.graphicsFont.
mkdir -p "$APP_DIR/Contents/Resources"
find "$BUILD_DIR" -maxdepth 1 -name "*.bundle" -type d | while read -r bundle; do
    cp -r "$bundle" "$APP_DIR/Contents/Resources/"
done


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
