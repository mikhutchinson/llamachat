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

cp "$BUILD_DIR/LlamaChatUI" "$APP_DIR/Contents/MacOS/"

WORKER_SRC="${SWIFTPYTHON_WORKER_PATH:-$BUILD_DIR/SwiftPythonWorker}"
if [ -f "$WORKER_SRC" ]; then
    cp "$WORKER_SRC" "$APP_DIR/Contents/MacOS/SwiftPythonWorker"
else
    echo "warning: SwiftPythonWorker not found at $WORKER_SRC — app will not be able to run Python"
fi

# Copy SPM resource bundles into Contents/Resources so Bundle.module
# can locate them at runtime. Required by any package that embeds
# resources (e.g. Textual/iosMath math fonts). Without this step the
# app crashes with EXC_BREAKPOINT inside Math.FontRegistry.graphicsFont.
mkdir -p "$APP_DIR/Contents/Resources"
find "$BUILD_DIR" -maxdepth 1 -name "*.bundle" -type d | while read -r bundle; do
    cp -r "$bundle" "$APP_DIR/Contents/Resources/"
done

# Binary targets (xcframework) don't generate resource bundles via SPM.
# Look for them in the resolved dependency checkouts (e.g. SwiftPython
# binary package ships SwiftPython_SwiftPythonRuntime.bundle).
if [ -d "$PKG_DIR/.build/checkouts" ]; then
    find "$PKG_DIR/.build/checkouts" -name "*.bundle" -type d -maxdepth 2 | while read -r bundle; do
        BNAME="$(basename "$bundle")"
        if [ ! -d "$APP_DIR/Contents/Resources/$BNAME" ]; then
            echo "Copying dependency resource bundle: $BNAME"
            cp -r "$bundle" "$APP_DIR/Contents/Resources/"
        fi
    done
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
