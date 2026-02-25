# Llama Chat

A native macOS chat app powered by local GGUF language models. Runs entirely on-device with Metal GPU acceleration.

## Features

- Multi-turn chat with conversation persistence
- Model Hub: browse, download, and activate GGUF models from Hugging Face
- Vision: image captioning via VLM
- Document extraction: PDF, DOCX, XLSX
- Code sandbox with Python execution
- LaTeX rendering in responses
- Automatic context management for long conversations

## Requirements

- macOS 15.0+
- **Xcode 16+** — install from the [App Store](https://apps.apple.com/app/xcode/id497799835), then run `sudo xcode-select -s /Applications/Xcode.app`. Command Line Tools alone are not sufficient.
- **Homebrew**: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- **Homebrew Python 3.13**: `brew install python@3.13`
- **Python venv** (required for document extraction, code sandbox, and embeddings):
  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r python-requirements.txt
  ```
- A GGUF model file (e.g. from [Hugging Face](https://huggingface.co/models?search=gguf))

## Build

```bash
git clone https://github.com/mikhutchinson/llamachat.git
cd llamachat

export SWIFTPYTHON_COMMERCIAL_PACKAGE_URL=https://github.com/mikhutchinson/swiftpython-commercial.git
export SWIFTPYTHON_COMMERCIAL_PACKAGE_VERSION=0.1.4

./scripts/build-app-bundle.sh
cp -R "build/Llama Chat.app" /Applications/
open "/Applications/Llama Chat.app"
```

## Troubleshooting

**"failed to build module 'Foundation'" or "redefinition of module 'SwiftBridging'"**

Swift can't find a compatible SDK. Install Xcode from the App Store, then:
```bash
sudo xcode-select -s /Applications/Xcode.app
```

**"xcode-select: note: No developer tools were found"**

Install Xcode (not just Command Line Tools) from the App Store.

**"Homebrew Python 3.13 Framework not found"** (build-app-bundle.sh fails)

Install Homebrew and Python 3.13:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.13
```

**PDF/DOCX/XLSX shows "content could not be extracted"**

The Python venv is missing or incomplete. Create it from the repo root:
```bash
python3 -m venv .venv
.venv/bin/pip install -r python-requirements.txt
```
No rebuild needed — restart the app and document extraction will work. See `BUGFIX.md` (BUG-007) for details.

## Usage

1. Launch Llama Chat
2. Select a GGUF model from Settings or the toolbar dropdown
3. Start chatting

Use the Model Hub to browse and download models directly from Hugging Face.

## License

AGPL-3.0 License. See [LICENSE](LICENSE) for details.

This project uses SwiftPythonRuntime via a binary xcframework. The runtime is dual-licensed (AGPL-3.0 / Commercial) — the xcframework satisfies the linking requirements while keeping the demo code open source.
