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
- Homebrew Python 3.13: `brew install python@3.13`
- Python packages: `pip3 install llama-cpp-python`
- A GGUF model file (e.g. from [Hugging Face](https://huggingface.co/models?search=gguf))
- SwiftPython runtime (included automatically via SPM)

## Build

```bash
git clone https://github.com/mikhutchinson/llamachat.git
cd llamachat

export SWIFTPYTHON_COMMERCIAL_PACKAGE_URL=https://github.com/mikhutchinson/swiftpython-commercial.git
export SWIFTPYTHON_COMMERCIAL_PACKAGE_VERSION=0.1.3

./scripts/build-app-bundle.sh
cp -R "build/Llama Chat.app" /Applications/
open "/Applications/Llama Chat.app"
```

## Usage

1. Launch Llama Chat
2. Select a GGUF model from Settings or the toolbar dropdown
3. Start chatting

Use the Model Hub to browse and download models directly from Hugging Face.

## License

AGPL-3.0 License. See [LICENSE](LICENSE) for details.

This project uses SwiftPythonRuntime via a binary xcframework. The runtime is dual-licensed (AGPL-3.0 / Commercial) — the xcframework satisfies the linking requirements while keeping the demo code open source.
