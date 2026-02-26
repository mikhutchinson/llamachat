# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Dynamic VLM architecture parsing to pass raw `architecture` strings from GGUF metadata down directly to the Python kernel.
- Python `inspect` integration to dynamically discover and attach `llama_cpp.llama_chat_format` vision handlers as they are released without hardcoded Enums.
- Dynamically rendered capitalized model architecture badges on Model Hub cards.

### Fixed
- Fixed an exception where bleeding-edge model architectures like `qwen3vl` crashed the underlying `llama.cpp` metadata parser by recommending a switch to the `JamePeng/llama-cpp-python` open-source branch which patches this early.
- Broadened `.isMMProj` match rules to catch vision projection files explicitly labelled `-clip`, `vision-encoder`, or `-proj`.
