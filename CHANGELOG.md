# Changelog

All notable changes to LlamaInferenceDemo are documented here.

---

## [Unreleased] — 2026-02-24

### VLM Pipeline — Structured Output & Multi-Model Context Improvements

#### Overview

End-to-end rework of the two-model vision pipeline (Gemma 3 VLM → Qwen3 VL chat). Addresses caption quality degradation, context contamination across images, and PDF extraction failures.

---

#### 1. Structured VLM Output — `VLMKernel.swift`, `ChatViewModel.swift`

Added structured JSON captioning from the VLM with a new `formattedCaption()` method on `VLMCaptionResult`.

**VLM prompt** now requests four fields:
- `objects` — every distinct person (with clothing detail), object, or UI element; no repeats
- `colors` — all distinct colors present
- `text` — verbatim transcription of all visible text (headings, labels, numbers, legends)
- `description` — multi-sentence scene description

**`formattedCaption()`** parses the JSON and renders:
```
Objects: adult man in gray sweatpants and Nike sneakers, girl in pink pants, ...
Colors: gray, pink, brown, red, ...
Text: PIPELINE FULLY OPTIMIZED | 97.6% idle | 60 fps | ...
Description: A man and young girl sit side by side on a wooden bench ...
```
Falls back to raw string for non-JSON output (backward compatible).

**Files changed:**
- `Sources/Python/VLMKernel.swift` — `_STRUCTURED_PROMPT`, `_CAPTION_SCHEMA`, `formattedCaption()`
- `UI/ChatViewModel.swift` — `captionImage` logs structured vs free-form, `buildAttachmentContext` prefixes `Auto-caption:`
- `Tests/LlamaInferenceDemoTests.swift` — 6 unit tests: full JSON, partial fields, free-form fallback, empty, unknown keys, whitespace

---

#### 2. Removed Grammar-Constrained Sampling — `VLMKernel.swift`

**Problem:** `response_format={"type": "json_object", "schema": ...}` passed to `llama-cpp-python` enables GBNF grammar-constrained token sampling. This cripples Gemma 3's vision attention — the grammar forces token selection down a constrained path before the model has fully processed the image, causing hallucinations and omissions (e.g., missing an entire adult in a photo).

**Fix:** Removed `response_format` entirely. The model now generates JSON naturally via prompt guidance alone. `formattedCaption()` already falls back gracefully if JSON is malformed.

---

#### 3. Token Budget Raised to 1024 — `VLMKernel.swift`, `ChatViewModel.swift`

Previous 256-token cap truncated output for anything beyond simple photos. Text-heavy images (infographics, screenshots, UI mockups) need more headroom.

| Location | Before | After |
|---|---|---|
| Python `caption()` / `caption_file()` default | 256 | 1024 |
| Swift `VLMKernel.captionFile` default | 256 → 512 | 1024 |
| `captionImage()` default | 256 → 512 | 1024 |
| `convertPendingToMessageAttachments` cap | 512 | 1024 |

---

#### 4. Context Framing — `InferenceScheduler.swift`, `StreamingThinkingSupport.swift`

**Problem:** The text model was misinterpreting the structured VLM output as a task to perform (triggering reasoning about "how to describe an image") and echoing constraint language in its responses.

**Fixes:**
- Attachment context wrapped in `<current_attachment_context>` XML tag (scoped to the current message's image, not global context)
- Imperative framing (`[Reference context — do not describe or analyse...]`) replaced with declarative XML — no imperatives to echo back
- `attachmentOnlyDefaultPrompt = ""` — nothing shown in the UI message bubble when the user sends an image with no text
- `attachmentOnlyModelPrompt` added — sent to the model (not displayed) to frame its role:
  > "You are part of a multi-model pipeline. A vision model has already processed the image attached to this message — its analysis is in `<current_attachment_context>` above. This is a different image from any previously discussed in this conversation. Base your response only on the current image's context. If the user wants to discuss an earlier image, they will reference it by name."

---

#### 5. Cross-Image Context Contamination Fix — `InferenceScheduler.swift`, `StreamingThinkingSupport.swift`

**Problem:** After sending multiple images in a session, the model would conflate the current image's analysis with captions from previous turns in its KV cache (e.g., applying infographic metrics to a photo of apples).

**Fix:** 
- Renamed XML tag to `<current_attachment_context>` to make temporal scope explicit
- Model prompt explicitly states "this is a different image from any previously discussed" and directs the model to focus on the current context block only

---

#### 6. PDF Extraction — Empty Page Fallback — `DocumentExtractor.swift`

**Problem:** Some PDFs (unusual CID/Type3 font encoding, print-to-PDF output) returned empty text from pdfminer per-page extraction. `combinedText` emitted `--- Page N ---\n` markers with no content, which the LLM received as context but couldn't use.

**Fixes:**
- After pdfminer per-page pass, if total extracted chars == 0, retry with MarkItDown's whole-document converter as fallback (handles different encoding paths)
- `combinedText` now filters out empty pages before joining; single-page results drop the marker entirely for cleaner context injection
- If both pdfminer and MarkItDown return empty, the file is genuinely image-based (scanned) — OCR via pytesseract/pdf2image is the next step

---

### CodeAct Agent Loop — Python REPL Agent Execution

#### Overview

Implemented the agent loop from "Executable Code Actions Elicit Better LLM Agents" (arXiv 2402.01030v4 Appendix E). The model can emit `<execute>…</execute>` blocks containing Python code; the code runs in the existing `PythonSandbox` and the stdout/stderr is fed back as an `Observation:` user turn, creating a REPL-style interaction loop for autonomous tool use without OpenAI/Llama function-calling APIs.

---

#### 1. Core Agent Logic — `CodeActAgent.swift`

**New file:** `Sources/Engine/CodeActAgent.swift`

- `systemPrompt` — verbatim zero-shot system prompt from Appendix E
- `maxIterations = 10` — safety guard to prevent infinite loops
- `parseExecuteBlock(_ text:)` — extracts first `<execute>…</execute>` block
- `formatObservation(_ output: RunOutput)` — formats sandbox output as `Observation:\n{stdout/stderr/error}`

---

#### 2. Chat Integration — `ChatViewModel.swift`

**New `@Published` property:**
- `codeActEnabled: Bool` — persisted to `UserDefaults` via `SettingsKeys.codeActEnabled`

**New method:**
- `setCodeActEnabled(_ enabled: Bool)` — toggles agent mode, clears session when enabling (ensures clean state)

**Routing in `sendMessage()`:**
```swift
if codeActEnabled && pythonSandbox != nil {
    await _runCodeActLoop(prompt: resolvedPrompt, documentContext: docContext)
} else {
    await _sendMessage(prompt: resolvedPrompt, documentContext: docContext)
}
```

**New method:** `_runCodeActLoop(prompt:documentContext:)`
- Creates inference session with CodeAct system prompt via `createSessionWithHistory`
- Streams each turn using `completeStreamWithMemoryManagement`
- On `<execute>` block: runs code in `PythonSandbox.run()`, appends `Observation:` as `.user` turn
- Loops up to 10 iterations until model responds without `<execute>` (final answer)
- Respects stop button via existing `stopRequested` mechanism
- Shows intermediate agent steps (thoughts + code) as assistant messages with live metrics

---

#### 3. UI Toggle — `ContentView.swift`

**`ComposerView` additions:**
- `@Binding var codeActEnabled: Bool` — two-way binding to view model
- `let sandboxReady: Bool` — controls visibility

**New toolbar button:**
- `cpu` SF Symbol toggle (only shown when sandbox ready)
- Green (`theme.accent`) = agent mode on, gray (`theme.textTertiary`) = off
- Help text: "Enable Agent mode (CodeAct)" / "Agent mode ON — disable CodeAct loop"

---

#### 4. Rendering Fix — `LatexPreprocessor.swift`

**New preprocessing step (step 0a):**
- `replaceExecuteTags(_ input:)` converts `<execute>…</execute>` blocks to fenced ` ```python ` code blocks
- Prevents raw tag display in chat
- Prevents Python `#` comment lines from being misrendered as markdown H1 headings
- Handles multiline content, normalizes whitespace

---

#### 5. Settings — `SettingsDefaults.swift`

**New keys:**
- `SettingsKeys.codeActEnabled = "codeActEnabled"`
- `SettingsDefaults.codeActEnabled = false` (default off)

---

### Files Changed

| File | Change |
|---|---|
| `Sources/Engine/CodeActAgent.swift` | New — agent system prompt, parsing, observation formatting |
| `Sources/Engine/VLMKernel.swift` | Structured prompt, text field, grammar constraint removal, 1024 budget, `formattedCaption()` |
| `Sources/Python/DocumentExtractor.swift` | MarkItDown fallback for empty PDFs, `combinedText` empty-page filter |
| `Sources/Engine/InferenceScheduler.swift` | `<current_attachment_context>` tag |
| `Sources/Engine/StreamingThinkingSupport.swift` | `attachmentOnlyDefaultPrompt = ""`, `attachmentOnlyModelPrompt` |
| `UI/ChatViewModel.swift` | `codeActEnabled` property, `setCodeActEnabled()`, `_runCodeActLoop()` REPL implementation; `Auto-caption:` prefix, `captionImage` structured logging, 1024 token cap, `attachmentOnlyModelPrompt` routing |
| `UI/ContentView.swift` | `ComposerView` toggle button, `codeActEnabled`/`sandboxReady` params |
| `UI/SettingsDefaults.swift` | `codeActEnabled` UserDefaults key and default |
| `ChatUIComponents/LatexPreprocessor.swift` | `replaceExecuteTags()` — renders `<execute>` as fenced code blocks |
| `Tests/LlamaInferenceDemoTests.swift` | 6 `VLMCaptionResultTests` |

---

## [Unreleased] — 2026-02-24

### Sidebar Conversation Rename — Inline Editing

#### Overview

Replaced the broken sheet-based rename flow with inline editing directly in the sidebar conversation list. The sheet-based approach had a race condition where dismissal happened before the async rename completed, causing the operation to fail silently.

---

#### 1. Inline Editing — `ContentView.swift`, `ConversationRow`

**`ConversationRow`** now conditionally renders either:
- Static `Text(conversation.title)` when not editing
- `TextField` with `.textFieldStyle(.plain)` when in edit mode

**State management:**
- `editingConversationID: String?` — which conversation is being edited
- `editingTitle: String` — temporary title during edit
- `@FocusState` — auto-focuses the TextField when entering edit mode

**Keyboard handling:**
- Enter/Return → saves the new title
- Focus loss (clicking elsewhere) → cancels edit
- Tapping another row while editing → cancels edit, selects new conversation

---

#### 2. Fixed Rename Dismiss Bug

**Problem:** The original sheet-based rename had `showRenameSheet = false` outside the `Task` that performed the async rename:

```swift
// BROKEN (original)
onSave: {
    Task {
        await viewModel.renameConversation(id: id, newTitle: title)
    }
    showRenameSheet = false  // ← Called immediately, NOT awaited
}
```

The sheet dismissed before the async operation completed, potentially causing the captured `id` to become stale if view state changed during the race.

**Fix:** Inline editing eliminates the sheet entirely. The save action happens synchronously on the main actor, with the actual persistence async in a fire-and-forget Task.

---

#### 3. Context Menu Options Preserved

Right-click menu on conversation rows still provides:
- **Rename** — triggers inline editing (was broken sheet, now works)
- **Export…** — saves conversation as Markdown file via `NSSavePanel`
- **Delete** — removes conversation with destructive button styling

---

### Files Changed

| File | Change |
|---|---|---|
| `UI/ContentView.swift` | Inline editing state (`editingConversationID`, `editingTitle`), `ConversationRow` passes edit binding, removed `RenameConversationSheet` |
| `Tests/ChatPersistenceTests.swift` | `testUpdateConversationTitle` — verifies title persistence |
