## Bug: Conversation Rename Not Working — Sheet Dismiss Race + Inline Editing (2026-02-24)

### Problem Statement

**Symptoms:**
- Clicking "Rename" from the conversation context menu did not actually rename the conversation
- The rename sheet appeared, but after editing the title and clicking Save, the conversation retained its original title
- No error was shown to the user — the operation appeared to succeed while silently failing

**Secondary UX Issue:**
- The sheet-based rename flow felt heavy for a quick title edit
- User requested inline editing (edit directly in the sidebar row) instead of a popover/sheet

---

### Root Cause Analysis

#### Primary Cause: Race Condition in Sheet Dismissal

The original sheet-based rename had a critical bug in the `onSave` closure:

```swift
// BROKEN — ContentView.swift, original implementation (line ~199-203)
.sheet(isPresented: $showRenameSheet) {
    if let id = renameConversationID {
        RenameConversationSheet(
            title: $renameConversationTitle,
            onSave: {
                Task {
                    await viewModel.renameConversation(id: id, newTitle: renameConversationTitle)
                }
                showRenameSheet = false  // ← CALLED IMMEDIATELY, NOT INSIDE TASK
            },
            // ...
        )
    }
}
```

**Chain of failure:**
1. User clicks Save → `onSave` closure executes
2. `Task { await viewModel.renameConversation(...) }` is created and scheduled (async)
3. `showRenameSheet = false` runs **immediately** on the main thread
4. Sheet dismisses, view state updates, potentially causing `renameConversationID` to change or become nil
5. The `Task` executes asynchronously with the captured `id` and `renameConversationTitle`
6. If view state shifted during the race (e.g., user selected another conversation), the wrong ID might be used
7. More critically: there's no guarantee the Task completes before the view hierarchy changes
8. Result: rename appears to work (no error) but the title doesn't persist

**Why it was hard to notice:**
- The code "looks" correct at first glance — Task contains the async work
- The bug is subtle: the sheet dismissal happens outside the Task, creating a race
- No error handling or completion callback meant silent failures

---

### Solution

#### Part 1: Inline Editing (UX Improvement)

Replaced the sheet-based flow with inline editing directly in the sidebar row:

**`ConversationRow` changes:**
```swift
struct ConversationRow: View {
    let conversation: Conversation
    let isSelected: Bool
    let isEditing: Bool                    // NEW
    @Binding var editingTitle: String      // NEW
    var onSaveEdit: () -> Void             // NEW
    var onCancelEdit: () -> Void           // NEW
    @FocusState private var isTitleFieldFocused: Bool  // NEW

    var body: some View {
        HStack(spacing: 8) {
            // ... icon ...
            VStack(alignment: .leading, spacing: 2) {
                if isEditing {
                    TextField("Title", text: $editingTitle)
                        .textFieldStyle(.plain)        // Inline style
                        .font(.system(size: 13))
                        .focused($isTitleFieldFocused)
                        .onSubmit { onSaveEdit() }
                        .task { isTitleFieldFocused = true }  // Auto-focus
                } else {
                    Text(conversation.title)
                        .font(.system(size: 13))
                        .lineLimit(1)
                }
                // ... date label ...
            }
            Spacer()
        }
        // ... styling ...
    }
}
```

**`ContentView` state changes:**
```swift
// BEFORE (broken):
@State private var renameConversationID: String?
@State private var renameConversationTitle = ""
@State private var showRenameSheet = false

// AFTER (working):
@State private var editingConversationID: String?
@State private var editingTitle: String = ""
```

**Edit flow:**
```swift
private func startEditing(_ conv: Conversation) {
    editingConversationID = conv.id
    editingTitle = conv.title
}

private func saveEdit() {
    guard let id = editingConversationID else { return }
    let trimmed = editingTitle.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { cancelEdit(); return }
    
    Task {
        await viewModel.renameConversation(id: id, newTitle: trimmed)
    }
    clearEditState()
}

private func cancelEdit() {
    clearEditState()
}
```

**Interaction details:**
- Right-click → "Rename" triggers `startEditing(conv)`
- Row becomes a focused `TextField` with current title selected
- Enter/Return → saves (calls `saveEdit()` via `.onSubmit`)
- Focus loss (clicking elsewhere) → cancels (via `.onChange(of: isEditing)`)
- Tapping another row while editing → cancels edit, then selects that row

---

#### Part 2: Removed `RenameConversationSheet`

The entire `RenameConversationSheet` struct was removed as it's no longer needed. The inline editing provides a cleaner, more macOS-native experience (similar to renaming files in Finder's sidebar).

---

### Files Changed

1. **`UI/ContentView.swift`**
   - Replaced `renameConversationID`, `renameConversationTitle`, `showRenameSheet` with `editingConversationID`, `editingTitle`
   - Modified `ConversationRow` to accept `isEditing`, `editingTitle` binding, and save/cancel callbacks
   - Added inline editing helper functions: `startEditing()`, `saveEdit()`, `cancelEdit()`, `clearEditState()`
   - Updated context menu "Rename" action to trigger inline editing instead of sheet
   - Removed `.sheet(isPresented: $showRenameSheet)` modifier
   - Removed `RenameConversationSheet` struct entirely

---

### Verification

**Test:**
1. Right-click conversation → Rename
2. Type new title → press Enter
3. Title updates immediately in sidebar
4. Switch conversations → verify persistence after app restart

**Edge cases tested:**
- Empty title (cancels edit)
- Whitespace-only title (cancels edit)
- Tapping away while editing (cancels edit)
- Tapping another row while editing (cancels, then selects other)
- Rapid rename operations (no race conditions)

---

### Lessons Learned

#### 1. Fire-and-Forget Tasks Need Careful State Management

When using `Task { ... }` without `await`, the code after the Task runs immediately. If that code affects view state (like dismissing a sheet), it can race with the async work.

**Safe pattern for sheet dismissal with async work:**
```swift
// CORRECT: Dismiss inside the Task
Task {
    await viewModel.doAsyncWork()
    await MainActor.run {
        showSheet = false  // Dismiss AFTER work completes
    }
}
```

Or better yet: avoid sheets for quick edits entirely and use inline editing.

#### 2. SwiftUI `@State` and Async Closures

Captured values in closures (like `id` in the original `onSave`) can become stale if view state changes between closure creation and execution. Inline editing avoids this by using the ID at the moment the edit starts, not at save time.

#### 3. macOS Sidebar UX Pattern

Finder, Notes, and other native macOS apps use inline editing for sidebar items. Users expect this interaction model — sheets feel foreign for quick property edits.

---

## Bug: Model Selector Pill — Uneven Outer Halo (2026-02-24)

### Problem Statement

**Symptoms:**
- Model selector pill in toolbar showed "capsule inside a capsule" appearance
- Outer halo was tight on left/right sides but visually thicker top/bottom — uneven thickness
- Multiple attempts to remove the halo (Button chrome, stroke positioning) failed
- User wanted uniform halo thickness or clean removal

**Screenshot:**
```
[  Qwen3 >  ]  ← uneven outer ring visible around pill
```

### Root Cause

**Layering issue with capsule strokes in SwiftUI:**

1. **Initial attempt:** `.background(Capsule().fill())` + `.overlay(Capsule().strokeBorder())` created two separate capsule geometries with anti-aliasing artifacts
2. **Second attempt:** `.stroke()` centered on edge drew half outside, half inside — creating uneven appearance
3. **Third attempt:** Even with `.inset(by: 0.5)`, the stroke still created visual artifacts at different screen scales
4. **macOS toolbar:** Adding `Button` introduced AppKit bezel chrome that added its own outer ring

The fundamental issue: trying to fight the halo with strokes — strokes inherently have edge cases with anti-aliasing and positioning.

### Solution

**Embrace the halo — make it uniform with a dedicated outer layer:**

Instead of fighting the visual artifact, add an explicit outer capsule layer with fixed padding to ensure uniform thickness:

```swift
private var modelSelectorPill: some View {
    ZStack {
        // Outer halo layer — uniform thickness on all sides
        Capsule(style: .continuous)
            .fill(currentTheme == .light ? Color.black.opacity(0.05) : Color.white.opacity(0.08))
        
        // Inner pill — your actual content
        HStack(spacing: 6) {
            Text(activeModelShortName)
                .font(.system(size: 13, weight: .semibold))
            Image(systemName: "chevron.right")
                .font(.system(size: 11, weight: .semibold))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
        .background(
            Capsule(style: .continuous)
                .fill(currentTheme.buttonOverlay)
        )
        .overlay(
            Capsule(style: .continuous)
                .inset(by: 0.5)
                .stroke(currentTheme.divider.opacity(0.78), lineWidth: 1)
        )
        .padding(2.5)  // Fixed halo thickness
    }
    .contentShape(Capsule(style: .continuous))
    .onTapGesture { isModelPopoverPresented.toggle() }
    // ... popover, accessibility
}
```

**Key changes:**
1. Removed `Button` wrapper — use `.onTapGesture` on custom view (eliminates AppKit toolbar bezel)
2. `ZStack` with two capsule layers — outer halo + inner pill
3. Fixed `.padding(2.5)` between layers ensures **exactly 2.5pt halo on all sides**
4. No more fighting stroke positioning — the halo is now a separate filled shape

### Files Changed

- **`UI/ContentView.swift`**
  - `modelSelectorPill` view: Replaced `Button` with custom tappable `ZStack`
  - Added explicit outer capsule layer for uniform halo
  - Preserved popover behavior and accessibility traits

### Trade-off

**Before:** Uneven, visually inconsistent outer ring that looked like a glitch.

**After:** Uniform, intentional halo thickness (2.5pt). The pill now looks designed rather than buggy.

**Future:** Halo thickness can be tuned by changing single value `2.5` → `1.0` (thinner) or `4.0` (thicker).

---

## Bug: LaTeX \underbrace / \overbrace Renders as Blank Box (2026-02-24)

### Problem Statement

**Symptoms:**
- LaTeX equations containing `\underbrace{...}_{...}` or `\overbrace{...}^{...}` rendered as blank/empty boxes
- The equation `$$f(x) = \underbrace{g(h(x))}_{\text{outer function}}$$` produced only a blank box where the math should appear
- Other equations without these commands rendered correctly

**Example from user report:**
```latex
$$
f(x) = \underbrace{g(h(x))}_{\text{outer function}} \quad \text{(where } g(u) \text{ is the outer layer)}
$$
```
Rendered as:
```
[blank box]
```

### Root Cause

**SwiftUIMath (iosMath) does not support the `\underbrace` and `\overbrace` amsmath commands.**

When Textual's math pipeline (via SwiftUIMath) encounters an unrecognized LaTeX command:
1. Parser encounters `\underbrace{...}` — command not in supported set
2. Typesetter fails to produce a display list
3. Render produces zero-size output
4. Result: blank box (user highlighted the empty space in their screenshot)

Note: `\text{...}` is **supported** by SwiftUIMath — it maps to roman font style. The issue was specifically `\underbrace` causing the entire expression to fail.

### Solution

Add normalization in `LatexPreprocessor` to strip unsupported brace annotations before they reach the renderer:

**`stripBraceAnnotation()` helper** — strips `\underbrace{content}_{label}` → `content`:
```swift
private static func stripBraceAnnotation(_ latex: String, command: String, trailingMarker: String) -> String {
    // Find \underbrace or \overbrace
    // Extract content from first brace group
    // Optionally consume trailing _{...} or ^{...} label group
    // Replace entire construct with just the content
}
```

**Applied in both contexts:**
1. `normalizeBlockMathContent()` — for `$$...$$` and `\[...\]` blocks
2. `normalizeInlineMathContent()` — for `$...$` inline math

**Result:** The problematic equation becomes:
```latex
f(x) = g(h(x)) \quad \text{(where } g(u) \text{ is the outer layer)}
```
— which SwiftUIMath renders correctly (brace label is lost but equation is visible).

### Files Changed

- **`ChatUIComponents/LatexPreprocessor.swift`**
  - Added `stripBraceAnnotation()` helper method
  - Applied to both `normalizeBlockMathContent()` and `normalizeInlineMathContent()`

- **`Tests/LatexPreprocessorTests.swift`**
  - Added 6 regression tests covering:
    - `\underbrace` with label (block math)
    - `\underbrace` without label (block math)
    - `\overbrace` with label (block math)
    - `\underbrace` in inline math
    - Exact chain rule equation from bug report

### Verification

**Tests:**
```
Test Suite 'LatexPreprocessorTests' passed
Executed 58 tests, with 0 failures (0 unexpected)
```

**New tests specifically covering this fix:**
- `testBlockMath_underbraceWithLabel` ✅
- `testBlockMath_underbraceWithoutLabel` ✅
- `testBlockMath_overbraceWithLabel` ✅
- `testInlineMath_underbraceStripped` ✅
- `testBlockMath_chainRuleExact` ✅ (exact equation from bug report)

### Trade-off

**Loss:** Brace annotation labels are not rendered (e.g., "outer function" under the brace disappears).

**Gain:** The equation content itself becomes visible instead of a blank box. For chat UI, seeing the equation without the brace label is preferable to seeing nothing.

Future improvement could render braces as plain text annotations outside the math block.

---

## Bug: Inline Math Trade-offs in MarkdownUI (Sizing/Jitter/Copy) (2026-02-13)

### Problem Statement

After attempting true inline LaTeX rendering with MarkdownUI inline image providers, chat rendering regressed:

- inline formulas did not reliably match surrounding font sizing
- line flow could jitter when inline images resolved asynchronously
- copy/selection behavior remained constrained by MarkdownUI text-fragment boundaries

### Root Cause

These limitations are tied to MarkdownUI's current text/attachment architecture and are documented in upstream issues:

- inline image font-size mismatch (`gonzalezreal/swift-markdown-ui#418`)
- image insertion/removal jitter (`gonzalezreal/swift-markdown-ui#415`)
- broader text-selection architecture limits (`gonzalezreal/swift-markdown-ui#264`)

### Fix

Migrate chat markdown rendering from MarkdownUI to Textual:

1. Replace `swift-markdown-ui` dependency with `textual`.
2. Update `MessageContentView` to render via `StructuredText(markdown:..., syntaxExtensions: [.math])`.
3. Keep `LatexPreprocessor`, but repurpose it as a Textual-oriented normalization pass:
   - `$$...$$` / `\[...\]` -> fenced `math` code blocks
   - `\(...\)` / `$`...`$` -> `$...$`
   - escape likely currency-dollar values
   - normalize markdown-escaped subscripts (`\_`) inside inline math back to TeX (`_`)
   - escape markdown control chars inside inline math before parse (excluding `_` so TeX subscripts are preserved)
4. Update LaTeX preprocess tests for normalization and protection behavior.

### Files Changed

- `Package.swift`
- `UI/MessageContentView.swift`
- `ChatUIComponents/LatexPreprocessor.swift`
- `Tests/LatexPreprocessorTests.swift`

### Outcome

- Inline + block math now render through Textual's math pipeline.
- Selection/copy integration is handled by Textual's richer text engine.
- Preprocessing no longer simplifies math expressions to inline code text.

---

## Bug: Inline LaTeX Image Rendering Regressions (2026-02-13)

### Problem Statement

**Symptoms:** Rendering inline LaTeX as Markdown inline images regressed readability and copy behavior:

- inline formulas appeared undersized vs surrounding text
- baseline/line rhythm drifted in paragraph and table-like content
- selecting/copying text produced object-replacement glyphs (`￼`) where inline math appeared

### Root Cause

The inline-image approach represented formulas as text attachments in Markdown's inline run. That improved math fidelity but introduced typography and text-selection artifacts that were worse than the prior compromise.

### Fix

Rollback to the previous stable inline rendering path:

1. Restore `LatexPreprocessor` inline conversion to simplified inline code text (no inline image tokens).
2. Remove MarkdownUI custom inline-image math rendering from `MessageContentView`.
3. Keep hardening for false-positive `$...$` captures (currency/plain-text guard).

### Files Changed

- `ChatUIComponents/LatexPreprocessor.swift`
- `UI/MessageContentView.swift`
- `Tests/LatexPreprocessorTests.swift`

### Outcome

- Inline formulas return to stable sizing/flow and copy behavior.
- Block math rendering remains unchanged (pre-rendered SwiftMath bitmap path).
- True inline math rendering is deferred until a typography-safe + copy-safe strategy is ready.

---

## Bug: Model Hub Scroll Jank During Active Downloads (2026-02-13)

### Problem Statement

**Symptom:** Scrolling the Model Hub results list became sluggish while one or more model downloads were active.

### Root Cause

`ModelHubView` / `ModelCardRow` read `downloadManager.activeDownloads` directly during row rendering. Download delegate callbacks update byte counters very frequently, which caused repeated invalidation and re-layout of the results list while the user was scrolling.

### Fix

Apply the same transient-state principle used by the streamed live preview fix:

1. Keep high-frequency download progress in a **transient live snapshot** inside `ModelHubView` (synchronized loop).
2. Quantize progress to percent-level in that snapshot to coalesce byte-level noise.
3. Render row progress UI from the snapshot (`liveDownloads`) instead of directly binding rows to `activeDownloads`.
4. Add `Equatable` conformance for `DownloadState` / `DownloadTask` so unchanged snapshots are suppressed naturally.

### Files Changed

- `UI/ModelHubView.swift`
- `UI/ModelCardRow.swift`
- `Sources/Models/HubModels.swift`

### Outcome

- Model Hub scrolling remains responsive while downloads are in-flight.
- Progress UI still updates in near real time, without forcing full row-tree churn per network progress callback.

---

## Bug: Streaming Scroll Jank from In-Transcript Token Mutation (2026-02-13)

### Problem Statement

**Symptom:** During streamed generation, scrolling felt sluggish even after introducing preview coalescing.

### Root Cause

The active assistant row lived inside `messages` and was rewritten on every preview flush. In this UI architecture:

1. `messages[idx] = ...` fires `ChatViewModel.objectWillChange`
2. `messageList` (`ScrollView` + eager `VStack`) receives a row-content size change
3. SwiftUI re-runs stack measurement/alignment over many rows
4. Scroll interactions compete with repeated layout passes

### Fix

Use a dedicated transient container for the active stream row:

- **`LiveAssistantState`** (`ObservableObject`) holds the in-progress assistant `ChatMessage`
- Stream updates mutate `liveAssistantState.message` instead of `messages`
- `messages` is mutated once at terminal state (`done` / cancelled partial commit / failed partial commit)
- `ContentView` renders a **`LiveAssistantInlineRow`** inline with transcript rows (still backed by transient state, not persisted history)

This preserves per-token rendering while keeping transcript history stable during generation.

### Observed Regression During Iteration

An intermediate revision rendered the live preview as a visually separate bottom
band (outside the transcript container, with its own divider). In practice, that
reintroduced "wipe-like" behavior during streaming.

### Hypothesis (Why Inline Fixed It)

Likely mechanism:

1. Stream tokens changed the out-of-band preview container height/content
2. Parent layout had to reconcile two sibling regions (`messageList` + preview band)
3. ScrollView + transcript stack reflow/compositing expanded beyond the active row
4. Under frequent updates, this manifested as transient full-area redraw/wipe artifacts

By rendering the live preview **inline with transcript rows**, updates stay in one
layout domain (single transcript stack), which reduces cross-container reflow and
eliminates the observed wipe in manual verification.

### Important Constraint

Did **not** switch back to `LazyVStack`. Earlier macOS regressions documented below still apply for programmatic scrolling + lazy recycling.

---

## Bug: Thinking Disclosure Expand/Collapse — Stacking Artifacts During Layout Reflow (2026-02-13)

### Problem Statement

**Symptoms:**
- When expanding or collapsing the "Thought for X.Xs" disclosure in assistant messages, the entire message list vertically shifted (animated height change).
- During the transition, text appeared to pass behind the disclosure region, creating a transient "overlay" or dimmed layering effect.
- The effect resembled z-index misordering or clipping during animated layout recalculation.
- Messages without the thinking disclosure did not exhibit the issue.

**Impact:**
- Visually jarring expand/collapse animation
- Content appeared masked or incorrectly composited during the transition

### Root Cause Analysis

**Layout reflow + animated height change inside a ScrollView caused stacking/clipping artifacts.**

When the disclosure's conditional content appeared or disappeared:
1. The disclosure height changed (animated via `.animation(_:value:)`)
2. The enclosing message cell resized
3. The ScrollView content reflowed
4. Sibling message rows shifted position
5. During the animated layout, subviews were composited in an unexpected order — content appeared to slide "behind" the disclosure region rather than cleanly above/below it

**Contributing factors:**
- `.animation(_:value:)` on a parent VStack initially propagated layout animation broadly, causing prior messages to dim (fixed earlier by scoping to a `Group` around the conditional content).
- The `.transition(.opacity.combined(with: .move(edge: .top)))` animated position during insertion/removal, which could interact badly with the concurrent layout reflow.
- The disclosure content had no compositing isolation, so its subviews (accent line, text) could be drawn in wrong order relative to adjacent scrolling content during the animation.

### Solution

1. **`.compositingGroup()`** — Flattens the disclosure into a single composited layer. Prevents z-order and blending artifacts during layout animation by ensuring the disclosure renders as one unit.

2. **Simplified transition** — Changed from `.transition(.opacity.combined(with: .move(edge: .top)))` to `.transition(.opacity)`. The `.move` transition animated position during insertion/removal, which interacted poorly with the ScrollView layout reflow. A simple opacity fade avoids that interaction.

3. **Scoped animation** — `.animation(.easeInOut(duration: 0.2), value: isExpanded)` remains on the `Group` wrapping only the conditional content, not the outer VStack, to avoid propagating layout animation to sibling messages.

### Files Changed

- **`UI/ContentView.swift`** — `ThinkingDisclosure`: added `.compositingGroup()`, simplified to `.transition(.opacity)`, kept animation scoped to conditional `Group`.

---

## Bug: LaTeX Content Causes Full Interface Re-render / Content Wipe (2026-02-11)

### Problem Statement

**Symptoms:**
- Any chat message containing LaTeX (e.g., `$$\lim_{x \to \infty} \frac{1}{x^2} = 0$$`) caused the entire chat interface to refresh and "wipe" content on:
  - Sending/receiving a message
  - Model response streaming
  - Scrolling
- Conversations without LaTeX rendered smoothly without any refresh artifacts
- The bug was **specific to LaTeX** content, not general markdown or code blocks

**Impact:**
- Unusable for math/science/CS discussions
- Visual flickering on every message update
- Scroll position lost frequently

### Root Cause Analysis

#### Initial Hypothesis (Incorrect)

Initially suspected:
- SwiftUI `LazyVStack` recycling issues with mixed content heights
- MarkdownUI theme closure recreation causing re-layout
- Scroll position tracking bug in `ContentView.onScrollGeometryChange`

#### Actual Root Cause

**`NSViewRepresentable` inside MarkdownUI's `.codeBlock` theme closure creates a layout feedback loop:**

```
1. @Published change (new message, streaming update)
      ↓
2. MessageRow body re-evaluation
      ↓
3. MarkdownUI .codeBlock closure executes (new Theme on every body call)
      ↓
4. MathLabelRepresentable (NSViewRepresentable) created/updated
      ↓
5. MTMathUILabel property set → invalidateIntrinsicContentSize()
      ↓
6. NSView intrinsic size changes → SwiftUI detects content size change
      ↓
7. ScrollView.contentSize changes → triggers onScrollGeometryChange
      ↓
8. Auto-scroll fires → more body re-evaluations
      ↓
9. Loop back to step 3 (infinite layout churn)
      ↓
10. Visible result: interface "wipes" as views are destroyed/recreated
```

**Key insight:** The `NSViewRepresentable` lifecycle (makeNSView/updateNSView) cannot be prevented by `EquatableView` because:
- `@Environment(\.theme)` changes bypass `EquatableView` checks
- MarkdownUI theme closures are recreated on every body evaluation with new closure identities
- `MTMathUILabel` internally calls `setNeedsLayout`/`invalidateIntrinsicContentSize` on every property set

### Diagnostic Process

#### Step 1: Isolate to LaTeX

Tested identical conversations with/without LaTeX:
```swift
// Plain text conversation — NO wipe
"What is the capital of France?"

// LaTeX conversation — WIPE occurs
"What is $$\\lim_{x \\to \\infty} \\frac{1}{x}$$?"
```

**Finding:** Bug only manifests when message content contains LaTeX delimiters (`$$`, `\[`, `$...$`).

#### Step 2: Verify NSViewRepresentable is the trigger

Modified `MathBlockView` to use plain `Text` instead of `MathLabelRepresentable`:
```swift
// Workaround test — NO wipe, but no math rendering either
Text(equation)
    .font(.system(size: fontSize, design: .monospaced))
```

**Finding:** Removing `NSViewRepresentable` eliminates the wipe, confirming it's the root cause.

#### Step 3: Test EquatableView approach (Insufficient)

Wrapped `MessageContentView` in `EquatableView`:
```swift
EquatableView(content: MessageContentView(...))
```

Added `Equatable` conformance to skip body re-evaluation:
```swift
struct MessageContentView: View, Equatable {
    static func == (lhs: Self, rhs: Self) -> Bool { ... }
}
```

**Result:** Wipe still occurred. Environment changes (theme) bypass `EquatableView`.

#### Step 4: Root Cause Confirmation

Logged `MTMathUILabel` lifecycle:
```swift
func makeNSView(context: Context) -> MTMathUILabel {
    print("makeNSView called — creating new MTMathUILabel")
    // ...
}

func updateNSView(_ label: MTMathUILabel, context: Context) {
    print("updateNSView called — setting latex=\(equation)")
    // Each property set triggers invalidateIntrinsicContentSize()
}
```

**Finding:** 
- `makeNSView` called once per visible LaTeX block on initial load
- `updateNSView` called **repeatedly** on every `@Published` change (even unrelated messages)
- Each `updateNSView` call chain triggers 5+ `invalidateIntrinsicContentSize()` calls
- These propagate up through `NSView` → `NSViewRepresentable` → `SwiftUI` → `ScrollView.contentSize`

### Solution: Pre-rendered Image Approach

**Strategy:** Eliminate `NSViewRepresentable` entirely. Render LaTeX equations off-screen to bitmaps once, cache them, and display as static `Image` views.

#### Implementation

**MathRenderCache** (new):
```swift
private final class MathRenderCache: @unchecked Sendable {
    private let cache = NSCache<NSString, MathRenderEntry>()
    
    @MainActor
    func render(equation: String, fontSize: CGFloat, textColor: NSColor) -> MathRenderEntry? {
        // 1. Check cache first
        // 2. Configure MTMathUILabel off-screen
        // 3. Measure using fittingSize (NOT intrinsicContentSize — macOS quirk)
        // 4. Create NSBitmapImageRep at Retina resolution
        // 5. Render via label.draw() into graphics context
        // 6. Cache and return MathRenderEntry (NSImage + height)
    }
}
```

**MathBlockView** (updated):
```swift
struct MathBlockView: View {
    var body: some View {
        if let entry = Self.renderCache.render(...) {
            Image(nsImage: entry.image)  // Pure SwiftUI, no AppKit views
                .frame(height: entry.pointHeight)
        } else {
            Text(equation)  // Fallback for unparseable LaTeX
        }
    }
}
```

### Key macOS-Specific Findings

#### 1. MTMathUILabel Uses `fittingSize`, Not `intrinsicContentSize`

On macOS, `MTMathUILabel` overrides:
```swift
override public var fittingSize: CGSize { _sizeThatFits(CGSizeZero) }
```

But **NOT** `intrinsicContentSize` (that's in the iOS `#else` block). 

**Bug:** Using `label.intrinsicContentSize` returns `(-1, -1)` (NSView default), causing 1px-wide bitmaps.

**Fix:** Use `label.fittingSize` for measurement.

#### 2. NSGraphicsContext Handles Scale Automatically

Setting `bitmapRep.size = pointSize` tells `NSGraphicsContext(bitmapImageRep:)` to handle point-to-pixel mapping:
```swift
let bitmapRep = NSBitmapImageRep(
    pixelsWide: Int(pointWidth * scale),  // Retina pixels
    pixelsHigh: Int(pointHeight * scale),
    ...
)
bitmapRep.size = pointSize  // Points for coordinate system

// Context automatically maps point coordinates → pixels
// NO explicit scaleBy(x: scale, y: scale) needed
```

**Bug:** Adding `scaleBy` causes double-scaling, making glyphs oversized and overlapping.

#### 3. Off-Screen NSView Rendering Requires Manual Layout

Unlike iOS `UIView.layoutIfNeeded()`, macOS `NSView` needs explicit layout call:
```swift
label.frame = NSRect(origin: .zero, size: pointSize)
label.layout()  // Triggers _layoutSubviews() which creates the display list
// NOW safe to call label.draw()
```

Without `layout()`, the `displayList` is nil and `draw()` produces blank output.

### Verification

**Test Cases:**
- ✅ Multi-turn chat with heavy LaTeX (limits, integrals, fractions)
- ✅ Streaming response with inline math updates
- ✅ Scroll through history with mixed plain/LaTeX messages
- ✅ Send new message while viewing history (no forced scroll jump)
- ✅ Switch conversations (no content wipe)
- ✅ Light/dark theme toggle (math renders with correct text color)

**Performance:**
- First render: ~2-5ms per equation (parsing + typesetting + bitmap creation)
- Cached render: ~0.01ms (NSCache hit)
- Memory: 200 cached equations × ~50KB average = ~10MB max

### Files Modified

1. **`UI/MessageContentView.swift`**
   - Replaced `MathLabelRepresentable` (NSViewRepresentable) with `MathRenderCache`
   - Replaced `MathHeightCache` with `MathRenderEntry` (stores NSImage)
   - `MathBlockView` now uses `Image(nsImage:)` instead of `NSViewRepresentable`
   - Added `Equatable` conformance to `MessageContentView`

2. **`UI/ContentView.swift`**
   - Wrapped `MessageContentView` in `EquatableView(content:)` in both `userRow` and `assistantRow`

3. **`Storage/Models.swift`**
   - Added `Equatable` to `MessageRole` enum

### Lessons Learned

#### 1. NSViewRepresentable and Dynamic Layout Don't Mix

`NSViewRepresentable` is safe for:
- Static, fixed-size content (video players, maps)
- Content that doesn't affect parent layout dimensions
- Scenes where view recreation is acceptable

`NSViewRepresentable` is problematic for:
- Content with dynamic intrinsic sizes
- Scroll views with auto-scroll behavior
- High-frequency update scenarios (streaming text)

#### 2. Off-Screen Rendering is a Powerful Escape Hatch

For AppKit views that need to participate in SwiftUI layouts:
1. Configure view off-screen
2. Force layout
3. Render to bitmap context
4. Display bitmap via SwiftUI `Image`

This eliminates the AppKit-SwiftUI layout boundary entirely.

#### 3. macOS vs iOS Differences in SwiftMath

SwiftMath's `MTMathUILabel` has platform-specific implementations:
- iOS: Uses `intrinsicContentSize`, `layoutSubviews`
- macOS: Uses `fittingSize`, `layout()`

Cross-platform code must test both paths or check `os(macOS)`/`os(iOS)`.

### Recommendations

#### For SwiftUI + MarkdownUI Projects

If you need custom content in MarkdownUI code blocks:

**Option 1: Pre-rendered images (this fix)**
- Best for: Complex AppKit views with dynamic sizes
- Trade-off: Slightly higher initial render cost, but smooth updates

**Option 2: Pure SwiftUI views**
- Best for: Simple content (text, shapes, basic charts)
- Trade-off: Limited to SwiftUI capabilities

**Avoid:** `NSViewRepresentable` inside markdown theme closures unless content is truly static.

#### For SwiftMath Users

Document the macOS/iOS differences:
```swift
#if os(macOS)
let size = label.fittingSize  // macOS override
#else
let size = label.intrinsicContentSize  // iOS override
#endif
```

### Conclusion

The LaTeX wipe bug was caused by `NSViewRepresentable` creating a layout feedback loop with `ScrollView` auto-scroll behavior. The pre-rendered image approach eliminates the AppKit-SwiftUI boundary for math content, resulting in smooth, stable chat UI regardless of LaTeX presence.

**Key takeaway:** When `NSViewRepresentable` causes layout churn, consider off-screen rendering to bitmap as an alternative. It trades a small upfront cost for consistent, jank-free rendering.

---

## Follow-up: Content Wipe Persisted After LaTeX Fix (2026-02-11)

### Problem

After implementing the pre-rendered bitmap approach above, the chat content wipe **continued to occur** — even for conversations without LaTeX. Messages would vanish from the chat area entirely after sending a message or receiving a response, leaving a blank pane.

### Root Causes (Two Additional)

#### 1. `ThemeColors` Not `Equatable` — Environment Cascade

`ThemeColors` (the custom environment value injected via `@Environment(\.theme)`) did not conform to `Equatable`. Without `Equatable`, SwiftUI cannot determine whether the environment value has actually changed — it conservatively treats every injection as a change.

**Chain of failure:**
1. Any `@Published` change on `ChatViewModel` (message append, `isGenerating`, `inputText`) triggers `ContentView.body` re-evaluation
2. `.environment(\.theme, currentTheme)` injects a new `ThemeColors` struct instance
3. SwiftUI cannot compare old vs new (no `Equatable`) → propagates as a "change" to all children
4. Every `MessageRow` and `MessageContentView` reading `@Environment(\.theme)` gets body re-evaluated
5. Each re-evaluation creates a new `chatMarkdownTheme` with new closure identities
6. MarkdownUI re-renders all content from scratch
7. Mass layout recalculation → visible content wipe

**Fix:** Added `name: String` discriminator to `ThemeColors` and `Equatable` conformance comparing only `name`. Since the theme is always `.dark` or `.light`, this lets SwiftUI detect the value is unchanged and skip child invalidation entirely.

```swift
struct ThemeColors: Equatable {
    let name: String
    // ... color properties ...

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.name == rhs.name
    }

    static let dark = ThemeColors(name: "dark", ...)
    static let light = ThemeColors(name: "light", ...)
}
```

**Note:** `EquatableView` wrapping `MessageContentView` was already in place but insufficient — `@Environment` changes bypass `EquatableView` checks. The fix must prevent the environment from "changing" in the first place.

#### 2. `LazyVStack` + Content-Height Scroll Handler — Feedback Loop

The message list used `LazyVStack` with an `onScrollGeometryChange` handler that monitored content height and called `scrollTo("bottom")` when content grew by >40pt.

**Chain of failure:**
1. Any state change causes `ContentView.body` re-evaluation
2. Message rows re-render (due to cause #1 above, or legitimate content changes)
3. Content height changes by >40pt → `onScrollGeometryChange` fires
4. `DispatchQueue.main.async { proxy.scrollTo("bottom") }` executes
5. Programmatic scroll causes `LazyVStack` to deallocate off-screen views
6. Deallocated views change content height → handler fires again
7. On macOS, `LazyVStack` can fail to re-allocate views after rapid scroll changes (known SwiftUI bug)
8. All views permanently deallocated → blank chat area

**Fix (two parts):**

**a) `LazyVStack` → `VStack`:** With pagination capped at 50 messages per page, `VStack` has negligible performance difference and never deallocates its children. Eliminates the macOS lazy view recycling bug entirely.

**b) Content-height scroll handler → `onChange(of: messages.count)`:** Replaced the reactive content-size monitoring with event-driven scrolling that fires only on actual message events:

```swift
// Before (feedback loop):
.onScrollGeometryChange(for: CGFloat.self, of: { geo in
    geo.contentSize.height
}) { oldHeight, newHeight in
    if (newHeight - oldHeight) > 40 && isNearBottom {
        DispatchQueue.main.async {
            proxy.scrollTo("bottom", anchor: .bottom)
        }
    }
}

// After (event-driven):
.onChange(of: viewModel.messages.count) { oldCount, _ in
    if isNearBottom || oldCount == 0 {
        proxy.scrollTo("bottom", anchor: .bottom)
    }
}
.onChange(of: viewModel.isGenerating) { _, generating in
    if generating && isNearBottom {
        proxy.scrollTo("typing", anchor: .bottom)
    }
}
```

- `onChange(of: messages.count)` scrolls when messages are added or first loaded (`oldCount == 0`)
- `onChange(of: isGenerating)` scrolls to typing indicator when generation starts
- Both respect `isNearBottom` so reading history is not disrupted
- `.defaultScrollAnchor(.bottom)` handles initial positioning

### Files Changed

1. **`UI/ContentView.swift`**
   - `ThemeColors`: Added `name: String`, `Equatable` conformance, updated `.dark`/`.light`
   - `LazyVStack(spacing: 0)` → `VStack(spacing: 0)`
   - Removed `onScrollGeometryChange` content-height handler and `.onAppear` scroll
   - Added `onChange(of: messages.count)` and `onChange(of: isGenerating)` scroll triggers

### Lessons Learned

#### 1. Custom Environment Values Must Be `Equatable`

Any struct injected via `@Environment` should conform to `Equatable`. Without it, SwiftUI treats every body re-evaluation as an environment change, invalidating all children that read it — even through `EquatableView` wrappers.

#### 2. `LazyVStack` Is Unsafe with Programmatic Scrolling on macOS

`LazyVStack` + `ScrollViewReader.scrollTo()` can permanently lose views on macOS. The lazy allocation/deallocation during programmatic scrolling has known bugs. For bounded lists (≤50 items), prefer `VStack`.

#### 3. Content-Size Scroll Handlers Create Feedback Loops

Monitoring `ScrollView` content height and scrolling in response is inherently unstable — scrolling changes content layout, which changes content height, which triggers more scrolling. Use event-driven scrolling (`onChange` on data) instead of reactive scrolling (`onScrollGeometryChange` on layout).

---

# Bug Fix: TypeError in methodResult<String>

## Summary

Fixed a critical `TypeError: bad argument type for built-in operation` that occurred when calling Python methods via `methodResult<String>` that returned dictionary objects. The fix establishes a pattern for all SwiftPython projects: **Python kernel methods must return JSON strings when Swift expects `String` types**.

## Problem Statement

### Symptom

Calling `llama_cpp.Llama.create_chat_completion` from Swift via `methodResult<String>` triggered:

```
TypeError: bad argument type for built-in operation
```

The error surfaced on the Swift main-process side during deserialization, not in the worker process.

### Initial Hypothesis (Incorrect)

Initially suspected issues with:
- llama-cpp-python API compatibility
- Argument marshaling
- Worker process state

### Actual Root Cause

**Type mismatch in the `methodResult<T>` deserialization path.**

When a Python method returns a `dict` object but Swift calls `methodResult<String>`, the runtime:
1. Worker pickles the Python dict
2. Sends pickled bytes to main process
3. Main process unpickles → Python dict object
4. Attempts `String(pythonObject:)` conversion
5. Calls `PyUnicode_AsUTF8` on a dict pointer
6. `PyUnicode_AsUTF8` validates type → fails → `PyErr_BadArgument()`
7. Error propagates as `TypeError: bad argument type for built-in operation`

**Key insight:** `methodResult<T>` requires the Python return value to be compatible with `T`'s `PythonConvertible` initializer.

## Diagnostic Process

### Step 1: Isolate the Failure Point

Added diagnostic tests to `LlamaInferenceDemoApp.swift` with `--diag` flag:

```swift
// Test A: evalResult<Int> — PASS
let r: Int = try await worker.evalResult("1+1")

// Test B: method (handle return, no pickling) — PASS
let h = try await pool.method(handle: kernel, name: "create_session", ...)

// Test C: methodResult<String> on create_session — FAIL
let r: String = try await pool.methodResult(handle: kernel, name: "create_session", ...)
```

**Finding:** `method()` succeeded but `methodResult<String>()` failed on the same Python method → issue is in the pickle→deserialize→convert path, not in method dispatch.

### Step 2: Test Return Type Variations

```swift
// Test F: evalResult pickling a dict via str() — PASS
let r: String = try await worker.evalResult("str({'a': 1, 'b': 2})")

// Test G: methodResult<Int> on scalar return — PASS
let r: Int = try await worker.methodResult(handle: listHandle, name: "__len__")

// Test H: methodResult<String> on method returning string — PASS
let obj = try await worker.eval("type('T', (), {'greet': lambda self: 'hello'})()")
let r: String = try await worker.methodResult(handle: obj, name: "greet")

// Test I: methodResult<String> on method returning dict — FAIL
let obj = try await worker.eval("type('T', (), {'info': lambda self: {'a': 1}})()")
let r: String = try await worker.methodResult(handle: obj, name: "info")
```

**Finding:** The issue is specific to `methodResult<String>` when the Python method returns a dict. Scalar returns (Int, String) work fine.

### Step 3: Verify Hypothesis

Examined `TypeConversion.swift` in SwiftPythonRuntime:

```swift
extension String: PythonConvertible {
    public init(pythonObject: PyObjectRef) {
        guard let cString = PyUnicode_AsUTF8(pythonObject) else {
            // If PyUnicode_AsUTF8 fails, it sets a Python error
            fatalError("Failed to convert Python object to String")
        }
        self = String(cString: cString)
    }
}
```

`PyUnicode_AsUTF8` expects a Unicode object. Passing a dict triggers `PyErr_BadArgument()`.

## Solution

### Pattern: JSON Bridge for Dictionary Returns

**Python side:** Return `json.dumps(result_dict)` instead of raw dicts

```python
import json

class LlamaSessionKernel:
    def create_session(self, session_id, system_prompt=None):
        result = {
            "status": "created",
            "session_id": session_id
        }
        return json.dumps(result)  # ← Return JSON string, not dict
```

**Swift side:** Parse JSON string into `[String: Any]`

```swift
let jsonString: String = try await pool.methodResult(
    handle: kernel,
    name: "create_session",
    kwargs: ["session_id": .python(sessionID)]
)

let result = try LlamaSessionKernel.parseJSON(jsonString)
let status = result["status"] as? String
```

**Helper function:**

```swift
static func parseJSON(_ jsonString: String) throws -> [String: Any] {
    guard let data = jsonString.data(using: .utf8) else {
        throw InferenceError.decodeFailed(
            sessionID: SessionID(),
            reason: "Invalid UTF-8 in JSON string"
        )
    }
    guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw InferenceError.decodeFailed(
            sessionID: SessionID(),
            reason: "JSON is not a dictionary"
        )
    }
    return dict
}
```

## Implementation

### Files Modified

1. **`LlamaSessionKernel.swift` (Python methods)**
   - `create_session` → returns `json.dumps({"status": "created", ...})`
   - `prefill` → returns `json.dumps({"prompt_tokens": N, ...})`
   - `decode` → returns `json.dumps({"text": "...", ...})`
   - `complete` → returns `json.dumps({...})`
   - `evict` → returns `json.dumps({"status": "evicted"})`
   - `session_info` → returns `json.dumps({...})`

2. **`LlamaSessionKernel.swift` (Swift wrappers)**
   - All method signatures changed from `-> [String: String]` to `-> String`
   - Added `parseJSON()` helper
   - Callers updated to parse JSON and cast values from `Any`

3. **`LlamaSessionManager.swift`**
   - Updated to use `[String: Any]` instead of `[String: String]`
   - Value extraction uses optional casting: `result["status"] as? String`

### Verification

**Regression Tests (6 added):**

```swift
func testParseJSONValidDict() throws {
    let json = #"{"status": "created", "session_id": "test-123"}"#
    let result = try LlamaSessionKernel.parseJSON(json)
    XCTAssertEqual(result["status"] as? String, "created")
}

func testParseJSONMixedValueTypes() throws {
    let json = #"{"session_id": "s1", "prompt_tokens": 42, "prefill_ms": 1.23}"#
    let result = try LlamaSessionKernel.parseJSON(json)
    XCTAssertEqual(result["prompt_tokens"] as? Int, 42)
    XCTAssertEqual(result["prefill_ms"] as! Double, 1.23, accuracy: 0.001)
}
```

**Integration Tests:**
- ✅ Single-shot: 3/3 completions
- ✅ Multi-turn: 3 turns, context preserved
- ✅ Burst: 12/12 balanced across 4 workers
- ✅ DAG batch: 12/12 success, 24 nodes
- ✅ Stress: 20/20 concurrent
- ✅ Long decode: 1484 tokens sustained

**Test suite:** 29/29 passing

## Lessons Learned

### 1. Type Safety in Cross-Language Boundaries

`methodResult<T>` is type-safe at the Swift level but requires **runtime type compatibility** with Python. The generic `T` must have a `PythonConvertible` initializer that can handle the actual Python object returned.

**Safe patterns:**
- `methodResult<Int>` ← Python returns `int`
- `methodResult<String>` ← Python returns `str`
- `methodResult<Bool>` ← Python returns `bool`

**Unsafe patterns:**
- `methodResult<String>` ← Python returns `dict` ❌
- `methodResult<[String]>` ← Python returns `tuple` ❌ (unless tuple contains strings)

### 2. JSON as a Universal Bridge

JSON provides a safe, well-defined serialization format for complex data structures:
- **Type-safe:** JSON types map cleanly to Swift types
- **Debuggable:** JSON strings are human-readable
- **Flexible:** Supports nested structures, mixed types
- **Standard:** No custom pickling logic

**Trade-off:** Slight serialization overhead vs. raw pickle, but negligible compared to inference compute time.

### 3. Diagnostic Mode is Essential

The `--diag` flag with step-by-step tests (eval vs method vs methodResult) was critical for isolating the exact failure point. Without it, we would have continued guessing at the wrong layer.

**Recommendation:** All SwiftPython projects should include a diagnostic mode that tests:
- Basic eval
- Method dispatch (handle return)
- Method result (typed return)
- Different return types (scalar, string, dict)

### 4. Reference Existing Patterns

The Smallville demo's `InferenceKernel.generate` method already used the correct pattern:

```python
def generate(self, messages, ...):
    result = self.llama.create_chat_completion(messages=messages, ...)
    return result['choices'][0]['message']['content']  # ← Returns string, not dict
```

**Lesson:** When in doubt, check existing SwiftPython demos for established patterns.

## Recommendations for SwiftPython Projects

### 1. Document `methodResult<T>` Constraints

Add to SwiftPython wiki:

> **`methodResult<T>` Type Compatibility**
>
> The Python method's return value must be compatible with `T`'s `PythonConvertible` initializer:
> - `T=String` → Python must return `str`
> - `T=Int` → Python must return `int`
> - `T=Bool` → Python must return `bool`
>
> For complex data structures (dicts, lists), return JSON strings and parse on the Swift side.

### 2. Provide JSON Helper Template

Add to SwiftPython examples:

```python
# Python side
import json

def my_method(self, arg):
    result = {"key": "value", "count": 42}
    return json.dumps(result)
```

```swift
// Swift side
let jsonString: String = try await pool.methodResult(handle: h, name: "my_method")
let dict = try JSONSerialization.jsonObject(with: jsonString.data(using: .utf8)!) as! [String: Any]
```

### 3. Add Type Mismatch Error Message

Improve error message in `TypeConversion.swift`:

```swift
extension String: PythonConvertible {
    public init(pythonObject: PyObjectRef) {
        guard let cString = PyUnicode_AsUTF8(pythonObject) else {
            let typeName = String(cString: PyType_GetName(Py_TYPE(pythonObject)))
            fatalError("Cannot convert Python \(typeName) to String. " +
                      "If the Python method returns a dict, use json.dumps() and parse on Swift side.")
        }
        self = String(cString: cString)
    }
}
```

### 4. Add Diagnostic Template

Provide a diagnostic template for new projects:

```swift
static func runDiagnostic(pool: PythonProcessPool) async throws {
    let worker = pool.worker(0)
    
    // Test 1: Basic eval
    let _: Int = try await worker.evalResult("1+1")
    
    // Test 2: Method with handle return
    let handle = try await pool.method(handle: kernel, name: "my_method")
    
    // Test 3: Method with typed result
    let _: String = try await pool.methodResult(handle: kernel, name: "my_method")
    
    // Test 4: Method returning dict (should fail if not JSON)
    let _: String = try await pool.methodResult(handle: kernel, name: "method_returning_dict")
}
```

## Related Issues

This pattern applies to any SwiftPython project using `methodResult<T>` with Python methods that return complex data structures:
- MLX model outputs (logits, hidden states)
- NetworkX graph properties
- Matplotlib figure metadata
- Any custom Python class returning dicts/lists

**General rule:** If the Python method returns anything other than a primitive (int, float, str, bool), return JSON and parse on the Swift side.

## Performance Impact

**Negligible.** JSON serialization/deserialization overhead is ~0.1-1ms per call, which is insignificant compared to:
- Inference time: 100-1000ms per request
- IPC overhead: 1-10ms per call
- Model load time: 1-10s

**Measurement:** Burst scenario with 12 requests showed no measurable difference in throughput (101.4 tok/s) compared to theoretical pickle-only approach.

- JSON path escaping: `\` → `\\`, `"` → `\"`, unicode preserved, `None` when `nil`

## SwiftUI Chat UI (Feb 2025)

**Problem:** Built a native macOS chat interface but keyboard input went to terminal when launched via `swift run`.

**Investigation:**
- `NSApp.activate(ignoringOtherApps: true)` brought window forward visually but didn't detach from TTY
- `FocusState` was set correctly but app never received key events
- Root cause: `swift run` attaches child process stdin to parent terminal

**Solution:** Run built binary directly via Finder/`open` instead of `swift run`

**Implementation notes:**
- `WindowGroup` + `NavigationSplitView` for sidebar layout
- `@FocusState` for input focus with auto-focus on model ready
- `NSOpenPanel` with `UTType(filenameExtension: "gguf")` for model selection
- Auto-discovery via `FileManager` recursive scan of HF cache structure
- SF Symbols: `hare.fill` was wrong (not a llama), switched to 🦙 emoji

**Files added:**
- `UI/LlamaChatUI.swift`, `UI/ContentView.swift`, `UI/ChatViewModel.swift`, `UI/SettingsView.swift`

**Files modified:**
- `Package.swift` — added `LlamaChatUI` executable target
- `InferenceWorkerPool.swift` — Python-side stderr suppression (already documented above)
- `LlamaSessionKernel.swift` — `SuppressStderr` class (already documented above)

## Conclusion

The `methodResult<T>` + JSON pattern is now the **recommended approach** for all SwiftPython projects that need to pass structured data from Python to Swift. This fix not only resolved the immediate TypeError but established a robust, debuggable pattern for future development.

**Key takeaway:** Type safety at language boundaries requires both compile-time and runtime compatibility. Always verify that Python return types match Swift's expectations, and use JSON as a universal bridge for complex structures.

---

## Bug: DocumentExtractor Install Failure — `PythonWorkerError error 0` (2026-02-11)

### Problem Statement

**Symptom:** `DocumentExtractor install failed (non-fatal): The operation couldn't be completed. (SwiftPythonRuntime.PythonWorkerError error 0.)` logged on every app launch. PDF attachments showed "[Attached file: ... — content could not be extracted]" instead of document text.

**Impact:** PDF extraction completely non-functional in the app, despite `markitdown` and `pdfminer.six` being installed in `.venv`.

### Root Cause Analysis

Two independent issues combined to prevent DocumentExtractor from working:

#### Issue 1: Eager `markitdown` Import at Install Time

`DocumentExtractor.__init__()` performed `from markitdown import MarkItDown` eagerly:

```python
def __init__(self):
    from markitdown import MarkItDown  # ← Fails if package not in sys.path
    self._md = MarkItDown()
```

The install method concatenated class definition + instantiation into a single `eval`:

```swift
let installCode = kernelSource + "\n_doc_extractor = DocumentExtractor()\n_doc_extractor"
return try await worker.eval(installCode)
```

If the `from markitdown import MarkItDown` line failed, the entire `eval` failed with a generic `PythonWorkerError error 0` — no indication of *what* went wrong.

#### Issue 2: `venvPath` Never Passed to Workers

`ChatViewModel.configFromUserDefaults()` built `InferenceConfig` without `venvPath`:

```swift
return InferenceConfig(
    modelPath: ...,
    // venvPath: missing — defaults to nil
)
```

`InferenceWorkerPool` checks `config.venvPath ?? ProcessInfo.processInfo.environment["VIRTUAL_ENV"]`. When launched from Finder or `swift run`, `VIRTUAL_ENV` is not set, so workers had no venv site-packages in `sys.path`. Even though `markitdown` was installed in `.venv/lib/python3.13/site-packages`, workers couldn't import it.

### Diagnostic Process

1. **App logs** showed `DocumentExtractor install failed (non-fatal)` with no Python traceback — just a generic error code.
2. **Checked `InferenceWorkerPool.swift`** — `venvPath` logic depends on `config.venvPath` or `VIRTUAL_ENV` env var. Neither was set.
3. **Checked `configFromUserDefaults()`** — confirmed `venvPath` not populated.
4. **Compared with test code** — `MemoryArchitectureE2ETests.findVenvPath()` auto-discovers `.venv` via `#filePath`. The app had no equivalent.
5. **Unit tests confirmed** — after adding `discoverVenvPath()` and injecting site-packages, all 5 `DocumentExtractorTests` passed, including PDF extraction.

### Solution

#### Fix 1: Lazy Import + `_ensure_md()` (DocumentExtractor.swift)

```python
def __init__(self):
    self._md = None  # ← No import at install time
    self._log("DocumentExtractor init (markitdown loaded lazily)")

def _ensure_md(self):
    if self._md is None:
        self._log(f"Loading markitdown... sys.path={sys.path[:5]}")
        from markitdown import MarkItDown
        self._md = MarkItDown()
        self._log("MarkItDown loaded")
    return self._md

def extract_file(self, file_path):
    md = self._ensure_md()  # ← Import deferred to first use
    result = md.convert(file_path)
```

**Benefits:** Install always succeeds. Import errors surface at extraction time with `sys.path` diagnostic. Users see a meaningful error ("extraction failed") instead of silent install failure.

#### Fix 2: Two-Step Install (DocumentExtractor.swift)

```swift
public static func install(on worker: PythonProcessPool.WorkerContext) async throws -> PyHandle {
    _ = try await worker.eval(kernelSource + "\nTrue")       // Define class
    return try await worker.eval("DocumentExtractor()")      // Instantiate
}
```

Separating class definition from instantiation isolates errors to the specific step that fails.

#### Fix 3: `discoverVenvPath()` (ChatViewModel.swift)

```swift
private static func discoverVenvPath(sourceFile: String = #filePath) -> String? {
    // 1. Check VIRTUAL_ENV env var
    // 2. Walk up from compile-time source path looking for .venv/
    // 3. Fallback: cwd-relative paths
}
```

Uses `#filePath` (resolved at compile time to the source file's absolute path) to reliably find the repo root regardless of how the app is launched (Finder, `swift run`, Xcode).

#### Fix 4: `pdfminer.six` in requirements

Added to `python-requirements.txt` — required by MarkItDown for PDF text extraction.

### Verification

- **Unit tests:** 5 new `DocumentExtractorTests` — install, text extraction, PDF extraction, error handling, venv discovery
- **App test:** 60,090 chars extracted from a real 20-page medical PDF in 647ms
- **Regression:** 162 total tests, 0 failures

### Files Changed

1. **`Sources/Python/DocumentExtractor.swift`** — lazy import, `_ensure_md()`, two-step install, `sys.path` diagnostic logging
2. **`UI/ChatViewModel.swift`** — `discoverVenvPath()`, passed to `InferenceConfig.venvPath`
3. **`python-requirements.txt`** — added `pdfminer.six`
4. **`Tests/DocumentExtractorTests.swift`** — 5 new tests (new file)

### Lessons Learned

#### 1. Python Kernel `__init__` Must Not Import Optional Dependencies

Eager imports in `__init__` cause install failures that are invisible to the caller. Always defer optional imports to first use with diagnostic logging.

#### 2. `#filePath` Is Reliable for Repo-Relative Discovery

Unlike executable path (varies with app bundle depth) or `cwd` (varies with launch method), `#filePath` is resolved at compile time and always points to the source tree. Walking up from it reliably finds the repo root.

#### 3. Two-Step Kernel Install Isolates Errors

Splitting `eval(class_def + instantiation)` into two `eval` calls makes it clear whether the class definition or the constructor failed. This is especially important when the Python source is a large multi-line string.

#### 4. Generic `PythonWorkerError` Needs Better Diagnostics

`PythonWorkerError error 0` gives no indication of the underlying Python exception. Future improvement: propagate the Python traceback string through the IPC error path.

---

## Bug: Model Hub Pagination Broken — HF API Ignores `offset` Parameter (2026-02-12)

### Problem Statement

**Symptom:** "Loading more…" spinner appeared briefly when scrolling to the bottom of the Model Hub results, then disappeared without adding any new results. Pagination was completely non-functional.

**Impact:** Users could only ever see the first 20 search results. Scrolling to the bottom triggered the spinner but no new models appeared.

### Root Cause Analysis

#### Primary Cause: HuggingFace API Does Not Support `offset`

The pagination implementation used an `offset` query parameter:

```
GET /api/models?filter=gguf&author=lmstudio-community&sort=downloads&limit=20&offset=20
```

**The HuggingFace `/api/models` endpoint silently ignores the `offset` parameter.** Verified empirically — requests with `offset=0` and `offset=3` return identical results:

```
# offset=0 → gemma-3-4b-it, gpt-oss-20b, GLM-4.7-Flash
# offset=3 → gemma-3-4b-it, gpt-oss-20b, GLM-4.7-Flash  ← IDENTICAL
```

The HF API uses **cursor-based pagination** via the `Link` response header, not offset-based pagination. The `offset` parameter is not a recognized query parameter and is simply discarded.

**Chain of failure:**
1. `fetchCurated()` fetches page 1 (20 results) → `canLoadMore = true`
2. User scrolls to bottom → `loadMore()` calls API with `offset: 20`
3. API ignores `offset` → returns the same 20 results as page 1
4. Deduplication filter (`existingIds.contains`) removes all 20 → `newResults` is empty
5. `results.append(contentsOf: [])` → no change to the list
6. Spinner appears during API call, disappears via `defer` → user sees "flicker"
7. No new items → `.onAppear` won't re-fire for the same last item → pagination dead

#### Secondary Issues (also fixed)

**`isLoadingMore` not reset on Task cancellation:**
Two `guard !Task.isCancelled else { return }` statements returned without resetting `isLoadingMore = false`. If the Task was ever cancelled, `isLoadingMore` would stay `true` permanently, blocking all future `loadMore()` calls.

**Unmanaged `loadMore` Task:**
`loadMore()` was called from an anonymous `Task { }` in `.onAppear` with no stored reference. If the user started a new search while `loadMore()` was in-flight, stale results could be appended to fresh search results.

### Solution

#### Fix 1: Cursor-Based Pagination via `Link` Header

Replaced the broken `offset` parameter with proper cursor-based pagination:

**`HuggingFaceAPI.swift`:**
```swift
public struct SearchPage: Sendable {
    public let models: [HFModelSummary]
    public let nextPageURL: URL?
}

public func search(...) async throws -> SearchPage {
    // ... build URL without offset ...
    return try await fetchPage(url: url)
}

public func fetchPage(url: URL) async throws -> SearchPage {
    let (data, response) = try await session.data(from: url)
    try Self.validateResponse(response)
    let models = try decoder.decode([HFModelSummary].self, from: data)
    let nextURL = Self.parseNextPageURL(from: response)
    return SearchPage(models: models, nextPageURL: nextURL)
}

private static func parseNextPageURL(from response: URLResponse) -> URL? {
    guard let http = response as? HTTPURLResponse,
          let linkHeader = http.value(forHTTPHeaderField: "Link") else { return nil }
    // Parse: <https://...>; rel="next"
    for part in linkHeader.components(separatedBy: ",") {
        let trimmed = part.trimmingCharacters(in: .whitespaces)
        guard trimmed.contains("rel=\"next\""),
              let urlStart = trimmed.firstIndex(of: "<"),
              let urlEnd = trimmed.firstIndex(of: ">") else { continue }
        return URL(string: String(trimmed[trimmed.index(after: urlStart)..<urlEnd]))
    }
    return nil
}
```

**`ModelHubViewModel.swift`:**
- Replaced `currentOffset: Int` with `nextPageURL: URL?`
- `canLoadMore = page.nextPageURL != nil` (no more count-based heuristic)
- `loadMore()` calls `api.fetchPage(url: nextPageURL!)` instead of re-constructing a search URL

#### Fix 2: `defer { isLoadingMore = false }`

Guarantees the flag is reset regardless of exit path.

#### Fix 3: Tracked `loadMoreTask` with Cancellation

View calls `triggerLoadMore()` (idempotent, stores task reference). `performSearch()` cancels in-flight `loadMoreTask` before resetting state.

### Files Changed

1. **`Sources/Network/HuggingFaceAPI.swift`**
   - Added `SearchPage` struct (models + nextPageURL)
   - `search()` returns `SearchPage`, removed `offset` parameter
   - Added `fetchPage(url:)` for cursor-based next-page fetch
   - Added `parseNextPageURL(from:)` — parses `Link` header for `rel="next"`

2. **`UI/ModelHubViewModel.swift`**
   - Replaced `currentOffset: Int` with `nextPageURL: URL?`
   - Added `loadMoreTask` property, `triggerLoadMore()` method
   - `performSearch()`, `fetchCurated()`, `loadMore()` all use `SearchPage`
   - `canLoadMore` derived from `nextPageURL != nil`
   - `defer { isLoadingMore = false }` in `loadMore()`

3. **`UI/ModelHubView.swift`**
   - `.onAppear` calls `hubVM.triggerLoadMore()` instead of anonymous Task

### Lessons Learned

#### 1. Verify API Pagination Behavior Empirically

The `offset` parameter was assumed to work based on common REST API conventions. A single test request would have revealed it was silently ignored. Always verify pagination behavior against the actual API, especially for APIs without explicit offset documentation.

#### 2. Cursor-Based Pagination Is the Standard for HuggingFace

The HF Hub API paginates via `Link: <url>; rel="next"` response headers. The cursor URL contains an opaque token managed by the server. This is the only reliable pagination method for the `/api/models` endpoint.

#### 3. Silent Parameter Ignoring Is Dangerous

APIs that silently ignore unrecognized query parameters (rather than returning an error) create hard-to-diagnose bugs. The code appeared to work — it made requests, got responses, and processed them. The deduplication filter masked the real issue by cleanly handling the duplicate results without error.

#### 4. Async Flag Cleanup Requires `defer`

Any boolean flag set before an `await` point must use `defer` for cleanup. Early returns from `guard`, `throw`, or cancellation bypass the cleanup line at the end of the function.

---

## Bug: Download Directory Picker Flicker (2026-02-12)

### Problem Statement

**Symptom:** When opening the directory picker for "Download Directory" in Settings → Models, the Select and New Folder buttons flicker as if two windows or views are competing. The vision (VLM) file picker does not exhibit this behavior.

### Solution

Make `browseForDownloadDir()` **identical** to `browseForClipProjection()`: same NSOpenPanel flow, same `directoryURL` logic (modelsDir or hfCache), no `DispatchQueue.main.async`, no custom initial-directory helpers. Only difference: `canChooseDirectories = true` / `canChooseFiles = false` (directory picker) vs `allowedContentTypes = [gguf]` (file picker).

### Files Changed

- **`UI/SettingsView.swift`**
  - `browseForDownloadDir()` — matches `browseForClipProjection()` structure exactly

---

## Bug: Slow Typing / Beachball with Document Attachments (2026-02-12)

### Problem Statement

**Symptoms:**
- 1–2 second spinning beachball when adding an image attachment (before sending)
- UI unresponsive during `sendMessage()` while file I/O, text extraction, and VLM captioning ran
- Typing jank in conversations with prior attachments due to multi-MB `Data` blobs in the `@Published var messages` array
- `ChatInputTextView.updateNSView` scheduling unnecessary `recalculateHeight()` on every SwiftUI re-render pass

**Impact:**
- Beachball on every image attachment add
- Input field frozen for seconds during send with attachments
- Memory pressure from retaining full file contents in the SwiftUI view tree

### Root Cause Analysis

Four independent issues contributed:

#### 1. Synchronous image decode in `addAttachment(url:)` (beachball)

`addAttachment(url:)` was a synchronous `@MainActor` method that called `NSImage(contentsOf: url)` to decode the full-resolution image (e.g., 12MP HEIC = 500ms–1s+), then `lockFocus`/`draw`/`unlockFocus` to scale it. All on the main thread.

#### 2. `sendMessage()` blocked main actor during attachment processing

`convertPendingToMessageAttachments()` ran `Data(contentsOf:)` for each file, called `DocumentExtractor` (Python kernel) for PDFs, and `VLMKernel` for image captioning — all before setting `isGenerating = true` or clearing `inputText`. The UI appeared frozen until all processing completed.

#### 3. `MessageAttachment.data` retained full file blobs in memory

`MessageAttachment` stored `let data: Data` (up to 50 MB per attachment). These blobs lived in the `@Published var messages` array, meaning every `objectWillChange` notification forced SwiftUI to diff structures containing multi-MB payloads. On DB load, `loadAttachments` read the full BLOB column into memory even though the UI only displays `thumbnailData` and `filename`.

#### 4. `updateNSView` unconditionally scheduled `recalculateHeight()`

Every `@Published` change on `ChatViewModel` triggered `ContentView.body` re-evaluation → `ChatInputTextView.updateNSView` → `DispatchQueue.main.async { recalculateHeight() }`. This caused a redundant second render pass on every state change, even when text/font/maxLines hadn't changed.

### Fix

#### Fix 1: Async thumbnail via `CGImageSource` (`ChatViewModel.swift`)

- `addAttachment(url:)` appends the chip **immediately** with `thumbnailImage: nil` (shows SF Symbol placeholder)
- Thumbnail generated async via `CGImageSourceCreateThumbnailAtIndex` — thread-safe single-pass decode+scale, no `NSImage`/`lockFocus` required
- `PendingAttachment` updated in-place once thumbnail is ready; SwiftUI swaps icon → image

```swift
private nonisolated static func generateThumbnail(url: URL, maxDim: CGFloat) async -> NSImage? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    let options: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceThumbnailMaxPixelSize: Int(maxDim),
        kCGImageSourceCreateThumbnailWithTransform: true,
    ]
    guard let cgThumb = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
    return NSImage(cgImage: cgThumb, size: NSSize(width: cgThumb.width, height: cgThumb.height))
}
```

#### Fix 2: Immediate UI response in `sendMessage()` (`ChatViewModel.swift`)

- `isGenerating = true`, `inputText = ""`, `pendingAttachments = []` set **before** any async work
- User message appended immediately with lightweight placeholder attachments (no `data`, thumbnail from `PendingAttachment`)
- Heavy work (file I/O via `nonisolated static readFileData`, extraction, captioning, semantic retrieval) runs **after** the UI has updated
- Message updated in-place with fully processed attachments once ready
- `readFileData(url:)` is `nonisolated static` so `Data(contentsOf:)` doesn't hold the main actor

#### Fix 3: Lightweight in-memory attachments (`Models.swift`, `ChatPersistence.swift`, `ChatViewModel.swift`)

- `MessageAttachment.data` changed from `Data` to `Data?` (default `nil`)
- `loadAttachments` SELECT omits the `data` BLOB column — UI only needs `thumbnailData`, `filename`, `extractedText`
- `saveAttachments` skips attachments with `data == nil` (placeholder phase)
- `stripAttachmentData()` called after persistence — replaces in-memory attachments with `data: nil` copies for messages in `lastSavedMessageIDs`
- Net effect: multi-MB blobs exist only transiently during send, never retained in the view tree

#### Fix 4: Conditional `recalculateHeight()` (`ChatInputTextView.swift`)

- `updateNSView` tracks whether text, font, or maxLines actually changed via `needsHeightRecalc` flag
- `recalculateHeight()` only scheduled when flag is true
- Preserves existing `DispatchQueue.main.async` pattern (avoids re-entrant binding updates) and `abs > 0.5` guard (prevents feedback loops)

### Files Changed

1. **`UI/ChatViewModel.swift`**
   - `addAttachment(url:)` — append immediately, async thumbnail via `Task` + `CGImageSource`
   - `generateThumbnail(url:maxDim:)` — new `nonisolated static` method
   - `sendMessage()` — set `isGenerating`/clear input before async work; placeholder attachments; update message in-place
   - `mimeType(for:)` — extracted as `static` helper (was inline in `convertPendingToMessageAttachments`)
   - `convertPendingToMessageAttachments(_:)` — takes explicit `[PendingAttachment]` parameter; uses `readFileData`
   - `readFileData(url:)` — new `nonisolated static` method for off-main-actor file I/O
   - `stripAttachmentData()` — new method to nil out data blobs after persistence

2. **`Storage/Models.swift`**
   - `MessageAttachment.data` — `Data` → `Data?` (default `nil`)

3. **`Storage/ChatPersistence.swift`**
   - `saveAttachments` — skips attachments with `data == nil`
   - `loadAttachments` — omits `data` BLOB column from SELECT

4. **`ChatUIComponents/ChatInputTextView.swift`**
   - `updateNSView` — conditional `recalculateHeight()` based on `needsHeightRecalc` flag

5. **`Tests/ChatPersistenceTests.swift`**
   - `testSaveAndLoadAttachments` — expects `data == nil` after load
   - `testAttachmentWithThumbnailData` — expects `data == nil` after load

### Regression Safety

These changes do **not** touch any of the layout/scrolling/theme machinery fixed in the LaTeX content wipe bugs:
- No `NSViewRepresentable` added to the render tree
- No `LazyVStack` usage
- No content-height-based scroll handlers
- `ThemeColors: Equatable`, `MessageContentView: Equatable`, `EquatableView` wrappers all untouched
- Event-driven scrolling (`onChange(of: messages.count)`, `onChange(of: isGenerating)`) unchanged

### Lessons Learned

#### 1. `NSImage(contentsOf:)` is deceptively expensive

Full-resolution image decode (especially HEIC/HEIF) can take hundreds of milliseconds. `CGImageSourceCreateThumbnailAtIndex` with `kCGImageSourceThumbnailMaxPixelSize` performs a single-pass decode+scale that is both faster and thread-safe — no `lockFocus`/`unlockFocus` needed.

#### 2. `@MainActor` async methods still block during synchronous segments

An `async` method on `@MainActor` only yields at `await` points. Synchronous code between awaits (file I/O, image decode, data conversion) blocks the main thread just as effectively as a synchronous method. Move synchronous work to `nonisolated` helpers.

#### 3. `@Published` arrays amplify memory pressure

Every `objectWillChange` notification from any `@Published` property causes SwiftUI to evaluate views that depend on *any* property of the `ObservableObject`. Storing large `Data` blobs in a `@Published` array means those blobs participate in every diff cycle, even when unrelated properties change. Keep published collections lightweight — persist heavy data elsewhere and hold only metadata in memory.

#### 4. `NSViewRepresentable.updateNSView` is called far more often than expected

SwiftUI calls `updateNSView` for any state change that affects the parent view, not just changes to the representable's bindings. Unconditionally scheduling work (especially work that mutates bindings) in `updateNSView` creates unnecessary render cycles. Guard expensive operations behind change detection.

---

## Fix: Monolithic `@ObservedObject` — Composer Re-evaluation Scope (2026-02-12)

### Problem Statement

**Symptom:** Text field sluggishness during long conversations. Each keystroke and each streaming token update caused the entire `ContentView.body` to re-evaluate, including the heavyweight message list (`VStack` with 50+ `MessageRow` views containing Markdown/LaTeX).

**Root cause:** `ContentView` observed `ChatViewModel` via a single `@ObservedObject`. Since `ObservableObject` fires `objectWillChange` for *any* `@Published` property change, updating `inputText` (keystroke) or `messages` (streaming) both triggered a full `ContentView.body` run. The composer and message list lived in the same view tree, so both participated in every re-evaluation.

**Prior mitigations (from earlier fixes) reduced the cost but not the frequency:**
- `ThemeColors: Equatable` — prevents environment cascade
- `EquatableView` wrapping `MessageContentView` — short-circuits unchanged message bodies
- LaTeX pre-rendered to bitmap — eliminates `NSViewRepresentable` layout churn
- `VStack` (not `LazyVStack`) — prevents view deallocation bugs
- Conditional `recalculateHeight()` in `updateNSView` — avoids redundant height passes

These made each body run cheaper, but didn't prevent the run itself.

### Solution: Separate `ObservableObject` for Composer State

Extracted `inputText` and `pendingAttachments` from `ChatViewModel` into a dedicated `ComposerState: ObservableObject`. The composer view observes `ComposerState` directly; `ContentView` does not observe it.

**Observation boundaries after the fix:**

```
Keystroke → ComposerState.objectWillChange
         → ComposerView.body re-evaluates (lightweight: text field + attachment chips)
         ✗ ContentView.body does NOT re-evaluate
         ✗ Message list VStack does NOT re-evaluate

Streaming → ChatViewModel.objectWillChange (messages changed)
         → ContentView.body re-evaluates (message list updates)
         ✗ ComposerView.body does NOT re-evaluate (it observes ComposerState, not ChatViewModel)
```

#### Implementation

**`ComposerState`** (new class in `ChatViewModel.swift`):
```swift
@MainActor
class ComposerState: ObservableObject {
    @Published var inputText = ""
    @Published var pendingAttachments: [PendingAttachment] = []
}
```

**`ChatViewModel`** changes:
- Removed `@Published var inputText` and `@Published var pendingAttachments`
- Added `let composerState = ComposerState()`
- `sendMessage()`, `addAttachment(url:)`, `removeAttachment(id:)` updated to read/write `composerState.inputText` and `composerState.pendingAttachments`

**`ComposerView`** (new struct in `ContentView.swift`):
```swift
struct ComposerView: View {
    @ObservedObject var composerState: ComposerState
    let isGenerating: Bool
    let isReady: Bool
    let onSend: () -> Void
    let onAddAttachment: (URL) -> Void
    let onRemoveAttachment: (UUID) -> Void
    // ... body contains the full composer bar
}
```

- Owns `@FocusState` and `@State inputContentHeight` (moved from `ContentView`)
- Uses `@Environment(\.theme)` for styling (no dependency on `ContentView`)
- `isGenerating` and `isReady` passed as plain values — changes to these trigger `ContentView.body` anyway (message list needs them), but that's infrequent (start/end of generation)
- `onSend`, `onAddAttachment`, `onRemoveAttachment` are closures wired to `ChatViewModel` methods

**`ContentView`** changes:
- Removed `composerBar`, `canSend`, `pendingAttachmentChips`, `showAttachmentMenu`, `sendAction`
- Removed `@FocusState private var isInputFocused` and `@State private var inputContentHeight`
- `chatDetail` body uses `ComposerView(composerState: viewModel.composerState, ...)`

### Why This Doesn't Regress Existing Fixes

| Existing Fix | Status |
|---|---|
| `ThemeColors: Equatable` | Untouched — still prevents environment cascade |
| `EquatableView` on `MessageContentView` | Untouched — still short-circuits unchanged messages |
| LaTeX bitmap cache (`MathRenderCache`) | Untouched — no `NSViewRepresentable` in render tree |
| `VStack` (not `LazyVStack`) | Untouched — no view deallocation risk |
| Event-driven scrolling (`onChange`) | Untouched — no content-height feedback loop |
| Conditional `recalculateHeight()` | Untouched — still guarded in `updateNSView` |
| Lightweight attachments (`data: nil`) | Untouched — blobs still stripped after persistence |

### Files Changed

1. **`UI/ChatViewModel.swift`**
   - Added `ComposerState` class (before `ChatViewModel`)
   - Removed `@Published var inputText` and `@Published var pendingAttachments`
   - Added `let composerState = ComposerState()`
   - Updated `sendMessage()`, `addAttachment(url:)`, `removeAttachment(id:)` to use `composerState`

2. **`UI/ContentView.swift`**
   - Removed `@FocusState private var isInputFocused`, `@State private var inputContentHeight`
   - Removed `composerBar`, `canSend`, `pendingAttachmentChips`, `showAttachmentMenu`, `sendAction`
   - Added `ComposerView` struct with `@ObservedObject var composerState: ComposerState`
   - `chatDetail` instantiates `ComposerView` with closures for actions

### Lessons Learned

#### 1. `ObservableObject` Has Coarse Invalidation Granularity

With `ObservableObject`, any `@Published` change fires a single `objectWillChange` publisher that invalidates all observing views — regardless of which property changed. The only way to scope invalidation is to use separate `ObservableObject` instances for independent state domains.

#### 2. View Extraction Alone Is Insufficient

Extracting a child view (e.g., `ComposerView`) but keeping `@ObservedObject var viewModel: ChatViewModel` on both parent and child does NOT help — both still subscribe to the same `objectWillChange`. The child must observe a *different* `ObservableObject` to create a true invalidation boundary.

#### 3. Closures + Values Are the Safe Bridge

Passing `isGenerating` and `isReady` as plain `let` values (not bindings or observed properties) ensures that `ComposerView` only re-evaluates when its parent explicitly passes new values — which only happens when `ContentView.body` runs for other reasons (e.g., message list update). This is the correct trade-off: generation start/stop is infrequent and already triggers a full update.

---

## Fix: Sidebar Search Field Lag — Debounced Local State (2026-02-12)

### Problem Statement

**Symptom:** Sidebar search field sluggish during typing, identical root cause to the composer keystroke bug (see "Monolithic `@ObservedObject`" fix above).

**Root cause:** `TextField("Search", text: $viewModel.searchQuery)` bound directly to a `@Published` property on `ChatViewModel`. Every keystroke fired `ChatViewModel.objectWillChange`, triggering full `ContentView.body` re-evaluation — including the heavyweight message list (`VStack` with 50+ `MessageRow` views containing Markdown/LaTeX). Additionally, `onChange(of: viewModel.searchQuery)` called `refreshConversations()` on every keystroke, issuing a synchronous FTS5 database query and setting `conversations` (another `@Published`), doubling the invalidation.

### Solution: Local `@State` + Debounced Task

Replaced the direct binding with a local `@State` that only invalidates the sidebar search field, plus a debounced `Task` that syncs to `viewModel.searchQuery` after 300ms of idle typing.

**Observation boundaries after the fix:**

```
Keystroke → @State searchText changes
         → Only searchField re-evaluates (lightweight: magnifying glass icon + text field)
         ✗ ContentView.body does NOT re-evaluate
         ✗ Message list VStack does NOT re-evaluate
         ✗ No database query fires

After 300ms idle → viewModel.searchQuery = searchText
                 → ChatViewModel.objectWillChange fires (once)
                 → refreshConversations() runs FTS5 query
                 → conversations updates → sidebar list re-renders
```

#### Implementation

**Before:**
```swift
TextField("Search", text: $viewModel.searchQuery)
// ...
.onChange(of: viewModel.searchQuery) {
    Task { await viewModel.refreshConversations() }
}
```

**After:**
```swift
@State private var searchText = ""
@State private var searchDebounceTask: Task<Void, Never>?

TextField("Search", text: $searchText)
// ...
.onChange(of: searchText) { _, newValue in
    searchDebounceTask?.cancel()
    searchDebounceTask = Task {
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        viewModel.searchQuery = newValue
        await viewModel.refreshConversations()
    }
}
```

### Why This Doesn't Regress Existing Fixes

- **ComposerState isolation preserved** — composer still observes its own `ComposerState`, unaffected
- **ThemeColors: Equatable** — unchanged, still prevents environment cascade
- **EquatableView on MessageContentView** — unchanged, still gates Markdown re-rendering
- **No new `@Published` properties** — `searchText` is `@State` (local to `ContentView`), adds zero `objectWillChange` pressure
- **No new `NSViewRepresentable`** — pure SwiftUI `TextField`

### Files Changed

1. **`UI/ContentView.swift`**
   - Added `@State private var searchText` and `@State private var searchDebounceTask`
   - Changed `TextField` binding from `$viewModel.searchQuery` to `$searchText`
   - Changed `onChange(of: viewModel.searchQuery)` to `onChange(of: searchText)` with 300ms debounced `Task`
   - Updated empty-state text to read `searchText` instead of `viewModel.searchQuery`

### Lessons Learned

Same principle as the `ComposerState` fix: any text field that binds directly to a `@Published` property on a monolithic `ObservableObject` will cause full-tree invalidation on every keystroke. The fix is always the same — isolate the keystroke state from `objectWillChange` using either a separate `ObservableObject` or a local `@State` with debounced sync.

---

## Fix: SwiftPythonWorker @main Build Failure (2026-02-12)

### Problem Statement

**Symptom:** `'@main' attribute cannot be used in a module that contains top-level code` when building LlamaChatUI (or any target depending on SwiftPythonWorker). Build failed during `SwiftEmitModule` / `SwiftCompile` for SwiftPythonWorker.

### Root Cause

When the LlamaInferenceDemo workspace builds the SwiftPython package, Swift treats `main.swift` as executable top-level code by default. The `@main` attribute requires the module to be compiled in library mode (declarations only, no top-level statements). The two modes are incompatible.

### Solution

Add `-parse-as-library` to the SwiftPythonWorker target in the root `Package.swift`:

```swift
.executableTarget(
    name: "SwiftPythonWorker",
    dependencies: ["SwiftPythonRuntime"],
    swiftSettings: [
        .unsafeFlags(["-parse-as-library"])
    ],
    linkerSettings: pythonLinkerSettings
),
```

This matches the pattern already used by the Examples target and LlamaInferenceCore.

### Files Changed

- **`Package.swift`** (SwiftPython repo root) — added `swiftSettings` to SwiftPythonWorker target

---

## Bug: Thinking Tokens Dumped Into Answer Text / Not Streaming to Disclosure Box (2026-02-24)

### Problem Statement

**Symptoms (two related issues):**

1. **Orphan `</think>` tag** — Model output that started with raw thinking tokens (no opening `<think>` tag) followed by `</think>` had the entire response — thinking + answer — rendered in the visible answer area. The collapsible "Thought for X" disclosure box did not appear.

2. **Thinking not live-streaming** — Even after fix #1, thinking tokens only appeared in the disclosure box *after* `</think>` arrived. During generation, thinking tokens streamed into the visible answer area; the disclosure box appeared only when the model switched to regular output.

**Model:** Qwen3 (and similar reasoning models). These models emit thinking tokens immediately, without an opening `<think>` tag, closing only with `</think>`.

### Root Cause

#### Issue 1 — Orphan `</think>` Tag

Both the Python `_extract_think_blocks` method and the Swift `ThinkingTextParser.split` only matched thinking content that started with a literal `<think>` tag. The regex patterns were:

```python
re.finditer(r'<think>(.*?)</think>', text, re.DOTALL)  # Python
```
```swift
pattern: "<think>(.*?)</think>"  // Swift
```

Qwen3's output format: `[thinking tokens]</think>[answer tokens]` — no opening tag. Neither parser fired; the entire raw string was returned as answer text.

#### Issue 2 — Streaming Delta Routing

The streaming delta flush path called `ThinkingTextParser.split(rawText: rawText)` on the accumulated buffer. While the model was still generating thinking tokens, `rawText` contained no `<think>` or `</think>` tags at all. The parser saw no tags → returned everything as `text` → rendered in the answer area.

Once fix #1 landed, `</think>` arriving in the buffer would finally trigger the split — but by then the entire thinking block had already been rendered and re-routed, causing a jarring visual swap rather than live streaming into the disclosure box.

### Fix

#### Fix 1 — Orphan `</think>` Tag (`LlamaSessionKernel.swift` + `StreamingThinkingSupport.swift`)

**Python:** After existing regex matching finds nothing, detect an orphan `</think>` and treat everything before it as thinking:

```python
if not parts and '</think>' in cleaned:
    idx = cleaned.index('</think>')
    before = cleaned[:idx].strip()
    after = cleaned[idx + len('</think>'):].strip()
    if before:
        parts.append(before)
    cleaned = after
```

**Swift:** Same logic after unclosed-tag detection:

```swift
if thinking == nil, let closeRange = answerText.range(of: "</think>") {
    let before = String(answerText[..<closeRange.lowerBound])
        .trimmingCharacters(in: .whitespacesAndNewlines)
    let after = String(answerText[closeRange.upperBound...])
        .trimmingCharacters(in: .whitespacesAndNewlines)
    if !before.isEmpty { thinking = normalized(before) }
    answerText = after
}
```

#### Fix 2 — Live Streaming Routing (`StreamingThinkingSupport.swift` + `ChatViewModel.swift`)

Added a `streamingInProgress: Bool = false` parameter to `ThinkingTextParser.split`. When `true` and `rawText` contains no think tags at all, the entire accumulated text is returned as in-progress thinking with an empty answer:

```swift
if thinking == nil && streamingInProgress
    && !answerText.contains("<think>")
    && !answerText.contains("</think>") {
    let candidate = answerText.trimmingCharacters(in: .whitespacesAndNewlines)
    return ThinkingTextSplit(text: "", thinking: candidate.isEmpty ? nil : candidate)
}
```

In `ChatViewModel`, both delta flush paths now pass `streamingInProgress: true`. The `done` event path is unchanged — Python already provides the correctly-split `chunk.thinking`.

### Streaming Behaviour After Fix

| Phase | Thinking box | Answer area |
|---|---|---|
| Pre-`</think>` (delta) | Tokens stream live | Empty |
| `</think>` arrives (delta) | Thinking finalised | Answer begins |
| `done` event | Python-parsed `chunk.thinking` used | Final clean text |

### Files Changed

- **`Sources/Python/LlamaSessionKernel.swift`** — `_extract_think_blocks`: orphan `</think>` detection
- **`Sources/Engine/StreamingThinkingSupport.swift`** — `ThinkingTextParser.split`: orphan `</think>` detection + `streamingInProgress` parameter
- **`UI/ChatViewModel.swift`** — delta flush paths pass `streamingInProgress: true`
- **`Tests/LlamaInferenceDemoTests.swift`** — 8 new `ThinkingTextParserTests` (11 total, 0 failures)
