import XCTest
import ChatUIComponents

final class LatexPreprocessorTests: XCTestCase {
    // MARK: - Block Math

    func testBlockMath_doubleDollar() {
        let input = "Before $$E=mc^2$$ after"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("```math\nE=mc^2\n```"), "Expected fenced math block, got: \(result)")
        XCTAssertTrue(result.contains("Before"))
        XCTAssertTrue(result.contains("after"))
    }

    func testBlockMath_displayBrackets() {
        let input = "Text \\[x^2\\] more"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("```math\nx^2\n```"), "Expected fenced math block, got: \(result)")
        XCTAssertTrue(result.contains("Text"))
        XCTAssertTrue(result.contains("more"))
    }

    func testBlockMath_multiLine() {
        let input = "$$\na + b\n= c\n$$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("```math\n"), "Expected fenced math block, got: \(result)")
        XCTAssertTrue(result.contains("a + b"))
        XCTAssertTrue(result.contains("= c"))
    }

    func testBlockMath_boxedUnwrapped() {
        // SwiftUIMath/iosMath do not implement \boxed; unwrap to inner content for rendering
        let input = "$$\\boxed{x^2 + y^2 = z^2}$$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("```math\nx^2 + y^2 = z^2\n```"),
            "Expected \\boxed{...} unwrapped to inner content, got: \(result)"
        )
        XCTAssertFalse(result.contains("\\boxed"), "boxed should be stripped, got: \(result)")
    }

    func testBlockMath_boxedWithAlignedUnwrapped() {
        let input = """
        $$\\boxed{
        \\begin{aligned}
        \\mathbf{v}(t) &= \\int_0^t a(t') \\, dt' + \\mathbf{v}_0
        \\end{aligned}}$$
        """
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("\\begin{aligned}"),
            "Aligned content should be preserved after unwrap, got: \(result)"
        )
        XCTAssertFalse(result.contains("\\boxed"), "boxed should be stripped, got: \(result)")
    }

    // MARK: - Inline Math

    func testInlineMath_parens() {
        let input = "Value \\(y\\) end"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("$y$"), "Expected \\(...\\) normalized to $...$, got: \(result)")
    }

    func testInlineMath_singleDollar() {
        let input = "Say $a+b$ then"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("$a+b$"), "Expected inline math preserved, got: \(result)")
    }

    func testInlineMath_preservesTeXSubscriptSyntax() {
        let input = "Tensor $g_{\\mu\\nu}$ entry"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("$g_{\\mu\\nu}$"),
            "Expected TeX subscript underscore to be preserved in inline math, got: \(result)"
        )
    }

    func testInlineMath_displaystyleStripped() {
        // \displaystyle unsupported in many inline renderers; strip to avoid raw-LaTeX fallback
        let input = "Force $\\displaystyle \\mathbf{F} = -m\\nabla\\Phi$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertFalse(
            result.contains("\\displaystyle"),
            "\\displaystyle should be stripped from inline math, got: \(result)"
        )
        XCTAssertTrue(
            result.contains("$\\mathbf{F} = -m\\nabla\\Phi$") || result.contains("$ \\mathbf{F} = -m\\nabla\\Phi$"),
            "Inner content should be preserved, got: \(result)"
        )
    }

    func testInlineMath_hatBoldRewritten() {
        // SwiftUIMath fails on \hat{\mathbf{r}} and dumps remainder as raw text (\mathbf{r^}); rewrite to \mathbf{\hat{r}}
        let input = "$-\\frac{GM}{r^2} \\hat{\\mathbf{r}}$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("\\mathbf{\\hat{r}}"),
            "\\hat{\\mathbf{r}} should be rewritten to \\mathbf{\\hat{r}}, got: \(result)"
        )
        XCTAssertFalse(
            result.contains("\\hat{\\mathbf{r}}"),
            "Original \\hat{\\mathbf{r}} should be replaced, got: \(result)"
        )
    }

    func testInlineMath_vecBoldRewritten() {
        // Same parse bailout for \vec{\mathbf{g}}; rewrite to \mathbf{\vec{g}}
        let input = "$\\vec{\\mathbf{g}} = \\vec{F}_{\\text{grav}}/m$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("\\mathbf{\\vec{g}}"),
            "\\vec{\\mathbf{g}} should be rewritten to \\mathbf{\\vec{g}}, got: \(result)"
        )
        XCTAssertFalse(
            result.contains("\\vec{\\mathbf{g}}"),
            "Original \\vec{\\mathbf{g}} should be replaced, got: \(result)"
        )
    }

    func testHealTruncatedMath_appendsClosingDollar() {
        // Model cut mid-command: $g_{tt} = -(1 + 2\ with no closing $
        let input = "Here, $g_{tt} = -(1 + 2\\"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("$g_{tt} = -(1 + 2\\$"),
            "Should append $ to close truncated span, got: \(result)"
        )
        // Verify it then gets processed as inline math (normalized)
        XCTAssertTrue(result.contains("g_{tt}"), "Math content should be preserved")
    }

    func testHealTruncatedMath_doesNotHealWhenEvenDollars() {
        // Both spans closed ($a$ and $b$), ends with \ (e.g. line break); must not append
        let input = "First $a$ then $b$. End\\"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertFalse(
            result.hasSuffix("\\$"),
            "Should NOT append when all spans closed (even $), got: \(result)"
        )
    }

    func testHealTruncatedMath_doesNotHealWhenEndsWithDoubleBackslash() {
        // \\ at end can be valid LaTeX line break
        let input = "Equation $x = 5\\\\"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertFalse(
            result.hasSuffix("\\\\$"),
            "Should NOT append when ends with \\\\, got: \(result)"
        )
    }

    func testHealTruncatedMath_doesNotHealWhenNoTrailingBackslash() {
        let input = "Incomplete $x = "
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertEqual(result, "Incomplete $x = ", "Should not append without truncation signature")
    }

    func testHealTruncatedMath_midDocumentLine() {
        let input = """
        Intro paragraph.
            $\\Phi = g_0$ is the Newtonian potential. Here, $g_{tt} = -(1 + 2\\
        2. Next section
        """
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertFalse(
            result.contains("\n    $\\Phi"),
            "Accidental 4-space indent should be reduced to avoid markdown code block: \(result)"
        )
        XCTAssertTrue(
            result.contains("$g_{tt} = -(1 + 2\\$"),
            "Truncated inline math should be closed on its line, got: \(result)"
        )
    }

    func testHealTruncatedMath_doesNotMutateFencedCodeBlockLines() {
        let input = """
        ```markdown
            $\\Phi = g_0$ and $g_{tt} = -(1 + 2\\
        ```
        """
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("    $\\Phi = g_0$ and $g_{tt} = -(1 + 2\\"),
            "Indented fenced-code content should not be deindented/healed, got: \(result)"
        )
        XCTAssertFalse(
            result.contains("2\\$"),
            "Should not append closing $ inside fenced code, got: \(result)"
        )
    }

    func testInlineMath_boxDAlambertianReplaced() {
        // SwiftUIMath doesn't implement \Box; replace with □ (U+25A1)
        let input = "(e.g., $\\Box g_{\\mu\\nu} = 0$)"
        let result = LatexPreprocessor.preprocess(input)
        let boxChar = "\u{25A1}"
        XCTAssertTrue(
            result.contains("$\(boxChar) g_{\\mu\\nu} = 0$") || result.contains("$ \(boxChar) g_{\\mu\\nu} = 0$"),
            "\\Box should be replaced with □, got: \(result)"
        )
        XCTAssertFalse(
            result.contains("\\Box"),
            "\\Box should be replaced, got: \(result)"
        )
        XCTAssertFalse(result.contains("\\boxed"), "\\boxed should not be altered")
    }

    func testInlineMath_normalizesMarkdownEscapedSubscripts() {
        let input = "Weak field $\\Phi = g\\_0$ and metric $g\\_{tt}$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("$\\Phi = g_0$"),
            "Expected markdown-escaped underscore to normalize to TeX subscript, got: \(result)"
        )
        XCTAssertTrue(
            result.contains("$g_{tt}$"),
            "Expected braced markdown-escaped underscore to normalize to TeX subscript, got: \(result)"
        )
    }

    // MARK: - Currency Preservation

    func testCurrencyNotConverted() {
        let input = "Cost $10 or $1.50"
        let result = LatexPreprocessor.preprocess(input)
        // Currency should be escaped so markdown math parsing does not consume it.
        XCTAssertTrue(result.contains("\\$10"), "Currency $10 should be escaped, got: \(result)")
        XCTAssertTrue(result.contains("\\$1.50"), "Currency $1.50 should be escaped, got: \(result)")
    }

    // MARK: - Edge Cases

    func testEmptyDollarPair_unchanged() {
        let input = "Empty $$ here"
        // Should not crash and should handle gracefully
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertNotNil(result, "Should not crash on empty $$")
    }

    func testAdjacentInlineMath() {
        let input = "$a$ and $b$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("$a$"), "First inline math not preserved, got: \(result)")
        XCTAssertTrue(result.contains("$b$"), "Second inline math not preserved, got: \(result)")
    }

    func testMixedBlockAndInline() {
        let input = "$$E=mc^2$$ and $p=mv$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("```math\nE=mc^2\n```"), "Block math not converted, got: \(result)")
        XCTAssertTrue(result.contains("$p=mv$"), "Inline math should be preserved, got: \(result)")
    }

    func testPlainTextUnchanged() {
        let input = "Hello world, no math here."
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertEqual(result, input)
    }

    func testNestedBraces_singleDollar() {
        let input = "The fraction $\\frac{1}{2}$ is one half."
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("$\\frac{1}{2}$"), "Fraction should remain raw LaTeX, got: \(result)")
    }

    // MARK: - GitHub Backtick Delimiter (Option 3)

    func testGitHubBacktickDelimiter() {
        let input = "Use $`\\sqrt{2}`$ here"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("$\\sqrt{2}$"), "GitHub backtick math should normalize to $...$, got: \(result)")
        XCTAssertFalse(result.contains("$`"), "Raw $` should be consumed, got: \(result)")
    }

    func testGitHubBacktick_beforeGenericDollar() {
        let input = "$`x^2`$ and $y$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("$x^2$"), "GitHub backtick not normalized, got: \(result)")
        XCTAssertTrue(result.contains("$y$"), "Generic dollar math not preserved, got: \(result)")
    }

    func testGitHubBacktick_multipleExpressions() {
        let input = "If $`a`$ and $`b`$ then $`c`$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("$a$"), "First backtick math not normalized, got: \(result)")
        XCTAssertTrue(result.contains("$b$"), "Second backtick math not normalized, got: \(result)")
        XCTAssertTrue(result.contains("$c$"), "Third backtick math not normalized, got: \(result)")
    }

    // MARK: - Robustness (Option 4)

    func testEmptyInlineParens_unchanged() {
        let input = "Empty \\(\\) here"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertEqual(result, input, "Empty parens should remain unchanged, got: \(result)")
    }

    func testEmptyDisplayBrackets_unchanged() {
        let input = "Empty \\[\\] here"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertEqual(result, input, "Empty display brackets should remain unchanged, got: \(result)")
    }

    func testWhitespaceOnlyDollar_unchanged() {
        let input = "Just $ $ whitespace"
        let result = LatexPreprocessor.preprocess(input)
        // Whitespace-only content should be preserved as-is
        XCTAssertEqual(result, input, "Whitespace-only $ should remain unchanged, got: \(result)")
    }

    func testCurrencyWithCommas() {
        let input = "The price is $1,000.50 per unit"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("\\$1,000.50"), "Currency with commas should be escaped, got: \(result)")
    }

    // MARK: - Block Math with Operators (Spacing Plan)

    func testBlockMath_limOperator() {
        let input = "$$\\lim_{x \\to 0} \\frac{\\sqrt{x+1}-1}{x}$$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("```math\n"),
            "\\lim expression should be in a fenced math block, got: \(result)"
        )
        XCTAssertTrue(
            result.contains("\\lim_{x \\to 0}"),
            "\\lim operator and subscript should be preserved intact, got: \(result)"
        )
        XCTAssertTrue(
            result.contains("\\frac{\\sqrt{x+1}-1}{x}"),
            "Fraction with sqrt should be preserved intact, got: \(result)"
        )
    }

    func testBlockMath_chainedFractions() {
        let input = "$$\\frac{1}{\\sqrt{0+1}+1} = \\frac{1}{1+1} = \\frac{1}{2}$$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("```math\n"),
            "Chained fractions should be in a fenced math block, got: \(result)"
        )
        XCTAssertTrue(
            result.contains("\\frac{1}{2}"),
            "Final fraction should be preserved, got: \(result)"
        )
    }

    // MARK: - Inline Simplification (simplifyForInline)

    func testSimplify_fraction() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("\\frac{1}{2}"), "1/2")
    }

    func testSimplify_fractionComplex() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("\\frac{a+b}{c}"), "(a+b)/c")
    }

    func testSimplify_nestedFraction() {
        // Inner \frac{1}{2} → 1/2, then outer \frac{(1/2)}{3} → (1/2)/3
        let result = LatexPreprocessor.simplifyForInline("\\frac{\\frac{1}{2}}{3}")
        XCTAssertEqual(result, "(1/2)/3")
    }

    func testSimplify_sqrt() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("\\sqrt{x+1}"), "√(x+1)")
    }

    func testSimplify_sqrtSingle() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("\\sqrt{2}"), "√2")
    }

    func testSimplify_greek() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("\\alpha + \\beta"), "α + β")
    }

    func testSimplify_neq() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("x \\neq 0"), "x ≠ 0")
    }

    func testSimplify_arrow() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("x \\to \\infty"), "x → ∞")
    }

    func testSimplify_text() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("\\text{hello}"), "hello")
    }

    func testSimplify_functionName() {
        let result = LatexPreprocessor.simplifyForInline("\\lim_{x \\to 0}")
        XCTAssertTrue(result.contains("lim"), "Expected lim, got: \(result)")
        XCTAssertTrue(result.contains("→"), "Expected arrow, got: \(result)")
        XCTAssertFalse(result.contains("\\"), "Expected no backslashes, got: \(result)")
    }

    func testSimplify_preservesPlainText() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("x + y = z"), "x + y = z")
    }

    func testSimplify_dfrac() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("\\dfrac{a}{b}"), "a/b")
    }

    func testSimplify_mathrm() {
        XCTAssertEqual(LatexPreprocessor.simplifyForInline("\\mathrm{d}x"), "dx")
    }

    func testSimplify_limitsNotMangled() {
        // \limits should be stripped without mangling \lim
        let result = LatexPreprocessor.simplifyForInline("\\lim\\limits_{n}")
        XCTAssertTrue(result.contains("lim"), "Expected lim, got: \(result)")
        XCTAssertFalse(result.contains("\\"), "Backslashes should be stripped, got: \(result)")
    }

    func testSimplify_leftRight() {
        let result = LatexPreprocessor.simplifyForInline("\\left(x+1\\right)")
        XCTAssertEqual(result, "(x+1)")
    }

    // MARK: - Full Preprocess with Delimiter Normalization

    func testPreprocess_inlineFractionSimplified() {
        let input = "The value \\(\\frac{1}{2}\\) is one half"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("$\\frac{1}{2}$"), "Expected \\(...\\) normalized to $...$, got: \(result)")
    }

    func testPreprocess_inlineNeqSimplified() {
        let input = "When $x \\neq 0$ we can divide"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("$x \\neq 0$"), "Expected inline math preserved, got: \(result)")
    }

    func testPreprocess_inlineSqrtSimplified() {
        let input = "The function $f(x) = \\sqrt{x}$ is defined for $x \\geq 0$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("$f(x) = \\sqrt{x}$"), "Expected first inline math preserved, got: \(result)")
        XCTAssertTrue(result.contains("$x \\geq 0$"), "Expected second inline math preserved, got: \(result)")
    }

    func testPreprocess_blockMathNotSimplified() {
        // Block math should NOT be simplified — it goes to SwiftMath
        let input = "$$\\frac{1}{2}$$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("\\frac{1}{2}"),
            "Block math should preserve raw LaTeX for SwiftMath, got: \(result)"
        )
    }

    // MARK: - \underbrace / \overbrace Stripping

    func testBlockMath_underbraceWithLabel() {
        let input = "$$f(x) = \\underbrace{g(h(x))}_{\\text{outer function}}$$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(
            result.contains("g(h(x))"),
            "Expected underbrace content preserved, got: \(result)"
        )
        XCTAssertFalse(result.contains("\\underbrace"), "underbrace should be stripped, got: \(result)")
        XCTAssertFalse(
            result.contains("outer function"),
            "underbrace label should be stripped, got: \(result)"
        )
    }

    func testBlockMath_underbraceWithoutLabel() {
        let input = "$$\\underbrace{a + b + c}$$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("a + b + c"), "Expected content preserved, got: \(result)")
        XCTAssertFalse(result.contains("\\underbrace"), "underbrace should be stripped, got: \(result)")
    }

    func testBlockMath_overbraceWithLabel() {
        let input = "$$\\overbrace{x_1 + x_2}^{\\text{sum}}$$"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("x_1 + x_2"), "Expected content preserved, got: \(result)")
        XCTAssertFalse(result.contains("\\overbrace"), "overbrace should be stripped, got: \(result)")
        XCTAssertFalse(result.contains("sum"), "overbrace label should be stripped, got: \(result)")
    }

    func testInlineMath_underbraceStripped() {
        let input = "The term $\\underbrace{2x}_{\\text{linear}}$ grows slowly"
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertTrue(result.contains("2x"), "Expected content preserved, got: \(result)")
        XCTAssertFalse(result.contains("\\underbrace"), "underbrace should be stripped, got: \(result)")
    }

    func testBlockMath_chainRuleExact() {
        // The exact equation from the user's bug report
        let input = """
        $$
        f(x) = \\underbrace{g(h(x))}_{\\text{outer function}} \\quad \\text{(where } g(u) \\text{ is the outer layer)}
        $$
        """
        let result = LatexPreprocessor.preprocess(input)
        XCTAssertFalse(result.contains("\\underbrace"), "underbrace should be stripped, got: \(result)")
        XCTAssertTrue(result.contains("g(h(x))"), "Content should survive, got: \(result)")
        XCTAssertTrue(result.contains("\\text{"), "\\text commands should be preserved for SwiftUIMath, got: \(result)")
    }
}
