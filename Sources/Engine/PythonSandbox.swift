import Foundation
import SwiftPythonRuntime
import OSLog

// MARK: - Python Sandbox — Dedicated code execution actor

/// Executes arbitrary Python code on a dedicated worker process.
/// Captures stdout, stderr, figures (matplotlib, Plotly, PIL), and errors.
/// Jupyter-style: last expression is evaluated and its repr printed.
/// Uses a persistent namespace so variables survive across runs (REPL-style).
public actor PythonSandbox {
    private let pool: PythonProcessPool
    private let workerIdx: Int
    private let log = Logger(subsystem: "com.llama-inference-demo", category: "PythonSandbox")
    private var didSetupPlotBackend = false

    public init(pool: PythonProcessPool, workerIdx: Int) {
        self.pool = pool
        self.workerIdx = workerIdx
    }

    // MARK: - Execution

    /// Run `code` in the persistent sandbox namespace and return captured output.
    public func run(_ code: String) async -> RunOutput {
        let clock = ContinuousClock()
        let t0 = clock.now

        // Lazily configure matplotlib Agg backend on first run
        if !didSetupPlotBackend {
            _ = try? await pool.worker(workerIdx).eval("""
            try:
                import matplotlib as _mpl; _mpl.use('Agg'); del _mpl
            except ImportError:
                pass
            """)
            didSetupPlotBackend = true
        }

        // Escape the user code for embedding inside a triple-quoted Python string
        let safeCode = code
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"\"\"", with: "\\\"\\\"\\\"")

        let script = buildHarness(escapedCode: safeCode)

        let json: String
        do {
            json = try await pool.worker(workerIdx).evalResult(script)
        } catch {
            let ms = elapsedMs(from: t0, clock: clock)
            let fullDesc = (error as? PythonWorkerError).map { String(describing: $0) } ?? String(describing: error)
            log.error("Sandbox worker eval error: \(fullDesc, privacy: .public)")
            let userDesc = Self.userFacingDescription(error)
            return RunOutput(stdout: "", stderr: "", figures: [], error: userDesc, elapsedMs: ms)
        }

        let ms = elapsedMs(from: t0, clock: clock)
        return Self.decodeJSON(json, elapsedMs: ms)
    }

    // MARK: - Code Analysis

    /// Run a lightweight AST-based static analysis on Python code.
    /// Returns structured analysis in ~50ms; returns nil on failure.
    public func analyzeForReview(_ code: String) async -> CodeAnalysis? {
        let safeCode = code
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"\"\"", with: "\\\"\\\"\\\"")

        let script = """
        import ast as _ast, json as _json, textwrap as _tw

        _code = \"\"\"\(safeCode)\"\"\"
        _result = {"functions": [], "classes": [], "imports": [], "issues": [], "line_count": 0}
        try:
            _tree = _ast.parse(_tw.dedent(_code))
            _result["line_count"] = len(_code.splitlines())
            for _node in _ast.walk(_tree):
                if isinstance(_node, _ast.FunctionDef) or isinstance(_node, _ast.AsyncFunctionDef):
                    _result["functions"].append(_node.name)
                    for _d in _node.args.defaults:
                        if isinstance(_d, (_ast.List, _ast.Dict, _ast.Set)):
                            _result["issues"].append(f"mutable default arg in {_node.name}() (line {_node.lineno})")
                elif isinstance(_node, _ast.ClassDef):
                    _result["classes"].append(_node.name)
                elif isinstance(_node, _ast.Import):
                    for _alias in _node.names:
                        _result["imports"].append(_alias.name)
                elif isinstance(_node, _ast.ImportFrom):
                    if _node.module:
                        _result["imports"].append(_node.module)
                elif isinstance(_node, _ast.ExceptHandler):
                    if _node.type is None:
                        _result["issues"].append(f"bare except clause (line {_node.lineno})")
        except SyntaxError as _e:
            _result["issues"].append(f"SyntaxError: {_e.msg} (line {_e.lineno})")
            _result["line_count"] = len(_code.splitlines())

        _payload = _json.dumps(_result)
        del _ast, _json, _tw, _code, _result, _tree
        try: del _node, _alias, _d, _e
        except: pass
        _payload
        """

        do {
            let json: String = try await pool.worker(workerIdx).evalResult(script)
            return CodeAnalysis.from(json: json)
        } catch {
            log.warning("AST analysis failed (non-fatal): \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    // MARK: - Namespace Reset

    /// Wipe user-defined variables from the sandbox namespace.
    public func clearNamespace() async {
        _ = try? await pool.worker(workerIdx).eval("""
        for _k in list(globals().keys()):
            if not _k.startswith('__'):
                del globals()[_k]
        try:
            del _k
        except NameError:
            pass
        """)
        didSetupPlotBackend = false
        log.debug("Sandbox namespace cleared")
    }

    // MARK: - Python Harness

    private func buildHarness(escapedCode: String) -> String {
        """
        import sys as _sys, io as _io, json as _json, traceback as _tb, base64 as _b64

        _out = _io.StringIO()
        _err = _io.StringIO()
        _figs = []
        _exc = None

        import ast as _ast, textwrap as _tw
        _code = \"\"\"\(escapedCode)\"\"\"
        # Patch plt.show() to a no-op so figures stay open for capture
        try:
            import matplotlib.pyplot as _plt_patch
            _orig_show = _plt_patch.show
            _plt_patch.show = lambda *a, **kw: None
        except ImportError:
            _plt_patch = None
            _orig_show = None

        _prev_out, _prev_err = _sys.stdout, _sys.stderr
        _sys.stdout, _sys.stderr = _out, _err
        try:
            # Jupyter-style: if the last statement is an expression, capture its
            # value by rewriting it to `_result = <expr>` in the AST, then exec
            # the entire tree in one call. This avoids exec/eval split namespace
            # isolation when running inside evalResult's double-wrapped exec.
            _tree = _ast.parse(_tw.dedent(_code))
            _result = None
            if _tree.body and isinstance(_tree.body[-1], _ast.Expr):
                _last_val = _tree.body[-1].value
                _tree.body[-1] = _ast.Assign(
                    targets=[_ast.Name(id='_result', ctx=_ast.Store())],
                    value=_last_val
                )
                _ast.fix_missing_locations(_tree)
            exec(compile(_tree, '<exec>', 'exec'), globals())
            if '_result' in dir() and _result is not None:
                print(repr(_result))
        except Exception:
            _exc = _tb.format_exc()
        finally:
            _sys.stdout, _sys.stderr = _prev_out, _prev_err

        # Restore original plt.show
        if _plt_patch is not None and _orig_show is not None:
            _plt_patch.show = _orig_show

        # 1. Matplotlib figures (also captures seaborn, pandas .plot(), etc.)
        try:
            import matplotlib.pyplot as _plt
            try:
                for _fn in _plt.get_fignums():
                    _buf = _io.BytesIO()
                    _plt.figure(_fn).savefig(_buf, format='png', bbox_inches='tight', dpi=150)
                    _buf.seek(0)
                    _figs.append(_b64.b64encode(_buf.read()).decode())
                    _buf.close()
            except Exception as _fig_err:
                _err.write("[figure capture] matplotlib: " + str(_fig_err) + chr(10))
            finally:
                _plt.close('all')
        except ImportError:
            pass

        # 2. Plotly figures — export any plotly.graph_objects.Figure in the namespace
        try:
            import plotly.graph_objects as _go
            for _name, _obj in list(globals().items()):
                if isinstance(_obj, _go.Figure):
                    _png = _obj.to_image(format='png', scale=2)
                    _figs.append(_b64.b64encode(_png).decode())
        except ImportError:
            pass
        except Exception as _fig_err:
            _err.write("[figure capture] plotly: " + str(_fig_err) + chr(10))

        # 3. PIL/Pillow Images — any PIL.Image.Image instance in the namespace
        try:
            from PIL import Image as _PILImage
            for _name, _obj in list(globals().items()):
                if _name.startswith('_'):
                    continue
                if isinstance(_obj, _PILImage.Image):
                    _buf = _io.BytesIO()
                    _obj.save(_buf, format='PNG')
                    _buf.seek(0)
                    _figs.append(_b64.b64encode(_buf.read()).decode())
                    _buf.close()
        except ImportError:
            pass
        except Exception as _fig_err:
            _err.write("[figure capture] PIL: " + str(_fig_err) + chr(10))

        _payload = _json.dumps({'out': _out.getvalue(), 'err': _err.getvalue(), 'figs': _figs, 'exc': _exc})

        for _v in ('_out','_err','_figs','_exc','_prev_out','_prev_err','_buf','_fn','_plt','_plt_patch','_orig_show','_go','_PILImage','_name','_obj','_png','_sys','_io','_json','_tb','_b64','_ast','_tw','_code','_tree','_last_val','_result','_v'):
            try: del globals()[_v]
            except KeyError: pass
        try: del globals()['_fig_err']
        except KeyError: pass

        _payload
        """
    }

    // MARK: - JSON Decode

    static func decodeJSON(_ raw: String, elapsedMs: Int) -> RunOutput {
        guard let data = raw.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return RunOutput(stdout: "", stderr: "", figures: [], error: "Failed to decode sandbox JSON", elapsedMs: elapsedMs)
        }

        let stdout = obj["out"] as? String ?? ""
        let stderr = obj["err"] as? String ?? ""
        let exc = obj["exc"] as? String
        let b64Figs = obj["figs"] as? [String] ?? []
        let figures = b64Figs.compactMap { Data(base64Encoded: $0) }

        return RunOutput(stdout: stdout, stderr: stderr, figures: figures, error: exc, elapsedMs: elapsedMs)
    }

    // MARK: - Helpers

    /// Clean error description without worker-internal details (e.g. "Worker 3:").
    /// Used for user-facing output and the "Fix with AI" prompt sent to the model.
    private static func userFacingDescription(_ error: Error) -> String {
        if let workerError = error as? PythonWorkerError {
            switch workerError {
            case .pythonException(let type, let message, let traceback, _):
                var desc = "\(type): \(message)"
                if let tb = traceback { desc += "\n\(tb)" }
                return desc
            case .workerCrashed(_, let exitCode):
                return "Python process crashed (exit code \(exitCode))"
            case .timeout(_, let seconds):
                return "Execution timed out after \(Int(seconds))s"
            case .workerUnreachable:
                return "Python process is not responding"
            case .poolShuttingDown:
                return "Python runtime is shutting down"
            default:
                return String(describing: workerError)
            }
        }
        return String(describing: error)
    }

    private func elapsedMs(from start: ContinuousClock.Instant, clock: ContinuousClock) -> Int {
        let d = clock.now - start
        return Int(d.components.seconds * 1000 + d.components.attoseconds / 1_000_000_000_000_000)
    }
}

// MARK: - RunOutput

/// Captured output from a single sandbox execution.
public struct RunOutput: Sendable {
    public let stdout: String
    public let stderr: String
    public let figures: [Data]   // PNG bytes
    public let error: String?
    public let elapsedMs: Int

    public init(stdout: String, stderr: String, figures: [Data], error: String?, elapsedMs: Int) {
        self.stdout = stdout
        self.stderr = stderr
        self.figures = figures
        self.error = error
        self.elapsedMs = elapsedMs
    }
}
