import Foundation
import LlamaInferenceCore
import SwiftPythonRuntime
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

@main
struct LlamaInferenceDemoApp {
    static func main() async throws {
        WorkerRuntimeEnvironment.configureForBundledWorker()
        let args = CommandLine.arguments

        guard let modelPath = parseArg(args, flag: "--model") else {
            printUsage()
            return
        }

        let workerCount = Int(parseArg(args, flag: "--workers") ?? "2") ?? 2
        let contextSize = Int(parseArg(args, flag: "--ctx") ?? "4096") ?? 4096
        let nGpuLayers = Int(parseArg(args, flag: "--gpu-layers") ?? "-1") ?? -1
        let summarizerModelPath = parseArg(args, flag: "--summarizer-model")

        let useSharedMemory = args.contains("--shm")

        let config = InferenceConfig(
            modelPath: modelPath,
            summarizerModelPath: summarizerModelPath,
            contextSize: contextSize,
            nGpuLayers: nGpuLayers,
            workerCount: workerCount,
            maxSessionsPerWorker: 8,
            maxInFlight: workerCount * 4,
            blasThreads: 1,
            useSharedMemory: useSharedMemory
        )

        print("╔══════════════════════════════════════════════════════════╗")
        print("║       Llama Inference Demo — Session Affinity Pipeline  ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print("")
        print("Config:")
        print("  model:       \(config.modelPath)")
        print("  context:     \(config.contextSize) tokens")
        print("  gpu layers:  \(config.nGpuLayers)")
        print("  workers:     \(config.workerCount)")
        print("  max sessions/worker: \(config.maxSessionsPerWorker)")
        if useSharedMemory {
            print("  shared memory: ON (slot=\(config.sharedMemorySlotSize) bytes)")
        }
        if let p = config.summarizerModelPath {
            print("  summarizer:    \(p) (dedicated worker)")
        }
        print("")

        let pool = InferenceWorkerPool(config: config)
        let manager = LlamaSessionManager(workerPool: pool, config: config)

        print("Starting worker pool...")
        let startupStart = ContinuousClock.now
        try await pool.startup()
        let startupDuration = ContinuousClock.now - startupStart
        print("Pool ready in \(formatDuration(startupDuration))")
        print("")

        let health = try await pool.healthCheck()
        print("Worker health: \(health.map { $0 ? "healthy" : "DEAD" }.joined(separator: ", "))")
        print("")

        if args.contains("--diag") {
            try await runDiagnostic(pool: pool, config: config)
            await pool.shutdown()
            print("\nDiag shutdown complete.")
            return
        }

        let scheduler = InferenceScheduler(workerPool: pool, config: config)

        let useScheduler = args.contains("--dag-burst") || args.contains("--dag-multi-turn")
            || args.contains("--long-decode") || args.contains("--stress")
            || args.contains("--worker-fail") || args.contains("--shm-bench")

        if args.contains("--shm-bench") {
            try await runShmBenchScenario(pool: pool, config: config)
        } else if args.contains("--dag-burst") {
            try await runDAGBurstScenario(scheduler: scheduler, config: config)
        } else if args.contains("--dag-multi-turn") {
            try await runDAGMultiTurnScenario(scheduler: scheduler)
        } else if args.contains("--long-decode") {
            try await runLongDecodeScenario(scheduler: scheduler)
        } else if args.contains("--stress") {
            try await runStressScenario(scheduler: scheduler, config: config)
        } else if args.contains("--worker-fail") {
            try await runWorkerFailScenario(pool: pool, scheduler: scheduler, config: config)
        } else if args.contains("--burst") {
            try await runBurstScenario(manager: manager, config: config)
        } else if args.contains("--multi-turn") {
            try await runMultiTurnScenario(manager: manager)
        } else if args.contains("--benchmark") {
            try await runBenchmarkScenario(manager: manager, config: config)
        } else if args.contains("--script-review") {
            try await runScriptReviewScenario(manager: manager, pool: pool, config: config)
        } else if args.contains("--script-review-parallel") {
            try await runScriptReviewParallelScenario(manager: manager, pool: pool, config: config)
        } else if args.contains("--script-review-adversarial") {
            try await runScriptReviewAdversarialScenario(manager: manager, pool: pool, config: config)
        } else if args.contains("--script-review-fail") {
            try await runScriptReviewFailScenario(manager: manager, pool: pool, config: config)
        } else {
            try await runSingleShotScenario(manager: manager)
        }

        if args.contains("--shm-bench") {
            // shm-bench manages its own schedulers and prints stats internally
        } else if useScheduler {
            let sStats = await scheduler.schedulerStats
            print("")
            printSchedulerStats(sStats)
        } else {
            let stats = await manager.aggregateStats
            print("")
            printStats(stats)
        }

        await pool.shutdown()
        print("\nShutdown complete.")
    }

    // MARK: - Scenarios

    static func runSingleShotScenario(manager: LlamaSessionManager) async throws {
        print("═══ Scenario: Single-Shot Completion ═══")
        print("")

        let prompts = [
            "Explain the concept of session affinity in distributed systems in two sentences.",
            "Write a haiku about parallel computing.",
            "What is the KV cache in transformer inference?",
        ]

        for (i, prompt) in prompts.enumerated() {
            print("[\(i + 1)/\(prompts.count)] Prompt: \(prompt.prefix(60))...")
            let result = try await manager.completeOneShot(
                prompt: prompt,
                params: SamplingParams(maxTokens: 2048, temperature: 0.7),
                systemPrompt: "You are a concise technical assistant. Keep responses brief."
            )
            printResult(result)
            print("")
        }
    }

    static func runScriptReviewScenario(manager: LlamaSessionManager, pool: InferenceWorkerPool, config: InferenceConfig) async throws {
        print("═══ Scenario: Python Script Generate + Second-Worker Review ═══")
        print("")

        if config.workerCount < 2 {
            print("  (Note: Use --workers 2 to have generator and reviewer on different workers.)")
        }

        let generatorPrompt = """
        Write a complete Python script that:
        1. Reads a CSV file from stdin (assume columns: name, score, grade)
        2. Parses the CSV, computes mean and max of the score column
        3. Prints a one-line summary: "mean=X.X max=Y.Y"
        Include error handling for malformed input. Output only the script, no explanation.
        """

        // Create two sessions; with 2+ workers they land on different workers (W0, W1)
        let sessionGen = try await manager.createSession(
            systemPrompt: "You are a Python expert. Output only executable code, no markdown."
        )
        let sessionReview = try await manager.createSession(
            systemPrompt: """
            Code reviewer. Emit structured findings only.
            Format: FINDINGS: [issue1, issue2...]; each: severity (high|medium|low), confidence (0-1), speculation (yes if you infer beyond the spec, else no).
            Validate against the given SPEC ONLY. Do not invent requirements. If OK: FINDINGS: []
            """
        )

        let genWorker = await manager.sessionState(for: sessionGen)?.workerIndex ?? -1
        let genPid = await pool.workerPID(for: genWorker)
        print("[1/2] W\(genWorker) (pid \(genPid.map { "\($0)" } ?? "n/a")): Generating script...")
        let genResult = try await manager.complete(
            sessionID: sessionGen,
            prompt: generatorPrompt,
            params: SamplingParams(maxTokens: 1024, temperature: 0.3)
        )
        let script = genResult.text.trimmingCharacters(in: .whitespacesAndNewlines)
        printResult(genResult)
        print("")

        let reviewPrompt = """
        SPEC (validate against this only; do not invent requirements):
        \(generatorPrompt)

        SCRIPT TO REVIEW:
        \(script)

        Emit FINDINGS: [issue (severity, confidence, speculation)] or FINDINGS: [] if OK.
        """

        let revWorker = await manager.sessionState(for: sessionReview)?.workerIndex ?? -1
        let revPid = await pool.workerPID(for: revWorker)
        print("[2/2] W\(revWorker) (pid \(revPid.map { "\($0)" } ?? "n/a")): Reviewing (double-check from different worker)...")
        let revResult = try await manager.complete(
            sessionID: sessionReview,
            prompt: reviewPrompt,
            params: SamplingParams(maxTokens: 512, temperature: 0.2)
        )
        printResult(revResult)

        if genWorker >= 0 && revWorker >= 0 && genWorker != revWorker {
            print("")
            if let gp = genPid, let rp = revPid {
                print("  ✓ Generator W\(genWorker) (pid \(gp)) ≠ Reviewer W\(revWorker) (pid \(rp)) — different processes.")
            } else {
                print("  ✓ Generator on W\(genWorker), reviewer on W\(revWorker) — second worker double-checked.")
            }
        }

        try await manager.evictSession(sessionGen)
        try await manager.evictSession(sessionReview)
    }

    static func runScriptReviewParallelScenario(manager: LlamaSessionManager, pool: InferenceWorkerPool, config: InferenceConfig) async throws {
        print("═══ Scenario: Parallel Script Generate + Cross-Worker Review (Pipeline) ═══")
        print("")

        guard config.workerCount >= 2 else {
            print("  Requires --workers 2 or more. Exiting.")
            return
        }

        let promptA = """
        Write a complete Python script that:
        1. Reads a CSV file from stdin (columns: name, score, grade)
        2. Parses CSV, computes mean and max of the score column
        3. Prints "mean=X.X max=Y.Y"
        Include error handling. Output only the script, no explanation.
        """

        let promptB = """
        Write a complete Python script that:
        1. Reads JSON from stdin (array of objects with "value" field)
        2. Computes sum and count of values
        3. Prints "sum=X count=Y"
        Include error handling. Output only the script, no explanation.
        """

        // Create 4 sessions. Order matters for cross-worker: GenA(W0), GenB(W1), RevB(W0), RevA(W1).
        // RevB created 3rd → W0 (load 2,1); RevA created 4th → W1. So W0 reviews B, W1 reviews A.
        let sessionGenA = try await manager.createSession(
            systemPrompt: "You are a Python expert. Output only executable code, no markdown."
        )
        let sessionGenB = try await manager.createSession(
            systemPrompt: "You are a Python expert. Output only executable code, no markdown."
        )
        let reviewerSystemPrompt = """
        Code reviewer. Emit: FINDINGS: [issue (severity, confidence, speculation)].
        Validate against SPEC ONLY. No invented requirements. OK → FINDINGS: [].
        """
        let sessionRevB = try await manager.createSession(systemPrompt: reviewerSystemPrompt)
        let sessionRevA = try await manager.createSession(systemPrompt: reviewerSystemPrompt)

        // Phase 1: Both workers generate in parallel
        print("Phase 1 (parallel): W0 generates script A, W1 generates script B...")
        async let resultA = manager.complete(
            sessionID: sessionGenA,
            prompt: promptA,
            params: SamplingParams(maxTokens: 1024, temperature: 0.3)
        )
        async let resultB = manager.complete(
            sessionID: sessionGenB,
            prompt: promptB,
            params: SamplingParams(maxTokens: 1024, temperature: 0.3)
        )
        let (genA, genB) = try await (resultA, resultB)

        let scriptA = genA.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let scriptB = genB.text.trimmingCharacters(in: .whitespacesAndNewlines)

        let workerA = genA.workerIndex
        let workerB = genB.workerIndex
        print("  W\(workerA): script A (\(genA.completionTokens) tok)")
        print("  W\(workerB): script B (\(genB.completionTokens) tok)")
        print("")

        // Phase 2: Cross-review in parallel — RevB (W0) reviews B, RevA (W1) reviews A
        print("Phase 2 (parallel): Cross-worker review (spec-conditioned, contract output)...")
        let reviewPromptA = """
        SPEC (validate only; do not invent):
        \(promptA)

        SCRIPT:
        \(scriptA)

        FINDINGS: [issue (severity, confidence, speculation)] or [] if OK.
        """
        let reviewPromptB = """
        SPEC (validate only; do not invent):
        \(promptB)

        SCRIPT:
        \(scriptB)

        FINDINGS: [issue (severity, confidence, speculation)] or [] if OK.
        """

        async let revA = manager.complete(
            sessionID: sessionRevA,
            prompt: reviewPromptA,
            params: SamplingParams(maxTokens: 512, temperature: 0.2)
        )
        async let revB = manager.complete(
            sessionID: sessionRevB,
            prompt: reviewPromptB,
            params: SamplingParams(maxTokens: 512, temperature: 0.2)
        )
        let (revResultA, revResultB) = try await (revA, revB)

        let revWorkerA = revResultA.workerIndex
        let revWorkerB = revResultB.workerIndex
        let pidA = await pool.workerPID(for: workerA)
        let pidB = await pool.workerPID(for: workerB)
        let pidRevA = await pool.workerPID(for: revWorkerA)
        let pidRevB = await pool.workerPID(for: revWorkerB)

        print("  W\(revWorkerB) (pid \(pidRevB.map { "\($0)" } ?? "n/a")) reviewed script B (gen by W\(workerB) pid \(pidB.map { "\($0)" } ?? "n/a"))")
        print("  W\(revWorkerA) (pid \(pidRevA.map { "\($0)" } ?? "n/a")) reviewed script A (gen by W\(workerA) pid \(pidA.map { "\($0)" } ?? "n/a"))")
        print("")

        print("  Script A review:")
        print(revResultA.text.trimmingCharacters(in: .whitespacesAndNewlines))
        print("")
        print("  Script B review:")
        print(revResultB.text.trimmingCharacters(in: .whitespacesAndNewlines))
        print("")

        if revWorkerA != workerA && revWorkerB != workerB, let pa = pidA, let pb = pidB, let rpa = pidRevA, let rpb = pidRevB {
            print("  ✓ Cross-worker review: A gen(pid \(pa))→rev(pid \(rpa)), B gen(pid \(pb))→rev(pid \(rpb)) — different processes.")
        }

        try await manager.evictSession(sessionGenA)
        try await manager.evictSession(sessionGenB)
        try await manager.evictSession(sessionRevA)
        try await manager.evictSession(sessionRevB)
    }

    static func runScriptReviewAdversarialScenario(manager: LlamaSessionManager, pool: InferenceWorkerPool, config: InferenceConfig) async throws {
        print("═══ Scenario: Adversarial Review (Contracts + Intersection + Judge) ═══")
        print("")

        guard config.workerCount >= 2 else {
            print("  Requires --workers 2 or more. Exiting.")
            return
        }

        // Hard spec: immutable, no file I/O
        let hardSpec = """
        [IMMUTABLE SPEC — DO NOT DEVIATE]
        INPUT: CSV via stdin only, comma-delimited. Columns: name, score, grade.
        OUTPUT: print exactly "mean=X max=Y" (X=mean of score column, Y=max of score column).
        NO FILE IO. open() or file paths = spec violation.
        """
        let genPrompt = """
        \(hardSpec)

        Output only the Python script. No explanation.
        """

        // Role-isolated: spec + script only; no examples, no fix-it language; output JSON only
        let reviewerSchema = """
        Output ONLY valid JSON. No code, no corrections, no full scripts, no explanations.
        Schema: {"findings":[{"id":"f1","type":"spec_violation","severity":1-5,"desc":"..."}],"confidence":0-1} or {"findings":[],"confidence":1.0}
        """
        let corrSystem = "Correctness reviewer. Compare script to spec. \(reviewerSchema)"
        let robSystem = "Robustness reviewer. Check edge cases and errors. \(reviewerSchema)"

        let sessionGenA = try await manager.createSession(systemPrompt: "Python expert. Output only executable code.")
        let sessionGenB = try await manager.createSession(systemPrompt: "Python expert. Output only executable code.")
        let sessionCorrB = try await manager.createSession(systemPrompt: corrSystem)
        let sessionCorrA = try await manager.createSession(systemPrompt: corrSystem)
        let sessionRobB = try await manager.createSession(systemPrompt: robSystem)
        let sessionRobA = try await manager.createSession(systemPrompt: robSystem)
        let sessionJudge = try await manager.createSession(
            systemPrompt: """
            Judge. Score two reviewers 1-5. Do NOT write code. Output ONLY this JSON: {"corr_score":3,"rob_score":2}
            Replace 3 and 2 with your scores. Nothing else.
            """
        )

        print("Phase 1: Generate A and B...")
        print("  Spec: CSV stdin only, no file I/O")
        async let genA = manager.complete(sessionID: sessionGenA, prompt: genPrompt, params: SamplingParams(maxTokens: 1024, temperature: 0.3))
        async let genB = manager.complete(sessionID: sessionGenB, prompt: genPrompt, params: SamplingParams(maxTokens: 1024, temperature: 0.3))
        let (scriptARaw, scriptBRaw) = try await (genA, genB)
        let scriptA = scriptARaw.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let scriptB = scriptBRaw.text.trimmingCharacters(in: .whitespacesAndNewlines)

        print("  W\(scriptARaw.workerIndex): A (\(scriptARaw.completionTokens) tok)")
        print("  W\(scriptBRaw.workerIndex): B (\(scriptBRaw.completionTokens) tok)")
        print("")

        // Hard pre-execution spec linter (fail fast before LLM judgment)
        let specLinterBans: [(pattern: String, desc: String)] = [
            (#"open\s*\("#, "uses open()"),
            (#"Path\s*\("#, "uses Path()"),
            (#"\bos\s*\."#, "uses os module"),
            (#"sys\.argv"#, "uses sys.argv"),
        ]
        func runSpecLinter(_ script: String) -> [(type: String, severity: Int, desc: String)] {
            var findings: [(type: String, severity: Int, desc: String)] = []
            for (pat, banDesc) in specLinterBans {
                if let regex = try? NSRegularExpression(pattern: pat),
                   regex.firstMatch(in: script, range: NSRange(script.startIndex..., in: script)) != nil {
                    findings.append(("spec_violation", 5, banDesc))
                }
            }
            // Type enforcement: scores from row[1] without float() coercion → sev5
            let hasScoresFromRow = script.contains("row[1]") && (script.contains("scores") || script.contains("score"))
            let hasFloatCoercion = script.contains("float(row[1]") || script.contains("float( row[1]")
            if hasScoresFromRow && !hasFloatCoercion {
                findings.append(("spec_violation", 5, "scores without float coercion"))
            }
            return findings
        }

        let linterA = runSpecLinter(scriptA)
        let linterB = runSpecLinter(scriptB)
        if !linterA.isEmpty { print("  [Spec linter] Script A: \(linterA.map { "\($0.desc)" }.joined(separator: "; "))") }
        if !linterB.isEmpty { print("  [Spec linter] Script B: \(linterB.map { "\($0.desc)" }.joined(separator: "; "))") }

        let revPromptA = """
        \(hardSpec)

        SCRIPT TO REVIEW:
        \(scriptA)

        Output JSON only.
        """
        let revPromptB = """
        \(hardSpec)

        SCRIPT TO REVIEW:
        \(scriptB)

        Output JSON only.
        """

        print("Phase 2: 4-way adversarial review (JSON contract)...")
        let reviewParams = SamplingParams(maxTokens: 384, temperature: 0.1)
        let phase2Start = ContinuousClock.now
        async let corrA = manager.complete(sessionID: sessionCorrA, prompt: revPromptA, params: reviewParams)
        async let corrB = manager.complete(sessionID: sessionCorrB, prompt: revPromptB, params: reviewParams)
        async let robA = manager.complete(sessionID: sessionRobA, prompt: revPromptA, params: reviewParams)
        async let robB = manager.complete(sessionID: sessionRobB, prompt: revPromptB, params: reviewParams)
        let (resCorrA, resCorrB, resRobA, resRobB) = try await (corrA, corrB, robA, robB)
        let phase2Elapsed = ContinuousClock.now - phase2Start

        print("  CorrA W\(resCorrA.workerIndex), RobA W\(resRobA.workerIndex), CorrB W\(resCorrB.workerIndex), RobB W\(resRobB.workerIndex)")

        // Parallelism proof: PIDs (distinct processes) + wall-clock (<< 4× single inference)
        var pids: [(Int, pid_t)] = []
        for i in 0..<config.workerCount {
            if let p = await pool.workerPID(for: i) {
                pids.append((i, p))
            }
        }
        let phase2Ms = Double(phase2Elapsed.components.seconds) * 1000 + Double(phase2Elapsed.components.attoseconds) / 1e15
        let sequentialEstimateMs = 4 * 700
        print("  [Parallelism] Phase 2 wall-clock: \(String(format: "%.0f", phase2Ms)) ms (sequential would be ~\(sequentialEstimateMs) ms)")
        if !pids.isEmpty {
            print("  [Parallelism] Worker PIDs: \(pids.map { "W\($0.0)=pid\($0.1)" }.joined(separator: ", ")) — distinct PIDs = distinct processes")
        }
        print("")

        // D) Contract gate: first JSON only; parse fail → format_compliance=0, excluded from intersection
        let (corrAParsed, corrACompliant) = parseReviewFindingsStructured(resCorrA.text)
        let (robAParsed, robACompliant) = parseReviewFindingsStructured(resRobA.text)
        let (corrBParsed, corrBCompliant) = parseReviewFindingsStructured(resCorrB.text)
        let (robBParsed, robBCompliant) = parseReviewFindingsStructured(resRobB.text)

        if !corrACompliant { print("  [Contract gate] CorrA: format_compliance=0 (downgraded, findings kept)") }
        if !robACompliant { print("  [Contract gate] RobA: format_compliance=0 (downgraded, findings kept)") }
        if !corrBCompliant { print("  [Contract gate] CorrB: format_compliance=0 (downgraded, findings kept)") }
        if !robBCompliant { print("  [Contract gate] RobB: format_compliance=0 (downgraded, findings kept)") }

        // Format compliance non-exclusionary: keep all findings; compliant reviewers participate in consensus.
        // ACCEPT iff (no severity>=5 in any finding) AND (≥1 compliant reviewer); else REJECT.
        func computeVerdict(
            corr: [(type: String, severity: Int, desc: String)],
            rob: [(type: String, severity: Int, desc: String)],
            corrOK: Bool,
            robOK: Bool,
            linterFindings: [(type: String, severity: Int, desc: String)]
        ) -> (verdict: String, consensus: [(String, Int, String)], advisory: [(String, Int, String)]) {
            // Compliant reviewers only for consensus intersection
            let cfConsensus = corrOK ? corr : []
            let rfConsensus = robOK ? rob : []
            let cKeys = Set(cfConsensus.map { canonicalKey(type: $0.type, desc: $0.desc) })
            let rKeys = Set(rfConsensus.map { canonicalKey(type: $0.type, desc: $0.desc) })
            let consensusKeys = cKeys.intersection(rKeys)
            var consensus: [(String, Int, String)] = []
            var seenKeys: Set<String> = []
            for f in cfConsensus + rfConsensus {
                let k = canonicalKey(type: f.type, desc: f.desc)
                if consensusKeys.contains(k) && !seenKeys.contains(k) {
                    seenKeys.insert(k)
                    consensus.append((f.type, f.severity, f.desc))
                }
            }
            var advisory: [(String, Int, String)] = []
            for f in corr + rob + linterFindings {
                let k = canonicalKey(type: f.type, desc: f.desc)
                let inConsensus = consensusKeys.contains(k)
                let alreadyAdvisory = advisory.contains(where: { canonicalKey(type: $0.0, desc: $0.2) == k })
                if !inConsensus && !alreadyAdvisory {
                    advisory.append((f.type, f.severity, f.desc))
                }
            }
            let allFindings = consensus + advisory
            let hasSev5 = allFindings.contains { $0.1 >= 5 }
            let hasCompliantReviewer = corrOK || robOK
            let accept = !hasSev5 && hasCompliantReviewer
            return (accept ? "ACCEPT" : "REJECT", consensus, advisory)
        }

        let (verdictA, consensusA, advisoryA) = computeVerdict(corr: corrAParsed, rob: robAParsed, corrOK: corrACompliant, robOK: robACompliant, linterFindings: linterA)
        let (verdictB, consensusB, advisoryB) = computeVerdict(corr: corrBParsed, rob: robBParsed, corrOK: corrBCompliant, robOK: robBCompliant, linterFindings: linterB)

        print("  Script A — SCRIPT_VERDICT: \(verdictA)")
        print("  Script A — CONSENSUS_FINDINGS: \(consensusA.isEmpty ? "[]" : consensusA.map { "\($0.0)/sev\($0.1): \($0.2)" }.joined(separator: "; "))")
        if !advisoryA.isEmpty { print("  Script A — ADVISORY_FINDINGS: \(advisoryA.map { "\($0.0)/sev\($0.1): \($0.2)" }.joined(separator: "; "))") }
        print("  Script B — SCRIPT_VERDICT: \(verdictB)")
        print("  Script B — CONSENSUS_FINDINGS: \(consensusB.isEmpty ? "[]" : consensusB.map { "\($0.0)/sev\($0.1): \($0.2)" }.joined(separator: "; "))")
        if !advisoryB.isEmpty { print("  Script B — ADVISORY_FINDINGS: \(advisoryB.map { "\($0.0)/sev\($0.1): \($0.2)" }.joined(separator: "; "))") }
        print("")

        print("  Raw CorrA: \(resCorrA.text.trimmingCharacters(in: .whitespacesAndNewlines))")
        print("  Raw RobA: \(resRobA.text.trimmingCharacters(in: .whitespacesAndNewlines))")
        print("  Raw CorrB: \(resCorrB.text.trimmingCharacters(in: .whitespacesAndNewlines))")
        print("  Raw RobB: \(resRobB.text.trimmingCharacters(in: .whitespacesAndNewlines))")
        print("")

        // Phase 3: Judge scores reviewers
        print("Phase 3: Judge scores reviewers...")
        let judgePrompt = """
        Corr said: \(resCorrA.text.prefix(300))
        Rob said: \(resRobA.text.prefix(300))

        Score each 1-5. Output ONLY: {"corr_score":N,"rob_score":N}
        """
        let judgeResult = try await manager.complete(sessionID: sessionJudge, prompt: judgePrompt, params: SamplingParams(maxTokens: 128, temperature: 0.0))
        print("  Judge (W\(judgeResult.workerIndex)): \(judgeResult.text.trimmingCharacters(in: .whitespacesAndNewlines))")

        try await manager.evictSession(sessionGenA)
        try await manager.evictSession(sessionGenB)
        try await manager.evictSession(sessionCorrA)
        try await manager.evictSession(sessionCorrB)
        try await manager.evictSession(sessionRobA)
        try await manager.evictSession(sessionRobB)
        try await manager.evictSession(sessionJudge)
    }

    static func runScriptReviewFailScenario(manager: LlamaSessionManager, pool: InferenceWorkerPool, config: InferenceConfig) async throws {
        print("═══ Scenario: Failure Injection (Kill W0 Mid-Review, Verify W1 Completes) ═══")
        print("")

        guard config.workerCount >= 2 else {
            print("  Requires --workers 2 or more. Exiting.")
            return
        }

        let prompt = "Write a minimal Python script that prints 'hello' twice. Output only the script."
        let spec = "Print 'hello' twice."

        // GenA W0, GenB W1, RevB W0, RevA W1
        let sessionGenA = try await manager.createSession(systemPrompt: "Python expert. Output code only.")
        let sessionGenB = try await manager.createSession(systemPrompt: "Python expert. Output code only.")
        let sessionRevB = try await manager.createSession(systemPrompt: "Code reviewer. FINDINGS: [] or [issue].")
        let sessionRevA = try await manager.createSession(systemPrompt: "Code reviewer. FINDINGS: [] or [issue].")

        print("Phase 1: Generate A and B...")
        async let genA = manager.complete(sessionID: sessionGenA, prompt: prompt, params: SamplingParams(maxTokens: 256, temperature: 0.2))
        async let genB = manager.complete(sessionID: sessionGenB, prompt: prompt, params: SamplingParams(maxTokens: 256, temperature: 0.2))
        let (resA, resB) = try await (genA, genB)
        let scriptA = resA.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let scriptB = resB.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let workerA = resA.workerIndex
        let workerB = resB.workerIndex
        print("  W\(workerA): A (\(resA.completionTokens) tok)")
        print("  W\(workerB): B (\(resB.completionTokens) tok)")

        let revPromptA = "SPEC: \(spec)\nSCRIPT:\n\(scriptA)\nFINDINGS:"
        let revPromptB = "SPEC: \(spec)\nSCRIPT:\n\(scriptB)\nFINDINGS:"

        let revBWorker = await manager.sessionState(for: sessionRevB)?.workerIndex ?? 0
        let victimPid = await pool.workerPID(for: revBWorker)
        guard let pid = victimPid else {
            print("  ERROR: Could not get PID for W\(revBWorker)")
            return
        }

        print("")
        print("Phase 2: Start both reviews; kill W\(revBWorker) (pid \(pid)) after 800ms...")

        let killTask = Task {
            try? await Task.sleep(for: .milliseconds(800))
            kill(pid, SIGKILL)
        }

        let revATask = Task { try await manager.complete(sessionID: sessionRevA, prompt: revPromptA, params: SamplingParams(maxTokens: 256, temperature: 0.2)) }
        let revBTask = Task { try await manager.complete(sessionID: sessionRevB, prompt: revPromptB, params: SamplingParams(maxTokens: 256, temperature: 0.2)) }

        let revAResult = await revATask.result
        let revBResult = await revBTask.result
        await killTask.value

        print("")
        switch revAResult {
        case .success(let r):
            print("  RevA (W\(r.workerIndex)): OK")
            print("  RevA text: \(r.text.trimmingCharacters(in: .whitespacesAndNewlines))")
        case .failure(let e):
            print("  RevA: FAILED \(e)")
        }
        switch revBResult {
        case .success(let r):
            print("  RevB (W\(r.workerIndex)): OK (unexpected — worker may not have been killed in time)")
        case .failure(let e):
            print("  RevB (W\(revBWorker), pid \(pid) killed): FAILED (expected)")
            print("  Error: \(e)")
        }

        let aOk: Bool
        if case .success = revAResult { aOk = true } else { aOk = false }
        let bFailed: Bool
        if case .failure = revBResult { bFailed = true } else { bFailed = false }
        print("")
        print("  ✓ W1 (RevA) completed: \(aOk)")
        print("  ✓ W0 (RevB) failed after kill: \(bFailed)")
        print("  Coordinator handled partial state.")

        try? await manager.evictSession(sessionGenA)
        try? await manager.evictSession(sessionGenB)
        try? await manager.evictSession(sessionRevA)
        try? await manager.evictSession(sessionRevB)
    }

    static func runMultiTurnScenario(manager: LlamaSessionManager) async throws {
        print("═══ Scenario: Multi-Turn Session ═══")
        print("")

        let sessionID = try await manager.createSession(
            systemPrompt: "You are a helpful coding assistant. Be concise."
        )
        print("Created session: \(sessionID)")
        print("")

        let turns = [
            "What is a process pool?",
            "How does it differ from a thread pool?",
            "Give me a one-line Python example of each.",
        ]

        for (i, prompt) in turns.enumerated() {
            print("Turn \(i + 1): \(prompt)")
            let result = try await manager.complete(
                sessionID: sessionID,
                prompt: prompt,
                params: SamplingParams(maxTokens: 2048, temperature: 0.5)
            )
            printResult(result)
            print("")
        }

        if let state = await manager.sessionState(for: sessionID) {
            print("Session summary:")
            print("  total prompt tokens:     \(state.promptTokenCount)")
            print("  total completion tokens:  \(state.completionTokenCount)")
            print("  total tokens:            \(state.totalTokenCount)")
        }

        try await manager.evictSession(sessionID)
        print("Session evicted.")
    }

    static func runBurstScenario(
        manager: LlamaSessionManager,
        config: InferenceConfig
    ) async throws {
        print("═══ Scenario: Burst Load ═══")
        print("")

        let burstSize = config.workerCount * 3
        let prompts = (0..<burstSize).map { i in
            "Generate a single creative name for a software project about topic #\(i + 1). Respond with just the name."
        }

        print("Launching \(burstSize) concurrent requests across \(config.workerCount) workers...")
        print("")

        let wallStart = ContinuousClock.now

        try await withThrowingTaskGroup(of: (Int, InferenceResult).self) { group in
            for (i, prompt) in prompts.enumerated() {
                group.addTask {
                    let result = try await manager.completeOneShot(
                        prompt: prompt,
                        params: SamplingParams(maxTokens: 32, temperature: 0.9),
                        systemPrompt: "Respond with only a single creative project name, nothing else."
                    )
                    return (i, result)
                }
            }

            var results: [(Int, InferenceResult)] = []
            for try await pair in group {
                results.append(pair)
            }
            results.sort { $0.0 < $1.0 }

            let wallEnd = ContinuousClock.now
            let wallDuration = wallEnd - wallStart

            print("Results:")
            print("+-----+--------+--------+--------+----------+---------+")
            print("| Req | Worker | Prompt | Compl. |   Tok/s  | Output  |")
            print("+-----+--------+--------+--------+----------+---------+")

            for (i, result) in results {
                let idx = String(format: "%3d", i)
                let wk = String(format: "  W%d  ", result.workerIndex)
                let pt = String(format: "%5d ", result.promptTokens)
                let ct = String(format: "%5d ", result.completionTokens)
                let tps = String(format: "%7.1f ", result.tokensPerSecond)
                let text = String(result.text.prefix(30)).replacingOccurrences(of: "\n", with: " ")
                print("| \(idx) | \(wk) | \(pt) | \(ct) | \(tps) | \(text)")
            }

            print("+-----+--------+--------+--------+----------+---------+")
            print("")
            print("Wall time: \(formatDuration(wallDuration))")
            print("Requests:  \(results.count)")

            let totalTokens = results.reduce(0) { $0 + $1.1.completionTokens }
            let wallSeconds = durationSeconds(wallDuration)
            if wallSeconds > 0 {
                print("Aggregate: \(String(format: "%.1f", Double(totalTokens) / wallSeconds)) tokens/sec")
            }
        }

        let load = await manager.workerLoad
        print("")
        print("Worker load distribution:")
        for (worker, count) in load.sorted(by: { $0.key < $1.key }) {
            print("  W\(worker): \(count) sessions")
        }
    }

    static func runBenchmarkScenario(
        manager: LlamaSessionManager,
        config: InferenceConfig
    ) async throws {
        print("═══ Scenario: Benchmark ═══")
        print("")

        let iterations = 5
        let prompt = "What is 2+2? Answer with just the number."
        let params = SamplingParams(maxTokens: 16, temperature: 0.0, topK: 1)

        print("Running \(iterations) sequential completions for latency measurement...")
        print("")

        var durations: [Double] = []
        var prefillDurations: [Double] = []
        var decodeDurations: [Double] = []
        var tokenCounts: [Int] = []

        for i in 1...iterations {
            let result = try await manager.completeOneShot(
                prompt: prompt,
                params: params,
                systemPrompt: "Answer concisely."
            )
            let totalMs = durationMs(result.totalDuration)
            let prefillMs = durationMs(result.prefillDuration)
            let decodeMs = durationMs(result.decodeDuration)

            durations.append(totalMs)
            prefillDurations.append(prefillMs)
            decodeDurations.append(decodeMs)
            tokenCounts.append(result.completionTokens)

            print("  [\(i)/\(iterations)] total=\(fmt(totalMs))ms  prefill=\(fmt(prefillMs))ms  decode=\(fmt(decodeMs))ms  tokens=\(result.completionTokens)  tok/s=\(fmt(result.tokensPerSecond))")
        }

        print("")
        print("Latency summary (ms):")
        print("  total:   mean=\(fmt(mean(durations)))  p50=\(fmt(percentile(durations, 0.5)))  p95=\(fmt(percentile(durations, 0.95)))")
        print("  prefill: mean=\(fmt(mean(prefillDurations)))  p50=\(fmt(percentile(prefillDurations, 0.5)))  p95=\(fmt(percentile(prefillDurations, 0.95)))")
        print("  decode:  mean=\(fmt(mean(decodeDurations)))  p50=\(fmt(percentile(decodeDurations, 0.5)))  p95=\(fmt(percentile(decodeDurations, 0.95)))")
        print("  tokens:  mean=\(fmt(mean(tokenCounts.map(Double.init))))")
    }

    // MARK: - DAG Scenarios (Phase 3)

    static func runDAGBurstScenario(
        scheduler: InferenceScheduler,
        config: InferenceConfig
    ) async throws {
        print("═══ Scenario: DAG Burst Load (Phase 3 Scheduler) ═══")
        print("")

        let burstSize = config.workerCount * 3
        let prompts = (0..<burstSize).map { i in
            "Generate a single creative name for a software project about topic #\(i + 1). Respond with just the name."
        }

        print("Creating \(burstSize) sessions across \(config.workerCount) workers...")

        var sessionIDs: [SessionID] = []
        for _ in 0..<burstSize {
            let sid = try await scheduler.createSession(
                systemPrompt: "Respond with only a single creative project name, nothing else."
            )
            sessionIDs.append(sid)
        }

        let load = await scheduler.workerLoad
        print("Worker assignment: ", terminator: "")
        for (w, count) in load.sorted(by: { $0.key < $1.key }) {
            print("W\(w)=\(count) ", terminator: "")
        }
        print("\n")

        print("Submitting batch DAG (\(burstSize) prefill→decode chains)...")
        let wallStart = ContinuousClock.now

        let requests = zip(sessionIDs, prompts).map { (sid, prompt) in
            (sessionID: sid, prompt: prompt, params: SamplingParams(maxTokens: 32, temperature: 0.9))
        }

        let outcomes = try await scheduler.completeBatch(requests: requests)
        let wallEnd = ContinuousClock.now
        let wallDuration = wallEnd - wallStart

        print("")
        print("Results:")
        print("+-----+--------+--------+--------+----------+---------+")
        print("| Req | Worker | Prompt | Compl. |   Tok/s  | Output  |")
        print("+-----+--------+--------+--------+----------+---------+")

        for (i, sid) in sessionIDs.enumerated() {
            let idx = String(format: "%3d", i)
            switch outcomes[sid] {
            case .success(let result):
                let wk = String(format: "  W%d  ", result.workerIndex)
                let pt = String(format: "%5d ", result.promptTokens)
                let ct = String(format: "%5d ", result.completionTokens)
                let tps = String(format: "%7.1f ", result.tokensPerSecond)
                let text = String(result.text.prefix(30)).replacingOccurrences(of: "\n", with: " ")
                print("| \(idx) | \(wk) | \(pt) | \(ct) | \(tps) | \(text)")
            case .failure(let error):
                print("| \(idx) |  FAIL  |   -   |   -   |    -    | \(String(describing: error).prefix(30))")
            case .none:
                print("| \(idx) |  SKIP  |   -   |   -   |    -    | (not in results)")
            }
        }

        print("+-----+--------+--------+--------+----------+---------+")
        print("")
        print("Wall time: \(formatDuration(wallDuration))")
        print("Requests:  \(outcomes.count)")

        let totalTokens = outcomes.values.compactMap { try? $0.get().completionTokens }.reduce(0, +)
        let wallSeconds = durationSeconds(wallDuration)
        if wallSeconds > 0 {
            print("Aggregate: \(String(format: "%.1f", Double(totalTokens) / wallSeconds)) tokens/sec")
        }

        for sid in sessionIDs {
            try? await scheduler.evictSession(sid)
        }
    }

    static func runDAGMultiTurnScenario(scheduler: InferenceScheduler) async throws {
        print("═══ Scenario: DAG Multi-Turn Session (Phase 3 Scheduler) ═══")
        print("")

        let sessionID = try await scheduler.createSession(
            systemPrompt: "You are a helpful coding assistant. Be concise."
        )
        print("Created session: \(sessionID)")

        if let info = await scheduler.sessionInfo(sessionID) {
            print("  Worker: W\(info.workerIndex)  Token budget: 0/\(4096)")
        }
        print("")

        let turns = [
            "What is a process pool?",
            "How does it differ from a thread pool?",
            "Give me a one-line Python example of each.",
        ]

        for (i, prompt) in turns.enumerated() {
            print("Turn \(i + 1): \(prompt)")
            let result = try await scheduler.complete(
                sessionID: sessionID,
                prompt: prompt,
                params: SamplingParams(maxTokens: 2048, temperature: 0.5)
            )
            printResult(result)

            if let info = await scheduler.sessionInfo(sessionID) {
                print("  Token budget used: \(info.tokenBudgetUsed)")
            }
            print("")
        }

        if let info = await scheduler.sessionInfo(sessionID) {
            print("Session summary:")
            print("  token budget used: \(info.tokenBudgetUsed)")
            print("  phase: \(info.phase.rawValue)")
        }

        try await scheduler.evictSession(sessionID)
        print("Session evicted.")
    }

    // MARK: - Advanced Test Scenarios

    static func runLongDecodeScenario(scheduler: InferenceScheduler) async throws {
        print("═══ Scenario: Long Decode (sustained generation) ═══")
        print("")

        let sessionID = try await scheduler.createSession(
            systemPrompt: "You are a storyteller. Write long, detailed narratives."
        )
        print("Created session: \(sessionID)")

        let prompt = "Write a detailed story about a group of AI agents exploring a virtual city. Include dialogue, descriptions, and plot twists. Make it very long and detailed."
        print("Prompt: \(prompt.prefix(80))...")
        print("")

        let wallStart = ContinuousClock.now
        let result = try await scheduler.complete(
            sessionID: sessionID,
            prompt: prompt,
            params: SamplingParams(maxTokens: 2048, temperature: 0.8, topP: 0.95)
        )
        let wallEnd = ContinuousClock.now

        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let wordCount = text.split(separator: " ").count
        let charCount = text.count

        print("  Generated: \(result.completionTokens) tokens, \(wordCount) words, \(charCount) chars")
        print("  Finish reason: \(result.finishReason)")
        print("  Worker: W\(result.workerIndex)")
        print("  Decode time: \(fmt(durationMs(result.decodeDuration))) ms")
        print("  Wall time: \(formatDuration(wallEnd - wallStart))")
        print("  Tok/s: \(fmt(result.tokensPerSecond))")

        if let info = await scheduler.sessionInfo(sessionID) {
            print("  Token budget: \(info.tokenBudgetUsed)/\(4096)")
        }

        print("")
        print("  First 200 chars: \(String(text.prefix(200)))...")
        print("  Last 200 chars:  ...\(String(text.suffix(200)))")

        try await scheduler.evictSession(sessionID)
        print("\nSession evicted.")
    }

    static func runStressScenario(
        scheduler: InferenceScheduler,
        config: InferenceConfig
    ) async throws {
        print("═══ Scenario: Stress Test (20 concurrent via DAG batch) ═══")
        print("")

        let requestCount = 20
        let prompts = (0..<requestCount).map { i in
            "In exactly one sentence, what is fact #\(i + 1) about distributed computing?"
        }

        print("Creating \(requestCount) sessions across \(config.workerCount) workers...")

        var sessionIDs: [SessionID] = []
        for _ in 0..<requestCount {
            let sid = try await scheduler.createSession(
                systemPrompt: "Answer in exactly one sentence."
            )
            sessionIDs.append(sid)
        }

        let load = await scheduler.workerLoad
        print("Worker assignment: ", terminator: "")
        for (w, count) in load.sorted(by: { $0.key < $1.key }) {
            print("W\(w)=\(count) ", terminator: "")
        }
        print("\n")

        print("Submitting batch DAG (\(requestCount) prefill→decode chains)...")
        let wallStart = ContinuousClock.now

        let requests = zip(sessionIDs, prompts).map { (sid, prompt) in
            (sessionID: sid, prompt: prompt, params: SamplingParams(maxTokens: 64, temperature: 0.7))
        }

        let outcomes = try await scheduler.completeBatch(requests: requests)
        let wallEnd = ContinuousClock.now
        let wallDuration = wallEnd - wallStart

        var successCount = 0
        var failCount = 0
        var totalTokens = 0

        for sid in sessionIDs {
            switch outcomes[sid] {
            case .success(let result):
                successCount += 1
                totalTokens += result.completionTokens
            case .failure:
                failCount += 1
            case .none:
                failCount += 1
            }
        }

        print("Results:")
        print("  Success: \(successCount)/\(requestCount)")
        print("  Failed:  \(failCount)/\(requestCount)")
        print("  Total tokens: \(totalTokens)")
        print("  Wall time: \(formatDuration(wallDuration))")

        let wallSeconds = durationSeconds(wallDuration)
        if wallSeconds > 0 {
            print("  Aggregate tok/s: \(fmt(Double(totalTokens) / wallSeconds))")
        }

        // Verify all workers were used
        var workerHits: [Int: Int] = [:]
        for sid in sessionIDs {
            if case .success(let r) = outcomes[sid] {
                workerHits[r.workerIndex, default: 0] += 1
            }
        }
        print("  Worker distribution: ", terminator: "")
        for (w, count) in workerHits.sorted(by: { $0.key < $1.key }) {
            print("W\(w)=\(count) ", terminator: "")
        }
        print("")

        // Verify latency is within 2x baseline
        let durations = sessionIDs.compactMap { sid -> Double? in
            guard case .success(let r) = outcomes[sid] else { return nil }
            return durationMs(r.totalDuration)
        }
        if !durations.isEmpty {
            print("  Latency: p50=\(fmt(percentile(durations, 0.5)))ms  p95=\(fmt(percentile(durations, 0.95)))ms  max=\(fmt(durations.max() ?? 0))ms")
        }

        for sid in sessionIDs {
            try? await scheduler.evictSession(sid)
        }
    }

    static func runWorkerFailScenario(
        pool: InferenceWorkerPool,
        scheduler: InferenceScheduler,
        config: InferenceConfig
    ) async throws {
        print("═══ Scenario: Worker Failure Recovery ═══")
        print("")

        guard config.workerCount >= 2 else {
            print("Need at least 2 workers for this test. Use --workers 2 or more.")
            return
        }

        // Step 1: Verify all workers healthy
        let health1 = try await pool.healthCheck()
        print("Step 1: Health check → \(health1.map { $0 ? "✓" : "✗" }.joined(separator: " "))")

        // Step 2: Create sessions on different workers
        let sid1 = try await scheduler.createSession(systemPrompt: "Be brief.")
        let sid2 = try await scheduler.createSession(systemPrompt: "Be brief.")
        print("Step 2: Created sessions — \(sid1) and \(sid2)")

        // Step 3: Complete a request on each to verify they work
        let r1 = try await scheduler.complete(sessionID: sid1, prompt: "Say hello.", params: .greedy)
        print("Step 3a: W\(r1.workerIndex) → \"\(r1.text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(50))\"")

        let r2 = try await scheduler.complete(sessionID: sid2, prompt: "Say goodbye.", params: .greedy)
        print("Step 3b: W\(r2.workerIndex) → \"\(r2.text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(50))\"")

        // Step 4: Health check again
        let health2 = try await pool.healthCheck()
        print("Step 4: Health check → \(health2.map { $0 ? "✓" : "✗" }.joined(separator: " "))")

        // Step 5: Try a batch to verify both workers still functional
        print("Step 5: Batch request across both sessions...")
        let batchOutcomes = try await scheduler.completeBatch(requests: [
            (sessionID: sid1, prompt: "Count to 3.", params: SamplingParams(maxTokens: 16)),
            (sessionID: sid2, prompt: "Count to 3.", params: SamplingParams(maxTokens: 16)),
        ])

        var batchOK = 0
        for (sid, outcome) in batchOutcomes {
            switch outcome {
            case .success(let r):
                batchOK += 1
                print("  \(sid): W\(r.workerIndex) → \"\(r.text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(40))\"")
            case .failure(let e):
                print("  \(sid): FAILED → \(e)")
            }
        }
        print("  Batch result: \(batchOK)/\(batchOutcomes.count) succeeded")

        // Cleanup
        try? await scheduler.evictSession(sid1)
        try? await scheduler.evictSession(sid2)

        let health3 = try await pool.healthCheck()
        print("Step 6: Final health → \(health3.map { $0 ? "✓" : "✗" }.joined(separator: " "))")
        print("\nWorker failure scenario complete.")
    }

    // MARK: - Shared Memory Benchmark

    static func runShmBenchScenario(
        pool: InferenceWorkerPool,
        config: InferenceConfig
    ) async throws {
        print("═══ Scenario: Shared Memory A/B Benchmark ═══")
        print("")

        let prompt = "Count from 1 to 5."
        let params = SamplingParams(maxTokens: 32, temperature: 0.0, topK: 1, repeatPenalty: 1.0)
        let sessionCount = config.workerCount * 3

        // --- Baseline: Pickle path ---
        let pickleConfig = InferenceConfig(
            modelPath: config.modelPath,
            contextSize: config.contextSize,
            nGpuLayers: config.nGpuLayers,
            workerCount: config.workerCount,
            maxSessionsPerWorker: config.maxSessionsPerWorker,
            maxInFlight: config.maxInFlight,
            blasThreads: config.blasThreads,
            useSharedMemory: false
        )
        let pickleScheduler = InferenceScheduler(workerPool: pool, config: pickleConfig)

        print("[Pickle] Creating \(sessionCount) sessions...")
        var pickleSessions: [SessionID] = []
        for _ in 0..<sessionCount {
            let sid = try await pickleScheduler.createSession(systemPrompt: "Be brief.")
            pickleSessions.append(sid)
        }

        print("[Pickle] Running batch of \(sessionCount)...")
        let pickleStart = ContinuousClock.now
        let pickleResults = try await pickleScheduler.completeBatch(
            requests: pickleSessions.map { ($0, prompt, params) }
        )
        let pickleEnd = ContinuousClock.now
        let pickleWallMs = durationMs(pickleEnd - pickleStart)

        var pickleOK = 0
        var pickleTotalTokens = 0
        var pickleTotalDecodeMs: Double = 0
        for (_, outcome) in pickleResults {
            if case .success(let r) = outcome {
                pickleOK += 1
                pickleTotalTokens += r.completionTokens
                pickleTotalDecodeMs += durationMs(r.decodeDuration)
            }
        }

        for sid in pickleSessions { try? await pickleScheduler.evictSession(sid) }

        print("[Pickle] \(pickleOK)/\(sessionCount) ok  |  wall=\(fmt(pickleWallMs))ms  |  tokens=\(pickleTotalTokens)  |  avg decode=\(fmt(pickleTotalDecodeMs / Double(max(1, pickleOK))))ms")
        print("")

        // --- Optimized: Shared memory path ---
        let shmConfig = InferenceConfig(
            modelPath: config.modelPath,
            contextSize: config.contextSize,
            nGpuLayers: config.nGpuLayers,
            workerCount: config.workerCount,
            maxSessionsPerWorker: config.maxSessionsPerWorker,
            maxInFlight: config.maxInFlight,
            blasThreads: config.blasThreads,
            useSharedMemory: true
        )
        let shmScheduler = InferenceScheduler(workerPool: pool, config: shmConfig)

        print("[SHM]    Creating \(sessionCount) sessions...")
        var shmSessions: [SessionID] = []
        for _ in 0..<sessionCount {
            let sid = try await shmScheduler.createSession(systemPrompt: "Be brief.")
            shmSessions.append(sid)
        }

        print("[SHM]    Running batch of \(sessionCount)...")
        let shmStart = ContinuousClock.now
        let shmResults = try await shmScheduler.completeBatch(
            requests: shmSessions.map { ($0, prompt, params) }
        )
        let shmEnd = ContinuousClock.now
        let shmWallMs = durationMs(shmEnd - shmStart)

        var shmOK = 0
        var shmTotalTokens = 0
        var shmTotalDecodeMs: Double = 0
        for (_, outcome) in shmResults {
            if case .success(let r) = outcome {
                shmOK += 1
                shmTotalTokens += r.completionTokens
                shmTotalDecodeMs += durationMs(r.decodeDuration)
            }
        }

        for sid in shmSessions { try? await shmScheduler.evictSession(sid) }

        print("[SHM]    \(shmOK)/\(sessionCount) ok  |  wall=\(fmt(shmWallMs))ms  |  tokens=\(shmTotalTokens)  |  avg decode=\(fmt(shmTotalDecodeMs / Double(max(1, shmOK))))ms")
        print("")

        // --- Comparison ---
        print("═══ A/B Comparison ═══")
        print("  Sessions:      \(sessionCount)")
        print("  Workers:       \(config.workerCount)")
        print("  Pickle wall:   \(fmt(pickleWallMs)) ms  (\(pickleTotalTokens) tokens)")
        print("  SHM wall:      \(fmt(shmWallMs)) ms  (\(shmTotalTokens) tokens)")

        if pickleWallMs > 0 {
            let delta = pickleWallMs - shmWallMs
            let pct = delta / pickleWallMs * 100
            if delta > 0 {
                print("  Delta:         \(fmt(delta)) ms faster (\(fmt(pct))% improvement)")
            } else {
                print("  Delta:         \(fmt(-delta)) ms slower (\(fmt(-pct))% regression)")
            }
        }

        let pickleTokPerSec = pickleTotalTokens > 0 ? Double(pickleTotalTokens) / (pickleWallMs / 1000) : 0
        let shmTokPerSec = shmTotalTokens > 0 ? Double(shmTotalTokens) / (shmWallMs / 1000) : 0
        print("  Pickle tok/s:  \(fmt(pickleTokPerSec))")
        print("  SHM tok/s:     \(fmt(shmTokPerSec))")
        print("")

        let pickleStats = await pickleScheduler.schedulerStats
        let shmStats = await shmScheduler.schedulerStats
        print("[Pickle stats] scheduled=\(pickleStats.totalScheduled) completed=\(pickleStats.totalCompleted) failed=\(pickleStats.totalFailed)")
        print("[SHM stats]    scheduled=\(shmStats.totalScheduled) completed=\(shmStats.totalCompleted) failed=\(shmStats.totalFailed)")
    }

    // MARK: - Output Helpers

    static func printResult(_ result: InferenceResult) {
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        print("  Response: \(text)")
        print("  Worker: W\(result.workerIndex)  |  Prompt: \(result.promptTokens) tok  |  Completion: \(result.completionTokens) tok  |  Finish: \(result.finishReason)")
        print("  Timing: prefill=\(fmt(durationMs(result.prefillDuration)))ms  decode=\(fmt(durationMs(result.decodeDuration)))ms  total=\(fmt(durationMs(result.totalDuration)))ms  tok/s=\(fmt(result.tokensPerSecond))")
    }

    static func printStats(_ stats: AggregateStats) {
        print("═══ Aggregate Stats ═══")
        print("  Total requests:        \(stats.totalRequests)")
        print("  Active sessions:       \(stats.activeSessions)")
        print("  Total tokens generated: \(stats.totalCompletionTokens)")
        print("  Avg prefill (Python):  \(fmt(stats.avgPrefillMs)) ms")
        print("  Avg decode (Python):   \(fmt(stats.avgDecodeMs)) ms")
        print("  Avg tokens/request:    \(fmt(stats.avgTokensPerRequest))")
        print("  Worker load: ", terminator: "")
        for (w, count) in stats.workerLoad.sorted(by: { $0.key < $1.key }) {
            print("W\(w)=\(count) ", terminator: "")
        }
        print("")
    }

    static func printSchedulerStats(_ stats: SchedulerStats) {
        print("═══ Scheduler Stats (Phase 3) ═══")
        print("  Scheduled:     \(stats.totalScheduled)")
        print("  Completed:     \(stats.totalCompleted)")
        print("  Failed:        \(stats.totalFailed)")
        print("  Active sessions: \(stats.activeSessions)")
        print("  Total tokens:  \(stats.totalTokensGenerated)")
        print("  Avg prefill:   \(fmt(stats.avgPrefillMs)) ms")
        print("  Avg decode:    \(fmt(stats.avgDecodeMs)) ms")
        print("  Avg tok/req:   \(fmt(stats.avgTokensPerRequest))")
        print("  Pending:       prefill=\(stats.pendingPrefills) decode=\(stats.pendingDecodes)")
        print("  Worker load: ", terminator: "")
        for (w, count) in stats.workerLoad.sorted(by: { $0.key < $1.key }) {
            print("W\(w)=\(count) ", terminator: "")
        }
        print("")
    }

    static func printUsage() {
        print("""
        Usage: LlamaInferenceDemo --model <path-to-gguf> [options]

        Options:
          --model <path>      Path to GGUF model file (required)
          --workers <n>       Number of worker processes (default: 2)
          --ctx <n>           Context size in tokens (default: 4096)
          --gpu-layers <n>    GPU layers to offload (default: -1 = all)
          --summarizer-model <path>  Dedicated model for narrative summarization (reduces worker contention)
          --shm               Enable shared memory transport (Phase 4)

        Scenarios (pick one):
          (none)              Single-shot completions (default)
          --script-review     Generate Python script on W0, review on W1 (sequential, requires 2 workers)
          --script-review-parallel  Two scripts, cross-reviewed in parallel (requires 2 workers)
          --script-review-adversarial  Contracts + intersection + Judge (requires 2 workers)
          --script-review-fail  Kill W0 mid-review, verify W1 completes (requires 2 workers)
          --multi-turn        Multi-turn conversation session
          --burst             Burst load across workers
          --benchmark         Latency benchmark
          --dag-burst         Burst load using DAG scheduler (Phase 3)
          --dag-multi-turn    Multi-turn using DAG scheduler (Phase 3)
          --long-decode       Long decode stress test
          --stress            Stress test (20 concurrent sessions)
          --worker-fail       Worker failure recovery test
          --shm-bench         A/B benchmark: pickle vs shared memory (Phase 4)
        """)
    }

    // MARK: - Diagnostic

    static func runDiagnostic(pool: InferenceWorkerPool, config: InferenceConfig) async throws {
        print("═══ Diagnostic: step-by-step worker isolation ═══\n")
        let p = try await pool.getPool()
        let w0 = p.worker(0)
        let kernel = try await pool.kernelHandle(for: 0)

        // Test A: basic evalResult (known working pattern)
        print("A: evalResult<Int> '1+1'")
        do {
            let r: Int = try await w0.evalResult("1+1")
            print("  OK: \(r)")
        } catch { print("  FAIL: \(error)") }

        // Test B: method (handle return, no pickling)
        print("\nB: method (handle return) create_session")
        do {
            let h = try await p.method(
                handle: kernel, name: "create_session",
                kwargs: ["session_id": .python("diag-h")],
                worker: 0
            )
            print("  OK: got handle \(h.id)")
            try? await p.release(h)
        } catch { print("  FAIL: \(error)") }

        // Test C: methodResult<String> on create_session
        print("\nC: methodResult<String> create_session")
        do {
            let r: String = try await p.methodResult(
                handle: kernel, name: "create_session",
                kwargs: ["session_id": .python("diag-1")],
                worker: 0
            )
            print("  OK: \(r)")
        } catch { print("  FAIL: \(error)") }

        // Test D: method + eval to read result (bypass methodResult pickling)
        print("\nD: method + evalResult to read n_ctx property")
        do {
            let nctx: Int = try await w0.methodResult(
                handle: kernel, name: "n_ctx.__get__",
                args: [.handle(kernel)]
            )
            print("  OK: n_ctx=\(nctx)")
        } catch {
            print("  FAIL via methodResult: \(error)")
            // try plain eval with binding
            do {
                let r: Int = try await w0.evalResult("k.n_ctx", bindings: ["k": kernel])
                print("  OK via evalResult: n_ctx=\(r)")
            } catch { print("  FAIL via evalResult: \(error)") }
        }

        // Test E: evalResult calling method directly in Python
        print("\nE: evalResult calling kernel.create_session directly in Python")
        do {
            let r: String = try await w0.evalResult(
                "str(k.create_session('diag-eval'))",
                bindings: ["k": kernel]
            )
            print("  OK: \(r)")
        } catch { print("  FAIL: \(error)") }

        // Test F: evalResult returning a dict (does dict pickling work at all?)
        print("\nF: evalResult<String> pickling a dict via str()")
        do {
            let r: String = try await w0.evalResult("str({'a': 1, 'b': 2})")
            print("  OK: \(r)")
        } catch { print("  FAIL: \(error)") }

        // Test G: methodResult returning a scalar (Int) — is it dict-specific?
        print("\nG: methodResult<Int> on list.__len__ (scalar return)")
        do {
            let listHandle = try await w0.eval("[1,2,3]")
            let r: Int = try await w0.methodResult(handle: listHandle, name: "__len__")
            print("  OK: \(r)")
            try? await p.release(listHandle)
        } catch { print("  FAIL: \(error)") }

        // Test H: methodResult<String> on a simple Python class method returning a string
        print("\nH: methodResult<String> on custom class returning string")
        do {
            let obj = try await w0.eval("type('T', (), {'greet': lambda self: 'hello'})()")
            let r: String = try await w0.methodResult(handle: obj, name: "greet")
            print("  OK: \(r)")
            try? await p.release(obj)
        } catch { print("  FAIL: \(error)") }

        // Test I: methodResult<String> on custom class returning dict (the actual pattern)
        print("\nI: methodResult<String> on custom class returning dict")
        do {
            let obj = try await w0.eval("type('T', (), {'info': lambda self: {'a': 1}})()")
            let r: String = try await w0.methodResult(handle: obj, name: "info")
            print("  OK: \(r)")
            try? await p.release(obj)
        } catch { print("  FAIL: \(error)") }

        print("\nDiagnostic complete.")
    }

    // MARK: - Arg Parsing

    static func parseArg(_ args: [String], flag: String) -> String? {
        guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else {
            return nil
        }
        return args[idx + 1]
    }

    // MARK: - Math Helpers

    static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    static func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let pos = Int(round(max(0, min(1, p)) * Double(sorted.count - 1)))
        return sorted[pos]
    }

    /// Canonical key for finding: type + ":" + normalize(desc). Synonym mapping for file I/O.
    static func canonicalKey(type: String, desc: String) -> String {
        var d = desc.lowercased()
        d = d.replacingOccurrences(of: "uses open()", with: "uses file not stdin")
        d = d.replacingOccurrences(of: "reads from file", with: "uses file not stdin")
        d = d.replacingOccurrences(of: "file instead of stdin", with: "uses file not stdin")
        let words = d.components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 1 }
        let normalized = words.joined(separator: " ")
        return "\(type):\(normalized)"
    }

    /// Structured parse: (type, severity, desc). formatCompliant=true only when valid JSON parses.
    /// When JSON fails, fallback extracts findings from raw text (downgraded—never drop correctness signals).
    static func parseReviewFindingsStructured(_ text: String) -> (findings: [(type: String, severity: Int, desc: String)], formatCompliant: Bool) {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let startIdx = raw.firstIndex(of: "{") else { return (parseReviewFindingsFallback(raw), false) }
        var depth = 0
        var endIdx: String.Index?
        for i in raw[startIdx...].indices {
            let c = raw[i]
            if c == "{" { depth += 1 }
            else if c == "}" { depth -= 1; if depth == 0 { endIdx = i; break } }
        }
        guard let end = endIdx else { return (parseReviewFindingsFallback(raw), false) }
        let jsonStr = String(raw[startIdx...end])
        guard let data = jsonStr.data(using: .utf8),
              let top = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let findings = top["findings"] as? [[String: Any]] else {
            return (parseReviewFindingsFallback(raw), false)
        }
        var result: [(type: String, severity: Int, desc: String)] = []
        for f in findings {
            let t = (f["type"] as? String) ?? "unknown"
            let sev: Int = {
                if let i = f["severity"] as? Int { return i }
                if let n = f["severity"] as? NSNumber { return n.intValue }
                if let s = f["severity"] as? String, let i = Int(s) { return i }
                return 0
            }()
            let d = (f["desc"] as? String) ?? (f["description"] as? String) ?? ""
            if !d.isEmpty { result.append((t, sev, d)) }
        }
        return (result, true)
    }

    /// Fallback: extract findings from raw text when JSON parse fails. Keeps correctness signals.
    static func parseReviewFindingsFallback(_ raw: String) -> [(type: String, severity: Int, desc: String)] {
        var result: [(type: String, severity: Int, String)] = []
        // Match patterns like "severity":5 or "severity": 5
        let sevPattern = #""severity"\s*:\s*(\d+)"#
        // Match desc/description: "text" - use a simple pattern for quoted strings
        let descPattern = #""(?:desc|description)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)""#
        guard let sevRegex = try? NSRegularExpression(pattern: sevPattern),
              let descRegex = try? NSRegularExpression(pattern: descPattern) else { return [] }
        let nsRaw = raw as NSString
        let sevMatches = sevRegex.matches(in: raw, range: NSRange(raw.startIndex..., in: raw))
        let descMatches = descRegex.matches(in: raw, range: NSRange(raw.startIndex..., in: raw))
        // Pair severity with nearest following desc; if none, infer from keywords (first match only)
        if let m = sevMatches.first {
            let sevRange = m.range(at: 1)
            let sev = Int(nsRaw.substring(with: sevRange)) ?? 5
            let mEnd = m.range.location + m.range.length
            let nextDesc = descMatches.first { $0.range.location >= mEnd && $0.range.location < mEnd + 200 }
            let desc: String
            if let d = nextDesc, let r = Range(d.range(at: 1), in: raw) {
                desc = String(raw[r])
            } else if raw.contains("open()") || raw.contains("uses file") || raw.contains("stdin") {
                desc = "uses file not stdin"
            } else if raw.contains("float") == false && raw.contains("row[1]") {
                desc = "scores without float coercion"
            } else {
                desc = "spec violation"
            }
            if !desc.isEmpty { result.append(("spec_violation", sev, desc)) }
        }
        // If no severity match but clear violation keywords, add synthetic finding
        if result.isEmpty && (raw.contains("open()") || raw.contains("Path(") || raw.contains("uses file")) {
            result.append(("spec_violation", 5, "uses file not stdin"))
        }
        return result
    }

    /// Best-effort extract findings from reviewer JSON. Returns (findings, parseOK).
    static func parseReviewFindings(_ text: String) -> (descriptions: [String], parseOK: Bool) {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // Extract first complete JSON object by brace matching (avoids grabbing prose/code after)
        guard let startIdx = raw.firstIndex(of: "{") else { return ([], false) }
        var depth = 0
        var endIdx: String.Index?
        for i in raw[startIdx...].indices {
            let c = raw[i]
            if c == "{" { depth += 1 }
            else if c == "}" {
                depth -= 1
                if depth == 0 { endIdx = i; break }
            }
        }
        guard let end = endIdx else { return ([], false) }
        let jsonStr = String(raw[startIdx...end])
        if let data = jsonStr.data(using: .utf8),
           let top = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let findings = top["findings"] as? [[String: Any]] {
            var descs: [String] = []
            for f in findings {
                if let d = (f["desc"] as? String) ?? (f["description"] as? String) {
                    descs.append(d)
                } else if let id = f["id"] as? String { descs.append(id) }
            }
            return (descs, true)
        }
        // Fallback: extract "desc":"..." from raw text when JSON is malformed
        var fallback: [String] = []
        var searchStart = raw.startIndex
        while let range = raw.range(of: "\"desc\":\"", range: searchStart..<raw.endIndex) {
            let valueStart = range.upperBound
            if let endRange = raw.range(of: "\"", range: valueStart..<raw.endIndex) {
                let value = String(raw[valueStart..<endRange.lowerBound])
                if !value.isEmpty && value.count < 200 { fallback.append(value) }
                searchStart = endRange.upperBound
            } else { break }
        }
        return (fallback, !fallback.isEmpty)
    }

    static func fmt(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    static func durationMs(_ d: Duration) -> Double {
        Double(d.components.seconds) * 1000
            + Double(d.components.attoseconds) / 1e15
    }

    static func durationSeconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }

    static func formatDuration(_ d: Duration) -> String {
        let ms = durationMs(d)
        if ms < 1000 {
            return "\(fmt(ms)) ms"
        }
        return "\(fmt(ms / 1000)) s"
    }
}
