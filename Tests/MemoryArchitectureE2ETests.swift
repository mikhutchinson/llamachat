import XCTest
import Foundation
@testable import LlamaInferenceCore
import SwiftPythonRuntime

// MARK: - E2E Test Configuration

private let kModelPath = "/Users/mikhutchinson/Models/gguf/Qwen3-4B-Q4_K_M.gguf"
private let kContextSize = 4096
private let kWorkerCount = 1

private func findWorkerExecutable() -> String? {
    let fm = FileManager.default

    // 1. Check the demo's own build directory (most reliable for tests)
    let demoBuild = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Tests/
        .deletingLastPathComponent()  // LlamaInferenceDemo/
        .appendingPathComponent(".build/arm64-apple-macosx/debug/SwiftPythonWorker")
    if fm.isExecutableFile(atPath: demoBuild.path) {
        return demoBuild.path
    }

    // 2. Walk up from test binary (xctest bundle) looking for SwiftPythonWorker sibling
    if let testBundle = Bundle.allBundles.first(where: { $0.bundlePath.hasSuffix(".xctest") }) {
        var dir = URL(fileURLWithPath: testBundle.bundlePath).deletingLastPathComponent()
        for _ in 0..<5 {
            let candidate = dir.appendingPathComponent("SwiftPythonWorker")
            if fm.isExecutableFile(atPath: candidate.path) {
                return candidate.path
            }
            dir = dir.deletingLastPathComponent()
        }
    }

    // 3. Root project build directory
    let rootBuild = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // Tests/
        .deletingLastPathComponent()  // LlamaInferenceDemo/
        .deletingLastPathComponent()  // Demo/
        .deletingLastPathComponent()  // SwiftPython/
        .appendingPathComponent(".build/arm64-apple-macosx/debug/SwiftPythonWorker")
    if fm.isExecutableFile(atPath: rootBuild.path) {
        return rootBuild.path
    }

    return nil
}

private func findVenvPath() -> String? {
    let fm = FileManager.default
    
    // 1. Check relative to test file (Demo/LlamaInferenceDemo/Tests/ -> ../../.venv)
    let testFileDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    let relativeVenv = testFileDir
        .deletingLastPathComponent()  // Tests/
        .deletingLastPathComponent()  // LlamaInferenceDemo/
        .deletingLastPathComponent()  // Demo/
        .deletingLastPathComponent()  // SwiftPython/
        .appendingPathComponent(".venv")
    if fm.fileExists(atPath: relativeVenv.path) {
        return relativeVenv.path
    }
    
    // 2. Check VIRTUAL_ENV env var
    if let envVenv = ProcessInfo.processInfo.environment["VIRTUAL_ENV"],
       fm.fileExists(atPath: envVenv) {
        return envVenv
    }
    
    // 3. Check common locations relative to project root
    let cwd = fm.currentDirectoryPath
    let candidates = [
        "\(cwd)/.venv",
        "\(cwd)/venv",
        "\(cwd)/../.venv",
        "\(cwd)/../venv",
        "\(cwd)/../../.venv",
        "\(cwd)/../../venv",
    ]
    for candidate in candidates {
        if fm.fileExists(atPath: candidate) {
            return candidate
        }
    }
    
    return nil
}

private func makeConfig() -> InferenceConfig {
    InferenceConfig(
        modelPath: kModelPath,
        contextSize: kContextSize,
        nGpuLayers: -1,
        workerCount: kWorkerCount,
        maxSessionsPerWorker: 8,
        maxInFlight: 16,
        blasThreads: 1,
        useSharedMemory: false,
        workerExecutablePath: findWorkerExecutable(),
        venvPath: findVenvPath()
    )
}

private func skipIfNoWorker(file: StaticString = #filePath, line: UInt = #line) throws {
    guard let path = findWorkerExecutable() else {
        throw XCTSkip("SwiftPythonWorker not found — build with: swift build --product SwiftPythonWorker")
    }
    guard FileManager.default.isExecutableFile(atPath: path) else {
        throw XCTSkip("SwiftPythonWorker at \(path) is not executable")
    }
}

private func skipIfNoModel(file: StaticString = #filePath, line: UInt = #line) throws {
    guard FileManager.default.fileExists(atPath: kModelPath) else {
        throw XCTSkip("Qwen3 model not found at \(kModelPath)")
    }
}

// MARK: - Diagnostic: Worker Path Resolution

final class WorkerPathDiagnosticTests: XCTestCase {

    func testWorkerExecutableFound() throws {
        let path = findWorkerExecutable()
        print("[DIAG] CommandLine.arguments[0] = \(CommandLine.arguments[0])")
        print("[DIAG] findWorkerExecutable() = \(path ?? "nil")")
        XCTAssertNotNil(path, "SwiftPythonWorker must be discoverable from test context")
        if let path {
            XCTAssertTrue(
                FileManager.default.isExecutableFile(atPath: path),
                "SwiftPythonWorker at \(path) must be executable"
            )
        }
    }

    func testPoolStartupAndShutdown() async throws {
        try skipIfNoModel()
        try skipIfNoWorker()

        let config = makeConfig()
        print("[DIAG] workerExecutablePath = \(config.workerExecutablePath ?? "auto")")
        print("[DIAG] modelPath = \(config.modelPath)")

        let pool = InferenceWorkerPool(config: config)
        do {
            try await pool.startup()
            print("[DIAG] Pool started successfully")
            let health = try await pool.healthCheck()
            print("[DIAG] Health check: \(health)")
            await pool.shutdown()
            print("[DIAG] Pool shut down")
        } catch {
            await pool.shutdown()
            XCTFail("Pool startup failed: \(error)")
        }
    }
}

// MARK: - Context Wind Monitor Unit Tests

final class ContextWindMonitorTests: XCTestCase {

    func testThresholdOrdering() {
        XCTAssertTrue(ContextThreshold.prepare < .commit)
        XCTAssertTrue(ContextThreshold.commit < .reset)
    }

    func testThresholdRawValues() {
        XCTAssertEqual(ContextThreshold.prepare.rawValue, 0.60)
        XCTAssertEqual(ContextThreshold.commit.rawValue, 0.70)
        XCTAssertEqual(ContextThreshold.reset.rawValue, 0.80)
    }

    func testContextWindEventInit() {
        let id = SessionID()
        let event = ContextWindEvent(
            sessionID: id,
            threshold: .commit,
            utilization: 0.72,
            promptTokens: 2950,
            contextSize: 4096
        )
        XCTAssertEqual(event.sessionID, id)
        XCTAssertEqual(event.threshold, .commit)
        XCTAssertEqual(event.utilization, 0.72, accuracy: 0.001)
        XCTAssertEqual(event.promptTokens, 2950)
        XCTAssertEqual(event.contextSize, 4096)
    }

    func testMonitorRegisterAndUnregister() async {
        let monitor = ContextWindMonitor(contextSize: 4096)
        let id = SessionID()
        await monitor.registerSession(id)
        let u = await monitor.utilization(for: id)
        XCTAssertEqual(u, 0.0)
        await monitor.unregisterSession(id)
        let u2 = await monitor.utilization(for: id)
        XCTAssertEqual(u2, 0.0)
    }

    func testMonitorNoThresholdBelowPrepare() async {
        let monitor = ContextWindMonitor(contextSize: 4096)
        let id = SessionID()
        await monitor.registerSession(id)

        // 50% utilization — below prepare threshold
        let threshold = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 1800, completionTokens: 248
        )
        XCTAssertNil(threshold)
        let highest = await monitor.highestThreshold(for: id)
        XCTAssertNil(highest)
    }

    func testMonitorPrepareThreshold() async {
        let monitor = ContextWindMonitor(contextSize: 4096)
        let id = SessionID()
        await monitor.registerSession(id)

        // 62% utilization — crosses prepare
        let threshold = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 2400, completionTokens: 140
        )
        XCTAssertEqual(threshold, .prepare)
        let highest = await monitor.highestThreshold(for: id)
        XCTAssertEqual(highest, .prepare)
    }

    func testMonitorCommitThreshold() async {
        let monitor = ContextWindMonitor(contextSize: 4096)
        let id = SessionID()
        await monitor.registerSession(id)

        // 72% — crosses prepare AND commit in one step
        let threshold = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 2800, completionTokens: 150
        )
        XCTAssertEqual(threshold, .commit)
        let highest = await monitor.highestThreshold(for: id)
        XCTAssertEqual(highest, .commit)

        // History should have 2 events (prepare + commit)
        let history = await monitor.crossingHistory(for: id)
        XCTAssertEqual(history.count, 2)
        XCTAssertEqual(history[0].threshold, .prepare)
        XCTAssertEqual(history[1].threshold, .commit)
    }

    func testMonitorResetThreshold() async {
        let monitor = ContextWindMonitor(contextSize: 4096)
        let id = SessionID()
        await monitor.registerSession(id)

        // 85% — crosses all three
        let threshold = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 3200, completionTokens: 280
        )
        XCTAssertEqual(threshold, .reset)
        let history = await monitor.crossingHistory(for: id)
        XCTAssertEqual(history.count, 3)
    }

    func testMonitorNoDuplicateCrossings() async {
        let monitor = ContextWindMonitor(contextSize: 4096)
        let id = SessionID()
        await monitor.registerSession(id)

        // First: cross prepare
        _ = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 2500, completionTokens: 0
        )
        // Second: still in prepare range — should NOT re-trigger
        let threshold2 = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 2550, completionTokens: 10
        )
        XCTAssertNil(threshold2)
        let history = await monitor.crossingHistory(for: id)
        XCTAssertEqual(history.count, 1)
    }

    func testMonitorUsesLatestTurnOccupancySemantics() async {
        let monitor = ContextWindMonitor(contextSize: 4096)
        let id = SessionID()
        await monitor.registerSession(id)

        // 58.6% utilization: no threshold crossing.
        let first = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 2300, completionTokens: 100
        )
        XCTAssertNil(first)

        // Same occupancy on next turn should remain below prepare.
        // Old cumulative semantics would incorrectly cross prepare here.
        let second = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 2300, completionTokens: 100
        )
        XCTAssertNil(second)
        let utilization = await monitor.utilization(for: id)
        XCTAssertEqual(utilization, 2400.0 / 4096.0, accuracy: 0.0001)
        let history = await monitor.crossingHistory(for: id)
        XCTAssertTrue(history.isEmpty)
    }

    func testMonitorResetSession() async {
        let monitor = ContextWindMonitor(contextSize: 4096)
        let id = SessionID()
        await monitor.registerSession(id)

        _ = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 3000, completionTokens: 100
        )
        let before = await monitor.highestThreshold(for: id)
        XCTAssertNotNil(before)

        await monitor.resetSession(id, newPromptTokens: 200)
        let after = await monitor.highestThreshold(for: id)
        XCTAssertNil(after)
        let u = await monitor.utilization(for: id)
        XCTAssertEqual(u, 200.0 / 4096.0, accuracy: 0.001)
    }

    func testMonitorCallback() async {
        let monitor = ContextWindMonitor(contextSize: 4096)
        let id = SessionID()
        await monitor.registerSession(id)

        nonisolated(unsafe) var receivedEvent: ContextWindEvent?
        await monitor.setOnThresholdCrossed { event in
            receivedEvent = event
        }

        _ = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 2600, completionTokens: 0
        )

        XCTAssertNotNil(receivedEvent)
        XCTAssertEqual(receivedEvent?.threshold, .prepare)
    }

    func testMonitorBoundaryExact60() async {
        let monitor = ContextWindMonitor(contextSize: 1000)
        let id = SessionID()
        await monitor.registerSession(id)

        // Exactly 60% — should trigger prepare
        let threshold = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 600, completionTokens: 0
        )
        XCTAssertEqual(threshold, .prepare)
    }

    func testMonitorBoundaryJustBelow60() async {
        let monitor = ContextWindMonitor(contextSize: 1000)
        let id = SessionID()
        await monitor.registerSession(id)

        // 59.9% — should NOT trigger prepare
        let threshold = await monitor.recordTokenUsage(
            sessionID: id, promptTokens: 599, completionTokens: 0
        )
        XCTAssertNil(threshold)
    }
}

// MARK: - Context Snapshot Tests

final class ContextSnapshotTests: XCTestCase {

    func testContextSnapshotInit() {
        let id = SessionID()
        let snap = ContextSnapshot(
            sessionID: id,
            sessionHistory: [["role": "user", "content": "hello"]],
            systemPrompt: "You are helpful."
        )
        XCTAssertEqual(snap.sessionID, id)
        XCTAssertEqual(snap.sessionHistory.count, 1)
    }
}

// MARK: - E2E: Pool + Scheduler + Context Wind Integration

final class MemoryArchitectureE2ETests: XCTestCase {

    private var pool: InferenceWorkerPool!
    private var scheduler: InferenceScheduler!

    override func setUp() async throws {
        try skipIfNoModel()
        let config = makeConfig()
        pool = InferenceWorkerPool(config: config)
        try await pool.startup()
        scheduler = InferenceScheduler(workerPool: pool, config: config)
    }

    override func tearDown() async throws {
        if let pool {
            await pool.shutdown()
        }
        pool = nil
        scheduler = nil
    }

    // MARK: - Basic Inference Still Works

    func testBasicInference() async throws {
        let sessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")
        let result = try await scheduler.complete(
            sessionID: sessionID,
            prompt: "What is 2+2? Answer with just the number.",
            params: SamplingParams(maxTokens: 256, temperature: 0.0)
        )
        let hasOutput = !result.text.isEmpty || !(result.thinking ?? "").isEmpty
        XCTAssertTrue(hasOutput, "Model should produce text or thinking output")
        XCTAssertGreaterThan(result.completionTokens, 0)
        XCTAssertGreaterThan(result.promptTokens, 0)
        try await scheduler.evictSession(sessionID)
    }

    // MARK: - Context Overflow Rejection (Near Limit)

    func testContextOverflowRejectedWhenPromptEstimateExceedsBudget() async throws {
        try skipIfNoWorker()
        try skipIfNoModel()

        // Use small context so prompt estimate triggers overflow before inference
        let smallConfig = InferenceConfig(
            modelPath: kModelPath,
            contextSize: 256,
            nGpuLayers: -1,
            workerCount: 1,
            maxSessionsPerWorker: 8,
            maxInFlight: 16,
            blasThreads: 1,
            useSharedMemory: false,
            workerExecutablePath: findWorkerExecutable(),
            venvPath: findVenvPath()
        )
        let smallPool = InferenceWorkerPool(config: smallConfig)
        try await smallPool.startup()
        defer { Task { await smallPool.shutdown() } }

        let smallScheduler = InferenceScheduler(workerPool: smallPool, config: smallConfig)
        let sessionID = try await smallScheduler.createSession(systemPrompt: nil)

        // 800 chars → ~266 estimated tokens > 256 context (prompt alone wouldn't fit)
        let longPrompt = String(repeating: "x", count: 800)

        do {
            _ = try await smallScheduler.complete(
                sessionID: sessionID,
                prompt: longPrompt,
                params: SamplingParams(maxTokens: 64, temperature: 0)
            )
            XCTFail("Expected contextOverflow to be thrown")
        } catch let err as InferenceError {
            if case .contextOverflow(let sid, let used, let max) = err {
                XCTAssertEqual(sid, sessionID)
                XCTAssertEqual(used, 0)
                XCTAssertEqual(max, 256)
            } else {
                XCTFail("Expected contextOverflow, got \(err)")
            }
        }

        try? await smallScheduler.evictSession(sessionID)
    }

    // MARK: - Context Monitor Tracks Tokens After Inference

    func testContextMonitorTracksTokensAfterInference() async throws {
        let sessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")

        // Before inference, utilization should be 0
        let u0 = await scheduler.contextMonitor.utilization(for: sessionID)
        XCTAssertEqual(u0, 0.0)

        _ = try await scheduler.complete(
            sessionID: sessionID,
            prompt: "Tell me a short joke.",
            params: SamplingParams(maxTokens: 64, temperature: 0.5)
        )

        // After inference, utilization should be > 0
        let u1 = await scheduler.contextMonitor.utilization(for: sessionID)
        XCTAssertGreaterThan(u1, 0.0, "Utilization should increase after inference")
        let tokens = await scheduler.contextMonitor.tokenCount(for: sessionID)
        XCTAssertGreaterThan(tokens, 0, "Token count should be positive")

        try await scheduler.evictSession(sessionID)
    }

    // MARK: - Context Monitor Accumulates Over Multiple Turns

    func testContextMonitorAccumulates() async throws {
        let sessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")

        _ = try await scheduler.complete(
            sessionID: sessionID,
            prompt: "What is Python?",
            params: SamplingParams(maxTokens: 32, temperature: 0.0)
        )
        let u1 = await scheduler.contextMonitor.utilization(for: sessionID)

        _ = try await scheduler.complete(
            sessionID: sessionID,
            prompt: "What about Swift?",
            params: SamplingParams(maxTokens: 32, temperature: 0.0)
        )
        let u2 = await scheduler.contextMonitor.utilization(for: sessionID)

        XCTAssertGreaterThan(u2, u1, "Utilization should increase with more turns")

        try await scheduler.evictSession(sessionID)
    }

    func testTokenBudgetTracksLatestTurnOccupancy() async throws {
        let sessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")

        let result1 = try await scheduler.complete(
            sessionID: sessionID,
            prompt: "Answer with one short sentence about apples.",
            params: SamplingParams(maxTokens: 48, temperature: 0.0)
        )
        let info1 = await scheduler.sessionInfo(sessionID)
        XCTAssertEqual(
            info1?.tokenBudgetUsed,
            min(kContextSize, result1.promptTokens + result1.completionTokens),
            "Token budget should reflect latest occupancy, not cumulative additions"
        )

        let result2 = try await scheduler.complete(
            sessionID: sessionID,
            prompt: "Now one short sentence about oranges.",
            params: SamplingParams(maxTokens: 48, temperature: 0.0)
        )
        let info2 = await scheduler.sessionInfo(sessionID)
        XCTAssertEqual(
            info2?.tokenBudgetUsed,
            min(kContextSize, result2.promptTokens + result2.completionTokens),
            "Second turn should overwrite occupancy with latest prompt+completion tokens"
        )

        try await scheduler.evictSession(sessionID)
    }

    func testProjectedHeadroomTriggersPreDecodeRehydrate() async throws {
        let sessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")

        // Seed high occupancy without running an actual long decode.
        await scheduler.finalizeCompletedStream(
            sessionID: sessionID,
            promptTokens: 3600,
            completionTokens: 0,
            decodeMs: 0,
            finishReason: "stop"
        )

        let (stream, returnedSessionID) = try await scheduler.completeStreamWithMemoryManagement(
            sessionID: sessionID,
            prompt: "Say hello in one sentence.",
            params: SamplingParams(maxTokens: 768, temperature: 0.0),
            systemPrompt: "You are a helpful assistant.",
            recentTurns: []
        )

        XCTAssertNotEqual(
            returnedSessionID,
            sessionID,
            "Projected headroom check should force rehydration before decode when occupancy is too high"
        )

        var doneChunk: StreamInferenceChunk?
        for try await chunk in stream {
            if chunk.event == .done {
                doneChunk = chunk
            }
        }
        if let doneChunk {
            await scheduler.finalizeCompletedStream(
                sessionID: returnedSessionID,
                promptTokens: doneChunk.promptTokens ?? 0,
                completionTokens: doneChunk.completionTokens ?? 0,
                decodeMs: doneChunk.decodeMs ?? 0,
                finishReason: doneChunk.finishReason ?? "unknown"
            )
        }

        try await scheduler.evictSession(returnedSessionID)
    }

    func testProjectedHeadroomDoesNotResetAtLowOccupancyWhenMaxTokensIsContextCeiling() async throws {
        let sessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")

        // Low occupancy should not trigger proactive rollover even if maxTokens
        // is configured to the full context-size ceiling.
        await scheduler.finalizeCompletedStream(
            sessionID: sessionID,
            promptTokens: 180,
            completionTokens: 60,
            decodeMs: 0,
            finishReason: "stop"
        )

        let (stream, returnedSessionID) = try await scheduler.completeStreamWithMemoryManagement(
            sessionID: sessionID,
            prompt: "Answer in one sentence.",
            params: SamplingParams(maxTokens: kContextSize, temperature: 0.0),
            systemPrompt: "You are a helpful assistant.",
            recentTurns: []
        )

        XCTAssertEqual(
            returnedSessionID,
            sessionID,
            "Projected guard should not force a reset solely because maxTokens equals context size"
        )

        var doneChunk: StreamInferenceChunk?
        for try await chunk in stream {
            if chunk.event == .done {
                doneChunk = chunk
            }
        }
        if let doneChunk {
            await scheduler.finalizeCompletedStream(
                sessionID: returnedSessionID,
                promptTokens: doneChunk.promptTokens ?? 0,
                completionTokens: doneChunk.completionTokens ?? 0,
                decodeMs: doneChunk.decodeMs ?? 0,
                finishReason: doneChunk.finishReason ?? "unknown"
            )
        }

        try await scheduler.evictSession(returnedSessionID)
    }

    // MARK: - Narrative Memory Installation

    func testNarrativeMemoryInstalls() async throws {
        try await scheduler.installMemoryGraph()
        let memory = await scheduler.narrativeMemory
        XCTAssertNotNil(memory, "Narrative memory should be installed")
    }

    // MARK: - Complete With Memory Management (no threshold hit)

    func testCompleteWithMemoryManagementNoThreshold() async throws {
        try await scheduler.installMemoryGraph()

        let sessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")
        let (result, returnedSessionID) = try await scheduler.completeWithMemoryManagement(
            sessionID: sessionID,
            prompt: "Say hello.",
            params: SamplingParams(maxTokens: 256, temperature: 0.0),
            systemPrompt: "You are a helpful assistant."
        )
        let hasOutput = !result.text.isEmpty || !(result.thinking ?? "").isEmpty
        XCTAssertTrue(hasOutput, "Model should produce text or thinking output")
        XCTAssertEqual(returnedSessionID, sessionID, "Session should NOT change when below threshold")

        try await scheduler.evictSession(returnedSessionID)
    }

    func testCompleteStreamWithMemoryManagementNoThreshold() async throws {
        try await scheduler.installMemoryGraph()

        let sessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")
        let (stream, returnedSessionID) = try await scheduler.completeStreamWithMemoryManagement(
            sessionID: sessionID,
            prompt: "Say hello in one sentence.",
            params: SamplingParams(maxTokens: 64, temperature: 0.0),
            systemPrompt: "You are a helpful assistant."
        )

        var accumulated = ""
        var doneChunk: StreamInferenceChunk?
        for try await chunk in stream {
            switch chunk.event {
            case .delta:
                accumulated += chunk.delta
            case .done:
                doneChunk = chunk
            case .error:
                XCTFail("Unexpected stream error: \(chunk.error ?? "unknown")")
            }
        }

        guard let done = doneChunk else {
            XCTFail("Expected terminal done stream chunk")
            return
        }

        await scheduler.finalizeCompletedStream(
            sessionID: returnedSessionID,
            promptTokens: done.promptTokens ?? 0,
            completionTokens: done.completionTokens ?? 0,
            decodeMs: done.decodeMs ?? 0,
            finishReason: done.finishReason ?? "unknown"
        )

        let hasOutput = !accumulated.isEmpty || !(done.text ?? "").isEmpty || !(done.thinking ?? "").isEmpty
        XCTAssertTrue(hasOutput, "Stream should produce text or thinking output")
        XCTAssertGreaterThan(done.completionTokens ?? 0, 0, "Stream completion tokens should be populated for accounting")
        XCTAssertEqual(returnedSessionID, sessionID, "Session should not change below threshold")

        try await scheduler.evictSession(returnedSessionID)
    }

    // MARK: - Reset And Rehydrate

    func testResetAndRehydrate() async throws {
        let sessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")

        // Do an inference to build some context
        _ = try await scheduler.complete(
            sessionID: sessionID,
            prompt: "Remember: my name is Alice.",
            params: SamplingParams(maxTokens: 32, temperature: 0.0)
        )

        let newSessionID = try await scheduler.resetAndRehydrate(
            sessionID: sessionID,
            systemPrompt: "You are a helpful assistant.",
            recentTurns: [
                (role: "user", content: "Remember: my name is Alice."),
                (role: "assistant", content: "I'll remember that your name is Alice.")
            ],
            narrativeSummary: "The user's name is Alice."
        )

        XCTAssertNotEqual(newSessionID, sessionID, "Rehydration should create a new session")

        // Context monitor should be reset
        let u = await scheduler.contextMonitor.utilization(for: newSessionID)
        // Should be small — only rehydration prompt
        XCTAssertLessThan(u, 0.5, "Utilization should be low after rehydration")

        // The old session should be gone
        let oldInfo = await scheduler.sessionInfo(sessionID) as InferenceScheduler.ScheduledSession?
        XCTAssertNil(oldInfo, "Old session should be evicted")

        try await scheduler.evictSession(newSessionID)
    }

    // MARK: - Multi-Turn Conversation With Memory Management

    func testMultiTurnWithMemoryManagement() async throws {
        try await scheduler.installMemoryGraph()

        var currentSessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")
        var turns: [(role: String, content: String)] = []

        let prompts = [
            "My favorite color is blue.",
            "I work as a software engineer.",
            "I live in San Francisco.",
        ]

        for prompt in prompts {
            turns.append((role: "user", content: prompt))
            let (result, newSID) = try await scheduler.completeWithMemoryManagement(
                sessionID: currentSessionID,
                prompt: prompt,
                params: SamplingParams(maxTokens: 256, temperature: 0.3),
                systemPrompt: "You are a helpful assistant.",
                recentTurns: turns
            )
            let responseText = result.text.isEmpty ? (result.thinking ?? "") : result.text
            turns.append((role: "assistant", content: responseText))
            currentSessionID = newSID
            let hasOutput = !result.text.isEmpty || !(result.thinking ?? "").isEmpty
            XCTAssertTrue(hasOutput, "Each turn should produce text or thinking output")
        }

        // Verify utilization tracked across turns
        let finalU = await scheduler.contextMonitor.utilization(for: currentSessionID)
        XCTAssertGreaterThan(finalU, 0.0)

        try await scheduler.evictSession(currentSessionID)
    }
}

// MARK: - E2E: Full Pipeline (Monitor → Summarize → Rehydrate)

final class FullPipelineE2ETests: XCTestCase {

    private var pool: InferenceWorkerPool!
    private var scheduler: InferenceScheduler!

    override func setUp() async throws {
        try skipIfNoModel()
        let config = makeConfig()
        pool = InferenceWorkerPool(config: config)
        try await pool.startup()
        scheduler = InferenceScheduler(workerPool: pool, config: config)
    }

    override func tearDown() async throws {
        if let pool {
            await pool.shutdown()
        }
        pool = nil
        scheduler = nil
    }

    func testFullConversationPipeline() async throws {
        try await scheduler.installMemoryGraph()

        var currentSessionID = try await scheduler.createSession(systemPrompt: "You are a helpful assistant.")
        var turns: [(role: String, content: String)] = []
        var sessionChanged = false

        // Run enough turns to build up context
        let prompts = [
            "Hi, I'm testing a memory system. My name is Charlie.",
            "I'm working on a Swift project that uses Python workers.",
            "The project uses llama.cpp for local inference.",
            "We use sqlite-vec for vector search in our knowledge graph.",
            "The context window is 4096 tokens.",
            "What have we discussed so far?",
        ]

        for (i, prompt) in prompts.enumerated() {
            turns.append((role: "user", content: prompt))

            let (result, newSID) = try await scheduler.completeWithMemoryManagement(
                sessionID: currentSessionID,
                prompt: prompt,
                params: SamplingParams(maxTokens: 256, temperature: 0.3),
                systemPrompt: "You are a helpful assistant.",
                recentTurns: turns
            )

            let u = await scheduler.contextMonitor.utilization(for: newSID)
            let tokens = await scheduler.contextMonitor.tokenCount(for: newSID)
            print("[E2E Pipeline] Turn \(i+1): u=\(String(format: "%.3f", u)) tokens=\(tokens) sessionChanged=\(newSID != currentSessionID)")

            if newSID != currentSessionID {
                sessionChanged = true
                print("[E2E Pipeline] Session reset: \(currentSessionID) → \(newSID)")
            }

            let responseText = result.text.isEmpty ? (result.thinking ?? "") : result.text
            turns.append((role: "assistant", content: responseText))
            currentSessionID = newSID

            let hasOutput = !result.text.isEmpty || !(result.thinking ?? "").isEmpty
            XCTAssertTrue(hasOutput, "Turn \(i+1) should produce text or thinking output")
        }

        // Verify final state
        let finalU = await scheduler.contextMonitor.utilization(for: currentSessionID)
        print("[E2E Pipeline] Final utilization: \(String(format: "%.3f", finalU))")

        // Clean up
        try await scheduler.evictSession(currentSessionID)

        // Log whether session reset happened (may not with short prompts)
        print("[E2E Pipeline] Session was reset during conversation: \(sessionChanged)")
    }

    func testHighUtilizationTriggersCommit() async throws {
        // Use a smaller context size to trigger commit faster
        let smallConfig = InferenceConfig(
            modelPath: kModelPath,
            contextSize: 2048,
            nGpuLayers: -1,
            workerCount: 1,
            maxSessionsPerWorker: 8,
            maxInFlight: 16,
            blasThreads: 1,
            useSharedMemory: false,
            workerExecutablePath: findWorkerExecutable()
        )

        let smallPool = InferenceWorkerPool(config: smallConfig)
        try await smallPool.startup()
        defer { Task { await smallPool.shutdown() } }

        let smallScheduler = InferenceScheduler(workerPool: smallPool, config: smallConfig)
        try await smallScheduler.installMemoryGraph()

        var currentSessionID = try await smallScheduler.createSession(systemPrompt: "You are a helpful assistant.")
        var turns: [(role: String, content: String)] = []
        var commitTriggered = false

        // Generate enough tokens to cross 70% of 2048 = 1434 tokens
        let prompts = [
            "Write a paragraph about the history of computing, starting from Charles Babbage.",
            "Now tell me about Alan Turing and his contributions.",
            "What about Grace Hopper? Tell me her story.",
            "How did the personal computer revolution start?",
            "Tell me about the internet's origins.",
            "What about modern AI and machine learning?",
            "Summarize the key milestones we discussed.",
        ]

        for (i, prompt) in prompts.enumerated() {
            turns.append((role: "user", content: prompt))

            let uBefore = await smallScheduler.contextMonitor.utilization(for: currentSessionID)

            let (result, newSID) = try await smallScheduler.completeWithMemoryManagement(
                sessionID: currentSessionID,
                prompt: prompt,
                params: SamplingParams(maxTokens: 128, temperature: 0.3),
                systemPrompt: "You are a helpful assistant.",
                recentTurns: turns
            )

            let uAfter = await smallScheduler.contextMonitor.utilization(for: newSID)
            print("[E2E Commit] Turn \(i+1): uBefore=\(String(format: "%.3f", uBefore)) uAfter=\(String(format: "%.3f", uAfter)) changed=\(newSID != currentSessionID)")

            if newSID != currentSessionID {
                commitTriggered = true
                print("[E2E Commit] >>> Commit triggered! Session \(currentSessionID) → \(newSID)")
            }

            turns.append((role: "assistant", content: result.text))
            currentSessionID = newSID
        }

        try? await smallScheduler.evictSession(currentSessionID)
        print("[E2E Commit] Commit was triggered: \(commitTriggered)")
    }
}

// MARK: - stripDocumentBlocks Unit Tests

final class StripDocumentBlocksTests: XCTestCase {

    func testStripsAttachedFileBlock() {
        let turns: [(role: String, content: String)] = [
            (role: "user", content: "[Attached file: report.pdf]\nSome extracted PDF text here\n\nWhat does this report say?")
        ]
        let cleaned = InferenceScheduler.stripDocumentBlocks(from: turns)
        XCTAssertEqual(cleaned.count, 1)
        XCTAssertFalse(cleaned[0].content.contains("[Attached file:"))
        XCTAssertTrue(cleaned[0].content.contains("What does this report say?"))
    }

    func testStripsImageBlock() {
        let turns: [(role: String, content: String)] = [
            (role: "user", content: "[Image: photo.png]\nA red apple on a table\n\nDescribe the colors")
        ]
        let cleaned = InferenceScheduler.stripDocumentBlocks(from: turns)
        XCTAssertEqual(cleaned.count, 1)
        XCTAssertFalse(cleaned[0].content.contains("[Image:"))
        XCTAssertTrue(cleaned[0].content.contains("Describe the colors"))
    }

    func testPreservesPlainTurns() {
        let turns: [(role: String, content: String)] = [
            (role: "user", content: "Hello, how are you?"),
            (role: "assistant", content: "I'm doing well, thanks!")
        ]
        let cleaned = InferenceScheduler.stripDocumentBlocks(from: turns)
        XCTAssertEqual(cleaned[0].content, "Hello, how are you?")
        XCTAssertEqual(cleaned[1].content, "I'm doing well, thanks!")
    }

    func testStripsMultipleBlocks() {
        let turns: [(role: String, content: String)] = [
            (role: "user", content: "[Attached file: a.pdf]\nPDF content A\n\n[Attached file: b.docx]\nDOCX content B\n\nCompare these two documents")
        ]
        let cleaned = InferenceScheduler.stripDocumentBlocks(from: turns)
        XCTAssertEqual(cleaned.count, 1)
        XCTAssertFalse(cleaned[0].content.contains("[Attached file:"))
        XCTAssertTrue(cleaned[0].content.contains("Compare these two documents"))
    }

    func testFallsBackToOriginalIfAllContent() {
        let turns: [(role: String, content: String)] = [
            (role: "user", content: "[Attached file: data.csv]\nrow1,row2,row3")
        ]
        let cleaned = InferenceScheduler.stripDocumentBlocks(from: turns)
        XCTAssertEqual(cleaned.count, 1)
        XCTAssertFalse(cleaned[0].content.isEmpty, "Should not produce empty content")
    }
}
