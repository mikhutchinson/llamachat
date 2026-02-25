import Foundation

// MARK: - Session Identity

public struct SessionID: Hashable, Sendable, CustomStringConvertible {
    public let raw: UUID

    public init() { self.raw = UUID() }
    public init(_ uuid: UUID) { self.raw = uuid }

    public var description: String { "session-\(raw.uuidString.prefix(8))" }
}

// MARK: - Session State

public enum SessionPhase: String, Sendable, CaseIterable {
    case idle
    case prefilling
    case decoding
    case completed
    case failed
    case evicted
}

public struct SessionState: Sendable {
    public let id: SessionID
    public let workerIndex: Int
    public private(set) var phase: SessionPhase
    public private(set) var promptTokenCount: Int
    public private(set) var completionTokenCount: Int
    public private(set) var createdAt: Date
    public private(set) var lastActivityAt: Date
    public private(set) var generatedText: String
    public private(set) var finishReason: String?

    public var totalTokenCount: Int { promptTokenCount + completionTokenCount }

    public init(id: SessionID, workerIndex: Int) {
        self.id = id
        self.workerIndex = workerIndex
        self.phase = .idle
        self.promptTokenCount = 0
        self.completionTokenCount = 0
        self.createdAt = Date()
        self.lastActivityAt = Date()
        self.generatedText = ""
        self.finishReason = nil
    }

    mutating func transitionTo(_ phase: SessionPhase) {
        self.phase = phase
        self.lastActivityAt = Date()
    }

    mutating func recordPrefill(promptTokens: Int) {
        self.promptTokenCount = promptTokens
        self.lastActivityAt = Date()
    }

    mutating func recordDecodeStep(newText: String, tokens: Int, finishReason: String?) {
        self.generatedText += newText
        self.completionTokenCount += tokens
        self.finishReason = finishReason
        self.lastActivityAt = Date()
    }
}

// MARK: - Sampling Parameters

public struct SamplingParams: Sendable {
    public let maxTokens: Int
    public let temperature: Double
    public let topP: Double
    public let topK: Int
    public let repeatPenalty: Double
    public let stop: [String]

    public init(
        maxTokens: Int = 256,
        temperature: Double = 0.7,
        topP: Double = 0.95,
        topK: Int = 40,
        repeatPenalty: Double = 1.1,
        stop: [String] = []
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repeatPenalty = repeatPenalty
        self.stop = stop
    }

    public static let `default` = SamplingParams()

    public static let greedy = SamplingParams(
        temperature: 0.0,
        topP: 1.0,
        topK: 1,
        repeatPenalty: 1.0
    )
}

// MARK: - Inference Configuration

public struct InferenceConfig: Sendable {
    public let modelPath: String
    /// Path for dedicated summarizer model. When set, a dedicated worker loads this model
    /// for narrative summarization; otherwise summarizer shares the chat model on worker 0.
    public let summarizerModelPath: String?
    public let contextSize: Int
    public let nGpuLayers: Int
    public let workerCount: Int
    public let maxSessionsPerWorker: Int
    public let maxMemoryBytesPerWorker: Int?
    public let maxInFlight: Int?
    public let blasThreads: Int?
    public let useSharedMemory: Bool
    public let sharedMemorySlotSize: Int
    /// Explicit path to SwiftPythonWorker executable. When nil, auto-detected
    /// relative to the main executable. Override for test runners.
    public let workerExecutablePath: String?
    /// Path to Python virtual environment. When nil, uses VIRTUAL_ENV env var.
    /// Used to inject site-packages into worker sys.path.
    public let venvPath: String?

    public init(
        modelPath: String,
        summarizerModelPath: String? = nil,
        contextSize: Int = 4096,
        nGpuLayers: Int = -1,
        workerCount: Int = 2,
        maxSessionsPerWorker: Int = 8,
        maxMemoryBytesPerWorker: Int? = nil,
        maxInFlight: Int? = 16,
        blasThreads: Int? = 1,
        useSharedMemory: Bool = false,
        sharedMemorySlotSize: Int = 65536,
        workerExecutablePath: String? = nil,
        venvPath: String? = nil
    ) {
        self.modelPath = modelPath
        self.summarizerModelPath = summarizerModelPath
        self.contextSize = contextSize
        self.nGpuLayers = nGpuLayers
        self.workerCount = workerCount
        self.maxSessionsPerWorker = maxSessionsPerWorker
        self.maxMemoryBytesPerWorker = maxMemoryBytesPerWorker
        self.maxInFlight = maxInFlight
        self.blasThreads = blasThreads
        self.useSharedMemory = useSharedMemory
        self.sharedMemorySlotSize = sharedMemorySlotSize
        self.workerExecutablePath = workerExecutablePath
        self.venvPath = venvPath
    }

    public var maxTotalSessions: Int { workerCount * maxSessionsPerWorker }
}

// MARK: - Request / Response

public struct InferenceRequest: Sendable {
    public let sessionID: SessionID
    public let prompt: String
    public let params: SamplingParams
    public let chatMessages: [[String: String]]?
    public let createdAt: Date

    public init(
        sessionID: SessionID = SessionID(),
        prompt: String,
        params: SamplingParams = .default,
        chatMessages: [[String: String]]? = nil
    ) {
        self.sessionID = sessionID
        self.prompt = prompt
        self.params = params
        self.chatMessages = chatMessages
        self.createdAt = Date()
    }
}

public struct InferenceResult: Sendable {
    public let sessionID: SessionID
    public let text: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let finishReason: String
    public let workerIndex: Int
    public let prefillDuration: Duration
    public let decodeDuration: Duration
    public let totalDuration: Duration
    public let thinking: String?

    public init(
        sessionID: SessionID,
        text: String,
        promptTokens: Int,
        completionTokens: Int,
        finishReason: String,
        workerIndex: Int,
        prefillDuration: Duration,
        decodeDuration: Duration,
        totalDuration: Duration,
        thinking: String? = nil
    ) {
        self.sessionID = sessionID
        self.text = text
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.finishReason = finishReason
        self.workerIndex = workerIndex
        self.prefillDuration = prefillDuration
        self.decodeDuration = decodeDuration
        self.totalDuration = totalDuration
        self.thinking = thinking
    }

    public var tokensPerSecond: Double {
        let seconds = Double(decodeDuration.components.seconds)
            + Double(decodeDuration.components.attoseconds) / 1e18
        guard seconds > 0 else { return 0 }
        return Double(completionTokens) / seconds
    }
}

// MARK: - Streamed Inference

public enum StreamEventKind: String, Sendable {
    case delta
    case done
    case error
}

/// Stream chunk emitted during `completeStream` decode.
public struct StreamInferenceChunk: Sendable {
    public let event: StreamEventKind
    public let delta: String
    public let finishReason: String?
    public let promptTokens: Int?
    public let completionTokens: Int?
    public let decodeMs: Double?
    public let prefillMs: Double?
    public let text: String?
    public let thinking: String?
    public let error: String?
    public let traceback: String?

    public init(
        event: StreamEventKind,
        delta: String = "",
        finishReason: String? = nil,
        promptTokens: Int? = nil,
        completionTokens: Int? = nil,
        decodeMs: Double? = nil,
        prefillMs: Double? = nil,
        text: String? = nil,
        thinking: String? = nil,
        error: String? = nil,
        traceback: String? = nil
    ) {
        self.event = event
        self.delta = delta
        self.finishReason = finishReason
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.decodeMs = decodeMs
        self.prefillMs = prefillMs
        self.text = text
        self.thinking = thinking
        self.error = error
        self.traceback = traceback
    }

    public var isTerminal: Bool {
        event == .done || event == .error
    }
}

// MARK: - Context Wind Thresholds

public enum ContextThreshold: Double, Sendable, CaseIterable, Comparable {
    case prepare = 0.60
    case commit  = 0.70
    case reset   = 0.80

    public static func < (lhs: ContextThreshold, rhs: ContextThreshold) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public struct ContextWindEvent: Sendable {
    public let sessionID: SessionID
    public let threshold: ContextThreshold
    public let utilization: Double
    public let promptTokens: Int
    public let contextSize: Int
    public let timestamp: Date

    public init(
        sessionID: SessionID,
        threshold: ContextThreshold,
        utilization: Double,
        promptTokens: Int,
        contextSize: Int
    ) {
        self.sessionID = sessionID
        self.threshold = threshold
        self.utilization = utilization
        self.promptTokens = promptTokens
        self.contextSize = contextSize
        self.timestamp = Date()
    }
}

// MARK: - Errors

public enum InferenceError: Error, LocalizedError, Sendable {
    case poolNotReady
    case modelLoadFailed(String)
    case sessionNotFound(SessionID)
    case workerFull(workerIndex: Int)
    case contextOverflow(sessionID: SessionID, used: Int, max: Int)
    case prefillFailed(sessionID: SessionID, reason: String)
    case decodeFailed(sessionID: SessionID, reason: String)
    case evicted(sessionID: SessionID)
    case timeout(sessionID: SessionID)

    public var errorDescription: String? {
        switch self {
        case .poolNotReady:
            return "Inference pool is not ready. Call startup() first."
        case .modelLoadFailed(let msg):
            return "Model load failed: \(msg)"
        case .sessionNotFound(let id):
            return "Session \(id) not found"
        case .workerFull(let idx):
            return "Worker \(idx) has reached max session capacity"
        case .contextOverflow(let id, let used, let max):
            return "Session \(id) exceeded context: \(used)/\(max) tokens"
        case .prefillFailed(let id, let reason):
            return "Prefill failed for \(id): \(reason)"
        case .decodeFailed(let id, let reason):
            return "Decode failed for \(id): \(reason)"
        case .evicted(let id):
            return "Session \(id) was evicted"
        case .timeout(let id):
            return "Session \(id) timed out"
        }
    }
}
