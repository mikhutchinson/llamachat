import Foundation

/// A chat conversation summary for sidebar display.
public struct Conversation: Identifiable, Sendable, Equatable {
    public let id: String
    public var title: String
    public var updatedAt: Date
    public var parentConversationID: String?
    public var forkMessageIndex: Int?
    public var forkNarrative: String?

    public init(
        id: String,
        title: String,
        updatedAt: Date,
        parentConversationID: String? = nil,
        forkMessageIndex: Int? = nil,
        forkNarrative: String? = nil
    ) {
        self.id = id
        self.title = title
        self.updatedAt = updatedAt
        self.parentConversationID = parentConversationID
        self.forkMessageIndex = forkMessageIndex
        self.forkNarrative = forkNarrative
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id
            && lhs.title == rhs.title
            && lhs.updatedAt == rhs.updatedAt
            && lhs.parentConversationID == rhs.parentConversationID
            && lhs.forkMessageIndex == rhs.forkMessageIndex
            && lhs.forkNarrative == rhs.forkNarrative
    }
}

/// A single chat message (user or assistant).
public struct ChatMessage: Identifiable, Sendable {
    public let id: UUID
    public let role: MessageRole
    public let content: String
    public var metrics: String?
    public var thinking: String?
    public var thinkingDurationSecs: Double?
    public var attachments: [MessageAttachment]

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        metrics: String? = nil,
        thinking: String? = nil,
        thinkingDurationSecs: Double? = nil,
        attachments: [MessageAttachment] = []
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.metrics = metrics
        self.thinking = thinking
        self.thinkingDurationSecs = thinkingDurationSecs
        self.attachments = attachments
    }
}

public enum MessageRole: Sendable, Equatable {
    case user
    case assistant
}

// MARK: - Attachments

public enum AttachmentType: String, Sendable, Equatable {
    case textFile = "text"
    case image = "image"
    case pdf = "pdf"
}

public struct MessageAttachment: Identifiable, Sendable {
    public let id: UUID
    public let type: AttachmentType
    public let filename: String
    public let mimeType: String
    public let data: Data?
    public var extractedText: String?
    public var thumbnailData: Data?

    public init(
        id: UUID = UUID(),
        type: AttachmentType,
        filename: String,
        mimeType: String,
        data: Data? = nil,
        extractedText: String? = nil,
        thumbnailData: Data? = nil
    ) {
        self.id = id
        self.type = type
        self.filename = filename
        self.mimeType = mimeType
        self.data = data
        self.extractedText = extractedText
        self.thumbnailData = thumbnailData
    }
}

/// Flattened attachment metadata for rebuilding per-session mention registries.
public struct AttachmentReferenceRecord: Sendable, Equatable {
    public let messageID: UUID
    public let messageSortOrder: Int
    public let attachmentID: UUID
    public let attachmentSortOrder: Int
    public let type: AttachmentType
    public let filename: String
    public let mimeType: String
    public let extractedText: String?

    public init(
        messageID: UUID,
        messageSortOrder: Int,
        attachmentID: UUID,
        attachmentSortOrder: Int,
        type: AttachmentType,
        filename: String,
        mimeType: String,
        extractedText: String?
    ) {
        self.messageID = messageID
        self.messageSortOrder = messageSortOrder
        self.attachmentID = attachmentID
        self.attachmentSortOrder = attachmentSortOrder
        self.type = type
        self.filename = filename
        self.mimeType = mimeType
        self.extractedText = extractedText
    }
}

#if canImport(AppKit)
import AppKit
#endif

public struct PendingAttachment: Identifiable, Sendable, Equatable {
    public let id: UUID
    public let filename: String
    public let type: AttachmentType
    public let fileURL: URL
    public let fileSize: Int64
    #if canImport(AppKit)
    public let thumbnailImage: NSImage?
    #endif

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id
    }

    #if canImport(AppKit)
    public init(
        id: UUID = UUID(),
        filename: String,
        type: AttachmentType,
        fileURL: URL,
        fileSize: Int64,
        thumbnailImage: NSImage? = nil
    ) {
        self.id = id
        self.filename = filename
        self.type = type
        self.fileURL = fileURL
        self.fileSize = fileSize
        self.thumbnailImage = thumbnailImage
    }
    #else
    public init(
        id: UUID = UUID(),
        filename: String,
        type: AttachmentType,
        fileURL: URL,
        fileSize: Int64
    ) {
        self.id = id
        self.filename = filename
        self.type = type
        self.fileURL = fileURL
        self.fileSize = fileSize
    }
    #endif
}
