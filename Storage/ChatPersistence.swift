import Foundation
import CSQLite
import OSLog

// MARK: - ChatPersistence

/// Persistent storage for chat conversations and messages using native SQLite3 + FTS5.
/// Actor isolation serializes all database access — callers `await` without blocking the main actor.
public actor ChatPersistence {
    private nonisolated(unsafe) let db: OpaquePointer
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "ChatPersistence")

    private enum SQLiteValue: Sendable {
        case text(String)
        case real(Double)
        case int(Int64)
        case blob(Data)
        case null
    }

    /// Use ``open(dbPath:)`` to create an instance — handles DB open + migrations.
    private init(db: OpaquePointer) {
        self.db = db
    }

    deinit {
        sqlite3_close(db)
    }

    /// Open (or create) the chat database and run migrations.
    /// - Parameter dbPath: Custom database file path. Pass `":memory:"` for an in-memory database (useful for tests).
    ///   When `nil`, defaults to `~/Library/Application Support/LlamaInferenceDemo/chat.db`.
    public static func open(dbPath: String? = nil) async throws -> ChatPersistence {
        let path: String
        if let dbPath {
            path = dbPath
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            let dbDir = appSupport.appendingPathComponent("LlamaInferenceDemo")
            try FileManager.default.createDirectory(at: dbDir, withIntermediateDirectories: true)
            path = dbDir.appendingPathComponent("chat.db").path
        }

        var dbPtr: OpaquePointer?
        guard sqlite3_open(path, &dbPtr) == SQLITE_OK else {
            let msg = dbPtr.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            sqlite3_close(dbPtr)
            throw ChatPersistenceError.openFailed(msg)
        }

        let instance = ChatPersistence(db: dbPtr!)
        try await instance.setup(dbPath: path)
        return instance
    }

    /// Actor-isolated setup: PRAGMAs and schema migrations.
    private func setup(dbPath: String) throws {
        try exec("PRAGMA foreign_keys = ON")
        try exec("PRAGMA journal_mode = WAL")
        try runMigrations()
        logger.debug("ChatPersistence initialized at \(dbPath, privacy: .public)")
    }

    // MARK: - Migrations

    private func runMigrations() throws {
        try exec("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                createdAt REAL NOT NULL,
                updatedAt REAL NOT NULL
            )
        """)
        try exec("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversationID TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metrics TEXT,
                thinking TEXT,
                thinkingDurationSecs REAL,
                sortOrder INTEGER NOT NULL
            )
        """)
        try exec("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content, thinking, content='messages', content_rowid='rowid'
            )
        """)
        try exec("""
            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content, thinking) VALUES (new.rowid, new.content, new.thinking);
            END
        """)
        try exec("""
            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, thinking) VALUES ('delete', old.rowid, old.content, old.thinking);
            END
        """)
        try exec("""
            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, thinking) VALUES ('delete', old.rowid, old.content, old.thinking);
                INSERT INTO messages_fts(rowid, content, thinking) VALUES (new.rowid, new.content, new.thinking);
            END
        """)

        // Attachment storage (idempotent)
        try exec("""
            CREATE TABLE IF NOT EXISTS message_attachments (
                id TEXT PRIMARY KEY,
                messageID TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                type TEXT NOT NULL,
                filename TEXT NOT NULL,
                mimeType TEXT NOT NULL,
                data BLOB NOT NULL,
                extractedText TEXT,
                thumbnailData BLOB,
                sortOrder INTEGER NOT NULL
            )
        """)

        // Branch lineage columns (idempotent — check existence first to avoid
        // SQLite "duplicate column" log noise on every launch).
        let existingColumns = columnNames(table: "conversations")
        for (name, definition) in [
            ("parentConversationID", "parentConversationID TEXT"),
            ("forkMessageIndex", "forkMessageIndex INTEGER"),
            ("forkNarrative", "forkNarrative TEXT"),
        ] {
            guard !existingColumns.contains(name) else { continue }
            try exec("ALTER TABLE conversations ADD COLUMN \(definition)")
        }
    }

    // MARK: - Conversations

    public func loadConversations() throws -> [Conversation] {
        var conversations: [Conversation] = []
        try query("SELECT id, title, updatedAt, parentConversationID, forkMessageIndex, forkNarrative FROM conversations ORDER BY updatedAt DESC") { stmt in
            conversations.append(Conversation(
                id: columnText(stmt, 0),
                title: columnText(stmt, 1),
                updatedAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2)),
                parentConversationID: columnOptionalText(stmt, 3),
                forkMessageIndex: columnOptionalInt(stmt, 4),
                forkNarrative: columnOptionalText(stmt, 5)
            ))
        }
        return conversations
    }

    /// Load a single conversation by ID (used to resolve parent title for branches).
    public func loadConversation(id: String) throws -> Conversation? {
        var result: Conversation?
        try query(
            "SELECT id, title, updatedAt, parentConversationID, forkMessageIndex, forkNarrative FROM conversations WHERE id = ?1",
            [.text(id)]
        ) { stmt in
            if result == nil {
                result = Conversation(
                    id: columnText(stmt, 0),
                    title: columnText(stmt, 1),
                    updatedAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2)),
                    parentConversationID: columnOptionalText(stmt, 3),
                    forkMessageIndex: columnOptionalInt(stmt, 4),
                    forkNarrative: columnOptionalText(stmt, 5)
                )
            }
        }
        return result
    }

    public func saveConversation(id: String, title: String, messages: [ChatMessage]) throws {
        let now = Date().timeIntervalSince1970

        try exec("BEGIN TRANSACTION")
        do {
            // Upsert conversation — preserves createdAt on conflict
            try run(
                """
                INSERT INTO conversations (id, title, createdAt, updatedAt) VALUES (?1, ?2, ?3, ?4)
                ON CONFLICT(id) DO UPDATE SET title = excluded.title, updatedAt = excluded.updatedAt
                """,
                [.text(id), .text(title), .real(now), .real(now)]
            )

            // Replace all messages for this conversation
            try run("DELETE FROM messages WHERE conversationID = ?1", [.text(id)])

            for (index, msg) in messages.enumerated() {
                try run(
                    """
                    INSERT INTO messages (id, conversationID, role, content, metrics, thinking, thinkingDurationSecs, sortOrder)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                    """,
                    [
                        .text(msg.id.uuidString),
                        .text(id),
                        .text(msg.role == .user ? "user" : "assistant"),
                        .text(msg.content),
                        msg.metrics.map { .text($0) } ?? .null,
                        msg.thinking.map { .text($0) } ?? .null,
                        msg.thinkingDurationSecs.map { .real($0) } ?? .null,
                        .int(Int64(index)),
                    ]
                )
                if !msg.attachments.isEmpty {
                    try saveAttachments(for: msg.id, attachments: msg.attachments)
                }
            }

            try exec("COMMIT")
        } catch {
            try? exec("ROLLBACK")
            throw error
        }
    }

    /// Incremental save: only INSERT new messages and DELETE removed ones, avoiding full replace.
    /// `existingMessageIDs` should be the set of message IDs from the last load or save.
    public func saveConversationIncremental(
        id: String,
        title: String,
        messages: [ChatMessage],
        existingMessageIDs: Set<UUID>
    ) throws {
        let now = Date().timeIntervalSince1970
        let currentIDs = Set(messages.map(\.id))

        try exec("BEGIN TRANSACTION")
        do {
            // Upsert conversation header
            try run(
                """
                INSERT INTO conversations (id, title, createdAt, updatedAt) VALUES (?1, ?2, ?3, ?4)
                ON CONFLICT(id) DO UPDATE SET title = excluded.title, updatedAt = excluded.updatedAt
                """,
                [.text(id), .text(title), .real(now), .real(now)]
            )

            // DELETE messages that were removed
            let removedIDs = existingMessageIDs.subtracting(currentIDs)
            for removedID in removedIDs {
                try run("DELETE FROM messages WHERE id = ?1", [.text(removedID.uuidString)])
            }

            // INSERT new messages (those not in existingMessageIDs)
            for (index, msg) in messages.enumerated() {
                if !existingMessageIDs.contains(msg.id) {
                    try run(
                        """
                        INSERT INTO messages (id, conversationID, role, content, metrics, thinking, thinkingDurationSecs, sortOrder)
                        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                        """,
                        [
                            .text(msg.id.uuidString),
                            .text(id),
                            .text(msg.role == .user ? "user" : "assistant"),
                            .text(msg.content),
                            msg.metrics.map { .text($0) } ?? .null,
                            msg.thinking.map { .text($0) } ?? .null,
                            msg.thinkingDurationSecs.map { .real($0) } ?? .null,
                            .int(Int64(index)),
                        ]
                    )
                    if !msg.attachments.isEmpty {
                        try saveAttachments(for: msg.id, attachments: msg.attachments)
                    }
                }
            }

            try exec("COMMIT")
        } catch {
            try? exec("ROLLBACK")
            throw error
        }
    }

    public func deleteConversation(id: String) throws {
        try run("DELETE FROM conversations WHERE id = ?1", [.text(id)])
    }

    /// Update only the title and updatedAt for a conversation (e.g. after semantic naming).
    public func updateConversationTitle(id: String, title: String) throws {
        let now = Date().timeIntervalSince1970
        try run(
            "UPDATE conversations SET title = ?1, updatedAt = ?2 WHERE id = ?3",
            [.text(title), .real(now), .text(id)]
        )
    }

    /// Save a branch conversation with fresh message UUIDs. Does NOT cascade-link to parent
    /// via foreign key — branches survive parent deletion (orphan tolerance).
    public func saveBranchConversation(
        id: String,
        parentID: String,
        forkMessageIndex: Int,
        forkNarrative: String?,
        title: String,
        messages: [ChatMessage]
    ) throws {
        let now = Date().timeIntervalSince1970

        try exec("BEGIN TRANSACTION")
        do {
            try run(
                """
                INSERT INTO conversations (id, title, createdAt, updatedAt, parentConversationID, forkMessageIndex, forkNarrative)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                """,
                [
                    .text(id), .text(title), .real(now), .real(now),
                    .text(parentID), .int(Int64(forkMessageIndex)),
                    forkNarrative.map { .text($0) } ?? .null,
                ]
            )

            for (index, msg) in messages.enumerated() {
                try run(
                    """
                    INSERT INTO messages (id, conversationID, role, content, metrics, thinking, thinkingDurationSecs, sortOrder)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                    """,
                    [
                        .text(msg.id.uuidString),
                        .text(id),
                        .text(msg.role == .user ? "user" : "assistant"),
                        .text(msg.content),
                        msg.metrics.map { .text($0) } ?? .null,
                        msg.thinking.map { .text($0) } ?? .null,
                        msg.thinkingDurationSecs.map { .real($0) } ?? .null,
                        .int(Int64(index)),
                    ]
                )
                if !msg.attachments.isEmpty {
                    try saveAttachments(for: msg.id, attachments: msg.attachments)
                }
            }

            try exec("COMMIT")
        } catch {
            try? exec("ROLLBACK")
            throw error
        }
    }

    /// Update the fork narrative for a branch conversation (called asynchronously after summarization).
    public func updateForkNarrative(branchID: String, narrative: String) throws {
        try run(
            "UPDATE conversations SET forkNarrative = ?1 WHERE id = ?2",
            [.text(narrative), .text(branchID)]
        )
    }

    // MARK: - Messages

    public func loadMessages(for conversationID: String) throws -> [ChatMessage] {
        var messages: [ChatMessage] = []
        try query(
            """
            SELECT id, role, content, metrics, thinking, thinkingDurationSecs
            FROM messages WHERE conversationID = ?1 ORDER BY sortOrder ASC
            """,
            [.text(conversationID)]
        ) { stmt in
            let msgID = UUID(uuidString: columnText(stmt, 0)) ?? UUID()
            messages.append(ChatMessage(
                id: msgID,
                role: columnText(stmt, 1) == "user" ? .user : .assistant,
                content: columnText(stmt, 2),
                metrics: columnOptionalText(stmt, 3),
                thinking: columnOptionalText(stmt, 4),
                thinkingDurationSecs: columnOptionalDouble(stmt, 5)
            ))
        }
        // Load attachments for messages that have them
        for i in messages.indices {
            let atts = try loadAttachments(for: messages[i].id)
            if !atts.isEmpty {
                messages[i].attachments = atts
            }
        }
        return messages
    }

    /// Load a page of messages for a conversation, ordered by sortOrder ascending.
    /// Returns messages with `sortOrder < beforeSortOrder` (or all if `beforeSortOrder` is nil),
    /// limited to `limit` results. Results are returned in ascending sortOrder.
    public func loadMessagesPage(
        for conversationID: String,
        limit: Int,
        beforeSortOrder: Int? = nil
    ) throws -> [ChatMessage] {
        var messages: [ChatMessage] = []
        if let before = beforeSortOrder {
            // Load older messages: fetch in descending order (newest-first within the page),
            // then reverse so the caller gets ascending order.
            try query(
                """
                SELECT id, role, content, metrics, thinking, thinkingDurationSecs, sortOrder
                FROM messages WHERE conversationID = ?1 AND sortOrder < ?2
                ORDER BY sortOrder DESC LIMIT ?3
                """,
                [.text(conversationID), .int(Int64(before)), .int(Int64(limit))]
            ) { stmt in
                messages.append(ChatMessage(
                    id: UUID(uuidString: columnText(stmt, 0)) ?? UUID(),
                    role: columnText(stmt, 1) == "user" ? .user : .assistant,
                    content: columnText(stmt, 2),
                    metrics: columnOptionalText(stmt, 3),
                    thinking: columnOptionalText(stmt, 4),
                    thinkingDurationSecs: columnOptionalDouble(stmt, 5)
                ))
            }
            messages.reverse()
        } else {
            // Load the most recent `limit` messages (tail of the conversation).
            try query(
                """
                SELECT id, role, content, metrics, thinking, thinkingDurationSecs, sortOrder
                FROM messages WHERE conversationID = ?1
                ORDER BY sortOrder DESC LIMIT ?2
                """,
                [.text(conversationID), .int(Int64(limit))]
            ) { stmt in
                messages.append(ChatMessage(
                    id: UUID(uuidString: columnText(stmt, 0)) ?? UUID(),
                    role: columnText(stmt, 1) == "user" ? .user : .assistant,
                    content: columnText(stmt, 2),
                    metrics: columnOptionalText(stmt, 3),
                    thinking: columnOptionalText(stmt, 4),
                    thinkingDurationSecs: columnOptionalDouble(stmt, 5)
                ))
            }
            messages.reverse()
        }
        // Load attachments for messages that have them
        for i in messages.indices {
            let atts = try loadAttachments(for: messages[i].id)
            if !atts.isEmpty {
                messages[i].attachments = atts
            }
        }
        return messages
    }

    /// Count messages for a specific conversation.
    public func countMessages(for conversationID: String) throws -> Int {
        var count = 0
        try query("SELECT COUNT(*) FROM messages WHERE conversationID = ?1", [.text(conversationID)]) { stmt in
            count = Int(sqlite3_column_int64(stmt, 0))
        }
        return count
    }

    // MARK: - Statistics

    /// Count total conversations in the database.
    public func countConversations() throws -> Int {
        var count = 0
        try query("SELECT COUNT(*) FROM conversations") { stmt in
            count = Int(sqlite3_column_int64(stmt, 0))
        }
        return count
    }

    /// Count total messages across all conversations.
    public func countMessages() throws -> Int {
        var count = 0
        try query("SELECT COUNT(*) FROM messages") { stmt in
            count = Int(sqlite3_column_int64(stmt, 0))
        }
        return count
    }

    /// Run PRAGMA integrity_check and return whether the database is healthy.
    public func checkIntegrity() throws -> Bool {
        var result = ""
        try query("PRAGMA integrity_check") { stmt in
            if result.isEmpty {
                result = columnText(stmt, 0)
            }
        }
        return result == "ok"
    }

    /// Get the current journal mode (expected: "wal").
    public func journalMode() throws -> String {
        var mode = ""
        try query("PRAGMA journal_mode") { stmt in
            mode = columnText(stmt, 0)
        }
        return mode.uppercased()
    }

    // MARK: - Search

    public func search(query searchQuery: String) throws -> [Conversation] {
        let trimmed = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return try loadConversations()
        }

        // FTS5 prefix search: each word gets a trailing * for prefix matching
        let pattern = trimmed
            .split(separator: " ")
            .map { "\($0)*" }
            .joined(separator: " ")

        var conversations: [Conversation] = []
        try query(
            """
            SELECT DISTINCT c.id, c.title, c.updatedAt, c.parentConversationID, c.forkMessageIndex, c.forkNarrative
            FROM conversations c
            JOIN messages m ON m.conversationID = c.id
            JOIN messages_fts ON messages_fts.rowid = m.rowid
            WHERE messages_fts MATCH ?1
            ORDER BY c.updatedAt DESC
            """,
            [.text(pattern)]
        ) { stmt in
            conversations.append(Conversation(
                id: columnText(stmt, 0),
                title: columnText(stmt, 1),
                updatedAt: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2)),
                parentConversationID: columnOptionalText(stmt, 3),
                forkMessageIndex: columnOptionalInt(stmt, 4),
                forkNarrative: columnOptionalText(stmt, 5)
            ))
        }
        return conversations
    }

    // MARK: - SQLite Helpers

    /// Execute one or more SQL statements with no parameter bindings (DDL, PRAGMA, etc.)
    private func exec(_ sql: String) throws {
        var errMsg: UnsafeMutablePointer<CChar>?
        guard sqlite3_exec(db, sql, nil, nil, &errMsg) == SQLITE_OK else {
            let msg = errMsg.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(errMsg)
            throw ChatPersistenceError.queryFailed(msg)
        }
    }

    /// Execute a single parameterized DML statement (INSERT, UPDATE, DELETE).
    private func run(_ sql: String, _ bindings: [SQLiteValue]) throws {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw ChatPersistenceError.queryFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindValues(stmt, bindings)
        let result = sqlite3_step(stmt)
        guard result == SQLITE_DONE || result == SQLITE_ROW else {
            throw ChatPersistenceError.queryFailed(String(cString: sqlite3_errmsg(db)))
        }
    }

    /// Execute a parameterized SELECT and call `row` for each result row.
    private func query(_ sql: String, _ bindings: [SQLiteValue] = [], row: (OpaquePointer) -> Void) throws {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw ChatPersistenceError.queryFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        bindValues(stmt, bindings)
        while true {
            let result = sqlite3_step(stmt)
            if result == SQLITE_ROW {
                row(stmt!)
            } else if result == SQLITE_DONE {
                break
            } else {
                throw ChatPersistenceError.queryFailed(String(cString: sqlite3_errmsg(db)))
            }
        }
    }

    private func bindValues(_ stmt: OpaquePointer?, _ bindings: [SQLiteValue]) {
        for (i, value) in bindings.enumerated() {
            let idx = Int32(i + 1)
            switch value {
            case .text(let s):
                sqlite3_bind_text(stmt, idx, s, -1, csqlite_TRANSIENT())
            case .real(let d):
                sqlite3_bind_double(stmt, idx, d)
            case .int(let n):
                sqlite3_bind_int64(stmt, idx, n)
            case .blob(let data):
                _ = data.withUnsafeBytes { ptr in
                    sqlite3_bind_blob(stmt, idx, ptr.baseAddress, Int32(data.count), csqlite_TRANSIENT())
                }
            case .null:
                sqlite3_bind_null(stmt, idx)
            }
        }
    }

    /// Returns the set of column names for the given table via `PRAGMA table_info`.
    private func columnNames(table: String) -> Set<String> {
        var names: Set<String> = []
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "PRAGMA table_info(\(table))", -1, &stmt, nil) == SQLITE_OK else {
            return names
        }
        defer { sqlite3_finalize(stmt) }
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let ptr = sqlite3_column_text(stmt, 1) {
                names.insert(String(cString: ptr))
            }
        }
        return names
    }

    private func columnText(_ stmt: OpaquePointer, _ index: Int32) -> String {
        guard let ptr = sqlite3_column_text(stmt, index) else { return "" }
        return String(cString: ptr)
    }

    private func columnOptionalText(_ stmt: OpaquePointer, _ index: Int32) -> String? {
        guard sqlite3_column_type(stmt, index) != SQLITE_NULL,
              let ptr = sqlite3_column_text(stmt, index) else { return nil }
        return String(cString: ptr)
    }

    private func columnOptionalDouble(_ stmt: OpaquePointer, _ index: Int32) -> Double? {
        guard sqlite3_column_type(stmt, index) != SQLITE_NULL else { return nil }
        return sqlite3_column_double(stmt, index)
    }

    private func columnOptionalInt(_ stmt: OpaquePointer, _ index: Int32) -> Int? {
        guard sqlite3_column_type(stmt, index) != SQLITE_NULL else { return nil }
        return Int(sqlite3_column_int64(stmt, index))
    }

    private func columnOptionalBlob(_ stmt: OpaquePointer, _ index: Int32) -> Data? {
        guard sqlite3_column_type(stmt, index) != SQLITE_NULL,
              let ptr = sqlite3_column_blob(stmt, index) else { return nil }
        let count = Int(sqlite3_column_bytes(stmt, index))
        return Data(bytes: ptr, count: count)
    }

    private func columnBlob(_ stmt: OpaquePointer, _ index: Int32) -> Data {
        guard let ptr = sqlite3_column_blob(stmt, index) else { return Data() }
        let count = Int(sqlite3_column_bytes(stmt, index))
        return Data(bytes: ptr, count: count)
    }

    // MARK: - Attachments

    public func saveAttachments(for messageID: UUID, attachments: [MessageAttachment]) throws {
        guard !attachments.isEmpty else { return }
        for (index, att) in attachments.enumerated() {
            // Skip attachments without data — these are lightweight placeholders
            // that will be updated once extraction completes.
            guard let data = att.data else { continue }
            try run(
                """
                INSERT OR REPLACE INTO message_attachments
                    (id, messageID, type, filename, mimeType, data, extractedText, thumbnailData, sortOrder)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                """,
                [
                    .text(att.id.uuidString),
                    .text(messageID.uuidString),
                    .text(att.type.rawValue),
                    .text(att.filename),
                    .text(att.mimeType),
                    .blob(data),
                    att.extractedText.map { .text($0) } ?? .null,
                    att.thumbnailData.map { .blob($0) } ?? .null,
                    .int(Int64(index)),
                ]
            )
        }
    }

    public func loadAttachments(for messageID: UUID) throws -> [MessageAttachment] {
        var attachments: [MessageAttachment] = []
        // Omit the data BLOB column — the UI only needs metadata + thumbnails.
        // The full file data lives in SQLite and is never needed after persistence.
        try query(
            """
            SELECT id, type, filename, mimeType, extractedText, thumbnailData
            FROM message_attachments WHERE messageID = ?1 ORDER BY sortOrder ASC
            """,
            [.text(messageID.uuidString)]
        ) { stmt in
            let typeStr = columnText(stmt, 1)
            let attType = AttachmentType(rawValue: typeStr) ?? .textFile
            attachments.append(MessageAttachment(
                id: UUID(uuidString: columnText(stmt, 0)) ?? UUID(),
                type: attType,
                filename: columnText(stmt, 2),
                mimeType: columnText(stmt, 3),
                data: nil,
                extractedText: columnOptionalText(stmt, 4),
                thumbnailData: columnOptionalBlob(stmt, 5)
            ))
        }
        return attachments
    }

    /// Load attachment references for rebuilding mention registries when a
    /// conversation is loaded from persistence.
    public func loadAttachmentReferences(for conversationID: String) throws -> [AttachmentReferenceRecord] {
        var records: [AttachmentReferenceRecord] = []
        try query(
            """
            SELECT
                m.id,
                m.sortOrder,
                a.id,
                a.sortOrder,
                a.type,
                a.filename,
                a.mimeType,
                a.extractedText
            FROM messages m
            JOIN message_attachments a ON a.messageID = m.id
            WHERE m.conversationID = ?1
            ORDER BY m.sortOrder ASC, a.sortOrder ASC
            """,
            [.text(conversationID)]
        ) { stmt in
            let type = AttachmentType(rawValue: columnText(stmt, 4)) ?? .textFile
            records.append(AttachmentReferenceRecord(
                messageID: UUID(uuidString: columnText(stmt, 0)) ?? UUID(),
                messageSortOrder: Int(sqlite3_column_int64(stmt, 1)),
                attachmentID: UUID(uuidString: columnText(stmt, 2)) ?? UUID(),
                attachmentSortOrder: Int(sqlite3_column_int64(stmt, 3)),
                type: type,
                filename: columnText(stmt, 5),
                mimeType: columnText(stmt, 6),
                extractedText: columnOptionalText(stmt, 7)
            ))
        }
        return records
    }

    public func deleteAttachments(for messageID: UUID) throws {
        try run("DELETE FROM message_attachments WHERE messageID = ?1", [.text(messageID.uuidString)])
    }
}

// MARK: - Errors

public enum ChatPersistenceError: Error, LocalizedError {
    case openFailed(String)
    case queryFailed(String)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let msg): return "Failed to open database: \(msg)"
        case .queryFailed(let msg): return "Database query failed: \(msg)"
        }
    }
}
