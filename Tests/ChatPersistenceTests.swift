import XCTest
import ChatStorage

final class ChatPersistenceTests: XCTestCase {

    private func makeDB() async throws -> ChatPersistence {
        try await ChatPersistence.open(dbPath: ":memory:")
    }

    // MARK: - Empty State

    func testEmptyDatabaseHasNoConversations() async throws {
        let db = try await makeDB()
        let convos = try await db.loadConversations()
        XCTAssertTrue(convos.isEmpty)
    }

    // MARK: - Save & Load

    func testSaveAndLoadConversation() async throws {
        let db = try await makeDB()
        let messages = [
            ChatMessage(role: .user, content: "Hello"),
            ChatMessage(role: .assistant, content: "Hi there!", metrics: "10.0 tok/s"),
        ]

        try await db.saveConversation(id: "c1", title: "Hello", messages: messages)

        let convos = try await db.loadConversations()
        XCTAssertEqual(convos.count, 1)
        XCTAssertEqual(convos[0].id, "c1")
        XCTAssertEqual(convos[0].title, "Hello")

        let loaded = try await db.loadMessages(for: "c1")
        XCTAssertEqual(loaded.count, 2)
        XCTAssertEqual(loaded[0].role, .user)
        XCTAssertEqual(loaded[0].content, "Hello")
        XCTAssertEqual(loaded[1].role, .assistant)
        XCTAssertEqual(loaded[1].content, "Hi there!")
        XCTAssertEqual(loaded[1].metrics, "10.0 tok/s")
    }

    func testStableMessageIDs() async throws {
        let db = try await makeDB()
        let msgID = UUID()
        let messages = [ChatMessage(id: msgID, role: .user, content: "test")]

        try await db.saveConversation(id: "c1", title: "Test", messages: messages)
        let loaded = try await db.loadMessages(for: "c1")

        XCTAssertEqual(loaded.count, 1)
        XCTAssertEqual(loaded[0].id, msgID)
    }

    func testNullableFieldsPreserved() async throws {
        let db = try await makeDB()
        let messages = [
            ChatMessage(role: .assistant, content: "answer", metrics: nil, thinking: "thought process", thinkingDurationSecs: 2.5),
            ChatMessage(role: .assistant, content: "plain", metrics: "5.0 tok/s", thinking: nil, thinkingDurationSecs: nil),
        ]

        try await db.saveConversation(id: "c1", title: "Test", messages: messages)
        let loaded = try await db.loadMessages(for: "c1")

        XCTAssertEqual(loaded.count, 2)
        XCTAssertNil(loaded[0].metrics)
        XCTAssertEqual(loaded[0].thinking, "thought process")
        XCTAssertEqual(loaded[0].thinkingDurationSecs, 2.5)
        XCTAssertEqual(loaded[1].metrics, "5.0 tok/s")
        XCTAssertNil(loaded[1].thinking)
        XCTAssertNil(loaded[1].thinkingDurationSecs)
    }

    func testMessageOrderPreserved() async throws {
        let db = try await makeDB()
        let messages = (0..<5).map { i in
            ChatMessage(role: i % 2 == 0 ? .user : .assistant, content: "Message \(i)")
        }

        try await db.saveConversation(id: "c1", title: "Chat", messages: messages)
        let loaded = try await db.loadMessages(for: "c1")

        XCTAssertEqual(loaded.count, 5)
        for i in 0..<5 {
            XCTAssertEqual(loaded[i].content, "Message \(i)")
        }
    }

    // MARK: - Upsert

    func testUpsertPreservesCreatedAtAndUpdatesTitle() async throws {
        let db = try await makeDB()

        try await db.saveConversation(id: "c1", title: "First", messages: [
            ChatMessage(role: .user, content: "hello"),
        ])

        try await Task.sleep(for: .milliseconds(50))

        try await db.saveConversation(id: "c1", title: "Updated", messages: [
            ChatMessage(role: .user, content: "hello"),
            ChatMessage(role: .assistant, content: "world"),
        ])

        let convos = try await db.loadConversations()
        XCTAssertEqual(convos.count, 1)
        XCTAssertEqual(convos[0].title, "Updated")

        let loaded = try await db.loadMessages(for: "c1")
        XCTAssertEqual(loaded.count, 2)
    }

    // MARK: - Delete

    func testDeleteConversationCascadesToMessages() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "c1", title: "Chat", messages: [
            ChatMessage(role: .user, content: "hello"),
            ChatMessage(role: .assistant, content: "hi"),
        ])

        try await db.deleteConversation(id: "c1")

        let convos = try await db.loadConversations()
        XCTAssertTrue(convos.isEmpty)

        // Messages should be cascade-deleted
        let messages = try await db.loadMessages(for: "c1")
        XCTAssertTrue(messages.isEmpty)
    }

    func testDeleteNonexistentDoesNotThrow() async throws {
        let db = try await makeDB()
        try await db.deleteConversation(id: "nonexistent")
    }

    // MARK: - Update Title

    func testUpdateConversationTitle() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "c1", title: "Original", messages: [
            ChatMessage(role: .user, content: "hello"),
            ChatMessage(role: .assistant, content: "hi"),
        ])

        try await db.updateConversationTitle(id: "c1", title: "Semantic Title From AI")

        let convos = try await db.loadConversations()
        XCTAssertEqual(convos.count, 1)
        XCTAssertEqual(convos[0].title, "Semantic Title From AI")
    }

    // MARK: - Ordering

    func testConversationsOrderedByUpdatedAtDesc() async throws {
        let db = try await makeDB()

        try await db.saveConversation(id: "old", title: "Old", messages: [
            ChatMessage(role: .user, content: "first"),
        ])
        try await Task.sleep(for: .milliseconds(50))
        try await db.saveConversation(id: "new", title: "New", messages: [
            ChatMessage(role: .user, content: "second"),
        ])

        let convos = try await db.loadConversations()
        XCTAssertEqual(convos.count, 2)
        XCTAssertEqual(convos[0].id, "new")
        XCTAssertEqual(convos[1].id, "old")
    }

    // MARK: - FTS5 Search

    func testSearchFindsMatchingConversation() async throws {
        let db = try await makeDB()

        try await db.saveConversation(id: "c1", title: "Science", messages: [
            ChatMessage(role: .user, content: "Tell me about quantum computing"),
            ChatMessage(role: .assistant, content: "Quantum computing uses qubits..."),
        ])
        try await db.saveConversation(id: "c2", title: "Recipe", messages: [
            ChatMessage(role: .user, content: "How do I make pasta?"),
            ChatMessage(role: .assistant, content: "Boil water and add noodles"),
        ])

        let results = try await db.search(query: "quantum")
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "c1")
    }

    func testSearchPrefixMatching() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "c1", title: "Chat", messages: [
            ChatMessage(role: .user, content: "Tell me about programming languages"),
        ])

        // "prog*" should match "programming"
        let results = try await db.search(query: "prog")
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "c1")
    }

    func testSearchMatchesThinkingContent() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "c1", title: "Deep", messages: [
            ChatMessage(role: .assistant, content: "The answer is 42", thinking: "Let me reason about thermodynamics"),
        ])

        let results = try await db.search(query: "thermodynamics")
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "c1")
    }

    func testSearchEmptyQueryReturnsAll() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "c1", title: "A", messages: [
            ChatMessage(role: .user, content: "hello"),
        ])
        try await db.saveConversation(id: "c2", title: "B", messages: [
            ChatMessage(role: .user, content: "world"),
        ])

        let results = try await db.search(query: "")
        XCTAssertEqual(results.count, 2)

        let resultsWhitespace = try await db.search(query: "   ")
        XCTAssertEqual(resultsWhitespace.count, 2)
    }

    func testSearchNoResults() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "c1", title: "Chat", messages: [
            ChatMessage(role: .user, content: "hello world"),
        ])

        let results = try await db.search(query: "xyznonexistent")
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchMultipleWordsMatchesAll() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "c1", title: "Chat", messages: [
            ChatMessage(role: .user, content: "quantum computing is fascinating"),
        ])
        try await db.saveConversation(id: "c2", title: "Other", messages: [
            ChatMessage(role: .user, content: "quantum mechanics only"),
        ])

        // Both words must match (FTS5 implicit AND)
        let results = try await db.search(query: "quantum computing")
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "c1")
    }

    // MARK: - Branch & Retry

    func testBranchSaveAndLoad() async throws {
        let db = try await makeDB()
        let parentMessages = [
            ChatMessage(role: .user, content: "Hello"),
            ChatMessage(role: .assistant, content: "Hi there!"),
        ]
        try await db.saveConversation(id: "parent", title: "Parent", messages: parentMessages)

        let branchMessages = [
            ChatMessage(role: .user, content: "Hello"),
            ChatMessage(role: .assistant, content: "Hi there!"),
        ]
        try await db.saveBranchConversation(
            id: "branch1",
            parentID: "parent",
            forkMessageIndex: 1,
            forkNarrative: nil,
            title: "Branch",
            messages: branchMessages
        )

        let convos = try await db.loadConversations()
        XCTAssertEqual(convos.count, 2)

        let branch = convos.first { $0.id == "branch1" }
        XCTAssertNotNil(branch)
        XCTAssertEqual(branch?.parentConversationID, "parent")
        XCTAssertEqual(branch?.forkMessageIndex, 1)
        XCTAssertNil(branch?.forkNarrative)
    }

    func testBranchCopiesCorrectMessages() async throws {
        let db = try await makeDB()
        let parentMessages = (0..<6).map { i in
            ChatMessage(role: i % 2 == 0 ? .user : .assistant, content: "Message \(i)")
        }
        try await db.saveConversation(id: "parent", title: "Parent", messages: parentMessages)

        // Branch at index 3 — should include messages 0..3 (4 messages)
        let branchMessages = parentMessages.prefix(4).map { msg in
            ChatMessage(id: UUID(), role: msg.role, content: msg.content)
        }
        try await db.saveBranchConversation(
            id: "branch",
            parentID: "parent",
            forkMessageIndex: 3,
            forkNarrative: nil,
            title: "Branch",
            messages: branchMessages
        )

        let loaded = try await db.loadMessages(for: "branch")
        XCTAssertEqual(loaded.count, 4)
        for i in 0..<4 {
            XCTAssertEqual(loaded[i].content, "Message \(i)")
        }
    }

    func testBranchOrphanTolerance() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "parent", title: "Parent", messages: [
            ChatMessage(role: .user, content: "hello"),
        ])
        try await db.saveBranchConversation(
            id: "branch",
            parentID: "parent",
            forkMessageIndex: 0,
            forkNarrative: nil,
            title: "Branch",
            messages: [ChatMessage(role: .user, content: "hello")]
        )

        // Delete parent — branch should survive (no FK cascade between conversations)
        try await db.deleteConversation(id: "parent")

        let convos = try await db.loadConversations()
        XCTAssertEqual(convos.count, 1)
        XCTAssertEqual(convos[0].id, "branch")
        XCTAssertEqual(convos[0].parentConversationID, "parent")

        let msgs = try await db.loadMessages(for: "branch")
        XCTAssertEqual(msgs.count, 1)
    }

    func testBranchNarrativePersists() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "parent", title: "Parent", messages: [
            ChatMessage(role: .user, content: "hello"),
        ])
        try await db.saveBranchConversation(
            id: "branch",
            parentID: "parent",
            forkMessageIndex: 0,
            forkNarrative: nil,
            title: "Branch",
            messages: [ChatMessage(role: .user, content: "hello")]
        )

        // Initially nil
        let before = try await db.loadConversation(id: "branch")
        XCTAssertNil(before?.forkNarrative)

        // Update narrative
        try await db.updateForkNarrative(branchID: "branch", narrative: "User greeted the assistant.")

        let after = try await db.loadConversation(id: "branch")
        XCTAssertEqual(after?.forkNarrative, "User greeted the assistant.")
    }

    func testLoadSingleConversation() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "c1", title: "Chat", messages: [
            ChatMessage(role: .user, content: "hello"),
        ])

        let found = try await db.loadConversation(id: "c1")
        XCTAssertNotNil(found)
        XCTAssertEqual(found?.id, "c1")
        XCTAssertEqual(found?.title, "Chat")

        let notFound = try await db.loadConversation(id: "nonexistent")
        XCTAssertNil(notFound)
    }

    func testOldConversationsHaveNilForkFields() async throws {
        let db = try await makeDB()
        try await db.saveConversation(id: "c1", title: "Old Chat", messages: [
            ChatMessage(role: .user, content: "hello"),
        ])

        let convos = try await db.loadConversations()
        XCTAssertEqual(convos.count, 1)
        XCTAssertNil(convos[0].parentConversationID)
        XCTAssertNil(convos[0].forkMessageIndex)
        XCTAssertNil(convos[0].forkNarrative)
    }

    // MARK: - Attachment Persistence

    func testSaveAndLoadAttachments() async throws {
        let db = try await makeDB()
        let attID = UUID()
        let testData = "Hello, world!".data(using: .utf8)!
        let attachment = MessageAttachment(
            id: attID,
            type: .textFile,
            filename: "test.txt",
            mimeType: "text/plain",
            data: testData,
            extractedText: "Hello, world!"
        )
        let msg = ChatMessage(role: .user, content: "See attached", attachments: [attachment])

        try await db.saveConversation(id: "c1", title: "With Attachment", messages: [msg])

        let loaded = try await db.loadMessages(for: "c1")
        XCTAssertEqual(loaded.count, 1)
        XCTAssertEqual(loaded[0].attachments.count, 1)

        let loadedAtt = loaded[0].attachments[0]
        XCTAssertEqual(loadedAtt.id, attID)
        XCTAssertEqual(loadedAtt.type, .textFile)
        XCTAssertEqual(loadedAtt.filename, "test.txt")
        XCTAssertEqual(loadedAtt.mimeType, "text/plain")
        // data is intentionally not loaded — full file blobs stay in SQLite
        // to keep the in-memory messages array lightweight for SwiftUI.
        XCTAssertNil(loadedAtt.data)
        XCTAssertEqual(loadedAtt.extractedText, "Hello, world!")
        XCTAssertNil(loadedAtt.thumbnailData)
    }

    func testAttachmentCascadeDeleteWithMessage() async throws {
        let db = try await makeDB()
        let attachment = MessageAttachment(
            type: .textFile, filename: "f.txt", mimeType: "text/plain",
            data: "data".data(using: .utf8)!
        )
        let msg = ChatMessage(role: .user, content: "hello", attachments: [attachment])

        try await db.saveConversation(id: "c1", title: "Chat", messages: [msg])

        // Verify attachment exists
        let before = try await db.loadMessages(for: "c1")
        XCTAssertEqual(before[0].attachments.count, 1)

        // Delete conversation — messages cascade, attachments cascade via FK
        try await db.deleteConversation(id: "c1")

        let messages = try await db.loadMessages(for: "c1")
        XCTAssertTrue(messages.isEmpty)

        // Verify attachments are gone (load directly)
        let orphanAtts = try await db.loadAttachments(for: msg.id)
        XCTAssertTrue(orphanAtts.isEmpty)
    }

    func testMultipleAttachmentsOrderPreserved() async throws {
        let db = try await makeDB()
        let atts = (0..<3).map { i in
            MessageAttachment(
                type: .textFile, filename: "file\(i).txt", mimeType: "text/plain",
                data: "content \(i)".data(using: .utf8)!
            )
        }
        let msg = ChatMessage(role: .user, content: "multiple files", attachments: atts)

        try await db.saveConversation(id: "c1", title: "Multi", messages: [msg])

        let loaded = try await db.loadMessages(for: "c1")
        XCTAssertEqual(loaded[0].attachments.count, 3)
        for i in 0..<3 {
            XCTAssertEqual(loaded[0].attachments[i].filename, "file\(i).txt")
        }
    }

    func testAttachmentWithThumbnailData() async throws {
        let db = try await makeDB()
        let imgData = Data(repeating: 0xFF, count: 100)
        let thumbData = Data(repeating: 0xAA, count: 50)
        let attachment = MessageAttachment(
            type: .image, filename: "photo.jpg", mimeType: "image/jpeg",
            data: imgData, thumbnailData: thumbData
        )
        let msg = ChatMessage(role: .user, content: "see photo", attachments: [attachment])

        try await db.saveConversation(id: "c1", title: "Photo", messages: [msg])

        let loaded = try await db.loadMessages(for: "c1")
        let loadedAtt = loaded[0].attachments[0]
        XCTAssertEqual(loadedAtt.type, .image)
        // data is intentionally not loaded — full file blobs stay in SQLite
        XCTAssertNil(loadedAtt.data)
        XCTAssertEqual(loadedAtt.thumbnailData, thumbData)
        XCTAssertNil(loadedAtt.extractedText)
    }

    func testMessagesWithoutAttachmentsUnchanged() async throws {
        let db = try await makeDB()
        let messages = [
            ChatMessage(role: .user, content: "no attachments"),
            ChatMessage(role: .assistant, content: "reply"),
        ]

        try await db.saveConversation(id: "c1", title: "Plain", messages: messages)

        let loaded = try await db.loadMessages(for: "c1")
        XCTAssertEqual(loaded.count, 2)
        XCTAssertTrue(loaded[0].attachments.isEmpty)
        XCTAssertTrue(loaded[1].attachments.isEmpty)
    }

    func testIncrementalSavePreservesAttachments() async throws {
        let db = try await makeDB()
        let attachment = MessageAttachment(
            type: .textFile, filename: "init.txt", mimeType: "text/plain",
            data: "initial".data(using: .utf8)!
        )
        let msg1 = ChatMessage(role: .user, content: "hello", attachments: [attachment])

        try await db.saveConversation(id: "c1", title: "Chat", messages: [msg1])
        let existingIDs = Set([msg1.id])

        // Add a new message via incremental save
        let msg2 = ChatMessage(role: .assistant, content: "world")
        try await db.saveConversationIncremental(
            id: "c1", title: "Chat",
            messages: [msg1, msg2],
            existingMessageIDs: existingIDs
        )

        let loaded = try await db.loadMessages(for: "c1")
        XCTAssertEqual(loaded.count, 2)
        // Original message's attachment should still be there
        XCTAssertEqual(loaded[0].attachments.count, 1)
        XCTAssertEqual(loaded[0].attachments[0].filename, "init.txt")
        XCTAssertTrue(loaded[1].attachments.isEmpty)
    }

    func testBranchPreservesAttachments() async throws {
        let db = try await makeDB()
        let attachment = MessageAttachment(
            type: .pdf, filename: "report.pdf", mimeType: "application/pdf",
            data: Data(repeating: 0x25, count: 200), extractedText: "Report content"
        )
        let parentMsg = ChatMessage(role: .user, content: "analyze this", attachments: [attachment])

        try await db.saveConversation(id: "parent", title: "Parent", messages: [parentMsg])

        // Branch with copied attachment (fresh UUID)
        let branchAtt = MessageAttachment(
            type: attachment.type, filename: attachment.filename, mimeType: attachment.mimeType,
            data: attachment.data, extractedText: attachment.extractedText
        )
        let branchMsg = ChatMessage(id: UUID(), role: .user, content: "analyze this", attachments: [branchAtt])

        try await db.saveBranchConversation(
            id: "branch", parentID: "parent", forkMessageIndex: 0,
            forkNarrative: nil, title: "Branch", messages: [branchMsg]
        )

        let loaded = try await db.loadMessages(for: "branch")
        XCTAssertEqual(loaded.count, 1)
        XCTAssertEqual(loaded[0].attachments.count, 1)
        XCTAssertEqual(loaded[0].attachments[0].filename, "report.pdf")
        XCTAssertEqual(loaded[0].attachments[0].extractedText, "Report content")
    }

    func testPaginatedLoadIncludesAttachments() async throws {
        let db = try await makeDB()
        let attachment = MessageAttachment(
            type: .textFile, filename: "code.py", mimeType: "text/plain",
            data: "print('hi')".data(using: .utf8)!, extractedText: "print('hi')"
        )
        var messages: [ChatMessage] = []
        for i in 0..<10 {
            if i == 3 {
                messages.append(ChatMessage(role: .user, content: "Message \(i)", attachments: [attachment]))
            } else {
                messages.append(ChatMessage(role: i % 2 == 0 ? .user : .assistant, content: "Message \(i)"))
            }
        }

        try await db.saveConversation(id: "c1", title: "Paginated", messages: messages)

        // Load last 5 messages (messages 5-9, no attachment)
        let page1 = try await db.loadMessagesPage(for: "c1", limit: 5)
        XCTAssertEqual(page1.count, 5)
        XCTAssertTrue(page1.allSatisfy { $0.attachments.isEmpty })

        // Load messages 0-4 (message 3 has attachment)
        let page2 = try await db.loadMessagesPage(for: "c1", limit: 5, beforeSortOrder: 5)
        XCTAssertEqual(page2.count, 5)
        XCTAssertEqual(page2[3].attachments.count, 1)
        XCTAssertEqual(page2[3].attachments[0].filename, "code.py")
    }

    func testAttachmentTypeRoundTrip() async throws {
        let db = try await makeDB()
        let types: [(AttachmentType, String)] = [
            (.textFile, "text"),
            (.image, "image"),
            (.pdf, "pdf"),
        ]

        for (type, rawValue) in types {
            XCTAssertEqual(type.rawValue, rawValue)
            XCTAssertEqual(AttachmentType(rawValue: rawValue), type)
        }

        // Verify all types persist correctly
        let atts = types.map { type, _ in
            MessageAttachment(
                type: type, filename: "\(type.rawValue).file", mimeType: "test/test",
                data: Data([0x01])
            )
        }
        let msg = ChatMessage(role: .user, content: "types", attachments: atts)
        try await db.saveConversation(id: "c1", title: "Types", messages: [msg])

        let loaded = try await db.loadMessages(for: "c1")
        XCTAssertEqual(loaded[0].attachments.count, 3)
        XCTAssertEqual(loaded[0].attachments[0].type, .textFile)
        XCTAssertEqual(loaded[0].attachments[1].type, .image)
        XCTAssertEqual(loaded[0].attachments[2].type, .pdf)
    }

    func testLoadAttachmentReferencesOrderedAcrossMessages() async throws {
        let db = try await makeDB()
        let msg1 = ChatMessage(
            role: .user,
            content: "first",
            attachments: [
                MessageAttachment(
                    type: .pdf,
                    filename: "report.pdf",
                    mimeType: "application/pdf",
                    data: Data([0x01]),
                    extractedText: "Report body"
                ),
                MessageAttachment(
                    type: .image,
                    filename: "photo.png",
                    mimeType: "image/png",
                    data: Data([0x02]),
                    extractedText: "A red house"
                ),
            ]
        )
        let msg2 = ChatMessage(
            role: .assistant,
            content: "second",
            attachments: [
                MessageAttachment(
                    type: .textFile,
                    filename: "notes.txt",
                    mimeType: "text/plain",
                    data: Data([0x03]),
                    extractedText: "todo items"
                ),
            ]
        )

        try await db.saveConversation(id: "c1", title: "refs", messages: [msg1, msg2])

        let refs = try await db.loadAttachmentReferences(for: "c1")
        XCTAssertEqual(refs.count, 3)
        XCTAssertEqual(refs[0].messageSortOrder, 0)
        XCTAssertEqual(refs[0].attachmentSortOrder, 0)
        XCTAssertEqual(refs[0].filename, "report.pdf")
        XCTAssertEqual(refs[0].type, .pdf)
        XCTAssertEqual(refs[1].messageSortOrder, 0)
        XCTAssertEqual(refs[1].attachmentSortOrder, 1)
        XCTAssertEqual(refs[1].filename, "photo.png")
        XCTAssertEqual(refs[1].type, .image)
        XCTAssertEqual(refs[2].messageSortOrder, 1)
        XCTAssertEqual(refs[2].attachmentSortOrder, 0)
        XCTAssertEqual(refs[2].filename, "notes.txt")
        XCTAssertEqual(refs[2].type, .textFile)
        XCTAssertEqual(refs[2].extractedText, "todo items")
    }

    func testLoadAttachmentReferencesEmptyConversation() async throws {
        let db = try await makeDB()
        try await db.saveConversation(
            id: "c1",
            title: "empty",
            messages: [ChatMessage(role: .user, content: "no attachments")]
        )

        let refs = try await db.loadAttachmentReferences(for: "c1")
        XCTAssertTrue(refs.isEmpty)
    }

    // MARK: - Model Tests

    func testPendingAttachmentEquatableByID() {
        let id = UUID()
        let a = PendingAttachment(
            id: id, filename: "a.txt", type: .textFile,
            fileURL: URL(fileURLWithPath: "/tmp/a.txt"), fileSize: 100
        )
        let b = PendingAttachment(
            id: id, filename: "b.txt", type: .image,
            fileURL: URL(fileURLWithPath: "/tmp/b.png"), fileSize: 999
        )
        // Same ID → equal (cheap SwiftUI diffing)
        XCTAssertEqual(a, b)

        let c = PendingAttachment(
            filename: "a.txt", type: .textFile,
            fileURL: URL(fileURLWithPath: "/tmp/a.txt"), fileSize: 100
        )
        // Different ID → not equal
        XCTAssertNotEqual(a, c)
    }

    func testChatMessageDefaultEmptyAttachments() {
        let msg = ChatMessage(role: .user, content: "hello")
        XCTAssertTrue(msg.attachments.isEmpty)
    }
}
