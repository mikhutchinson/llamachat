import Testing
import Foundation
@testable import LlamaInferenceCore

// MARK: - Model Hub Search Parsing

@Suite("Model Hub Search Parser")
struct ModelHubSearchParserTests {

    @Test("Parses provider after query")
    func providerAfterQuery() {
        let parsed = ModelHubSearchParser.parse("nanbeige @bartowski")
        #expect(parsed.query == "nanbeige")
        #expect(parsed.authorOverride == "bartowski")
    }

    @Test("Parses provider before query")
    func providerBeforeQuery() {
        let parsed = ModelHubSearchParser.parse("@bartowski nanbeige")
        #expect(parsed.query == "nanbeige")
        #expect(parsed.authorOverride == "bartowski")
    }

    @Test("Parses provider-only search")
    func providerOnly() {
        let parsed = ModelHubSearchParser.parse("@bartowski")
        #expect(parsed.query == "")
        #expect(parsed.authorOverride == "bartowski")
    }

    @Test("Uses first provider when multiple are present")
    func firstProviderWins() {
        let parsed = ModelHubSearchParser.parse("@bartowski @lmstudio-community")
        #expect(parsed.query == "")
        #expect(parsed.authorOverride == "bartowski")
    }

    @Test("Leaves normal query unchanged")
    func plainQuery() {
        let parsed = ModelHubSearchParser.parse("nanbeige")
        #expect(parsed.query == "nanbeige")
        #expect(parsed.authorOverride == nil)
    }

    @Test("Handles empty input")
    func emptyInput() {
        let parsed = ModelHubSearchParser.parse("")
        #expect(parsed.query == "")
        #expect(parsed.authorOverride == nil)
    }

    @Test("Handles whitespace-only input")
    func whitespaceOnlyInput() {
        let parsed = ModelHubSearchParser.parse("   \n\t  ")
        #expect(parsed.query == "")
        #expect(parsed.authorOverride == nil)
    }

    @Test("Ignores bare @ token")
    func bareAtToken() {
        let parsed = ModelHubSearchParser.parse("@")
        #expect(parsed.query == "")
        #expect(parsed.authorOverride == nil)
    }

    @Test("Ignores trailing bare @ token")
    func trailingBareAtToken() {
        let parsed = ModelHubSearchParser.parse("nanbeige @")
        #expect(parsed.query == "nanbeige")
        #expect(parsed.authorOverride == nil)
    }
}

// MARK: - Model Hub Persisted Filter State

@Suite("Model Hub Persisted Filter State")
struct ModelHubPersistedFilterStateTests {

    @Test("HubSortOrder restores valid stored value")
    func restoresValidSort() {
        #expect(HubSortOrder.fromStored("likes") == .likes)
        #expect(HubSortOrder.fromStored("lastModified") == .lastModified)
    }

    @Test("HubSortOrder falls back for unknown value")
    func fallbackSort() {
        #expect(HubSortOrder.fromStored("unknown") == .downloads)
    }

    @Test("HubSourceFilter restores valid stored value")
    func restoresValidSource() {
        #expect(HubSourceFilter.fromStored("all") == .all)
        #expect(HubSourceFilter.fromStored("recommended") == .recommended)
    }

    @Test("HubSourceFilter falls back for unknown value")
    func fallbackSource() {
        #expect(HubSourceFilter.fromStored("invalid") == .recommended)
    }
}

// MARK: - Model Hub Selection Policy

@Suite("Model Hub Selection Policy")
struct ModelHubSelectionPolicyTests {

    @Test("Keeps current selection when still present")
    func keepCurrentSelection() {
        let models = [
            makeSummary(id: "author/a"),
            makeSummary(id: "author/b"),
            makeSummary(id: "author/c"),
        ]
        let selected = ModelHubSelectionPolicy.nextSelection(current: "author/b", available: models)
        #expect(selected == "author/b")
    }

    @Test("Falls back to first item when current missing")
    func fallbackToFirstSelection() {
        let models = [
            makeSummary(id: "author/a"),
            makeSummary(id: "author/b"),
        ]
        let selected = ModelHubSelectionPolicy.nextSelection(current: "author/x", available: models)
        #expect(selected == "author/a")
    }

    @Test("Returns nil when list is empty")
    func noSelectionForEmptyList() {
        let selected = ModelHubSelectionPolicy.nextSelection(current: "author/a", available: [])
        #expect(selected == nil)
    }

    private func makeSummary(id: String) -> HFModelSummary {
        HFModelSummary(
            id: id,
            likes: 0,
            downloads: 0,
            tags: ["gguf"],
            pipelineTag: nil,
            createdAt: nil
        )
    }
}

// MARK: - GGUF Model Info (base name for downloaded matching)

@Suite("GGUFModelInfo Base Name")
struct GGUFModelInfoBaseNameTests {

    @Test("baseName(fromFilename:) strips path, .gguf, and quant")
    func baseNameFromFilename() {
        #expect(GGUFModelInfo.baseName(fromFilename: "Qwen3-4B-Q4_K_M.gguf") == "Qwen3-4B")
        #expect(GGUFModelInfo.baseName(fromFilename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf") == "Llama-3.2-1B-Instruct")
        #expect(GGUFModelInfo.baseName(fromFilename: "gemma-3-4b-it-Q4_K_M.gguf") == "gemma-3-4b-it")
        #expect(GGUFModelInfo.baseName(fromFilename: "model-Q8_0.gguf") == "model")
    }

    @Test("baseName(fromFilename:) strips path when present")
    func baseNameStripsPath() {
        let withPath = "lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf"
        #expect(GGUFModelInfo.baseName(fromFilename: withPath) == "gemma-3-4b-it")
    }

    @Test("baseNameFromHFModel strips -GGUF suffix")
    func baseNameFromHFModel() {
        #expect(GGUFModelInfo.baseNameFromHFModel("Qwen3-4B-GGUF") == "Qwen3-4B")
        #expect(GGUFModelInfo.baseNameFromHFModel("Qwen3-4B-Thinking-2507-GGUF") == "Qwen3-4B-Thinking-2507")
        #expect(GGUFModelInfo.baseNameFromHFModel("gemma-3-4b-it-GGUF") == "gemma-3-4b-it")
        #expect(GGUFModelInfo.baseNameFromHFModel("Llama-3.2-1B-Instruct-GGUF") == "Llama-3.2-1B-Instruct")
    }

    @Test("baseNameFromHFModel handles case variants")
    func baseNameFromHFModelCaseVariants() {
        #expect(GGUFModelInfo.baseNameFromHFModel("model-gguf") == "model")
        #expect(GGUFModelInfo.baseNameFromHFModel("model-Gguf") == "model")
    }

    @Test("baseNameFromHFModel returns unchanged when no -GGUF suffix")
    func baseNameFromHFModelNoSuffix() {
        #expect(GGUFModelInfo.baseNameFromHFModel("some-model-name") == "some-model-name")
    }

    @Test("exact match: Qwen3-4B file matches Qwen3-4B-GGUF not Thinking or Instruct")
    func exactMatchPreventsFalsePositives() {
        let fileBase = GGUFModelInfo.baseName(fromFilename: "Qwen3-4B-Q4_K_M.gguf")
        #expect(fileBase == "Qwen3-4B")
        #expect(fileBase.caseInsensitiveCompare(GGUFModelInfo.baseNameFromHFModel("Qwen3-4B-GGUF")) == .orderedSame)
        #expect(fileBase.caseInsensitiveCompare(GGUFModelInfo.baseNameFromHFModel("Qwen3-4B-Thinking-2507-GGUF")) != .orderedSame)
        #expect(fileBase.caseInsensitiveCompare(GGUFModelInfo.baseNameFromHFModel("Qwen3-4B-Instruct-2507-GGUF")) != .orderedSame)
    }

    @Test("parse produces correct displayName and quantization")
    func parseDisplayName() {
        let info = GGUFModelInfo.parse(filename: "Qwen3-4B-Q4_K_M.gguf", sizeBytes: 2_500_000_000)
        #expect(info.displayName == "Qwen3 4B")
        #expect(info.quantization == "Q4_K_M")

        let info2 = GGUFModelInfo.parse(filename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf", sizeBytes: 800_000_000)
        #expect(info2.displayName == "Llama 3.2 1B Instruct")
        #expect(info2.quantization == "Q4_K_M")
    }
}

// MARK: - Quant Level Detection

@Suite("GGUF Quant Level Detection")
struct QuantLevelDetectionTests {

    @Test("Detects standard quant levels from filenames")
    func standardQuantLevels() {
        let cases: [(String, QuantLevel)] = [
            ("Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf", .Q4_K_M),
            ("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", .Q8_0),
            ("model-Q2_K.gguf", .Q2_K),
            ("model-Q3_K_S.gguf", .Q3_K_S),
            ("model-Q3_K_M.gguf", .Q3_K_M),
            ("model-Q3_K_L.gguf", .Q3_K_L),
            ("model-Q4_0.gguf", .Q4_0),
            ("model-Q4_K_S.gguf", .Q4_K_S),
            ("model-Q4_K_L.gguf", .Q4_K_L),
            ("model-Q5_K_S.gguf", .Q5_K_S),
            ("model-Q5_K_M.gguf", .Q5_K_M),
            ("model-Q5_K_L.gguf", .Q5_K_L),
            ("model-Q6_K.gguf", .Q6_K),
            ("model-f16.gguf", .f16),
            ("model-bf16.gguf", .bf16),
        ]
        for (filename, expected) in cases {
            let detected = QuantLevel.detect(from: filename)
            #expect(detected == expected, "Expected \(expected) for '\(filename)', got \(String(describing: detected))")
        }
    }

    @Test("Handles underscore-separated quant in filename")
    func underscoreSeparated() {
        let result = QuantLevel.detect(from: "some_model_Q5_K_M.gguf")
        #expect(result == .Q5_K_M)
    }

    @Test("Returns nil for non-quantized filenames")
    func noQuantLevel() {
        #expect(QuantLevel.detect(from: "readme.md") == nil)
        #expect(QuantLevel.detect(from: "config.json") == nil)
        #expect(QuantLevel.detect(from: "tokenizer.gguf") == nil)
    }

    @Test("Case insensitive detection")
    func caseInsensitive() {
        #expect(QuantLevel.detect(from: "model-q4_k_m.gguf") == .Q4_K_M)
        #expect(QuantLevel.detect(from: "model-q8_0.gguf") == .Q8_0)
    }

    @Test("Prefers longer match (Q4_K_M over Q4_K or Q4_0)")
    func longerMatchPreferred() {
        let result = QuantLevel.detect(from: "model-Q4_K_M.gguf")
        #expect(result == .Q4_K_M)
    }

    @Test("Quant tier classification")
    func tierClassification() {
        #expect(QuantLevel.Q2_K.tier == "Tiny")
        #expect(QuantLevel.Q4_K_M.tier == "Medium")
        #expect(QuantLevel.Q5_K_M.tier == "Good")
        #expect(QuantLevel.Q8_0.tier == "High")
        #expect(QuantLevel.f16.tier == "Full")
    }
}

// MARK: - GGUF File Parsing

@Suite("GGUF File Parsing")
struct GGUFFileParsingTests {

    @Test("Parses standard GGUF sibling")
    func standardFile() {
        let sibling = HFSibling(rfilename: "Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf", lfs: nil)
        let file = GGUFFile.from(sibling: sibling, repoId: "lmstudio-community/Qwen3-VL-4B-Instruct-GGUF")
        #expect(file != nil)
        #expect(file?.filename == "Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf")
        #expect(file?.quantLevel == .Q4_K_M)
        #expect(file?.isMMProj == false)
        #expect(file?.repoId == "lmstudio-community/Qwen3-VL-4B-Instruct-GGUF")
    }

    @Test("Detects mmproj files")
    func mmprojFile() {
        let sibling = HFSibling(rfilename: "mmproj-Qwen_Qwen3-VL-4B-Instruct-f16.gguf", lfs: nil)
        let file = GGUFFile.from(sibling: sibling, repoId: "test/repo")
        #expect(file != nil)
        #expect(file?.isMMProj == true)
        #expect(file?.quantLevel == .f16)
    }

    @Test("Detects mmproj when token is not prefix")
    func mmprojTokenInMiddle() {
        let sibling = HFSibling(rfilename: "gemma-3-4b-it-mmproj-f16.gguf", lfs: nil)
        let file = GGUFFile.from(sibling: sibling, repoId: "test/repo")
        #expect(file != nil)
        #expect(file?.isMMProj == true)
        #expect(file?.quantLevel == .f16)
    }

    @Test("Skips non-GGUF files")
    func nonGGUF() {
        let cases = ["README.md", "config.json", "tokenizer.model", ".gitattributes"]
        for name in cases {
            let sibling = HFSibling(rfilename: name, lfs: nil)
            #expect(GGUFFile.from(sibling: sibling, repoId: "test/repo") == nil)
        }
    }

    @Test("Download URL is correct")
    func downloadURL() {
        let sibling = HFSibling(rfilename: "model-Q4_K_M.gguf", lfs: nil)
        let file = GGUFFile.from(sibling: sibling, repoId: "author/model-GGUF")!
        #expect(file.downloadURL.absoluteString == "https://huggingface.co/author/model-GGUF/resolve/main/model-Q4_K_M.gguf")
    }
}

// MARK: - VLM Bundle Detection

@Suite("VLM Bundle Detection")
struct VLMBundleDetectionTests {

    @Test("Separates main GGUF and mmproj files")
    func bundleSeparation() {
        let siblings: [HFSibling] = [
            HFSibling(rfilename: "Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf", lfs: nil),
            HFSibling(rfilename: "Qwen_Qwen3-VL-4B-Instruct-Q8_0.gguf", lfs: nil),
            HFSibling(rfilename: "mmproj-Qwen_Qwen3-VL-4B-Instruct-f16.gguf", lfs: nil),
            HFSibling(rfilename: "README.md", lfs: nil),
            HFSibling(rfilename: "config.json", lfs: nil),
        ]
        let files = siblings.compactMap { GGUFFile.from(sibling: $0, repoId: "test/repo") }
        let mainFiles = files.filter { !$0.isMMProj }
        let projFiles = files.filter { $0.isMMProj }

        #expect(mainFiles.count == 2)
        #expect(projFiles.count == 1)
        #expect(projFiles[0].filename.hasPrefix("mmproj"))
    }

    @Test("Non-VLM repo has no mmproj files")
    func noMmproj() {
        let siblings: [HFSibling] = [
            HFSibling(rfilename: "model-Q4_K_M.gguf", lfs: nil),
            HFSibling(rfilename: "model-Q8_0.gguf", lfs: nil),
        ]
        let files = siblings.compactMap { GGUFFile.from(sibling: $0, repoId: "test/repo") }
        #expect(files.allSatisfy { !$0.isMMProj })
    }
}

// MARK: - Model Role Suggestion

@Suite("Model Role Suggestion")
struct ModelRoleSuggestionTests {

    @Test("mmproj file suggests VLM Projection")
    func mmprojRole() {
        let file = GGUFFile(id: "mmproj-f16.gguf", filename: "mmproj-f16.gguf",
                            repoId: "test/repo", quantLevel: .f16,
                            isMMProj: true, estimatedSize: nil, expectedSHA256: nil)
        #expect(ModelRole.suggest(for: file, isVLMRepo: true) == .vlmProjection)
        #expect(ModelRole.suggest(for: file, isVLMRepo: false) == .vlmProjection)
    }

    @Test("VLM repo non-mmproj suggests VLM Model")
    func vlmModelRole() {
        let file = GGUFFile(id: "model-Q4_K_M.gguf", filename: "model-Q4_K_M.gguf",
                            repoId: "test/repo", quantLevel: .Q4_K_M,
                            isMMProj: false, estimatedSize: nil, expectedSHA256: nil)
        #expect(ModelRole.suggest(for: file, isVLMRepo: true) == .vlmModel)
    }

    @Test("Non-VLM repo suggests Chat Model")
    func chatModelRole() {
        let file = GGUFFile(id: "model-Q4_K_M.gguf", filename: "model-Q4_K_M.gguf",
                            repoId: "test/repo", quantLevel: .Q4_K_M,
                            isMMProj: false, estimatedSize: nil, expectedSHA256: nil)
        #expect(ModelRole.suggest(for: file, isVLMRepo: false) == .chatModel)
    }

    @Test("summarization pipeline_tag suggests Summarizer")
    func summarizerFromPipeline() {
        let file = GGUFFile(id: "model-Q4_K_M.gguf", filename: "model-Q4_K_M.gguf",
                            repoId: "test/repo", quantLevel: .Q4_K_M,
                            isMMProj: false, estimatedSize: nil, expectedSHA256: nil)
        #expect(ModelRole.suggest(for: file, isVLMRepo: false, pipelineTag: "summarization") == .summarizer)
    }

    @Test("image-text-to-text pipeline_tag suggests VLM Model")
    func vlmFromPipeline() {
        let file = GGUFFile(id: "model-Q4_K_M.gguf", filename: "model-Q4_K_M.gguf",
                            repoId: "test/repo", quantLevel: .Q4_K_M,
                            isMMProj: false, estimatedSize: nil, expectedSHA256: nil)
        #expect(ModelRole.suggest(for: file, isVLMRepo: false, pipelineTag: "image-text-to-text") == .vlmModel)
    }

    @Test("text-generation pipeline_tag falls through to Chat Model")
    func chatFromPipeline() {
        let file = GGUFFile(id: "model-Q4_K_M.gguf", filename: "model-Q4_K_M.gguf",
                            repoId: "test/repo", quantLevel: .Q4_K_M,
                            isMMProj: false, estimatedSize: nil, expectedSHA256: nil)
        #expect(ModelRole.suggest(for: file, isVLMRepo: false, pipelineTag: "text-generation") == .chatModel)
    }

    @Test("mmproj overrides pipeline_tag")
    func mmprojOverridesPipeline() {
        let file = GGUFFile(id: "mmproj-f16.gguf", filename: "mmproj-f16.gguf",
                            repoId: "test/repo", quantLevel: .f16,
                            isMMProj: true, estimatedSize: nil, expectedSHA256: nil)
        #expect(ModelRole.suggest(for: file, isVLMRepo: false, pipelineTag: "summarization") == .vlmProjection)
    }

    @Test("isVLMRepo overrides pipeline_tag")
    func vlmRepoOverridesPipeline() {
        let file = GGUFFile(id: "model-Q4_K_M.gguf", filename: "model-Q4_K_M.gguf",
                            repoId: "test/repo", quantLevel: .Q4_K_M,
                            isMMProj: false, estimatedSize: nil, expectedSHA256: nil)
        #expect(ModelRole.suggest(for: file, isVLMRepo: true, pipelineTag: "summarization") == .vlmModel)
    }
}

// MARK: - Pipeline Description

@Suite("Pipeline Description")
struct PipelineDescriptionTests {

    @Test("Known pipeline tags map to friendly names")
    func knownTags() {
        #expect(HFModelSummary.descriptionForTag("text-generation") == "Text Generation")
        #expect(HFModelSummary.descriptionForTag("image-text-to-text") == "Vision-Language")
        #expect(HFModelSummary.descriptionForTag("summarization") == "Summarization")
        #expect(HFModelSummary.descriptionForTag("conversational") == "Conversational")
    }

    @Test("Unknown pipeline tag is capitalized with dashes replaced")
    func unknownTag() {
        #expect(HFModelSummary.descriptionForTag("custom-new-task") == "Custom New Task")
    }
}

// MARK: - HF API Response Parsing

@Suite("HF API Response Parsing")
struct HFAPIParsingTests {

    @Test("Decodes HFModelSummary from JSON")
    func decodeSummary() throws {
        let json = """
        {
            "id": "lmstudio-community/Qwen3-VL-4B-Instruct-GGUF",
            "likes": 2,
            "downloads": 42686,
            "tags": ["gguf", "vision"],
            "pipeline_tag": "image-text-to-text",
            "created_at": "2025-10-30T20:21:34.000Z"
        }
        """
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let summary = try decoder.decode(HFModelSummary.self, from: json.data(using: .utf8)!)

        #expect(summary.id == "lmstudio-community/Qwen3-VL-4B-Instruct-GGUF")
        #expect(summary.author == "lmstudio-community")
        #expect(summary.modelName == "Qwen3-VL-4B-Instruct-GGUF")
        #expect(summary.likes == 2)
        #expect(summary.downloads == 42686)
        #expect(summary.isVLM == true)
    }

    @Test("Decodes HFModelSummary without optional fields")
    func decodeSummaryMinimal() throws {
        let json = """
        {
            "id": "user/model",
            "likes": 0,
            "downloads": 100,
            "tags": ["gguf"]
        }
        """
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let summary = try decoder.decode(HFModelSummary.self, from: json.data(using: .utf8)!)

        #expect(summary.pipelineTag == nil)
        #expect(summary.createdAt == nil)
        #expect(summary.isVLM == false)
    }

    @Test("Decodes HFModelDetail with siblings and gguf info")
    func decodeDetail() throws {
        let json = """
        {
            "id": "lmstudio-community/Qwen3-VL-4B-Instruct-GGUF",
            "siblings": [
                {"rfilename": "Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf"},
                {"rfilename": "mmproj-Qwen_Qwen3-VL-4B-Instruct-f16.gguf"},
                {"rfilename": "README.md"}
            ],
            "gguf": {
                "total": 4022468096,
                "architecture": "qwen3vl",
                "context_length": 262144
            },
            "likes": 2,
            "downloads": 42686,
            "tags": ["gguf"],
            "pipeline_tag": "image-text-to-text"
        }
        """
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let detail = try decoder.decode(HFModelDetail.self, from: json.data(using: .utf8)!)

        #expect(detail.siblings.count == 3)
        #expect(detail.gguf?.architecture == "qwen3vl")
        #expect(detail.gguf?.contextLength == 262144)
        #expect(detail.gguf?.total == 4022468096)
        #expect(detail.isVLM == true)
    }

    @Test("Decodes HFModelDetail without gguf info")
    func decodeDetailNoGGUF() throws {
        let json = """
        {
            "id": "user/model",
            "siblings": [],
            "likes": 0,
            "downloads": 0,
            "tags": ["gguf"]
        }
        """
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let detail = try decoder.decode(HFModelDetail.self, from: json.data(using: .utf8)!)

        #expect(detail.gguf == nil)
        #expect(detail.cardData == nil)
        #expect(detail.pipelineTag == nil)
    }
}

// MARK: - VLM Bundle Selection

@Suite("VLM Bundle File Selection")
struct VLMBundleSelectionTests {

    private func makeFile(_ name: String, quant: QuantLevel?, mmproj: Bool = false) -> GGUFFile {
        GGUFFile(id: name, filename: name, repoId: "test/vlm-repo",
                 quantLevel: quant, isMMProj: mmproj, estimatedSize: nil, expectedSHA256: nil)
    }

    @Test("Prefers Q4_K_M for bundle main file")
    func prefersQ4KM() {
        let files = [
            makeFile("model-Q3_K_M.gguf", quant: .Q3_K_M),
            makeFile("model-Q4_K_M.gguf", quant: .Q4_K_M),
            makeFile("model-Q8_0.gguf", quant: .Q8_0),
        ]
        let preferred = files.first(where: { $0.quantLevel == .Q4_K_M })
            ?? files.first(where: { $0.quantLevel == .Q4_K_S })
            ?? files.first!
        #expect(preferred.quantLevel == .Q4_K_M)
    }

    @Test("Falls back to Q4_K_S when no Q4_K_M")
    func fallsBackToQ4KS() {
        let files = [
            makeFile("model-Q3_K_M.gguf", quant: .Q3_K_M),
            makeFile("model-Q4_K_S.gguf", quant: .Q4_K_S),
            makeFile("model-Q8_0.gguf", quant: .Q8_0),
        ]
        let preferred = files.first(where: { $0.quantLevel == .Q4_K_M })
            ?? files.first(where: { $0.quantLevel == .Q4_K_S })
            ?? files.first!
        #expect(preferred.quantLevel == .Q4_K_S)
    }

    @Test("Falls back to first file when no Q4 variant")
    func fallsBackToFirst() {
        let files = [
            makeFile("model-Q8_0.gguf", quant: .Q8_0),
            makeFile("model-Q6_K.gguf", quant: .Q6_K),
        ]
        let preferred = files.first(where: { $0.quantLevel == .Q4_K_M })
            ?? files.first(where: { $0.quantLevel == .Q4_K_S })
            ?? files.first!
        #expect(preferred.quantLevel == .Q8_0)
    }

    @Test("Bundle identifies main + proj files correctly")
    func bundleSplitForVLM() {
        let files = [
            makeFile("model-Q4_K_M.gguf", quant: .Q4_K_M),
            makeFile("model-Q8_0.gguf", quant: .Q8_0),
            makeFile("mmproj-model-f16.gguf", quant: .f16, mmproj: true),
        ]
        let mainFiles = files.filter { !$0.isMMProj }
        let projFiles = files.filter { $0.isMMProj }
        #expect(mainFiles.count == 2)
        #expect(projFiles.count == 1)
        #expect(projFiles[0].isMMProj == true)
    }
}

// MARK: - Delete Downloaded

@Suite("Delete Downloaded")
struct DeleteDownloadedTests {

    @Test("deleteDownloaded returns false for non-existent file")
    @MainActor
    func deleteNonExistent() {
        let manager = ModelDownloadManager()
        let result = manager.deleteDownloaded(filename: "nonexistent-file-xyz.gguf")
        #expect(result == false)
    }

    @Test("deleteDownloaded removes file from disk and returns true")
    @MainActor
    func deleteExistingFile() throws {
        let manager = ModelDownloadManager()
        // Create a temp file in the download directory
        let fm = FileManager.default
        if !fm.fileExists(atPath: manager.downloadDirectory.path) {
            try fm.createDirectory(at: manager.downloadDirectory, withIntermediateDirectories: true)
        }
        let testFile = manager.downloadDirectory.appendingPathComponent("__test_delete__.gguf")
        fm.createFile(atPath: testFile.path, contents: Data("test".utf8))
        #expect(fm.fileExists(atPath: testFile.path))

        let result = manager.deleteDownloaded(filename: "__test_delete__.gguf")
        #expect(result == true)
        #expect(!fm.fileExists(atPath: testFile.path))
    }
}

// MARK: - SHA256 Verification

@Suite("SHA256 Verification")
struct SHA256VerificationTests {

    @Test("Verifies correct SHA256")
    func correctHash() throws {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("sha256_test_\(UUID().uuidString).bin")
        let content = Data("Hello, SHA256!".utf8)
        fm.createFile(atPath: tmp.path, contents: content)
        defer { try? fm.removeItem(at: tmp) }

        // Pre-computed SHA256 of "Hello, SHA256!"
        // echo -n "Hello, SHA256!" | shasum -a 256
        let expected = "d14c0cd6caa3d2006a34ba29886d8b2e698e9d2259be8b1f852375d6026502bd"
        #expect(ModelDownloadManager.verifySHA256(path: tmp.path, expected: expected) == true)
    }

    @Test("Rejects incorrect SHA256")
    func incorrectHash() throws {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("sha256_bad_\(UUID().uuidString).bin")
        fm.createFile(atPath: tmp.path, contents: Data("test data".utf8))
        defer { try? fm.removeItem(at: tmp) }

        #expect(ModelDownloadManager.verifySHA256(path: tmp.path, expected: "0000000000000000") == false)
    }

    @Test("Returns false for non-existent file")
    func nonExistentFile() {
        #expect(ModelDownloadManager.verifySHA256(path: "/tmp/nonexistent_\(UUID().uuidString)", expected: "abc") == false)
    }
}

// MARK: - LFS Info Parsing

@Suite("LFS Info Parsing")
struct LFSInfoTests {

    @Test("GGUFFile picks up LFS size and sha256")
    func lfsFields() {
        let lfs = HFLfsInfo(sha256: "abc123def456", size: 4_000_000_000)
        let sibling = HFSibling(rfilename: "model-Q4_K_M.gguf", lfs: lfs)
        let file = GGUFFile.from(sibling: sibling, repoId: "author/repo")
        #expect(file?.estimatedSize == 4_000_000_000)
        #expect(file?.expectedSHA256 == "abc123def456")
    }

    @Test("GGUFFile handles nil LFS gracefully")
    func nilLfs() {
        let sibling = HFSibling(rfilename: "model-Q8_0.gguf", lfs: nil)
        let file = GGUFFile.from(sibling: sibling, repoId: "author/repo")
        #expect(file?.estimatedSize == nil)
        #expect(file?.expectedSHA256 == nil)
    }
}

// MARK: - Download Task

@Suite("Download Task")
struct DownloadTaskTests {

    @Test("Progress calculation")
    func progress() {
        var task = DownloadTask(id: "test.gguf", filename: "test.gguf", repoId: "a/b",
                                totalBytes: 1000, downloadedBytes: 500, state: .downloading)
        #expect(task.progress == 0.5)

        task.downloadedBytes = 0
        #expect(task.progress == 0.0)

        task.downloadedBytes = 1000
        #expect(task.progress == 1.0)
    }

    @Test("Progress with zero total bytes")
    func progressZeroTotal() {
        let task = DownloadTask(id: "test.gguf", filename: "test.gguf", repoId: "a/b",
                                totalBytes: 0, downloadedBytes: 100, state: .downloading)
        #expect(task.progress == 0.0)
    }
}
