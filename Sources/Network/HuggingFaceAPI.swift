import Foundation
import OSLog

/// Client for the Hugging Face public REST API (models endpoint).
/// All requests are unauthenticated — public repos only.
public actor HuggingFaceAPI {
    public static let shared = HuggingFaceAPI()

    private let session = URLSession.shared
    private let baseURL = "https://huggingface.co/api/models"
    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "HuggingFaceAPI")

    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    // MARK: - Search

    /// A page of search results with an optional cursor URL for the next page.
    public struct SearchPage: Sendable {
        public let models: [HFModelSummary]
        public let nextPageURL: URL?
    }

    /// Search for GGUF models on Hugging Face.
    /// - Parameters:
    ///   - query: Search text (model name, author, etc.)
    ///   - author: Scope to a specific author (default: lmstudio-community)
    ///   - sort: Sort field (downloads, likes, lastModified)
    ///   - limit: Max results per page
    public func search(
        query: String,
        author: String? = "lmstudio-community",
        sort: String = "downloads",
        limit: Int = 20
    ) async throws -> SearchPage {
        var components = URLComponents(string: baseURL)!
        var items: [URLQueryItem] = [
            URLQueryItem(name: "filter", value: "gguf"),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: String(limit)),
        ]
        if !query.isEmpty {
            items.append(URLQueryItem(name: "search", value: query))
        }
        if let author, !author.isEmpty {
            items.append(URLQueryItem(name: "author", value: author))
        }
        components.queryItems = items

        guard let url = components.url else {
            throw HuggingFaceAPIError.invalidURL
        }

        return try await fetchPage(url: url)
    }

    /// Fetch a page of models from a direct URL (used for cursor-based pagination).
    public func fetchPage(url: URL) async throws -> SearchPage {
        logger.debug("Fetching page: \(url.absoluteString, privacy: .public)")
        let (data, response) = try await session.data(from: url)
        try Self.validateResponse(response)
        let models = try decoder.decode([HFModelSummary].self, from: data)
        let nextURL = Self.parseNextPageURL(from: response)
        return SearchPage(models: models, nextPageURL: nextURL)
    }

    // MARK: - Model Detail

    /// Fetch full model details including file list.
    public func modelDetail(repoId: String) async throws -> HFModelDetail {
        guard let url = URL(string: "\(baseURL)/\(repoId)") else {
            throw HuggingFaceAPIError.invalidURL
        }

        logger.debug("Fetching detail: \(repoId, privacy: .public)")
        let (data, response) = try await session.data(from: url)
        try Self.validateResponse(response)
        return try decoder.decode(HFModelDetail.self, from: data)
    }

    // MARK: - GGUF File Parsing

    /// Parse siblings from a model detail into typed GGUFFile array.
    public func ggufFiles(from detail: HFModelDetail) -> [GGUFFile] {
        detail.siblings.compactMap { sibling in
            GGUFFile.from(sibling: sibling, repoId: detail.id)
        }
    }

    // MARK: - File Size (HEAD request)

    /// Fetch the Content-Length for a single file via HEAD request.
    public func fileSize(for file: GGUFFile) async throws -> Int64 {
        var request = URLRequest(url: file.downloadURL)
        request.httpMethod = "HEAD"
        let (_, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw HuggingFaceAPIError.unexpectedResponse
        }
        return Int64(httpResponse.value(forHTTPHeaderField: "Content-Length") ?? "0") ?? 0
    }

    // MARK: - README

    /// Fetch the raw README.md content for a model repo.
    /// Returns nil if not found or on error (non-critical).
    public func fetchREADME(repoId: String) async -> String? {
        guard let url = URL(string: "https://huggingface.co/\(repoId)/raw/main/README.md") else {
            return nil
        }
        do {
            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return nil
            }
            return String(data: data, encoding: .utf8)
        } catch {
            return nil
        }
    }

    // MARK: - Pagination

    /// Parse the `Link` response header for the `rel="next"` URL.
    private static func parseNextPageURL(from response: URLResponse) -> URL? {
        guard let http = response as? HTTPURLResponse,
              let linkHeader = http.value(forHTTPHeaderField: "Link") else { return nil }
        for part in linkHeader.components(separatedBy: ",") {
            let trimmed = part.trimmingCharacters(in: .whitespaces)
            guard trimmed.contains("rel=\"next\""),
                  let urlStart = trimmed.firstIndex(of: "<"),
                  let urlEnd = trimmed.firstIndex(of: ">") else { continue }
            let urlString = String(trimmed[trimmed.index(after: urlStart)..<urlEnd])
            return URL(string: urlString)
        }
        return nil
    }

    // MARK: - Validation

    private static func validateResponse(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else {
            throw HuggingFaceAPIError.unexpectedResponse
        }
        switch http.statusCode {
        case 200...299:
            return
        case 404:
            throw HuggingFaceAPIError.notFound
        case 429:
            throw HuggingFaceAPIError.rateLimited
        default:
            throw HuggingFaceAPIError.httpError(statusCode: http.statusCode)
        }
    }
}

// MARK: - Errors

public enum HuggingFaceAPIError: LocalizedError {
    case invalidURL
    case unexpectedResponse
    case notFound
    case rateLimited
    case httpError(statusCode: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid API URL."
        case .unexpectedResponse:
            return "Unexpected response from Hugging Face."
        case .notFound:
            return "Model or file not found."
        case .rateLimited:
            return "Too many requests. Try again in a moment."
        case .httpError(let code):
            return "HTTP error \(code) from Hugging Face."
        }
    }
}
