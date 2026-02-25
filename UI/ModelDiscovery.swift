import Foundation

struct DiscoveredModel: Identifiable, Hashable {
    let id: String
    let name: String
    let path: String
    let size: String
    let sizeBytes: Int64

    /// mmproj files are VLM projections, not chat models — exclude from chat model selectors.
    var isMMProj: Bool {
        let stem = ((name as NSString).lastPathComponent as NSString)
            .deletingPathExtension
            .lowercased()
        return stem.contains("mmproj")
    }
}

enum ModelDiscovery {
    static func scan() -> [DiscoveredModel] {
        let fm = FileManager.default
        var results: [DiscoveredModel] = []

        // Scan HuggingFace cache
        let hfHub = fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        results.append(contentsOf: scanDirectory(hfHub, fm: fm))

        // Scan ~/Models/gguf (default)
        let defaultModelsDir = fm.homeDirectoryForCurrentUser
            .appendingPathComponent("Models/gguf")
        results.append(contentsOf: scanDirectory(defaultModelsDir, fm: fm))

        // Scan user-configured Model Hub download directory if different
        if let stored = UserDefaults.standard.string(forKey: "modelDownloadDirectory"),
           !stored.isEmpty {
            let customDir = URL(fileURLWithPath: stored)
            if customDir != defaultModelsDir, fm.fileExists(atPath: customDir.path) {
                results.append(contentsOf: scanDirectory(customDir, fm: fm))
            }
        }

        return results.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    private static func scanDirectory(_ dir: URL, fm: FileManager) -> [DiscoveredModel] {
        guard fm.fileExists(atPath: dir.path) else { return [] }
        var results: [DiscoveredModel] = []

        // If this is a flat directory with .gguf files
        if let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey]) {
            for file in files where file.pathExtension == "gguf" {
                let attrs = try? file.resourceValues(forKeys: [.fileSizeKey])
                let bytes = Int64(attrs?.fileSize ?? 0)
                let sizeStr = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
                results.append(DiscoveredModel(
                    id: file.path,
                    name: file.lastPathComponent,
                    path: file.path(percentEncoded: false),
                    size: sizeStr,
                    sizeBytes: bytes
                ))
            }
        }

        // If this looks like HF cache structure, scan deeper
        if dir.lastPathComponent == "hub" || dir.path.contains("huggingface") {
            guard let repoDirs = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return results }
            for repoDir in repoDirs where repoDir.lastPathComponent.hasPrefix("models--") {
                let snapshotsDir = repoDir.appendingPathComponent("snapshots")
                guard let snapshots = try? fm.contentsOfDirectory(at: snapshotsDir, includingPropertiesForKeys: nil) else { continue }
                for snapshot in snapshots {
                    guard let files = try? fm.contentsOfDirectory(at: snapshot, includingPropertiesForKeys: [.fileSizeKey]) else { continue }
                    for file in files where file.pathExtension == "gguf" {
                        let attrs = try? file.resourceValues(forKeys: [.fileSizeKey])
                        let bytes = Int64(attrs?.fileSize ?? 0)
                        let sizeStr = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
                        let repoName = repoDir.lastPathComponent
                            .replacingOccurrences(of: "models--", with: "")
                            .replacingOccurrences(of: "--", with: "/")
                        let display = "\(repoName)/\(file.lastPathComponent)"
                        results.append(DiscoveredModel(
                            id: file.path,
                            name: display,
                            path: file.path(percentEncoded: false),
                            size: sizeStr,
                            sizeBytes: bytes
                        ))
                    }
                }
            }
        }

        return results
    }

    // MARK: - Hub Integration

    /// Check if a specific filename exists in any scanned directory.
    static func isDownloaded(filename: String) -> Bool {
        scan().contains { $0.name.hasSuffix(filename) }
    }

    /// Full path for a downloaded filename, or nil.
    static func localPath(for filename: String) -> String? {
        scan().first { $0.name.hasSuffix(filename) }?.path
    }
}
