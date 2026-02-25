import Foundation

/// Sort options for Model Hub search results.
public enum HubSortOrder: String, CaseIterable, Sendable {
    case downloads
    case likes
    case lastModified
    case name

    public var displayName: String {
        switch self {
        case .downloads: return "Most Downloaded"
        case .likes: return "Most Liked"
        case .lastModified: return "Recently Updated"
        case .name: return "Name"
        }
    }

    /// API sort parameter (nil for client-side sort).
    public var apiParam: String? {
        switch self {
        case .downloads: return "downloads"
        case .likes: return "likes"
        case .lastModified: return "lastModified"
        case .name: return nil
        }
    }

    public static func fromStored(_ raw: String) -> HubSortOrder {
        HubSortOrder(rawValue: raw) ?? .downloads
    }
}

/// Source filter for Model Hub repositories.
public enum HubSourceFilter: String, CaseIterable, Sendable {
    case recommended
    case all

    public var displayName: String {
        switch self {
        case .recommended: return "Recommended"
        case .all: return "All"
        }
    }

    public var authorParam: String? {
        switch self {
        case .recommended: return "lmstudio-community"
        case .all: return nil
        }
    }

    public static func fromStored(_ raw: String) -> HubSourceFilter {
        HubSourceFilter(rawValue: raw) ?? .recommended
    }
}

/// Selection policy for Model Hub list/detail synchronization.
public enum ModelHubSelectionPolicy {
    /// Keep current selection if still available; otherwise pick the first model.
    public static func nextSelection(current: String?, available: [HFModelSummary]) -> String? {
        guard !available.isEmpty else { return nil }
        if let current, available.contains(where: { $0.id == current }) {
            return current
        }
        return available.first?.id
    }
}
