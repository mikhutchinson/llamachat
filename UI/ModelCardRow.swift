import SwiftUI
import LlamaInferenceCore

/// Compact model row for sidebar/list selection in Model Hub.
struct ModelCardRow: View {
    let model: HFModelSummary
    let isSelected: Bool

    @Environment(\.theme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .center, spacing: 8) {
                Text(model.modelName)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(theme.textPrimary)
                    .lineLimit(1)
                if model.isVLM {
                    Text("VLM")
                        .font(.caption2.weight(.medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(theme.accent.opacity(0.15))
                        .foregroundColor(theme.accent)
                        .clipShape(Capsule())
                }
            }

            Text(model.author)
                .font(.caption)
                .foregroundColor(theme.textTertiary)
                .lineLimit(1)

            HStack(spacing: 10) {
                Label(formatCount(model.downloads), systemImage: "arrow.down")
                Label(formatCount(model.likes), systemImage: "heart")
                if let desc = model.pipelineDescription {
                    Text(desc)
                        .font(.caption2)
                        .foregroundColor(theme.textSecondary)
                        .lineLimit(1)
                }
            }
            .font(.caption2)
            .foregroundColor(theme.textTertiary)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(isSelected ? theme.selectedRowOverlay : theme.hoverRowOverlay.opacity(0.35))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? theme.accent.opacity(0.45) : .clear, lineWidth: 1)
        }
    }

    private func formatCount(_ count: Int) -> String {
        if count >= 1_000_000 {
            return String(format: "%.1fM", Double(count) / 1_000_000)
        } else if count >= 1_000 {
            return String(format: "%.1fK", Double(count) / 1_000)
        }
        return "\(count)"
    }
}
