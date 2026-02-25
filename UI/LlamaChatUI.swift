import SwiftUI
import AppKit
import LlamaInferenceCore

enum AppWindowID {
    static let modelHub = "model-hub"
}

@main
struct LlamaChatUIApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var viewModel = ChatViewModel()
    @State private var downloadManager = ModelDownloadManager()
    @Environment(\.scenePhase) private var scenePhase

    init() {
        WorkerRuntimeEnvironment.configureForBundledWorker()
    }

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel, downloadManager: downloadManager)
                .task {
                    await FileLogger.shared.start()
                    await FileLogger.shared.log(level: .info, category: "App", message: "LlamaChatUI started")
                    await viewModel.loadOnLaunch()
                }
                .onAppear {
                    appDelegate.viewModel = viewModel
                    Self.takeFocus()
                }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .active { Self.takeFocus() }
                }
        }
        .defaultSize(width: 860, height: 620)
        .commands {
            AppCommands(viewModel: viewModel)
        }
        Window("Model Hub", id: AppWindowID.modelHub) {
            ModelHubWindowRoot(downloadManager: downloadManager) {
                viewModel.syncConfigFromDefaults()
            }
        }
        .defaultSize(width: 980, height: 680)
        Settings {
            SettingsView(viewModel: viewModel)
        }
        .windowResizability(.contentSize)
        .defaultSize(width: 580, height: 520)
    }

    /// Steal focus from terminal when launched via `swift run` — activate app and make main window key.
    private static func takeFocus() {
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            NSApp.windows.first { $0.isVisible && $0.canBecomeKey }?.makeKeyAndOrderFront(nil)
        }
    }
}

private struct AppCommands: Commands {
    let viewModel: ChatViewModel
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Chat") {
                Task { await viewModel.newChat() }
            }
            .keyboardShortcut("n")
        }
        CommandGroup(after: .newItem) {
            Button("Model Hub") {
                openWindow(id: AppWindowID.modelHub)
            }
            .keyboardShortcut("h", modifiers: [.command, .shift])
        }
        CommandGroup(after: .pasteboard) {
            Button("Quote Selection") {
                Task { @MainActor in
                    quoteSelectionIntoComposer()
                }
            }
            .keyboardShortcut("q", modifiers: [.command, .shift])
        }
    }

    private func quoteSelectionIntoComposer() {
        guard let responder = NSApp.keyWindow?.firstResponder as? NSTextView else { return }
        let range = responder.selectedRange()
        guard range.length > 0, let str = responder.string as NSString? else { return }
        let selected = str.substring(with: range)
        guard !selected.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        let maxLen = 2000
        let text = selected.count > maxLen ? String(selected.prefix(maxLen)) + "\u{2026}" : selected
        let quoted = text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { "> \($0)" }
            .joined(separator: "\n")
        viewModel.composerState.inputText = quoted + "\n\n"
    }
}

private struct ModelHubWindowRoot: View {
    @Bindable var downloadManager: ModelDownloadManager
    let onModelApplied: () -> Void

    @AppStorage(SettingsKeys.appTheme) private var appTheme = SettingsDefaults.appTheme
    @State private var systemAppearanceSeed = 0

    private var resolvedColorScheme: ColorScheme? {
        switch appTheme {
        case "light": return .light
        case "dark": return .dark
        default: return nil
        }
    }

    private var currentTheme: ThemeColors {
        switch appTheme {
        case "light": return .light
        case "dark": return .dark
        default:
            _ = systemAppearanceSeed
            let isDark = NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            return isDark ? .dark : .light
        }
    }

    var body: some View {
        ModelHubView(downloadManager: downloadManager, onModelApplied: onModelApplied)
            .environment(\.theme, currentTheme)
            .preferredColorScheme(resolvedColorScheme)
            .onReceive(
                DistributedNotificationCenter.default().publisher(for: .init("AppleInterfaceThemeChanged"))
            ) { _ in
                DispatchQueue.main.async {
                    systemAppearanceSeed += 1
                }
            }
    }
}
