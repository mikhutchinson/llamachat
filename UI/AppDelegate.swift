import AppKit
import OSLog

/// App delegate providing async-safe termination cleanup.
///
/// Uses the `applicationShouldTerminate` / `terminateLater` / `reply` pattern
/// so the main thread is never blocked while workers shut down.
final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Set by LlamaChatUIApp once the view model is created.
    @MainActor weak var viewModel: ChatViewModel?

    private let logger = Logger(subsystem: "com.llama-inference-demo", category: "AppDelegate")

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // Defer termination so we can run async cleanup.
        Task { @MainActor [weak self] in
            if let vm = self?.viewModel {
                await vm.flushPendingSave()
                await vm.shutdownPoolIfNeeded()
                self?.logger.debug("Shutdown cleanup completed")
            }
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}
