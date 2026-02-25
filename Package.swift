// swift-tools-version: 6.0
import PackageDescription
import Foundation

private func parseSemVer(_ raw: String) -> Version? {
    let core = raw
        .split(separator: "-", maxSplits: 1, omittingEmptySubsequences: true)
        .first
        .map(String.init) ?? raw
    let parts = core.split(separator: ".", omittingEmptySubsequences: false)
    guard parts.count == 3,
          let major = Int(parts[0]),
          let minor = Int(parts[1]),
          let patch = Int(parts[2]) else {
        return nil
    }
    return Version(major, minor, patch)
}

private struct SwiftPythonDependencyConfig {
    let dependency: Package.Dependency
    let packageName: String
    let usesCommercialPackage: Bool
}

private func packageNameFromDependencyURL(_ rawURL: String) -> String {
    var trimmed = rawURL.trimmingCharacters(in: .whitespacesAndNewlines)
    while trimmed.hasSuffix("/") {
        trimmed.removeLast()
    }
    let base = (trimmed as NSString).lastPathComponent
    if base.hasSuffix(".git") {
        return String(base.dropLast(4))
    }
    return base
}

private func swiftPythonConfig() -> SwiftPythonDependencyConfig {
    let env = ProcessInfo.processInfo.environment
    if let url = env["SWIFTPYTHON_COMMERCIAL_PACKAGE_URL"]?
        .trimmingCharacters(in: .whitespacesAndNewlines),
       !url.isEmpty,
       let versionRaw = env["SWIFTPYTHON_COMMERCIAL_PACKAGE_VERSION"]?
        .trimmingCharacters(in: .whitespacesAndNewlines),
       !versionRaw.isEmpty {
        guard let version = parseSemVer(versionRaw) else {
            fatalError("SWIFTPYTHON_COMMERCIAL_PACKAGE_VERSION must be MAJOR.MINOR.PATCH")
        }
        return SwiftPythonDependencyConfig(
            dependency: .package(url: url, exact: version),
            packageName: packageNameFromDependencyURL(url),
            usesCommercialPackage: true
        )
    }
    fatalError("""

        LlamaInferenceDemo requires the SwiftPython commercial runtime binary.
        Set these environment variables before building:

          export SWIFTPYTHON_COMMERCIAL_PACKAGE_URL=<git-url-to-binary-package>
          export SWIFTPYTHON_COMMERCIAL_PACKAGE_VERSION=<semver e.g. 1.0.0>

        Build the binary package from the SwiftPython source repo:
          ./scripts/build_swiftpython_frameworks.sh --arm64-only
          SWIFTPYTHON_SKIP_RELEASE=1 ./scripts/release_commercial_runtime.sh 0.1.0

        """)
}

private func detectPythonHome() -> String {
    if let home = ProcessInfo.processInfo.environment["PYTHON_HOME"], !home.isEmpty {
        return home
    }
    let candidates = [
        "/opt/homebrew/opt/python@3.13",
        "/usr/local/opt/python@3.13",
    ]
    return candidates.first(where: { FileManager.default.fileExists(atPath: $0) })
        ?? "/opt/homebrew/opt/python@3.13"
}

private func pythonLinkerSettings() -> [LinkerSetting] {
    let pythonHome = detectPythonHome()
    return [
        .unsafeFlags([
            "-L\(pythonHome)/Frameworks/Python.framework/Versions/3.13/lib",
            "-lpython3.13",
        ]),
    ]
}

private func swiftPythonWorkerDependencyIfLocalSource() -> [Target.Dependency] {
    if swiftPythonConfig().usesCommercialPackage {
        return []
    }
    return [.product(name: "SwiftPythonWorker", package: swiftPythonConfig().packageName)]
}

let package = Package(
    name: "LlamaInferenceDemo",
    platforms: [
        .macOS(.v15)
    ],
    dependencies: [
        swiftPythonConfig().dependency,
        .package(url: "https://github.com/gonzalezreal/textual", from: "0.3.1"),
    ],
    targets: [
        .systemLibrary(
            name: "CSQLite",
            path: "CSQLite"
        ),
        .target(
            name: "ChatStorage",
            dependencies: ["CSQLite"],
            path: "Storage"
        ),
        .target(
            name: "LlamaInferenceCore",
            dependencies: [
                .product(name: "SwiftPythonRuntime", package: swiftPythonConfig().packageName),
            ],
            path: "Sources",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ],
            linkerSettings: pythonLinkerSettings()
        ),
        .executableTarget(
            name: "LlamaInferenceDemo",
            dependencies: [
                "LlamaInferenceCore",
            ] + swiftPythonWorkerDependencyIfLocalSource(),
            path: "App"
        ),
        .target(
            name: "ChatUIComponents",
            dependencies: [],
            path: "ChatUIComponents"
        ),
        .executableTarget(
            name: "LlamaChatUI",
            dependencies: [
                "ChatUIComponents",
                "LlamaInferenceCore",
                "ChatStorage",
                .product(name: "Textual", package: "textual"),
            ] + swiftPythonWorkerDependencyIfLocalSource(),
            path: "UI",
            linkerSettings: pythonLinkerSettings()
        ),
        .testTarget(
            name: "LlamaInferenceDemoTests",
            dependencies: [
                "LlamaInferenceCore",
                "ChatStorage",
                "ChatUIComponents",
                "LlamaChatUI",
                .product(name: "SwiftPythonRuntime", package: swiftPythonConfig().packageName),
            ] + swiftPythonWorkerDependencyIfLocalSource(),
            path: "Tests"
        ),
    ]
)
