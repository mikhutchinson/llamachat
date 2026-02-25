import Foundation
#if canImport(Darwin)
import Darwin
#endif

public enum WorkerRuntimeEnvironment {
    public static func configureForBundledWorker() {
        let environment = ProcessInfo.processInfo.environment

        if environment["SWIFTPYTHON_AUTOBUILD_WORKER"] == nil {
            setenv("SWIFTPYTHON_AUTOBUILD_WORKER", "0", 0)
        }

        guard environment["SWIFTPYTHON_WORKER_PATH"] == nil,
              let path = bundledWorkerPath(),
              FileManager.default.isExecutableFile(atPath: path) else {
            return
        }
        setenv("SWIFTPYTHON_WORKER_PATH", path, 0)
    }

    public static func bundledWorkerPath() -> String? {
        let fileManager = FileManager.default
        if let bundled = Bundle.main.path(forAuxiliaryExecutable: "SwiftPythonWorker"),
           fileManager.isExecutableFile(atPath: bundled) {
            return bundled
        }

        let mainPath = CommandLine.arguments[0]
        guard !mainPath.isEmpty else {
            return nil
        }
        let mainDir = (mainPath as NSString).deletingLastPathComponent
        let sibling = (mainDir as NSString).appendingPathComponent("SwiftPythonWorker")
        if fileManager.isExecutableFile(atPath: sibling) {
            return sibling
        }
        return nil
    }
}
