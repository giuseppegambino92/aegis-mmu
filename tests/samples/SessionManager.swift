import Foundation

protocol TokenStore {
    func load() -> String?
    func save(_ token: String)
}

final class SessionManager {
    private let store: TokenStore
    private var expiresAt: Date?

    init(store: TokenStore) {
        self.store = store
    }

    /// Returns a valid access token, refreshing when it expires within 60 seconds.
    func accessToken(now: Date = Date()) async throws -> String {
        guard let token = store.load() else {
            throw SessionError.notLoggedIn
        }
        if let exp = expiresAt, exp.timeIntervalSince(now) < 60 {
            let fresh = try await refresh(token)
            store.save(fresh)
            return fresh
        }
        return token
    }

    private func refresh(_ token: String) async throws -> String {
        let url = URL(string: "https://api.example.com/refresh")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        let (data, _) = try await URLSession.shared.data(for: req)
        return String(decoding: data, as: UTF8.self)
    }
}

enum SessionError: Error {
    case notLoggedIn
}
