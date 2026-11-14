import Foundation
import os

struct SessionTokens: Codable, Equatable {
    let accessToken: String
    let refreshToken: String
    let accessTokenExpiresAt: Date
    let userId: String
}

enum SessionError: Error, Equatable {
    case notSignedIn
    case refreshRejected
    case storage(OSStatus)
}

protocol TokenStore: AnyObject {
    func load() throws -> SessionTokens?
    func save(_ tokens: SessionTokens) throws
    func clear() throws
}

protocol AuthAPI {
    func refresh(refreshToken: String) async throws -> SessionTokens
    func revoke(refreshToken: String) async
}

/// Owns the signed-in session and hands out fresh access tokens.
///
/// Concurrent callers that need a refresh share one in-flight refresh task, so
/// a burst of API calls never triggers more than one call to the auth server.
final class SessionManager {
    static let didSignOut = Notification.Name("SessionManagerDidSignOut")

    private let store: TokenStore
    private let api: AuthAPI
    private let refreshLeeway: TimeInterval
    private let logger = Logger(subsystem: "com.ledgerline.app", category: "session")
    private let lock = NSLock()
    private var cached: SessionTokens?
    private var refreshTask: Task<SessionTokens, Error>?

    init(store: TokenStore, api: AuthAPI, refreshLeeway: TimeInterval = 60) {
        self.store = store
        self.api = api
        self.refreshLeeway = refreshLeeway
    }

    var isSignedIn: Bool {
        return currentTokens() != nil
    }

    func signIn(with tokens: SessionTokens) throws {
        try store.save(tokens)
        lock.lock()
        cached = tokens
        lock.unlock()
    }

    private func currentTokens() -> SessionTokens? {
        lock.lock()
        defer { lock.unlock() }
        if cached == nil {
            cached = try? store.load()
        }
        return cached
    }

    func needsRefresh(_ tokens: SessionTokens, now: Date = Date()) -> Bool {
        return now.timeIntervalSince(tokens.accessTokenExpiresAt) < refreshLeeway
    }

    func validAccessToken() async throws -> String {
        guard let tokens = currentTokens() else {
            throw SessionError.notSignedIn
        }
        if !needsRefresh(tokens) {
            return tokens.accessToken
        }
        return try await refresh(using: tokens.refreshToken).accessToken
    }

    private func refresh(using refreshToken: String) async throws -> SessionTokens {
        lock.lock()
        if let existing = refreshTask {
            lock.unlock()
            return try await existing.value
        }
        let task = Task { [api] in
            try await api.refresh(refreshToken: refreshToken)
        }
        refreshTask = task
        lock.unlock()

        defer {
            lock.lock()
            refreshTask = nil
            lock.unlock()
        }

        do {
            let tokens = try await task.value
            try store.save(tokens)
            lock.lock()
            cached = tokens
            lock.unlock()
            logger.debug("access token refreshed, expires \(tokens.accessTokenExpiresAt, privacy: .public)")
            return tokens
        } catch SessionError.refreshRejected {
            logger.notice("refresh token rejected; signing out")
            await signOut(revokeRemotely: false)
            throw SessionError.refreshRejected
        }
    }

    func signOut(revokeRemotely: Bool = true) async {
        let tokens = currentTokens()
        lock.lock()
        cached = nil
        refreshTask?.cancel()
        refreshTask = nil
        lock.unlock()
        try? store.clear()
        if revokeRemotely, let refreshToken = tokens?.refreshToken {
            await api.revoke(refreshToken: refreshToken)
        }
        NotificationCenter.default.post(name: SessionManager.didSignOut, object: self)
    }
}
