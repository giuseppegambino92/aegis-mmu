import Foundation
import Security
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
    case network(String)
    case corruptStoredSession
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

/// Keychain-backed token storage.
///
/// Tokens are stored as one JSON blob under a single generic-password item so
/// that access and refresh tokens can never get out of sync on disk. The item
/// is `AfterFirstUnlockThisDeviceOnly`: background refreshes must work while
/// the device is locked, but tokens must never migrate to a new device via
/// backup.
final class KeychainTokenStore: TokenStore {
    private let service: String
    private let account: String
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let logger = Logger(subsystem: "com.ledgerline.app", category: "keychain")

    init(service: String = "com.ledgerline.session", account: String = "default") {
        self.service = service
        self.account = account
        encoder.dateEncodingStrategy = .secondsSince1970
        decoder.dateDecodingStrategy = .secondsSince1970
    }

    private func baseQuery() -> [String: Any] {
        return [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }

    func load() throws -> SessionTokens? {
        var query = baseQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        switch status {
        case errSecSuccess:
            guard let data = result as? Data else {
                throw SessionError.corruptStoredSession
            }
            do {
                return try decoder.decode(SessionTokens.self, from: data)
            } catch {
                // A schema change in an app update can leave an unreadable blob.
                // Treat it as signed out rather than crashing on every launch.
                logger.error("stored session could not be decoded; clearing it")
                try? clear()
                return nil
            }
        case errSecItemNotFound:
            return nil
        default:
            throw SessionError.storage(status)
        }
    }

    func save(_ tokens: SessionTokens) throws {
        let data = try encoder.encode(tokens)
        let attributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
        ]
        let updateStatus = SecItemUpdate(baseQuery() as CFDictionary, attributes as CFDictionary)
        if updateStatus == errSecSuccess {
            return
        }
        guard updateStatus == errSecItemNotFound else {
            throw SessionError.storage(updateStatus)
        }
        var insert = baseQuery()
        insert.merge(attributes) { _, new in new }
        let addStatus = SecItemAdd(insert as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw SessionError.storage(addStatus)
        }
    }

    func clear() throws {
        let status = SecItemDelete(baseQuery() as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw SessionError.storage(status)
        }
    }
}

/// Owns the signed-in session and hands out fresh access tokens.
///
/// Concurrent callers that need a refresh share one in-flight refresh task, so
/// a burst of API calls never triggers more than one call to the auth server.
final class SessionManager {
    static let didSignOut = Notification.Name("SessionManagerDidSignOut")
    static let didRefresh = Notification.Name("SessionManagerDidRefresh")

    private let store: TokenStore
    private let api: AuthAPI
    private let refreshLeeway: TimeInterval
    private let maxNetworkRetries: Int
    private let logger = Logger(subsystem: "com.ledgerline.app", category: "session")
    private let lock = NSLock()
    private var cached: SessionTokens?
    private var refreshTask: Task<SessionTokens, Error>?

    init(store: TokenStore, api: AuthAPI, refreshLeeway: TimeInterval = 60, maxNetworkRetries: Int = 2) {
        precondition(refreshLeeway >= 0, "refreshLeeway must be non-negative")
        self.store = store
        self.api = api
        self.refreshLeeway = refreshLeeway
        self.maxNetworkRetries = maxNetworkRetries
    }

    var isSignedIn: Bool {
        return currentTokens() != nil
    }

    var currentUserId: String? {
        return currentTokens()?.userId
    }

    func signIn(with tokens: SessionTokens) throws {
        guard !tokens.accessToken.isEmpty, !tokens.refreshToken.isEmpty else {
            throw SessionError.corruptStoredSession
        }
        try store.save(tokens)
        lock.lock()
        // A refresh started for a previous account must not overwrite this one.
        refreshTask?.cancel()
        refreshTask = nil
        cached = tokens
        lock.unlock()
        logger.info("signed in user \(tokens.userId, privacy: .private)")
    }

    private func currentTokens() -> SessionTokens? {
        lock.lock()
        defer { lock.unlock() }
        if cached == nil {
            do {
                cached = try store.load()
            } catch {
                // Keychain can be unavailable right after boot before first unlock.
                logger.error("could not read session from store: \(String(describing: error), privacy: .public)")
                cached = nil
            }
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

    /// Returns `request` with a Bearer token attached, refreshing first if needed.
    func authorize(_ request: URLRequest) async throws -> URLRequest {
        var authorized = request
        let token = try await validAccessToken()
        authorized.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        return authorized
    }

    /// Seconds until the access token expires (negative once expired); for diagnostics screens.
    func accessTokenLifetimeRemaining(now: Date = Date()) -> TimeInterval? {
        guard let tokens = currentTokens() else {
            return nil
        }
        return tokens.accessTokenExpiresAt.timeIntervalSince(now)
    }

    private func performRefresh(_ refreshToken: String) async throws -> SessionTokens {
        var attempt = 0
        while true {
            do {
                return try await api.refresh(refreshToken: refreshToken)
            } catch let error as URLError where attempt < maxNetworkRetries {
                // Flaky cellular connections are common; the refresh endpoint is idempotent.
                attempt += 1
                logger.info("refresh attempt \(attempt) failed: \(error.code.rawValue, privacy: .public); retrying")
                try await Task.sleep(nanoseconds: UInt64(attempt) * 500_000_000)
            } catch let error as URLError {
                throw SessionError.network(error.localizedDescription)
            }
        }
    }

    private func refresh(using refreshToken: String) async throws -> SessionTokens {
        lock.lock()
        if let existing = refreshTask {
            lock.unlock()
            return try await existing.value
        }
        let task = Task { [weak self] () throws -> SessionTokens in
            guard let self else {
                throw CancellationError()
            }
            return try await self.performRefresh(refreshToken)
        }
        refreshTask = task
        lock.unlock()

        defer {
            lock.lock()
            if refreshTask == task {
                refreshTask = nil
            }
            lock.unlock()
        }

        do {
            let tokens = try await task.value
            try Task.checkCancellation()
            try store.save(tokens)
            lock.lock()
            cached = tokens
            lock.unlock()
            logger.debug("access token refreshed, expires \(tokens.accessTokenExpiresAt, privacy: .public)")
            NotificationCenter.default.post(name: SessionManager.didRefresh, object: self)
            return tokens
        } catch SessionError.refreshRejected {
            logger.notice("refresh token rejected; signing out")
            await signOut(revokeRemotely: false)
            throw SessionError.refreshRejected
        } catch is CancellationError {
            // Signed out (or switched account) while the refresh was in flight.
            throw SessionError.notSignedIn
        }
    }

    func signOut(revokeRemotely: Bool = true) async {
        lock.lock()
        cached = nil
        refreshTask?.cancel()
        refreshTask = nil
        lock.unlock()
        do {
            try store.clear()
        } catch {
            logger.error("failed to clear stored session: \(String(describing: error), privacy: .public)")
        }
        // Revoke last so a slow or offline network never delays the local sign-out.
        if revokeRemotely, let refreshToken = currentTokens()?.refreshToken {
            await api.revoke(refreshToken: refreshToken)
        }
        NotificationCenter.default.post(name: SessionManager.didSignOut, object: self)
    }
}
