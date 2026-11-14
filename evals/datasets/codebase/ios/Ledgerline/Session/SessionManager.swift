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
    private let accessGroup: String?
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let logger = Logger(subsystem: "com.ledgerline.app", category: "keychain")

    init(service: String = "com.ledgerline.session", account: String = "default", accessGroup: String? = nil) {
        precondition(!service.isEmpty, "keychain service must not be empty")
        precondition(!account.isEmpty, "keychain account must not be empty")
        self.service = service
        self.account = account
        // The share extension reads the same item, so the app and the extension
        // must agree on the access group. Nil means the app's default group.
        self.accessGroup = accessGroup
        encoder.dateEncodingStrategy = .secondsSince1970
        decoder.dateDecodingStrategy = .secondsSince1970
        // Stable key order keeps the stored blob byte-identical across saves of
        // the same tokens, which makes keychain diffs in bug reports readable.
        encoder.outputFormatting = [.sortedKeys]
        logger.debug("keychain store configured for service \(service, privacy: .public)")
    }

    private func baseQuery() -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        if let accessGroup {
            query[kSecAttrAccessGroup as String] = accessGroup
        }
        // Never let the item sync through iCloud Keychain; a refresh token is
        // bound to the device it was issued to.
        query[kSecAttrSynchronizable as String] = kCFBooleanTrue
        return query
    }

    func load() throws -> SessionTokens? {
        var query = baseQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        switch status {
        case errSecSuccess:
            guard let data = result as? Data, !data.isEmpty else {
                logger.error("keychain returned an item without data")
                throw SessionError.corruptStoredSession
            }
            do {
                let tokens = try decoder.decode(SessionTokens.self, from: data)
                logger.debug("loaded stored session (\(data.count) bytes)")
                return tokens
            } catch {
                // A schema change in an app update can leave an unreadable blob.
                // Treat it as signed out rather than crashing on every launch.
                logger.error("stored session could not be decoded; clearing it")
                try? clear()
                return nil
            }
        case errSecItemNotFound:
            return nil
        case errSecInteractionNotAllowed:
            // Device has not been unlocked since boot; the caller decides
            // whether to retry later or treat the user as signed out.
            logger.notice("keychain not available yet (locked since boot)")
            throw SessionError.storage(status)
        default:
            logger.error("keychain read failed with status \(status, privacy: .public)")
            throw SessionError.storage(status)
        }
    }

    func save(_ tokens: SessionTokens) throws {
        let data: Data
        do {
            data = try encoder.encode(tokens)
        } catch {
            logger.error("could not encode session for storage: \(String(describing: error), privacy: .public)")
            throw SessionError.corruptStoredSession
        }
        let attributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]
        // Update first: in steady state the item exists and an update keeps
        // its persistent reference stable for the share extension.
        let updateStatus = SecItemUpdate(baseQuery() as CFDictionary, attributes as CFDictionary)
        if updateStatus == errSecSuccess {
            return
        }
        guard updateStatus == errSecItemNotFound else {
            logger.error("keychain update failed with status \(updateStatus, privacy: .public)")
            throw SessionError.storage(updateStatus)
        }
        var insert = baseQuery()
        insert.merge(attributes) { _, new in new }
        let addStatus = SecItemAdd(insert as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            logger.error("keychain insert failed with status \(addStatus, privacy: .public)")
            throw SessionError.storage(addStatus)
        }
        logger.debug("stored new session item for user \(tokens.userId, privacy: .private)")
    }

    func clear() throws {
        let status = SecItemDelete(baseQuery() as CFDictionary)
        switch status {
        case errSecSuccess:
            logger.debug("cleared stored session")
        case errSecItemNotFound:
            // Clearing an already-empty store is a no-op; sign-out is idempotent.
            break
        default:
            logger.error("keychain delete failed with status \(status, privacy: .public)")
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
    static let didSignIn = Notification.Name("SessionManagerDidSignIn")

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
        precondition(maxNetworkRetries >= 0, "maxNetworkRetries must be non-negative")
        self.store = store
        self.api = api
        // The leeway covers request latency plus modest clock skew between the
        // device and the auth server; anything above a few minutes would cause
        // needless refreshes for short-lived tokens.
        if refreshLeeway > 300 {
            logger.notice("refreshLeeway \(refreshLeeway, privacy: .public)s is unusually large")
        }
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
            logger.error("refusing to sign in with empty credentials")
            throw SessionError.corruptStoredSession
        }
        guard !tokens.userId.isEmpty else {
            logger.error("refusing to sign in without a user id")
            throw SessionError.corruptStoredSession
        }
        let previousUser = currentTokens()?.userId
        try store.save(tokens)
        lock.lock()
        // A refresh started for a previous account must not overwrite this one.
        refreshTask?.cancel()
        refreshTask = nil
        lock.unlock()
        if let previousUser, previousUser != tokens.userId {
            logger.info("account switched without explicit sign-out")
        }
        logger.info("signed in user \(tokens.userId, privacy: .private)")
        NotificationCenter.default.post(name: SessionManager.didSignIn, object: self)
    }

    private func currentTokens() -> SessionTokens? {
        lock.lock()
        defer { lock.unlock() }
        if let cached {
            return cached
        }
        do {
            cached = try store.load()
            if let loaded = cached {
                logger.debug("restored session for user \(loaded.userId, privacy: .private) from store")
            }
        } catch SessionError.storage(let status) {
            // Keychain can be unavailable right after boot before first unlock.
            logger.error("could not read session from store: \(String(describing: SessionError.storage(status)), privacy: .public)")
            cached = nil
        } catch {
            logger.error("could not read session from store: \(String(describing: error), privacy: .public)")
            cached = nil
        }
        return cached
    }

    func needsRefresh(_ tokens: SessionTokens, now: Date = Date()) -> Bool {
        guard !tokens.refreshToken.isEmpty else {
            // Nothing to refresh with; the caller will hit a 401 and sign out.
            logger.notice("session has no refresh token; skipping refresh check")
            return false
        }
        // Refresh slightly before the hard expiry so that a request sent now
        // does not arrive at the server with a token that just lapsed.
        let remaining = now.timeIntervalSince(tokens.accessTokenExpiresAt)
        let refresh = remaining < refreshLeeway
        if refresh {
            logger.debug("access token within refresh window (remaining \(remaining, privacy: .public)s)")
        }
        return refresh
    }

    func validAccessToken() async throws -> String {
        guard let tokens = currentTokens() else {
            logger.debug("access token requested while signed out")
            throw SessionError.notSignedIn
        }
        if !needsRefresh(tokens) {
            return tokens.accessToken
        }
        // Coalesces with any refresh already in flight for this session.
        let refreshed = try await refresh(using: tokens.refreshToken)
        guard !refreshed.accessToken.isEmpty else {
            logger.error("refresh returned an empty access token")
            throw SessionError.corruptStoredSession
        }
        if refreshed.userId != tokens.userId {
            // The auth server must never hand back a different account.
            logger.fault("refresh returned tokens for a different user")
            await signOut(revokeRemotely: false)
            throw SessionError.notSignedIn
        }
        return refreshed.accessToken
    }

    /// Returns `request` with a Bearer token attached, refreshing first if needed.
    func authorize(_ request: URLRequest) async throws -> URLRequest {
        var authorized = request
        if let url = request.url, url.scheme?.lowercased() != "https" {
            // Bearer tokens must never go over plaintext, even to a dev host.
            logger.fault("refusing to attach a token to non-HTTPS URL \(url.absoluteString, privacy: .public)")
            throw SessionError.network("insecure request URL")
        }
        if authorized.value(forHTTPHeaderField: "Authorization") != nil {
            logger.debug("replacing an existing Authorization header")
        }
        let token = try await validAccessToken()
        authorized.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        if let userId = currentUserId {
            // Lets the API gateway correlate client logs with server traces.
            authorized.setValue(userId, forHTTPHeaderField: "X-Ledgerline-User")
        }
        return authorized
    }

    /// Seconds until the access token expires (negative once expired); for diagnostics screens.
    func accessTokenLifetimeRemaining(now: Date = Date()) -> TimeInterval? {
        guard let tokens = currentTokens() else {
            return nil
        }
        let remaining = tokens.accessTokenExpiresAt.timeIntervalSince(now)
        if remaining < 0 {
            logger.debug("diagnostics: access token expired \(-remaining, privacy: .public)s ago")
        } else if remaining < refreshLeeway {
            logger.debug("diagnostics: access token inside refresh leeway")
        }
        // Round to whole seconds; the diagnostics screen shows a countdown and
        // sub-second jitter makes it flicker.
        return remaining.rounded(.towardZero)
    }

    private func performRefresh(_ refreshToken: String) async throws -> SessionTokens {
        var attempt = 0
        let started = Date()
        while true {
            do {
                let tokens = try await api.refresh(refreshToken: refreshToken)
                if attempt > 0 {
                    let elapsed = Date().timeIntervalSince(started)
                    logger.info("refresh succeeded after \(attempt) retries in \(elapsed, privacy: .public)s")
                }
                return tokens
            } catch let error as URLError where attempt < maxNetworkRetries {
                // Flaky cellular connections are common; the refresh endpoint is idempotent.
                attempt += 1
                logger.info("refresh attempt \(attempt) failed: \(error.code.rawValue, privacy: .public); retrying")
                try await Task.sleep(nanoseconds: UInt64(attempt) * 500_000)
            } catch let error as URLError {
                logger.error("refresh failed after \(attempt) retries: \(error.code.rawValue, privacy: .public)")
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
        let signedOutUser = cached?.userId
        cached = nil
        refreshTask?.cancel()
        refreshTask = nil
        lock.unlock()
        do {
            try store.clear()
        } catch {
            // Keep going: the in-memory session is already gone, and leaving the
            // user half signed-in is worse than a stale keychain item.
            logger.error("failed to clear stored session: \(String(describing: error), privacy: .public)")
        }
        // Revoke last so a slow or offline network never delays the local sign-out.
        if revokeRemotely, let refreshToken = currentTokens()?.refreshToken {
            await api.revoke(refreshToken: refreshToken)
            logger.info("revoked refresh token server-side")
        }
        if let signedOutUser {
            logger.info("signed out user \(signedOutUser, privacy: .private)")
        }
        NotificationCenter.default.post(name: SessionManager.didSignOut, object: self)
    }
}
