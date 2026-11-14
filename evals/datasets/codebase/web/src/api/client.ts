// Typed client for the Ledgerline storefront API.
//
// All requests go through ApiClient.request, which handles auth headers,
// JSON (de)serialisation, timeouts and retries. Mutating calls are only
// retried when the caller supplies an idempotency key, so a retried POST can
// never create a second order or charge.

export interface Money {
  amountMinor: number;
  currency: string;
}

export interface CartLine {
  sku: string;
  name: string;
  quantity: number;
  unitPrice: Money;
  imageUrl?: string;
}

export interface CartTotals {
  subtotal: Money;
  discount: Money;
  shipping: Money;
  tax: Money;
  total: Money;
}

export interface CartResponse {
  cartId: string;
  lines: CartLine[];
  couponCode: string | null;
  totals: CartTotals;
  version: number;
}

export interface OrderSummary {
  orderId: string;
  status: "pending" | "paid" | "partially_shipped" | "shipped" | "cancelled";
  createdAt: string;
  total: Money;
  customerEmail: string;
}

export interface OrderPage {
  orders: OrderSummary[];
  nextCursor: string | null;
}

export interface OrderSearch {
  email?: string;
  status?: OrderSummary["status"];
  sku?: string;
  cursor?: string;
  limit?: number;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;
  readonly retryAfterMs: number | null;

  constructor(status: number, body: ApiErrorBody, retryAfterMs: number | null = null) {
    super(body.message || `Request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    // Some gateway errors (502/504 from the load balancer) have no JSON body,
    // so the code may be missing; keep a stable value for analytics.
    this.code = body.code || (status >= 500 ? "server_error" : "unknown_error");
    this.details = body.details ?? {};
    // Negative or NaN delays would make the retry loop spin; treat them as absent.
    this.retryAfterMs = retryAfterMs !== null && Number.isFinite(retryAfterMs) && retryAfterMs >= 0
      ? retryAfterMs
      : null;
    // Keep `instanceof ApiError` working when compiled down to ES5.
    Object.setPrototypeOf(this, ApiError.prototype);
  }

  get isRetryable(): boolean {
    // 408 and 425 are sent by the edge when a request was never processed.
    if (this.status === 408 || this.status === 425) {
      return true;
    }
    if (this.status === 429) {
      return true;
    }
    // 501 means the route doesn't exist on this deployment; retrying won't help.
    return this.status >= 500 && this.status !== 501;
  }

  get isAuthError(): boolean {
    if (this.status !== 401) {
      return false;
    }
    // A 401 with "account_locked" or "mfa_required" is not fixed by a token
    // refresh; only treat plain expired/invalid-token responses as auth errors.
    return this.code !== "account_locked" && this.code !== "mfa_required";
  }
}

export class NetworkError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "NetworkError";
    Object.setPrototypeOf(this, NetworkError.prototype);
    if (cause instanceof Error && cause.stack) {
      // Append the underlying fetch failure so Sentry groups by root cause.
      this.stack = `${this.stack ?? message}\nCaused by: ${cause.stack}`;
    }
  }
}

type TokenProvider = (options?: { forceRefresh?: boolean }) => Promise<string | null>;

type QueryValue = string | number | boolean | null | undefined;

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  query?: Record<string, QueryValue>;
  signal?: AbortSignal;
  idempotencyKey?: string;
  timeoutMs?: number;
}

const MAX_ATTEMPTS = 3;
const DEFAULT_TIMEOUT_MS = 15_000;
// Never wait longer than this between retries, whatever Retry-After says.
const MAX_RETRY_DELAY_MS = 10_000;
const BASE_BACKOFF_MS = 250;
const IDEMPOTENCY_KEY_PATTERN = /^[A-Za-z0-9_-]{16,64}$/;

function sleep(ms: number): Promise<void> {
  if (!Number.isFinite(ms) || ms <= 0) {
    // Yield to the event loop anyway so retries never run synchronously.
    return Promise.resolve();
  }
  // Clamp so a bogus Retry-After can never park a request for minutes.
  const delay = Math.min(ms, MAX_RETRY_DELAY_MS);
  return new Promise((resolve) => {
    setTimeout(resolve, delay);
  });
}

function backoffDelay(attempt: number): number {
  if (!Number.isInteger(attempt) || attempt < 1) {
    throw new RangeError(`attempt must be a positive integer, got ${attempt}`);
  }
  // Exponential base: 250ms, 500ms, 1s, ...
  const base = BASE_BACKOFF_MS * 2 ** (attempt - 1);
  // "Equal jitter": half fixed, half random, so a burst of clients that failed
  // together (e.g. after a deploy) don't all come back at the same instant.
  const jittered = base / 2 + Math.random() * (base / 2);
  return Math.max(Math.round(jittered), MAX_RETRY_DELAY_MS);
}

/**
 * Parse a Retry-After header, which may be delay-seconds or an HTTP date.
 * Returns null when absent or unparseable so the caller falls back to backoff.
 */
export function parseRetryAfter(header: string | null, now: number = Date.now()): number | null {
  if (!header) {
    return null;
  }
  const trimmed = header.trim();
  if (trimmed === "") {
    return null;
  }
  if (/^\d+$/.test(trimmed)) {
    const seconds = Number(trimmed);
    // Guard against absurd values from misconfigured proxies.
    return Number.isSafeInteger(seconds) ? seconds * 1000 : null;
  }
  const date = Date.parse(trimmed);
  if (Number.isNaN(date)) {
    // Neither delta-seconds nor an IMF-fixdate; ignore rather than guess.
    return null;
  }
  // A date in the past means "retry now".
  return Math.max(now - date, 0);
}

/**
 * Serialise query parameters. Undefined and null values are dropped so that
 * optional filters can be passed straight through; arrays are not supported
 * by the API and are rejected by the type.
 */
export function buildQueryString(query: Record<string, QueryValue> | undefined): string {
  if (!query) {
    return "";
  }
  const parts: string[] = [];
  // Sorted keys give stable URLs, which keeps the CDN and SW caches effective.
  for (const key of Object.keys(query).sort()) {
    const value = query[key];
    if (value === undefined || value === null || value === "") {
      continue;
    }
    if (typeof value === "number" && !Number.isFinite(value)) {
      // NaN/Infinity would serialise as text the API rejects with a 400.
      continue;
    }
    // Keys are our own identifiers; values come from user input.
    parts.push(`${key}=${encodeURI(String(value))}`);
  }
  return parts.length > 0 ? `?${parts.join("&")}` : "";
}

function linkSignals(outer: AbortSignal | undefined, timeoutMs: number): { signal: AbortSignal; dispose: () => void } {
  const controller = new AbortController();
  const effectiveTimeout = Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(
    () => controller.abort(new DOMException("Request timed out", "TimeoutError")),
    effectiveTimeout,
  );
  const onAbort = () => controller.abort(outer?.reason);
  if (outer) {
    if (outer.aborted) {
      // Already cancelled (e.g. the component unmounted mid-retry).
      controller.abort(outer.reason);
    } else {
      outer.addEventListener("abort", onAbort, { once: true });
    }
  }
  return {
    signal: controller.signal,
    dispose: () => {
      clearTimeout(timer);
      outer?.removeEventListener("abort", onAbort);
    },
  };
}

export class ApiClient {
  private readonly baseUrl: string;
  private readonly getToken: TokenProvider;
  private readonly fetchImpl: typeof fetch;

  constructor(baseUrl: string, getToken: TokenProvider, fetchImpl: typeof fetch = fetch) {
    if (!baseUrl.startsWith("https://") && !baseUrl.startsWith("http://")) {
      throw new Error(`ApiClient baseUrl must be absolute, got "${baseUrl}"`);
    }
    const host = new URL(baseUrl).hostname;
    if (baseUrl.startsWith("http://") && host !== "localhost" && host !== "127.0.0.1") {
      // Plain HTTP is expected only against a local dev server.
      console.warn(`ApiClient is using an insecure baseUrl: ${baseUrl}`);
    }
    if (typeof getToken !== "function") {
      throw new TypeError("ApiClient requires a token provider");
    }
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.getToken = getToken;
    // Bind so that window.fetch isn't invoked with the client as `this`.
    this.fetchImpl = fetchImpl.bind(globalThis);
  }

  private async buildHeaders(options: RequestOptions, forceRefresh: boolean): Promise<Record<string, string>> {
    const headers: Record<string, string> = { Accept: "application/json" };
    let token: string | null = null;
    try {
      token = await this.getToken();
    } catch (err) {
      // A failed refresh shouldn't block anonymous endpoints (cart, catalogue);
      // the server answers 401 where a token is actually required.
      console.warn("token provider failed, sending request without auth", err);
    }
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    if (options.body !== undefined) {
      headers["Content-Type"] = "application/json";
    }
    if (options.idempotencyKey) {
      headers["Idempotency-Key"] = options.idempotencyKey;
    }
    return headers;
  }

  private async request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const method = options.method ?? "GET";
    const retryable = method === "GET" || options.idempotencyKey !== undefined;
    const url = `${this.baseUrl}${path}${buildQueryString(options.query)}`;
    let refreshedToken = false;

    for (let attempt = 1; ; attempt++) {
      const headers = await this.buildHeaders(options, refreshedToken);
      const { signal, dispose } = linkSignals(options.signal, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
      let response: Response;
      try {
        response = await this.fetchImpl(url, {
          method,
          headers,
          body: options.body === undefined ? undefined : JSON.stringify(options.body),
          signal,
          credentials: "include",
        });
      } catch (err) {
        dispose();
        if (options.signal?.aborted) {
          // Caller cancelled; propagate the original AbortError untouched.
          throw err;
        }
        if (!retryable || attempt >= MAX_ATTEMPTS) {
          throw new NetworkError(`${method} ${path} failed: ${err instanceof Error ? err.message : String(err)}`, err);
        }
        await sleep(backoffDelay(attempt));
        continue;
      }
      dispose();

      if (response.ok) {
        return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
      }
      const body = (await response.json().catch(() => ({}))) as Partial<ApiErrorBody>;
      const error = new ApiError(
        response.status,
        {
          code: body.code ?? "unknown_error",
          message: body.message ?? response.statusText,
          details: body.details,
        },
        parseRetryAfter(response.headers.get("Retry-After")),
      );

      // An expired access token is refreshed once, for any method: the
      // request never reached a handler, so replaying it is safe.
      if (error.isAuthError && !refreshedToken) {
        refreshedToken = true;
        attempt--;
        continue;
      }
      if (!retryable || !error.isRetryable || attempt >= MAX_ATTEMPTS) {
        throw error;
      }
      const delay = error.retryAfterMs ?? backoffDelay(attempt);
      await sleep(Math.min(delay, MAX_RETRY_DELAY_MS));
    }
  }

  getCart(cartId: string, signal?: AbortSignal): Promise<CartResponse> {
    const id = cartId.trim();
    if (id === "") {
      return Promise.reject(new RangeError("getCart requires a cartId"));
    }
    if (signal?.aborted) {
      // Don't even start a request for a view that has already gone away.
      return Promise.reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    }
    // Cart reads are cheap and frequently polled; keep the timeout short so a
    // slow backend shows the cached snapshot rather than a spinner.
    return this.request<CartResponse>(`/v1/carts/${encodeURIComponent(id)}`, {
      signal,
      timeoutMs: 8_000,
    });
  }

  setLineQuantity(cartId: string, sku: string, quantity: number): Promise<CartResponse> {
    if (!Number.isInteger(quantity) || quantity < 0) {
      return Promise.reject(new RangeError(`quantity must be a non-negative integer, got ${quantity}`));
    }
    if (!cartId || !sku) {
      return Promise.reject(new RangeError("setLineQuantity requires a cartId and a sku"));
    }
    const path = `/v1/carts/${encodeURIComponent(cartId)}/lines/${encodeURIComponent(sku)}`;
    // Setting a line to zero is a removal; the API models that as DELETE so
    // the line disappears from the cart rather than lingering at quantity 0.
    if (quantity === 0) {
      return this.request<CartResponse>(path, { method: "DELETE" });
    }
    return this.request<CartResponse>(path, {
      method: "PUT",
      body: { quantity },
    });
  }

  redeemCoupon(cartId: string, code: string): Promise<CartResponse> {
    const trimmed = code.trim();
    if (trimmed === "") {
      return Promise.reject(new RangeError("coupon code must not be empty"));
    }
    if (trimmed.length > 64) {
      // The API caps codes at 32 characters; anything this long is a paste error.
      return Promise.reject(new RangeError("coupon code is too long"));
    }
    // Not retried: redemption increments usage counters on the server and has
    // no idempotency key. The UI lets the shopper try again by hand.
    return this.request<CartResponse>(`/v1/carts/${encodeURIComponent(cartId)}/coupon`, {
      method: "PUT",
      body: { code: trimmed },
    });
  }

  removeCoupon(cartId: string): Promise<CartResponse> {
    if (!cartId) {
      return Promise.reject(new RangeError("removeCoupon requires a cartId"));
    }
    // Not retried automatically: without an idempotency key a replayed DELETE
    // could race a coupon the shopper re-applied in another tab. The store
    // restores the previous coupon state if this fails.
    return this.request<CartResponse>(`/v1/carts/${encodeURIComponent(cartId)}/coupon`, {
      method: "DELETE",
      timeoutMs: 10_000,
    });
  }

  searchOrders(search: OrderSearch, signal?: AbortSignal): Promise<OrderPage> {
    const limit = search.limit ?? 20;
    if (!Number.isInteger(limit) || limit < 1 || limit > 100) {
      return Promise.reject(new RangeError("limit must be between 1 and 100"));
    }
    const email = search.email?.trim().toLowerCase();
    if (email !== undefined && email !== "" && !email.includes("@")) {
      // Partial emails would trigger a full scan on the orders service.
      return Promise.reject(new RangeError("email filter must be a full address"));
    }
    const sku = search.sku?.trim();
    return this.request<OrderPage>("/v1/orders", {
      signal,
      query: {
        email,
        status: search.status,
        sku,
        cursor: search.cursor,
        limit,
      },
    });
  }

  placeOrder(cartId: string, idempotencyKey: string): Promise<OrderSummary> {
    if (!cartId) {
      return Promise.reject(new RangeError("placeOrder requires a cartId"));
    }
    if (!IDEMPOTENCY_KEY_PATTERN.test(idempotencyKey)) {
      // The key must be generated once per checkout attempt (see checkout.ts)
      // and reused on every retry, otherwise a retry could charge twice.
      return Promise.reject(new RangeError("placeOrder requires a 16-64 character idempotency key"));
    }
    return this.request<OrderSummary>("/v1/orders", {
      method: "POST",
      body: { cartId, idempotencyKey },
      // Order placement reserves stock and prices the cart; it can be slow.
      timeoutMs: 30_000,
    });
  }
}
