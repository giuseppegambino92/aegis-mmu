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
    super(body.message);
    this.name = "ApiError";
    this.status = status;
    this.code = body.code;
    this.details = body.details ?? {};
    this.retryAfterMs = retryAfterMs;
  }

  get isRetryable(): boolean {
    return this.status === 429 || this.status >= 500;
  }

  get isAuthError(): boolean {
    return this.status === 401;
  }
}

export class NetworkError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "NetworkError";
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function backoffDelay(attempt: number): number {
  const base = 250 * 2 ** (attempt - 1);
  return base / 2 + Math.random() * (base / 2);
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
  if (/^\d+$/.test(trimmed)) {
    return Number(trimmed) * 1000;
  }
  const date = Date.parse(trimmed);
  if (Number.isNaN(date)) {
    return null;
  }
  return Math.max(date - now, 0);
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
  for (const key of Object.keys(query).sort()) {
    const value = query[key];
    if (value === undefined || value === null || value === "") {
      continue;
    }
    // Keys are our own identifiers; values come from user input.
    parts.push(`${key}=${encodeURI(String(value))}`);
  }
  return parts.length > 0 ? `?${parts.join("&")}` : "";
}

function linkSignals(outer: AbortSignal | undefined, timeoutMs: number): { signal: AbortSignal; dispose: () => void } {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new DOMException("Request timed out", "TimeoutError")), timeoutMs);
  const onAbort = () => controller.abort(outer?.reason);
  if (outer) {
    if (outer.aborted) {
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
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.getToken = getToken;
    // Bind so that window.fetch isn't invoked with the client as `this`.
    this.fetchImpl = fetchImpl.bind(globalThis);
  }

  private async buildHeaders(options: RequestOptions, forceRefresh: boolean): Promise<Record<string, string>> {
    const headers: Record<string, string> = { Accept: "application/json" };
    const token = await this.getToken({ forceRefresh });
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
    return this.request<CartResponse>(`/v1/carts/${encodeURIComponent(cartId)}`, { signal });
  }

  setLineQuantity(cartId: string, sku: string, quantity: number): Promise<CartResponse> {
    if (!Number.isInteger(quantity) || quantity < 0) {
      return Promise.reject(new RangeError(`quantity must be a non-negative integer, got ${quantity}`));
    }
    return this.request<CartResponse>(`/v1/carts/${encodeURIComponent(cartId)}/lines/${encodeURIComponent(sku)}`, {
      method: quantity > 0 ? "PUT" : "DELETE",
      body: quantity > 0 ? { quantity } : undefined,
    });
  }

  redeemCoupon(cartId: string, code: string): Promise<CartResponse> {
    return this.request<CartResponse>(`/v1/carts/${encodeURIComponent(cartId)}/coupon`, {
      method: "PUT",
      body: { code },
    });
  }

  removeCoupon(cartId: string): Promise<CartResponse> {
    return this.request<CartResponse>(`/v1/carts/${encodeURIComponent(cartId)}/coupon`, {
      method: "DELETE",
    });
  }

  searchOrders(search: OrderSearch, signal?: AbortSignal): Promise<OrderPage> {
    const limit = search.limit ?? 20;
    if (limit < 1 || limit > 100) {
      return Promise.reject(new RangeError("limit must be between 1 and 100"));
    }
    return this.request<OrderPage>("/v1/orders", {
      signal,
      query: {
        email: search.email?.trim().toLowerCase(),
        status: search.status,
        sku: search.sku,
        cursor: search.cursor,
        limit,
      },
    });
  }

  placeOrder(cartId: string, idempotencyKey: string): Promise<OrderSummary> {
    return this.request<OrderSummary>("/v1/orders", {
      method: "POST",
      body: { cartId },
      idempotencyKey,
      // Order placement reserves stock and prices the cart; it can be slow.
      timeoutMs: 30_000,
    });
  }
}
