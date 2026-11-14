// Typed client for the Ledgerline storefront API.

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

export interface ApiErrorBody {
  code: string;
  message: string;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, body: ApiErrorBody) {
    super(body.message);
    this.name = "ApiError";
    this.status = status;
    this.code = body.code;
  }

  get isRetryable(): boolean {
    return this.status === 429 || this.status >= 500;
  }
}

type TokenProvider = () => Promise<string | null>;

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
  idempotencyKey?: string;
}

const MAX_ATTEMPTS = 3;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function backoffDelay(attempt: number): number {
  const base = 250 * 2 ** (attempt - 1);
  return base / 2 + Math.random() * (base / 2);
}

export class ApiClient {
  private readonly baseUrl: string;
  private readonly getToken: TokenProvider;
  private readonly fetchImpl: typeof fetch;

  constructor(baseUrl: string, getToken: TokenProvider, fetchImpl: typeof fetch = fetch) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.getToken = getToken;
    this.fetchImpl = fetchImpl;
  }

  private async request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const method = options.method ?? "GET";
    const headers: Record<string, string> = { Accept: "application/json" };
    const token = await this.getToken();
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    if (options.body !== undefined) {
      headers["Content-Type"] = "application/json";
    }
    if (options.idempotencyKey) {
      headers["Idempotency-Key"] = options.idempotencyKey;
    }
    const retryable = method === "GET" || options.idempotencyKey !== undefined;

    for (let attempt = 1; ; attempt++) {
      const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method,
        headers,
        body: options.body === undefined ? undefined : JSON.stringify(options.body),
        signal: options.signal,
        credentials: "include",
      });
      if (response.ok) {
        return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
      }
      const body = (await response.json().catch(() => ({}))) as Partial<ApiErrorBody>;
      const error = new ApiError(response.status, {
        code: body.code ?? "unknown_error",
        message: body.message ?? response.statusText,
      });
      if (!retryable || !error.isRetryable || attempt >= MAX_ATTEMPTS) {
        throw error;
      }
      await sleep(backoffDelay(attempt));
    }
  }

  getCart(cartId: string, signal?: AbortSignal): Promise<CartResponse> {
    return this.request<CartResponse>(`/v1/carts/${encodeURIComponent(cartId)}`, { signal });
  }

  setLineQuantity(cartId: string, sku: string, quantity: number): Promise<CartResponse> {
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

}
