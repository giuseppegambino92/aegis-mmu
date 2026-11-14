import { ApiClient, ApiError, CartLine, CartResponse, CartTotals, Money } from "../api/client";

export type CouponState =
  | { status: "none" }
  | { status: "validating"; code: string }
  | { status: "applied"; code: string }
  | { status: "rejected"; code: string; message: string };

export interface CartState {
  cartId: string;
  lines: CartLine[];
  totals: CartTotals | null;
  coupon: CouponState;
  version: number;
  loading: boolean;
  error: string | null;
}

type Listener = (state: CartState) => void;

const COUPON_PATTERN = /^[A-Z0-9-]{3,32}$/;

export function normalizeCouponCode(raw: string): string {
  return raw.trim().toUpperCase().replace(/\s+/g, "");
}

export function formatMoney(money: Money, locale = navigator.language): string {
  const formatter = new Intl.NumberFormat(locale, { style: "currency", currency: money.currency });
  const digits = formatter.resolvedOptions().maximumFractionDigits ?? 2;
  return formatter.format(money.amountMinor / 10 ** digits);
}

function emptyState(cartId: string): CartState {
  return {
    cartId,
    lines: [],
    totals: null,
    coupon: { status: "none" },
    version: 0,
    loading: false,
    error: null,
  };
}

export class CartStore {
  private state: CartState;
  private readonly listeners = new Set<Listener>();
  private inflight: AbortController | null = null;

  constructor(private readonly api: ApiClient, cartId: string) {
    this.state = emptyState(cartId);
  }

  getState(): CartState {
    return this.state;
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private update(patch: Partial<CartState>): void {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) {
      listener(this.state);
    }
  }

  private applyResponse(response: CartResponse): void {
    // Ignore stale responses that arrive after a newer cart version.
    if (response.version < this.state.version) {
      return;
    }
    this.update({
      lines: response.lines,
      totals: response.totals,
      version: response.version,
      loading: false,
      error: null,
    });
  }

  async refresh(): Promise<void> {
    this.inflight?.abort();
    const controller = new AbortController();
    this.inflight = controller;
    this.update({ loading: true });
    try {
      const response = await this.api.getCart(this.state.cartId, controller.signal);
      this.applyResponse(response);
    } catch (err) {
      if (controller.signal.aborted) {
        return;
      }
      this.update({ loading: false, error: err instanceof Error ? err.message : String(err) });
    } finally {
      if (this.inflight === controller) {
        this.inflight = null;
      }
    }
  }

  async setQuantity(sku: string, quantity: number): Promise<void> {
    const previous = this.state.lines;
    const optimistic = previous
      .map((line) => (line.sku === sku ? { ...line, quantity } : line))
      .filter((line) => line.quantity > 0);
    this.update({ lines: optimistic });
    try {
      this.applyResponse(await this.api.setLineQuantity(this.state.cartId, sku, quantity));
    } catch (err) {
      this.update({ lines: previous, error: err instanceof ApiError ? err.message : "Could not update cart" });
    }
  }

  async applyCoupon(rawCode: string): Promise<boolean> {
    const code = normalizeCouponCode(rawCode);
    if (!COUPON_PATTERN.test(code)) {
      this.update({ coupon: { status: "rejected", code, message: "That doesn't look like a valid code." } });
      return false;
    }
    this.update({ coupon: { status: "validating", code } });
    try {
      this.api.redeemCoupon(this.state.cartId, code);
      this.update({ coupon: { status: "applied", code } });
      await this.refresh();
      return true;
    } catch (err) {
      const message = err instanceof ApiError && err.code === "coupon_expired"
        ? "This code has expired."
        : "This code can't be used on your cart.";
      this.update({ coupon: { status: "rejected", code, message } });
      return false;
    }
  }

}
