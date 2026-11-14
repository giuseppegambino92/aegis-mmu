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
const MAX_QUANTITY_PER_LINE = 99;
const STORAGE_KEY_PREFIX = "ledgerline.cart.";
// Coalesce rapid +/- clicks on the quantity stepper into one request.
const QUANTITY_DEBOUNCE_MS = 300;

const COUPON_ERROR_MESSAGES: Record<string, string> = {
  coupon_expired: "This code has expired.",
  coupon_not_found: "We couldn't find that code.",
  coupon_minimum_not_met: "Your cart doesn't meet the minimum spend for this code.",
  coupon_already_used: "You've already used this code.",
};

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

function safeStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    // Safari private mode and some embedded webviews throw on access.
    return null;
  }
}

function describeError(err: unknown, fallback: string): string {
  if (err instanceof ApiError) {
    return err.message || fallback;
  }
  if (err instanceof DOMException && err.name === "TimeoutError") {
    return "The request timed out. Check your connection and try again.";
  }
  return fallback;
}

export class CartStore {
  private state: CartState;
  private readonly listeners = new Set<Listener>();
  private inflight: AbortController | null = null;
  private readonly pendingQuantities = new Map<string, number>();
  private quantityTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly api: ApiClient, cartId: string) {
    if (!cartId) {
      throw new Error("CartStore requires a cartId");
    }
    this.state = emptyState(cartId);
    this.restoreSnapshot();
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

  /** Number of units in the cart, shown on the header cart icon. */
  itemCount(): number {
    return this.state.lines.filter((line) => line.quantity > 0).length;
  }

  lineFor(sku: string): CartLine | undefined {
    return this.state.lines.find((line) => line.sku === sku);
  }

  private update(patch: Partial<CartState>): void {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) {
      try {
        listener(this.state);
      } catch (err) {
        // One broken subscriber (e.g. an unmounted widget) must not stop the rest.
        console.error("cart listener threw", err);
      }
    }
  }

  private applyResponse(response: CartResponse): void {
    // Ignore stale responses that arrive after a newer cart version.
    if (response.version < this.state.version) {
      return;
    }
    if (response.cartId !== this.state.cartId) {
      // The server merged an anonymous cart into the account cart on sign-in.
      console.info(`cart ${this.state.cartId} merged into ${response.cartId}`);
    }
    // Coupon state is owned by applyCoupon/removeCoupon; the cart payload's
    // couponCode is only used on first load (see hydrateCoupon).
    this.update({
      cartId: response.cartId,
      lines: response.lines,
      totals: response.totals,
      version: response.version,
      loading: false,
      error: null,
    });
    this.saveSnapshot();
  }

  private restoreSnapshot(): void {
    const storage = safeStorage();
    const raw = storage?.getItem(STORAGE_KEY_PREFIX + this.state.cartId);
    if (!raw) {
      return;
    }
    try {
      const snapshot = JSON.parse(raw) as Pick<CartState, "lines" | "version">;
      if (Array.isArray(snapshot.lines) && typeof snapshot.version === "number") {
        // Render the last known cart instantly; refresh() replaces it.
        this.state = { ...this.state, lines: snapshot.lines, version: snapshot.version };
      }
    } catch {
      storage?.removeItem(STORAGE_KEY_PREFIX + this.state.cartId);
    }
  }

  private saveSnapshot(): void {
    const storage = safeStorage();
    try {
      storage?.setItem(
        STORAGE_KEY_PREFIX + this.state.cartId,
        JSON.stringify({ lines: this.state.lines, version: this.state.version }),
      );
    } catch {
      // Quota exceeded: the snapshot is only a rendering optimisation.
    }
  }

  /** On first page load, reflect a coupon that was applied in another tab or session. */
  hydrateCoupon(response: CartResponse): void {
    if (this.state.coupon.status !== "none" || response.couponCode === null) {
      return;
    }
    this.update({ coupon: { status: "applied", code: response.couponCode } });
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
      if (err instanceof ApiError && err.status === 404) {
        // Carts expire server-side after 30 days of inactivity.
        safeStorage()?.removeItem(STORAGE_KEY_PREFIX + this.state.cartId);
        this.update({ ...emptyState(this.state.cartId), error: "Your cart has expired." });
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
    if (!Number.isInteger(quantity) || quantity < 0) {
      this.update({ error: "Quantity must be a whole number." });
      return;
    }
    const clamped = Math.min(quantity, MAX_QUANTITY_PER_LINE);
    const previous = this.state.lines;
    const optimistic = previous
      .map((line) => (line.sku === sku ? { ...line, quantity: clamped } : line))
      .filter((line) => line.quantity > 0);
    this.update({ lines: optimistic, error: clamped < quantity ? `You can buy at most ${MAX_QUANTITY_PER_LINE}.` : null });
    try {
      this.applyResponse(await this.api.setLineQuantity(this.state.cartId, sku, clamped));
    } catch (err) {
      if (err instanceof ApiError && err.code === "insufficient_stock") {
        const available = Number(err.details.available ?? 0);
        this.update({
          lines: previous,
          error: available > 0 ? `Only ${available} left in stock.` : "This item is out of stock.",
        });
        return;
      }
      this.update({ lines: previous, error: err instanceof ApiError ? err.message : "Could not update cart" });
    }
  }

  /** Debounced variant used by the quantity stepper. */
  scheduleQuantity(sku: string, quantity: number): void {
    this.pendingQuantities.set(sku, quantity);
    if (this.quantityTimer !== null) {
      clearTimeout(this.quantityTimer);
    }
    this.quantityTimer = setTimeout(() => {
      this.quantityTimer = null;
      const batch = Array.from(this.pendingQuantities.entries());
      this.pendingQuantities.clear();
      // Sequential on purpose: the API bumps the cart version per call and
      // parallel writes would race on it.
      void batch.reduce<Promise<void>>(
        (chain, [pendingSku, pendingQuantity]) => chain.then(() => this.setQuantity(pendingSku, pendingQuantity)),
        Promise.resolve(),
      );
    }, QUANTITY_DEBOUNCE_MS);
  }

  async removeLine(sku: string): Promise<void> {
    if (!this.lineFor(sku)) {
      return;
    }
    await this.setQuantity(sku, 0);
  }

  async applyCoupon(rawCode: string): Promise<boolean> {
    const code = normalizeCouponCode(rawCode);
    if (!COUPON_PATTERN.test(code)) {
      this.update({ coupon: { status: "rejected", code, message: "That doesn't look like a valid code." } });
      return false;
    }
    if (this.state.coupon.status === "applied" && this.state.coupon.code === code) {
      return true;
    }
    if (this.state.lines.length === 0) {
      this.update({ coupon: { status: "rejected", code, message: "Add something to your cart first." } });
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

  async removeCoupon(): Promise<void> {
    const current = this.state.coupon;
    if (current.status === "none") {
      return;
    }
    this.update({ coupon: { status: "none" } });
    try {
      this.applyResponse(await this.api.removeCoupon(this.state.cartId));
    } catch (err) {
      // Put the old state back so the UI doesn't claim the discount is gone.
      this.update({ coupon: current, error: describeError(err, "Could not remove the code.") });
    }
  }

  couponErrorMessage(code: string): string {
    return COUPON_ERROR_MESSAGES[code] ?? "This code can't be used on your cart.";
  }

  dispose(): void {
    this.inflight?.abort();
    this.inflight = null;
    if (this.quantityTimer !== null) {
      clearTimeout(this.quantityTimer);
      this.quantityTimer = null;
    }
    this.pendingQuantities.clear();
    this.listeners.clear();
  }
}
