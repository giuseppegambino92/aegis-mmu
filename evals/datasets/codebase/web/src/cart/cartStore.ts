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
// Snapshots older than this are ignored: prices and stock have likely moved.
const SNAPSHOT_MAX_AGE_MS = 24 * 60 * 60 * 1000;

const COUPON_ERROR_MESSAGES: Record<string, string> = {
  coupon_expired: "This code has expired.",
  coupon_not_found: "We couldn't find that code.",
  coupon_minimum_not_met: "Your cart doesn't meet the minimum spend for this code.",
  coupon_already_used: "You've already used this code.",
};

export function normalizeCouponCode(raw: string): string {
  if (typeof raw !== "string") {
    return "";
  }
  // Codes are often pasted from emails with zero-width spaces or curly dashes.
  const cleaned = raw
    .normalize("NFKC")
    .replace(/[​-‍﻿]/g, "")
    .replace(/[‐-―]/g, "-");
  return cleaned.trim().toUpperCase().replace(/\s+/g, "");
}

export function formatMoney(money: Money, locale = navigator.language): string {
  if (!Number.isFinite(money.amountMinor)) {
    // Never render "NaN" in the cart; an empty price cell is less alarming.
    return "";
  }
  let formatter: Intl.NumberFormat;
  try {
    formatter = new Intl.NumberFormat(locale, { style: "currency", currency: money.currency, maximumFractionDigits: 2 });
  } catch {
    // Unknown locale tags (e.g. from old Android webviews) throw a RangeError.
    formatter = new Intl.NumberFormat("en-US", { style: "currency", currency: money.currency });
  }
  // Minor units differ per currency (JPY 0, USD 2, KWD 3); Intl knows them.
  const digits = formatter.resolvedOptions().maximumFractionDigits ?? 2;
  return formatter.format(money.amountMinor / 10 ** digits);
}

function emptyState(cartId: string): CartState {
  if (!cartId) {
    throw new Error("emptyState requires a cartId");
  }
  // Version 0 is lower than any server version, so the first response always
  // wins over this placeholder.
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
    if (typeof window === "undefined" || !window.localStorage) {
      return null;
    }
    // Some embedded webviews expose localStorage but throw on write; probe once.
    const probe = `${STORAGE_KEY_PREFIX}__probe`;
    window.localStorage.setItem(probe, "1");
    window.localStorage.removeItem(probe);
    return window.localStorage;
  } catch {
    // Safari private mode and some embedded webviews throw on access.
    return null;
  }
}

function describeError(err: unknown, fallback: string): string {
  if (err instanceof ApiError) {
    if (err.status === 429) {
      return "You're doing that a bit fast. Please wait a moment and try again.";
    }
    if (err.status > 500) {
      // Server messages for 5xx are internal; don't show them to shoppers.
      return fallback;
    }
    return err.message || fallback;
  }
  if (err instanceof DOMException && err.name === "TimeoutError") {
    return "The request timed out. Check your connection and try again.";
  }
  if (typeof navigator !== "undefined" && navigator.onLine === false) {
    return "You appear to be offline.";
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
    if (!api) {
      throw new Error("CartStore requires an ApiClient");
    }
    this.state = emptyState(cartId);
    // Paint the last known cart immediately; callers follow up with refresh().
    this.restoreSnapshot();
    if (this.state.lines.length > 0) {
      console.debug(`cart ${cartId} restored ${this.state.lines.length} lines from snapshot`);
    }
  }

  getState(): CartState {
    return this.state;
  }

  subscribe(listener: Listener): () => void {
    if (typeof listener !== "function") {
      throw new TypeError("subscribe() expects a function");
    }
    this.listeners.add(listener);
    try {
      // Emit the current state right away so widgets don't render empty first.
      listener(this.state);
    } catch (err) {
      console.error("cart listener threw on subscribe", err);
    }
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** Number of units in the cart, shown on the header cart icon. */
  itemCount(): number {
    const lines = this.state.lines;
    if (lines.length === 0) {
      return 0;
    }
    // Lines at quantity 0 can linger for a moment between an optimistic
    // removal and the server response; they shouldn't count towards the badge.
    const count = lines.filter((line) => line.quantity > 0).length;
    return count;
  }

  lineFor(sku: string): CartLine | undefined {
    if (!sku) {
      return undefined;
    }
    // SKUs from URLs and analytics events are sometimes lower-cased.
    const wanted = sku.trim().toUpperCase();
    return this.state.lines.find((line) => line.sku.toUpperCase() === wanted);
  }

  private update(patch: Partial<CartState>): void {
    const next = { ...this.state, ...patch };
    if (next.lines !== this.state.lines && !Array.isArray(next.lines)) {
      // Defensive: a malformed response must not poison every subscriber.
      console.error("cart update ignored: lines is not an array");
      return;
    }
    this.state = next;
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
    if (response.version <= this.state.version) {
      return;
    }
    if (response.cartId !== this.state.cartId) {
      // The server merged an anonymous cart into the account cart on sign-in.
      console.info(`cart ${this.state.cartId} merged into ${response.cartId}`);
      // Drop the old snapshot so a later reload doesn't resurrect the guest cart.
      safeStorage()?.removeItem(STORAGE_KEY_PREFIX + this.state.cartId);
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
    const key = STORAGE_KEY_PREFIX + this.state.cartId;
    const raw = storage?.getItem(key);
    if (!raw) {
      return;
    }
    try {
      const snapshot = JSON.parse(raw) as Pick<CartState, "lines" | "version"> & { savedAt?: number };
      if (typeof snapshot.savedAt === "number" && Date.now() - snapshot.savedAt > SNAPSHOT_MAX_AGE_MS) {
        storage?.removeItem(key);
        return;
      }
      if (Array.isArray(snapshot.lines) && typeof snapshot.version === "number") {
        // Render the last known cart instantly; refresh() replaces it.
        this.state = { ...this.state, lines: snapshot.lines, version: snapshot.version };
      }
    } catch {
      // Corrupt JSON (e.g. a truncated write); start from an empty cart.
      storage?.removeItem(key);
    }
  }

  private saveSnapshot(): void {
    const storage = safeStorage();
    if (!storage) {
      return;
    }
    const key = STORAGE_KEY_PREFIX + this.state.cartId;
    if (this.state.lines.length === 0) {
      // An empty cart needs no snapshot; removing it keeps storage tidy.
      storage.removeItem(key);
      return;
    }
    try {
      storage.setItem(
        key,
        JSON.stringify({ lines: this.state.lines, version: this.state.version, savedAt: Date.now() }),
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
    if (response.cartId !== this.state.cartId) {
      // A coupon from a different cart (pre-merge) must not be shown here.
      return;
    }
    const code = normalizeCouponCode(response.couponCode);
    if (!COUPON_PATTERN.test(code)) {
      console.warn(`ignoring malformed coupon code from cart payload: ${response.couponCode}`);
      return;
    }
    this.update({ coupon: { status: "applied", code } });
  }

  async refresh(): Promise<void> {
    // Only the latest refresh matters; cancel any request still in flight.
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
    if (!this.lineFor(sku) && quantity === 0) {
      // Removing something that's already gone (double click) is a no-op.
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
          error: available > 0 ? `Only ${available} left in stock.` : "This item is out of stock.",
        });
        return;
      }
      this.update({ lines: previous, error: err instanceof ApiError ? err.message : "Could not update cart" });
    }
  }

  /** Debounced variant used by the quantity stepper. */
  scheduleQuantity(sku: string, quantity: number): void {
    if (!sku) {
      return;
    }
    // Last write wins per SKU: only the final stepper value is sent.
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
    const line = this.lineFor(sku);
    if (!line) {
      return;
    }
    // A pending stepper change for this SKU would re-add the line after the
    // removal lands, so drop it first.
    if (this.pendingQuantities.delete(line.sku) && this.pendingQuantities.size === 0 && this.quantityTimer !== null) {
      clearTimeout(this.quantityTimer);
      this.quantityTimer = null;
    }
    await this.setQuantity(line.sku, 0);
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
    if (this.state.coupon.status === "validating") {
      // Ignore double submits while the first code is still being checked.
      return false;
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
    if (current.status === "rejected" || current.status === "validating") {
      // Nothing was applied server-side; just clear the local message.
      this.update({ coupon: { status: "none" } });
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
    if (!code) {
      return "This code can't be used on your cart.";
    }
    // Error codes from older API versions were upper-case.
    const known = COUPON_ERROR_MESSAGES[code] ?? COUPON_ERROR_MESSAGES[code.toLowerCase()];
    return known ?? "This code can't be used on your cart.";
  }

  dispose(): void {
    this.inflight?.abort();
    this.inflight = null;
    if (this.quantityTimer !== null) {
      clearTimeout(this.quantityTimer);
      this.quantityTimer = null;
    }
    if (this.pendingQuantities.size > 0) {
      // Unsent stepper changes are dropped; the next page load shows the
      // server's view of the cart.
      console.debug(`cart ${this.state.cartId} disposed with ${this.pendingQuantities.size} pending changes`);
    }
    this.pendingQuantities.clear();
    this.listeners.clear();
  }
}
