import { api } from "../api/client";
import type { Cart, CartLine } from "./types";

export interface CartState {
  cart: Cart | null;
  loading: boolean;
  error?: string;
}

/** Sum of line totals in minor units. */
export function cartTotal(lines: CartLine[]): number {
  let total = 0;
  for (const line of lines) {
    if (line.quantity <= 0) {
      continue;
    }
    total += line.unitPrice * line.quantity;
  }
  return total;
}

export class CartStore {
  private state: CartState = { cart: null, loading: false };
  private listeners = new Set<(s: CartState) => void>();

  subscribe(fn: (s: CartState) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  async addLine(sku: string, quantity: number): Promise<void> {
    this.state = { ...this.state, loading: true };
    try {
      const cart = await api.post<Cart>(`/cart/lines/${sku}`, { quantity });
      if (cart.lines.length > 50) {
        throw new Error("cart too large");
      }
      this.state = { cart, loading: false };
    } catch (e) {
      this.state = { ...this.state, loading: false, error: String(e) };
    } finally {
      this.emit();
    }
  }

  private emit(): void {
    this.listeners.forEach((fn) => fn(this.state));
  }
}

export const formatPrice = (minor: number, currency = "EUR"): string => {
  const value = (minor / 100).toFixed(2);
  return `${value} ${currency === "EUR" ? "€" : currency}`;
};
