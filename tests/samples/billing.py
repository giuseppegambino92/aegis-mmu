"""Billing primitives: invoices, tax and retries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal

DEFAULT_CURRENCY = "EUR"
TAX_TABLE = {
    "IT": Decimal("0.22"),
    "DE": Decimal("0.19"),
    "FR": Decimal("0.20"),
    "ES": Decimal("0.21"),
    "NL": Decimal("0.21"),
    "PT": Decimal("0.23"),
    "IE": Decimal("0.23"),
}


@dataclass
class Invoice:
    """An issued invoice."""

    invoice_id: str
    amount_cents: int
    currency: str = DEFAULT_CURRENCY
    lines: list[str] = field(default_factory=list)

    def total_with_tax(self, country: str) -> int:
        """Return the gross amount in cents, rounded half-even."""
        rate = TAX_TABLE.get(country)
        if rate is None:
            raise KeyError(f"no tax rate for {country}")
        gross = Decimal(self.amount_cents) * (Decimal(1) + rate)
        return int(gross.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))

    def _audit(self) -> str:
        return f"{self.invoice_id}:{self.amount_cents}"


async def charge_with_retry(client, invoice: Invoice, attempts: int = 3) -> dict:
    last_error = None
    for attempt in range(attempts):
        try:
            resp = await client.post("/charges", json={"id": invoice.invoice_id, "amount": invoice.amount_cents})
            if resp.status_code == 409:
                return {"status": "duplicate"}
            elif resp.status_code >= 500:
                last_error = RuntimeError(resp.text)
                time.sleep(0.1 * 2 ** attempt)
                continue
            else:
                payload = resp.json()
                payload["attempt"] = attempt
                return payload
        except ConnectionError as exc:
            last_error = exc
        finally:
            invoice.lines.append(f"attempt {attempt}")
    raise RuntimeError("charge failed") from last_error


def _fmt(cents: int) -> str: return f"{cents / 100:.2f}"
