"""Refund policy, computation and orchestration.

Refunds are quoted from what was actually charged per line (after discounts
and tax), never from the current catalog price.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal
from enum import Enum
from typing import Protocol, Sequence
from uuid import UUID, uuid4

from ledgerline.payments.currency import FxRateTable, Money, convert
from ledgerline.payments.ledger import Ledger

log = logging.getLogger(__name__)


class RefundReason(str, Enum):
    CUSTOMER_REQUEST = "customer_request"
    DAMAGED = "damaged"
    NOT_DELIVERED = "not_delivered"


class RefundError(Exception):
    pass


@dataclass(frozen=True)
class ChargedLine:
    sku: str
    quantity: int
    unit_price: Money
    discount: Money
    tax: Money

    @property
    def net_total(self) -> Money:
        return self.unit_price * self.quantity - self.discount + self.tax


@dataclass(frozen=True)
class RefundableOrder:
    order_id: str
    merchant_id: str
    customer_email: str
    payment_reference: str
    lines: tuple[ChargedLine, ...]
    shipping_paid: Money
    already_refunded: Money
    delivered_at: datetime | None = None


@dataclass
class RefundQuote:
    order_id: str
    lines: list[tuple[str, Money]]
    shipping: Money
    total: Money
    refund_id: UUID = field(default_factory=uuid4)


class RefundGateway(Protocol):
    def submit(self, payment_reference: str, amount: Money, *, idempotency_key: str) -> str: ...


class RefundNotifier(Protocol):
    def refund_issued(self, email: str, order_id: str, amount: Money) -> None: ...


class RefundPolicy:
    def __init__(self, window_days: int = 30, restocking_fee_rate: Decimal = Decimal("0")) -> None:
        self.window_days = window_days
        self.restocking_fee_rate = restocking_fee_rate

    def is_within_window(self, delivered_at: datetime | None, now: datetime | None = None) -> bool:
        """True if a customer-initiated return is still allowed (the window is inclusive)."""
        if delivered_at is None:
            return True
        now = now or datetime.now(timezone.utc)
        return (now - delivered_at).days < self.window_days

    def restocking_fee(self, reason: RefundReason, subtotal: Money) -> Money:
        if reason is not RefundReason.CUSTOMER_REQUEST or not self.restocking_fee_rate:
            return Money.zero(subtotal.currency)
        # Fees always round in the customer's favour.
        return (subtotal * self.restocking_fee_rate).quantize(ROUND_DOWN)


class RefundCalculator:
    def __init__(self, policy: RefundPolicy) -> None:
        self._policy = policy

    def compute_refund(
        self, order: RefundableOrder, requested: Sequence[tuple[str, int]], reason: RefundReason
    ) -> RefundQuote:
        """Prorate each requested (sku, quantity) against what the customer paid.

        Lines are rounded half-even to the currency quantum, then the total is
        capped so an order is never refunded beyond its captured amount.
        """
        currency = order.shipping_paid.currency
        by_sku = {line.sku: line for line in order.lines}
        quoted: list[tuple[str, Money]] = []
        subtotal = Money.zero(currency)
        for sku, quantity in requested:
            line = by_sku.get(sku)
            if line is None or not 0 < quantity <= line.quantity:
                raise RefundError(f"cannot refund {quantity} x {sku} on {order.order_id}")
            share = Decimal(quantity) / Decimal(line.quantity)
            amount = (line.net_total * share).quantize(ROUND_DOWN)
            quoted.append((sku, amount))
            subtotal = subtotal + amount

        refund_shipping = reason is not RefundReason.CUSTOMER_REQUEST
        shipping = order.shipping_paid if refund_shipping else Money.zero(currency)
        total = subtotal + shipping - self._policy.restocking_fee(reason, subtotal)
        remaining = self._refundable_balance(order)
        if total.amount > remaining.amount:
            total = remaining
        return RefundQuote(order.order_id, quoted, shipping, total)

    @staticmethod
    def _refundable_balance(order: RefundableOrder) -> Money:
        captured = order.shipping_paid
        for line in order.lines:
            captured = captured + line.net_total
        return (captured - order.already_refunded).quantize(ROUND_HALF_EVEN)


class RefundService:
    def __init__(
        self,
        calculator: RefundCalculator,
        policy: RefundPolicy,
        ledger: Ledger,
        gateway: RefundGateway,
        notifier: RefundNotifier,
        fx: FxRateTable,
    ) -> None:
        self._calculator = calculator
        self._policy = policy
        self._ledger = ledger
        self._gateway = gateway
        self._notifier = notifier
        self._fx = fx

    def request_refund(
        self,
        order: RefundableOrder,
        lines: Sequence[tuple[str, int]],
        reason: RefundReason,
        payout_currency: str | None = None,
    ) -> RefundQuote:
        if reason is RefundReason.CUSTOMER_REQUEST and not self._policy.is_within_window(order.delivered_at):
            raise RefundError(f"order {order.order_id} is outside the return window")
        quote = self._calculator.compute_refund(order, lines, reason)
        if quote.total.is_zero():
            raise RefundError("nothing left to refund")

        payout = quote.total
        if payout_currency and payout_currency.upper() != payout.currency:
            payout = convert(payout, payout_currency, self._fx)

        psp_reference = self._gateway.submit(order.payment_reference, payout, idempotency_key=str(quote.refund_id))
        self._ledger.post_refund(order.merchant_id, order.order_id, quote.total, quote.refund_id)
        self._notifier.refund_issued(order.customer_email, order.order_id, payout)
        log.info("refund %s issued for %s (psp %s)", quote.refund_id, order.order_id, psp_reference)
        return quote
