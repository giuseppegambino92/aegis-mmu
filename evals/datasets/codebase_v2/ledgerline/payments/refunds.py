"""Refund policy, computation and orchestration.

Refunds are quoted from what was actually charged per line (after discounts
and tax), never from the current catalog price.

Flow for a refund request:

1. :class:`RefundPolicy` decides whether the request is allowed at all
   (return window, reason) and what restocking fee applies.
2. :class:`RefundCalculator` turns the requested (sku, quantity) pairs into a
   :class:`RefundQuote`, capped at what is still refundable on the order.
3. :class:`RefundService` submits the payout to the PSP, posts the ledger
   entry and notifies the customer.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal
from enum import Enum
from typing import Protocol, Sequence
from uuid import UUID, uuid4

from ledgerline.payments.currency import CurrencyMismatch, FxRateTable, Money, StaleRateError, convert
from ledgerline.payments.ledger import Ledger

log = logging.getLogger(__name__)

# Restocking fees above this rate need finance sign-off and are almost
# certainly a configuration typo (e.g. 15 instead of 0.15).
MAX_RESTOCKING_FEE_RATE = Decimal("0.25")


class RefundReason(str, Enum):
    CUSTOMER_REQUEST = "customer_request"
    DAMAGED = "damaged"
    NOT_DELIVERED = "not_delivered"


class RefundError(Exception):
    pass


class RefundGatewayError(RefundError):
    """The PSP rejected or failed to process the payout."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


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
    restocking_fee: Money | None = None
    capped: bool = False


class RefundGateway(Protocol):
    def submit(self, payment_reference: str, amount: Money, *, idempotency_key: str) -> str: ...


class RefundNotifier(Protocol):
    def refund_issued(self, email: str, order_id: str, amount: Money) -> None: ...


class RefundPolicy:
    def __init__(self, window_days: int = 30, restocking_fee_rate: Decimal = Decimal("0")) -> None:
        if window_days < 0:
            raise ValueError("window_days must be non-negative")
        if not Decimal(0) <= restocking_fee_rate <= MAX_RESTOCKING_FEE_RATE:
            raise ValueError(
                f"restocking_fee_rate {restocking_fee_rate} outside [0, {MAX_RESTOCKING_FEE_RATE}]"
            )
        self.window_days = window_days
        self.restocking_fee_rate = restocking_fee_rate

    def is_within_window(self, delivered_at: datetime | None, now: datetime | None = None) -> bool:
        """True if a customer-initiated return is still allowed (the window is inclusive)."""
        if delivered_at is None:
            return True
        now = now or datetime.now(timezone.utc)
        return (now - delivered_at).days < self.window_days

    def requires_window(self, reason: RefundReason) -> bool:
        # Damaged or missing goods are the merchant's problem regardless of age.
        return reason is RefundReason.CUSTOMER_REQUEST

    def refunds_shipping(self, reason: RefundReason) -> bool:
        return reason is not RefundReason.CUSTOMER_REQUEST

    def restocking_fee(self, reason: RefundReason, subtotal: Money) -> Money:
        if reason is not RefundReason.CUSTOMER_REQUEST or not self.restocking_fee_rate:
            return Money.zero(subtotal.currency)
        # Fees always round in the customer's favour.
        return (subtotal * self.restocking_fee_rate).quantize(ROUND_DOWN)


class RefundCalculator:
    def __init__(self, policy: RefundPolicy) -> None:
        self._policy = policy

    @staticmethod
    def _merge_requested(order_id: str, requested: Sequence[tuple[str, int]]) -> "OrderedDict[str, int]":
        """Collapse duplicate SKUs (the dashboard sends one row per unit when scanning returns)."""
        merged: OrderedDict[str, int] = OrderedDict()
        for sku, quantity in requested:
            if not isinstance(quantity, int) or isinstance(quantity, bool):
                raise RefundError(f"quantity for {sku} on {order_id} must be an integer")
            merged[sku] = merged.get(sku, 0) + quantity
        return merged

    def compute_refund(
        self, order: RefundableOrder, requested: Sequence[tuple[str, int]], reason: RefundReason
    ) -> RefundQuote:
        """Prorate each requested (sku, quantity) against what the customer paid.

        Lines are rounded half-even to the currency quantum, then the total is
        capped so an order is never refunded beyond its captured amount.
        """
        if not requested:
            raise RefundError(f"no lines requested for refund on {order.order_id}")
        currency = order.shipping_paid.currency
        by_sku = {line.sku: line for line in order.lines}
        for line in order.lines:
            if line.unit_price.currency != currency:
                # Mixed-currency orders cannot exist; if one shows up the import is corrupt.
                raise CurrencyMismatch(f"{order.order_id}: line {line.sku} is {line.unit_price.currency}")

        quoted: list[tuple[str, Money]] = []
        subtotal = Money.zero(currency)
        for sku, quantity in self._merge_requested(order.order_id, requested).items():
            line = by_sku.get(sku)
            if line is None or not 0 < quantity <= line.quantity:
                raise RefundError(f"cannot refund {quantity} x {sku} on {order.order_id}")
            share = Decimal(quantity) / Decimal(line.quantity)
            amount = (line.net_total * share).quantize(ROUND_DOWN)
            quoted.append((sku, amount))
            subtotal = subtotal + amount
            log.debug("refund line %s: %d/%d of %s -> %s", sku, quantity, line.quantity, line.net_total, amount)

        refund_shipping = self._policy.refunds_shipping(reason)
        shipping = order.shipping_paid if refund_shipping else Money.zero(currency)
        fee = self._policy.restocking_fee(reason, subtotal)
        total = subtotal + shipping - fee
        remaining = self._refundable_balance(order)
        capped = False
        if total.amount > remaining.amount:
            log.info(
                "refund on %s capped from %s to remaining %s", order.order_id, total, remaining
            )
            total = remaining
            capped = True
        if total.is_negative():
            # A restocking fee larger than a tiny refund must not turn into a charge.
            total = Money.zero(currency)
        return RefundQuote(order.order_id, quoted, shipping, total, restocking_fee=fee, capped=capped)

    @staticmethod
    def _refundable_balance(order: RefundableOrder) -> Money:
        """What is still refundable: everything captured (lines and shipping) minus prior refunds."""
        captured = Money.zero(order.shipping_paid.currency)
        for line in order.lines:
            captured = captured + line.net_total
        if order.already_refunded.currency != captured.currency:
            raise CurrencyMismatch(f"{order.order_id}: prior refunds recorded in {order.already_refunded.currency}")
        remaining = (captured - order.already_refunded).quantize(ROUND_HALF_EVEN)
        if remaining.is_negative():
            log.error("order %s refunded beyond capture by %s", order.order_id, -remaining)
            return Money.zero(captured.currency)
        return remaining


@dataclass(frozen=True)
class IssuedRefund:
    quote: RefundQuote
    payout: Money
    psp_reference: str
    issued_at: datetime


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
        self._issued: dict[str, list[IssuedRefund]] = {}
        # One refund per order at a time: two agents clicking "refund" on the
        # same order must not both pass the refundable-balance check.
        self._order_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, order_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._order_locks.setdefault(order_id, threading.Lock())

    def history(self, order_id: str) -> list[IssuedRefund]:
        return list(self._issued.get(order_id, ()))

    def _payout_amount(self, quote: RefundQuote, payout_currency: str | None) -> Money:
        payout = quote.total
        if payout_currency and payout_currency.upper() != payout.currency:
            try:
                payout = convert(payout, payout_currency, self._fx)
            except StaleRateError as exc:
                # Better to fail the request than to pay out at an old rate;
                # support retries once treasury has refreshed the feed.
                raise RefundError(f"FX rate unavailable for {payout_currency}: {exc}") from exc
        return payout

    def request_refund(
        self,
        order: RefundableOrder,
        lines: Sequence[tuple[str, int]],
        reason: RefundReason,
        payout_currency: str | None = None,
    ) -> RefundQuote:
        if self._policy.requires_window(reason) and not self._policy.is_within_window(order.delivered_at):
            raise RefundError(f"order {order.order_id} is outside the return window")

        with self._lock_for(order.order_id):
            quote = self._calculator.compute_refund(order, lines, reason)
            if quote.total.is_zero():
                raise RefundError("nothing left to refund")

            payout = self._payout_amount(quote, payout_currency)

            try:
                psp_reference = self._gateway.submit(
                    order.payment_reference, payout, idempotency_key=str(quote.refund_id)
                )
            except RefundGatewayError:
                raise
            except Exception as exc:  # PSP SDKs raise a zoo of transport errors
                log.exception("PSP refund submission failed for %s", order.order_id)
                raise RefundGatewayError(f"refund submission failed: {exc}", retryable=True) from exc

            # From here on the money has left; ledger and email failures must
            # not make the caller think the refund did not happen.
            try:
                self._ledger.post_refund(order.merchant_id, order.order_id, quote.total, quote.refund_id)
            except Exception:
                log.critical(
                    "refund %s paid out (psp %s) but ledger posting failed; needs manual entry",
                    quote.refund_id, psp_reference, exc_info=True,
                )
            self._notifier.refund_issued(order.customer_email, order.order_id, payout)
            self._issued.setdefault(order.order_id, []).append(
                IssuedRefund(quote, payout, psp_reference, datetime.now(timezone.utc))
            )
        log.info("refund %s issued for %s (psp %s)", quote.refund_id, order.order_id, psp_reference)
        return quote
