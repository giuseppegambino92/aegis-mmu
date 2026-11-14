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

# Return windows longer than a year are not something any merchant contract
# allows; a value above this is a units mistake (hours instead of days).
MAX_WINDOW_DAYS = 365

# Per-order locks are kept in memory; once this many are tracked, idle ones
# are dropped so a long-running worker does not grow without bound.
MAX_TRACKED_ORDER_LOCKS = 10_000


class RefundReason(str, Enum):
    CUSTOMER_REQUEST = "customer_request"
    DAMAGED = "damaged"
    NOT_DELIVERED = "not_delivered"


class RefundError(Exception):
    pass


class RefundGatewayError(RefundError):
    """The PSP rejected or failed to process the payout."""

    def __init__(self, message: str, *, retryable: bool, psp_code: str | None = None) -> None:
        # The PSP error code is what support searches for in the PSP dashboard,
        # so it goes into the message as well as onto the attribute.
        if psp_code:
            message = f"{message} [psp:{psp_code}]"
        super().__init__(message)
        self.retryable = retryable
        self.psp_code = psp_code
        if not retryable:
            # Non-retryable rejections (card closed, refund window at the PSP
            # expired) always need a human; make them easy to find in logs.
            log.warning("non-retryable refund gateway error: %s", message)


@dataclass(frozen=True)
class ChargedLine:
    sku: str
    quantity: int
    unit_price: Money
    discount: Money
    tax: Money

    @property
    def net_total(self) -> Money:
        """What the customer actually paid for this line: units minus discount plus tax.

        The importer guarantees all three amounts share a currency; Money
        arithmetic raises CurrencyMismatch if that ever stops being true.
        A discount larger than the gross line would yield a negative total,
        which we log but do not clamp, because clamping would hide bad data.
        """
        gross = self.unit_price * self.quantity
        total = gross - self.discount + self.tax
        if total.is_negative():
            log.warning("line %s has negative net total %s (gross %s, discount %s)",
                        self.sku, total, gross, self.discount)
        return total


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
    def submit(self, payment_reference: str, amount: Money, *, idempotency_key: str) -> str:
        """Ask the PSP to return ``amount`` against the original charge.

        Implementations must be idempotent on ``idempotency_key``: retrying a
        submission that timed out must not refund twice. They return the PSP's
        refund reference, and raise :class:`RefundGatewayError` for rejections
        the PSP reported explicitly (anything else is treated as transport
        failure and retried).
        """
        ...


class RefundNotifier(Protocol):
    def refund_issued(self, email: str, order_id: str, amount: Money) -> None:
        """Tell the customer a refund of ``amount`` is on its way for ``order_id``.

        Called after the PSP accepted the payout, so implementations should
        swallow and log their own delivery failures rather than raise: the
        refund has happened whether or not the email goes out. ``amount`` is
        in the payout currency, which may differ from the order currency.
        """
        ...


class RefundPolicy:
    def __init__(self, window_days: int = 30, restocking_fee_rate: Decimal = Decimal("0")) -> None:
        if window_days < 0:
            raise ValueError("window_days must be non-negative")
        if not Decimal(0) <= restocking_fee_rate <= MAX_RESTOCKING_FEE_RATE:
            raise ValueError(
                f"restocking_fee_rate {restocking_fee_rate} outside [0, {MAX_RESTOCKING_FEE_RATE}]"
            )
        if window_days > MAX_WINDOW_DAYS:
            raise ValueError(f"window_days {window_days} exceeds {MAX_WINDOW_DAYS}")
        self.window_days = window_days
        self.restocking_fee_rate = restocking_fee_rate
        log.debug("refund policy: %d-day window, %s restocking fee", window_days, restocking_fee_rate)

    def is_within_window(self, delivered_at: datetime | None, now: datetime | None = None) -> bool:
        """True if a customer-initiated return is still allowed (the window is inclusive)."""
        if delivered_at is None:
            # Not delivered yet (or carrier never confirmed): the clock has not
            # started, so the customer is always inside the window.
            return True
        if delivered_at.tzinfo is None:
            raise ValueError("delivered_at must be timezone-aware")
        now = now or datetime.now(timezone.utc)
        if delivered_at > now:
            # Carrier scans occasionally arrive with a future timestamp from a
            # mis-set handheld; treat the parcel as delivered just now.
            log.warning("delivery timestamp %s is in the future; treating as now", delivered_at)
            return True
        return (now - delivered_at).days < self.window_days

    def requires_window(self, reason: RefundReason) -> bool:
        """Whether the return window applies to a refund for ``reason``.

        Damaged or missing goods are the merchant's problem regardless of age,
        so only customer-initiated returns ("changed my mind") are limited.
        Reasons arrive as strings from the dashboard; unknown values are a
        client bug and rejected here rather than defaulting to "no window".
        """
        if not isinstance(reason, RefundReason):
            try:
                reason = RefundReason(reason)
            except ValueError:
                raise RefundError(f"unknown refund reason {reason!r}") from None
        return reason is RefundReason.CUSTOMER_REQUEST

    def refunds_shipping(self, reason: RefundReason) -> bool:
        """Whether the original shipping charge goes back to the customer.

        Consumer law in most of our markets requires refunding outbound
        shipping when the goods were faulty or never arrived; for a change of
        mind it is at the merchant's discretion and our default is no.
        """
        if not isinstance(reason, RefundReason):
            try:
                reason = RefundReason(reason)
            except ValueError:
                raise RefundError(f"unknown refund reason {reason!r}") from None
        return reason is RefundReason.DAMAGED

    def restocking_fee(self, reason: RefundReason, subtotal: Money) -> Money:
        """Fee withheld from a customer-initiated return, as a share of the line subtotal.

        Only applies to change-of-mind returns; shipping is never part of the
        base. A negative subtotal cannot occur for valid orders, but the
        importer has produced them before, and charging a negative fee would
        increase the refund, so it is treated as zero.
        """
        if reason is not RefundReason.CUSTOMER_REQUEST or not self.restocking_fee_rate:
            return Money.zero(subtotal.currency)
        if not subtotal.is_positive():
            log.warning("restocking fee requested on non-positive subtotal %s", subtotal)
            return Money.zero(subtotal.currency)
        # Fees always round in the customer's favour.
        fee = (subtotal * self.restocking_fee_rate).quantize(ROUND_DOWN)
        log.debug("restocking fee %s on %s at rate %s", fee, subtotal, self.restocking_fee_rate)
        return fee


class RefundCalculator:
    def __init__(self, policy: RefundPolicy) -> None:
        if policy is None:
            raise ValueError("RefundCalculator needs a RefundPolicy")
        # The calculator only reads the policy, but keeping a reference (rather
        # than copying its fields) means a hot-reloaded policy takes effect on
        # the next quote without rebuilding the service graph.
        self._policy = policy
        log.debug(
            "refund calculator using %d-day window, restocking fee rate %s",
            policy.window_days, policy.restocking_fee_rate,
        )

    @staticmethod
    def _merge_requested(order_id: str, requested: Sequence[tuple[str, int]]) -> "OrderedDict[str, int]":
        """Collapse duplicate SKUs (the dashboard sends one row per unit when scanning returns)."""
        merged: OrderedDict[str, int] = OrderedDict()
        for sku, quantity in requested:
            if not sku or not isinstance(sku, str):
                raise RefundError(f"refund row on {order_id} is missing a SKU")
            if not isinstance(quantity, int) or isinstance(quantity, bool):
                raise RefundError(f"quantity for {sku} on {order_id} must be an integer")
            if quantity <= 0:
                # Zero/negative rows come from un-ticking a scanned item; they
                # must not cancel out a positive row for the same SKU.
                raise RefundError(f"quantity for {sku} on {order_id} must be positive, got {quantity}")
            merged[sku] = quantity
        if len(merged) != len(requested):
            log.debug("merged %d refund rows into %d SKUs for %s", len(requested), len(merged), order_id)
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
        missing = [
            name for name, dep in (
                ("calculator", calculator), ("policy", policy), ("ledger", ledger),
                ("gateway", gateway), ("notifier", notifier), ("fx", fx),
            ) if dep is None
        ]
        if missing:
            raise ValueError(f"RefundService is missing dependencies: {', '.join(missing)}")
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
        """The lock serialising refunds on ``order_id``, created on first use.

        Locks are never removed while held. When the table grows past
        ``MAX_TRACKED_ORDER_LOCKS`` the idle ones are dropped; a dropped lock
        is simply recreated the next time its order is refunded.
        """
        if not order_id:
            raise RefundError("order_id is required to serialise refunds")
        with self._locks_guard:
            lock = self._order_locks.get(order_id)
            if lock is not None:
                return lock
            if len(self._order_locks) >= MAX_TRACKED_ORDER_LOCKS:
                idle = [oid for oid, held in self._order_locks.items() if not held.locked()]
                for oid in idle:
                    del self._order_locks[oid]
                log.info("dropped %d idle refund locks (%d still held)", len(idle), len(self._order_locks))
            lock = threading.Lock()
            return lock

    def history(self, order_id: str) -> list[IssuedRefund]:
        """Refunds issued for ``order_id`` by this process, oldest first.

        This is an in-memory convenience for the support console; the ledger
        (entries referenced by the order id) is the source of truth across
        restarts. A copy is returned so callers cannot mutate our records,
        and the order lock is not taken: a refund being issued right now
        simply shows up on the next call.
        """
        if not order_id:
            raise ValueError("order_id is required")
        issued = self._issued.get(order_id, ())
        ordered = sorted(issued, key=lambda refund: refund.issued_at)
        log.debug("refund history for %s: %d entries", order_id, len(ordered))
        return ordered

    def _payout_amount(self, quote: RefundQuote, payout_currency: str | None) -> Money:
        payout = quote.total
        if payout_currency is not None and not payout_currency.strip():
            # An empty string from the dashboard means "same as the charge".
            payout_currency = None
        if payout_currency and payout_currency.upper() != payout.currency:
            log.info("refund %s pays out in %s (charged in %s)", quote.refund_id,
                     payout_currency.upper(), payout.currency)
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
                self._ledger.post_refund(order.merchant_id, order.order_id, payout, quote.refund_id)
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
