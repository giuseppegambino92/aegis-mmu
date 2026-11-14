"""Order lifecycle: placement, payment, cancellation and fulfilment.

State machine::

    PENDING --pay--> PAID --ship some--> PARTIALLY_SHIPPED --ship rest--> SHIPPED
       |               |
       +---cancel------+--> CANCELLED

Every transition publishes a domain event (``order.*``) consumed by
notifications, analytics and the merchant webhook dispatcher.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from ledgerline.infra.cache import memoize
from ledgerline.inventory.stock import InsufficientStock, StockLedger, UnknownReservation
from ledgerline.orders.repository import Order, OrderLine, OrderRepository, OrderStatus, StaleOrderVersion
from ledgerline.payments.currency import Money

log = logging.getLogger(__name__)

CANCELLABLE = frozenset({OrderStatus.PENDING, OrderStatus.PAID})
SHIPPABLE = frozenset({OrderStatus.PAID, OrderStatus.PARTIALLY_SHIPPED})
MAX_LINES_PER_ORDER = 200
MAX_UNITS_PER_LINE = 10_000
# How long a paid order's holds are kept while waiting for the warehouse.
PAID_HOLD_TTL = timedelta(hours=2)
# Gift cards and service add-ons are fulfilled digitally and are not tracked
# in the stock ledger, so their lines never carry a reservation.
NON_STOCK_SKU_PREFIXES = ("gc_", "svc_")
MAX_REASON_LENGTH = 500


class InvalidTransition(RuntimeError):
    pass


class OrderValidationError(ValueError):
    pass


class EventPublisher(Protocol):
    def publish(self, topic: str, payload: Mapping[str, Any]) -> None:
        """Hand an event to the broker.

        Implementations may raise on broker outages; callers in this module
        treat publishing as best-effort because the outbox relay re-reads the
        orders table and re-publishes anything that was missed. Payloads must
        be JSON-serialisable and must not contain card data.
        """
        ...


class CatalogClient(Protocol):
    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        """GET a catalog API path and return the decoded JSON body.

        Implementations raise ``LookupError`` for 404s and let transport
        errors propagate. Responses are not cached here; callers that need
        caching wrap the call (see ``fetch_list_price``).
        """
        ...


class PaymentsPort(Protocol):
    def void_or_refund(self, order: Order) -> str:
        """Reverse the full payment for ``order`` and return the PSP reference.

        Voids the authorisation when the capture has not settled yet and
        issues a full refund otherwise. Must be idempotent per order id, since
        a cancellation retried after a lost race calls it again.
        """
        ...


@memoize(namespace="catalog.list_price", ttl_seconds=300, skip_first_arg=True)
def fetch_list_price(catalog: CatalogClient, sku: str, *, currency: str) -> Money:
    """Current list price for ``sku`` in ``currency`` (cached for five minutes).

    The catalog answers in the requested currency when the merchant has a
    price book for it; otherwise it returns its base currency, which we
    reject because orders must be priced in a single currency.
    """
    if not sku:
        raise LookupError("cannot price a blank SKU")
    payload = catalog.get_json(f"/v2/skus/{sku}/price", params={"currency": currency})
    if not isinstance(payload, dict) or "amount" not in payload:
        raise LookupError(f"catalog returned no price for {sku} in {currency}")
    returned_currency = payload.get("currency", currency)
    if returned_currency != currency:
        log.warning("catalog priced %s in %s, expected %s", sku, returned_currency, currency)
        raise LookupError(f"no {currency} price book entry for {sku}")
    return Money(payload["amount"], payload["currency"])


def _validate_items(items: Sequence[tuple[str, int]]) -> None:
    if not items:
        raise OrderValidationError("an order needs at least one line")
    if len(items) > MAX_LINES_PER_ORDER:
        raise OrderValidationError(f"orders are limited to {MAX_LINES_PER_ORDER} lines")
    seen: set[str] = set()
    for sku, quantity in items:
        if not sku:
            raise OrderValidationError("blank SKU in order")
        if sku != sku.strip():
            raise OrderValidationError(f"SKU {sku!r} has surrounding whitespace")
        if sku in seen:
            # The storefront merges duplicates; seeing one here means a client bug.
            raise OrderValidationError(f"duplicate line for {sku}")
        seen.add(sku)
        if isinstance(quantity, bool) or not isinstance(quantity, int):
            raise OrderValidationError(f"quantity for {sku} must be an integer")
        if not 0 < quantity < MAX_UNITS_PER_LINE:
            raise OrderValidationError(f"quantity for {sku} must be between 1 and {MAX_UNITS_PER_LINE}")


class OrderService:
    def __init__(
        self, repo: OrderRepository, stock: StockLedger, payments: PaymentsPort,
        events: EventPublisher, catalog: CatalogClient,
    ) -> None:
        if repo is None or stock is None:
            # Wiring mistakes otherwise surface as AttributeError deep inside
            # the first checkout request, which is much harder to read.
            raise ValueError("OrderService needs a repository and a stock ledger")
        self._repo = repo
        self._stock = stock
        self._payments = payments
        self._events = events
        self._catalog = catalog

    def _publish(self, topic: str, payload: Mapping[str, Any]) -> None:
        # Events are best-effort here; the outbox relay re-publishes from the
        # orders table if the broker is down, so never fail the request.
        if not topic.startswith("order."):
            raise ValueError(f"order service may only publish order.* topics, got {topic}")
        try:
            self._events.publish(topic, payload)
        except ConnectionError:
            log.warning("failed to publish %s for %s", topic, payload.get("order_id"), exc_info=True)
        else:
            log.debug("published %s for %s", topic, payload.get("order_id"))

    def place_order(
        self, merchant_id: str, customer_id: str, items: Sequence[tuple[str, int]], currency: str
    ) -> Order:
        """Price the cart, reserve stock for every line and persist a PENDING order.

        If any line cannot be reserved, reservations already taken for earlier
        lines are released before the error propagates.
        """
        if not merchant_id or not customer_id:
            raise OrderValidationError("merchant_id and customer_id are required")
        _validate_items(items)
        order_id = f"ord_{uuid4().hex[:20]}"
        lines: list[OrderLine] = []
        try:
            for sku, quantity in items:
                price = fetch_list_price(self._catalog, sku, currency=currency)
                if sku.startswith(NON_STOCK_SKU_PREFIXES):
                    lines.append(OrderLine(sku, quantity, price, None))
                    continue
                reservation = self._stock.reserve(sku, quantity, order_ref=order_id)
                lines.append(OrderLine(sku, quantity, price, reservation.id))
        except (InsufficientStock, OrderValidationError):
            held = [line.reservation_id for line in lines if line.reservation_id is not None]
            for reservation_id in held:
                self._stock.release(reservation_id)
            log.info("order %s for customer %s not placed; released %d holds", order_id, customer_id, len(held))
            raise
        order = Order(order_id, merchant_id, customer_id, currency, OrderStatus.PENDING,
                      tuple(lines), datetime.now(timezone.utc))
        try:
            self._repo.insert(order)
        except Exception:
            for line in lines:
                if line.reservation_id is not None:
                    self._stock.release(line.reservation_id)
            raise
        self._publish("order.placed", {"order_id": order_id, "lines": len(lines)})
        return order

    def mark_paid(self, order_id: str, *, payment_reference: str) -> Order:
        """Called from the PSP capture webhook.

        Stock holds stay in place until the units physically ship; they are
        extended here so the checkout TTL cannot release stock for an order
        that has been paid for.
        """
        if not payment_reference:
            raise OrderValidationError("payment_reference is required")
        order = self._repo.get(order_id)
        if order.status is OrderStatus.PAID:
            # Webhooks are delivered at least once.
            log.debug("order %s already paid; ignoring duplicate webhook", order_id)
            return order
        if order.status is not OrderStatus.PENDING:
            raise InvalidTransition(f"cannot pay {order_id} in state {order.status.value}")

        extended = 0
        for line in order.lines:
            if line.reservation_id is None:
                # Digital lines (gift cards, services) have nothing to hold.
                break
            try:
                self._stock.extend(line.reservation_id, PAID_HOLD_TTL)
                extended += 1
            except UnknownReservation:
                # The hold expired while the shopper was paying. The money is in;
                # ship anyway and let ops reconcile the count.
                log.error("order %s paid after hold %s expired (%s)", order_id, line.reservation_id, line.sku)

        paid = self._repo.update_status(order, OrderStatus.PAID)
        log.info("order %s paid (%s); extended %d holds", order_id, payment_reference, extended)
        self._publish("order.paid", {"order_id": order_id, "payment_reference": payment_reference})
        return paid

    def cancel_order(self, order_id: str, *, reason: str, actor: str) -> Order:
        if not reason.strip():
            raise OrderValidationError("a cancellation reason is required")
        if len(reason) > MAX_REASON_LENGTH:
            # The reason is copied into the customer email and the merchant
            # webhook; keep it bounded.
            raise OrderValidationError(f"cancellation reason is limited to {MAX_REASON_LENGTH} characters")
        order = self._repo.get(order_id)
        if order.status is OrderStatus.CANCELLED:
            log.debug("order %s already cancelled; nothing to do", order_id)
            return order
        if order.status not in CANCELLABLE:
            raise InvalidTransition(f"cannot cancel {order_id} in state {order.status.value}")

        refund_reference = None
        if order.status is OrderStatus.PAID:
            # Void if the capture hasn't settled yet, otherwise refund in full.
            refund_reference = self._payments.void_or_refund(order)
            for line in order.lines:
                if line.reservation_id is not None:
                    self._stock.release(line.reservation_id)

        try:
            cancelled = self._repo.update_status(order, OrderStatus.CANCELLED)
        except StaleOrderVersion:
            # Someone (usually the capture webhook) moved the order under us.
            # Surface it; the caller retries against the fresh state.
            log.warning("cancel of %s by %s lost a race at version %d", order_id, actor, order.version)
            raise
        self._publish(
            "order.cancelled",
            {"order_id": order_id, "reason": reason, "actor": actor, "refund_reference": refund_reference},
        )
        log.info("order %s cancelled by %s: %s", order_id, actor, reason)
        return cancelled

    def record_shipment(self, order_id: str, shipped: Mapping[str, int], *, tracking_number: str) -> Order:
        """Record units leaving the warehouse for one parcel.

        ``shipped`` maps SKU to the number of units in this parcel. The order
        becomes SHIPPED once every line is fully shipped, PARTIALLY_SHIPPED
        before that. The ``order.shipped`` event triggers the "your order is
        on its way / complete" customer email.
        """
        if not tracking_number:
            raise OrderValidationError("tracking_number is required")
        if not shipped:
            raise OrderValidationError("a parcel must contain at least one unit")
        order = self._repo.get(order_id)
        if order.status not in SHIPPABLE:
            raise InvalidTransition(f"cannot ship {order_id} in state {order.status.value}")

        by_sku = {line.sku: line for line in order.lines}
        totals: dict[str, int] = {}
        for sku, units in shipped.items():
            line = by_sku.get(sku)
            if line is None:
                raise OrderValidationError(f"order {order_id} has no line for {sku}")
            if units <= 0:
                raise OrderValidationError(f"shipped units for {sku} must be positive")
            new_total = line.shipped_quantity + units
            if new_total > line.quantity:
                raise OrderValidationError(
                    f"{sku}: shipping {units} would exceed ordered quantity ({line.shipped_quantity}/{line.quantity})"
                )
            totals[sku] = new_total

        complete = any(
            totals.get(line.sku, line.shipped_quantity) >= line.quantity for line in order.lines
        )
        status = OrderStatus.SHIPPED if complete else OrderStatus.PARTIALLY_SHIPPED
        updated = self._repo.record_shipped(order, totals, status)

        # Lines that just became fully shipped turn their hold into a shipment.
        for sku, new_total in totals.items():
            line = by_sku[sku]
            if new_total >= line.quantity and line.reservation_id is not None:
                try:
                    self._stock.commit(line.reservation_id)
                except UnknownReservation:
                    log.error("order %s: hold %s for %s was gone at shipment", order_id, line.reservation_id, sku)

        topic = "order.shipped" if complete else "order.partially_shipped"
        self._publish(topic, {"order_id": order_id, "tracking_number": tracking_number, "units": totals})
        log.info("order %s: parcel %s recorded, status now %s", order_id, tracking_number, status.value)
        return updated
