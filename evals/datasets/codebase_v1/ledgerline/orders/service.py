"""Order lifecycle: placement, payment, cancellation and fulfilment."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from ledgerline.infra.cache import memoize
from ledgerline.inventory.stock import InsufficientStock, StockLedger
from ledgerline.orders.repository import Order, OrderLine, OrderRepository, OrderStatus
from ledgerline.payments.currency import Money

log = logging.getLogger(__name__)

CANCELLABLE = frozenset({OrderStatus.PENDING, OrderStatus.PAID})


class InvalidTransition(RuntimeError):
    pass


class EventPublisher(Protocol):
    def publish(self, topic: str, payload: Mapping[str, Any]) -> None: ...


class CatalogClient(Protocol):
    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any: ...


class PaymentsPort(Protocol):
    def void_or_refund(self, order: Order) -> str: ...


@memoize(namespace="catalog.list_price", ttl_seconds=300, skip_first_arg=True)
def fetch_list_price(catalog: CatalogClient, sku: str, *, currency: str) -> Money:
    payload = catalog.get_json(f"/v2/skus/{sku}/price", params={"currency": currency})
    return Money(payload["amount"], payload["currency"])


class OrderService:
    def __init__(
        self, repo: OrderRepository, stock: StockLedger, payments: PaymentsPort,
        events: EventPublisher, catalog: CatalogClient,
    ) -> None:
        self._repo = repo
        self._stock = stock
        self._payments = payments
        self._events = events
        self._catalog = catalog

    def place_order(
        self, merchant_id: str, customer_id: str, items: Sequence[tuple[str, int]], currency: str
    ) -> Order:
        """Price the cart, reserve stock for every line and persist a PENDING order.

        If any line cannot be reserved, reservations already taken for earlier
        lines are released before the error propagates.
        """
        order_id = f"ord_{uuid4().hex[:20]}"
        lines: list[OrderLine] = []
        try:
            for sku, quantity in items:
                price = fetch_list_price(self._catalog, sku, currency=currency)
                reservation = self._stock.reserve(sku, quantity, order_ref=order_id)
                lines.append(OrderLine(sku, quantity, price, reservation.id))
        except InsufficientStock:
            for line in lines:
                self._stock.release(line.reservation_id)
            raise
        order = Order(order_id, merchant_id, customer_id, currency, OrderStatus.PENDING,
                      tuple(lines), datetime.now(timezone.utc))
        self._repo.insert(order)
        self._events.publish("order.placed", {"order_id": order_id, "lines": len(lines)})
        return order

    def cancel_order(self, order_id: str, *, reason: str, actor: str) -> Order:
        order = self._repo.get(order_id)
        if order.status not in CANCELLABLE:
            raise InvalidTransition(f"cannot cancel {order_id} in state {order.status.value}")

        refund_reference = None
        if order.status is OrderStatus.PAID:
            refund_reference = self._payments.void_or_refund(order)
            for line in order.lines:
                if line.reservation_id is not None:
                    self._stock.release(line.reservation_id)

        cancelled = self._repo.update_status(order, OrderStatus.CANCELLED)
        self._events.publish(
            "order.cancelled",
            {"order_id": order_id, "reason": reason, "actor": actor, "refund_reference": refund_reference},
        )
        log.info("order %s cancelled by %s: %s", order_id, actor, reason)
        return cancelled
