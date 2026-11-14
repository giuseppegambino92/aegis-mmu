"""Persistence for orders and order lines (PostgreSQL via DB-API).

Orders use optimistic locking: every update bumps ``version`` and is
conditional on the version the caller read, so two workers racing on the same
order cannot silently overwrite each other.
"""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Sequence

from ledgerline.payments.currency import Money

log = logging.getLogger(__name__)

MAX_PAGE_SIZE = 100


class OrderStatus(str, Enum):
    PENDING = "pending"
    PAID = "paid"
    PARTIALLY_SHIPPED = "partially_shipped"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"


class OrderNotFound(LookupError):
    pass


class StaleOrderVersion(RuntimeError):
    """An optimistic-locking update lost the race."""


class InvalidCursor(ValueError):
    pass


@dataclass(frozen=True)
class OrderLine:
    sku: str
    quantity: int
    unit_price: Money
    reservation_id: str | None = None
    shipped_quantity: int = 0

    @property
    def fully_shipped(self) -> bool:
        return self.shipped_quantity >= self.quantity


@dataclass(frozen=True)
class Order:
    id: str
    merchant_id: str
    customer_id: str
    currency: str
    status: OrderStatus
    lines: tuple[OrderLine, ...]
    created_at: datetime
    version: int = 1


@dataclass(frozen=True)
class OrderPage:
    orders: list[Order]
    next_cursor: str | None


_COLUMNS = "id, merchant_id, customer_id, currency, status, created_at, version"
_LINE_COLUMNS = "sku, quantity, unit_price, reservation_id, shipped_quantity"


def _encode_cursor(order: Order) -> str:
    raw = json.dumps({"c": order.created_at.isoformat(), "i": order.id}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        data = json.loads(raw)
        return datetime.fromisoformat(data["c"]), str(data["i"])
    except (ValueError, KeyError, TypeError) as exc:
        raise InvalidCursor("cursor is not one we issued") from exc


class OrderRepository:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def get(self, order_id: str) -> Order:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT {_COLUMNS} FROM orders WHERE id = %s", (order_id,))
            row = cur.fetchone()
            if row is None:
                raise OrderNotFound(order_id)
            cur.execute(
                f"SELECT {_LINE_COLUMNS} "
                "FROM order_lines WHERE order_id = %s ORDER BY position",
                (order_id,),
            )
            return self._hydrate(row, cur.fetchall())

    def insert(self, order: Order) -> None:
        if not order.lines:
            raise ValueError(f"order {order.id} has no lines")
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO orders ({_COLUMNS}) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (order.id, order.merchant_id, order.customer_id, order.currency,
                     order.status.value, order.created_at, order.version),
                )
                cur.executemany(
                    "INSERT INTO order_lines (order_id, position, sku, quantity, unit_price, reservation_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [(order.id, pos, ln.sku, ln.quantity, str(ln.unit_price.amount), ln.reservation_id)
                     for pos, ln in enumerate(order.lines)],
                )
        except Exception:
            # Leave the connection usable for the next request in the pool.
            self._conn.rollback()
            log.exception("insert of order %s failed", order.id)
            raise
        self._conn.commit()

    def update_status(self, order: Order, status: OrderStatus) -> Order:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE orders SET status = %s, version = version + 1 WHERE id = %s AND version = %s",
                (status.value, order.id, order.version),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                raise StaleOrderVersion(f"order {order.id} changed since version {order.version}")
        self._conn.commit()
        return replace(order, status=status, version=order.version + 1)

    def record_shipped(self, order: Order, shipped: Mapping[str, int], status: OrderStatus) -> Order:
        """Persist new shipped quantities per SKU and the resulting status in one transaction."""
        unknown = set(shipped) - {line.sku for line in order.lines}
        if unknown:
            raise ValueError(f"order {order.id} has no lines for {sorted(unknown)}")
        new_lines = tuple(
            replace(line, shipped_quantity=shipped.get(line.sku, line.shipped_quantity))
            for line in order.lines
        )
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE orders SET status = %s, version = version + 1 WHERE id = %s AND version = %s",
                (status.value, order.id, order.version),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                raise StaleOrderVersion(f"order {order.id} changed since version {order.version}")
            cur.executemany(
                "UPDATE order_lines SET shipped_quantity = %s WHERE order_id = %s AND sku = %s",
                [(line.shipped_quantity, order.id, line.sku) for line in new_lines if line.sku in shipped],
            )
        self._conn.commit()
        return replace(order, lines=new_lines, status=status, version=order.version + 1)

    def list_for_customer(self, customer_id: str, *, limit: int = 20, cursor: str | None = None) -> OrderPage:
        """Newest-first order history for the customer account page.

        Pagination is keyset-based on ``(created_at, id)``: the cursor encodes
        the last order of the previous page and the next page starts strictly
        after it, so inserts between page loads never shift results.
        """
        if not 0 < limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        params: list[Any] = [customer_id]
        where = "customer_id = %s"
        if cursor is not None:
            created_at, last_id = _decode_cursor(cursor)
            where += " AND (created_at, id) <= (%s, %s)"
            params.extend([created_at, last_id])

        with self._conn.cursor() as cur:
            # Fetch one extra row to learn whether another page exists.
            cur.execute(
                f"SELECT {_COLUMNS} FROM orders WHERE {where} "
                "ORDER BY created_at DESC, id DESC LIMIT %s",
                (*params, limit + 1),
            )
            rows = cur.fetchall()
            has_more = len(rows) > limit
            rows = rows[:limit]
            if not rows:
                return OrderPage([], None)

            ids = [row[0] for row in rows]
            cur.execute(
                f"SELECT order_id, {_LINE_COLUMNS} FROM order_lines "
                "WHERE order_id = ANY(%s) ORDER BY order_id, position",
                (ids,),
            )
            lines_by_order: dict[str, list[Sequence[Any]]] = {order_id: [] for order_id in ids}
            for order_id, *line in cur.fetchall():
                lines_by_order[order_id].append(line)

        orders = [self._hydrate(row, lines_by_order[row[0]]) for row in rows]
        next_cursor = _encode_cursor(orders[-1]) if has_more else None
        return OrderPage(orders, next_cursor)

    @staticmethod
    def _hydrate(row: Sequence[Any], line_rows: Sequence[Sequence[Any]]) -> Order:
        order_id, merchant_id, customer_id, currency, status, created_at, version = row
        try:
            order_status = OrderStatus(status)
        except ValueError:
            # A newer deploy may have written a status this build doesn't know.
            raise RuntimeError(f"order {order_id} has unknown status {status!r}") from None
        lines = tuple(
            OrderLine(sku, qty, Money(Decimal(price), currency), res_id, shipped or 0)
            for sku, qty, price, res_id, shipped in line_rows
        )
        return Order(order_id, merchant_id, customer_id, currency, order_status, lines, created_at, version)

    def count_by_status(self, merchant_id: str, *, since: datetime | None = None) -> dict[OrderStatus, int]:
        """Order counts per status for the merchant dashboard tiles."""
        sql = "SELECT status, COUNT(*) FROM orders WHERE merchant_id = %s"
        params: list[Any] = [merchant_id]
        if since is not None:
            if since.tzinfo is None:
                raise ValueError("since must be timezone-aware")
            sql += " AND created_at >= %s"
            params.append(since)
        sql += " GROUP BY status"
        counts = {status: 0 for status in OrderStatus}
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            for status, count in cur.fetchall():
                try:
                    counts[OrderStatus(status)] = int(count)
                except ValueError:
                    log.warning("skipping unknown order status %r in counts", status)
        return counts

    def find_by_reservation(self, reservation_id: str) -> Order | None:
        """Reverse lookup used by the reservation sweeper to log which order lost a hold."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT order_id FROM order_lines WHERE reservation_id = %s LIMIT 1",
                (reservation_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        try:
            return self.get(row[0])
        except OrderNotFound:
            # Line without a parent order: a half-deleted test fixture or a bad migration.
            log.error("order line references missing order %s", row[0])
            return None
