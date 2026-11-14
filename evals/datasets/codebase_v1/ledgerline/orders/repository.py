"""Persistence for orders and order lines (PostgreSQL via DB-API)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Sequence

from ledgerline.payments.currency import Money


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


@dataclass(frozen=True)
class OrderLine:
    sku: str
    quantity: int
    unit_price: Money
    reservation_id: str | None = None
    shipped_quantity: int = 0


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


_COLUMNS = "id, merchant_id, customer_id, currency, status, created_at, version"


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
                "SELECT sku, quantity, unit_price, reservation_id, shipped_quantity "
                "FROM order_lines WHERE order_id = %s ORDER BY position",
                (order_id,),
            )
            return self._hydrate(row, cur.fetchall())

    def insert(self, order: Order) -> None:
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

    @staticmethod
    def _hydrate(row: Sequence[Any], line_rows: Sequence[Sequence[Any]]) -> Order:
        order_id, merchant_id, customer_id, currency, status, created_at, version = row
        lines = tuple(
            OrderLine(sku, qty, Money(Decimal(price), currency), res_id, shipped)
            for sku, qty, price, res_id, shipped in line_rows
        )
        return Order(order_id, merchant_id, customer_id, currency, OrderStatus(status), lines, created_at, version)
