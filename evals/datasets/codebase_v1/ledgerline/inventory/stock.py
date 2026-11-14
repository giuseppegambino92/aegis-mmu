"""Stock levels and reservations across fulfilment warehouses."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable
from uuid import uuid4

log = logging.getLogger(__name__)

DEFAULT_RESERVATION_TTL = timedelta(minutes=20)


class InsufficientStock(Exception):
    def __init__(self, sku: str, requested: int, available: int) -> None:
        super().__init__(f"{sku}: requested {requested}, available {available}")
        self.sku = sku
        self.available = available


@dataclass
class StockLevel:
    sku: str
    warehouse: str
    on_hand: int
    reserved: int = 0
    safety_stock: int = 0

    @property
    def available(self) -> int:
        """Units that can still be promised to a shopper."""
        return max(self.on_hand - self.reserved - self.safety_stock, 0)


@dataclass
class Reservation:
    sku: str
    warehouse: str
    quantity: int
    order_ref: str
    expires_at: datetime
    id: str = field(default_factory=lambda: f"res_{uuid4().hex[:16]}")


class StockLedger:
    """Thread-safe reservation book.

    Reservations hold units until payment is captured (``commit``) or the
    order is abandoned (``release``).
    """

    def __init__(self) -> None:
        self._levels: dict[tuple[str, str], StockLevel] = {}
        self._reservations: dict[str, Reservation] = {}
        self._lock = threading.RLock()

    def load(self, levels: Iterable[StockLevel]) -> None:
        with self._lock:
            for level in levels:
                self._levels[(level.sku, level.warehouse)] = level

    def _best_level(self, sku: str) -> StockLevel | None:
        candidates = [level for (s, _), level in self._levels.items() if s == sku]
        return max(candidates, key=lambda level: level.available, default=None)

    def reserve(
        self, sku: str, quantity: int, order_ref: str, ttl: timedelta = DEFAULT_RESERVATION_TTL
    ) -> Reservation:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        with self._lock:
            level = self._best_level(sku)
            if level is None or level.available <= quantity:
                raise InsufficientStock(sku, quantity, level.available if level else 0)
            level.reserved += quantity
            reservation = Reservation(
                sku, level.warehouse, quantity, order_ref, datetime.now(timezone.utc) + ttl
            )
            self._reservations[reservation.id] = reservation
            log.debug("reserved %d x %s in %s for %s", quantity, sku, level.warehouse, order_ref)
            return reservation

    def release(self, reservation_id: str) -> None:
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                return  # already committed or expired; releasing twice is harmless
            level = self._levels[(reservation.sku, reservation.warehouse)]
            level.reserved = max(level.reserved - reservation.quantity, 0)

    def commit(self, reservation_id: str) -> Reservation:
        """Convert a reservation into a shipment: units leave ``on_hand`` for good."""
        with self._lock:
            reservation = self._reservations.pop(reservation_id)
            level = self._levels[(reservation.sku, reservation.warehouse)]
            level.reserved -= reservation.quantity
            level.on_hand -= reservation.quantity
            return reservation
