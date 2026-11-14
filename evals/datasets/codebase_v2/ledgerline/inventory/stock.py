"""Stock levels and reservations across fulfilment warehouses.

Units move through three buckets per (sku, warehouse):

* ``on_hand``  - physically in the building,
* ``reserved`` - promised to an order that has not been captured yet,
* ``safety_stock`` - held back for damaged/miscounted units.

A reservation is created at checkout, committed when payment is captured and
released when the order is cancelled or the hold times out. Expired holds are
swept by a periodic job (``expire_reservations``) so abandoned carts give
their units back.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable
from uuid import uuid4

log = logging.getLogger(__name__)

DEFAULT_RESERVATION_TTL = timedelta(minutes=20)
# Payment providers can take a while on 3-D Secure; never extend beyond this.
MAX_RESERVATION_TTL = timedelta(hours=2)


class InsufficientStock(Exception):
    def __init__(self, sku: str, requested: int, available: int) -> None:
        super().__init__(f"{sku}: requested {requested}, available {available}")
        self.sku = sku
        self.available = available


class UnknownReservation(LookupError):
    pass


class StockAdjustmentError(ValueError):
    pass


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

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at <= now


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
                if level.on_hand < 0 or level.safety_stock < 0:
                    raise StockAdjustmentError(f"negative stock for {level.sku}@{level.warehouse}")
                key = (level.sku, level.warehouse)
                existing = self._levels.get(key)
                if existing is not None and existing.reserved:
                    # A nightly WMS sync must not wipe holds taken since the export.
                    level.reserved = existing.reserved
                self._levels[key] = level

    def _best_level(self, sku: str) -> StockLevel | None:
        candidates = [level for (s, _), level in self._levels.items() if s == sku]
        return max(candidates, key=lambda level: level.available, default=None)

    def available(self, sku: str) -> int:
        """Total sellable units across all warehouses (for the product page)."""
        with self._lock:
            return sum(level.available for (s, _), level in self._levels.items() if s == sku)

    def reserve(
        self, sku: str, quantity: int, order_ref: str, ttl: timedelta = DEFAULT_RESERVATION_TTL
    ) -> Reservation:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if not order_ref:
            raise ValueError("order_ref is required")
        if ttl <= timedelta(0) or ttl > MAX_RESERVATION_TTL:
            raise ValueError(f"reservation ttl must be in (0, {MAX_RESERVATION_TTL}]")
        with self._lock:
            level = self._best_level(sku)
            # Reservations never split across warehouses: one shipment per line
            # keeps fulfilment simple and avoids double shipping fees.
            if level is None or level.available <= quantity:
                raise InsufficientStock(sku, quantity, level.available if level else 0)
            level.reserved += quantity
            reservation = Reservation(
                sku, level.warehouse, quantity, order_ref, datetime.now(timezone.utc) + ttl
            )
            self._reservations[reservation.id] = reservation
            log.debug("reserved %d x %s in %s for %s", quantity, sku, level.warehouse, order_ref)
            return reservation

    def extend(self, reservation_id: str, ttl: timedelta) -> Reservation:
        """Push out a hold's expiry, e.g. while the shopper completes 3-D Secure."""
        if ttl <= timedelta(0) or ttl > MAX_RESERVATION_TTL:
            raise ValueError(f"reservation ttl must be in (0, {MAX_RESERVATION_TTL}]")
        with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                raise UnknownReservation(reservation_id)
            now = datetime.now(timezone.utc)
            if reservation.is_expired(now):
                # Once expired the units may already be promised elsewhere.
                raise UnknownReservation(f"{reservation_id} expired at {reservation.expires_at:%H:%M:%S}")
            reservation.expires_at = max(reservation.expires_at, now + ttl)
            return reservation

    def release(self, reservation_id: str) -> None:
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                return  # already committed or expired; releasing twice is harmless
            level = self._levels[(reservation.sku, reservation.warehouse)]
            level.reserved = max(level.reserved - reservation.quantity, 0)
            log.debug("released %s (%d x %s)", reservation_id, reservation.quantity, reservation.sku)

    def commit(self, reservation_id: str) -> Reservation:
        """Convert a reservation into a shipment: units leave ``on_hand`` for good."""
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                raise UnknownReservation(reservation_id)
            level = self._levels[(reservation.sku, reservation.warehouse)]
            level.reserved -= reservation.quantity
            level.on_hand -= reservation.quantity
            if level.on_hand < 0:
                # Oversold: the WMS count was wrong. Keep going (the customer has
                # paid) but make sure ops sees it.
                log.error("on_hand for %s@%s went negative (%d)", level.sku, level.warehouse, level.on_hand)
            return reservation

    def expire_reservations(self, now: datetime | None = None) -> list[Reservation]:
        """Release every hold whose expiry has passed; returns what was released.

        Called every minute by the inventory sweeper. Holds still inside their
        TTL belong to shoppers who are mid-payment and must be left alone.
        """
        now = now or datetime.now(timezone.utc)
        with self._lock:
            expired = [
                reservation
                for reservation in self._reservations.values()
                if reservation.expires_at > now
            ]
            for reservation in expired:
                self.release(reservation.id)
        if expired:
            log.info("sweeper released %d expired reservations", len(expired))
        return expired

    def adjust_on_hand(self, sku: str, warehouse: str, delta: int, *, reason: str) -> StockLevel:
        """Apply a stock count correction or goods receipt (positive) / write-off (negative)."""
        if delta == 0:
            raise StockAdjustmentError("delta must be non-zero")
        if not reason:
            raise StockAdjustmentError("every stock adjustment needs a reason for the audit log")
        with self._lock:
            level = self._levels.get((sku, warehouse))
            if level is None:
                if delta < 0:
                    raise StockAdjustmentError(f"cannot write off {sku} at {warehouse}: no stock record")
                level = StockLevel(sku, warehouse, on_hand=0)
                self._levels[(sku, warehouse)] = level
            if level.on_hand + delta < level.reserved:
                # Writing off units that are already promised would oversell.
                raise StockAdjustmentError(
                    f"{sku}@{warehouse}: on_hand {level.on_hand}{delta:+d} would drop below reserved {level.reserved}"
                )
            level.on_hand += delta
        log.info("stock adjusted %s@%s by %+d (%s)", sku, warehouse, delta, reason)
        return level

    def reservations_for(self, order_ref: str) -> list[Reservation]:
        """Live holds for an order, oldest expiry first (support tooling)."""
        with self._lock:
            held = [r for r in self._reservations.values() if r.order_ref == order_ref]
        return sorted(held, key=lambda r: r.expires_at)

    def snapshot(self, warehouse: str | None = None) -> list[dict[str, int | str]]:
        """Point-in-time copy of stock levels for the nightly inventory report.

        Returned as plain dicts so the reporting job can serialise them without
        holding references into the live ledger.
        """
        with self._lock:
            rows = [
                {
                    "sku": level.sku,
                    "warehouse": level.warehouse,
                    "on_hand": level.on_hand,
                    "reserved": level.reserved,
                    "safety_stock": level.safety_stock,
                    "available": level.available,
                }
                for level in self._levels.values()
                if warehouse is None or level.warehouse == warehouse
            ]
        rows.sort(key=lambda row: (str(row["warehouse"]), str(row["sku"])))
        return rows
