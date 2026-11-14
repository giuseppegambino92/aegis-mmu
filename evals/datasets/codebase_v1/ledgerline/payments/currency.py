"""Money arithmetic, ISO 4217 minor units and FX conversion. Never use floats here."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Iterable

# ISO 4217 exponents for every currency we can charge or settle in.
MINOR_UNITS = {"USD": 2, "EUR": 2, "GBP": 2, "CHF": 2, "SEK": 2, "JPY": 0, "KRW": 0, "KWD": 3}


class CurrencyMismatch(ValueError):
    pass


class StaleRateError(RuntimeError):
    pass


def quantum_for(currency: str) -> Decimal:
    """Smallest representable amount for ``currency``, e.g. ``Decimal('0.01')``."""
    return Decimal(1).scaleb(-MINOR_UNITS[currency.upper()])


@dataclass(frozen=True, slots=True)
class Money:
    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", Decimal(str(self.amount)))
        object.__setattr__(self, "currency", self.currency.upper())

    @classmethod
    def zero(cls, currency: str) -> Money:
        return cls(Decimal(0), currency)

    @classmethod
    def from_minor(cls, minor: int, currency: str) -> Money:
        return cls(Decimal(minor).scaleb(-MINOR_UNITS[currency.upper()]), currency)

    def to_minor(self) -> int:
        return int(self.quantize().amount.scaleb(MINOR_UNITS[self.currency]))

    def quantize(self, rounding: str = ROUND_HALF_EVEN) -> Money:
        return Money(self.amount.quantize(quantum_for(self.currency), rounding=rounding), self.currency)

    def is_zero(self) -> bool:
        return self.amount == 0

    def is_negative(self) -> bool:
        return self.amount < 0

    def __add__(self, other: Money) -> Money:
        self._require_same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._require_same_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: Decimal | int) -> Money:
        return Money(self.amount * Decimal(factor), self.currency)

    def _require_same_currency(self, other: Money) -> None:
        if other.currency != self.currency:
            raise CurrencyMismatch(f"{self.currency} vs {other.currency}")


@dataclass(frozen=True)
class FxQuote:
    base: str
    quote: str
    rate: Decimal
    as_of: datetime


class FxRateTable:
    """Snapshot of mid-market rates from the treasury feed.

    Usually only one direction of each pair is published; the inverse is
    derived on demand.
    """

    def __init__(self, max_age: timedelta = timedelta(minutes=15)) -> None:
        self._quotes: dict[tuple[str, str], FxQuote] = {}
        self._max_age = max_age

    def load(self, quotes: Iterable[FxQuote]) -> None:
        for quote in quotes:
            self._quotes[(quote.base.upper(), quote.quote.upper())] = quote

    def rate_for(self, base: str, quote: str, now: datetime | None = None) -> Decimal:
        base, quote = base.upper(), quote.upper()
        if base == quote:
            return Decimal(1)
        now = now or datetime.now(timezone.utc)
        if (direct := self._quotes.get((base, quote))) is not None:
            self._ensure_fresh(direct, now)
            return direct.rate
        if (inverse := self._quotes.get((quote, base))) is not None:
            self._ensure_fresh(inverse, now)
            return Decimal(1) / inverse.rate
        raise LookupError(f"no FX quote for {base}/{quote}")

    def _ensure_fresh(self, quote: FxQuote, now: datetime) -> None:
        if now - quote.as_of > self._max_age:
            raise StaleRateError(f"{quote.base}/{quote.quote} quote from {quote.as_of:%H:%M} is stale")


def convert(money: Money, target: str, table: FxRateTable, rounding: str = ROUND_HALF_EVEN) -> Money:
    """Convert ``money`` into ``target``, rounding once to the target's minor unit."""
    target = target.upper()
    if target == money.currency:
        return money
    raw = money.amount * table.rate_for(money.currency, target)
    return Money(raw.quantize(quantum_for(money.currency), rounding=rounding), target)
