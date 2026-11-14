"""Money arithmetic, ISO 4217 minor units and FX conversion. Never use floats here.

Every amount that crosses a service boundary is a :class:`Money`. Arithmetic
between two ``Money`` values requires the same currency; conversion between
currencies goes through :func:`convert` and an :class:`FxRateTable` snapshot
loaded from the treasury feed.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

# ISO 4217 exponents for every currency we can charge or settle in.
MINOR_UNITS = {"USD": 2, "EUR": 2, "GBP": 2, "CHF": 2, "SEK": 2, "JPY": 0, "KRW": 0, "KWD": 3}

# Currencies the treasury feed quotes against; used to triangulate cross rates
# when neither direction of a pair is published.
PIVOT_CURRENCY = "USD"

_CURRENCY_CODE = re.compile(r"^[A-Z]{3}$")
_AMOUNT_TEXT = re.compile(r"^-?\d+(\.\d+)?$")


class CurrencyMismatch(ValueError):
    pass


class UnknownCurrency(ValueError):
    """Raised for a syntactically valid code we do not settle in."""


class StaleRateError(RuntimeError):
    pass


class InvalidAmount(ValueError):
    """Raised when text or a number cannot be represented in the currency's precision."""


def quantum_for(currency: str) -> Decimal:
    """Smallest representable amount for ``currency``, e.g. ``Decimal('0.01')``."""
    code = currency.upper()
    try:
        exponent = MINOR_UNITS[code]
    except KeyError:
        raise UnknownCurrency(f"no ISO 4217 exponent configured for {code!r}") from None
    return Decimal(1).scaleb(-exponent)


def is_supported(currency: str) -> bool:
    return currency.upper() in MINOR_UNITS


@dataclass(frozen=True, slots=True)
class Money:
    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        amount = self.amount
        if isinstance(amount, float):
            # Floats sneak in from JSON payloads. Going through str() keeps the
            # shortest repr (0.1 -> "0.1") instead of the binary expansion.
            amount = Decimal(str(amount))
        elif not isinstance(amount, Decimal):
            try:
                amount = Decimal(str(amount))
            except InvalidOperation as exc:
                raise InvalidAmount(f"not a number: {self.amount!r}") from exc
        if not amount.is_finite():
            raise InvalidAmount(f"amount must be finite, got {amount}")
        object.__setattr__(self, "amount", amount)

        code = str(self.currency).upper()
        if not _CURRENCY_CODE.match(code):
            raise UnknownCurrency(f"malformed currency code {self.currency!r}")
        object.__setattr__(self, "currency", code)

    @classmethod
    def zero(cls, currency: str) -> Money:
        return cls(Decimal(0), currency)

    @classmethod
    def from_minor(cls, minor: int, currency: str) -> Money:
        if isinstance(minor, bool) or not isinstance(minor, int):
            raise InvalidAmount(f"minor units must be an int, got {type(minor).__name__}")
        return cls(Decimal(minor).scaleb(-MINOR_UNITS[currency.upper()]), currency)

    @classmethod
    def parse(cls, text: str, currency: str) -> Money:
        """Parse a decimal string such as ``"12.50"`` without losing precision.

        Amounts with more decimals than the currency allows are rejected rather
        than rounded: a client sending ``"1.005"`` USD has a bug we want to see.
        """
        cleaned = text.strip().replace("_", "")
        if not _AMOUNT_TEXT.match(cleaned):
            raise InvalidAmount(f"cannot parse amount {text!r}")
        value = Decimal(cleaned)
        quantum = quantum_for(currency)
        if value != value.quantize(quantum):
            raise InvalidAmount(
                f"{text!r} has more precision than {currency.upper()} allows ({quantum})"
            )
        return cls(value, currency)

    def to_minor(self) -> int:
        return int(self.quantize().amount.scaleb(MINOR_UNITS[self.currency]))

    def quantize(self, rounding: str = ROUND_HALF_EVEN) -> Money:
        return Money(self.amount.quantize(quantum_for(self.currency), rounding=rounding), self.currency)

    def is_zero(self) -> bool:
        return self.amount == 0

    def is_negative(self) -> bool:
        return self.amount < 0

    def is_positive(self) -> bool:
        return self.amount > 0

    def __add__(self, other: Money) -> Money:
        self._require_same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._require_same_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: Decimal | int) -> Money:
        if isinstance(factor, float):
            raise TypeError("multiply Money by Decimal or int, never float")
        return Money(self.amount * Decimal(factor), self.currency)

    def __neg__(self) -> Money:
        return Money(-self.amount, self.currency)

    def __lt__(self, other: Money) -> bool:
        self._require_same_currency(other)
        return self.amount < other.amount

    def __le__(self, other: Money) -> bool:
        self._require_same_currency(other)
        return self.amount <= other.amount

    def min(self, other: Money) -> Money:
        return self if self <= other else other

    def allocate(self, weights: Sequence[int | Decimal]) -> list[Money]:
        """Split this amount across ``weights`` without creating or losing minor units.

        Each share is rounded down to the currency quantum and the leftover
        minor units are handed out one at a time, starting with the share that
        lost the most to rounding (largest-remainder method). The result always
        sums exactly to ``self``.
        """
        if not weights:
            raise ValueError("cannot allocate across zero weights")
        total_weight = sum(Decimal(w) for w in weights)
        if total_weight <= 0:
            raise ValueError("weights must sum to a positive number")
        if any(Decimal(w) < 0 for w in weights):
            raise ValueError("weights must be non-negative")

        total_minor = self.to_minor()
        exact = [Decimal(total_minor) * Decimal(w) / total_weight for w in weights]
        floors = [int(share.to_integral_value(rounding="ROUND_FLOOR")) for share in exact]
        leftover = total_minor - sum(floors)

        # Order by fractional part, largest first; ties go to the earlier index
        # so the split is deterministic across processes.
        order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - floors[i]), i))
        for i in order[:leftover]:
            floors[i] += 1
        return [Money.from_minor(minor, self.currency) for minor in floors]

    def _require_same_currency(self, other: Money) -> None:
        if not isinstance(other, Money):
            raise TypeError(f"expected Money, got {type(other).__name__}")
        if other.currency != self.currency:
            raise CurrencyMismatch(f"{self.currency} vs {other.currency}")

    def __str__(self) -> str:
        return f"{self.amount} {self.currency}"


def sum_money(items: Iterable[Money], currency: str) -> Money:
    """Sum ``items``; an empty iterable yields zero in ``currency``."""
    total = Money.zero(currency)
    for item in items:
        total = total + item
    return total


@dataclass(frozen=True)
class FxQuote:
    base: str
    quote: str
    rate: Decimal
    as_of: datetime

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError(f"FX rate for {self.base}/{self.quote} must be positive, got {self.rate}")
        if self.as_of.tzinfo is None:
            raise ValueError("FX quote timestamps must be timezone-aware")


class FxRateTable:
    """Snapshot of mid-market rates from the treasury feed.

    Usually only one direction of each pair is published; the inverse is
    derived on demand. Every lookup checks the quote's age against
    ``max_age`` so that a stalled feed fails loudly instead of paying out at
    an old rate.
    """

    def __init__(self, max_age: timedelta = timedelta(minutes=15)) -> None:
        self._quotes: dict[tuple[str, str], FxQuote] = {}
        self._max_age = max_age
        self._loaded_at: datetime | None = None

    def load(self, quotes: Iterable[FxQuote]) -> int:
        """Merge quotes into the table, keeping the newest per pair. Returns the number applied."""
        applied = 0
        for quote in quotes:
            key = (quote.base.upper(), quote.quote.upper())
            if key[0] == key[1]:
                log.warning("ignoring identity FX quote %s/%s", *key)
                continue
            existing = self._quotes.get(key)
            if existing is not None and existing.as_of >= quote.as_of:
                # The feed occasionally replays a batch; never move a pair backwards in time.
                continue
            self._quotes[key] = quote
            applied += 1
        self._loaded_at = datetime.now(timezone.utc)
        log.debug("loaded %d FX quotes (%d pairs total)", applied, len(self._quotes))
        return applied

    def pairs(self) -> list[tuple[str, str]]:
        return sorted(self._quotes)

    def rate_for(self, base: str, quote: str, now: datetime | None = None) -> Decimal:
        base, quote = base.upper(), quote.upper()
        if base == quote:
            return Decimal(1)
        now = now or datetime.now(timezone.utc)
        if (direct := self._quotes.get((base, quote))) is not None:
            self._ensure_fresh(direct, now)
            return direct.rate
        if (inverse := self._quotes.get((quote, base))) is not None:
            # Derive the reverse direction from the published pair. The quote
            # object is the same one the direct lookup would have used.
            return Decimal(1) / inverse.rate
        if PIVOT_CURRENCY not in (base, quote):
            # Triangulate through the pivot, e.g. SEK/JPY = SEK/USD * USD/JPY.
            try:
                leg_one = self.rate_for(base, PIVOT_CURRENCY, now)
                leg_two = self.rate_for(PIVOT_CURRENCY, quote, now)
            except LookupError:
                pass
            else:
                return leg_one * leg_two
        raise LookupError(f"no FX quote for {base}/{quote}")

    def _ensure_fresh(self, quote: FxQuote, now: datetime) -> None:
        if now - quote.as_of > self._max_age:
            raise StaleRateError(f"{quote.base}/{quote.quote} quote from {quote.as_of:%H:%M} is stale")


def convert(money: Money, target: str, table: FxRateTable, rounding: str = ROUND_HALF_EVEN) -> Money:
    """Convert ``money`` into ``target``, rounding once to the target's minor unit."""
    target = target.upper()
    if not is_supported(target):
        raise UnknownCurrency(f"cannot convert into unsupported currency {target}")
    if target == money.currency:
        return money
    rate = table.rate_for(money.currency, target)
    raw = money.amount * rate
    converted = Money(raw.quantize(quantum_for(money.currency), rounding=rounding), target)
    log.debug("fx %s -> %s at %s = %s", money, target, rate, converted)
    return converted
