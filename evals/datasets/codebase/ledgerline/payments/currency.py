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
import decimal
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

# ISO 4217 exponents for every currency we can charge or settle in.
MINOR_UNITS = {"USD": 2, "EUR": 2, "GBP": 2, "CHF": 2, "SEK": 2, "JPY": 0, "KRW": 0, "KWD": 3}

# Currencies the treasury feed quotes against; used to triangulate cross rates
# when neither direction of a pair is published.
PIVOT_CURRENCY = "USD"

# Quotes stamped further than this in the future come from a feed host with a
# broken clock; accepting them would make the pair look fresh forever.
_FUTURE_QUOTE_TOLERANCE = timedelta(seconds=30)

_ROUNDING_MODES = frozenset(
    getattr(decimal, name) for name in dir(decimal) if name.startswith("ROUND_")
)

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
    """Smallest representable amount for ``currency``, e.g. ``Decimal('0.01')``.

    The exponent table is deliberately closed: an unknown code is far more
    likely to be a typo or an unsanitised client value than a currency we
    forgot, so we raise instead of defaulting to two decimals.
    """
    if not isinstance(currency, str):
        raise UnknownCurrency(f"currency code must be a string, got {type(currency).__name__}")
    code = currency.strip().upper()
    if not _CURRENCY_CODE.match(code):
        raise UnknownCurrency(f"malformed currency code {currency!r}")
    try:
        exponent = MINOR_UNITS[code]
    except KeyError:
        raise UnknownCurrency(f"no ISO 4217 exponent configured for {code!r}") from None
    # scaleb keeps the result exact (Decimal('1E-2')), unlike 10 ** -exponent
    # which would go through a float for negative exponents.
    return Decimal(1).scaleb(-exponent)


def is_supported(currency: str) -> bool:
    """True if we can charge, refund or settle in ``currency``.

    Accepts lower-case and padded input because the value usually comes
    straight from a merchant's configuration form; anything that is not a
    string (``None`` from an unset column, for instance) is simply unsupported
    rather than an error, so callers can use this as a guard.
    """
    if not isinstance(currency, str):
        return False
    code = currency.strip().upper()
    if not _CURRENCY_CODE.match(code):
        return False
    return code in MINOR_UNITS


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
        """A zero amount in ``currency``, already at the currency's precision.

        Used as the starting value for every running total in the payments
        code, so it validates the code eagerly: a typo here would otherwise
        only surface much later as a confusing ``CurrencyMismatch``.
        """
        code = str(currency).strip().upper()
        if not is_supported(code):
            # Still constructible (Money only checks the shape of the code),
            # but worth a trace: nothing downstream can settle this amount.
            log.warning("zero amount created in unsupported currency %r", currency)
        return cls(Decimal(0), code)

    @classmethod
    def from_minor(cls, minor: int, currency: str) -> Money:
        """Build an amount from integer minor units (cents, yen, fils).

        This is the only constructor PSP webhooks and the ledger use, so it is
        strict: bools are ints in Python but ``True`` cents is always a bug.
        """
        if isinstance(minor, bool) or not isinstance(minor, int):
            raise InvalidAmount(f"minor units must be an int, got {type(minor).__name__}")
        code = str(currency).strip().upper()
        try:
            exponent = MINOR_UNITS[code]
        except KeyError:
            raise UnknownCurrency(f"no ISO 4217 exponent configured for {code!r}") from None
        return cls(Decimal(minor).scaleb(-exponent), code)

    @classmethod
    def parse(cls, text: str, currency: str) -> Money:
        """Parse a decimal string such as ``"12.50"`` without losing precision.

        Amounts with more decimals than the currency allows are rejected rather
        than rounded: a client sending ``"1.005"`` USD has a bug we want to see.
        """
        if not isinstance(text, str):
            raise InvalidAmount(f"expected amount text, got {type(text).__name__}")
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
        """Integer minor units, rounding half-even first if the amount is unquantized.

        Intermediate results (a prorated refund share, an FX product) can carry
        more digits than the currency allows; callers that need a specific
        rounding mode must quantize themselves before asking for minor units.
        """
        exponent = MINOR_UNITS.get(self.currency)
        if exponent is None:
            raise UnknownCurrency(f"no ISO 4217 exponent configured for {self.currency!r}")
        quantized = self.quantize().amount
        minor = quantized.scaleb(exponent)
        # After quantizing this is integral by construction; int() would silently
        # truncate if that invariant were ever broken, so check it explicitly.
        if minor != minor.to_integral_value():
            raise InvalidAmount(f"{self} does not map onto whole minor units")
        return int(minor)

    def quantize(self, rounding: str = ROUND_HALF_EVEN) -> Money:
        """Round to the currency's minor unit with ``rounding`` (half-even by default).

        Banker's rounding is the default because it does not bias totals when
        thousands of prorated amounts are summed in settlement reports.
        """
        if rounding not in _ROUNDING_MODES:
            raise ValueError(f"unsupported rounding mode {rounding!r}")
        quantum = quantum_for(self.currency)
        return Money(self.amount.quantize(quantum, rounding=rounding), self.currency)

    def is_zero(self) -> bool:
        """True for zero, including negative zero produced by subtraction."""
        return self.amount == 0

    def is_negative(self) -> bool:
        """True if strictly below zero; ``-0.00`` is not negative."""
        return self.amount < 0

    def is_positive(self) -> bool:
        """True if strictly above zero; zero amounts are neither sign."""
        return self.amount > 0

    def __add__(self, other: Money) -> Money:
        """Exact addition; no rounding happens until someone calls :meth:`quantize`.

        ``sum()`` starts from the int ``0``; we refuse that explicitly so
        callers use :func:`sum_money` and get a currency-aware zero.
        """
        if isinstance(other, int) and not isinstance(other, bool) and other == 0:
            raise TypeError("use sum_money() instead of sum() for Money values")
        self._require_same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        """Exact subtraction in the same currency.

        The result may be negative; ledger code that cannot accept negative
        amounts checks :meth:`is_negative` itself rather than having this
        operator clamp silently.
        """
        if not isinstance(other, Money):
            raise TypeError(f"cannot subtract {type(other).__name__} from Money")
        self._require_same_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: Decimal | int) -> Money:
        """Scale by a Decimal or int (quantities, rates, proration shares).

        The product is left unquantized: multiplying by a share like 1/3 and
        rounding immediately would compound rounding error across lines.
        """
        if isinstance(factor, float):
            raise TypeError("multiply Money by Decimal or int, never float")
        if isinstance(factor, bool):
            raise TypeError("multiplying Money by a bool is almost certainly a bug")
        if not isinstance(factor, (Decimal, int)):
            raise TypeError(f"cannot multiply Money by {type(factor).__name__}")
        scaled = Decimal(factor)
        if not scaled.is_finite():
            raise InvalidAmount(f"scale factor must be finite, got {scaled}")
        return Money(self.amount * scaled, self.currency)

    def __neg__(self) -> Money:
        """Flip the sign, keeping the currency; used for display of debits."""
        return Money(-self.amount, self.currency)

    def __lt__(self, other: Money) -> bool:
        """Compare amounts; comparing different currencies is an error, not False."""
        self._require_same_currency(other)
        return self.amount < other.amount

    def __le__(self, other: Money) -> bool:
        """Compare amounts; comparing different currencies is an error, not False."""
        self._require_same_currency(other)
        return self.amount <= other.amount

    def min(self, other: Money) -> Money:
        """Return the smaller of the two amounts (``self`` on ties).

        Used to cap refunds and payouts at what is available. ``None`` is
        accepted for "no cap" so callers do not have to special-case it.
        """
        if other is None:
            return self
        self._require_same_currency(other)
        if self <= other:
            return self
        log.debug("capping %s at %s", self, other)
        return other

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
        order = sorted(range(len(weights)), key=lambda i: (exact[i] - floors[i], i))
        for i in order[:leftover]:
            floors[i] += 1
        return [Money.from_minor(minor, self.currency) for minor in floors]

    def _require_same_currency(self, other: Money) -> None:
        """Guard for every binary operation: same type, same currency, or raise.

        Silently converting here would hide the real problem (an order line
        imported in the wrong currency, a refund requested against the wrong
        charge), so mismatches are always an exception and never a rate lookup.
        """
        if not isinstance(other, Money):
            raise TypeError(f"expected Money, got {type(other).__name__}")
        if other.currency != self.currency:
            log.debug("currency mismatch: %s vs %s", self, other)
            raise CurrencyMismatch(f"{self.currency} vs {other.currency}")

    def __str__(self) -> str:
        """Log-friendly ``"12.50 EUR"``; customer-facing formatting lives in notifications.

        The amount is printed exactly as stored (unquantized intermediates
        included) because ``__str__`` runs inside log calls and must never be
        the thing that rounds, raises or otherwise hides what was computed.
        """
        return f"{self.amount} {self.currency}"


def sum_money(items: Iterable[Money], currency: str) -> Money:
    """Sum ``items``; an empty iterable yields zero in ``currency``.

    Every item must already be in ``currency``; mixed baskets have to be
    converted first. The error names the offending position because these
    lists usually come from order lines and support needs to find the row.
    """
    total = Money.zero(currency)
    for index, item in enumerate(items):
        if not isinstance(item, Money):
            raise TypeError(f"item {index} is {type(item).__name__}, expected Money")
        if item.currency != currency:
            raise CurrencyMismatch(
                f"item {index} is in {item.currency}, expected {total.currency}"
            )
        total = total + item
    return total


@dataclass(frozen=True)
class FxQuote:
    base: str
    quote: str
    rate: Decimal
    as_of: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.rate, Decimal):
            # The feed parser hands us strings; floats would be a bug upstream.
            if isinstance(self.rate, float):
                raise TypeError(f"FX rate for {self.base}/{self.quote} must not be a float")
            object.__setattr__(self, "rate", Decimal(str(self.rate)))
        if not self.rate.is_finite() or self.rate <= 0:
            raise ValueError(f"FX rate for {self.base}/{self.quote} must be positive, got {self.rate}")
        if self.as_of.tzinfo is None:
            raise ValueError("FX quote timestamps must be timezone-aware")
        if not (_CURRENCY_CODE.match(self.base.upper()) and _CURRENCY_CODE.match(self.quote.upper())):
            raise UnknownCurrency(f"malformed FX pair {self.base!r}/{self.quote!r}")


class FxRateTable:
    """Snapshot of mid-market rates from the treasury feed.

    Usually only one direction of each pair is published; the inverse is
    derived on demand. Every lookup checks the quote's age against
    ``max_age`` so that a stalled feed fails loudly instead of paying out at
    an old rate.
    """

    def __init__(self, max_age: timedelta = timedelta(minutes=15)) -> None:
        if max_age <= timedelta(0):
            raise ValueError(f"max_age must be positive, got {max_age}")
        if max_age > timedelta(hours=24):
            # Treasury only guarantees intraday rates; a day-old mid-market rate
            # is not something we are allowed to pay out at.
            log.warning("FX max_age of %s exceeds the 24h treasury guarantee", max_age)
        self._quotes: dict[tuple[str, str], FxQuote] = {}
        self._max_age = max_age
        self._loaded_at: datetime | None = None

    def load(self, quotes: Iterable[FxQuote]) -> int:
        """Merge quotes into the table, keeping the newest per pair. Returns the number applied.

        The feed pushes a full batch every minute; most pairs are unchanged
        between batches, so the return value is mainly useful for the
        ``fx_quotes_applied`` metric.
        """
        applied = 0
        replayed = 0
        now = datetime.now(timezone.utc)
        for quote in quotes:
            key = (quote.base.upper(), quote.quote.upper())
            if key[0] == key[1]:
                log.warning("ignoring identity FX quote %s/%s", *key)
                continue
            if quote.as_of - now > _FUTURE_QUOTE_TOLERANCE:
                log.warning("ignoring FX quote %s/%s stamped in the future (%s)", *key, quote.as_of)
                continue
            existing = self._quotes.get(key)
            if existing is not None and quote.as_of >= existing.as_of:
                # The feed occasionally replays a batch; never move a pair backwards in time.
                replayed += 1
                continue
            self._quotes[key] = quote
            applied += 1
        self._loaded_at = now
        log.debug(
            "loaded %d FX quotes (%d replayed, %d pairs total)", applied, replayed, len(self._quotes)
        )
        return applied

    def pairs(self, currency: str | None = None) -> list[tuple[str, str]]:
        """Published pairs, sorted; optionally only those involving ``currency``.

        Derived inverses and triangulated crosses are not listed: this is what
        the treasury feed actually sent, which is what ops dashboards show.
        """
        if currency is None:
            return sorted(self._quotes)
        code = currency.strip().upper()
        if not _CURRENCY_CODE.match(code):
            raise UnknownCurrency(f"malformed currency code {currency!r}")
        matching = [pair for pair in self._quotes if code in pair]
        if not matching:
            log.debug("no published FX pairs involve %s", code)
        return sorted(matching)

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
        """Raise :class:`StaleRateError` if ``quote`` is older than ``max_age``.

        ``now`` must be timezone-aware; comparing a naive ``now`` against the
        aware ``as_of`` would raise a bare TypeError deep inside a payout.
        """
        if now.tzinfo is None:
            raise ValueError("freshness checks need a timezone-aware 'now'")
        age = now - quote.as_of
        if age.seconds > self._max_age.seconds:
            log.warning(
                "FX quote %s/%s is %ss old (max %ss)",
                quote.base, quote.quote, int(age.total_seconds()), int(self._max_age.total_seconds()),
            )
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
