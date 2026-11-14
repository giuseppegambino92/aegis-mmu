"""Double-entry ledger for merchant balances.

Accounts are namespaced per merchant and currency, e.g. ``m_8812:EUR:payable``.

Conventions:

* ``clearing`` (asset) holds funds captured from shoppers that have not yet
  been paid out.
* ``payable`` (liability) is what we owe the merchant.
* ``fees`` (revenue) is our cut.

Every journal entry must balance in minor units; postings are always
non-negative and the side carries the sign.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Iterator
from uuid import UUID, uuid4

from ledgerline.payments.currency import CurrencyMismatch, Money

log = logging.getLogger(__name__)


class AccountKind(str, Enum):
    ASSET = "asset"
    LIABILITY = "liability"
    REVENUE = "revenue"


class EntrySide(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"

    def opposite(self) -> EntrySide:
        return EntrySide.CREDIT if self is EntrySide.DEBIT else EntrySide.DEBIT


@dataclass(frozen=True)
class Posting:
    account: str
    side: EntrySide
    amount: Money


@dataclass(frozen=True)
class JournalEntry:
    id: UUID
    posted_at: datetime
    memo: str
    postings: tuple[Posting, ...]
    reference: str | None = None
    reverses: UUID | None = None

    def touches(self, account: str) -> bool:
        return any(posting.account == account for posting in self.postings)


@dataclass(frozen=True)
class StatementLine:
    entry_id: UUID
    posted_at: datetime
    memo: str
    reference: str | None
    change: Money
    running_balance: Money


class UnbalancedEntryError(ValueError):
    pass


class UnknownAccount(KeyError):
    pass


class InsufficientBalance(RuntimeError):
    def __init__(self, account: str, available: Money, requested: Money) -> None:
        super().__init__(f"{account}: requested {requested}, available {available}")
        self.account = account
        self.available = available
        self.requested = requested


def merchant_account_codes(merchant_id: str, currency: str) -> tuple[str, str, str]:
    """Return ``(clearing, payable, fees)`` account codes for a merchant."""
    prefix = f"{merchant_id}:{currency.upper()}"
    return f"{prefix}:clearing", f"{prefix}:payable", f"{prefix}:fees"


class Ledger:
    def __init__(self) -> None:
        self._accounts: dict[str, tuple[AccountKind, str]] = {}
        self._journal: list[JournalEntry] = []
        self._by_id: dict[UUID, JournalEntry] = {}
        # Posting is cheap, but reversal and payout checks read-then-write, so
        # they must not interleave with another writer.
        self._lock = threading.RLock()

    def open_merchant(self, merchant_id: str, currency: str) -> None:
        if not merchant_id:
            raise ValueError("merchant_id is required")
        clearing, payable, fees = merchant_account_codes(merchant_id, currency)
        with self._lock:
            if payable in self._accounts:
                # Onboarding retries call this again; opening twice is a no-op.
                log.debug("accounts for %s/%s already open", merchant_id, currency.upper())
                return
            self._accounts[clearing] = (AccountKind.ASSET, currency.upper())
            self._accounts[payable] = (AccountKind.LIABILITY, currency.upper())
            self._accounts[fees] = (AccountKind.REVENUE, currency.upper())
        log.info("opened ledger accounts for merchant %s in %s", merchant_id, currency.upper())

    def has_account(self, code: str) -> bool:
        return code in self._accounts

    def post(
        self, memo: str, postings: Iterable[Posting], *, reference: str | None = None,
        posted_at: datetime | None = None,
    ) -> JournalEntry:
        frozen = tuple(postings)
        if posted_at is not None and posted_at.tzinfo is None:
            raise ValueError("posted_at must be timezone-aware")
        with self._lock:
            self._validate(frozen)
            entry = JournalEntry(uuid4(), posted_at or datetime.now(timezone.utc), memo, frozen, reference)
            self._journal.append(entry)
            self._by_id[entry.id] = entry
        log.debug("posted %s (%s) with %d postings", entry.id, memo, len(frozen))
        return entry

    def _validate(self, postings: tuple[Posting, ...]) -> None:
        if len(postings) < 2:
            raise UnbalancedEntryError("a journal entry needs at least two postings")
        balance = 0
        for posting in postings:
            if posting.account not in self._accounts:
                raise UnknownAccount(posting.account)
            _, currency = self._accounts[posting.account]
            if currency != posting.amount.currency:
                raise CurrencyMismatch(f"{posting.account} is {currency}")
            if posting.amount.is_negative():
                raise ValueError("postings must be non-negative; flip the side instead")
            minor = posting.amount.to_minor()
            balance += minor if posting.side is EntrySide.DEBIT else -minor
        if balance != 0:
            raise UnbalancedEntryError(f"entry is off by {balance} minor units")

    def post_capture(self, merchant_id: str, order_id: str, gross: Money, fee: Money) -> JournalEntry:
        if fee.is_negative() or gross.is_negative():
            raise ValueError("capture amounts must be non-negative")
        if gross < fee:
            # A fee larger than the charge would leave a negative payable credit;
            # pricing should have rejected this upstream.
            raise ValueError(f"fee {fee} exceeds gross {gross} on order {order_id}")
        clearing, payable, fees = merchant_account_codes(merchant_id, gross.currency)
        return self.post(
            f"capture for order {order_id}",
            [
                Posting(clearing, EntrySide.DEBIT, gross),
                Posting(payable, EntrySide.CREDIT, gross - fee),
                Posting(fees, EntrySide.CREDIT, fee),
            ],
            reference=order_id,
        )

    def post_refund(self, merchant_id: str, order_id: str, amount: Money, refund_id: UUID) -> JournalEntry:
        """Reverse the merchant's share of a captured payment."""
        if not amount.is_positive():
            raise ValueError(f"refund amount must be positive, got {amount}")
        clearing, payable, _ = merchant_account_codes(merchant_id, amount.currency)
        entry = self.post(
            f"refund {refund_id} for order {order_id}",
            [
                Posting(payable, EntrySide.CREDIT, amount),
                Posting(clearing, EntrySide.DEBIT, amount),
            ],
            reference=order_id,
        )
        log.info("ledger: refund %s of %s posted for merchant %s", refund_id, amount, merchant_id)
        return entry

    def post_payout(self, merchant_id: str, amount: Money, payout_id: str) -> JournalEntry:
        """Record a bank payout to the merchant: money leaves clearing and the debt to them shrinks.

        The payout must not exceed what we currently owe the merchant. The
        entry is referenced by ``payout_id`` so the bank-statement reconciler
        can find it.
        """
        if not amount.is_positive():
            raise ValueError(f"payout amount must be positive, got {amount}")
        if not payout_id:
            raise ValueError("payout_id is required")
        clearing, payable, _ = merchant_account_codes(merchant_id, amount.currency)
        with self._lock:
            owed = self._current_balance(payable)
            if owed < amount:
                raise InsufficientBalance(payable, owed, amount)
            entry = self.post(
                f"payout {payout_id} to merchant {merchant_id}",
                [
                    Posting(payable, EntrySide.DEBIT, amount),
                    Posting(clearing, EntrySide.CREDIT, amount),
                ],
            )
        log.info("ledger: payout %s of %s for merchant %s", payout_id, amount, merchant_id)
        return entry

    def reverse(self, entry_id: UUID, *, memo: str | None = None) -> JournalEntry:
        """Post the mirror image of an existing entry. Entries are never edited or deleted."""
        with self._lock:
            original = self._by_id.get(entry_id)
            if original is None:
                raise LookupError(f"no journal entry {entry_id}")
            if any(entry.reverses == entry_id for entry in self._journal):
                raise ValueError(f"entry {entry_id} has already been reversed")
            mirrored = tuple(
                Posting(p.account, p.side.opposite(), p.amount) for p in original.postings
            )
            self._validate(mirrored)
            entry = JournalEntry(
                uuid4(),
                datetime.now(timezone.utc),
                memo or f"reversal of {entry_id}",
                mirrored,
                original.reference,
                reverses=entry_id,
            )
            self._journal.append(entry)
            self._by_id[entry.id] = entry
        log.warning("ledger: reversed entry %s (%s)", entry_id, original.memo)
        return entry

    def entries_for_reference(self, reference: str) -> list[JournalEntry]:
        return [entry for entry in self._journal if entry.reference == reference]

    def _normal_side(self, code: str) -> tuple[EntrySide, str]:
        try:
            kind, currency = self._accounts[code]
        except KeyError:
            raise UnknownAccount(code) from None
        normal = EntrySide.DEBIT if kind is AccountKind.ASSET else EntrySide.CREDIT
        return normal, currency

    def _current_balance(self, code: str) -> Money:
        normal, currency = self._normal_side(code)
        total = 0
        for entry in self._journal:
            for posting in entry.postings:
                if posting.account == code:
                    minor = posting.amount.to_minor()
                    total += minor if posting.side is normal else -minor
        return Money.from_minor(total, currency)

    def balance_as_of(self, code: str, cutoff: datetime) -> Money:
        """Balance on the account's normal side, including entries posted at or before ``cutoff``."""
        kind, currency = self._accounts[code]
        normal = EntrySide.DEBIT if kind is AccountKind.ASSET else EntrySide.CREDIT
        total = 0
        for entry in self._journal:
            if entry.posted_at >= cutoff:
                continue
            for posting in entry.postings:
                if posting.account == code:
                    minor = posting.amount.to_minor()
                    total += minor if posting.side is normal else -minor
        return Money.from_minor(total, currency)

    def _entries_between(self, start: datetime, end: datetime) -> Iterator[JournalEntry]:
        for entry in sorted(self._journal, key=lambda e: e.posted_at):
            if start <= entry.posted_at < end:
                yield entry

    def statement(self, code: str, start: datetime, end: datetime) -> list[StatementLine]:
        """Line-by-line movements on ``code`` in ``[start, end)`` with a running balance.

        The opening balance is everything strictly before ``start``; that is
        what the monthly merchant statement prints as "brought forward".
        """
        if end <= start:
            raise ValueError("statement end must be after start")
        normal, currency = self._normal_side(code)
        opening = 0
        for entry in self._journal:
            if entry.posted_at < start:
                for posting in entry.postings:
                    if posting.account == code:
                        minor = posting.amount.to_minor()
                        opening += minor if posting.side is normal else -minor

        lines: list[StatementLine] = []
        running = opening
        for entry in self._entries_between(start, end):
            change = 0
            for posting in entry.postings:
                if posting.account == code:
                    minor = posting.amount.to_minor()
                    change += minor if posting.side is normal else -minor
            if change == 0 and not entry.touches(code):
                continue
            running += change
            lines.append(
                StatementLine(
                    entry.id,
                    entry.posted_at,
                    entry.memo,
                    entry.reference,
                    Money.from_minor(change, currency),
                    Money.from_minor(running, currency),
                )
            )
        return lines

    def trial_balance(self) -> dict[str, int]:
        """Net debit-minus-credit per currency in minor units; every value must be zero."""
        totals: dict[str, int] = {}
        for entry in self._journal:
            for posting in entry.postings:
                minor = posting.amount.to_minor()
                signed = minor if posting.side is EntrySide.DEBIT else -minor
                totals[posting.amount.currency] = totals.get(posting.amount.currency, 0) + signed
        broken = {currency: net for currency, net in totals.items() if net != 0}
        if broken:
            log.error("ledger trial balance does not net to zero: %s", broken)
        return totals
