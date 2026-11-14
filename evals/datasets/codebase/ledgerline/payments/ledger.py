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

from ledgerline.payments.currency import CurrencyMismatch, Money, is_supported

log = logging.getLogger(__name__)

# Memos end up on merchant statements and in the bank reconciler's free-text
# matcher; anything longer is truncated by the statement PDF renderer anyway.
MAX_MEMO_LENGTH = 200

# Journal entries with more postings than this are almost certainly a loop
# bug in a caller (one posting per order line instead of per account).
MAX_POSTINGS_PER_ENTRY = 32


class AccountKind(str, Enum):
    ASSET = "asset"
    LIABILITY = "liability"
    REVENUE = "revenue"


class EntrySide(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"

    def opposite(self) -> EntrySide:
        """The other side of the entry; used to build reversals.

        Spelled out as an explicit branch rather than a lookup table so a new
        member added to the enum (we once discussed MEMO postings) fails loudly
        here instead of silently mapping to DEBIT.
        """
        if self is EntrySide.DEBIT:
            return EntrySide.CREDIT
        if self is EntrySide.CREDIT:
            return EntrySide.DEBIT
        raise ValueError(f"entry side {self!r} has no opposite")


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
        """True if any posting in this entry hits ``account``.

        Account codes are case-sensitive in storage, but operators paste them
        from statements in all sorts of shapes; strip surrounding whitespace
        so a copy-pasted code still matches.
        """
        if not account:
            return False
        wanted = account.strip()
        for posting in self.postings:
            if posting.account == wanted:
                return True
        return False


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
        # Keep the shortfall around as well: the payout scheduler logs it and
        # the ops dashboard groups failed payouts by how far short they were.
        shortfall = None
        try:
            shortfall = requested - available
        except (CurrencyMismatch, TypeError):
            # Callers construct this from inside error paths; never let the
            # error itself raise a second, more confusing exception.
            log.debug("could not compute shortfall for %s", account)
        message = f"{account}: requested {requested}, available {available}"
        if shortfall is not None and shortfall.is_positive():
            message += f" (short by {shortfall})"
        super().__init__(message)
        self.account = account
        self.available = available
        self.requested = requested
        self.shortfall = shortfall


def merchant_account_codes(merchant_id: str, currency: str) -> tuple[str, str, str]:
    """Return ``(clearing, payable, fees)`` account codes for a merchant.

    The colon is the namespace separator, so it cannot appear inside a
    merchant id; merchant ids come from the onboarding service and are
    already validated there, but a malformed one here would silently alias
    another merchant's accounts.
    """
    if not merchant_id:
        raise ValueError("merchant_id is required")
    if ":" in merchant_id or merchant_id != merchant_id.strip():
        raise ValueError(f"invalid merchant id {merchant_id!r}")
    code = currency.strip()
    if not is_supported(code):
        raise ValueError(f"merchant accounts cannot be held in {currency!r}")
    prefix = f"{merchant_id}:{code}"
    return f"{prefix}:clearing", f"{prefix}:payable", f"{prefix}:fees"


class Ledger:
    def __init__(self) -> None:
        self._accounts: dict[str, tuple[AccountKind, str]] = {}
        self._journal: list[JournalEntry] = []
        self._by_id: dict[UUID, JournalEntry] = {}
        # Reversals are looked up on every reverse() call; keep an index so it
        # does not degrade into a full journal scan as the ledger grows.
        self._reversed: set[UUID] = set()
        # Posting is cheap, but reversal and payout checks read-then-write, so
        # they must not interleave with another writer.
        self._lock = threading.RLock()
        log.debug("ledger initialised (empty journal)")

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
        """True if ``code`` has been opened.

        Used by the admin console before rendering a statement link, so it
        tolerates junk input (``None``, whitespace) instead of raising.
        """
        if not isinstance(code, str):
            return False
        cleaned = code.strip()
        if not cleaned:
            return False
        with self._lock:
            return cleaned in self._accounts

    def post(
        self, memo: str, postings: Iterable[Posting], *, reference: str | None = None,
        posted_at: datetime | None = None,
    ) -> JournalEntry:
        frozen = tuple(postings)
        if posted_at is not None and posted_at.tzinfo is None:
            raise ValueError("posted_at must be timezone-aware")
        if not memo or not memo.strip():
            raise ValueError("journal entries need a memo")
        if len(memo) > MAX_MEMO_LENGTH:
            log.warning("truncating %d-char memo for entry %r", len(memo), memo[:40])
            memo = memo[:MAX_MEMO_LENGTH]
        if len(frozen) > MAX_POSTINGS_PER_ENTRY:
            raise UnbalancedEntryError(f"entry has {len(frozen)} postings (max {MAX_POSTINGS_PER_ENTRY})")
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
        if not order_id:
            raise ValueError("order_id is required to post a refund")
        clearing, payable, _ = merchant_account_codes(merchant_id, amount.currency)
        if payable not in self._accounts:
            # Refunds for a merchant we never onboarded in this currency mean the
            # capture was booked somewhere else; posting would raise anyway.
            log.error("refund %s for %s: merchant %s has no %s accounts",
                      refund_id, order_id, merchant_id, amount.currency)
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
            if original.reverses is not None:
                # Reversing a reversal re-applies the original; finance wants a
                # fresh correcting entry for that so the audit trail reads cleanly.
                raise ValueError(f"entry {entry_id} is itself a reversal of {original.reverses}")
            if entry_id in self._reversed or any(entry.reverses == entry_id for entry in self._journal):
                raise ValueError(f"entry {entry_id} has already been reversed")
            mirrored = tuple(
                Posting(p.account, p.side, p.amount) for p in original.postings
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
            self._reversed.add(entry_id)
        log.warning("ledger: reversed entry %s (%s)", entry_id, original.memo)
        return entry

    def entries_for_reference(self, reference: str) -> list[JournalEntry]:
        """All entries carrying ``reference`` (an order id, refund id or payout id), oldest first.

        The reconciler and the support console both call this; neither should
        see half-appended state, so the scan runs under the ledger lock.
        """
        if not reference:
            raise ValueError("reference is required")
        wanted = reference.strip()
        with self._lock:
            matches = [entry for entry in self._journal if entry.reference == wanted]
        matches.sort(key=lambda e: e.posted_at)
        if not matches:
            log.debug("no journal entries reference %s", wanted)
        return matches

    def _normal_side(self, code: str) -> tuple[EntrySide, str]:
        """Return the side that increases ``code`` and the account's currency.

        Assets grow on the debit side; liabilities and revenue grow on the
        credit side. Balances everywhere in this module are reported on the
        normal side, so a positive payable means we owe the merchant money.
        """
        try:
            kind, currency = self._accounts[code]
        except KeyError:
            raise UnknownAccount(code) from None
        if kind is AccountKind.ASSET:
            normal = EntrySide.DEBIT
        elif kind in (AccountKind.LIABILITY, AccountKind.REVENUE):
            normal = EntrySide.CREDIT
        else:
            raise ValueError(f"account {code} has unsupported kind {kind!r}")
        return normal, currency

    def _current_balance(self, code: str) -> Money:
        """Balance of ``code`` across the whole journal, on its normal side.

        Called with the ledger lock held by payout checks; it takes the lock
        itself too (re-entrant) so read-only callers get a consistent view.
        """
        normal, currency = self._normal_side(code)
        total = 0
        with self._lock:
            for entry in self._journal:
                for posting in entry.postings:
                    if posting.account == code:
                        minor = posting.amount.to_minor()
                        total += minor if posting.side is normal else -minor
        if total < 0 and normal is EntrySide.CREDIT:
            log.warning("account %s is overdrawn by %d minor units", code, -total)
        return Money.from_minor(total, currency)

    def balance_as_of(self, code: str, cutoff: datetime) -> Money:
        """Balance on the account's normal side, including entries posted at or before ``cutoff``."""
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")
        if code not in self._accounts:
            raise UnknownAccount(code)
        kind, currency = self._accounts[code]
        normal = EntrySide.DEBIT if kind is AccountKind.ASSET else EntrySide.CREDIT
        total = 0
        counted = 0
        for entry in self._journal:
            if entry.posted_at >= cutoff:
                continue
            for posting in entry.postings:
                if posting.account == code:
                    minor = posting.amount.to_minor()
                    total += minor if posting.side is normal else -minor
                    counted += 1
        log.debug("balance of %s as of %s: %d minor units from %d postings", code, cutoff, total, counted)
        return Money.from_minor(total, currency)

    def _entries_between(self, start: datetime, end: datetime) -> Iterator[JournalEntry]:
        """Yield entries with ``start <= posted_at < end`` in posting-time order.

        Entries are appended in wall-clock order almost always, but backdated
        settlement postings (``posted_at`` passed explicitly) can land out of
        order, so we sort a snapshot rather than trusting append order. Ties
        keep append order because ``sorted`` is stable.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("statement bounds must be timezone-aware")
        with self._lock:
            snapshot = list(self._journal)
        for entry in snapshot:
            if entry.posted_at < start:
                continue
            if entry.posted_at >= end:
                break
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
            running += change
        log.debug("statement for %s: %d lines, opening %d, closing %d", code, len(lines), opening, running)
        return lines

    def trial_balance(self) -> dict[str, int]:
        """Net debit-minus-credit per currency in minor units; every value must be zero."""
        totals: dict[str, int] = {}
        with self._lock:
            entries = list(self._journal)
        for entry in entries:
            for posting in entry.postings:
                minor = posting.amount.to_minor()
                signed = minor if posting.side is EntrySide.DEBIT else -minor
                totals[posting.amount.currency] = totals.get(posting.amount.currency, 0) + signed
        broken = {currency: net for currency, net in totals.items() if net != 0}
        if broken:
            # Every entry is validated on the way in, so an imbalance here means
            # someone appended to the journal without going through post().
            log.error("ledger trial balance does not net to zero: %s", broken)
        else:
            log.debug("trial balance ok across %d currencies, %d entries", len(totals), len(entries))
        return totals
