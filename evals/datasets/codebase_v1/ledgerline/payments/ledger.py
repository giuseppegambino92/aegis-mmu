"""Double-entry ledger for merchant balances.

Accounts are namespaced per merchant and currency, e.g. ``m_8812:EUR:payable``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable
from uuid import UUID, uuid4

from ledgerline.payments.currency import CurrencyMismatch, Money


class AccountKind(str, Enum):
    ASSET = "asset"
    LIABILITY = "liability"
    REVENUE = "revenue"


class EntrySide(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


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


class UnbalancedEntryError(ValueError):
    pass


def merchant_account_codes(merchant_id: str, currency: str) -> tuple[str, str, str]:
    """Return ``(clearing, payable, fees)`` account codes for a merchant."""
    prefix = f"{merchant_id}:{currency.upper()}"
    return f"{prefix}:clearing", f"{prefix}:payable", f"{prefix}:fees"


class Ledger:
    def __init__(self) -> None:
        self._accounts: dict[str, tuple[AccountKind, str]] = {}
        self._journal: list[JournalEntry] = []

    def open_merchant(self, merchant_id: str, currency: str) -> None:
        clearing, payable, fees = merchant_account_codes(merchant_id, currency)
        self._accounts[clearing] = (AccountKind.ASSET, currency.upper())
        self._accounts[payable] = (AccountKind.LIABILITY, currency.upper())
        self._accounts[fees] = (AccountKind.REVENUE, currency.upper())

    def post(
        self, memo: str, postings: Iterable[Posting], *, reference: str | None = None,
        posted_at: datetime | None = None,
    ) -> JournalEntry:
        frozen = tuple(postings)
        self._validate(frozen)
        entry = JournalEntry(uuid4(), posted_at or datetime.now(timezone.utc), memo, frozen, reference)
        self._journal.append(entry)
        return entry

    def _validate(self, postings: tuple[Posting, ...]) -> None:
        if len(postings) < 2:
            raise UnbalancedEntryError("a journal entry needs at least two postings")
        balance = 0
        for posting in postings:
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
        clearing, payable, _ = merchant_account_codes(merchant_id, amount.currency)
        return self.post(
            f"refund {refund_id} for order {order_id}",
            [
                Posting(payable, EntrySide.CREDIT, amount),
                Posting(clearing, EntrySide.DEBIT, amount),
            ],
            reference=order_id,
        )

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
