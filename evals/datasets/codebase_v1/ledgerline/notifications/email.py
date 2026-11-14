"""Transactional email: rendering, localised amounts and SMTP delivery."""
from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from string import Template

from ledgerline.payments.currency import Money

log = logging.getLogger(__name__)

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "KRW": "₩"}

# Locales that group with "." and use "," for decimals, symbol after the amount.
_SUFFIX_LOCALES = ("de", "fr", "it", "es", "nl", "pt")


def format_money(money: Money, locale: str = "en_US") -> str:
    """Render an amount for humans, e.g. ``$1,234.50`` or ``1.234,50 €``."""
    symbol = CURRENCY_SYMBOLS.get(money.currency, f"{money.currency} ")
    digits = f"{money.amount:,.2f}"
    if locale.split("_")[0] in _SUFFIX_LOCALES:
        digits = digits.replace(",", " ").replace(".", ",").replace(" ", ".")
        return f"{digits} {symbol.strip()}"
    return f"{symbol}{digits}"


@dataclass(frozen=True)
class OutboundEmail:
    to: str
    subject: str
    body: str
    tag: str
    reply_to: str | None = None


_REFUND_TEXT = Template(
    "Hi $name,\n\n"
    "We've issued a refund of $amount for order $order_id. Depending on your bank it can\n"
    "take 5-10 business days to appear on your statement.\n\n— The $merchant team\n"
)


class RefundEmailComposer:
    def __init__(self, merchant_name: str, support_address: str) -> None:
        self._merchant = merchant_name
        self._support = support_address

    def render(self, to: str, name: str, order_id: str, amount: Money, locale: str = "en_US") -> OutboundEmail:
        body = _REFUND_TEXT.substitute(
            name=name or "there",
            amount=format_money(amount, locale),
            order_id=order_id,
            merchant=self._merchant,
        )
        return OutboundEmail(to, f"Your refund for order {order_id}", body, "refund", self._support)


class SmtpMailer:
    def __init__(self, host: str, port: int, username: str, password: str, sender: str) -> None:
        self._host, self._port = host, port
        self._username, self._password = username, password
        self._sender = sender

    def send(self, email: OutboundEmail) -> None:
        msg = EmailMessage()
        msg["From"] = self._sender
        msg["To"] = email.to
        msg["Subject"] = email.subject
        msg["X-Ledgerline-Tag"] = email.tag
        if email.reply_to:
            msg["Reply-To"] = email.reply_to
        msg.set_content(email.body)
        with smtplib.SMTP(self._host, self._port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(self._username, self._password)
            smtp.send_message(msg)


class EmailRefundNotifier:
    """Adapter satisfying ``payments.refunds.RefundNotifier``."""

    def __init__(self, composer: RefundEmailComposer, mailer: SmtpMailer) -> None:
        self._composer = composer
        self._mailer = mailer

    def refund_issued(self, email: str, order_id: str, amount: Money) -> None:
        try:
            self._mailer.send(self._composer.render(email, "", order_id, amount))
        except smtplib.SMTPException:
            # A failed receipt must never roll back a refund that already went out.
            log.exception("refund email for %s not delivered", order_id)
