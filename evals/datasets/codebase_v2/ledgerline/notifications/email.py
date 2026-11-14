"""Transactional email: rendering, localised amounts and SMTP delivery.

Only transactional mail (receipts, refunds, password resets) goes through
here; marketing mail is sent by the CRM. Delivery is synchronous via the
internal relay, which accepts mail quickly and handles retries to the
outside world itself.
"""
from __future__ import annotations

import html
import logging
import re
import smtplib
import time
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from string import Template

from ledgerline.payments.currency import Money

log = logging.getLogger(__name__)

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "KRW": "₩"}

# Locales that group with "." and use "," for decimals, symbol after the amount.
_SUFFIX_LOCALES = ("de", "fr", "it", "es", "nl", "pt")

_ADDRESS = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# SMTP reply codes worth retrying against the relay (greylisting, overload).
_TRANSIENT_CODES = frozenset({421, 450, 451, 452})
SMTP_TIMEOUT_SECONDS = 10
MAX_SEND_ATTEMPTS = 3


class InvalidRecipient(ValueError):
    pass


def format_money(money: Money, locale: str = "en_US") -> str:
    """Render an amount for humans, e.g. ``$1,234.50`` or ``1.234,50 €``."""
    symbol = CURRENCY_SYMBOLS.get(money.currency, f"{money.currency} ")
    digits = f"{money.amount:,.2f}"
    if money.is_negative():
        # Refund adjustments can be negative; keep the sign in front of the symbol.
        digits = digits.lstrip("-")
        sign = "-"
    else:
        sign = ""
    if locale.split("_")[0] in _SUFFIX_LOCALES:
        digits = digits.replace(",", " ").replace(".", ",").replace(" ", ".")
        return f"{sign}{digits} {symbol.strip()}"
    return f"{sign}{symbol}{digits}"


def _validate_address(address: str) -> str:
    cleaned = address.strip()
    if not _ADDRESS.match(cleaned):
        raise InvalidRecipient(f"not a deliverable address: {address!r}")
    # Header injection guard: addresses come from customer profiles.
    if "\n" in cleaned or "\r" in cleaned:
        raise InvalidRecipient("newline in address")
    return cleaned


@dataclass(frozen=True)
class OutboundEmail:
    to: str
    subject: str
    body: str
    tag: str
    reply_to: str | None = None
    html_body: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


_REFUND_TEXT = Template(
    "Hi $name,\n\n"
    "We've issued a refund of $amount for order $order_id. Depending on your bank it can\n"
    "take 5-10 business days to appear on your statement.\n\n— The $merchant team\n"
)

_REFUND_HTML = Template(
    "<p>Hi $name,</p>"
    "<p>We've issued a refund of <strong>$amount</strong> for order <code>$order_id</code>. "
    "Depending on your bank it can take 5-10 business days to appear on your statement.</p>"
    "<p>— The $merchant team</p>"
)


class RefundEmailComposer:
    def __init__(self, merchant_name: str, support_address: str) -> None:
        if not merchant_name.strip():
            raise ValueError("merchant_name is required")
        self._merchant = merchant_name.strip()
        self._support = _validate_address(support_address)

    def render(self, to: str, name: str, order_id: str, amount: Money, locale: str = "en_US") -> OutboundEmail:
        recipient = _validate_address(to)
        if amount.is_negative() or amount.is_zero():
            # A zero/negative "refund" email confuses customers and support alike.
            raise ValueError(f"refusing to render refund email for amount {amount}")
        display_name = (name or "").strip() or "there"
        formatted = format_money(amount, locale)
        body = _REFUND_TEXT.substitute(
            name=display_name,
            amount=formatted,
            order_id=order_id,
            merchant=self._merchant,
        )
        html_body = _REFUND_HTML.substitute(
            name=html.escape(display_name),
            amount=html.escape(formatted),
            order_id=html.escape(order_id),
            merchant=html.escape(self._merchant),
        )
        return OutboundEmail(
            recipient,
            f"Your refund for order {order_id}",
            body,
            "refund",
            self._support,
            html_body=html_body,
            headers={"X-Ledgerline-Order": order_id},
        )


class SmtpMailer:
    def __init__(self, host: str, port: int, username: str, password: str, sender: str,
                 sender_name: str = "Ledgerline") -> None:
        self._host, self._port = host, port
        self._username, self._password = username, password
        self._sender = _validate_address(sender)
        self._sender_name = sender_name

    def _build(self, email: OutboundEmail) -> EmailMessage:
        msg = EmailMessage()
        msg["From"] = formataddr((self._sender_name, self._sender))
        msg["To"] = email.to
        msg["Subject"] = email.subject
        msg["Message-ID"] = make_msgid(domain=self._sender.split("@", 1)[1])
        msg["X-Ledgerline-Tag"] = email.tag
        for name, value in email.headers.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"header {name} contains a newline")
            msg[name] = value
        if email.reply_to:
            msg["Reply-To"] = email.reply_to
        msg.set_content(email.body)
        if email.html_body:
            msg.add_alternative(email.html_body, subtype="html")
        return msg

    def send(self, email: OutboundEmail) -> None:
        msg = self._build(email)
        with smtplib.SMTP(self._host, self._port, timeout=SMTP_TIMEOUT_SECONDS) as smtp:
            smtp.starttls()
            smtp.login(self._username, self._password)
            smtp.send_message(msg)
        log.debug("sent %s mail %s to relay", email.tag, msg["Message-ID"])

    def send_with_retry(self, email: OutboundEmail, attempts: int = MAX_SEND_ATTEMPTS) -> None:
        """Send, retrying transient relay rejections (4xx) and dropped connections."""
        for attempt in range(1, attempts + 1):
            try:
                self.send(email)
                return
            except smtplib.SMTPResponseException as exc:
                if exc.smtp_code not in _TRANSIENT_CODES or attempt == attempts:
                    raise
                log.info("relay deferred %s mail (%d), retrying", email.tag, exc.smtp_code)
            except smtplib.SMTPServerDisconnected:
                if attempt == attempts:
                    raise
                log.info("relay dropped connection sending %s mail, retrying", email.tag)
            time.sleep(0.5 * attempt)


class EmailRefundNotifier:
    """Adapter satisfying ``payments.refunds.RefundNotifier``."""

    def __init__(self, composer: RefundEmailComposer, mailer: SmtpMailer) -> None:
        self._composer = composer
        self._mailer = mailer

    def refund_issued(self, email: str, order_id: str, amount: Money) -> None:
        try:
            message = self._composer.render(email, "", order_id, amount)
        except (InvalidRecipient, ValueError):
            log.warning("refund email for %s not rendered (bad address or amount)", order_id)
            return
        try:
            self._mailer.send_with_retry(message)
        except smtplib.SMTPException:
            # A failed receipt must never roll back a refund that already went out.
            log.exception("refund email for %s not delivered", order_id)


_RECEIPT_LINE = Template("  $quantity x $name  $amount\n")
_RECEIPT_TEXT = Template(
    "Hi $name,\n\n"
    "Thanks for your order $order_id. Here's what you bought:\n\n"
    "$lines\n"
    "Total charged: $total\n\n"
    "We'll email you again when it ships.\n\n— The $merchant team\n"
)


@dataclass(frozen=True)
class ReceiptLine:
    name: str
    quantity: int
    amount: Money


class ReceiptEmailComposer:
    """Order confirmation sent right after capture."""

    # Long carts make unreadable emails; the full list is on the order page.
    MAX_LINES = 25

    def __init__(self, merchant_name: str, support_address: str) -> None:
        self._merchant = merchant_name.strip()
        self._support = _validate_address(support_address)

    def render(
        self, to: str, name: str, order_id: str, lines: list[ReceiptLine], total: Money,
        locale: str = "en_US",
    ) -> OutboundEmail:
        recipient = _validate_address(to)
        if not lines:
            raise ValueError(f"receipt for {order_id} has no lines")
        currencies = {line.amount.currency for line in lines} | {total.currency}
        if len(currencies) != 1:
            raise ValueError(f"receipt for {order_id} mixes currencies: {sorted(currencies)}")

        rendered = [
            _RECEIPT_LINE.substitute(
                quantity=line.quantity,
                name=line.name[:60],
                amount=format_money(line.amount, locale),
            )
            for line in lines[: self.MAX_LINES]
        ]
        hidden = len(lines) - self.MAX_LINES
        if hidden > 0:
            rendered.append(f"  ...and {hidden} more item{'s' if hidden != 1 else ''}\n")

        body = _RECEIPT_TEXT.substitute(
            name=(name or "").strip() or "there",
            order_id=order_id,
            lines="".join(rendered),
            total=format_money(total, locale),
            merchant=self._merchant,
        )
        return OutboundEmail(
            recipient,
            f"Order {order_id} confirmed",
            body,
            "receipt",
            self._support,
            headers={"X-Ledgerline-Order": order_id},
        )
