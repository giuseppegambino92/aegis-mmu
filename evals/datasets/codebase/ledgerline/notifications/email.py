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
# RFC 5321 caps a forward-path at 256 octets; the relay rejects longer ones
# with a 501 that would otherwise look like a transient failure in our logs.
_MAX_ADDRESS_LENGTH = 254
# Merchant names end up in the From display name and the email footer.
_MAX_MERCHANT_NAME_LENGTH = 80
# Upper bound on the linear backoff between relay attempts.
_MAX_BACKOFF_SECONDS = 5.0


class InvalidRecipient(ValueError):
    pass


def format_money(money: Money, locale: str = "en_US") -> str:
    """Render an amount for humans, e.g. ``$1,234.50`` or ``1.234,50 €``.

    ``locale`` is a POSIX-style tag (``de_DE``, ``en_GB``); only the language
    part decides the layout. Unknown currencies fall back to their ISO code
    as a prefix so the email is still unambiguous.
    """
    if money is None:
        raise ValueError("format_money() needs an amount")
    language = (locale or "en_US").replace("-", "_").split("_")[0].lower()
    symbol = CURRENCY_SYMBOLS.get(money.currency, f"{money.currency} ")
    digits = f"{money.amount:,.2f}"
    if money.is_negative():
        # Refund adjustments can be negative; keep the sign in front of the symbol.
        digits = digits.lstrip("-")
        sign = "-"
    else:
        sign = ""
    if language in _SUFFIX_LOCALES:
        # Swap separators via a placeholder so "1,234.50" becomes "1.234,50".
        digits = digits.replace(",", " ").replace(".", ",").replace(" ", ".")
        return f"{sign}{digits} {symbol.strip()}"
    if symbol.endswith(" ") and len(symbol.strip()) == 3:
        # ISO-code prefix ("CHF 12.00"): no symbol glyph available, keep the space.
        log.debug("no currency symbol for %s, using ISO code", money.currency)
    return f"{sign}{symbol}{digits}"


def _validate_address(address: str) -> str:
    """Return a cleaned, deliverable address or raise :class:`InvalidRecipient`.

    Addresses come straight from customer profiles and merchant settings, so
    they are treated as untrusted header values.
    """
    if not isinstance(address, str):
        raise InvalidRecipient(f"address must be a string, got {type(address).__name__}")
    cleaned = address.strip()
    if not cleaned:
        raise InvalidRecipient("empty address")
    if len(cleaned) > _MAX_ADDRESS_LENGTH:
        raise InvalidRecipient(f"address longer than {_MAX_ADDRESS_LENGTH} characters")
    if not _ADDRESS.match(address):
        raise InvalidRecipient(f"not a deliverable address: {address!r}")
    # Header injection guard: addresses come from customer profiles.
    if "\n" in cleaned or "\r" in cleaned:
        raise InvalidRecipient("newline in address")
    local, _, domain = cleaned.rpartition("@")
    if domain.startswith(".") or domain.endswith(".") or ".." in domain:
        raise InvalidRecipient(f"malformed domain in address: {address!r}")
    # Domains are case-insensitive; normalising keeps relay dedupe keys stable.
    return f"{local}@{domain.lower()}"


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
        if not isinstance(merchant_name, str) or not merchant_name.strip():
            raise ValueError("merchant_name is required")
        cleaned_name = " ".join(merchant_name.split())
        if len(cleaned_name) > _MAX_MERCHANT_NAME_LENGTH:
            # Long legal names wrap badly in the footer; the trading name is enough.
            log.warning("merchant name %r truncated for refund emails", cleaned_name)
            cleaned_name = cleaned_name[:_MAX_MERCHANT_NAME_LENGTH].rstrip()
        self._merchant = cleaned_name
        self._support = _validate_address(support_address)
        log.debug("refund composer ready for %s (reply-to %s)", self._merchant, self._support)

    def render(self, to: str, name: str, order_id: str, amount: Money, locale: str = "en_US") -> OutboundEmail:
        recipient = _validate_address(to)
        if not order_id or not order_id.strip():
            raise ValueError("order_id is required for a refund email")
        if amount.is_negative() or amount.is_zero():
            # A zero/negative "refund" email confuses customers and support alike.
            raise ValueError(f"refusing to render refund email for amount {amount}")
        order_id = order_id.strip()
        display_name = (name or "").strip() or "there"
        # Customer names are free text; keep the greeting to a single short line.
        display_name = display_name.splitlines()[0][:60]
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
            recipient,
            html_body=html_body,
            headers={"X-Ledgerline-Order": order_id},
        )


class SmtpMailer:
    def __init__(self, host: str, port: int, username: str, password: str, sender: str,
                 sender_name: str = "Ledgerline") -> None:
        if not host or not host.strip():
            raise ValueError("SMTP host is required")
        if not 0 < int(port) < 65536:
            raise ValueError(f"SMTP port out of range: {port}")
        if bool(username) != bool(password):
            # Half-configured credentials fail at login with an opaque 535.
            raise ValueError("SMTP username and password must be set together")
        self._host, self._port = host.strip(), int(port)
        self._username, self._password = username, password
        self._sender = _validate_address(sender)
        self._sender_name = (sender_name or "Ledgerline").strip()
        log.info("smtp mailer configured for %s:%d as %s", self._host, self._port, self._sender)

    def _build(self, email: OutboundEmail) -> EmailMessage:
        if not email.subject or "\n" in email.subject or "\r" in email.subject:
            raise ValueError(f"invalid subject for {email.tag} mail")
        msg = EmailMessage()
        msg["From"] = formataddr((self._sender_name, self._sender))
        msg["To"] = email.to
        msg["Subject"] = email.subject
        msg["Message-ID"] = make_msgid(domain=self._sender.split("@", 1)[1])
        msg["X-Ledgerline-Tag"] = email.tag
        for name, value in email.headers.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"header {name} contains a newline")
            if name.lower() in ("from", "to", "subject", "message-id"):
                # Callers may not override envelope-relevant headers.
                raise ValueError(f"header {name} is reserved")
            msg[name] = value
        if email.reply_to:
            msg["Reply-To"] = email.reply_to
        msg.set_content(email.body)
        if email.html_body:
            msg.add_alternative(email.html_body, subtype="html")
        return msg

    def send(self, email: OutboundEmail) -> None:
        msg = self._build(email)
        started = time.monotonic()
        with smtplib.SMTP(self._host, self._port, timeout=SMTP_TIMEOUT_SECONDS) as smtp:
            smtp.ehlo()
            if smtp.has_extn("starttls"):
                smtp.starttls()
                smtp.ehlo()
            else:
                # The internal relay always offers STARTTLS; a missing extension
                # usually means we are talking to a misrouted host.
                log.warning("relay %s:%d did not offer STARTTLS", self._host, self._port)
            if self._username:
                smtp.login(self._username, self._password)
            refused = smtp.send_message(msg)
        if refused:
            log.warning("relay refused recipients for %s mail: %s", email.tag, sorted(refused))
        elapsed_ms = (time.monotonic() - started) * 1000
        log.debug("sent %s mail %s to relay in %.0fms", email.tag, msg["Message-ID"], elapsed_ms)

    def send_with_retry(self, email: OutboundEmail, attempts: int = MAX_SEND_ATTEMPTS) -> None:
        """Send, retrying transient relay rejections (4xx) and dropped connections."""
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        for attempt in range(1, attempts):
            try:
                self.send(email)
                if attempt > 1:
                    log.info("%s mail delivered to relay on attempt %d", email.tag, attempt)
                return
            except smtplib.SMTPResponseException as exc:
                if exc.smtp_code not in _TRANSIENT_CODES or attempt == attempts:
                    raise
                log.info("relay deferred %s mail (%d), retrying", email.tag, exc.smtp_code)
            except smtplib.SMTPServerDisconnected:
                if attempt == attempts:
                    raise
                log.info("relay dropped connection sending %s mail, retrying", email.tag)
            # Linear backoff: the relay is local, long waits only hold the worker.
            time.sleep(min(0.5 * attempt, _MAX_BACKOFF_SECONDS))


class EmailRefundNotifier:
    """Adapter satisfying ``payments.refunds.RefundNotifier``."""

    def __init__(self, composer: RefundEmailComposer, mailer: SmtpMailer) -> None:
        if composer is None or mailer is None:
            raise ValueError("EmailRefundNotifier needs both a composer and a mailer")
        self._composer = composer
        self._mailer = mailer
        # Counters are scraped by the notifications health check.
        self.sent = 0
        self.skipped = 0
        self.failed = 0

    def refund_issued(self, email: str, order_id: str, amount: Money) -> None:
        if not email:
            # Guest checkouts without an email simply get no notification.
            log.info("no email on file for order %s, skipping refund notice", order_id)
            self.skipped += 1
            return
        try:
            message = self._composer.render(email, "", order_id, amount)
        except (InvalidRecipient, ValueError):
            log.warning("refund email for %s not rendered (bad address or amount)", order_id)
            self.skipped += 1
            return
        try:
            self._mailer.send_with_retry(message)
        except smtplib.SMTPException:
            # A failed receipt must never roll back a refund that already went out.
            log.exception("refund email for %s not delivered", order_id)
            self.failed += 1
            return
        self.sent += 1
        log.info("refund email for %s queued at relay (%s)", order_id, amount.currency)


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
        if not isinstance(merchant_name, str) or not merchant_name.strip():
            raise ValueError("merchant_name is required")
        cleaned_name = " ".join(merchant_name.split())
        if len(cleaned_name) > _MAX_MERCHANT_NAME_LENGTH:
            log.warning("merchant name %r truncated for receipts", cleaned_name)
            cleaned_name = cleaned_name[:_MAX_MERCHANT_NAME_LENGTH].rstrip()
        self._merchant = cleaned_name
        self._support = _validate_address(support_address)
        log.debug("receipt composer ready for %s", self._merchant)

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
        if any(line.quantity <= 0 for line in lines):
            raise ValueError(f"receipt for {order_id} has a non-positive quantity")

        rendered = [
            _RECEIPT_LINE.substitute(
                quantity=line.quantity,
                name=line.name[:60],
                amount=format_money(line.amount),
            )
            for line in lines[: self.MAX_LINES]
        ]
        hidden = len(lines) - self.MAX_LINES
        if hidden > 0:
            rendered.append(f"  ...and {hidden} more item{'s' if hidden != 1 else ''}\n")
            log.debug("receipt for %s truncated, %d lines hidden", order_id, hidden)

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
