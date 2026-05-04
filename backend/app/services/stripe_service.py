"""Stripe wrapper. Falls back to a deterministic mock when no secret key is set."""
from __future__ import annotations

import secrets
from dataclasses import dataclass

from ..config import settings


@dataclass
class PaymentIntent:
    id: str
    client_secret: str
    status: str  # "requires_payment_method" | "succeeded"


def create_registration_payment_intent(
    participant_id: int,
    amount_cents: int | None = None,
) -> PaymentIntent:
    amount = amount_cents if amount_cents is not None else settings.registration_price_cents
    if not settings.stripe_secret_key:
        fake_id = f"pi_mock_{participant_id}_{secrets.token_hex(6)}"
        return PaymentIntent(id=fake_id, client_secret=f"{fake_id}_secret", status="succeeded")

    import stripe  # imported lazily to avoid requiring a key in tests

    stripe.api_key = settings.stripe_secret_key
    intent = stripe.PaymentIntent.create(
        amount=amount,
        currency="usd",
        metadata={"participant_id": str(participant_id)},
        automatic_payment_methods={"enabled": True},
    )
    return PaymentIntent(id=intent.id, client_secret=intent.client_secret, status=intent.status)


def refund_payment_intent(payment_intent_id: str | None) -> dict:
    """Refund a Participant's payment. Mock-mode is a no-op success.
    Returns {ok, mode, refund_id?, error?}."""
    if not payment_intent_id or payment_intent_id.startswith(("pi_mock_", "group_via_")):
        return {"ok": True, "mode": "mock", "refund_id": None}
    if not settings.stripe_secret_key:
        return {"ok": True, "mode": "mock-no-key", "refund_id": None}

    import stripe

    stripe.api_key = settings.stripe_secret_key
    try:
        refund = stripe.Refund.create(payment_intent=payment_intent_id)
        return {"ok": True, "mode": "stripe", "refund_id": refund.id}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "mode": "stripe", "error": str(exc)}


def verify_webhook(payload: bytes, sig_header: str) -> dict:
    if not settings.stripe_webhook_secret:
        import json

        return json.loads(payload.decode("utf-8"))

    import stripe

    stripe.api_key = settings.stripe_secret_key
    return stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
