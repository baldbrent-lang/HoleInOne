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


def verify_webhook(payload: bytes, sig_header: str) -> dict:
    if not settings.stripe_webhook_secret:
        import json

        return json.loads(payload.decode("utf-8"))

    import stripe

    stripe.api_key = settings.stripe_secret_key
    return stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
