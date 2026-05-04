"""User authentication: bcrypt password hashing + JWT bearer tokens."""
from __future__ import annotations

from datetime import datetime, timedelta

import bcrypt
import jwt

from ..config import settings

JWT_ALG = "HS256"


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def issue_token(user_id: int) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": str(user_id),
        "kind": "user",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=settings.jwt_ttl_days)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALG)


def issue_operator_token(course_id: int) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": f"course:{course_id}",
        "kind": "operator",
        "course_id": course_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=settings.jwt_ttl_days)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALG)


def decode_token(token: str) -> int | None:
    """Returns user_id for legacy `user` tokens (and operator tokens are
    not user accounts so they return None here)."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALG])
        if payload.get("kind", "user") != "user":
            return None
        return int(payload["sub"])
    except (jwt.PyJWTError, ValueError, KeyError):
        return None


def decode_operator_token(token: str) -> int | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALG])
        if payload.get("kind") != "operator":
            return None
        return int(payload["course_id"])
    except (jwt.PyJWTError, ValueError, KeyError):
        return None
