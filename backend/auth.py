"""
Authentication utilities: password hashing, JWT token management, and FastAPI dependency.

This module provides the auth infrastructure for Finvisor AI's multi-user support.

Password hashing uses bcrypt via passlib -- bcrypt is deliberately slow (tunable cost factor)
which makes brute-force attacks against leaked hashes impractical, unlike a fast hash like SHA-256
that can be tested billions of times per second on commodity hardware.

JWTs are signed with HS256 (symmetric HMAC) rather than RS256 (asymmetric RSA). HS256 is the
right choice here because both the issuer and the verifier are the same backend process -- there's
no separate service that needs to verify tokens without the signing key. HS256 is simpler,
produces smaller tokens, and has no key rotation ceremony to get wrong.

The token is passed in the standard `Authorization: Bearer <token>` header. The frontend stores
it in localStorage, which is acceptable for this self-hosted tool (XSS is the real threat vector
regardless of cookie vs. localStorage, and httpOnly cookies introduce CSRF complexity that buys
nothing for a same-origin API).
"""
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Request, HTTPException
from jose import jwt, JWTError
import bcrypt

logger = logging.getLogger("Auth")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "finvisor-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "1440"))  # 24 hours default


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """Hashes a plain-text password with bcrypt. Returns the full bcrypt hash string."""
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a bcrypt hash. Returns True if they match."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------

def create_access_token(user_id: str, email: str) -> str:
    """
    Creates a signed JWT containing the user's id and email.

    The `sub` (subject) claim is the user id as a string -- MongoDB ObjectIds are not
    JSON-serializable directly, so the caller converts to str before passing in.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decodes and validates a JWT. Returns the payload dict.
    Raises JWTError if the token is invalid, expired, or tampered with.
    """
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_current_user(request: Request) -> dict:
    """
    FastAPI dependency that extracts and validates the Bearer token from the request.

    Returns a dict with at least `user_id` and `email` keys.
    Raises 401 if the token is missing, malformed, or expired.

    Usage in an endpoint:
        @app.get("/protected")
        def protected(user: dict = Depends(get_current_user)):
            ...
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.split(" ", 1)[1]
    try:
        payload = decode_access_token(token)
    except JWTError as e:
        logger.debug(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    email = payload.get("email")
    if not user_id or not email:
        raise HTTPException(
            status_code=401,
            detail="Invalid token payload.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"user_id": user_id, "email": email}


def get_optional_user(request: Request) -> Optional[dict]:
    """
    Like get_current_user but returns None instead of raising when no token is present.
    Used for endpoints that work both authenticated and anonymously but attach user_id
    when available (e.g. /chat, /feedback).
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ", 1)[1]
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        email = payload.get("email")
        if user_id and email:
            return {"user_id": user_id, "email": email}
    except JWTError:
        pass

    return None
