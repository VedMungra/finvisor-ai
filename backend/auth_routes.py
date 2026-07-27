"""
Authentication API routes: register, login, and profile retrieval.

These are mounted as a sub-router at /auth by api.py. Keeping them in a separate module
follows the same pattern as the rest of the backend -- db.py for data access, a dedicated
module for the logic, and api.py as the thin HTTP wiring layer.

Registration and login both return the same response shape: {token, user}. The frontend
stores the token and immediately has the user profile without a second round trip.
"""
import logging
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, field_validator

from auth import hash_password, verify_password, create_access_token, get_current_user
import db

logger = logging.getLogger("AuthRoutes")

router = APIRouter(tags=["auth"])

# At least 8 characters, at least one letter and one digit. Deliberately not more restrictive
# than this -- overly complex rules (must have uppercase, symbol, etc.) don't measurably improve
# security but do measurably increase password-reset support load.
_PASSWORD_MIN_LENGTH = 8

# Simple email format check. Full RFC 5322 validation is overkill for a self-hosted tool;
# the real validation is "can we send you an email", which v1 doesn't do yet.
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., min_length=5, max_length=200)
    password: str = Field(..., min_length=_PASSWORD_MIN_LENGTH, max_length=128)

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not _EMAIL_RE.match(value):
            raise ValueError("Invalid email format.")
        return value

    @field_validator("username")
    @classmethod
    def _validate_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Username cannot be blank.")
        return value


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=200)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()


def _user_response(user_doc: dict) -> dict:
    """Shapes a MongoDB user document into the public response contract."""
    user_id = str(user_doc["_id"])
    return {
        "token": create_access_token(user_id, user_doc["email"]),
        "user": {
            "id": user_id,
            "username": user_doc["username"],
            "email": user_doc["email"],
        },
    }


@router.post("/register")
def register(request: RegisterRequest):
    """
    Creates a new user account and returns a JWT + profile.

    Duplicate emails are caught by the unique index on `users.email` and surfaced as a clear
    409 rather than a MongoDB DuplicateKeyError 500.
    """
    existing = db.get_user_by_email(request.email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    hashed = hash_password(request.password)

    try:
        user_doc = db.create_user(
            username=request.username,
            email=request.email,
            hashed_password=hashed,
        )
    except Exception as e:
        logger.error(f"Failed to create user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not create account. Please try again.")

    logger.info(f"New user registered: {request.email}")
    return _user_response(user_doc)


@router.post("/login")
def login(request: LoginRequest):
    """
    Authenticates a user by email + password and returns a JWT + profile.

    The error message is deliberately vague ("Invalid email or password") to avoid confirming
    whether an email is registered -- enumeration attacks shouldn't be free.
    """
    user_doc = db.get_user_by_email(request.email)
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not verify_password(request.password, user_doc["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    logger.info(f"User logged in: {request.email}")
    return _user_response(user_doc)


@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    """Returns the authenticated user's profile. Used by the frontend on page load to verify
    a stored token is still valid and to refresh the user object."""
    user_doc = db.get_user_by_id(user["user_id"])
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")

    return {
        "user": {
            "id": str(user_doc["_id"]),
            "username": user_doc["username"],
            "email": user_doc["email"],
        }
    }
