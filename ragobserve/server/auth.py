"""API key auth for the RAGObserve server.

Key resolution order:
  1. ``RAGOBSERVE_API_KEY`` env var (set before starting the server)
  2. Auto-generated ``secrets.token_urlsafe(32)`` — stable for the process lifetime,
     printed by ``serve()`` so the user can copy it.

All ``/api/*`` routes depend on ``require_api_key``.
Dashboard HTML pages are unauthenticated shells; the JS reads the key from
localStorage (seeded via ``?key=<key>`` on first visit) and adds the
Authorization header to every fetch call.
"""
from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def get_api_key() -> str:
    """Return the server's API key, generating one if not already set."""
    key = os.environ.get("RAGOBSERVE_API_KEY", "")
    if not key:
        key = secrets.token_urlsafe(32)
        os.environ["RAGOBSERVE_API_KEY"] = key
    return key


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    x_api_key: str | None = Header(None, alias="X-Api-Key"),
) -> str:
    """FastAPI dependency — raises 401 if the API key is missing or wrong."""
    api_key = get_api_key()
    token = (credentials.credentials if credentials else None) or x_api_key or ""
    if not token or not secrets.compare_digest(token, api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. "
                   "Pass Authorization: Bearer <key> or X-Api-Key: <key>.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
