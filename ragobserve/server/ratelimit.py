"""Shared slowapi rate-limiter instance.

Import ``limiter`` here so api.py and app.py share the same object.
Key is client IP address; limits are per-IP.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
