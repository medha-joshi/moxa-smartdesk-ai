"""
auth.py — Email + password sign-in for pre-registered employees.

Employees are listed in .streamlit/secrets.toml (see secrets.toml.example) with a scrypt hash of their
password, never the password itself. Generate a hash with:

    uv run moxa-hash-password
"""

import getpass
import hashlib
import hmac
import secrets
from collections.abc import Mapping

SCRYPT_PARAMS = {"n": 2**14, "r": 8, "p": 1}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **SCRYPT_PARAMS)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), **SCRYPT_PARAMS)
        return hmac.compare_digest(candidate, bytes.fromhex(digest))
    except ValueError:
        return False


# Checked for unknown emails too, so a wrong email takes as long as a wrong password
_DUMMY_HASH = hash_password(secrets.token_hex(16))


def authenticate(users: Mapping[str, Mapping], email: str, password: str) -> dict | None:
    """Return {"email", "name"} for a registered employee with the right password, else None."""
    email = email.strip().lower()
    accounts = {key.lower(): user for key, user in users.items()}
    user = accounts.get(email)
    stored = user.get("password_hash", "") if user else _DUMMY_HASH
    if not verify_password(password, stored) or user is None:
        return None
    return {"email": email, "name": user.get("name") or email}


def main() -> None:
    """CLI: print a password hash to paste into .streamlit/secrets.toml."""
    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords don't match.")
    print(hash_password(password))


if __name__ == "__main__":
    main()
