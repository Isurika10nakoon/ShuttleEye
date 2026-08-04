# auth.py — ShuttleEye local account store
# ═══════════════════════════════════════════════════════════════════════
#  Local JSON-backed accounts with two roles:
#    • admin   — manages umpire accounts, can also run the dashboard
#    • umpire  — goes straight to the umpire dashboard
#
#  Passwords are never stored in plaintext: PBKDF2-HMAC-SHA256 with a
#  random per-user salt (stdlib only, no extra dependency).
# ═══════════════════════════════════════════════════════════════════════

import json
import os
import hashlib
import secrets

USERS_FILE   = os.path.join(os.path.dirname(__file__), "..", "users.json")
PBKDF2_ROUNDS = 200_000

ROLES = ("admin", "umpire")


def _hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ROUNDS
    ).hex()
    return salt, digest


def _default_users():
    users = {}
    for username, password, role in (
        ("admin", "admin123", "admin"),
        ("umpire", "umpire123", "umpire"),
    ):
        salt, digest = _hash_password(password)
        users[username] = {"salt": salt, "hash": digest, "role": role}
    return users


def load_users():
    if not os.path.exists(USERS_FILE):
        users = _default_users()
        save_users(users)
        return users
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def authenticate(username, password):
    """Returns role string on success, or None on failure."""
    users = load_users()
    record = users.get(username)
    if not record:
        return None
    _, digest = _hash_password(password, record["salt"])
    if secrets.compare_digest(digest, record["hash"]):
        return record["role"]
    return None


def add_user(username, password, role):
    if role not in ROLES:
        raise ValueError(f"Invalid role: {role}")
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password are required.")
    users = load_users()
    if username in users:
        raise ValueError(f"User '{username}' already exists.")
    salt, digest = _hash_password(password)
    users[username] = {"salt": salt, "hash": digest, "role": role}
    save_users(users)


def remove_user(username):
    users = load_users()
    if username not in users:
        raise ValueError(f"User '{username}' not found.")
    del users[username]
    save_users(users)


def reset_password(username, new_password):
    users = load_users()
    if username not in users:
        raise ValueError(f"User '{username}' not found.")
    salt, digest = _hash_password(new_password)
    users[username]["salt"] = salt
    users[username]["hash"] = digest
    save_users(users)


def list_users(role=None):
    users = load_users()
    return {u: r for u, r in users.items() if role is None or r["role"] == role}
