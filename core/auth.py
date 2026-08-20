# auth.py — ShuttleEye account store (PostgreSQL-backed)
# ═══════════════════════════════════════════════════════════════════════
#  Accounts with two roles:
#    • admin   — manages umpire accounts, can also run the dashboard
#    • umpire  — goes straight to the umpire dashboard
#
#  Passwords are never stored in plaintext: PBKDF2-HMAC-SHA256 with a
#  random per-user salt (stdlib only, no extra dependency).
# ═══════════════════════════════════════════════════════════════════════

import hashlib
import secrets

import db

PBKDF2_ROUNDS = 200_000

ROLES = ("admin", "umpire")

_DEFAULT_ACCOUNTS = (
    ("admin", "admin123", "admin"),
    ("umpire", "umpire123", "umpire"),
)


def _hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ROUNDS
    ).hex()
    return salt, digest


def ensure_default_accounts():
    """Seed the default admin/umpire accounts if the users table is empty.
    Safe to call on every app startup."""
    with db.get_conn() as conn:
        (count,) = conn.execute("SELECT count(*) AS c FROM users").fetchone().values()
        if count > 0:
            return
        for username, password, role in _DEFAULT_ACCOUNTS:
            salt, digest = _hash_password(password)
            conn.execute(
                "INSERT INTO users (username, salt, password_hash, role) "
                "VALUES (%s, %s, %s, %s)",
                (username, salt, digest, role),
            )


def authenticate(username, password):
    """Returns role string on success, or None on failure."""
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT salt, password_hash, role FROM users WHERE username = %s",
            (username,),
        ).fetchone()
    if not row:
        return None
    _, digest = _hash_password(password, row["salt"])
    if secrets.compare_digest(digest, row["password_hash"]):
        return row["role"]
    return None


def get_user_id(username):
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE username = %s", (username,)
        ).fetchone()
    return row["id"] if row else None


def add_user(username, password, role):
    if role not in ROLES:
        raise ValueError(f"Invalid role: {role}")
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password are required.")
    salt, digest = _hash_password(password)
    with db.get_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM users WHERE username = %s", (username,)
        ).fetchone()
        if exists:
            raise ValueError(f"User '{username}' already exists.")
        conn.execute(
            "INSERT INTO users (username, salt, password_hash, role) "
            "VALUES (%s, %s, %s, %s)",
            (username, salt, digest, role),
        )


def remove_user(username):
    with db.get_conn() as conn:
        cur = conn.execute("DELETE FROM users WHERE username = %s", (username,))
        if cur.rowcount == 0:
            raise ValueError(f"User '{username}' not found.")


def reset_password(username, new_password):
    salt, digest = _hash_password(new_password)
    with db.get_conn() as conn:
        cur = conn.execute(
            "UPDATE users SET salt = %s, password_hash = %s WHERE username = %s",
            (salt, digest, username),
        )
        if cur.rowcount == 0:
            raise ValueError(f"User '{username}' not found.")


def list_users(role=None):
    with db.get_conn() as conn:
        if role is None:
            rows = conn.execute("SELECT username, role FROM users").fetchall()
        else:
            rows = conn.execute(
                "SELECT username, role FROM users WHERE role = %s", (role,)
            ).fetchall()
    return {r["username"]: {"role": r["role"]} for r in rows}
