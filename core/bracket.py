# bracket.py — ShuttleEye single-elimination tournament bracket engine
# ═══════════════════════════════════════════════════════════════════════
#  Pure bracket logic (seeding, byes, winner advancement) on top of the
#  raw CRUD in db.py. Nothing here touches Tkinter or Flask — it's called
#  by web_dashboard.py (admin actions) and umpire_dashboard.py (reporting
#  a finished match back to whatever bracket slot it was playing).
#
#  Bracket shape: participants are padded up to the next power of two;
#  the padding slots are byes that auto-advance their real opponent.
#  Round 1 is the only round that can contain a bye — every later round
#  starts empty and fills in as earlier rounds resolve.
# ═══════════════════════════════════════════════════════════════════════

import random

from core import db


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def create_tournament(name, created_by=None):
    return db.create_tournament(name, created_by)


def add_participant(tournament_id, name):
    name = (name or "").strip()
    if not name:
        raise ValueError("Participant name is required.")
    return db.add_participant(tournament_id, name)


def generate_bracket(tournament_id):
    """Build every round's slots for this tournament, pair up round 1,
    and auto-resolve any byes (cascading them forward as far as they go).
    Call once, after all participants have been added."""
    tournament = db.get_tournament(tournament_id)
    if tournament is None:
        raise ValueError("Tournament not found.")
    if tournament["status"] != "draft":
        raise ValueError("This tournament's bracket has already been generated.")

    participants = db.list_participants(tournament_id)
    n = len(participants)
    if n < 2:
        raise ValueError("Need at least 2 participants to generate a bracket.")

    size = _next_pow2(n)
    num_rounds = size.bit_length() - 1  # log2(size)
    num_slots  = size // 2
    byes       = size - n

    # Randomise the draw, then give the first `byes` slots a single
    # participant (auto-advances) and every other slot two — this keeps
    # every round-1 slot valid (never two byes in the same slot, which a
    # flat "pad the list and pair sequentially" approach can produce).
    ids = [p["id"] for p in participants]
    random.shuffle(ids)

    i = 0
    for slot_num in range(num_slots):
        if slot_num < byes:
            a, b = ids[i], None
            i += 1
        else:
            a, b = ids[i], ids[i + 1]
            i += 2
        db.create_bracket_slot(tournament_id, 1, slot_num, a, b)

    round_size = size // 2
    for rnd in range(2, num_rounds + 1):
        round_size //= 2
        for slot_num in range(round_size):
            db.create_bracket_slot(tournament_id, rnd, slot_num, None, None)

    db.set_tournament_bracket_meta(tournament_id, size, num_rounds)

    # Round 1 byes: exactly one side present. Resolve and cascade forward.
    for slot_num in range(size // 2):
        slot = db.get_bracket_slot(tournament_id, 1, slot_num)
        a, b = slot["participant_a_id"], slot["participant_b_id"]
        if (a is None) != (b is None):
            winner = a if a is not None else b
            db.set_slot_bye(tournament_id, 1, slot_num, winner)
            advance_winner(tournament_id, 1, slot_num, winner, num_rounds)


def advance_winner(tournament_id, round_num, slot_num, winner_id, num_rounds=None):
    """Push a slot's winner into the next round. If this was the final
    round, the tournament is complete."""
    if num_rounds is None:
        num_rounds = db.get_tournament(tournament_id)["num_rounds"]

    if round_num >= num_rounds:
        winner = next(p for p in db.list_participants(tournament_id) if p["id"] == winner_id)
        db.finish_tournament(tournament_id, winner["name"])
        return

    next_round = round_num + 1
    next_slot  = slot_num // 2
    side       = "a" if slot_num % 2 == 0 else "b"
    db.set_slot_participant(tournament_id, next_round, next_slot, side, winner_id)


def report_slot_result(slot, winner_side):
    """Called once a live match linked to a bracket slot finishes.
    `slot` is a bracket_slots row (as returned by get_ready_slot_for_court
    or similar — must include tournament_id/round_num/slot_num/
    participant_a_id/participant_b_id). `winner_side` is 'A' or 'B'."""
    winner_id = slot["participant_a_id"] if winner_side == "A" else slot["participant_b_id"]
    db.complete_slot(slot["id"], winner_id)
    advance_winner(slot["tournament_id"], slot["round_num"], slot["slot_num"], winner_id)


def bracket_view(tournament_id):
    """Bracket slots grouped by round, e.g. {1: [...], 2: [...]}."""
    rounds = {}
    for slot in db.get_bracket(tournament_id):
        rounds.setdefault(slot["round_num"], []).append(slot)
    return rounds
