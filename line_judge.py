# line_judge.py  ─  ShuttleEye
# ═══════════════════════════════════════════════════════════════════
#  Judges IN / OUT relative to the single detected boundary line
#  (see calibration.py — classify_side() owns the IN/OUT geometry
#  and the margin tolerance for "touching the line counts as IN").
#
#  • All drawing / banner code removed — the Umpire Dashboard owns UI
#  • Decisions are pushed to an optional callback (on_decision) AND to
#    an internal deque so the dashboard can poll them
#  • draw_decision() kept as a thin overlay so the CV window still
#    shows a landing marker + small verdict text (no giant banner)
# ═══════════════════════════════════════════════════════════════════

import cv2
import numpy as np
from collections import deque
import calibration


class LineJudge:

    DISPLAY_DURATION = 60    # frames the CV overlay stays visible

    def __init__(self, on_decision=None):
        """
        on_decision: optional callable(decision, landing_cm, landing_px)
                     called immediately when a decision is made.
                     Useful for pushing events to the dashboard.
        """
        self.on_decision     = on_decision
        self.last_decision   = ""
        self.last_landing_px = None
        self.last_landing_cm = None
        self.display_counter = 0
        self.history         = []          # list of (decision, cm_point)
        self._pending        = deque()     # unread decisions for polling

    # ── Core judgement ───────────────────────────────────────────

    def judge(self, landing_px):
        """
        Args:  landing_px — (x, y) in pixel coordinates
        Returns: "IN" | "OUT" | None
        """
        if landing_px is None:
            return None
        decision = calibration.classify_side(*landing_px)
        if decision is None:
            print("[LineJudge] WARNING: not calibrated.")
            return None

        # Signed pixel offsets from the line (perp, along) — kept in the
        # same "cm" slot the dashboard expects for display purposes only;
        # there's no real-world scale for a single line without a second
        # reference dimension, so these are pixel offsets, not centimetres.
        offs = calibration.line_offsets(*landing_px) or (0.0, 0.0)

        self.last_decision   = decision
        self.last_landing_px = landing_px
        self.last_landing_cm = offs
        self.display_counter = self.DISPLAY_DURATION
        self.history.append((decision, offs))
        self._pending.append((decision, offs, landing_px))

        if self.on_decision:
            self.on_decision(decision, offs, landing_px)

        return decision

    # ── Poll interface (for dashboard) ───────────────────────────

    def poll(self):
        """
        Return next unread (decision, cm, px) tuple, or None.
        Non-destructive: call repeatedly each frame.
        """
        if self._pending:
            return self._pending.popleft()
        return None

    # ── Lightweight CV overlay ────────────────────────────────────

    def draw_decision(self, frame):
        """
        Draws a landing marker and small verdict text on the CV frame.
        No large banner — the dashboard shows the full verdict.
        """
        if self.display_counter <= 0 or not self.last_decision:
            return frame

        self.display_counter -= 1
        fade  = self.display_counter / self.DISPLAY_DURATION

        color = (0, 220, 0) if self.last_decision == "IN" else (0, 60, 220)

        if self.last_landing_px:
            px = self.last_landing_px
            # Cross-hair marker
            cv2.drawMarker(frame, px, color,
                           markerType=cv2.MARKER_CROSS,
                           markerSize=30, thickness=2)
            cv2.circle(frame, px, 14, color, 2)

            # Small verdict tag next to marker
            label = self.last_decision
            if self.last_landing_cm:
                dist = self.last_landing_cm[0]
                label += f"  {dist:+.0f}px"
            cv2.putText(frame, label, (px[0]+18, px[1]-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4)
            cv2.putText(frame, label, (px[0]+18, px[1]-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        return frame

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self):
        """Return (total, in_count, out_count)."""
        total = len(self.history)
        ins   = sum(1 for d, _ in self.history if d == "IN")
        return total, ins, total - ins
