# line_judge.py  ─  ShuttleEye v5
# ═══════════════════════════════════════════════════════════════════
#  Judges IN / OUT in real-world court space (centimetres).
#
#  CHANGES FROM v4
#  ───────────────
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
    MARGIN_CM        = 2.0   # cm tolerance — anything within 2 cm of line → IN

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
        if calibration.H is None:
            print("[LineJudge] WARNING: not calibrated.")
            return None

        cm = calibration.pixel_to_real(*landing_px)
        if cm is None:
            return None
        rx, ry = cm

        decision = self._classify(rx, ry)

        self.last_decision   = decision
        self.last_landing_px = landing_px
        self.last_landing_cm = (rx, ry)
        self.display_counter = self.DISPLAY_DURATION
        self.history.append((decision, (rx, ry)))
        self._pending.append((decision, (rx, ry), landing_px))

        if self.on_decision:
            self.on_decision(decision, (rx, ry), landing_px)

        return decision

    def _classify(self, rx, ry):
        m = self.MARGIN_CM
        W = calibration.COURT_W_CM
        L = calibration.COURT_L_CM
        return "IN" if (-m <= rx <= W+m and -m <= ry <= L+m) else "OUT"

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
                rx, ry = self.last_landing_cm
                label += f"  {rx:.0f},{ry:.0f}cm"
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
