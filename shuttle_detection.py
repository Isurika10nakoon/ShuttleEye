# shuttle_detection.py
# ═══════════════════════════════════════════════════════════════════
#  Shuttle detector with bottom-edge reporting
#
#  KEY CHANGE for first-touch accuracy:
#  Instead of reporting the bounding-box CENTRE as the shuttle
#  position, we report the BOTTOM-CENTRE of the bounding box.
#
#  Why: the shuttle's base (cork / rubber tip) is what physically
#  contacts the ground. The centre of the blob is always several
#  pixels above the actual contact point. Reporting bottom-centre
#  removes that constant offset so the landing y-coordinate
#  matches the real floor contact pixel exactly.
# ═══════════════════════════════════════════════════════════════════

import cv2
import numpy as np
from collections import deque


class ShuttleDetector:

    TRAIL_LEN  = 24
    MIN_AREA   = 12
    MAX_AREA   = 700
    MIN_ASPECT = 0.2
    MAX_ASPECT = 4.5

    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=40,
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.trail  = deque(maxlen=self.TRAIL_LEN)

    def detect(self, frame):
        # ── Background subtraction + clean mask ──────────────────
        mask = self.backSub.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        shuttle_pos = None   # bottom-centre (x, y_bottom)
        shuttle_ctr = None   # centre (for trail)
        best_score  = -1
        best_box    = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.MIN_AREA < area < self.MAX_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if not (self.MIN_ASPECT < aspect < self.MAX_ASPECT):
                continue

            roundness = 4 * np.pi * area / max(cv2.arcLength(cnt, True) ** 2, 1)
            score     = area * roundness

            if score > best_score:
                best_score = score
                best_box   = (x, y, w, h)

                # ── Bottom-centre as reported position ───────────
                # x_centre, y_bottom — the cork tip contact point
                shuttle_pos = (x + w // 2, y + h)
                shuttle_ctr = (x + w // 2, y + h // 2)

        # ── Trail (use centre for smooth visual path) ─────────────
        if shuttle_ctr:
            self.trail.append(shuttle_ctr)

        for i in range(1, len(self.trail)):
            alpha = i / len(self.trail)
            color = (int(255 * alpha), int(180 * alpha), 0)
            cv2.line(frame, self.trail[i-1], self.trail[i], color, 2)

        # ── Draw detection ────────────────────────────────────────
        if best_box:
            x, y, w, h = best_box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Centre dot (blue)
            cv2.circle(frame, shuttle_ctr, 5, (0, 0, 255), -1)

            # Bottom contact point (red — this is what lands_detection uses)
            cv2.circle(frame, shuttle_pos, 4, (0, 80, 255), -1)
            cv2.line(frame,
                     (shuttle_pos[0]-6, shuttle_pos[1]),
                     (shuttle_pos[0]+6, shuttle_pos[1]),
                     (0, 80, 255), 1)

        return frame, shuttle_pos
