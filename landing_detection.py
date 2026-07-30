# landing_detection.py
# ═══════════════════════════════════════════════════════════════════
#  FIRST-TOUCH landing detector
#
#  Goal: fire the decision at the EXACT frame the shuttle first
#        contacts the ground — not after the bounce, not after it
#        stops, but at impact frame zero.
#
#  How it works (3-layer pipeline)
#  ────────────────────────────────
#  Layer 1 — Kalman filter (4-state: x, y, vx, vy)
#    Smooths noisy detections and gives clean velocity estimates.
#    Bridges up to GAP_TOLERANCE missing frames via prediction.
#
#  Layer 2 — Trajectory phase classifier
#    Each frame is tagged as one of:
#      DESCENDING  → vy > +VY_THRESH  (moving down = increasing y)
#      ASCENDING   → vy < -VY_THRESH  (moving up  = decreasing y)
#      FLAT        → |vy| ≤ VY_THRESH (horizontal / near net)
#
#  Layer 3 — First-touch triggers (in priority order)
#
#    A) IMPACT-FRAME DETECTION (highest priority)
#       During descent, track the bottom-most y reached so far.
#       The moment detected y rises MORE than IMPACT_RISE_PX pixels
#       above that bottom (the shuttle is already bouncing back up),
#       report the saved bottom point — that IS the first-touch frame.
#       This catches the exact impact because:
#         • Before impact: y increases each frame (shuttle falling)
#         • At impact:     y is maximum (lowest physical point)
#         • After impact:  y decreases (shuttle bouncing up)
#       We report as soon as we confirm the direction reversed.
#
#    B) VELOCITY SIGN FLIP (vy crosses zero)
#       Kalman vy flips from positive → negative.
#       We store the position from ONE frame before the flip
#       (that is the ground contact frame, not the first rising frame).
#
#    C) SPEED COLLAPSE after confirmed descent
#       If total speed drops below STOP_THRESHOLD after the shuttle
#       was clearly descending. Catches shuttles that land softly
#       and do not bounce visibly.
#
#  All three triggers report the LOWEST y seen in the recent window,
#  not the current position, so the decision is always at the actual
#  ground level even when triggered one frame late.
# ═══════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import math
from collections import deque
from enum import Enum, auto


class Phase(Enum):
    UNKNOWN    = auto()
    ASCENDING  = auto()
    FLAT       = auto()
    DESCENDING = auto()


class LandingDetector:

    # ── Tuning constants ─────────────────────────────────────────
    HISTORY_LEN     = 30    # smoothed positions kept
    MIN_DESCENT     = 5     # frames of descent before triggers arm
    GAP_TOLERANCE   = 5     # frames of missing detection to bridge
    COOLDOWN        = 40    # frames locked after a decision

    VY_THRESH       = 1.5   # px/frame — min vy to be "moving"
    STOP_THRESHOLD  = 5.0   # px/frame — "stopped" speed
    IMPACT_RISE_PX  = 3     # px rise above floor_y to confirm bounce

    # Kalman noise tuning
    PROC_NOISE      = 5e-3
    MEAS_NOISE      = 5e-2
    # ─────────────────────────────────────────────────────────────

    def __init__(self):
        self._kf       = self._build_kalman()
        self._kf_ready = False

        self.positions  = deque(maxlen=self.HISTORY_LEN)  # (x,y) smoothed
        self.velocities = deque(maxlen=self.HISTORY_LEN)  # (vx,vy) from KF

        self.cooldown      = 0
        self.gap_count     = 0
        self.descent_count = 0   # consecutive descending frames
        self.phase         = Phase.UNKNOWN

        # First-touch tracker
        self._floor_y      = -1    # lowest y seen during current descent
        self._floor_pos    = None  # position at that lowest y
        self._prev_pos     = None  # position one frame ago (for vy sign flip)
        self._prev_vy      = 0.0

    # ── Kalman setup ─────────────────────────────────────────────

    def _build_kalman(self):
        """
        State vector: [x, y, vx, vy]
        Measurement:  [x, y]
        """
        kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        kf.transitionMatrix = np.float32([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ])
        kf.measurementMatrix = np.float32([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * self.PROC_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.MEAS_NOISE
        kf.errorCovPost        = np.eye(4, dtype=np.float32)
        return kf

    def _kf_step(self, pos):
        """Feed measurement, return (smoothed_xy, velocity_xy)."""
        if not self._kf_ready:
            self._kf.statePre = np.float32(
                [pos[0], pos[1], 0, 0]).reshape(4, 1)
            self._kf.statePost = self._kf.statePre.copy()
            self._kf_ready = True

        predicted = self._kf.predict()
        meas      = np.float32([[pos[0]], [pos[1]]])
        corrected = self._kf.correct(meas)

        sx  = float(corrected[0][0])
        sy  = float(corrected[1][0])
        vx  = float(corrected[2][0])
        vy  = float(corrected[3][0])
        return (int(sx), int(sy)), (vx, vy)

    def _kf_predict_only(self):
        """Advance filter one step with no measurement (gap bridging)."""
        if not self._kf_ready:
            return None, None
        pred = self._kf.predict()
        return (int(pred[0][0]), int(pred[1][0])), (float(pred[2][0]), float(pred[3][0]))

    # ── Public API ───────────────────────────────────────────────

    def update(self, shuttle_pos):
        """
        Call once per frame with shuttle pixel position or None.
        Internally updates Kalman filter and phase tracker.
        """
        if self.cooldown > 0:
            self.cooldown -= 1

        if shuttle_pos is not None:
            self.gap_count = 0
            spos, vel = self._kf_step(shuttle_pos)
        else:
            self.gap_count += 1
            if self.gap_count <= self.GAP_TOLERANCE and self._kf_ready:
                spos, vel = self._kf_predict_only()
                if spos is None:
                    return
            else:
                # Too many missing frames — reset descent tracker
                self._reset_descent()
                return

        self.positions.append(spos)
        self.velocities.append(vel)
        self._update_phase(vel[1])   # vy

    def detect_landing(self):
        """
        Returns first-touch (x, y) pixel coordinates, or None.
        Call once per frame immediately after update().
        """
        if self.cooldown > 0 or len(self.positions) < self.MIN_DESCENT:
            return None

        vy  = self.velocities[-1][1]
        pos = self.positions[-1]

        # ── Arm: require confirmed descent ──────────────────────
        if self.phase != Phase.DESCENDING:
            return None

        # Track the deepest (highest y) point during descent
        if pos[1] > self._floor_y:
            self._floor_y   = pos[1]
            self._floor_pos = pos

        landing = None

        # ── Trigger A: impact-frame detection ───────────────────
        # Shuttle has risen IMPACT_RISE_PX above the floor → it bounced
        if (self._floor_pos is not None and
                self._floor_y > 0 and
                pos[1] < self._floor_y - self.IMPACT_RISE_PX):
            landing = self._floor_pos

        # ── Trigger B: Kalman vy sign flip ───────────────────────
        # vy was positive (descending), now negative (ascending)
        elif (self._prev_vy > self.VY_THRESH and
              vy < -self.VY_THRESH):
            # Report the position from one frame ago (last descending frame)
            landing = self._prev_pos if self._prev_pos else self._floor_pos

        # ── Trigger C: speed collapse after descent ──────────────
        elif self.descent_count >= self.MIN_DESCENT:
            spd = math.hypot(*self.velocities[-1])
            if spd < self.STOP_THRESHOLD:
                landing = self._floor_pos if self._floor_pos else pos

        # ── Fire decision ────────────────────────────────────────
        if landing is not None:
            self.cooldown = self.COOLDOWN
            self._reset_descent()
            return landing

        # Store previous for next frame's sign-flip check
        self._prev_pos = pos
        self._prev_vy  = vy
        return None

    # ── Internal helpers ─────────────────────────────────────────

    def _update_phase(self, vy):
        """Classify current phase and maintain descent counter."""
        if vy > self.VY_THRESH:
            if self.phase != Phase.DESCENDING:
                self._reset_descent()          # entering descent fresh
            self.phase = Phase.DESCENDING
            self.descent_count += 1
            # Reset floor tracker when a new descent begins
            if self.descent_count == 1:
                self._floor_y   = -1
                self._floor_pos = None
        elif vy < -self.VY_THRESH:
            self.phase = Phase.ASCENDING
            self.descent_count = 0
        else:
            self.phase = Phase.FLAT
            self.descent_count = 0

    def _reset_descent(self):
        self.descent_count = 0
        self._floor_y      = -1
        self._floor_pos    = None
        self._prev_pos     = None
        self._prev_vy      = 0.0

    # ── Debug info ───────────────────────────────────────────────

    def debug_info(self):
        """Return a dict of current internal state for HUD display."""
        vy  = self.velocities[-1][1] if self.velocities else 0
        spd = math.hypot(*self.velocities[-1]) if self.velocities else 0
        return {
            "phase"   : self.phase.name,
            "descent" : self.descent_count,
            "vy"      : vy,
            "speed"   : spd,
            "floor_y" : self._floor_y,
            "cooldown": self.cooldown,
        }
