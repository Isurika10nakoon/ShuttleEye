# landing_detection.py
# ═══════════════════════════════════════════════════════════════════
#  FIRST-TOUCH (ground-only) landing detector
#
#  Goal: fire the decision the moment the shuttle first contacts the
#        COURT FLOOR — and only the floor. A racket/player contact
#        must never be reported as a landing.
#
#  Why "did it bounce back up" is the WRONG signal
#  ─────────────────────────────────────────────────
#  A badminton shuttlecock has very high drag and almost no
#  coefficient of restitution: it does not rebound off the court, it
#  just stops (a soft "plop"), possibly with a small roll. A clean
#  rebound after the lowest point of a fall — the shuttle rising
#  again with real speed — is the signature of a RACKET/PLAYER
#  contact, not the floor. So this detector does NOT treat "bounced
#  back up" as ground contact. It only fires on genuine ground
#  contact's real signature: the shuttle was clearly falling, and its
#  speed then collapses to near-zero instead of continuing to move.
#  A racket redirect keeps (or increases) speed, so it fails this
#  check and is correctly ignored — tracking simply continues into
#  whatever the shuttle does next.
#
#  How it works (2-layer pipeline)
#  ────────────────────────────────
#  Layer 1 — Kalman filter (4-state: x, y, vx, vy)
#    Smooths noisy detections and gives clean velocity estimates.
#    Bridges up to GAP_TOLERANCE missing frames via prediction, but
#    the extrapolated point is never treated as a real observation
#    (see update() — it never enters positions/velocities and can
#    never itself trigger a decision).
#
#  Layer 2 — Trajectory phase classifier
#    Each frame is tagged as one of:
#      DESCENDING  → vy > +VY_THRESH  (moving down = increasing y)
#      ASCENDING   → vy < -VY_THRESH  (moving up  = decreasing y)
#      FLAT        → |vy| ≤ VY_THRESH (horizontal / near net)
#
#  Layer 3 — Ground-contact trigger
#    While descending, the lowest y reached so far is tracked as the
#    candidate floor point. Once a CONFIRMED descent (>= MIN_DESCENT
#    frames) ends, if total speed stays below STOP_THRESHOLD for
#    STOP_CONFIRM_FRAMES consecutive frames, that candidate point is
#    reported as the landing. If speed does not collapse (racket
#    redirect), the streak resets and no decision fires — the
#    tracker just keeps watching for the next real descent.
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
    MIN_DESCENT     = 5     # frames of descent before the trigger arms
    GAP_TOLERANCE   = 5     # frames of missing detection to bridge
    COOLDOWN        = 40    # frames locked after a decision

    VY_THRESH        = 1.5  # px/frame — min vy to be "moving"
    STOP_THRESHOLD   = 5.0  # px/frame — "stopped" speed (ground contact)
    STOP_CONFIRM_FRAMES = 2 # consecutive low-speed frames required to fire

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

        # Ground-contact confirmation state
        self._was_confirmed_descent = False  # confirmed descent, prev frame
        self._stop_streak           = 0      # consecutive low-speed frames

        # True only on frames where shuttle_pos came from a real detection,
        # not a gap-bridged Kalman extrapolation. Triggers must never fire
        # on a fabricated position.
        self._last_measured = False

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
            self.positions.append(spos)
            self.velocities.append(vel)
            self._update_phase(vel[1])   # vy
            self._last_measured = True
        else:
            self.gap_count += 1
            self._last_measured = False
            if self.gap_count <= self.GAP_TOLERANCE and self._kf_ready:
                # Keep the Kalman filter's internal clock ticking across the
                # gap so its state stays time-consistent, but do NOT feed the
                # extrapolated point into positions/velocities/phase — a
                # fabricated (never actually seen) position must never arm
                # or fire a landing trigger.
                self._kf_predict_only()
            else:
                # Too many missing frames — reset descent tracker
                self._reset_descent()

    def detect_landing(self):
        """
        Returns first-touch (x, y) pixel coordinates on the FLOOR, or None.
        Call once per frame immediately after update(). Never fires on a
        racket/player redirect — only on a confirmed descent whose speed
        collapses to near-zero (see module docstring for why).
        """
        if (self.cooldown > 0 or len(self.positions) < self.MIN_DESCENT
                or not self._last_measured):
            self._stop_streak = 0
            return None

        vx, vy = self.velocities[-1]
        pos    = self.positions[-1]
        spd    = math.hypot(vx, vy)

        is_descending     = self.phase == Phase.DESCENDING
        confirmed_descent = is_descending and self.descent_count >= self.MIN_DESCENT

        # Track the deepest (highest y) point reached during this descent —
        # that is the actual ground-contact pixel, reported even if the
        # trigger confirms a frame or two later.
        if is_descending and pos[1] > self._floor_y:
            self._floor_y   = pos[1]
            self._floor_pos = pos

        landing = None

        # Only evaluate the stop condition once we've had (or are still
        # inside) a confirmed descent — this is what "just left descent"
        # means without depending on this-frame's phase label directly,
        # since the Kalman-smoothed phase can flip within the same frame.
        if self._was_confirmed_descent or self._stop_streak > 0:
            if spd < self.STOP_THRESHOLD:
                self._stop_streak += 1
            else:
                # Speed did not collapse — a racket redirect, not ground
                # contact. Reject and keep tracking normally.
                self._stop_streak = 0

            if self._stop_streak >= self.STOP_CONFIRM_FRAMES:
                landing = self._floor_pos if self._floor_pos else pos
        else:
            self._stop_streak = 0

        self._was_confirmed_descent = confirmed_descent

        # ── Fire decision ────────────────────────────────────────
        if landing is not None:
            self.cooldown = self.COOLDOWN
            self._reset_descent()
            return landing

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
        self.descent_count          = 0
        self._floor_y               = -1
        self._floor_pos             = None
        self._was_confirmed_descent = False
        self._stop_streak           = 0

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
