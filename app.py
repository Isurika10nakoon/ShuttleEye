# app.py  —  ShuttleEye v5  —  First-Touch + Umpire Dashboard
# ═══════════════════════════════════════════════════════════════════
#  KEYBOARD  (CV window)
#  ───────────────────
#  Q   quit
#  C   re-calibrate  (manual override)
#  P   pause / resume
#  D   toggle debug HUD
#  S   print session stats to console
# ═══════════════════════════════════════════════════════════════════

import cv2
import time
import os

from shuttle_detection  import ShuttleDetector
from landing_detection  import LandingDetector
from line_judge         import LineJudge
from umpire_dashboard   import UmpireDashboard
import calibration

# ── Config ───────────────────────────────────────────────────────
VIDEO_SOURCE = "videos/test1.mp4"  # 0 for webcam
# ─────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {VIDEO_SOURCE}")

# ── Calibration: auto-first, manual fallback ──────────────────────
if not calibration.load_court_points():
    calibration.calibrate(cap)   # tries auto → falls back to manual

# ── Umpire dashboard ──────────────────────────────────────────────
dashboard = UmpireDashboard()
dashboard.start()   # runs in background thread; non-blocking

# ── Core components ───────────────────────────────────────────────
def _on_decision(decision, cm, px):
    """Called by LineJudge immediately on each decision."""
    dashboard.push_decision(decision, cm, px)

shuttle = ShuttleDetector()
lander  = LandingDetector()
judge   = LineJudge(on_decision=_on_decision)

# ── Runtime state ─────────────────────────────────────────────────
prev_time  = time.time()
paused     = False
show_debug = False

print("\n[ShuttleEye v5] Running")
print("  Q=quit  C=calibrate  P=pause  D=debug  S=stats\n")


def _draw_debug(frame, lander):
    info = lander.debug_info()
    lines = [
        f"Phase   : {info['phase']}",
        f"Descent : {info['descent']} frames",
        f"vy      : {info['vy']:+.1f} px/fr",
        f"Speed   : {info['speed']:.1f} px/fr",
        f"Floor-y : {info['floor_y']}",
        f"Cooldown: {info['cooldown']}",
    ]
    x0, y0 = 20, 160
    cv2.rectangle(frame, (x0-5, y0-18), (x0+215, y0+len(lines)*20+4),
                  (20,20,20), -1)
    phase_colors = {
        "DESCENDING": (0,200,255),
        "ASCENDING" : (255,200,0),
        "FLAT"      : (180,180,180),
        "UNKNOWN"   : (120,120,120),
    }
    for i, ln in enumerate(lines):
        col = phase_colors.get(info['phase'], (200,200,200)) if i==0 else (200,200,200)
        cv2.putText(frame, ln, (x0, y0+i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)


# ── Main loop ─────────────────────────────────────────────────────
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("[ShuttleEye] Quit.")
        break

    elif key == ord('p'):
        paused = not paused
        print("[ShuttleEye]", "Paused" if paused else "Resumed")

    elif key == ord('d'):
        show_debug = not show_debug
        print("[ShuttleEye] Debug", "ON" if show_debug else "OFF")

    elif key == ord('s'):
        total, ins, outs = judge.get_stats()
        a, b = dashboard.score_a, dashboard.score_b
        print(f"[Stats] Score {a}–{b}  | Decisions: {total} total, {ins} IN, {outs} OUT")

    elif key == ord('c'):
        # Force re-calibration (manual)
        if os.path.exists(calibration.CONFIG_FILE):
            os.remove(calibration.CONFIG_FILE)
        calibration.calibrate(cap)

    if paused:
        continue

    ret, frame = cap.read()
    if not ret:
        print("[ShuttleEye] End of stream.")
        break

    # 1. Detect shuttle position (bottom-centre of bounding box)
    frame, shuttle_pos = shuttle.detect(frame)

    # 2. Update trajectory / Kalman filter
    lander.update(shuttle_pos)

    # 3. First-touch check → fires _on_decision callback → dashboard
    landing_pt = lander.detect_landing()
    if landing_pt is not None:
        decision = judge.judge(landing_pt)
        cm       = judge.last_landing_cm
        cm_str   = f"({cm[0]:.1f},{cm[1]:.1f})cm" if cm else "?"
        a, b     = dashboard.score_a, dashboard.score_b
        print(f"[FIRST-TOUCH] px={landing_pt}  {cm_str}  → {decision}   {a}–{b}")

    # 4. Draw court overlay
    frame = calibration.draw_court(frame)

    # 5. Lightweight landing marker (no banner — dashboard owns that)
    frame = judge.draw_decision(frame)

    # 6. HUD
    now       = time.time()
    fps       = 1.0 / max(now-prev_time, 1e-6)
    prev_time = now

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, "ShuttleEye v5",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # Score mirror from dashboard
    a, b = dashboard.score_a, dashboard.score_b
    cv2.putText(frame, f"{dashboard.name_a} {a} – {b} {dashboard.name_b}",
                (20, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,220,80), 2)

    if shuttle_pos:
        cm_pos = calibration.pixel_to_real(*shuttle_pos)
        if cm_pos:
            cv2.putText(frame,
                        f"Shuttle: px{shuttle_pos} ({cm_pos[0]:.0f},{cm_pos[1]:.0f})cm",
                        (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

    if show_debug:
        _draw_debug(frame, lander)

    if paused:
        cv2.putText(frame, "PAUSED",
                    (frame.shape[1]//2-80, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0,200,255), 4)

    cv2.imshow("ShuttleEye", frame)

cap.release()
cv2.destroyAllWindows()
dashboard.stop()
