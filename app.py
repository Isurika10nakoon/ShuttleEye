# app.py  —  ShuttleEye  —  First-Touch + Umpire Dashboard
# ═══════════════════════════════════════════════════════════════════
#  KEYBOARD  (CV window)
#  ───────────────────
#  Q   quit
#  C   re-run automatic calibration
#  M   manual calibration (click lines by hand)
#  P   pause / resume
#  D   toggle debug HUD
#  F   toggle fullscreen
#  S   print session stats to console
# ═══════════════════════════════════════════════════════════════════

import cv2
import time
import os
import sys

from vision.shuttle_detection  import ShuttleDetector
from vision.landing_detection  import LandingDetector
from vision.line_judge         import LineJudge
from desktop.umpire_dashboard  import UmpireDashboard
from desktop.login_window      import run_login_flow
from web.web_dashboard         import run_web_dashboard
from vision import calibration
from core import auth
from core import db

# ── Config ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(PROJECT_ROOT, "videos", "test4.mp4")  # 0 for webcam
COURT_NAME   = os.environ.get("COURT_NAME", "Court 1")  # shown to the
                                                          # admin's multi-court view
# ─────────────────────────────────────────────────────────────────

# ── Database (creates tables + seeds default accounts on first run) ─
db.init_db()
auth.ensure_default_accounts()

# ── Login (blocks in main thread until a valid session is chosen) ─
session = run_login_flow()
if session is None:
    print("[ShuttleEye] Login cancelled. Exiting.")
    sys.exit(0)
umpire_name, role = session
print(f"[ShuttleEye] Logged in as '{umpire_name}' ({role}) -- {COURT_NAME}")

cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {VIDEO_SOURCE}")

# ── Calibration: fully automatic, no manual line placement ────────
if not calibration.load_court_points():
    calibration.calibrate(cap)

# ── Umpire dashboard ──────────────────────────────────────────────
dashboard = UmpireDashboard(umpire_name=umpire_name, role=role, court_name=COURT_NAME)
dashboard.start()   # runs in background thread; non-blocking

# ── Remote dashboard — lets the umpire's own phone/tablet/laptop view
#    and control the match over the network, separate from this PC ─────
web_url = run_web_dashboard(dashboard)
dashboard.web_url = web_url  # lets the desktop window's "Spectator Board" button open it
print(f"[ShuttleEye] Remote umpire dashboard: {web_url}")
print( "[ShuttleEye] Open that address on the umpire's device (same Wi-Fi/network) to log in.")
print(f"[ShuttleEye] Public spectator scoreboard (no login): {web_url}/board")

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
fullscreen = False

WINDOW_NAME = "ShuttleEye"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print("\n[ShuttleEye] Running")
print("  Q=quit  C=calibrate  M=manual-calibrate  P=pause  D=debug  F=fullscreen  S=stats\n")


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

    elif key == ord('f'):
        fullscreen = not fullscreen
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        print("[ShuttleEye] Fullscreen", "ON" if fullscreen else "OFF")

    elif key == ord('s'):
        total, ins, outs = judge.get_stats()
        a, b = dashboard.score_a, dashboard.score_b
        print(f"[Stats] Score {a}-{b}  | Decisions: {total} total, {ins} IN, {outs} OUT")

    elif key == ord('c'):
        # Force a fresh automatic recalibration
        if os.path.exists(calibration.CONFIG_FILE):
            os.remove(calibration.CONFIG_FILE)
        calibration.calibrate(cap)

    elif key == ord('m'):
        # On-demand manual calibration -- click lines by hand when
        # automatic detection isn't reliable for this footage.
        calibration.calibrate_manual(cap)

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
        print(f"[FIRST-TOUCH] px={landing_pt}  {cm_str}  -> {decision}   {a}-{b}")

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
    cv2.putText(frame, "ShuttleEye",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # Player names (no live score here — the umpire dashboard owns that)
    cv2.putText(frame, f"{dashboard.name_a} vs {dashboard.name_b}",
                (20, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,220,80), 2)

    if show_debug:
        _draw_debug(frame, lander)

    if paused:
        cv2.putText(frame, "PAUSED",
                    (frame.shape[1]//2-80, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0,200,255), 4)

    cv2.imshow(WINDOW_NAME, frame)

cap.release()
cv2.destroyAllWindows()
dashboard.stop()
