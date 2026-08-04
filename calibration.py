# calibration.py  ─  ShuttleEye  ─  Single Boundary-Line Auto-Calibration
# ═══════════════════════════════════════════════════════════════════════
#
#  The camera is assumed to be pointed at ONE painted white boundary line
#  (not the whole court), e.g. a dedicated line-judge camera. This module:
#
#   1. Finds that line automatically — no clicking required in the normal
#      case. It isolates white pixels (court lines are painted white) that
#      also form a thin, locally-bright feature (rejects players' white
#      kit, ad boards, glare — anything broad rather than line-shaped),
#      then merges collinear Hough segments into a single best line.
#
#   2. Works out which side of the line is IN and which is OUT — this
#      flips depending on where the camera happens to be fixed, so it
#      can't be hardcoded. The court surface normally fills most of a
#      line-judge shot, so whichever side's sampled surface colour is
#      closer to the frame's own dominant colour is taken to be IN; the
#      minority-colour side is OUT.
#
#   3. Only if no line can be found at all (e.g. it's simply not visible
#      in this footage) does it fall back to a minimal manual UI — click
#      2 points on the line. Even then, which side is IN is still worked
#      out automatically from colour, not asked for.
# ═══════════════════════════════════════════════════════════════════════

import cv2
import json
import numpy as np
import os

CONFIG_FILE = "court_config.json"
CONFIG_VERSION = 6   # single-line schema (incompatible with old 4-corner configs)

# Tolerance around the line: a shuttle touching the line counts as IN.
MARGIN_PX = 4.0

# Ignore candidate lines shorter than this fraction of the frame diagonal —
# too short to trust as THE boundary line (could be a shoe, a racket edge).
MIN_LINE_LEN_FRAC = 0.15

# How far off the line (perpendicular, in px) to sample surface colour.
SAMPLE_OFFSET_PX = 25
SAMPLE_PATCH     = 9   # odd side length of each colour-sample patch

# White court line: low colour saturation, high brightness.
WHITE_S_MAX = 60
WHITE_V_MIN = 170

# ── Module state (single boundary line) ───────────────────────────────
LINE_POINT     = None   # np.array([x, y]) — a point on the line
LINE_DIR       = None   # np.array([dx, dy]) — unit vector along the line
LINE_NORMAL    = None   # np.array([nx, ny]) — unit vector pointing to the IN side
LINE_ENDPOINTS = None   # ((x1,y1), (x2,y2)) — visible extent, for drawing


# ═══════════════════════════════════════════════════════════════════════
#  White line isolation
# ═══════════════════════════════════════════════════════════════════════

def _white_line_mask(frame):
    """
    Isolate the painted white boundary line using two complementary cues
    combined with AND, so each cancels the other's false positives:

      1. Colour   — the line is white: low saturation, high brightness.
                     Alone, this would also match white shirts/shoes, sky,
                     or bright ad boards.
      2. Contrast — a top-hat transform keeps only features that are
                     narrow and brighter than their immediate surroundings.
                     Alone, this would also match skin, reflections, or any
                     other locally-bright edge regardless of colour.

    A pixel that is both "white" and "a thin bright feature" is, on a
    badminton court, a boundary line.
    """
    hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_color = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (180, WHITE_S_MAX, 255))

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat  = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    _, contrast_mask = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_and(white_color, contrast_mask)


def _sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ═══════════════════════════════════════════════════════════════════════
#  Segment detection & collinear clustering
# ═══════════════════════════════════════════════════════════════════════

def _detect_segments(mask, hough_threshold, min_len):
    edges = cv2.Canny(mask, 40, 120, apertureSize=3)
    segs  = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold,
                             minLineLength=min_len, maxLineGap=25)
    if segs is None:
        return []
    return [tuple(int(v) for v in s[0]) for s in segs]


def _segment_rho_theta(seg):
    x1, y1, x2, y2 = seg
    angle = np.arctan2(y2-y1, x2-x1)
    theta = (angle + np.pi/2) % np.pi        # perpendicular (normal) angle
    mx, my = (x1+x2)/2.0, (y1+y2)/2.0
    rho    = mx*np.cos(theta) + my*np.sin(theta)
    length = float(np.hypot(x2-x1, y2-y1))
    return theta, rho, length


def _cluster_segments(segs, angle_tol=np.radians(4), rho_tol=15):
    """Group collinear segments (same angle + same perpendicular offset)."""
    clusters = []   # each: {theta_sum, rho_sum, count, length, segs}
    for seg in segs:
        theta, rho, length = _segment_rho_theta(seg)
        placed = False
        for c in clusters:
            avg_theta = c['theta_sum'] / c['count']
            avg_rho   = c['rho_sum']   / c['count']
            dtheta = min(abs(theta-avg_theta), np.pi-abs(theta-avg_theta))
            if dtheta < angle_tol and abs(rho-avg_rho) < rho_tol:
                c['theta_sum'] += theta
                c['rho_sum']   += rho
                c['count']     += 1
                c['length']    += length
                c['segs'].append(seg)
                placed = True
                break
        if not placed:
            clusters.append({'theta_sum': theta, 'rho_sum': rho,
                              'count': 1, 'length': length, 'segs': [seg]})
    return clusters


def _fit_line(segs):
    """Least-squares line through all segment endpoints; returns the
    point/direction plus the visible extent (min/max projection)."""
    pts = []
    for (x1, y1, x2, y2) in segs:
        pts.append((x1, y1))
        pts.append((x2, y2))
    pts = np.array(pts, dtype=np.float32)

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    point = np.array([x0, y0], dtype=np.float64)

    t = (pts.astype(np.float64) - point) @ direction
    t_min, t_max = float(t.min()), float(t.max())
    p1 = point + direction*t_min
    p2 = point + direction*t_max
    endpoints = (tuple(int(v) for v in p1), tuple(int(v) for v in p2))
    return point, direction, t_min, t_max, endpoints


# ═══════════════════════════════════════════════════════════════════════
#  IN/OUT side detection (colour based — camera-placement independent)
# ═══════════════════════════════════════════════════════════════════════

def _dominant_frame_color(frame, exclude_mask):
    """
    Robust dominant colour of the frame, in Lab space, as a stand-in for
    'the court surface colour' — a line-judge shot is assumed to show
    mostly court, so the median non-line pixel is the court colour.
    """
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    valid = exclude_mask == 0
    if valid.sum() < 100:
        return None
    return np.median(lab[valid].reshape(-1, 3), axis=0)


def _sample_side_color(lab, line_mask, point, direction, normal, sign, t_values, offset):
    half = SAMPLE_PATCH // 2
    h, w = line_mask.shape[:2]
    samples = []
    for t in t_values:
        base = point + direction*t + normal*sign*offset
        cx, cy = int(base[0]), int(base[1])
        if cx-half < 0 or cy-half < 0 or cx+half >= w or cy+half >= h:
            continue
        patch      = lab[cy-half:cy+half+1, cx-half:cx+half+1]
        mask_patch = line_mask[cy-half:cy+half+1, cx-half:cx+half+1]
        px = patch[mask_patch == 0]
        if len(px) == 0:
            continue
        samples.append(np.median(px.reshape(-1, 3), axis=0))
    if not samples:
        return None
    return np.median(np.array(samples), axis=0)


def _determine_in_side(frame, line_mask, point, direction, t_min, t_max):
    """
    Returns the unit normal vector pointing toward the IN side, or None
    if it couldn't be determined (e.g. sample points fall outside frame).
    """
    normal   = np.array([-direction[1], direction[0]])
    t_values = np.linspace(t_min, t_max, 9)[1:-1]   # skip noisy extreme ends
    lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    color_pos = _sample_side_color(lab, line_mask, point, direction, normal, +1, t_values, SAMPLE_OFFSET_PX)
    color_neg = _sample_side_color(lab, line_mask, point, direction, normal, -1, t_values, SAMPLE_OFFSET_PX)
    dominant  = _dominant_frame_color(frame, line_mask)
    if color_pos is None or color_neg is None or dominant is None:
        return None

    d_pos = np.linalg.norm(color_pos - dominant)
    d_neg = np.linalg.norm(color_neg - dominant)

    if abs(d_pos - d_neg) < 3.0:
        print(f"[Calibration] WARNING: IN/OUT sides look colour-similar "
              f"(d_pos={d_pos:.1f}, d_neg={d_neg:.1f}) — result may be unreliable.")

    return normal if d_pos <= d_neg else -normal


# ═══════════════════════════════════════════════════════════════════════
#  Auto calibration
# ═══════════════════════════════════════════════════════════════════════

def _score_and_fit(frame, hough_threshold):
    h, w = frame.shape[:2]
    diag = float(np.hypot(w, h))
    mask = _white_line_mask(frame)
    min_len = max(30, int(w * 0.08))

    segs = _detect_segments(mask, hough_threshold, min_len)
    if not segs:
        return None

    clusters = _cluster_segments(segs)
    if not clusters:
        return None

    best = max(clusters, key=lambda c: c['length'])
    if best['length'] < diag * MIN_LINE_LEN_FRAC:
        return None

    point, direction, t_min, t_max, endpoints = _fit_line(best['segs'])
    in_normal = _determine_in_side(frame, mask, point, direction, t_min, t_max)
    if in_normal is None:
        return None

    return {
        'point': point, 'direction': direction,
        't_min': t_min, 't_max': t_max, 'endpoints': endpoints,
        'in_normal': in_normal, 'length': best['length'],
    }


def auto_calibrate(frames):
    """
    Try to automatically find the boundary line and its IN side.
    Tries the sharpest frames at several Hough sensitivities, keeping the
    longest (most confident) line found. Returns a result dict or None.
    """
    print("[AutoCalib] Scanning frames for sharpest …")
    order = sorted(range(len(frames)), key=lambda i: -_sharpness(frames[i]))
    top_frames = order[:min(5, len(order))]
    print(f"[AutoCalib] Trying {len(top_frames)} sharpest frames: {top_frames}")

    best = None
    for idx in top_frames:
        frame = frames[idx]
        w = frame.shape[1]
        for hough_threshold in (60, 45, 32, 22):
            result = _score_and_fit(frame, hough_threshold)
            if result is None:
                continue
            if best is None or result['length'] > best['length']:
                best = result
            if best['length'] > w * 0.6:
                break
        if best is not None and best['length'] > w * 0.6:
            break

    if best is None:
        print("[AutoCalib] No boundary line found across attempts.")
    else:
        print(f"[AutoCalib] Best line length={best['length']:.0f}px "
              f"endpoints={best['endpoints']}")
    return best


# ═══════════════════════════════════════════════════════════════════════
#  Manual fallback (last resort — only the line position is clicked;
#  the IN side is still determined automatically from colour)
# ═══════════════════════════════════════════════════════════════════════

def _manual_line_ui(frame):
    pts = []
    win = "Calibration (click 2 points on the boundary line)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 760)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.setMouseCallback(win, on_mouse)

    confirmed = False
    while True:
        disp = frame.copy()
        for p in pts:
            cv2.circle(disp, p, 6, (0, 255, 255), -1)
        if len(pts) == 2:
            cv2.line(disp, pts[0], pts[1], (0, 255, 255), 2)
        cv2.putText(disp, "Click 2 points on the line.  ENTER=confirm  R=reset  ESC=cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(16) & 0xFF

        if key == 13 and len(pts) == 2:
            confirmed = True
            break
        elif key == 27:
            break
        elif key == ord('r'):
            pts = []

    cv2.destroyWindow(win)
    if not confirmed:
        return None

    point = np.array(pts[0], dtype=np.float64)
    end   = np.array(pts[1], dtype=np.float64)
    direction = end - point
    length    = float(np.linalg.norm(direction))
    direction /= length
    endpoints = (pts[0], pts[1])

    mask = _white_line_mask(frame)
    in_normal = _determine_in_side(frame, mask, point, direction, 0.0, length)
    if in_normal is None:
        in_normal = np.array([-direction[1], direction[0]])
        print("[Calibration] WARNING: could not auto-detect the IN side by "
              "colour — defaulted; verify the IN/OUT overlay looks correct.")

    return {
        'point': point, 'direction': direction,
        't_min': 0.0, 't_max': length, 'endpoints': endpoints,
        'in_normal': in_normal, 'length': length,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Public entry points
# ═══════════════════════════════════════════════════════════════════════

def _apply(result):
    global LINE_POINT, LINE_DIR, LINE_NORMAL, LINE_ENDPOINTS
    LINE_POINT     = result['point']
    LINE_DIR       = result['direction']
    LINE_NORMAL    = result['in_normal']
    LINE_ENDPOINTS = result['endpoints']


def calibrate(cap):
    """
    Fully automatic: detects the white boundary line and which side of it
    is IN, with no user interaction. Only if no line can be found at all
    in any sampled frame does a minimal manual UI open (click 2 points on
    the line) — and even then, the IN side is still worked out from
    colour automatically, never asked for.
    """
    print("\n[Calibration] ──────────────────────────────────────────")
    print("  Loading frames …")

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while len(frames) < 300:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    if not frames:
        raise RuntimeError("No frames available for calibration.")

    result = auto_calibrate(frames)

    if result is not None:
        _apply(result)
        _save()
        print(f"[Calibration] Auto-calibration complete "
              f"(line length={result['length']:.0f}px)\n")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True

    # ── Last resort: no line detected in any frame at all ──────────
    print("[Calibration] Could not detect any boundary line automatically.")
    print("[Calibration] Opening manual UI …\n")

    best_idx = int(np.argmax([_sharpness(f) for f in frames]))
    result = _manual_line_ui(frames[best_idx])

    ok = result is not None
    if ok:
        _apply(result)
        _save()
        print(f"[Calibration] Manual line saved (length={result['length']:.0f}px)\n")
    else:
        print("[Calibration] Cancelled.\n")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return ok


# ═══════════════════════════════════════════════════════════════════════
#  Runtime classification
# ═══════════════════════════════════════════════════════════════════════

def is_calibrated():
    return LINE_POINT is not None


def classify_side(px, py):
    """Returns "IN" | "OUT" | None (not calibrated)."""
    if LINE_POINT is None:
        return None
    v = np.array([px, py], dtype=np.float64) - LINE_POINT
    signed = float(v @ LINE_NORMAL)
    return "IN" if signed >= -MARGIN_PX else "OUT"


def line_offsets(px, py):
    """Returns (perp_dist_from_line_px, along_line_px), signed toward IN, or None."""
    if LINE_POINT is None:
        return None
    v = np.array([px, py], dtype=np.float64) - LINE_POINT
    perp  = float(v @ LINE_NORMAL)
    along = float(v @ LINE_DIR)
    return perp, along


def draw_court(frame):
    """Draws the boundary line and IN/OUT side labels."""
    if LINE_POINT is None:
        return frame

    p1, p2 = LINE_ENDPOINTS
    cv2.line(frame, p1, p2, (0, 255, 255), 3)

    mid = np.array([(p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0])
    in_pt  = tuple(int(v) for v in (mid + LINE_NORMAL*40))
    out_pt = tuple(int(v) for v in (mid - LINE_NORMAL*40))

    cv2.putText(frame, "IN",  in_pt,  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),   2)
    cv2.putText(frame, "OUT", out_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 255), 2)
    return frame


# ═══════════════════════════════════════════════════════════════════════
#  Persistence
# ═══════════════════════════════════════════════════════════════════════

def _save():
    data = {
        "point":     LINE_POINT.tolist(),
        "direction": LINE_DIR.tolist(),
        "in_normal": LINE_NORMAL.tolist(),
        "endpoints": [list(LINE_ENDPOINTS[0]), list(LINE_ENDPOINTS[1])],
        "version":   CONFIG_VERSION,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Calibration] Config saved → {CONFIG_FILE}")


def load_court_points():
    if not os.path.exists(CONFIG_FILE):
        return False
    with open(CONFIG_FILE) as f:
        data = json.load(f)
    if data.get("version") != CONFIG_VERSION:
        print("[Calibration] Config is an older/incompatible format — recalibrating.")
        return False

    global LINE_POINT, LINE_DIR, LINE_NORMAL, LINE_ENDPOINTS
    LINE_POINT     = np.array(data["point"], dtype=np.float64)
    LINE_DIR       = np.array(data["direction"], dtype=np.float64)
    LINE_NORMAL    = np.array(data["in_normal"], dtype=np.float64)
    LINE_ENDPOINTS = (tuple(data["endpoints"][0]), tuple(data["endpoints"][1]))
    print("[Calibration] Loaded boundary line from config.")
    return True
