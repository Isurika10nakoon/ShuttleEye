# calibration.py  ─  ShuttleEye v5  ─  Auto + Manual Calibration
# ═══════════════════════════════════════════════════════════════════════
#
#  AUTO CALIBRATION  (no clicks needed)
#  ──────────────────────────────────────
#  1. Scans up to 300 frames, picks the sharpest (Laplacian variance)
#  2. Runs Canny + HoughLines to find all court lines
#  3. Computes every pairwise line intersection
#  4. Clusters nearby intersections (within 20 px) → candidate corners
#  5. Picks the 4 candidates that best form a rectangle matching the
#     expected court aspect ratio (W:L = 610:1340)
#  6. Orders them TL/TR/BR/BL and builds the homography
#  7. If reprojection error < AUTO_ERR_THRESHOLD → auto-confirms
#     Otherwise falls back to the interactive manual UI
#
#  MANUAL CALIBRATION  (fallback / re-calibrate)
#  ───────────────────────────────────────────────
#  • Click to place 4 court corners (TL, TR, BR, BL)
#  • Drag any placed point to fine-tune
#  • Scroll wheel nudges ±1 px (Ctrl = ×10)
#  • Z  toggle snap-to-line-intersection assist
#  • A/D step one frame back / forward
#  • B  jump to auto-detected best (sharpest) frame
#  • R  reset all points
#  • ENTER confirm & save
#  • ESC cancel
# ═══════════════════════════════════════════════════════════════════════

import cv2
import json
import numpy as np
import os
from itertools import combinations

CONFIG_FILE = "court_config.json"

# ── Official badminton court dimensions (cm) ─────────────────────────
COURT_W_CM  = 610
COURT_L_CM  = 1340
HALF_L      = 670
SINGLES_OFF = 46

REAL_CORNERS = np.float32([
    [0,          0         ],
    [COURT_W_CM, 0         ],
    [COURT_W_CM, COURT_L_CM],
    [0,          COURT_L_CM],
])

# Auto-calibration threshold: accept if mean reprojection error < this
AUTO_ERR_THRESHOLD = 8.0   # pixels

# ── Module globals ────────────────────────────────────────────────────
COURT_POINTS = []
H            = None
H_INV        = None


# ═══════════════════════════════════════════════════════════════════════
#  Homography
# ═══════════════════════════════════════════════════════════════════════

def _build_homography(pixel_corners):
    global H, H_INV, COURT_POINTS
    COURT_POINTS = [tuple(int(v) for v in p) for p in pixel_corners]
    src    = np.float32(pixel_corners)
    H,    _ = cv2.findHomography(src, REAL_CORNERS, cv2.RANSAC, 5.0)
    H_INV, _ = cv2.findHomography(REAL_CORNERS, src, cv2.RANSAC, 5.0)


def pixel_to_real(px, py):
    if H is None:
        return None
    out = cv2.perspectiveTransform(np.float32([[[px, py]]]), H)
    return float(out[0][0][0]), float(out[0][0][1])


def real_to_pixel(rx, ry):
    if H_INV is None:
        return None
    out = cv2.perspectiveTransform(np.float32([[[rx, ry]]]), H_INV)
    return int(out[0][0][0]), int(out[0][0][1])


def _reprojection_error(pixel_corners):
    """Mean pixel distance between placed corners and back-projected real corners."""
    if H is None or H_INV is None:
        return 999.0, [999.0]*4
    total = 0.0
    per_pt = []
    for i, pc in enumerate(pixel_corners):
        rc = REAL_CORNERS[i]
        pp = real_to_pixel(rc[0], rc[1])
        if pp:
            e = np.hypot(pp[0]-pc[0], pp[1]-pc[1])
            total += e
            per_pt.append(e)
        else:
            per_pt.append(999.0)
    return total / max(len(pixel_corners), 1), per_pt


# ═══════════════════════════════════════════════════════════════════════
#  Frame sharpness
# ═══════════════════════════════════════════════════════════════════════

def _sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _best_frame(frames):
    scores = [(_sharpness(f), i) for i, f in enumerate(frames)]
    scores.sort(reverse=True)
    return scores[0][1], scores[0][0]


# ═══════════════════════════════════════════════════════════════════════
#  Line detection & intersection helpers
# ═══════════════════════════════════════════════════════════════════════

def _detect_lines(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=70)
    return lines if lines is not None else []


def _line_intersection(r1, t1, r2, t2):
    ct1, st1 = np.cos(t1), np.sin(t1)
    ct2, st2 = np.cos(t2), np.sin(t2)
    det = ct1*st2 - ct2*st1
    if abs(det) < 1e-6:
        return None
    x = (r1*st2 - r2*st1) / det
    y = (r2*ct1 - r1*ct2) / det
    return x, y


def _all_intersections(hough_lines, frame_shape):
    h, w = frame_shape[:2]
    pts = []
    for i in range(len(hough_lines)):
        for j in range(i+1, len(hough_lines)):
            r1, t1 = hough_lines[i][0]
            r2, t2 = hough_lines[j][0]
            angle_diff = abs(t1-t2) % np.pi
            if angle_diff < np.radians(15) or angle_diff > np.radians(165):
                continue
            pt = _line_intersection(r1, t1, r2, t2)
            if pt and -50 < pt[0] < w+50 and -50 < pt[1] < h+50:
                pts.append(pt)
    return pts


def _cluster_intersections(pts, radius=20):
    """
    Merge nearby intersection points into cluster centroids.
    Returns list of (cx, cy) centroid points.
    """
    if not pts:
        return []
    visited = [False]*len(pts)
    clusters = []
    for i, p in enumerate(pts):
        if visited[i]:
            continue
        group = [p]
        visited[i] = True
        for j in range(i+1, len(pts)):
            if not visited[j]:
                if np.hypot(pts[j][0]-p[0], pts[j][1]-p[1]) < radius:
                    group.append(pts[j])
                    visited[j] = True
        cx = sum(g[0] for g in group)/len(group)
        cy = sum(g[1] for g in group)/len(group)
        clusters.append((cx, cy))
    return clusters


def _sort_corners_tl_tr_br_bl(pts):
    """Sort 4 (x,y) points into [TL, TR, BR, BL] order."""
    pts = list(pts)
    cx  = sum(p[0] for p in pts)/4
    cy  = sum(p[1] for p in pts)/4
    def quad(p):
        l = p[0] < cx;  t = p[1] < cy
        return 0 if (t and l) else 1 if (t and not l) else 2 if (not t and not l) else 3
    ordered = [None]*4
    for p in pts:
        q = quad(p)
        if ordered[q] is None:
            ordered[q] = p
    if any(v is None for v in ordered):
        return pts  # degenerate, return as-is
    return ordered


def _score_quad(pts, frame_shape):
    """
    Score a set of 4 candidate corners for being a valid court:
    - prefers larger area
    - penalises deviation from expected court aspect ratio
    - penalises non-convexity
    Returns a score (higher = better); -inf if invalid.
    """
    h, w = frame_shape[:2]
    arr  = np.float32(pts)
    area = cv2.contourArea(arr)
    if area < 5000:
        return -np.inf

    # Convexity check
    hull = cv2.convexHull(arr)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1:
        return -np.inf
    convexity = area / hull_area
    if convexity < 0.85:
        return -np.inf

    # Aspect ratio of bounding box
    xs = [p[0] for p in pts];  ys = [p[1] for p in pts]
    bw = max(xs)-min(xs);      bh = max(ys)-min(ys)
    if bh < 1:
        return -np.inf

    expected_ratio = COURT_W_CM / COURT_L_CM   # ~0.455
    actual_ratio   = bw / bh
    ratio_error    = abs(actual_ratio - expected_ratio)

    # Score: large area, close aspect ratio, convex
    score = area * convexity / (1 + ratio_error * 5)
    return score


# ═══════════════════════════════════════════════════════════════════════
#  AUTO CALIBRATION
# ═══════════════════════════════════════════════════════════════════════

def auto_calibrate(frames):
    """
    Try to automatically detect the 4 court corners.

    Returns (corners_list, error) where:
      corners_list  = [[x,y], [x,y], [x,y], [x,y]] in TL/TR/BR/BL order
      error         = mean reprojection error in pixels (999 if failed)

    The caller should check error < AUTO_ERR_THRESHOLD to decide
    whether to accept or fall back to manual.
    """
    print("[AutoCalib] Scanning frames for sharpest …")
    best_idx, best_score = _best_frame(frames)
    print(f"[AutoCalib] Best frame #{best_idx}  sharpness={best_score:.0f}")

    frame = frames[best_idx]
    h, w  = frame.shape[:2]

    # ── Detect lines ───────────────────────────────────────────────
    hough = _detect_lines(frame)
    if len(hough) < 4:
        print("[AutoCalib] Too few lines detected. Falling back to manual.")
        return None, 999.0

    # ── All intersections → cluster → candidate corners ────────────
    raw_pts  = _all_intersections(hough, frame.shape)
    clusters = _cluster_intersections(raw_pts, radius=20)
    print(f"[AutoCalib] {len(raw_pts)} intersections → {len(clusters)} clusters")

    if len(clusters) < 4:
        print("[AutoCalib] Not enough cluster points. Falling back to manual.")
        return None, 999.0

    # ── Try every combination of 4 clusters, pick best quad ────────
    best_score_quad = -np.inf
    best_corners    = None

    # Limit search to top 30 clusters by distance from frame centre
    cx_f, cy_f = w/2, h/2
    clusters_sorted = sorted(clusters, key=lambda p: np.hypot(p[0]-cx_f, p[1]-cy_f))
    candidates = clusters_sorted[:min(30, len(clusters))]

    for combo in combinations(candidates, 4):
        s = _score_quad(combo, frame.shape)
        if s > best_score_quad:
            best_score_quad = s
            best_corners    = combo

    if best_corners is None:
        print("[AutoCalib] No valid quad found. Falling back to manual.")
        return None, 999.0

    # ── Sort TL/TR/BR/BL, build homography, measure error ─────────
    corners_sorted = _sort_corners_tl_tr_br_bl(best_corners)
    _build_homography(corners_sorted)
    err, _ = _reprojection_error(corners_sorted)

    print(f"[AutoCalib] Reprojection error = {err:.2f} px  (threshold={AUTO_ERR_THRESHOLD})")

    if err < AUTO_ERR_THRESHOLD:
        print(f"[AutoCalib] ✓ Auto-calibration accepted.")
        return corners_sorted, err
    else:
        print(f"[AutoCalib] ✗ Error too high — falling back to manual.")
        return corners_sorted, err   # return as initial hint for manual UI


# ═══════════════════════════════════════════════════════════════════════
#  Court line definitions (real-world cm)
# ═══════════════════════════════════════════════════════════════════════

def _court_lines_real():
    W, L, HL, S = COURT_W_CM, COURT_L_CM, HALF_L, SINGLES_OFF
    cx = W / 2
    return [
        (0, 0,    W, 0,    'outer'),
        (W, 0,    W, L,    'outer'),
        (W, L,    0, L,    'outer'),
        (0, L,    0, 0,    'outer'),
        (S,   0,   S,   L,   'inner'),
        (W-S, 0,   W-S, L,   'inner'),
        (0, HL,   W, HL,   'net'),
        (0, HL-160, W, HL-160, 'inner'),
        (0, HL+160, W, HL+160, 'inner'),
        (0,   80,   W,   80,   'inner'),
        (0, L-80,   W, L-80,   'inner'),
        (cx, HL,     cx, HL-160, 'inner'),
        (cx, HL,     cx, HL+160, 'inner'),
    ]


# ═══════════════════════════════════════════════════════════════════════
#  Drawing helpers
# ═══════════════════════════════════════════════════════════════════════

_LINE_COLOR = {'outer': (255,255,255), 'inner': (160,160,160), 'net': (40,210,210)}
_LINE_WIDTH = {'outer': 3,             'inner': 1,              'net': 3           }

POINT_LABELS      = ["1·TL", "2·TR", "3·BR", "4·BL"]
POINT_COLORS      = [(0,255,255), (0,200,255), (0,150,255), (0,100,255)]
POINT_COLORS_GRAB = [(0,255,100),(0,200,100),(0,150,100),(0,100,100)]


def _err_color(e):
    if e < 1.5:   return (0, 255,  60)
    if e < 3.0:   return (0, 165, 255)
    return (0, 60, 255)


def _draw_court_lines(frame):
    if H_INV is None:
        return
    for (x1r, y1r, x2r, y2r, style) in _court_lines_real():
        p1 = real_to_pixel(x1r, y1r)
        p2 = real_to_pixel(x2r, y2r)
        if p1 and p2:
            cv2.line(frame, p1, p2, _LINE_COLOR[style], _LINE_WIDTH[style])


def _draw_points(frame, pts, grabbed_idx, per_err=None):
    for i, pt in enumerate(pts):
        col = POINT_COLORS_GRAB[i] if i == grabbed_idx else POINT_COLORS[i]
        ec  = _err_color(per_err[i]) if per_err else col
        cv2.circle(frame, pt, 14, (0,0,0), -1)
        cv2.circle(frame, pt, 13, ec, 2)
        cv2.circle(frame, pt, 10, col, 2)
        cv2.circle(frame, pt,  3, col, -1)
        lx, ly = pt[0]+15, pt[1]-12
        cv2.putText(frame, POINT_LABELS[i], (lx+1, ly+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(frame, POINT_LABELS[i], (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(frame, f"({pt[0]},{pt[1]})", (lx+1, ly+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 3)
        cv2.putText(frame, f"({pt[0]},{pt[1]})", (lx, ly+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)


def _draw_loupe(frame, base_frame, mouse_xy, zoom=2.5, size=190):
    mx, my = mouse_xy
    h, w   = base_frame.shape[:2]
    half   = int(size / zoom / 2)
    x1 = max(0, mx-half);  x2 = min(w, mx+half)
    y1 = max(0, my-half);  y2 = min(h, my+half)
    if x2-x1 < 4 or y2-y1 < 4:
        return
    crop   = base_frame[y1:y2, x1:x2].copy()
    zoomed = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
    cxz, cyz = size//2, size//2
    cv2.line(zoomed, (cxz,0), (cxz,size), (0,255,80), 1)
    cv2.line(zoomed, (0,cyz), (size,cyz), (0,255,80), 1)
    cv2.circle(zoomed, (cxz,cyz), 5, (0,255,80), 1)
    px = w-size-10;  py = 10
    cv2.rectangle(frame, (px-2,py-2), (px+size+2,py+size+2), (180,220,255), 2)
    frame[py:py+size, px:px+size] = zoomed
    cv2.putText(frame, f"ZOOM {zoom:.1f}x  ({mx},{my})",
                (px, py+size+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)


def _snap_to_intersection(x, y, intersections, radius=30):
    best_d, best_pt = radius, (x, y)
    for ix, iy in intersections:
        d = np.hypot(ix-x, iy-y)
        if d < best_d:
            best_d  = d
            best_pt = (int(round(ix)), int(round(iy)))
    return best_pt


def _draw_hud(frame, pts, grabbed_idx, snap_on, frame_idx,
              total_frames, sharpness, best_idx, err_total, per_err):
    h, w = frame.shape[:2]
    bar_y = h - 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    instructions = [
        "Drag=move", "Scroll=nudge Y", "Shift+Scroll=X",
        f"Z=snap({'ON' if snap_on else 'OFF'})",
        "A/D=frame  B=best", "R=reset  ENTER=save  ESC=cancel",
    ]
    cv2.putText(frame, "  |  ".join(instructions),
                (10, bar_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190,190,190), 1)

    placed = len(pts)
    if placed < 4:
        status = f"Place corner {placed+1}/4  →  {POINT_LABELS[placed]}"
        scol   = (0, 200, 255)
    else:
        star   = " ★" if frame_idx == best_idx else ""
        status = (f"Reprojection: {err_total:.2f} px  "
                  f"| Frame {frame_idx}/{total_frames-1}{star}  "
                  f"| Sharpness {sharpness:.0f}")
        scol   = _err_color(err_total)

    cv2.putText(frame, status, (10, bar_y+48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, scol, 1)

    if placed == 4 and per_err:
        ex = 10
        for i, e in enumerate(per_err):
            lbl = f"{POINT_LABELS[i]}:{e:.1f}px"
            cv2.putText(frame, lbl, (ex, bar_y+70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, _err_color(e), 1)
            ex += cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]+20

    if grabbed_idx >= 0:
        cv2.putText(frame, f"Moving: {POINT_LABELS[grabbed_idx]}",
                    (10, bar_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    POINT_COLORS_GRAB[grabbed_idx], 2)


# ═══════════════════════════════════════════════════════════════════════
#  Calibration state machine (manual)
# ═══════════════════════════════════════════════════════════════════════

class _CalibState:
    GRAB_RADIUS = 28

    def __init__(self, base_frame, hough_lines, initial_pts=None):
        self.base          = base_frame.copy()
        self.hough         = hough_lines
        self.intersections = _all_intersections(hough_lines, base_frame.shape)
        self.clusters      = _cluster_intersections(self.intersections)
        self.pts           = [list(p) for p in initial_pts] if initial_pts else []
        self.grabbed       = -1
        self.mouse         = (0, 0)
        self.snap          = True
        self.frame_idx     = 0
        self.sharpness     = 0.0

    def on_mouse(self, event, x, y, flags):
        self.mouse = (x, y)
        ctrl       = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        mult       = 10 if ctrl else 1

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, p in enumerate(self.pts):
                if np.hypot(p[0]-x, p[1]-y) < self.GRAB_RADIUS:
                    self.grabbed = i;  return
            if len(self.pts) < 4:
                sx, sy = self._snap(x, y)
                self.pts.append([sx, sy])
                if len(self.pts) == 4:
                    self.pts = [list(p) for p in
                                _sort_corners_tl_tr_br_bl(self.pts)]
                self._rebuild()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.grabbed >= 0:
                self.pts[self.grabbed] = [x, y];  self._rebuild()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.grabbed >= 0:
                sx, sy = self._snap(x, y)
                self.pts[self.grabbed] = [sx, sy]
                if len(self.pts) == 4:
                    self.pts = [list(p) for p in
                                _sort_corners_tl_tr_br_bl(self.pts)]
                self._rebuild()
                self.grabbed = -1

        elif event == cv2.EVENT_MOUSEWHEEL:
            idx = self.grabbed if self.grabbed >= 0 else self._nearest(x, y)
            if 0 <= idx < len(self.pts):
                self.pts[idx][1] += (-mult if flags > 0 else mult)
                self._rebuild()

        elif event == cv2.EVENT_MOUSEHWHEEL:
            idx = self.grabbed if self.grabbed >= 0 else self._nearest(x, y)
            if 0 <= idx < len(self.pts):
                self.pts[idx][0] += (mult if flags > 0 else -mult)
                self._rebuild()

    def _snap(self, x, y):
        if self.snap and self.clusters:
            return _snap_to_intersection(x, y, self.clusters, radius=30)
        return x, y

    def _nearest(self, x, y):
        best_d, best_i = self.GRAB_RADIUS*2, -1
        for i, p in enumerate(self.pts):
            d = np.hypot(p[0]-x, p[1]-y)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _rebuild(self):
        if len(self.pts) == 4:
            _build_homography(self.pts)

    def error(self):
        if len(self.pts) < 4:
            return 999.0, [999.0]*4
        return _reprojection_error(self.pts)

    def update_frame(self, frame):
        self.base          = frame.copy()
        self.hough         = _detect_lines(frame)
        self.intersections = _all_intersections(self.hough, frame.shape)
        self.clusters      = _cluster_intersections(self.intersections)
        self.sharpness     = _sharpness(frame)

    def render(self, zoom, total_frames, best_idx):
        frame   = self.base.copy()
        pts_t   = [tuple(p) for p in self.pts]
        err, pe = self.error()

        if len(self.pts) == 4:
            _draw_court_lines(frame)
        _draw_points(frame, pts_t, self.grabbed, pe if len(self.pts)==4 else None)
        _draw_loupe(frame, self.base, self.mouse, zoom=zoom)
        _draw_hud(frame, pts_t, self.grabbed, self.snap,
                  self.frame_idx, total_frames, self.sharpness, best_idx,
                  err, pe)
        return frame


# ═══════════════════════════════════════════════════════════════════════
#  Public calibrate() entry point
# ═══════════════════════════════════════════════════════════════════════

def calibrate(cap):
    """
    First tries auto-calibration.
    If that achieves reprojection error < AUTO_ERR_THRESHOLD, saves and returns True.
    Otherwise opens the interactive manual UI (with auto result as starting hint).
    """
    print("\n[Calibration v5] ──────────────────────────────────────────")
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

    best_idx, best_sharp = _best_frame(frames)

    # ── Try auto ─────────────────────────────────────────────────
    auto_corners, auto_err = auto_calibrate(frames)

    if auto_corners is not None and auto_err < AUTO_ERR_THRESHOLD:
        _save()
        print(f"[Calibration] Auto-calibration complete  (error={auto_err:.2f} px)\n")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True

    # ── Fall back to manual UI ────────────────────────────────────
    print("[Calibration] Opening manual UI …")
    print("  Drag corners. Z=snap  A/D=frame  B=best  ENTER=save  ESC=cancel\n")

    frame_idx = best_idx
    zoom      = 2.5

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 1280, 760)

    # Use auto corners as initial hint if available (even if error was high)
    state = _CalibState(
        frames[frame_idx],
        _detect_lines(frames[frame_idx]),
        initial_pts=auto_corners
    )
    state.frame_idx  = frame_idx
    state.sharpness  = best_sharp

    cv2.setMouseCallback("Calibration",
                         lambda e, x, y, f, p: state.on_mouse(e, x, y, f))

    confirmed = False
    while True:
        frame = state.render(zoom, len(frames), best_idx)
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(16) & 0xFF

        if key == 13:    # ENTER
            if len(state.pts) == 4:
                confirmed = True;  break
            else:
                print(f"  [!] Need 4 corners, only {len(state.pts)} placed.")
        elif key == 27:  print("[Calibration] Cancelled."); break
        elif key == ord('r'):
            state.pts = [];  state.grabbed = -1
            print("[Calibration] Points reset.")
        elif key == ord('z'):
            state.snap = not state.snap
            print(f"[Calibration] Snap {'ON' if state.snap else 'OFF'}")
        elif key == ord('b'):
            frame_idx = best_idx;  state.frame_idx = frame_idx
            state.update_frame(frames[frame_idx])
        elif key == ord('a'):
            frame_idx = max(0, frame_idx-1);  state.frame_idx = frame_idx
            state.update_frame(frames[frame_idx])
        elif key == ord('d'):
            frame_idx = min(len(frames)-1, frame_idx+1);  state.frame_idx = frame_idx
            state.update_frame(frames[frame_idx])
        elif key in (ord('+'), ord('=')):  zoom = min(zoom+0.5, 8.0)
        elif key in (ord('-'), ord('_')):  zoom = max(zoom-0.5, 1.5)

    cv2.destroyWindow("Calibration")

    if confirmed:
        _build_homography(state.pts)
        _save()
        err, pe = _reprojection_error(state.pts)
        print(f"[Calibration] Saved. Error={err:.2f} px  per-pt={[f'{e:.2f}' for e in pe]}\n")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return confirmed


# ═══════════════════════════════════════════════════════════════════════
#  Persistence
# ═══════════════════════════════════════════════════════════════════════

def _save():
    with open(CONFIG_FILE, "w") as f:
        json.dump({"court_points": COURT_POINTS, "version": 5}, f, indent=2)
    print(f"[Calibration] Config saved → {CONFIG_FILE}")


def load_court_points():
    if not os.path.exists(CONFIG_FILE):
        return False
    with open(CONFIG_FILE) as f:
        data = json.load(f)
    pts = [tuple(p) for p in data["court_points"]]
    _build_homography(pts)
    err, _ = _reprojection_error(pts)
    print(f"[Calibration] Loaded  (reproj error={err:.2f} px)")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  Runtime drawing
# ═══════════════════════════════════════════════════════════════════════

def draw_court(frame):
    _draw_court_lines(frame)
    return frame
