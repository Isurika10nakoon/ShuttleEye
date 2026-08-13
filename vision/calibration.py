# calibration.py  ─  ShuttleEye  ─  Boundary-Line Auto-Calibration
# ═══════════════════════════════════════════════════════════════════════
#
#  Works for any camera placement: a corner view where only 1-2 painted
#  white boundary lines are visible, all the way up to a camera placed
#  behind the court seeing the WHOLE court (all 4 outer boundary lines,
#  plus internal lines like service lines / centre line / net, which must
#  be told apart from the actual boundary). This module:
#
#   1. Finds every line automatically — no clicking required in the
#      normal case. It isolates white pixels (court lines are painted
#      white) that also form a thin, locally-bright feature (rejects
#      players' white kit, ad boards, glare — anything broad rather than
#      line-shaped), merges collinear Hough segments into candidate
#      lines, and buckets them as roughly-horizontal ('H') or
#      roughly-vertical ('V') in the frame.
#
#   2. Rejects any candidate whose surroundings aren't a real floor —
#      a genuine boundary line separates two FLAT surfaces (court paint
#      vs. floor/outside). The net's white top tape sits directly above
#      the mesh webbing, which has fine grid texture no matter its exact
#      colour, so it measures as high local edge-energy and gets rejected
#      here — this is what keeps the net out even though it's just as
#      white and line-shaped as a real floor line, and even if perspective
#      would otherwise make it look like an outer boundary line (see 3).
#
#   3. For each orientation there may be several remaining parallel lines
#      in a full-court view (e.g. near baseline, service lines, far
#      baseline are all 'H'). Only the two most extreme — the outermost
#      pair — are kept as the court boundary; everything in between is an
#      internal line and is dropped. A corner view with just one line per
#      orientation keeps that single line, unchanged from before.
#
#   4. Works out which side of EACH kept line is IN and which is OUT —
#      this flips depending on where the camera happens to be fixed, so
#      it can't be hardcoded. The court surface normally fills most of
#      the shot, so whichever side's sampled surface colour is closer to
#      the frame's own dominant colour is taken to be IN; the
#      minority-colour side is OUT.
#
#   5. Combines all kept lines for the final verdict: a point is IN only
#      if it is on the IN side of every one of them (crossing any single
#      boundary puts it OUT). With one line that's a half-plane test; with
#      two it's a court-corner wedge; with all 4 outer lines it's exactly
#      "inside the court rectangle."
#
#   6. Manual line placement is not used — it's too easy to place a line
#      slightly wrong and never notice. If the fast pass (sharpest frames
#      only) finds nothing, it retries with a wider automatic search
#      (every loaded frame, lower Hough sensitivities) before giving up.
#      If that still finds nothing, the system simply runs uncalibrated
#      until a later automatic attempt succeeds (e.g. lighting improves,
#      or the camera framing changes) — pressing C re-triggers detection.
# ═══════════════════════════════════════════════════════════════════════

import cv2
import json
import numpy as np
import os

CONFIG_FILE = "court_config.json"
CONFIG_VERSION = 8   # multi-line schema + per-line measured margin

# Tolerance around a line: a shuttle touching the line counts as IN.
# The real tolerance is derived per-line from the painted line's measured
# thickness (see _refine_centerline) — half the line's own width, since
# the shuttle only needs to graze the physical paint to count. These
# bound that measurement so a bad measurement can't make the call fuzzy
# (too small) or overly lenient (too large) — sharp, not vague.
MIN_MARGIN_PX     = 1.5
MAX_MARGIN_PX     = 6.0
DEFAULT_MARGIN_PX = 2.5   # fallback when thickness can't be measured

# Ignore candidate lines shorter than this fraction of the frame diagonal —
# too short to trust as a real boundary line (could be a shoe, a racket edge).
MIN_LINE_LEN_FRAC = 0.15

# How far off a line (perpendicular, in px) to sample surface colour.
SAMPLE_OFFSET_PX = 25
SAMPLE_PATCH     = 9   # odd side length of each colour-sample patch

# A genuine court boundary separates two flat, unmarked surfaces (court
# paint vs. floor/court paint vs. outside). If either side of a candidate
# line has fine texture — many small edges, like a grid — it isn't really
# bordering a flat surface. This is what specifically excludes the net:
# the white top tape sits directly above the mesh webbing, whose grid
# pattern reads as high local edge-energy no matter its exact colour.
# Same test also screens out lines bordering spectators/foliage/signage.
# Measured the same way frame sharpness already is elsewhere in this file
# (Laplacian variance) but over a wider window, since a texture pattern
# needs more area to show up in than a single colour-sample patch does.
TEXTURE_WINDOW    = 21     # odd side length of the texture-sample window
MAX_SIDE_TEXTURE  = 150.0  # Laplacian variance

# White court line: low colour saturation, high brightness.
WHITE_S_MAX = 60
WHITE_V_MIN = 170

# ── Module state ────────────────────────────────────────────────────────
# List of calibrated lines, each a dict:
#   {'point', 'direction', 'normal', 'endpoints', 'orientation', 'length'}
# 'orientation' is 'H' or 'V'. 0, 1, or 2 lines may be present.
LINES = []


# ═══════════════════════════════════════════════════════════════════════
#  White line isolation
# ═══════════════════════════════════════════════════════════════════════

def _white_line_mask(frame):
    """
    Isolate painted white boundary lines using two complementary cues
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


def _orientation(direction):
    """Buckets a line direction as 'H' (nearer horizontal) or 'V' (nearer vertical)."""
    angle  = abs(np.degrees(np.arctan2(direction[1], direction[0]))) % 180
    dist_h = min(angle, abs(180 - angle))   # distance from 0°/180°
    dist_v = abs(angle - 90)                # distance from 90°
    return 'H' if dist_h < dist_v else 'V'


def _refine_centerline(mask, point, direction, t_min, t_max, search_radius=15):
    """
    A Hough segment fit locates the line approximately, from edge pixels
    that can sit slightly off-centre due to blur/anti-aliasing. For a
    sharp IN/OUT call we need the true painted-line CENTRE, not that
    approximation.

    Walks along the line and, at each cross-section, finds the actual
    white-line pixel run in the mask and takes its midpoint — the real
    centreline — plus its width (the physical line thickness). Fitting a
    new line through these midpoints gives a sub-pixel-accurate centre,
    and the measured thickness lets the margin match the real line
    instead of a guessed constant.

    Returns (point, direction, t_min, t_max, thickness_px) or None if too
    few cross-sections could be measured to trust the refinement.
    """
    normal = np.array([-direction[1], direction[0]])
    h, w = mask.shape[:2]
    offs = np.arange(-search_radius, search_radius + 1)

    centers, widths = [], []
    n_samples = max(20, int((t_max - t_min) / 10))
    for t in np.linspace(t_min, t_max, n_samples):
        base = point + direction * t
        xs = np.round(base[0] + normal[0]*offs).astype(int)
        ys = np.round(base[1] + normal[1]*offs).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not valid.any():
            continue
        on = np.where(valid & (mask[np.clip(ys,0,h-1), np.clip(xs,0,w-1)] > 0))[0]
        if len(on) == 0:
            continue
        # Midpoint (and width) of the contiguous run closest to the centre
        mid_off = (offs[on].min() + offs[on].max()) / 2.0
        width   = float(offs[on].max() - offs[on].min() + 1)
        centers.append(base + normal*mid_off)
        widths.append(width)

    if len(centers) < 5:
        return None

    centers = np.array(centers, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(centers, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    new_dir = np.array([vx, vy], dtype=np.float64)
    new_dir /= np.linalg.norm(new_dir)
    new_point = np.array([x0, y0], dtype=np.float64)

    tt = (centers.astype(np.float64) - new_point) @ new_dir
    thickness = float(np.median(widths))
    return new_point, new_dir, float(tt.min()), float(tt.max()), thickness


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


def _sample_side_color(lab, gray, line_mask, point, direction, normal, sign, t_values, offset):
    """
    Returns (median_color, texture) for the sampled side, or (None, None)
    if no valid sample points were found. `texture` is the mean Laplacian
    variance across a wider window at each sample point — near-zero for a
    flat painted surface, high for something with fine structure like net
    mesh (same edge-energy measure used for frame sharpness elsewhere in
    this file, just applied locally instead of to the whole frame).
    """
    half      = SAMPLE_PATCH // 2
    tex_half  = TEXTURE_WINDOW // 2
    h, w = line_mask.shape[:2]
    all_px, tex_scores = [], []
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
        all_px.append(px.reshape(-1, 3))

        if cx-tex_half >= 0 and cy-tex_half >= 0 and cx+tex_half < w and cy+tex_half < h:
            tex_patch = gray[cy-tex_half:cy+tex_half+1, cx-tex_half:cx+tex_half+1]
            tex_scores.append(cv2.Laplacian(tex_patch, cv2.CV_64F).var())
    if not all_px:
        return None, None
    all_px  = np.concatenate(all_px, axis=0)
    median  = np.median(all_px, axis=0)
    # Median, not mean: a sample point near a line intersection (e.g. a
    # court corner, where a perpendicular line crosses through the texture
    # window) is a legitimate single-sample outlier, not evidence the
    # whole side is textured. Median ignores one or two such outliers but
    # still flags a side where MOST samples show real texture (mesh).
    texture = float(np.median(tex_scores)) if tex_scores else 0.0
    return median, texture


def _determine_in_side(frame, line_mask, point, direction, t_min, t_max):
    """
    Returns the unit normal vector pointing toward the IN side, or None
    if it couldn't be determined — either sample points fell outside the
    frame, or one side is too visually non-uniform to be a real court-
    adjacent surface (e.g. the net's mesh, spectators, foliage), in which
    case this candidate line is rejected outright rather than risked as a
    boundary.
    """
    normal   = np.array([-direction[1], direction[0]])
    t_values = np.linspace(t_min, t_max, 9)[1:-1]   # skip noisy extreme ends
    lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    color_pos, tex_pos = _sample_side_color(lab, gray, line_mask, point, direction, normal, +1, t_values, SAMPLE_OFFSET_PX)
    color_neg, tex_neg = _sample_side_color(lab, gray, line_mask, point, direction, normal, -1, t_values, SAMPLE_OFFSET_PX)
    dominant  = _dominant_frame_color(frame, line_mask)
    if color_pos is None or color_neg is None or dominant is None:
        return None

    if tex_pos > MAX_SIDE_TEXTURE or tex_neg > MAX_SIDE_TEXTURE:
        print(f"[Calibration] Rejected a candidate line: one side has fine "
              f"texture, not a flat court surface (texture={tex_pos:.0f}/{tex_neg:.0f}, "
              f"limit={MAX_SIDE_TEXTURE}) — likely the net's mesh or background "
              f"clutter, not a floor boundary line.")
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

def _fit_candidates(frame, hough_threshold):
    """
    Returns every sufficiently-long, colour-resolvable line found in this
    frame at this Hough sensitivity — of any orientation, and there may be
    several per orientation (e.g. every horizontal line visible in a
    full-court shot: both baselines, service lines, the net). Callers
    dedupe and pick the outer boundary from this full set.
    """
    h, w = frame.shape[:2]
    diag = float(np.hypot(w, h))
    mask = _white_line_mask(frame)
    min_len = max(30, int(w * 0.08))

    segs = _detect_segments(mask, hough_threshold, min_len)
    if not segs:
        return []

    clusters = _cluster_segments(segs)
    candidates = []
    for c in clusters:
        if c['length'] < diag * MIN_LINE_LEN_FRAC:
            continue
        point, direction, t_min, t_max, endpoints = _fit_line(c['segs'])
        length = c['length']

        # Sharpen the fit onto the true painted-line centre + measure its
        # real thickness, so the IN/OUT edge sits exactly on the line.
        refined = _refine_centerline(mask, point, direction, t_min, t_max)
        if refined is not None:
            point, direction, t_min, t_max, thickness = refined
            margin = min(MAX_MARGIN_PX, max(MIN_MARGIN_PX, thickness / 2.0))
            p1 = point + direction*t_min
            p2 = point + direction*t_max
            endpoints = (tuple(int(v) for v in p1), tuple(int(v) for v in p2))
            # The refined extent (validated cross-section by cross-section)
            # is the real length — not the raw Hough cluster's segment sum,
            # which can overstate it. Using the stale value here corrupted
            # length-based comparisons downstream (dedup, boundary select).
            length = t_max - t_min
        else:
            margin = DEFAULT_MARGIN_PX

        in_normal = _determine_in_side(frame, mask, point, direction, t_min, t_max)
        if in_normal is None:
            continue
        candidates.append({
            'point': point, 'direction': direction, 'normal': in_normal,
            'endpoints': endpoints, 'length': length, 'margin': margin,
            'orientation': _orientation(direction),
        })
    return candidates


def _line_position(line):
    """
    A scalar position along the axis that separates PARALLEL lines of the
    same orientation — y for horizontal lines (near baseline vs. far
    baseline), x for vertical lines (left sideline vs. right sideline).
    Used to tell distinct parallel lines apart and find the outermost
    pair.

    Deliberately uses a fixed image-space axis rather than each line's
    own normal: cv2.fitLine's direction sign is arbitrary per line, so a
    per-line-derived axis can point opposite ways for two lines of the
    same orientation and scramble the ordering between them.
    """
    return float(line['point'][1] if line['orientation'] == 'H' else line['point'][0])


def _dedupe_lines(candidates, pos_tol=20):
    """
    Merges candidates that are really the same physical line seen more
    than once (different frame/threshold attempts), keeping the longest
    representative of each. Candidates only merge within the same
    orientation and a similar _line_position.
    """
    groups = []   # [{'orientation', 'pos', 'best'}]
    for cand in candidates:
        pos = _line_position(cand)
        merged = False
        for g in groups:
            if g['orientation'] == cand['orientation'] and abs(pos - g['pos']) < pos_tol:
                if cand['length'] > g['best']['length']:
                    g['best'], g['pos'] = cand, pos
                merged = True
                break
        if not merged:
            groups.append({'orientation': cand['orientation'], 'pos': pos, 'best': cand})
    return [g['best'] for g in groups]


def _select_boundary_lines(candidates, frame_shape):
    """
    Keeps only the boundary of THIS court per orientation.

    With 1 line it's the boundary as-is (corner/single-line view). With 2,
    they're the outer pair (e.g. near/far baseline) — kept as-is.

    With 3+ parallel lines, this is very likely a facility with several
    courts SHARING one floor (common for badminton halls — courts laid
    out side by side reusing the same lines/space). Those extra lines
    aren't internal markings of one court, they can be another court's
    sideline entirely — taking the outermost pair in that case would span
    multiple courts' width, not just this one's. Instead, bracket the
    frame centre: keep the nearest line on each side of the frame's
    midline, since a dedicated court camera is framed on the court
    actually in play, which should sit roughly centred in the shot.
    """
    h, w = frame_shape[:2]
    center = {'H': h / 2.0, 'V': w / 2.0}

    result = []
    for o in ('H', 'V'):
        group = [c for c in candidates if c['orientation'] == o]
        if not group:
            continue
        group.sort(key=_line_position)

        if len(group) <= 2:
            result.append(group[0])
            if len(group) > 1:
                result.append(group[-1])
            continue

        c0    = center[o]
        left  = [g for g in group if _line_position(g) <= c0]
        right = [g for g in group if _line_position(g) >  c0]
        if left:
            result.append(max(left, key=_line_position))    # nearest to centre, left/above
        if right:
            result.append(min(right, key=_line_position))   # nearest to centre, right/below
    return result


def auto_calibrate(frames, exhaustive=False):
    """
    Try to automatically find the court boundary line(s) and each one's
    IN side — works for a camera that only sees one corner (1-2 lines) as
    well as a camera placed behind the court seeing the whole thing (up to
    4 outer boundary lines, with any internal lines correctly ignored).

    Scans frames at several Hough sensitivities, collects every line seen,
    deduplicates repeated detections of the same physical line, then keeps
    this court's boundary line(s) per orientation (see
    _select_boundary_lines for how that's told apart from a shared
    multi-court floor's other lines).

    By default only the sharpest few frames are tried (fast — the normal
    case). With exhaustive=True, every loaded frame is tried at a wider,
    more sensitive range of Hough thresholds — a slower search used only
    as a second automatic attempt when the fast pass finds nothing, so
    manual line placement is never needed.
    """
    print("[AutoCalib] Scanning frames for sharpest …")
    order = sorted(range(len(frames)), key=lambda i: -_sharpness(frames[i]))
    top_frames = order if exhaustive else order[:min(5, len(order))]
    thresholds = (60, 45, 32, 22, 15, 10) if exhaustive else (60, 45, 32, 22)
    print(f"[AutoCalib] Trying {len(top_frames)} frame(s)"
          f"{' (exhaustive)' if exhaustive else ': ' + str(top_frames)}")

    all_candidates = []
    for idx in top_frames:
        frame = frames[idx]
        for hough_threshold in thresholds:
            all_candidates.extend(_fit_candidates(frame, hough_threshold))

    if not all_candidates:
        print("[AutoCalib] No boundary line found across attempts.")
        return []

    deduped = _dedupe_lines(all_candidates)
    lines   = _select_boundary_lines(deduped, frames[0].shape)

    for l in lines:
        print(f"[AutoCalib] {l['orientation']} boundary line "
              f"length={l['length']:.0f}px endpoints={l['endpoints']}")
    return lines


# ═══════════════════════════════════════════════════════════════════════
#  Public entry points
# ═══════════════════════════════════════════════════════════════════════

def _apply(lines):
    global LINES
    LINES = lines


def calibrate(cap):
    """
    Fully automatic, always — no manual line placement. Detects the white
    boundary line(s) and which side of each is IN with no user interaction.

    Tries a fast pass (sharpest few frames) first. If that finds nothing,
    retries with an exhaustive pass (every loaded frame, wider Hough
    sensitivity range) rather than asking for clicks. If even that finds
    nothing, the system is left uncalibrated — IN/OUT calls are
    unavailable until a later automatic attempt succeeds (e.g. via the
    app's recalibrate key), never via manual line placement.
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

    lines = auto_calibrate(frames)

    if not lines:
        print("[Calibration] Fast pass found nothing — retrying with a wider "
              "automatic search (every frame, more Hough sensitivities) …")
        lines = auto_calibrate(frames, exhaustive=True)

    if lines:
        _apply(lines)
        _save()
        print(f"[Calibration] Auto-calibration complete "
              f"({len(lines)} line(s): {[l['orientation'] for l in lines]})\n")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True

    print("[Calibration] Automatic detection could not find any boundary line "
          "in this footage. Continuing uncalibrated — IN/OUT calls are "
          "unavailable until recalibration succeeds.\n")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return False


# ═══════════════════════════════════════════════════════════════════════
#  Runtime classification
# ═══════════════════════════════════════════════════════════════════════

def is_calibrated():
    return len(LINES) > 0


def classify_side(px, py):
    """
    Returns "IN" | "OUT" | None (not calibrated).
    A point is IN only if it's on the IN side of every detected line —
    crossing any one boundary (horizontal or vertical) puts it OUT.
    """
    if not LINES:
        return None
    pt = np.array([px, py], dtype=np.float64)
    for line in LINES:
        signed = float((pt - line['point']) @ line['normal'])
        margin = line.get('margin', DEFAULT_MARGIN_PX)
        if signed < -margin:
            return "OUT"
    return "IN"


def line_offsets(px, py):
    """
    Returns (perp_dist_px, along_line_px) for the *binding* line — the one
    with the smallest (most restrictive) signed distance, i.e. whichever
    line either caused an OUT verdict or is nearest if still IN. Returns
    None if not calibrated.
    """
    if not LINES:
        return None
    pt = np.array([px, py], dtype=np.float64)
    best = None
    for line in LINES:
        v     = pt - line['point']
        perp  = float(v @ line['normal'])
        along = float(v @ line['direction'])
        if best is None or perp < best[0]:
            best = (perp, along)
    return best


def draw_court(frame):
    """Draws each boundary line and its IN/OUT side labels."""
    for line in LINES:
        p1, p2 = line['endpoints']
        cv2.line(frame, p1, p2, (0, 255, 255), 3)

        mid = np.array([(p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0])
        in_pt  = tuple(int(v) for v in (mid + line['normal']*40))
        out_pt = tuple(int(v) for v in (mid - line['normal']*40))

        cv2.putText(frame, "IN",  in_pt,  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "OUT", out_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 255), 2)
    return frame


# ═══════════════════════════════════════════════════════════════════════
#  Persistence
# ═══════════════════════════════════════════════════════════════════════

def _save():
    data = {
        "lines": [
            {
                "point":       line['point'].tolist(),
                "direction":   line['direction'].tolist(),
                "normal":      line['normal'].tolist(),
                "endpoints":   [list(line['endpoints'][0]), list(line['endpoints'][1])],
                "orientation": line['orientation'],
                "margin":      line.get('margin', DEFAULT_MARGIN_PX),
            }
            for line in LINES
        ],
        "version": CONFIG_VERSION,
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

    global LINES
    LINES = [
        {
            'point':       np.array(l['point'], dtype=np.float64),
            'direction':   np.array(l['direction'], dtype=np.float64),
            'normal':      np.array(l['normal'], dtype=np.float64),
            'endpoints':   (tuple(l['endpoints'][0]), tuple(l['endpoints'][1])),
            'orientation': l['orientation'],
            'margin':      l.get('margin', DEFAULT_MARGIN_PX),
        }
        for l in data["lines"]
    ]
    print(f"[Calibration] Loaded {len(LINES)} boundary line(s) from config.")
    return True
