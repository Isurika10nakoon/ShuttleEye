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
#   2. Rejects any candidate that isn't painted CONTINUOUSLY along its own
#      length. A real boundary line is one continuous stripe of paint (bar
#      the odd scuff or shadow); the net's white top tape is not — strung
#      between two posts, it sags, self-shadows, and lets the mesh and
#      background show through at an angle, so it comes apart into shorter
#      fragments in the same white-pixel mask a real line stays solid in.
#      Measured by walking the fitted line and checking how much of it is
#      actually covered by a contiguous run of mask pixels (small gaps —
#      a shadow, a shoe — are bridged; large ones aren't). This is what
#      keeps the net out even though it's just as white and line-shaped as
#      a real floor line, and even if perspective would otherwise make it
#      look like an outer boundary line (see 3). (An earlier version tried
#      to tell the net apart by local pixel texture — high edge-energy
#      near the mesh — but on a real glossy indoor floor under uneven gym
#      lighting, glare/reflections/wood-grain measured just as textured as
#      the net itself, sometimes more so, making that signal useless; this
#      continuity check is about the line's own paint, not its surroundings.)
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
#   6. calibrate() — the normal entry point — is automatic-only. If the
#      fast pass (sharpest frames only) finds nothing, it retries with a
#      wider automatic search (every loaded frame, lower Hough
#      sensitivities) before giving up. If that still finds nothing, the
#      system simply runs uncalibrated until a later automatic attempt
#      succeeds (e.g. lighting improves) — pressing C re-triggers it.
#
#      calibrate_manual() is a SEPARATE, on-demand entry point — a click
#      UI for placing (or correcting) lines by hand, for footage where
#      automatic detection isn't reliable (a shared multi-court floor,
#      heavy glare). It's never opened by the system on its own, only
#      when explicitly invoked (e.g. a keybinding in the app). Even a
#      manually-placed line still gets its IN side worked out from colour
#      automatically, and its centre/thickness sharpened the same way an
#      auto-detected line's is.
# ═══════════════════════════════════════════════════════════════════════

import cv2
import json
import numpy as np
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(_PROJECT_ROOT, "court_config.json")
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

# A genuine boundary line is painted as one continuous stripe. The net's
# top tape, strung between two posts, comes apart into shorter fragments
# in the white-pixel mask (sag, self-shadowing, mesh/background showing
# through) — this is what specifically excludes it. Checked by walking
# the fitted line and measuring the longest contiguous run of mask hits,
# as a fraction of the line's own length; gaps up to COVERAGE_GAP_BRIDGE_PX
# are bridged (a shadow or a shoe crossing the line shouldn't count against
# it), bigger gaps aren't. A candidate below MIN_COVERAGE_RATIO is dropped.
COVERAGE_TOL_PX        = 3    # perpendicular tolerance for a "hit"
COVERAGE_GAP_BRIDGE_PX = 6    # along-line gap size still bridged
MIN_COVERAGE_RATIO     = 0.70

# In boundary selection, a candidate shorter than this fraction of the
# longest candidate in its own H/V group is treated as an internal marking
# (e.g. the centre service line), not a real boundary — see
# _select_boundary_lines.
MIN_GROUP_LEN_FRAC = 0.5

# White court line: low colour saturation. (Brightness is judged locally,
# not by a fixed floor here — see _white_line_mask.)
WHITE_S_MAX = 60

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

      1. Colour   — the line is white: low saturation. Alone, this would
                     also match white shirts/shoes, sky, or bright ad
                     boards.
      2. Contrast — a top-hat transform keeps only features that are
                     narrow and LOCALLY brighter than their immediate
                     surroundings. Alone, this would also match skin,
                     reflections, or any other locally-bright edge
                     regardless of colour.

    Brightness is deliberately judged locally (via the top-hat), not
    against a fixed floor: a real painted line can be near-white (V~255)
    right under the camera and much dimmer (V~85) in a shadowed or
    distant stretch of the same physical line — a fixed absolute
    brightness floor would cut that dim stretch off entirely, even though
    it's still clearly brighter than ITS surroundings, which is what
    actually makes it a line.

    A pixel that is both low-saturation and a locally-bright thin feature
    is, on a badminton court, a boundary line.
    """
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_sat = cv2.inRange(hsv[:, :, 1], 0, WHITE_S_MAX)

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat  = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    _, contrast_mask = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_and(low_sat, contrast_mask)


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


HORIZONTAL_MAX_DEG = 15.0   # see _orientation

def _orientation(direction):
    """
    Buckets a line direction as 'H' (a back/front boundary line -- runs
    laterally across the shot, close to horizontal) or 'V' (a sideline --
    runs away from the camera toward the net).

    Deliberately NOT a 45/45 split. For the camera placement this module
    targets ("behind the court, sees the whole court"), a real sideline
    recedes toward a vanishing point and is usually tilted well short of
    vertical in image space -- 30-40 degrees off horizontal is typical,
    sometimes less. A 45/45 split buckets those as 'H', where they collide
    with the genuine horizontal boundary line during selection (and can
    crowd it out entirely) instead of being bracketed against the OTHER
    sideline the way a 'V' candidate would be. Only lines within
    HORIZONTAL_MAX_DEG of true horizontal -- which a back/front boundary
    line always is, since it runs side-to-side across the shot rather than
    receding into it -- are called 'H'; everything else, including a
    fairly shallow sideline, is 'V'.
    """
    angle  = abs(np.degrees(np.arctan2(direction[1], direction[0]))) % 180
    dist_h = min(angle, abs(180 - angle))   # distance from 0°/180°
    return 'H' if dist_h < HORIZONTAL_MAX_DEG else 'V'


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

    # A real boundary line is straight; a curved court marking (a service
    # circle/arc) is not, but Hough can still chain its gently-curving
    # sub-segments into one "collinear enough" cluster. Measure how far
    # the actual centreline points stray perpendicular to the straight
    # fit -- noise/anti-aliasing keeps this to a pixel or two; systematic
    # curvature grows it well past that, growing with the arc's length.
    # Checked as a fraction of length rather than a flat pixel cap so it
    # scales with how much of the arc got captured.
    perp = (centers.astype(np.float64) - new_point) @ np.array([-new_dir[1], new_dir[0]])
    length = float(tt.max() - tt.min())
    if length > 0 and float(np.max(np.abs(perp))) > max(2.0, 0.015*length):
        return None

    thickness = float(np.median(widths))
    return new_point, new_dir, float(tt.min()), float(tt.max()), thickness


def _line_coverage_ratio(mask, point, direction, t_min, t_max,
                          tol=COVERAGE_TOL_PX, gap_bridge_px=COVERAGE_GAP_BRIDGE_PX):
    """
    How much of a fitted line is actually painted, as a fraction of its
    own length — the longest contiguous run of mask hits along it, gaps
    up to `gap_bridge_px` bridged. 1.0 = solid stripe end to end.

    Used to tell a real boundary line (continuous, bar the odd shadow)
    from the net's top tape (fragments in the mask — sag, self-shadowing,
    mesh/background showing through) without relying on appearance around
    the line, which a glossy, unevenly-lit real floor makes unreliable.
    """
    h, w = mask.shape[:2]
    normal = np.array([-direction[1], direction[0]])
    length = t_max - t_min
    if length <= 0:
        return 0.0
    step = 1.5
    ts = np.arange(0.0, length, step)
    offs = np.arange(-tol, tol + 1)

    hits = []
    for t in ts:
        base = point + direction*(t_min + t)
        xs = np.round(base[0] + normal[0]*offs).astype(int)
        ys = np.round(base[1] + normal[1]*offs).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        hit = valid.any() and bool((mask[np.clip(ys,0,h-1), np.clip(xs,0,w-1)][valid] > 0).any())
        hits.append(hit)

    if not hits:
        return 0.0
    gap_bridge = max(1, int(round(gap_bridge_px / step)))
    best = cur = gap = 0
    for hit in hits:
        if hit:
            cur += 1
            gap = 0
        else:
            gap += 1
            if gap <= gap_bridge:
                cur += 1
            else:
                best = max(best, cur)
                cur = 0
    best = max(best, cur)
    return (best*step) / length


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
    """
    Returns the median colour of the sampled side, or None if no valid
    sample points were found.
    """
    half = SAMPLE_PATCH // 2
    h, w = line_mask.shape[:2]
    all_px = []
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
    if not all_px:
        return None
    all_px = np.concatenate(all_px, axis=0)
    return np.median(all_px, axis=0)


def _determine_in_side(frame, line_mask, point, direction, t_min, t_max):
    """
    Returns the unit normal vector pointing toward the IN side, or None if
    it couldn't be determined — sample points fell outside the frame, or
    both sides read too similarly in colour to call (see below).
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
              f"(d_pos={d_pos:.1f}, d_neg={d_neg:.1f}) -- result may be unreliable.")

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
        # A failed refinement isn't given a lenient pass on the raw Hough
        # fit -- it means either too little of the candidate could be
        # cross-section-validated to trust, or (see _refine_centerline)
        # it's measurably curved, not straight (a service circle/arc, not
        # a boundary line). Either way it's dropped, not kept as-is.
        refined = _refine_centerline(mask, point, direction, t_min, t_max)
        if refined is None:
            continue
        point, direction, t_min, t_max, thickness = refined
        margin = min(MAX_MARGIN_PX, max(MIN_MARGIN_PX, thickness / 2.0))
        p1 = point + direction*t_min
        p2 = point + direction*t_max
        endpoints = (tuple(int(v) for v in p1), tuple(int(v) for v in p2))
        # The refined extent (validated cross-section by cross-section) is
        # the real length — not the raw Hough cluster's segment sum, which
        # can overstate it. Using the stale value here corrupted
        # length-based comparisons downstream (dedup, boundary select).
        length = t_max - t_min

        coverage = _line_coverage_ratio(mask, point, direction, t_min, t_max)
        if coverage < MIN_COVERAGE_RATIO:
            print(f"[Calibration] Rejected a candidate line: only "
                  f"{coverage*100:.0f}% of its length is continuously "
                  f"painted (need >={MIN_COVERAGE_RATIO*100:.0f}%) -- "
                  f"likely the net's top tape, not a floor boundary line.")
            continue

        in_normal = _determine_in_side(frame, mask, point, direction, t_min, t_max)
        if in_normal is None:
            continue
        candidates.append({
            'point': point, 'direction': direction, 'normal': in_normal,
            'endpoints': endpoints, 'length': length, 'margin': margin,
            'orientation': _orientation(direction),
        })
    return candidates


def _line_position(line, frame_shape):
    """
    A scalar position along the axis that separates PARALLEL lines of the
    same orientation — y for horizontal lines (near baseline vs. far
    baseline), x for vertical lines (left sideline vs. right sideline).
    Used to tell distinct parallel lines apart and find the outermost
    pair.

    Evaluated at a FIXED reference (the frame's own centre column/row),
    not read off the line's raw fitted point. A sideline in this module's
    target camera placement is often meaningfully diagonal (see
    _orientation), so its fitted point's x can land anywhere along a wide
    y-range depending on which sub-segment a particular Hough pass
    happened to capture -- two detections of the exact same physical
    sideline, one from its near-net half and one from its near-camera
    half, would otherwise disagree by however far the line has drifted
    sideways between those two segments, far more than a real difference
    between two distinct sidelines. Projecting every candidate to where
    it crosses the same fixed row/column removes that source of
    disagreement, so dedup and boundary selection compare apples to
    apples regardless of which part of the line was actually detected.
    """
    h, w = frame_shape[:2]
    point, direction = line['point'], line['direction']
    if line['orientation'] == 'H':
        if abs(direction[0]) < 1e-9:
            return float(point[1])
        t = (w/2.0 - point[0]) / direction[0]
        return float(point[1] + direction[1]*t)
    else:
        if abs(direction[1]) < 1e-9:
            return float(point[0])
        t = (h/2.0 - point[1]) / direction[1]
        return float(point[0] + direction[0]*t)


MIN_DETECTION_COUNT = 2   # see _dedupe_lines

def _dedupe_lines(candidates, frame_shape, pos_tol=20):
    """
    Merges candidates that are really the same physical line seen more
    than once (different frame/threshold attempts), keeping the longest
    representative of each. Candidates only merge within the same
    orientation and a similar _line_position.

    Also drops any group seen fewer than MIN_DETECTION_COUNT times. A real
    boundary line is painted the same way in every frame, so it tends to
    pass at most of the (frame, Hough-threshold) attempts it's actually
    visible in; a one-off false positive — e.g. a gently curved court
    marking whose curvature happens to fall inside every per-point check's
    tolerance only for one unlucky frame's particular lighting/occlusion —
    typically doesn't repeat. Requiring more than a single sighting costs
    nothing for a genuinely visible line but filters out that kind of
    fluke.
    """
    groups = []   # [{'orientation', 'pos', 'best', 'count'}]
    for cand in candidates:
        pos = _line_position(cand, frame_shape)
        merged = False
        for g in groups:
            if g['orientation'] == cand['orientation'] and abs(pos - g['pos']) < pos_tol:
                g['count'] += 1
                if cand['length'] > g['best']['length']:
                    g['best'], g['pos'] = cand, pos
                merged = True
                break
        if not merged:
            groups.append({'orientation': cand['orientation'], 'pos': pos, 'best': cand, 'count': 1})
    return [g['best'] for g in groups if g['count'] >= MIN_DETECTION_COUNT]


def _motion_center(frames, max_frames=150, warmup=30):
    """
    Rough centre (x, y) of player activity -- literally where the game is
    being played. Used two ways: to anchor which pair of sidelines belongs
    to the court actually in play on a shared multi-court floor (the x
    component), and to sanity-check/correct each selected boundary line's
    IN direction (both components) -- the court interior necessarily
    contains wherever the players actually are, which is a far more
    reliable signal than comparing colours across a line, especially for
    a line near the edge of the frame where both sides are plain court
    surface and colour has almost nothing to go on (see
    _select_boundary_lines). Returns None if no real motion is found (e.g.
    an empty court), so callers can fall back to frame geometry.

    MOG2 needs a run of CONSECUTIVE frames to build a stable background
    model — feeding it a sparse/decimated sample starves it of that and
    its foreground output is unreliable noise, not real motion. This
    processes frames in original order and only trusts the output after
    a warm-up period, the same way the shuttle detector's own background
    subtractor is used elsewhere in this codebase.
    """
    n = min(max_frames, len(frames))
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=max(warmup, 1), varThreshold=40, detectShadows=False)

    xs, ys, weights = [], [], []
    for i in range(n):
        fg = backSub.apply(frames[i])
        if i < warmup:
            continue   # let the model converge before trusting its output
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        ys_idx, xs_idx = np.nonzero(fg)
        if len(xs_idx) > 50:   # ignore near-empty/noise frames
            xs.append(float(np.mean(xs_idx)))
            ys.append(float(np.mean(ys_idx)))
            weights.append(len(xs_idx))

    if not xs:
        return None
    return (float(np.average(xs, weights=weights)),
            float(np.average(ys, weights=weights)))


def _collapse_same_side(group, pos, c0):
    """
    Collapses candidates that agree on which direction is IN down to just
    the outermost (rearmost) one.

    Two lines of the same orientation are only a genuine near/far (or
    left/right) pair if they disagree about which side is IN -- that's
    what it means for them to be opposite edges of the court. If two
    candidates' normals point the SAME way, they're not opposite edges,
    just two lines on the same side (e.g. a service line short of the
    real back boundary, or a duplicate detection). The area inside a line
    is IN and outside it is OUT, so when two lines on the same side both
    got kept, the space between them was being called OUT by the nearer
    one even though the rearmost line is the one that actually bounds the
    court -- only the area beyond THAT should read OUT.

    Picks the survivor by POSITION (further from the group's centre
    anchor `c0`, via the same `pos` used for bracketing), not by asking
    each candidate's own normal which one "permits the larger region".
    That would sound right but isn't robust here: this situation is
    specifically two lines that are BOTH bordered by plain court floor on
    both sides (that's exactly why they read as agreeing on IN direction
    in the first place), which is the one case the colour-based IN/OUT
    check has the least to go on -- so its output is the one signal we
    shouldn't lean on to break the tie. Position doesn't have that
    problem: a real boundary is, by construction, the most extreme
    marking on the court, and an internal line (service line, centre
    line) always sits closer to the middle -- true regardless of which
    way either line's own normal ended up pointing.
    """
    kept = []
    for cand in group:
        merged = False
        for i, k in enumerate(kept):
            if np.dot(cand['normal'], k['normal']) > 0.5:
                if abs(pos(cand) - c0) > abs(pos(k) - c0):
                    kept[i] = cand
                merged = True
                break
        if not merged:
            kept.append(cand)
    return kept


def _select_boundary_lines(candidates, frame_shape, motion_center=None):
    """
    Keeps only the boundary of THIS court per orientation.

    With 1 line it's the boundary as-is (corner/single-line view). With 2,
    they're the outer pair (e.g. near/far baseline) — kept as-is.

    With 3+ parallel lines, this is very likely a facility with several
    courts SHARING one floor (common for badminton halls — courts laid
    out side by side reusing the same lines/space). Those extra lines
    aren't internal markings of one court, they can be another court's
    sideline entirely — taking the outermost pair in that case would span
    multiple courts' width, not just this one's. Instead, bracket a
    centre point with the nearest line on each side of it.

    For the sidelines ('V'), that centre point is the horizontal component
    of where players are actually seen moving (motion_center), when
    available — a much more direct anchor for "this court" than frame
    geometry, since it's literally where the game is being played. Falls
    back to the frame's own centre otherwise (e.g. an empty court, or 'H').

    Before bracketing, drops any candidate under MIN_GROUP_LEN_FRAC of the
    longest candidate in its own orientation group. A short internal
    marking (the centre service line, a short service line) can sit
    closer to the "bracket around centre" anchor point than the real
    sideline/boundary is — which is exactly backwards, since that anchor
    exists to prefer THIS court's boundary over a farther-away adjacent
    court's, not to prefer an internal line over a real one. A genuine
    boundary is close to the longest thing visible in its orientation;
    an internal line consistently measures much shorter (observed well
    under half, in practice) because it doesn't run the court's full
    visible extent.

    After picking this court's line(s) for the orientation, runs
    _collapse_same_side over just that pick. Deliberately not run any
    earlier: two lines from DIFFERENT courts on a shared floor can easily
    share a normal direction too (e.g. two courts' left sidelines both
    have "IN" pointing the same way), and collapsing on that basis before
    the bracket step would silently undo the multi-court disambiguation
    above, keeping whichever court's line is most extreme instead of THIS
    court's. Once narrowed to this court's own candidate(s), a leftover
    same-direction pair can only mean one thing: a nearer line short of
    the true rearmost boundary on the very side already selected.

    Finally, for every kept line, corrects its IN-side normal against
    motion_center if the two disagree: the court interior necessarily
    contains wherever the players actually are, so if a line's own
    colour-based normal puts that point on the OUT side, the normal was
    wrong, not the player. This matters most for exactly the line that's
    hardest for colour to call correctly -- a boundary near the edge of
    the frame, where both sides are the same plain court surface and the
    colour-similarity comparison has very little to go on (see
    _sample_side_color) -- which is also, not coincidentally, the most
    consequential line to get backwards: the outer boundary that decides
    IN vs. OUT for real shots.
    """
    h, w = frame_shape[:2]
    motion_center_x = motion_center[0] if motion_center is not None else None
    center = {
        'H': h / 2.0,
        'V': motion_center_x if motion_center_x is not None else w / 2.0,
    }

    pos = lambda c: _line_position(c, frame_shape)

    result = []
    for o in ('H', 'V'):
        group = [c for c in candidates if c['orientation'] == o]
        if not group:
            continue
        c0 = center[o]
        max_len = max(c['length'] for c in group)
        group = [c for c in group if c['length'] >= MIN_GROUP_LEN_FRAC*max_len]
        group.sort(key=pos)

        if len(group) <= 2:
            chosen = [group[0]] if len(group) == 1 else [group[0], group[-1]]
        else:
            left  = [g for g in group if pos(g) <= c0]
            right = [g for g in group if pos(g) >  c0]
            chosen = []
            if left:
                chosen.append(max(left, key=pos))    # nearest to centre, left/above
            if right:
                chosen.append(min(right, key=pos))   # nearest to centre, right/below

        result.extend(_collapse_same_side(chosen, pos, c0))

    if motion_center is not None:
        mc = np.array(motion_center, dtype=np.float64)
        for line in result:
            if np.dot(mc - line['point'], line['normal']) < 0:
                line['normal'] = -line['normal']

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

    By default only a modest, spread-out sample of frames is tried (fast —
    the normal case). With exhaustive=True, every loaded frame is tried at
    a wider, more sensitive range of Hough thresholds — a slower search
    used only as a second automatic attempt when the fast pass finds
    nothing, so manual line placement is never needed.
    """
    print("[AutoCalib] Scanning frames for sharpest ...")
    if exhaustive:
        top_frames = sorted(range(len(frames)), key=lambda i: -_sharpness(frames[i]))
    else:
        # The sharpest frames overall tend to cluster in the same short
        # stretch of the clip (e.g. a moment nobody's mid-stride) — trying
        # only those risks every sampled frame sharing the same player
        # position, so a line only unoccluded elsewhere never gets seen at
        # all. Bucketing across the whole loaded span first, then taking
        # the sharpest frame per bucket, spreads samples over different
        # moments (different player positions/occlusion) while still
        # skipping obviously blurry frames within each one.
        n_buckets = min(20, len(frames))
        bucket_edges = np.linspace(0, len(frames), n_buckets + 1).astype(int)
        top_frames = []
        for b in range(n_buckets):
            lo, hi = bucket_edges[b], bucket_edges[b+1]
            if lo >= hi:
                continue
            idxs = range(lo, hi)
            top_frames.append(max(idxs, key=lambda i: _sharpness(frames[i])))
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

    motion_center = _motion_center(frames)
    if motion_center is not None:
        print(f"[AutoCalib] Player motion centre x={motion_center[0]:.0f} y={motion_center[1]:.0f}")
    else:
        print("[AutoCalib] No player motion detected -- using frame centre")

    deduped = _dedupe_lines(all_candidates, frames[0].shape)
    lines   = _select_boundary_lines(deduped, frames[0].shape, motion_center)

    for l in lines:
        print(f"[AutoCalib] {l['orientation']} boundary line "
              f"length={l['length']:.0f}px endpoints={l['endpoints']}")
    return lines


# ═══════════════════════════════════════════════════════════════════════
#  Manual calibration (on-demand — not a fallback the system opens on its
#  own; only used when the app explicitly asks for it, e.g. a keybinding)
# ═══════════════════════════════════════════════════════════════════════

def _manual_line_ui(frame, motion_center=None):
    """
    Click up to 2 lines (2 points each). ENTER confirms after the 2nd or
    4th point placed — so a single visible line still works, but a corner
    view can capture both. The IN side of each clicked line is still
    worked out automatically from colour, never asked for -- and, when
    motion_center is available, corrected against it the same way an
    auto-detected line is (see _select_boundary_lines): the court
    interior necessarily contains wherever the players actually are.
    """
    pts = []
    win = "Calibration (click 2 points per line — up to 2 lines)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 760)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.setMouseCallback(win, on_mouse)

    confirmed = False
    while True:
        disp = frame.copy()
        for p in pts:
            cv2.circle(disp, p, 6, (0, 255, 255), -1)
        if len(pts) >= 2:
            cv2.line(disp, pts[0], pts[1], (0, 255, 255), 2)
        if len(pts) == 4:
            cv2.line(disp, pts[2], pts[3], (0, 200, 255), 2)
        cv2.putText(disp,
                    "Click 2 pts/line (up to 2 lines).  ENTER=confirm  R=reset  ESC=cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(16) & 0xFF

        if key == 13 and len(pts) in (2, 4):
            confirmed = True
            break
        elif key == 27:
            break
        elif key == ord('r'):
            pts = []

    cv2.destroyWindow(win)
    if not confirmed:
        return []

    mask  = _white_line_mask(frame)
    lines = []
    for i in range(0, len(pts), 2):
        point = np.array(pts[i], dtype=np.float64)
        end   = np.array(pts[i+1], dtype=np.float64)
        direction = end - point
        length    = float(np.linalg.norm(direction))
        if length < 1:
            continue
        direction /= length
        endpoints = (pts[i], pts[i+1])
        t_min, t_max = 0.0, length

        # Snap the clicked line onto the true painted-line centre, same as
        # the auto path, so a manually-placed line is just as sharp.
        refined = _refine_centerline(mask, point, direction, t_min, t_max)
        if refined is not None:
            point, direction, t_min, t_max, thickness = refined
            margin = min(MAX_MARGIN_PX, max(MIN_MARGIN_PX, thickness / 2.0))
            p1 = point + direction*t_min
            p2 = point + direction*t_max
            endpoints = (tuple(int(v) for v in p1), tuple(int(v) for v in p2))
            length = t_max - t_min
        else:
            margin = DEFAULT_MARGIN_PX

        in_normal = _determine_in_side(frame, mask, point, direction, t_min, t_max)
        if in_normal is None:
            in_normal = np.array([-direction[1], direction[0]])
            print("[Calibration] WARNING: could not auto-detect the IN side by "
                  "colour for a manually-placed line -- defaulted; verify the "
                  "IN/OUT overlay looks correct.")
        if motion_center is not None:
            mc = np.array(motion_center, dtype=np.float64)
            if np.dot(mc - point, in_normal) < 0:
                in_normal = -in_normal

        lines.append({
            'point': point, 'direction': direction, 'normal': in_normal,
            'endpoints': endpoints, 'length': length, 'margin': margin,
            'orientation': _orientation(direction),
        })
    return lines


def calibrate_manual(cap):
    """
    On-demand manual calibration: opens the click UI directly, regardless
    of whether automatic detection would succeed. Unlike calibrate(), this
    is never invoked by the system on its own — only when the caller
    explicitly wants to place or correct lines by hand (e.g. a keybinding
    in the app), for footage where automatic detection isn't reliable
    (a shared multi-court floor, heavy glare, etc.).
    """
    print("\n[Calibration] -- Manual (on-demand) -------------------------")
    print("  Loading a frame ...")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    while len(frames) < 60:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    if not frames:
        raise RuntimeError("No frames available for calibration.")

    best_idx = int(np.argmax([_sharpness(f) for f in frames]))
    motion_center = _motion_center(frames)
    lines = _manual_line_ui(frames[best_idx], motion_center)

    ok = bool(lines)
    if ok:
        _apply(lines)
        _save()
        print(f"[Calibration] Manual line(s) saved "
              f"({len(lines)} line(s): {[l['orientation'] for l in lines]})\n")
    else:
        print("[Calibration] Cancelled.\n")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return ok


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
    print("\n[Calibration] ------------------------------------------")
    print("  Loading frames ...")

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
        print("[Calibration] Fast pass found nothing -- retrying with a wider "
              "automatic search (every frame, more Hough sensitivities) ...")
        lines = auto_calibrate(frames, exhaustive=True)

    if lines:
        _apply(lines)
        _save()
        print(f"[Calibration] Auto-calibration complete "
              f"({len(lines)} line(s): {[l['orientation'] for l in lines]})\n")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True

    print("[Calibration] Automatic detection could not find any boundary line "
          "in this footage. Continuing uncalibrated -- IN/OUT calls are "
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
    print(f"[Calibration] Config saved -> {CONFIG_FILE}")


def load_court_points():
    if not os.path.exists(CONFIG_FILE):
        return False
    with open(CONFIG_FILE) as f:
        data = json.load(f)
    if data.get("version") != CONFIG_VERSION:
        print("[Calibration] Config is an older/incompatible format -- recalibrating.")
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
