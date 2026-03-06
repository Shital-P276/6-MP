"""
Raster Parser v5 — Floor plan PNG/PDF → ParsedGeometry

Supported inputs
────────────────
  • Clean architectural PNGs/PDFs (CAD exports with green dimension annotations)
  • Screenshots of the floor-plan viewer (auto-cropped to the floor-plan panel)

Pipeline
────────
  1. AUTO-CROP     Find the white-background floor plan panel inside any UI screenshot.
                   Uses brightness ≥ 175 (excludes cyan 3-D viewer walls) and picks
                   the contour with the highest fill-ratio (most uniformly white).

  2. PPM           Read green dimension tick marks along the top/bottom border.
                   Each inter-tick gap represents one labelled bay (default 400 cm = 4 m).
                   Falls back to FALLBACK_PPM (65) if tick detection fails.

  3. SEGMENT       Classify every pixel:
                     WALL : gray 82–148, not strongly green
                     ROOM : gray 153–244, not green
                   Strip the outer BORDER_CROP_PX (annotation grid).

  4. SKELETON      Morphological thinning (cross-kernel, up to 80 iterations)
                   → single-pixel wall centerlines.

  5. HOUGH+SNAP    HoughLinesP on skeleton; snap lines within ±12° of horizontal or
                   vertical to exact H/V axes.  Discard diagonals.

  6. MERGE         Group segments by fixed coordinate (±11 px tolerance), bridge gaps
                   ≤ 18 px, discard results shorter than MIN_WALL_PX.

  7. DEDUP         Collapse parallel double-edges (both faces of a thick wall) into
                   one centerline (average position).

  8. COMPLETE      Add any missing outer wall side inferred from the room bounding box.

  9. OPENINGS      Scan a ±14 px band along every wall centerline.
                   a) GAP SCAN: runs of bright pixels (>148) ≥ MIN_OPENING_M wide →
                      door (≤ MAX_DOOR_M) or structural window (> MAX_DOOR_M).
                   b) WINDOW-SYMBOL SCAN: detect the CAD double-line window symbol
                      (two narrow dark stripes bracketing a bright band, all within
                      the wall band) → architectural window.

  10. ROOMS        Seal doorway gaps in the wall mask (48×48 close kernel), then
                   connected-component analysis on room-coloured pixels.

All thresholds are general — no values hard-coded for a specific floor plan.
"""

from __future__ import annotations
from pathlib import Path
import math
from .dxf_parser import Segment, Point2D, ParsedGeometry

try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import pdf2image
    PDF2_OK = True
except ImportError:
    PDF2_OK = False

# ── Tunable constants ─────────────────────────────────────────────────────────
CROP_BRIGHT_TH    = 175   # floor-plan panel: pixels brighter than this
WALL_LO, WALL_HI  = 82,  148
ROOM_LO, ROOM_HI  = 153, 244
GREEN_DIFF        = 22    # G-R and G-B delta for annotation suppression
BORDER_CROP_PX    = 4     # absolute minimum border strip (overridden dynamically in _segment)
SKEL_MAX_ITER     = 80
HOUGH_TH          = 7
HOUGH_MIN_LEN     = 12
HOUGH_MAX_GAP     = 14
AXIS_SNAP_TOL     = 12    # snap near-H/V lines to exact axes
COORD_GROUP_TOL   = 11    # group collinear lines within this px distance
MERGE_GAP         = 60    # max gap to bridge — covers doorway gaps (~50px at 61ppm)
DEDUP_DIST        = 20    # max px between parallel double-edges to collapse
MIN_WALL_PX       = 40    # discard wall segments shorter than this (covers noise arc artifacts)
WALL_SEAL_PX      = 48    # kernel for doorway-sealing before room detection
MIN_ROOM_PX       = 2000
MAX_ROOM_ASPECT   = 14.0
FALLBACK_PPM      = 65.0  # px/m when tick detection fails

# Opening detection
MIN_OPENING_M     = 0.35  # minimum gap width treated as an opening (m)
MAX_DOOR_M        = 1.40  # gaps wider than this → window (not door)
MAX_WINDOW_M      = 3.50  # gaps wider than this → noise, ignore
SCAN_HALF_PX      = 14    # half-width of wall-band scan
OPENING_BRIGHT    = 148   # mean brightness above this → opening pixel

# Window-symbol detection
WIN_SYM_MIN_BRIGHT_PX = 6   # min bright pixels between two dark stripes
WIN_SYM_MAX_TOTAL_M   = 3.0 # max total window symbol extent (m)
WIN_SYM_MIN_TOTAL_M   = 0.3 # min total window symbol extent (m)
WIN_SYM_DARK_TH       = 148 # pixel ≤ this = part of window frame


# ── I/O ───────────────────────────────────────────────────────────────────────

def _req_cv():
    if not CV2_OK:
        raise ImportError("pip install opencv-python")

def _load(path: str):
    _req_cv()
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    return img

def _pdf_to_img(path: str, dpi: int = 200):
    if not PDF2_OK:
        raise ImportError("pip install pdf2image")
    _req_cv()
    pages = pdf2image.convert_from_path(path, dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise ValueError("Empty PDF")
    return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)


# ── Step 1 — auto-crop ────────────────────────────────────────────────────────

def _autocrop(img):
    """
    Return (cropped, (x0,y0,x1,y1)).
    Picks the contour whose bounding-box is most uniformly filled with
    bright (≥ CROP_BRIGHT_TH) pixels.  This reliably selects a white
    floor-plan panel inside a dark UI even when the UI has cyan lines.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W  = gray.shape
    bright = (gray > CROP_BRIGHT_TH).astype(np.uint8) * 255
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    filled = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, (0, 0, W, H)

    best_fill, best_rect = -1.0, None
    for c in cnts:
        if cv2.contourArea(c) < H * W * 0.04:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        fill = float(filled[y:y+ch, x:x+cw].mean()) / 255.0
        if fill > best_fill:
            best_fill  = fill
            best_rect  = (x, y, cw, ch)

    if best_rect is None:
        return img, (0, 0, W, H)
    x, y, cw, ch = best_rect
    return img[y:y+ch, x:x+cw], (x, y, x+cw, y+ch)


# ── Step 2 — PPM detection ────────────────────────────────────────────────────

def _detect_ppm(fp) -> float:
    """
    Detect pixels-per-metre from green dimension tick marks.
    Scans the top 5 % of the image for a row with ≥ 3 green pixel spans.
    The median inter-span gap (> 50 px) is assumed to represent one 4-m bay.
    """
    if fp.ndim < 3:
        return FALLBACK_PPM

    b, g, r = cv2.split(fp)
    gi      = g.astype(np.int16)
    gmask   = ((gi - r.astype(np.int16) > GREEN_DIFF) &
               (gi - b.astype(np.int16) > GREEN_DIFF)).astype(np.uint8) * 255

    FH, FW  = fp.shape[:2]
    scan_y  = max(1, FH // 20)
    best_spans, best_n = None, 0

    for y in range(max(1, scan_y - 5), min(FH, scan_y + 10)):
        row    = gmask[y, :]
        spans  = []
        in_g   = False
        for x, v in enumerate(row):
            if v > 0 and not in_g:
                spans.append(x); in_g = True
            elif v == 0 and in_g:
                in_g = False
        if len(spans) >= 3 and len(spans) > best_n:
            best_n = len(spans); best_spans = spans

    if best_spans:
        gaps = [best_spans[i+1] - best_spans[i]
                for i in range(len(best_spans)-1)
                if best_spans[i+1] - best_spans[i] > 50]
        if gaps:
            med = sorted(gaps)[len(gaps)//2]
            ppm = med / 4.0        # one bay = 4 m
            if 20.0 < ppm < 400.0:
                return ppm

    return FALLBACK_PPM


# ── Step 2b — dynamic border crop ────────────────────────────────────────────

def _detect_border_crop(fp, green_mask_dilated) -> int:
    """
    Dynamically compute the border crop width for this image.

    Instead of a fixed pixel count, we find the inner edge of the green
    annotation frame by scanning inward from each edge until we find a row/
    column where green coverage drops below a 10% threshold.
    Then we add a small buffer (GREEN_BORDER_BUFFER px) to catch any
    fringe pixels that sneak through the green mask.

    For images with no green frame (or very thin frames like 3px) this
    naturally returns a small value, protecting the outer walls.
    For images with wide annotation bands (>30px) it returns a larger value.

    Falls back to BORDER_CROP_PX_MIN if detection fails.
    """
    FH, FW = fp.shape[:2]
    GREEN_BORDER_BUFFER = 8   # px of buffer beyond the last green row/col
    MIN_GREEN_COV = 0.10      # row/col is "green" if ≥ 10% of its pixels are green

    def scan_inward_top():
        last_green_row = -1
        for y in range(FH // 2):
            cov = float(green_mask_dilated[y, :].sum()) / (255 * FW)
            if cov >= MIN_GREEN_COV:
                last_green_row = y
            elif last_green_row >= 0:
                break   # first row with no significant green after green zone
        return last_green_row + 1 + GREEN_BORDER_BUFFER if last_green_row >= 0 else BORDER_CROP_PX

    def scan_inward_bottom():
        last_green_row = -1
        for y in range(FH - 1, FH // 2, -1):
            cov = float(green_mask_dilated[y, :].sum()) / (255 * FW)
            if cov >= MIN_GREEN_COV:
                last_green_row = y
            elif last_green_row >= 0:
                break
        return (FH - last_green_row) + GREEN_BORDER_BUFFER if last_green_row >= 0 else BORDER_CROP_PX

    def scan_inward_left():
        last_green_col = -1
        for x in range(FW // 2):
            cov = float(green_mask_dilated[:, x].sum()) / (255 * FH)
            if cov >= MIN_GREEN_COV:
                last_green_col = x
            elif last_green_col >= 0:
                break
        return last_green_col + 1 + GREEN_BORDER_BUFFER if last_green_col >= 0 else BORDER_CROP_PX

    def scan_inward_right():
        last_green_col = -1
        for x in range(FW - 1, FW // 2, -1):
            cov = float(green_mask_dilated[:, x].sum()) / (255 * FW)
            if cov >= MIN_GREEN_COV:
                last_green_col = x
            elif last_green_col >= 0:
                break
        return (FW - last_green_col) + GREEN_BORDER_BUFFER if last_green_col >= 0 else BORDER_CROP_PX

    top    = scan_inward_top()
    bottom = scan_inward_bottom()
    left   = scan_inward_left()
    right  = scan_inward_right()

    # Use the median of the four sides to avoid asymmetric layouts skewing things
    # but never go below the absolute minimum
    sides = sorted([top, bottom, left, right])
    crop  = max(BORDER_CROP_PX, sides[1])   # median of 4 = middle two, take lower
    # Safety cap: never crop more than 8% of the smaller image dimension
    max_crop = int(min(FH, FW) * 0.08)
    return min(crop, max_crop)


# ── Step 3 — segmentation ─────────────────────────────────────────────────────

def _segment(fp):
    """Return (wall_mask, room_mask, border_crop) with annotation border stripped.
    
    The border width is determined dynamically from the green annotation frame
    rather than a fixed constant — this handles floor plans of any size/scale.
    """
    gray  = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY) if fp.ndim == 3 else fp
    FH, FW = gray.shape

    if fp.ndim == 3:
        b, g, r = cv2.split(fp)
        gi     = g.astype(np.int16)
        green  = ((gi - r.astype(np.int16) > GREEN_DIFF) &
                  (gi - b.astype(np.int16) > GREEN_DIFF)).astype(np.uint8) * 255
    else:
        green  = np.zeros_like(gray)

    green_d = cv2.dilate(green, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
    not_g   = cv2.bitwise_not(green_d)

    wall_mask = cv2.bitwise_and(cv2.inRange(gray, WALL_LO, WALL_HI), not_g)
    room_mask = cv2.bitwise_and(cv2.inRange(gray, ROOM_LO, ROOM_HI), not_g)

    # Dynamic border crop: detect actual green frame extent instead of fixed px
    b2 = _detect_border_crop(fp, green_d)
    b2 = max(BORDER_CROP_PX, min(b2, FH // 4, FW // 4))  # sanity clamp

    inner = np.zeros((FH, FW), dtype=np.uint8)
    inner[b2:FH-b2, b2:FW-b2] = 255
    wall_mask = cv2.bitwise_and(wall_mask, inner)
    room_mask = cv2.bitwise_and(room_mask, inner)
    return wall_mask, room_mask, b2


# ── Step 4 — skeleton ─────────────────────────────────────────────────────────

def _skeleton(mask):
    kc   = cv2.getStructuringElement(cv2.MORPH_RECT,  (3, 3))
    kern = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    skel = np.zeros_like(temp)
    for _ in range(SKEL_MAX_ITER):
        er   = cv2.erode(temp, kern)
        diff = cv2.subtract(temp, cv2.dilate(er, kern))
        skel = cv2.bitwise_or(skel, diff)
        temp = er
        if not cv2.countNonZero(temp):
            break
    return skel


# ── Steps 5-7 — Hough → cleaned wall list ────────────────────────────────────

def _snap(x1, y1, x2, y2):
    if abs(y2-y1) <= AXIS_SNAP_TOL:
        my = (y1+y2)//2
        return min(x1,x2), my, max(x1,x2), my, 'H'
    if abs(x2-x1) <= AXIS_SNAP_TOL:
        mx = (x1+x2)//2
        return mx, min(y1,y2), mx, max(y1,y2), 'V'
    return None

def _merge(lines, orient, FH, FW, border_crop=BORDER_CROP_PX):
    border = border_crop
    lim    = FH if orient == 'H' else FW
    groups: dict[int, list] = {}
    for l in lines:
        fixed = l[1] if orient == 'H' else l[0]
        a     = l[0] if orient == 'H' else l[1]
        b     = l[2] if orient == 'H' else l[3]
        if fixed < border or fixed > lim - border:
            continue
        key = next((k for k in groups if abs(k - fixed) <= COORD_GROUP_TOL), None)
        if key is None:
            key = fixed; groups[key] = []
        groups[key].append((a, b))

    result = []
    for fixed, segs in groups.items():
        segs.sort(); ca, cb = segs[0]
        for a, b in segs[1:]:
            if a <= cb + MERGE_GAP:
                cb = max(cb, b)
            else:
                if cb - ca >= MIN_WALL_PX:
                    result.append((ca, fixed, cb, fixed) if orient == 'H'
                                  else (fixed, ca, fixed, cb))
                ca, cb = a, b
        if cb - ca >= MIN_WALL_PX:
            result.append((ca, fixed, cb, fixed) if orient == 'H'
                          else (fixed, ca, fixed, cb))
    return result

def _dedup(segs, orient):
    """
    Collapse parallel double-edges (both faces of a thick wall) into one
    centerline.  Uses a length-weighted average for the fixed coordinate so
    long wall lines dominate over short arc/corner fragments that happen to
    fall within DEDUP_DIST.
    """
    used, result = set(), []
    for i, s in enumerate(segs):
        if i in used: continue
        group = [s]
        for j, t in enumerate(segs):
            if j <= i or j in used: continue
            if orient == 'H':
                if abs(t[1]-s[1]) > DEDUP_DIST: continue
                if min(s[2],t[2]) - max(s[0],t[0]) >= MIN_WALL_PX//2:
                    group.append(t); used.add(j)
            else:
                if abs(t[0]-s[0]) > DEDUP_DIST: continue
                if min(s[3],t[3]) - max(s[1],t[1]) >= MIN_WALL_PX//2:
                    group.append(t); used.add(j)
        # TODO [2]: length-weighted average so dominant long line sets position
        if orient == 'H':
            weights = [g[2]-g[0] for g in group]
            total_w = sum(weights) or 1
            ay = int(sum(g[1]*w for g,w in zip(group,weights)) / total_w)
            result.append((min(g[0] for g in group), ay,
                           max(g[2] for g in group), ay))
        else:
            weights = [g[3]-g[1] for g in group]
            total_w = sum(weights) or 1
            ax = int(sum(g[0]*w for g,w in zip(group,weights)) / total_w)
            result.append((ax, min(g[1] for g in group),
                           ax, max(g[3] for g in group)))
        used.add(i)
    return result

def _refine_centerlines(h_walls, v_walls, wall_mask):
    """
    TODO [1]: Sub-pixel refinement of wall centerlines.
    After Hough gives approximate positions, scan the wall body pixels
    perpendicular to each wall to find the true weighted center.
    This corrects drift caused by the length-weighted dedup averaging
    two skeleton lines that are not equidistant from the true center.
    """
    REFINE_BAND = 18   # px — search ±this from approximate position
    LO, HI      = WALL_LO, WALL_HI

    refined_h = []
    for x1, y0, x2, _ in h_walls:
        y_min = max(0, y0 - REFINE_BAND)
        y_max = min(wall_mask.shape[0] - 1, y0 + REFINE_BAND)
        # Collect y positions of wall pixels in a few representative columns
        wy = []
        step = max(1, (x2 - x1) // 30)
        for x in range(x1, x2 + 1, step):
            col = wall_mask[y_min:y_max+1, x]
            for dy in range(len(col)):
                if col[dy] > 0:
                    wy.append(y_min + dy)
        yr = int(round(sum(wy) / len(wy))) if wy else y0
        refined_h.append((x1, yr, x2, yr))

    refined_v = []
    for x0, y1, _, y2 in v_walls:
        x_min = max(0, x0 - REFINE_BAND)
        x_max = min(wall_mask.shape[1] - 1, x0 + REFINE_BAND)
        wx = []
        step = max(1, (y2 - y1) // 30)
        for y in range(y1, y2 + 1, step):
            row = wall_mask[y, x_min:x_max+1]
            for dx in range(len(row)):
                if row[dx] > 0:
                    wx.append(x_min + dx)
        xr = int(round(sum(wx) / len(wx))) if wx else x0
        refined_v.append((xr, y1, xr, y2))

    return refined_h, refined_v


def _filter_edge_stubs(h_walls, v_walls, FH, FW, ppm, border_crop=BORDER_CROP_PX):
    """
    Remove short wall stubs that are artifacts of the image border annotation
    (tick marks, frame corners) rather than real walls.

    A stub is suspicious if:
      1. It is shorter than STUB_MAX_M metres (too short to be a real wall), AND
      2. It is within STUB_EDGE_PX pixels of any image edge.

    STUB_EDGE_PX is derived from border_crop (the dynamically measured green-frame
    width) so no values are hardcoded for a specific floor plan.
    """
    STUB_MAX_M   = 0.60   # walls shorter than this are candidates for removal
    STUB_EDGE_PX = border_crop * 2   # tick-mark stubs live within 2x the frame width

    stub_max_px = int(STUB_MAX_M * ppm)

    def near_edge(fixed, a, b, orient):
        if orient == 'H':
            # H-wall at y=fixed; near top or bottom edge?
            return (fixed < STUB_EDGE_PX or fixed > FH - STUB_EDGE_PX or
                    a < STUB_EDGE_PX or b > FW - STUB_EDGE_PX)
        else:
            # V-wall at x=fixed; near left or right edge?
            return (fixed < STUB_EDGE_PX or fixed > FW - STUB_EDGE_PX or
                    a < STUB_EDGE_PX or b > FH - STUB_EDGE_PX)

    filtered_h = []
    for x1, y, x2, _ in h_walls:
        length_px = x2 - x1
        if length_px < stub_max_px and near_edge(y, x1, x2, 'H'):
            continue   # drop phantom edge stub
        filtered_h.append((x1, y, x2, y))

    filtered_v = []
    for x, y1, _, y2 in v_walls:
        length_px = y2 - y1
        if length_px < stub_max_px and near_edge(x, y1, y2, 'V'):
            continue   # drop phantom edge stub
        filtered_v.append((x, y1, x, y2))

    return filtered_h, filtered_v


def _detect_walls(wall_mask, border_crop=BORDER_CROP_PX):
    FH, FW = wall_mask.shape
    skel   = _skeleton(wall_mask)
    raw    = cv2.HoughLinesP(skel, 1, math.pi/180,
                             threshold=HOUGH_TH,
                             minLineLength=HOUGH_MIN_LEN,
                             maxLineGap=HOUGH_MAX_GAP)
    if raw is None:
        return [], []
    axis  = [r for r in (_snap(*l[0]) for l in raw) if r]
    H     = _dedup(_merge([l for l in axis if l[4]=='H'], 'H', FH, FW, border_crop), 'H')
    V     = _dedup(_merge([l for l in axis if l[4]=='V'], 'V', FH, FW, border_crop), 'V')
    # Refine approximate centerlines to true wall-body centers
    H, V  = _refine_centerlines(H, V, wall_mask)
    return H, V


# ── Step 8 — complete missing outer walls ─────────────────────────────────────

def _complete_outer_walls(h_walls, v_walls, FH, FW):
    """
    If any side of the building's bounding box has no detected wall within
    OUTER_TOL pixels, synthesise it from the extent of all other walls.
    This repairs the common case where the left exterior wall is only 2 px
    wide (anti-aliased edge) and is missed by the skeleton+Hough stage.
    """
    if not h_walls and not v_walls:
        return h_walls, v_walls

    OUTER_TOL  = 30   # px — how close the nearest wall must be to count as that side
    EXTEND_TOL = 60   # px — extend wall endpoints to snap to nearest perpendicular wall

    all_xs = [x for l in h_walls for x in (l[0], l[2])] + [l[0] for l in v_walls]
    all_ys = [l[1] for l in h_walls] + [y for l in v_walls for y in (l[1], l[3])]
    if not all_xs:
        return h_walls, v_walls

    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)

    # ── Step A: Extend V-wall endpoints to snap to nearest H-wall ────────────
    # This fixes partial walls like LEFT-V whose top is cut off because the
    # thick corner block gets absorbed into the TOP-H skeleton.
    h_ys = sorted(set(l[1] for l in h_walls))  # all H-wall y positions

    def nearest_h(y):
        """Nearest H-wall y to a given y, within EXTEND_TOL."""
        best = None; best_d = EXTEND_TOL + 1
        for hy in h_ys:
            d = abs(hy - y)
            if d < best_d:
                best_d = d; best = hy
        return best if best_d <= EXTEND_TOL else None

    new_v = []
    for x, y1, _, y2 in v_walls:
        nh1 = nearest_h(y1)
        nh2 = nearest_h(y2)
        sy1 = nh1 if nh1 is not None else y1
        sy2 = nh2 if nh2 is not None else y2
        # TODO [1]: skip zero/near-zero length walls produced by snap collision
        if abs(sy2 - sy1) >= MIN_WALL_PX:
            new_v.append((x, sy1, x, sy2))

    # ── Step B: Extend H-wall endpoints to snap to nearest V-wall ────────────
    v_xs = sorted(set(l[0] for l in new_v))

    def nearest_v(x):
        best = None; best_d = EXTEND_TOL + 1
        for vx in v_xs:
            d = abs(vx - x)
            if d < best_d:
                best_d = d; best = vx
        return best if best_d <= EXTEND_TOL else None

    new_h = []
    for x1, y, x2, _ in h_walls:
        nv1 = nearest_v(x1)
        nv2 = nearest_v(x2)
        sx1 = nv1 if nv1 is not None else x1
        sx2 = nv2 if nv2 is not None else x2
        # TODO [1]: skip zero/near-zero length walls
        if abs(sx2 - sx1) >= MIN_WALL_PX:
            new_h.append((sx1, y, sx2, y))

    # ── Step C: Add completely missing outer walls ────────────────────────────
    # Recalculate extents after snapping
    all_xs2 = [x for l in new_h for x in (l[0],l[2])] + [l[0] for l in new_v]
    all_ys2 = [l[1] for l in new_h] + [y for l in new_v for y in (l[1],l[3])]
    min_x2, max_x2 = min(all_xs2), max(all_xs2)
    min_y2, max_y2 = min(all_ys2), max(all_ys2)

    def has_h(y): return any(abs(l[1]-y) <= OUTER_TOL for l in new_h)
    def has_v(x): return any(abs(l[0]-x) <= OUTER_TOL for l in new_v)

    if not has_h(min_y2):
        new_h.insert(0, (min_x2, min_y2, max_x2, min_y2))

    if max_y2 > FH * 0.55 and not has_h(max_y2):
        bx0 = min((l[0] for l in new_h if l[1] > FH*0.5), default=min_x2)
        bx1 = max((l[2] for l in new_h if l[1] > FH*0.5), default=max_x2)
        new_h.append((bx0, max_y2, bx1, max_y2))

    if not has_v(min_x2):
        left_ys = [l[1] for l in new_h if l[0] <= min_x2 + 30]
        ly0 = min(left_ys, default=min_y2)
        ly1 = max(left_ys, default=max_y2)
        new_v.insert(0, (min_x2, ly0, min_x2, ly1))

    if not has_v(max_x2):
        new_v.append((max_x2, min_y2, max_x2, max_y2))

    return new_h, new_v


# ── Step 9 — opening detection ────────────────────────────────────────────────

def _sample_wall_band(fp_gray, orient, fixed, pos, half=SCAN_HALF_PX):
    """
    Sample the wall band at a given position along the wall.
    Returns the MEDIAN brightness (more robust than mean against sparse
    dark jamb/frame pixels that skew mean below threshold at door edges).
    """
    FH, FW = fp_gray.shape
    if orient == 'H':
        y0 = max(0, fixed - half); y1 = min(FH-1, fixed + half)
        x  = max(0, min(FW-1, pos))
        strip = fp_gray[y0:y1+1, x]
    else:
        x0 = max(0, fixed - half); x1 = min(FW-1, fixed + half)
        y  = max(0, min(FH-1, pos))
        strip = fp_gray[y, x0:x1+1]
    # Use median: robust against a few dark jamb pixels at opening edges
    import numpy as _np
    return float(_np.median(strip))


def _find_gap_openings(fp_gray, orient, fixed, a, b, ppm):
    """
    Scan along wall from a->b.  Find contiguous runs of positions where the
    wall band is bright (mean > OPENING_BRIGHT) -- these are structural gaps
    (doors or wide windows).

    Endpoint filter: gaps that touch either wall endpoint are the FREE END of
    the wall (open room space), not a real door/window.  We require the gap
    to be flanked by wall pixels on BOTH sides.
    """
    min_px  = max(2, int(MIN_OPENING_M * ppm))
    max_win = int(MAX_WINDOW_M * ppm)
    # Guard: a real opening must have at least this many wall-px before & after it
    endpoint_guard = min_px

    bright_run, gap_start = 0, None
    gaps = []

    for i in range(a, b + 1):
        mean = _sample_wall_band(fp_gray, orient, fixed, i)
        is_open = mean > OPENING_BRIGHT

        if is_open:
            if gap_start is None: gap_start = i
            bright_run += 1
        else:
            if gap_start is not None and bright_run >= min_px:
                gaps.append((gap_start, i-1, bright_run))
            gap_start = None; bright_run = 0

    if gap_start is not None and bright_run >= min_px:
        gaps.append((gap_start, b, bright_run))

    results = []
    max_door_px = int(MAX_DOOR_M * ppm)
    for gs, ge, gpx in gaps:
        if gpx > max_win: continue
        # FIX: Discard gaps touching either wall endpoint -- those are open
        # wall ends (room interior spilling in), not structural openings.
        if gs <= a + endpoint_guard or ge >= b - endpoint_guard:
            continue
        kind = "door" if gpx <= max_door_px else "window"
        results.append((gs, ge, gpx, kind))
    return results


def _find_window_symbols(fp_gray, orient, fixed, a, b, ppm):
    """
    Detect CAD window symbols: two short dark stripes bracketing a bright
    band, all within the wall band.  This catches windows drawn as double
    parallel lines on the wall face (the wall is not cut through, but the
    symbol clearly marks a window).

    A symbol looks like:
      ...WALL|dark|bright|dark|WALL...
      where dark ≤ WIN_SYM_DARK_TH and bright > OPENING_BRIGHT.
    """
    min_total_px  = max(2, int(WIN_SYM_MIN_TOTAL_M * ppm))
    max_total_px  = int(WIN_SYM_MAX_TOTAL_M * ppm)
    FH, FW        = fp_gray.shape

    # Collect per-position brightness in the band
    profile = []
    for i in range(a, b+1):
        mean = _sample_wall_band(fp_gray, orient, fixed, i)
        profile.append(mean)

    # Find alternating dark→bright→dark pattern
    n = len(profile)
    windows = []
    i = 0
    while i < n - min_total_px:
        # Find first dark stripe
        if profile[i] <= WIN_SYM_DARK_TH:
            i += 1; continue
        # Look for pattern: bright_start, then dark stripe inside
        bright_start = None
        j = i
        while j < n and profile[j] > WIN_SYM_DARK_TH:
            if bright_start is None: bright_start = j
            j += 1
        if bright_start is None or j >= n:
            i = j + 1; continue
        bright_end  = j - 1
        bright_span = bright_end - bright_start + 1
        if bright_span < WIN_SYM_MIN_BRIGHT_PX:
            i = j + 1; continue
        # Now find the closing dark stripe
        k = j
        while k < n and profile[k] <= WIN_SYM_DARK_TH:
            k += 1
        closing_span = k - j
        if closing_span < 1:
            i = j + 1; continue
        total = k - i
        if min_total_px <= total <= max_total_px:
            windows.append((a + bright_start, a + bright_end, bright_span))
        i = k

    return [(gs, ge, gpx, "window") for gs, ge, gpx in windows]


def _detect_openings_from_image(fp_gray, h_walls, v_walls, ppm, border_crop=None):
    """
    Returns list of opening dicts:
      wall_id, orient, t_start, t_end, t_center, width_m, kind,
      x_px, y_px (image coords), x, y (metres, pre-Y-flip),
      swing_side: 'left'|'right' relative to wall travel direction.

    Detects openings in two passes:
      1. Intra-segment: bright gaps WITHIN a single wall segment (existing logic).
      2. Inter-segment: bright gaps BETWEEN two collinear wall stubs on the same
         line — these are door/window openings too large for MERGE_GAP to bridge.
    """
    openings = []
    wall_id  = 0
    FH, FW   = fp_gray.shape
    SIDE_SAMPLE = 25  # px — how far off centreline to sample for room brightness

    def room_side(orient, fixed, mid):
        """Which side of the wall has the brighter (room) interior."""
        if orient == 'H':
            above = float(np.mean(fp_gray[
                max(0, fixed - SIDE_SAMPLE): max(0, fixed - 5),
                max(0, mid - 8): min(FW, mid + 8)
            ])) if fixed > SIDE_SAMPLE else 0.0
            below = float(np.mean(fp_gray[
                min(FH, fixed + 5): min(FH, fixed + SIDE_SAMPLE),
                max(0, mid - 8): min(FW, mid + 8)
            ])) if fixed + SIDE_SAMPLE < FH else 0.0
            return 'left' if above >= below else 'right'
        else:
            left_x = float(np.mean(fp_gray[
                max(0, mid - 8): min(FH, mid + 8),
                max(0, fixed - SIDE_SAMPLE): max(0, fixed - 5)
            ])) if fixed > SIDE_SAMPLE else 0.0
            right_x = float(np.mean(fp_gray[
                max(0, mid - 8): min(FH, mid + 8),
                min(FW, fixed + 5): min(FW, fixed + SIDE_SAMPLE)
            ])) if fixed + SIDE_SAMPLE < FW else 0.0
            return 'left' if left_x >= right_x else 'right'

    def emit(wall_id, orient, fixed, a, b, gs, ge, gpx, kind):
        """Emit one opening dict relative to wall segment (a, b)."""
        wall_len = b - a
        if wall_len < 1: return
        t0 = (gs - a) / wall_len
        t1 = (ge - a) / wall_len
        tc = (t0 + t1) / 2
        wm = gpx / ppm
        if orient == 'H':
            xp = (gs + ge) // 2; yp = fixed
            xm = xp / ppm;       ym = yp / ppm
        else:
            xp = fixed;           yp = (gs + ge) // 2
            xm = xp / ppm;       ym = yp / ppm
        side = room_side(orient, fixed, xp if orient == 'H' else yp)
        openings.append({
            "wall_id":    wall_id,
            "orient":     orient,
            "t_start":    round(t0, 4),
            "t_end":      round(t1, 4),
            "t_center":   round(tc, 4),
            "width_m":    round(wm, 3),
            "kind":       kind,
            "x_px": xp,   "y_px": yp,
            "x":    round(xm, 3),
            "y":    round(ym, 3),
            "swing_side": side,
        })

    def emit_inter(wall_id_left, orient, fixed, left_end, right_start,
                   bright_start, bright_end, kind):
        """
        Emit an opening that lives in the GAP between two collinear wall stubs.
        We attach it to the LEFT/LOWER stub (wall_id_left) and express t > 1.0
        so split_wall_at_openings can clip it. The position x/y is in metres.
        """
        gpx = bright_end - bright_start + 1
        wm  = gpx / ppm
        if orient == 'H':
            xp = (bright_start + bright_end) // 2; yp = fixed
            xm = xp / ppm;                         ym = yp / ppm
        else:
            xp = fixed; yp = (bright_start + bright_end) // 2
            xm = xp / ppm; ym = yp / ppm
        side = room_side(orient, fixed, xp if orient == 'H' else yp)
        # Store as a standalone opening — pipeline's _build_raster_openings
        # will match it to the nearest wall by world position.
        openings.append({
            "wall_id":    wall_id_left,  # nearest stub (for dedup checks)
            "orient":     orient,
            "t_start":    -1.0,   # sentinel: inter-segment (not inside any stub)
            "t_end":      -1.0,
            "t_center":   -1.0,
            "width_m":    round(wm, 3),
            "kind":       kind,
            "x_px": xp,   "y_px": yp,
            "x":    round(xm, 3),
            "y":    round(ym, 3),
            "swing_side": side,
        })

    # ── PASS 1: Intra-segment gaps ────────────────────────────────────────────
    for x1, y, x2, _ in h_walls:
        gaps = _find_gap_openings(fp_gray, 'H', y, x1, x2, ppm)
        for gs, ge, gpx, kind in gaps:
            emit(wall_id, 'H', y, x1, x2, gs, ge, gpx, kind)
        wall_id += 1

    for x, y1, _, y2 in v_walls:
        gaps = _find_gap_openings(fp_gray, 'V', x, y1, y2, ppm)
        for gs, ge, gpx, kind in gaps:
            emit(wall_id, 'V', x, y1, y2, gs, ge, gpx, kind)
        syms = _find_window_symbols(fp_gray, 'V', x, y1, y2, ppm)
        for gs, ge, gpx, kind in syms:
            already = any(
                abs(op["x_px"] - x) < 5 and abs(op["y_px"] - (gs + ge) // 2) < gpx
                for op in openings if op["wall_id"] == wall_id
            )
            if not already:
                emit(wall_id, 'V', x, y1, y2, gs, ge, gpx, kind)
        wall_id += 1

    # ── PASS 2: Inter-segment gaps (collinear wall stubs) ────────────────────
    # Group H-walls by y, V-walls by x.  For each group of 2+ stubs on the
    # same line, check the GAP between them for bright (room-visible) pixels.
    # If the bright span is a valid door/window width, emit it.
    from collections import defaultdict
    max_door_px   = int(MAX_DOOR_M * ppm)
    max_win_px    = int(MAX_WINDOW_M * ppm)
    min_open_px   = max(2, int(MIN_OPENING_M * ppm))

    h_by_y = defaultdict(list)   # y → [(x1, x2, wall_id), ...]
    _wid = 0
    for x1, y, x2, _ in h_walls:
        h_by_y[y].append((x1, x2, _wid))
        _wid += 1

    v_by_x = defaultdict(list)   # x → [(y1, y2, wall_id), ...]
    for x, y1, _, y2 in v_walls:
        v_by_x[x].append((y1, y2, _wid))
        _wid += 1

    DEDUP_RADIUS_PX = max(12, int(ppm * 0.5))   # ~0.5m; adapts to image scale

    for y, segs in h_by_y.items():
        segs.sort()
        for i in range(len(segs) - 1):
            x_left_end   = segs[i][1]
            x_right_start = segs[i + 1][0]
            wid_left     = segs[i][2]
            if x_right_start <= x_left_end:
                continue   # overlapping — already one wall
            # Skip if this gap is already an intra-segment gap somewhere
            gap_mid = (x_left_end + x_right_start) // 2
            already = any(abs(op["x_px"] - gap_mid) < DEDUP_RADIUS_PX and abs(op["y_px"] - y) < DEDUP_RADIUS_PX
                          for op in openings)
            if already:
                continue
            # Scan the gap for bright pixels
            b_start = b_end = None
            for x in range(x_left_end, x_right_start + 1):
                v = _sample_wall_band(fp_gray, 'H', y, x)
                if v > OPENING_BRIGHT:
                    if b_start is None: b_start = x
                    b_end = x
            if b_start is None:
                continue
            gpx = b_end - b_start + 1
            if gpx < min_open_px or gpx > max_win_px:
                continue
            kind = "door" if gpx <= max_door_px else "window"
            emit_inter(wid_left, 'H', y, x_left_end, x_right_start,
                       b_start, b_end, kind)

    for x, segs in v_by_x.items():
        segs.sort()
        for i in range(len(segs) - 1):
            y_top_end    = segs[i][1]
            y_bot_start  = segs[i + 1][0]
            wid_left     = segs[i][2]
            if y_bot_start <= y_top_end:
                continue
            gap_mid = (y_top_end + y_bot_start) // 2
            already = any(abs(op["x_px"] - x) < DEDUP_RADIUS_PX and abs(op["y_px"] - gap_mid) < DEDUP_RADIUS_PX
                          for op in openings)
            if already:
                continue
            b_start = b_end = None
            for y in range(y_top_end, y_bot_start + 1):
                v = _sample_wall_band(fp_gray, 'V', x, y)
                if v > OPENING_BRIGHT:
                    if b_start is None: b_start = y
                    b_end = y
            if b_start is None:
                continue
            gpx = b_end - b_start + 1
            if gpx < min_open_px or gpx > max_win_px:
                continue
            kind = "door" if gpx <= max_door_px else "window"
            emit_inter(wid_left, 'V', x, y_top_end, y_bot_start,
                       b_start, b_end, kind)

    # ── PASS 3: T-junction gaps (inner wall endpoint near outer wall body) ──────
    # Pattern: an inner V-wall stub starts at y_start which is close to (but NOT
    # touching) the bottom edge of a horizontal outer wall.  The gap between the
    # outer wall's bottom face and the inner wall's top endpoint is a door opening.
    # Same logic applies in all four T-junction orientations.
    #
    # Strategy:
    #   For every V-wall endpoint (y_start or y_end), check if there is an H-wall
    #   whose centerline is within WALL_HALF_PX (≈ half wall thickness) of the
    #   endpoint, and if the gap between them (in image pixels) looks bright.
    #   Similarly for every H-wall endpoint near a V-wall.

    WALL_HALF_PX  = 20   # half wall thickness in pixels — endpoint tolerance
    T_GAP_MAX_PX  = int(MAX_DOOR_M * ppm) + WALL_HALF_PX   # maximum gap to scan
    T_GAP_MIN_PX  = max(2, int(MIN_OPENING_M * ppm))
    # Guard: reject T-junction gaps whose scan range touches the image edge.
    # Annotation tick marks at corners produce short stubs near the border;
    # their "gap" spans from the border all the way to the first real wall.
    # Any gap that starts within EDGE_GUARD px of an image edge is an artifact.
    # Use the dynamically measured border_crop so no values are hardcoded per image.
    EDGE_GUARD = border_crop if border_crop is not None else BORDER_CROP_PX

    # Build lookup: H-walls by y, V-walls by x (reuse above dicts)
    # h_by_y and v_by_x already built above

    def scan_tjunction_gap(orient_inner, fixed_inner, endpoint_px,
                            fixed_outer, a_outer, b_outer):
        """
        Scan the gap between inner wall endpoint and outer wall face for brightness.
        orient_inner: orientation of the inner wall ('V' or 'H')
        fixed_inner:  x (V-wall) or y (H-wall) coordinate of inner wall
        endpoint_px:  y (V) or x (H) of the inner wall's tip
        fixed_outer:  y (H-wall) or x (V-wall) of the outer wall
        a_outer, b_outer: range of outer wall along its axis
        Returns (bright_start, bright_end, kind) or None.
        """
        # Inner wall's tip must be within the outer wall's extent
        if orient_inner == 'V':
            # Inner=V at x=fixed_inner; tip at y=endpoint_px; outer=H at y=fixed_outer
            if not (a_outer - WALL_HALF_PX <= fixed_inner <= b_outer + WALL_HALF_PX):
                return None
            # Determine gap direction: is endpoint_px above or below outer wall?
            if endpoint_px < fixed_outer:
                # inner wall tip is ABOVE outer wall (gap runs from endpoint_px to fixed_outer)
                scan_start = endpoint_px
                scan_end   = fixed_outer + WALL_HALF_PX
            else:
                # inner wall tip is BELOW outer wall
                scan_start = fixed_outer - WALL_HALF_PX
                scan_end   = endpoint_px
            gap_range = scan_end - scan_start
            if gap_range < T_GAP_MIN_PX or gap_range > T_GAP_MAX_PX:
                return None
            # Reject gaps that touch the image edge (annotation tick mark artifacts)
            if scan_start < EDGE_GUARD or scan_end > FH - EDGE_GUARD:
                return None
            # Scan brightness along x=fixed_inner from scan_start to scan_end
            b_start = b_end = None
            for y in range(max(0, scan_start), min(FH - 1, scan_end) + 1):
                v = _sample_wall_band(fp_gray, 'V', fixed_inner, y)
                if v > OPENING_BRIGHT:
                    if b_start is None: b_start = y
                    b_end = y
            if b_start is None:
                return None
            gpx = b_end - b_start + 1
            if gpx < T_GAP_MIN_PX or gpx > T_GAP_MAX_PX:
                return None
            kind = "door" if gpx <= int(MAX_DOOR_M * ppm) else "window"
            return (b_start, b_end, kind)

        else:
            # orient_inner == 'H'
            # Inner=H at y=fixed_inner; tip at x=endpoint_px; outer=V at x=fixed_outer
            if not (a_outer - WALL_HALF_PX <= fixed_inner <= b_outer + WALL_HALF_PX):
                return None
            if endpoint_px < fixed_outer:
                scan_start = endpoint_px
                scan_end   = fixed_outer + WALL_HALF_PX
            else:
                scan_start = fixed_outer - WALL_HALF_PX
                scan_end   = endpoint_px
            gap_range = scan_end - scan_start
            if gap_range < T_GAP_MIN_PX or gap_range > T_GAP_MAX_PX:
                return None
            # Reject gaps that touch the image edge (annotation tick mark artifacts)
            if scan_start < EDGE_GUARD or scan_end > FW - EDGE_GUARD:
                return None
            b_start = b_end = None
            for x in range(max(0, scan_start), min(FW - 1, scan_end) + 1):
                v = _sample_wall_band(fp_gray, 'H', fixed_inner, x)
                if v > OPENING_BRIGHT:
                    if b_start is None: b_start = x
                    b_end = x
            if b_start is None:
                return None
            gpx = b_end - b_start + 1
            if gpx < T_GAP_MIN_PX or gpx > T_GAP_MAX_PX:
                return None
            kind = "door" if gpx <= int(MAX_DOOR_M * ppm) else "window"
            return (b_start, b_end, kind)

    # V-wall endpoints near H-walls
    DEDUP_RADIUS_PX = max(12, int(ppm * 0.5))   # ~0.5m; adapts to image scale
    h_wall_list = list(h_by_y.items())   # [(y, [(x1, x2, wid), ...]), ...]
    for x, v_segs in v_by_x.items():
        for y1, y2, _wid in v_segs:
            for endpoint_y in (y1, y2):
                # Find H-walls whose y is within WALL_HALF_PX of this endpoint
                for hy, h_segs in h_wall_list:
                    if abs(hy - endpoint_y) > WALL_HALF_PX:
                        continue
                    for hx1, hx2, h_wid in h_segs:
                        result = scan_tjunction_gap('V', x, endpoint_y, hy, hx1, hx2)
                        if result is None:
                            continue
                        b_start, b_end, kind = result
                        gap_mid_x = x
                        gap_mid_y = (b_start + b_end) // 2
                        # Dedup: skip if already have an opening very close
                        already = any(
                            abs(op["x_px"] - gap_mid_x) < DEDUP_RADIUS_PX and
                            abs(op["y_px"] - gap_mid_y) < DEDUP_RADIUS_PX
                            for op in openings
                        )
                        if already:
                            continue
                        emit_inter(_wid, 'V', x, endpoint_y, hy,
                                   b_start, b_end, kind)

    # H-wall endpoints near V-walls
    v_wall_list = list(v_by_x.items())   # [(x, [(y1, y2, wid), ...]), ...]
    for y, h_segs in h_by_y.items():
        for x1, x2, _wid in h_segs:
            for endpoint_x in (x1, x2):
                for vx, v_segs in v_wall_list:
                    if abs(vx - endpoint_x) > WALL_HALF_PX:
                        continue
                    for vy1, vy2, v_wid in v_segs:
                        result = scan_tjunction_gap('H', y, endpoint_x, vx, vy1, vy2)
                        if result is None:
                            continue
                        b_start, b_end, kind = result
                        gap_mid_x = (b_start + b_end) // 2
                        gap_mid_y = y
                        already = any(
                            abs(op["x_px"] - gap_mid_x) < DEDUP_RADIUS_PX and
                            abs(op["y_px"] - gap_mid_y) < DEDUP_RADIUS_PX
                            for op in openings
                        )
                        if already:
                            continue
                        emit_inter(_wid, 'H', y, endpoint_x, vx,
                                   b_start, b_end, kind)

    return openings


# ── Step 10 — room detection ──────────────────────────────────────────────────

def _detect_rooms(room_mask, wall_mask):
    """
    Seal doorway gaps (WALL_SEAL_PX close) then separate room regions.
    """
    k_seal    = cv2.getStructuringElement(cv2.MORPH_RECT, (WALL_SEAL_PX, WALL_SEAL_PX))
    sealed    = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k_seal)
    separated = cv2.bitwise_and(room_mask, cv2.bitwise_not(sealed))
    cleaned   = cv2.morphologyEx(separated, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    n, _, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
    rooms = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_ROOM_PX: continue
        sw = int(stats[i, cv2.CC_STAT_WIDTH])
        sh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(sw, sh) / max(min(sw, sh), 1) > MAX_ROOM_ASPECT: continue
        rooms.append({
            "cx_px": float(centroids[i][0]),
            "cy_px": float(centroids[i][1]),
            "area_px": area, "w_px": sw, "h_px": sh,
        })
    return rooms


# ── Coordinate converters ─────────────────────────────────────────────────────

def _to_seg(x1, y1, x2, y2, ppm, fp_h):
    """Pixel → metric Segment.  Y-flip: image-top = world-south."""
    return Segment(
        start=Point2D(x1/ppm, (fp_h-y1)/ppm),
        end=  Point2D(x2/ppm, (fp_h-y2)/ppm),
        layer="WALL", source_type="RASTER",
    )

def _opening_marker_seg(op, fp_h, ppm):
    """Zero-length marker Segment for door/window (world coords)."""
    wy = fp_h/ppm - op["y"]
    seg = Segment(
        start=Point2D(op["x"], wy),
        end=  Point2D(op["x"], wy),
        layer="DOOR" if op["kind"] == "door" else "WINDOW",
        source_type="RASTER_OPENING",
    )
    seg.width_m  = op["width_m"]
    seg.t_center = op["t_center"]
    seg.wall_id  = op["wall_id"]
    seg.orient   = op["orient"]
    return seg


# ── Debug visualisation ───────────────────────────────────────────────────────

def _debug_vis(fp, h_walls, v_walls, rooms, openings=None):
    vis = fp.copy() if fp.ndim == 3 else cv2.cvtColor(fp, cv2.COLOR_GRAY2BGR)
    for x1, y, x2, _ in h_walls:
        cv2.line(vis, (x1, y), (x2, y), (0, 220, 0), 2)
    for x, y1, _, y2 in v_walls:
        cv2.line(vis, (x, y1), (x, y2), (0, 160, 255), 2)
    for i, r in enumerate(rooms):
        cx, cy = int(r["cx_px"]), int(r["cy_px"])
        cv2.circle(vis, (cx, cy), 14, (220, 50, 50), -1)
        cv2.putText(vis, f"R{i+1}", (cx-8, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if openings:
        for op in openings:
            col = (30, 90, 255) if op["kind"] == "door" else (255, 130, 0)
            cv2.circle(vis, (op["x_px"], op["y_px"]), 9, col, -1)
            cv2.putText(vis, op["kind"][0].upper(),
                        (op["x_px"]+6, op["y_px"]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return vis


# ── Public API ────────────────────────────────────────────────────────────────

class RasterParser:
    def __init__(self,
                 pixels_per_meter: float = 0.0,   # 0 = auto-detect
                 pdf_dpi:          int   = 200,
                 # Legacy params (kept for API compatibility)
                 hough_threshold:  int   = HOUGH_TH,
                 hough_min_length: int   = HOUGH_MIN_LEN,
                 hough_max_gap:    int   = HOUGH_MAX_GAP):
        self._ppm_override = pixels_per_meter
        self.pdf_dpi       = pdf_dpi
        self._last_ppm     = FALLBACK_PPM

    @property
    def detected_ppm(self) -> float:
        return self._last_ppm

    def parse(self, filepath: str) -> ParsedGeometry:
        _req_cv()
        suffix = Path(filepath).suffix.lower()
        img    = _pdf_to_img(filepath, self.pdf_dpi) if suffix == ".pdf" else _load(filepath)
        OH, OW = img.shape[:2]

        fp, (cx0, cy0, cx1, cy1) = _autocrop(img)
        FH, FW = fp.shape[:2]
        fp_gray = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY) if fp.ndim == 3 else fp

        ppm = self._ppm_override if self._ppm_override > 0.0 else _detect_ppm(fp)
        self._last_ppm = ppm

        wall_mask, room_mask, border_crop = _segment(fp)

        h_walls, v_walls  = _detect_walls(wall_mask, border_crop)
        h_walls, v_walls  = _complete_outer_walls(h_walls, v_walls, FH, FW)
        h_walls, v_walls  = _filter_edge_stubs(h_walls, v_walls, FH, FW, ppm, border_crop)

        img_openings      = _detect_openings_from_image(fp_gray, h_walls, v_walls, ppm, border_crop)
        rooms_px          = _detect_rooms(room_mask, wall_mask)

        all_walls  = h_walls + v_walls
        wall_segs  = [_to_seg(x1, y1, x2, y2, ppm, FH) for x1, y1, x2, y2 in all_walls]

        door_segs, win_segs = [], []
        for op in img_openings:
            seg = _opening_marker_seg(op, FH, ppm)
            (door_segs if op["kind"] == "door" else win_segs).append(seg)

        result = ParsedGeometry()
        result.wall_segments   = wall_segs
        result.door_segments   = door_segs
        result.window_segments = win_segs

        xs = [c for s in wall_segs for c in (s.start.x, s.end.x)]
        ys = [c for s in wall_segs for c in (s.start.y, s.end.y)]
        if xs:
            result.bounds = {
                "minx": min(xs), "miny": min(ys),
                "maxx": max(xs), "maxy": max(ys),
            }
        result.units = "meters"
        result.metadata_extra = {
            "source":           "raster_v5",
            "original_size":    f"{OW}x{OH}px",
            "cropped_size":     f"{FW}x{FH}px",
            "crop_bbox":        [cx0, cy0, cx1, cy1],
            "pixels_per_meter": round(ppm, 2),
            "ppm_source":       "override" if self._ppm_override > 0 else "auto",
            "h_walls":          len(h_walls),
            "v_walls":          len(v_walls),
            "total_walls":      len(all_walls),
            "doors_detected":   len(door_segs),
            "windows_detected": len(win_segs),
            "rooms_detected":   len(rooms_px),
        }
        # Attach raw data for downstream pipeline use
        result._raster_openings = img_openings
        result._rooms_px        = rooms_px
        result._ppm             = ppm
        result._fp_h            = FH
        result._wall_mask       = wall_mask   # for per-wall thickness measurement

        return result

    def save_debug_image(self, filepath: str,
                         output_path: str = "debug_raster.png") -> str:
        _req_cv()
        suffix  = Path(filepath).suffix.lower()
        img     = _pdf_to_img(filepath, self.pdf_dpi) if suffix == ".pdf" else _load(filepath)
        fp, _   = _autocrop(img)
        FH, FW  = fp.shape[:2]
        fp_gray = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY) if fp.ndim == 3 else fp
        ppm     = self._ppm_override if self._ppm_override > 0 else _detect_ppm(fp)
        wm, rm, border_crop = _segment(fp)
        hw, vw  = _detect_walls(wm, border_crop)
        hw, vw  = _complete_outer_walls(hw, vw, FH, FW)
        hw, vw  = _filter_edge_stubs(hw, vw, FH, FW, ppm, border_crop)
        ops     = _detect_openings_from_image(fp_gray, hw, vw, ppm, border_crop)
        rooms   = _detect_rooms(rm, wm)
        vis     = _debug_vis(fp, hw, vw, rooms, ops)
        cv2.imwrite(output_path, vis)
        return output_path
