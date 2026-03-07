"""
Raster Parser v6 — Floor plan PNG/PDF → ParsedGeometry

Supported inputs
────────────────
  • CAD exports (green dimension annotations, walls gray 82-148, beige rooms)
  • SimpleDraw exports (high-contrast B&W, walls black 0-20, white background)
  • Screenshots of the floor-plan viewer (auto-cropped to the floor-plan panel)

Format auto-detection
─────────────────────
  • SimpleDraw: >80% white pixels, >3% black pixels, <5% mid-tone → use
    inverted thresholds (wall_lo=0, wall_hi=30; room_lo=220, room_hi=255).
  • CAD / coloured: use standard thresholds (wall 82-148, room 153-244).

Opening detection strategy
──────────────────────────
  Pass 1 (intra-segment): bright gaps WITHIN each wall segment.
    Endpoint guard: discard gaps that span the ENTIRE wall (i.e. gap covers
    ≥90% of wall length) — those are open wall ends, not real openings.

  Pass 2 (inter-segment): scan gaps BETWEEN collinear wall stubs on the same
    line — openings too wide for MERGE_GAP to bridge.

  T-junction filter: if a perpendicular wall crosses a gap → it is a structural
    junction, not an opening; skip it.

  Bbox filter: gaps whose centre lies outside the building bounding box are
    beyond the outer walls and are discarded.

Door vs Window classification (format-independent)
───────────────────────────────────────────────────
  Two complementary rules applied in order:

  1. Jamb rule — if the shorter of the two wall stubs adjacent to the gap is
     less than MIN_JAMB_M (0.50 m), the gap is a doorway.  Short stubs are
     door frames/jambs; this pattern appears in every floor-plan format without
     any format-specific tuning.

  2. Width rule — otherwise, gaps ≤ MAX_DOOR_M (1.40 m) are doors and gaps up
     to MAX_WINDOW_M (3.50 m) are windows.
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
# CAD format thresholds
WALL_LO, WALL_HI  = 82,  148
ROOM_LO, ROOM_HI  = 153, 244
# SimpleDraw B&W format thresholds
SD_WALL_LO, SD_WALL_HI = 0, 30
SD_ROOM_LO, SD_ROOM_HI = 220, 255
# Format detection
SD_WHITE_TH   = 0.80   # fraction of pixels > 200 to trigger SimpleDraw mode
SD_DARK_TH    = 0.02   # fraction of pixels < 30
SD_MID_TH     = 0.08   # fraction in mid-tone (30-200); SimpleDraw has almost none

GREEN_DIFF        = 22    # G-R and G-B delta for annotation suppression
BORDER_CROP_PX    = 36    # outer annotation-grid strip to ignore
SKEL_MAX_ITER     = 80
HOUGH_TH          = 7
HOUGH_MIN_LEN     = 12
HOUGH_MAX_GAP     = 14
AXIS_SNAP_TOL     = 12    # snap near-H/V lines to exact axes
COORD_GROUP_TOL   = 11    # group collinear lines within this px distance
MERGE_GAP         = 60    # max gap to bridge — covers doorway gaps (~50px at 61ppm)
DEDUP_DIST        = 35    # max px between parallel double-edges to collapse (thick walls ~34px)
MIN_WALL_PX       = 40    # discard wall segments shorter than this
WALL_SEAL_PX      = 48    # kernel for doorway-sealing before room detection
MIN_ROOM_PX       = 2000
MAX_ROOM_ASPECT   = 14.0
FALLBACK_PPM      = 65.0  # px/m when tick detection fails

# Opening detection
MIN_OPENING_M     = 0.35  # minimum gap width treated as an opening (m)
MAX_DOOR_M        = 1.40  # gaps wider than this → window unless jamb rule fires
MIN_JAMB_M        = 0.50  # if the shorter adjacent stub is under this → doorway
                           # (short stubs are door frames/jambs in any floor plan)
MAX_WINDOW_M      = 3.50  # gaps wider than this → noise, ignore
SCAN_HALF_PX      = 14    # half-width of wall-band scan
OPENING_BRIGHT    = 148   # mean brightness above this → opening pixel (CAD)
SD_OPENING_BRIGHT = 200   # SimpleDraw: white space threshold

# Window-symbol detection
WIN_SYM_MIN_BRIGHT_PX = 6   # min bright pixels between two dark stripes
WIN_SYM_MAX_TOTAL_M   = 3.0
WIN_SYM_MIN_TOTAL_M   = 0.3
WIN_SYM_DARK_TH       = 148


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


# ── Step 3 — segmentation ─────────────────────────────────────────────────────

def _detect_format(fp) -> str:
    """
    Return 'simpledraw' for high-contrast B&W drawings, 'cad' otherwise.

    SimpleDraw exports: ~92% white (>200), ~7% black (<30), <1% mid-tone.
    CAD exports: significant mid-tone (beige rooms + gray walls), coloured annotations.
    """
    gray = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY) if fp.ndim == 3 else fp
    total = gray.size
    pct_white = float((gray > 200).sum()) / total
    pct_dark  = float((gray < 30).sum())  / total
    pct_mid   = float(((gray >= 30) & (gray <= 200)).sum()) / total
    if pct_white >= SD_WHITE_TH and pct_dark >= SD_DARK_TH and pct_mid <= SD_MID_TH:
        return 'simpledraw'
    return 'cad'


def _segment(fp, fmt: str = 'cad'):
    """Return (wall_mask, room_mask) with annotation border stripped."""
    gray  = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY) if fp.ndim == 3 else fp
    FH, FW = gray.shape

    if fmt == 'simpledraw':
        # B&W format: walls are near-black, background is near-white
        wall_mask = cv2.inRange(gray, SD_WALL_LO, SD_WALL_HI)
        room_mask = cv2.inRange(gray, SD_ROOM_LO, SD_ROOM_HI)
        # No green annotation suppression needed (pure B&W)
        # No border crop for SimpleDraw (no annotation margins)
        return wall_mask, room_mask

    # CAD format: coloured, with green annotations and beige rooms
    if fp.ndim == 3:
        b, g, r = cv2.split(fp)
        gi     = g.astype(np.int16)
        green  = ((gi - r.astype(np.int16) > GREEN_DIFF) &
                  (gi - b.astype(np.int16) > GREEN_DIFF)).astype(np.uint8) * 255
    else:
        green  = np.zeros_like(gray)

    green  = cv2.dilate(green, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
    not_g  = cv2.bitwise_not(green)

    wall_mask = cv2.bitwise_and(cv2.inRange(gray, WALL_LO, WALL_HI), not_g)
    room_mask = cv2.bitwise_and(cv2.inRange(gray, ROOM_LO, ROOM_HI), not_g)

    inner = np.zeros((FH, FW), dtype=np.uint8)
    b2    = BORDER_CROP_PX
    inner[b2:FH-b2, b2:FW-b2] = 255
    wall_mask = cv2.bitwise_and(wall_mask, inner)
    room_mask = cv2.bitwise_and(room_mask, inner)
    return wall_mask, room_mask


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

def _merge(lines, orient, FH, FW, min_wall_px=None):
    if min_wall_px is None:
        min_wall_px = MIN_WALL_PX
    border = BORDER_CROP_PX
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
                if cb - ca >= min_wall_px:
                    result.append((ca, fixed, cb, fixed) if orient == 'H'
                                  else (fixed, ca, fixed, cb))
                ca, cb = a, b
        if cb - ca >= min_wall_px:
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


def _detect_walls(wall_mask, min_wall_px=None):
    """
    Detect walls from wall_mask.
    min_wall_px: minimum wall segment length in pixels.
                 Defaults to MIN_WALL_PX (40px).
                 Pass a larger value (e.g. ppm*1.0) for SimpleDraw to filter
                 door-arc artefacts which create 40-80px stub segments.
    """
    if min_wall_px is None:
        min_wall_px = MIN_WALL_PX
    FH, FW = wall_mask.shape
    skel   = _skeleton(wall_mask)
    raw    = cv2.HoughLinesP(skel, 1, math.pi/180,
                             threshold=HOUGH_TH,
                             minLineLength=HOUGH_MIN_LEN,
                             maxLineGap=HOUGH_MAX_GAP)
    if raw is None:
        return [], []
    axis  = [r for r in (_snap(*l[0]) for l in raw) if r]
    H     = _dedup(_merge([l for l in axis if l[4]=='H'], 'H', FH, FW, min_wall_px), 'H')
    V     = _dedup(_merge([l for l in axis if l[4]=='V'], 'V', FH, FW, min_wall_px), 'V')
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


def _find_gap_openings(fp_gray, orient, fixed, a, b, ppm, opening_bright=None):
    """
    Scan along wall from a→b.  Find contiguous bright runs (openings).

    Returns list of (gs, ge, gpx, kind) where kind is 'door' or 'window',
    classified by two format-independent rules:

      Jamb rule  — if the shorter of the two wall stubs adjacent to the gap
                   is < MIN_JAMB_M, the gap is a doorway regardless of width.
                   Short stubs are door frames; this pattern exists in every
                   floor-plan format without any format-specific tuning.

      Width rule — otherwise, gap ≤ MAX_DOOR_M → door; > MAX_DOOR_M → window.

    Endpoint guard: gaps spanning ≥90% of the wall AND touching an endpoint
    are discarded (open wall ends, not real openings).  Gaps at endpoints that
    don't span the whole wall are kept (real openings near corners).
    """
    if opening_bright is None:
        opening_bright = OPENING_BRIGHT
    min_px   = max(2, int(MIN_OPENING_M * ppm))
    max_win  = int(MAX_WINDOW_M * ppm)
    max_door = int(MAX_DOOR_M   * ppm)
    min_jamb = int(MIN_JAMB_M   * ppm)
    wall_len = b - a

    bright_run, gap_start = 0, None
    raw_gaps = []

    for i in range(a, b + 1):
        mean = _sample_wall_band(fp_gray, orient, fixed, i)
        is_open = mean > opening_bright
        if is_open:
            if gap_start is None: gap_start = i
            bright_run += 1
        else:
            if gap_start is not None and bright_run >= min_px:
                raw_gaps.append((gap_start, i - 1, bright_run))
            gap_start = None; bright_run = 0
    if gap_start is not None and bright_run >= min_px:
        raw_gaps.append((gap_start, b, bright_run))

    results = []
    for gs, ge, gpx in raw_gaps:
        if gpx > max_win:
            continue
        at_start    = gs <= a + 2
        at_end      = ge >= b - 2
        spans_whole = gpx >= wall_len * 0.90

        # Filter 1: gap spans ≥90% of wall and touches either endpoint → open end
        if spans_whole and (at_start or at_end):
            continue
        # Filter 2: gap reaches wall END (not start) and spans ≥60% → jamb stub
        if at_end and not at_start and gpx >= wall_len * 0.60:
            continue
        # Filter 3: gap touches wall START and spans ≥60% → jamb stub on far side
        if at_start and not at_end and gpx >= wall_len * 0.60:
            continue

        # Classify: width rule takes priority over jamb rule when gap clearly exceeds door max
        left_stub_px  = gs - a   # pixels from wall start to gap start
        right_stub_px = b - ge   # pixels from gap end to wall end
        min_stub_px   = min(left_stub_px, right_stub_px)
        # If gap width is clearly window-sized, don't let a missing jamb override it
        if gpx > max_door and min_stub_px < min_jamb:
            kind = "window"   # wide gap touching wall edge = inter-segment window
        elif min_stub_px < min_jamb or gpx <= max_door:
            kind = "door"
        else:
            kind = "window"
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


def _detect_openings_from_image(fp_gray, h_walls, v_walls, ppm,
                                opening_bright=None,
                                scan_h=None, scan_v=None,
                                bbox=None, fmt='cad'):
    """
    Returns list of opening dicts.

    opening_bright: brightness threshold (CAD: 148, SimpleDraw: 200).
    scan_h / scan_v: wall lists for inter-segment scanning; may include shorter
      stubs than h_walls/v_walls (the rendered walls).  Falls back to h/v_walls.
    bbox: (xmin, ymin, xmax, ymax) px — openings outside are discarded.

    Classification uses two format-independent rules (see module docstring):
      1. Jamb rule  — short adjacent stub → door
      2. Width rule — gap width vs MAX_DOOR_M / MAX_WINDOW_M
    """
    if opening_bright is None:
        opening_bright = OPENING_BRIGHT
    if scan_h is None:
        scan_h = h_walls
    if scan_v is None:
        scan_v = v_walls

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
        # Discard openings outside the building bounding box
        if bbox is not None:
            bx0, by0, bx1, by1 = bbox
            margin = 5  # px tolerance
            if xp < bx0 - margin or xp > bx1 + margin:
                return
            if yp < by0 - margin or yp > by1 + margin:
                return
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
        """
        gpx = bright_end - bright_start + 1
        wm  = gpx / ppm
        if orient == 'H':
            xp = (bright_start + bright_end) // 2; yp = fixed
            xm = xp / ppm;                         ym = yp / ppm
        else:
            xp = fixed; yp = (bright_start + bright_end) // 2
            xm = xp / ppm; ym = yp / ppm
        # Discard openings outside the building bounding box
        if bbox is not None:
            bx0, by0, bx1, by1 = bbox
            margin = 5
            if xp < bx0 - margin or xp > bx1 + margin:
                return
            if yp < by0 - margin or yp > by1 + margin:
                return
        side = room_side(orient, fixed, xp if orient == 'H' else yp)
        openings.append({
            "wall_id":    wall_id_left,
            "orient":     orient,
            "t_start":    -1.0,
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
        gaps = _find_gap_openings(fp_gray, 'H', y, x1, x2, ppm, opening_bright)
        for gs, ge, gpx, kind in gaps:
            emit(wall_id, 'H', y, x1, x2, gs, ge, gpx, kind)
        wall_id += 1

    for x, y1, _, y2 in v_walls:
        gaps = _find_gap_openings(fp_gray, 'V', x, y1, y2, ppm, opening_bright)
        for gs, ge, gpx, kind in gaps:
            emit(wall_id, 'V', x, y1, y2, gs, ge, gpx, kind)
        syms = _find_window_symbols(fp_gray, 'V', x, y1, y2, ppm) if fmt == 'cad' else []
        for gs, ge, gpx, kind in syms:
            already = any(
                abs(op["x_px"] - x) < 5 and abs(op["y_px"] - (gs + ge) // 2) < gpx
                for op in openings if op["wall_id"] == wall_id
            )
            if not already:
                emit(wall_id, 'V', x, y1, y2, gs, ge, gpx, kind)
        wall_id += 1

    # ── PASS 2: Inter-segment gaps (collinear wall stubs) ────────────────────
    from collections import defaultdict
    max_door_px = int(MAX_DOOR_M    * ppm)
    max_win_px  = int(MAX_WINDOW_M  * ppm)
    min_open_px = max(2, int(MIN_OPENING_M * ppm))
    min_jamb_px = int(MIN_JAMB_M    * ppm)

    h_by_y = defaultdict(list)
    _wid = 0
    for x1, y, x2, _ in scan_h:
        canonical_y = y
        for existing_y in h_by_y:
            if abs(existing_y - y) <= COORD_GROUP_TOL:
                canonical_y = existing_y
                break
        h_by_y[canonical_y].append((x1, x2, x2 - x1, _wid))
        _wid += 1

    v_by_x = defaultdict(list)
    for x, y1, _, y2 in scan_v:
        # Snap to nearest existing key within COORD_GROUP_TOL to handle sub-pixel
        # jitter between collinear stubs detected at slightly different x positions.
        canonical_x = x
        for existing_x in v_by_x:
            if abs(existing_x - x) <= COORD_GROUP_TOL:
                canonical_x = existing_x
                break
        v_by_x[canonical_x].append((y1, y2, y2 - y1, _wid))
        _wid += 1

    for y, segs in h_by_y.items():
        segs.sort()
        for i in range(len(segs) - 1):
            x_left_end    = segs[i][1]
            x_right_start = segs[i + 1][0]
            left_len_px   = segs[i][2]
            right_len_px  = segs[i + 1][2]
            wid_left      = segs[i][3]
            if x_right_start <= x_left_end:
                # Stubs overlap — scan combined range for a large bright gap
                # (can happen when a thick-wall skeleton segment crosses a window opening)
                combined_start = segs[i][0]
                combined_end   = segs[i + 1][1]
                gap_mid = (combined_start + combined_end) // 2
                if any(abs(op["x_px"] - gap_mid) < 10 and abs(op["y_px"] - y) < 10
                       for op in openings):
                    continue
                b_start = b_end = None; bright_run = 0
                best_gs = best_ge = best_gpx = None
                for x in range(combined_start, combined_end + 1):
                    v = _sample_wall_band(fp_gray, 'H', y, x)
                    if v > opening_bright:
                        if b_start is None: b_start = x
                        b_end = x; bright_run += 1
                    else:
                        if b_start is not None and bright_run >= min_open_px:
                            if best_gpx is None or bright_run > best_gpx:
                                best_gs, best_ge, best_gpx = b_start, b_end, bright_run
                        b_start = b_end = None; bright_run = 0
                if b_start is not None and bright_run >= min_open_px:
                    if best_gpx is None or bright_run > best_gpx:
                        best_gs, best_ge, best_gpx = b_start, b_end, bright_run
                if best_gs is None or best_gpx > max_win_px:
                    continue
                kind = "window" if best_gpx > max_door_px else "door"
                emit_inter(wid_left, 'H', y, combined_start, combined_end,
                           best_gs, best_ge, kind)
                continue
            # Skip T-junctions: a perpendicular (V) wall crosses through this gap
            if any(x_left_end < vx <= x_right_start
                   for vx, vy1, _, vy2 in scan_v if vy1 <= y <= vy2):
                continue
            gap_mid = (x_left_end + x_right_start) // 2
            if any(abs(op["x_px"] - gap_mid) < 10 and abs(op["y_px"] - y) < 10
                   for op in openings):
                continue
            b_start = b_end = None
            for x in range(x_left_end, x_right_start + 1):
                v = _sample_wall_band(fp_gray, 'H', y, x)
                if v > opening_bright:
                    if b_start is None: b_start = x
                    b_end = x
            if b_start is None:
                continue
            gpx = b_end - b_start + 1
            if gpx < min_open_px or gpx > max_win_px:
                continue
            min_stub = min(left_len_px, right_len_px)
            kind = "door" if (min_stub < min_jamb_px or gpx <= max_door_px) else "window"
            emit_inter(wid_left, 'H', y, x_left_end, x_right_start,
                       b_start, b_end, kind)

    for x, segs in v_by_x.items():
        segs.sort()
        for i in range(len(segs) - 1):
            y_top_end    = segs[i][1]
            y_bot_start  = segs[i + 1][0]
            top_len_px   = segs[i][2]
            bot_len_px   = segs[i + 1][2]
            wid_left     = segs[i][3]
            if y_bot_start <= y_top_end:
                # Stubs overlap — scan combined range for a large bright gap
                combined_start = segs[i][0]
                combined_end   = segs[i + 1][1]
                gap_mid = (combined_start + combined_end) // 2
                if any(abs(op["x_px"] - x) < 10 and abs(op["y_px"] - gap_mid) < 10
                       for op in openings):
                    continue
                b_start = b_end = None; bright_run = 0
                best_gs = best_ge = best_gpx = None
                for y in range(combined_start, combined_end + 1):
                    v = _sample_wall_band(fp_gray, 'V', x, y)
                    if v > opening_bright:
                        if b_start is None: b_start = y
                        b_end = y; bright_run += 1
                    else:
                        if b_start is not None and bright_run >= min_open_px:
                            if best_gpx is None or bright_run > best_gpx:
                                best_gs, best_ge, best_gpx = b_start, b_end, bright_run
                        b_start = b_end = None; bright_run = 0
                if b_start is not None and bright_run >= min_open_px:
                    if best_gpx is None or bright_run > best_gpx:
                        best_gs, best_ge, best_gpx = b_start, b_end, bright_run
                if best_gs is None or best_gpx > max_win_px:
                    continue
                kind = "window" if best_gpx > max_door_px else "door"
                emit_inter(wid_left, 'V', x, combined_start, combined_end,
                           best_gs, best_ge, kind)
                continue
            # Skip T-junctions: a perpendicular (H) wall crosses through this gap
            if any(y_top_end < hy <= y_bot_start
                   for hx1, hy, hx2, _ in scan_h if hx1 <= x <= hx2):
                continue
            gap_mid = (y_top_end + y_bot_start) // 2
            if any(abs(op["x_px"] - x) < 10 and abs(op["y_px"] - gap_mid) < 10
                   for op in openings):
                continue
            b_start = b_end = None
            for y in range(y_top_end, y_bot_start + 1):
                v = _sample_wall_band(fp_gray, 'V', x, y)
                if v > opening_bright:
                    if b_start is None: b_start = y
                    b_end = y
            if b_start is None:
                continue
            gpx = b_end - b_start + 1
            if gpx < min_open_px or gpx > max_win_px:
                continue
            min_stub = min(top_len_px, bot_len_px)
            kind = "door" if (min_stub < min_jamb_px or gpx <= max_door_px) else "window"
            emit_inter(wid_left, 'V', x, y_top_end, y_bot_start,
                       b_start, b_end, kind)

    # ── Deduplicate: remove openings at same position ────────────────────────
    # Can arise when thick walls produce two face-lines, each independently detecting
    # the same opening. Use a generous radius that adapts to image scale.
    FINAL_DEDUP_PX = max(20, int(ppm * 0.35))   # ~35cm in image pixels
    deduped = []
    for op in openings:
        duplicate = any(
            abs(op["x_px"] - ex["x_px"]) < FINAL_DEDUP_PX and
            abs(op["y_px"] - ex["y_px"]) < FINAL_DEDUP_PX and
            op["kind"] == ex["kind"]
            for ex in deduped
        )
        if not duplicate:
            deduped.append(op)

    return deduped


def _estimate_ppm_simpledraw(fp_gray) -> float:
    """
    Estimate pixels-per-metre for SimpleDraw exports.
    SimpleDraw walls are drawn at a fixed architectural scale.
    We detect the median wall thickness (30px is typical) and assume
    walls are 0.30m thick → ppm = wall_thickness_px / 0.30.
    Falls back to FALLBACK_PPM if detection fails.
    """
    wall_mask = cv2.inRange(fp_gray, SD_WALL_LO, SD_WALL_HI)
    FH, FW = fp_gray.shape
    thicknesses = []
    # Sample multiple horizontal and vertical profiles
    for y in [FH//4, FH//3, FH//2, 2*FH//3, 3*FH//4]:
        row = wall_mask[y, :]
        run, start = 0, None
        for x in range(FW):
            if row[x] > 0:
                if start is None: start = x
                run += 1
            else:
                if start is not None and run > 3:
                    thicknesses.append(run)
                run, start = 0, None
    for x in [FW//4, FW//3, FW//2, 2*FW//3, 3*FW//4]:
        col = wall_mask[:, x]
        run, start = 0, None
        for y in range(FH):
            if col[y] > 0:
                if start is None: start = y
                run += 1
            else:
                if start is not None and run > 3:
                    thicknesses.append(run)
                run, start = 0, None
    if not thicknesses:
        return FALLBACK_PPM
    # Use median of measured thicknesses; typical architectural wall = 0.30m
    med = sorted(thicknesses)[len(thicknesses) // 2]
    wall_m = 0.30   # assumed standard wall thickness
    ppm = med / wall_m
    # Clamp to plausible range
    return max(40.0, min(200.0, round(ppm, 2)))


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

        # ── Format detection ──────────────────────────────────────────────────
        fmt = _detect_format(fp)
        is_simpledraw = (fmt == 'simpledraw')

        # Choose brightness threshold for opening detection
        ob = SD_OPENING_BRIGHT if is_simpledraw else OPENING_BRIGHT

        # PPM: SimpleDraw has no tick annotations; estimate from wall thickness
        if self._ppm_override > 0.0:
            ppm = self._ppm_override
        elif is_simpledraw:
            ppm = _estimate_ppm_simpledraw(fp_gray)
        else:
            ppm = _detect_ppm(fp)
        self._last_ppm = ppm

        wall_mask, room_mask = _segment(fp, fmt)

        if is_simpledraw:
            # Two-pass wall detection for SimpleDraw:
            #   Render walls: min 1.0m (100px) — suppresses door-arc artefacts
            #   Scan walls:   min 0.4m ( 40px) — keeps short door-frame stubs
            #     needed for inter-segment gap detection (left wall top stub = 45px)
            min_wp_render = int(ppm * 1.0)
            min_wp_scan   = int(ppm * 0.4)
            h_walls, v_walls       = _detect_walls(wall_mask, min_wall_px=min_wp_render)
            h_scan,  v_scan        = _detect_walls(wall_mask, min_wall_px=min_wp_scan)
        else:
            h_walls, v_walls       = _detect_walls(wall_mask)
            h_scan,  v_scan        = h_walls, v_walls

        h_walls, v_walls  = _complete_outer_walls(h_walls, v_walls, FH, FW)
        h_scan,  v_scan   = _complete_outer_walls(h_scan,  v_scan,  FH, FW)

        # Building bounding box (pixel coords) — used to discard openings whose
        # centre lies outside the building perimeter (e.g. gaps beyond outer walls).
        all_scan_coords_x = [c for x1,y,x2,_ in h_scan for c in (x1, x2)] + \
                            [x for x,y1,_,y2 in v_scan]
        all_scan_coords_y = [y for x1,y,x2,_ in h_scan] + \
                            [c for x,y1,_,y2 in v_scan for c in (y1, y2)]
        bbox = (min(all_scan_coords_x, default=0),
                min(all_scan_coords_y, default=0),
                max(all_scan_coords_x, default=FW),
                max(all_scan_coords_y, default=FH)) if all_scan_coords_x else None

        img_openings = _detect_openings_from_image(
            fp_gray, h_walls, v_walls, ppm, opening_bright=ob,
            scan_h=h_scan, scan_v=v_scan,
            bbox=bbox, fmt=fmt,
        )
        rooms_px = _detect_rooms(room_mask, wall_mask)

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
            "source":           "raster_v6",
            "format":           fmt,
            "original_size":    f"{OW}x{OH}px",
            "cropped_size":     f"{FW}x{FH}px",
            "crop_bbox":        [cx0, cy0, cx1, cy1],
            "pixels_per_meter": round(ppm, 2),
            "ppm_source":       "override" if self._ppm_override > 0 else ("simpledraw_est" if is_simpledraw else "auto"),
            "h_walls":          len(h_walls),
            "v_walls":          len(v_walls),
            "total_walls":      len(all_walls),
            "doors_detected":   len(door_segs),
            "windows_detected": len(win_segs),
            "rooms_detected":   len(rooms_px),
        }
        result._raster_openings = img_openings
        result._rooms_px        = rooms_px
        result._ppm             = ppm
        result._fp_h            = FH

        return result

    def save_debug_image(self, filepath: str,
                         output_path: str = "debug_raster.png") -> str:
        _req_cv()
        suffix  = Path(filepath).suffix.lower()
        img     = _pdf_to_img(filepath, self.pdf_dpi) if suffix == ".pdf" else _load(filepath)
        fp, _   = _autocrop(img)
        FH, FW  = fp.shape[:2]
        fp_gray = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY) if fp.ndim == 3 else fp
        fmt     = _detect_format(fp)
        ob      = SD_OPENING_BRIGHT if fmt == 'simpledraw' else OPENING_BRIGHT
        ppm     = (self._ppm_override if self._ppm_override > 0
                   else _estimate_ppm_simpledraw(fp_gray) if fmt == 'simpledraw'
                   else _detect_ppm(fp))
        wm, rm  = _segment(fp, fmt)
        min_wp  = int(ppm * 1.0) if fmt == 'simpledraw' else None
        hw, vw  = _detect_walls(wm, min_wall_px=min_wp)
        hw, vw  = _complete_outer_walls(hw, vw, FH, FW)
        ops     = _detect_openings_from_image(fp_gray, hw, vw, ppm, opening_bright=ob)
        rooms   = _detect_rooms(rm, wm)
        vis     = _debug_vis(fp, hw, vw, rooms, ops)
        cv2.imwrite(output_path, vis)
        return output_path
