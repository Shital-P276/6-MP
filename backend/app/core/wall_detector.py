"""
Wall Detector v4 — clean, correct, verified

Key insight from debugging:
  - Our generator exports single centerlines (after wall deduplication)
  - Collinear merging destroys clean DXFs by joining touching wall segments
    that happen to be collinear (e.g., bottom of living room + bottom of hallway)
  - Solution: skip collinear merging for DXF input; keep it only for raster/image
    input where lines may genuinely be broken mid-segment by noise

Pipeline:
  1. Scale inference and application
  2. Filter tiny segments (< 0.1m)
  3. Attempt parallel pairing (for external double-line DXFs, MAX_PAIR_DIST=0.8m)
  4. All remaining → single-line walls with default thickness (our generator, most DXFs)

For raster/image input, collinear merging is applied before step 3.
"""
from __future__ import annotations
from dataclasses import dataclass
import math, collections
from .dxf_parser import Segment, Point2D, ParsedGeometry

MIN_WALL_LENGTH    = 0.1
MAX_PAIR_DIST      = 0.8   # max gap for double-line pairing — tight to avoid false positives
MIN_PAIR_DIST      = 0.01
PARALLEL_TOL_DEG   = 5.0
AXIAL_OVERLAP_TOL  = 0.3
DEFAULT_HEIGHT     = 3.0
DEFAULT_THICKNESS  = 0.2

# Raster-only: merge broken fragments. Tight thresholds so clean DXFs are unaffected.
RASTER_COLLINEAR_PERP_TOL = 0.015  # 1.5cm
RASTER_COLLINEAR_GAP_TOL  = 0.5    # 50cm gap bridge

# Wall thickness measurement from pixel mask
THICK_SCAN_HALF   = 15    # ±px perpendicular scan radius (not 30 — too wide picks up floor)
THICK_SCAN_STEPS  = 9     # sample points along wall (skip 15% from each end avoids corners)
THICK_WALL_LO     = 82    # gray lower bound for wall pixel (matches raster_parser WALL_LO)
THICK_WALL_HI     = 148   # gray upper bound for wall pixel (matches raster_parser WALL_HI)
THICK_MIN_M       = 0.05  # minimum allowed thickness (metres)
THICK_MAX_M       = 0.55  # maximum allowed thickness (metres)


def _len(s):  return math.hypot(s.end.x-s.start.x, s.end.y-s.start.y)
def _angle(s): return math.degrees(math.atan2(s.end.y-s.start.y, s.end.x-s.start.x)) % 180
def _mid(s):  return Point2D((s.start.x+s.end.x)/2, (s.start.y+s.end.y)/2)

def _unit(s):
    dx=s.end.x-s.start.x; dy=s.end.y-s.start.y; L=math.hypot(dx,dy)
    return (dx/L,dy/L) if L>1e-9 else (1.,0.)

def _perp_dist(a,b):
    ux,uy=_unit(a); nx,ny=-uy,ux; ma,mb=_mid(a),_mid(b)
    return abs((mb.x-ma.x)*nx+(mb.y-ma.y)*ny)

def _axial_proj(a,pt):
    ux,uy=_unit(a); return (pt.x-a.start.x)*ux+(pt.y-a.start.y)*uy

def _axial_overlap(a,b,tol=AXIAL_OVERLAP_TOL):
    a0,a1=sorted([_axial_proj(a,a.start),_axial_proj(a,a.end)])
    b0,b1=sorted([_axial_proj(a,b.start),_axial_proj(a,b.end)])
    return a1+tol>=b0 and b1+tol>=a0

def _angle_diff(a,b):
    d=abs(_angle(a)-_angle(b)); return min(d,180-d)

def _centerline(a,b):
    if _axial_proj(a,b.start) > _axial_proj(a,b.end):
        b=Segment(b.end,b.start,b.layer,b.source_type)
    return (
        Point2D((a.start.x+b.start.x)/2,(a.start.y+b.start.y)/2),
        Point2D((a.end.x+b.end.x)/2,    (a.end.y+b.end.y)/2),
    )

def infer_scale(bounds):
    if not bounds: return 1.0
    span=max(bounds.get("maxx",0)-bounds.get("minx",0),
             bounds.get("maxy",0)-bounds.get("miny",0))
    if span<=0: return 1.0
    if span<=100: return 1.0
    if span<=2000: return 0.0254
    return 0.001


@dataclass
class Wall:
    start:     Point2D
    end:       Point2D
    thickness: float = DEFAULT_THICKNESS
    height:    float = DEFAULT_HEIGHT
    layer:     str   = "WALL"
    paired:    bool  = False
    confidence:float = 1.0

    @property
    def length(self): return math.hypot(self.end.x-self.start.x,self.end.y-self.start.y)
    @property
    def angle_deg(self): return math.degrees(math.atan2(self.end.y-self.start.y,self.end.x-self.start.x))

    def to_dict(self):
        return {"start":{"x":round(self.start.x,4),"y":round(self.start.y,4)},
                "end":{"x":round(self.end.x,4),"y":round(self.end.y,4)},
                "thickness":round(self.thickness,4),"height":round(self.height,4),
                "length":round(self.length,4),"paired":self.paired,
                "confidence":round(self.confidence,3)}


def merge_collinear_fragments(segs, perp_tol=RASTER_COLLINEAR_PERP_TOL, gap_tol=RASTER_COLLINEAR_GAP_TOL):
    """Only for raster input — join broken line fragments from Hough detection."""
    used=set(); result=[]
    for i,sa in enumerate(segs):
        if i in used: continue
        chain=[sa]; cids={i}; chg=True
        while chg:
            chg=False; tail=chain[-1]
            for j,sb in enumerate(segs):
                if j in used or j in cids: continue
                if _angle_diff(tail,sb)>PARALLEL_TOL_DEG: continue
                if _perp_dist(tail,sb)>perp_tol: continue
                ce=tail.end
                ds=math.hypot(ce.x-sb.start.x,ce.y-sb.start.y)
                de=math.hypot(ce.x-sb.end.x,  ce.y-sb.end.y)
                if min(ds,de)>gap_tol: continue
                if de<ds: sb=Segment(sb.end,sb.start,sb.layer,sb.source_type)
                chain.append(sb); cids.add(j); chg=True; break
        used.update(cids)
        merged=Segment(chain[0].start,chain[-1].end,chain[0].layer,chain[0].source_type)
        if _len(merged)>1e-6: result.append(merged)
    return result


def pair_double_lines(segs, height, default_thickness):
    """Try pairing parallel lines for external double-line DXFs."""
    used=set(); walls=[]
    buckets=collections.defaultdict(list)
    for i,s in enumerate(segs):
        b=int(_angle(s)/5); buckets[b].append(i); buckets[(b+1)%36].append(i)
    for i,sa in enumerate(segs):
        if i in used: continue
        best_j=None; best_d=float("inf")
        for j in buckets.get(int(_angle(sa)/5),[]):
            if j<=i or j in used: continue
            sb=segs[j]
            if _angle_diff(sa,sb)>PARALLEL_TOL_DEG: continue
            d=_perp_dist(sa,sb)
            if d<MIN_PAIR_DIST or d>MAX_PAIR_DIST: continue
            if not _axial_overlap(sa,sb): continue
            if d<best_d: best_d=d; best_j=j
        if best_j is not None:
            used.add(i); used.add(best_j)
            cs,ce=_centerline(sa,segs[best_j])
            ra=min(_len(sa),_len(segs[best_j]))/(max(_len(sa),_len(segs[best_j]))+1e-9)
            walls.append(Wall(start=cs,end=ce,thickness=round(best_d,4),height=height,
                              layer=sa.layer,paired=True,confidence=round(min(1.,0.7+0.3*ra),3)))
    unpaired=[segs[i] for i in range(len(segs)) if i not in used]
    return walls,unpaired


def measure_wall_thicknesses(walls: list, wall_mask, ppm: float) -> list:
    """
    Refine wall.thickness by measuring the actual pixel width of each wall
    in the raster wall_mask, perpendicular to the wall's centerline.

    Algorithm per wall:
      1. Sample THICK_SCAN_STEPS evenly-spaced points along the wall (15%-85%)
      2. At each sample, scan ±THICK_SCAN_HALF pixels perpendicular to the wall
         in IMAGE space (note: image Y is flipped vs world Y)
      3. Count the longest consecutive run of wall-range pixels (THICK_WALL_LO..HI)
         — uses run-length, not span, to avoid counting gaps in wall body
      4. Take the median run across all sample points → wall thickness in pixels
      5. Convert px→metres; clamp to [THICK_MIN_M, THICK_MAX_M]

    Returns a new list of Wall objects with updated thickness values.
    """
    import numpy as np

    if wall_mask is None or ppm < 1.0:
        return walls

    FH, FW = wall_mask.shape[:2]
    # Pre-read wall_mask as numpy for fast access
    wm = (wall_mask > 0) if wall_mask.dtype != bool else wall_mask

    updated = []
    for wall in walls:
        L = wall.length
        if L < 1e-6:
            updated.append(wall)
            continue

        # Unit vector along wall (world space, Y increases upward)
        dx = (wall.end.x - wall.start.x) / L
        dy = (wall.end.y - wall.start.y) / L

        # Perpendicular in world space: rotate 90° = (-dy, dx)
        # Convert to image space: img_x = world_x * ppm, img_y = FH - world_y * ppm
        # So img_perp_x = -dy (same), img_perp_y = -dx  (Y is FLIPPED)
        img_px = float(-dy)   # image x-component of perp direction
        img_py = float(-dx)   # image y-component of perp direction (flip: -dy→-dx wait...)
        # Correct derivation:
        #   world perp = (-dy, dx)
        #   image perp x = world_perp_x (no flip on x) = -dy
        #   image perp y = -world_perp_y (flip y)      = -dx
        img_px = float(-dy)
        img_py = float(-dx)

        # Sample points along wall (skip corners)
        runs = []
        for step in range(THICK_SCAN_STEPS):
            t = 0.15 + (step / max(THICK_SCAN_STEPS - 1, 1)) * 0.70
            # World coords of sample point
            wx = wall.start.x + t * L * dx
            wy = wall.start.y + t * L * dy
            # Image coords
            cx_img = wx * ppm
            cy_img = FH - wy * ppm

            # Scan perpendicular in image space
            best_run = 0
            run = 0
            for k in range(-THICK_SCAN_HALF, THICK_SCAN_HALF + 1):
                ix = int(round(cx_img + k * img_px))
                iy = int(round(cy_img + k * img_py))
                if 0 <= ix < FW and 0 <= iy < FH and wm[iy, ix]:
                    run += 1
                    if run > best_run:
                        best_run = run
                else:
                    run = 0
            if best_run > 0:
                runs.append(best_run)

        if not runs:
            updated.append(wall)
            continue

        # Median run → metres
        runs.sort()
        median_px = runs[len(runs) // 2]
        measured_m = median_px / ppm

        # Use measured value directly; clamp to physical limits.
        # We do NOT blend with default_thickness here — blending would collapse
        # the distinction between thin interior walls and thick exterior walls,
        # which is the primary value of per-wall measurement.
        new_thick = max(THICK_MIN_M, min(THICK_MAX_M, measured_m))

        # Rebuild wall with new thickness (Wall is a dataclass — replace field)
        updated.append(Wall(
            start=wall.start, end=wall.end,
            thickness=round(new_thick, 4),
            height=wall.height,
            layer=wall.layer,
            paired=wall.paired,
            confidence=wall.confidence,
        ))

    return updated


class WallDetector:
    def __init__(self, scale=1.0, auto_scale=True,
                 default_thickness=DEFAULT_THICKNESS, default_height=DEFAULT_HEIGHT,
                 is_raster=False):
        self.scale=scale; self.auto_scale=auto_scale
        self.default_thickness=default_thickness; self.default_height=default_height
        self.is_raster=is_raster; self._applied_scale=scale

    @property
    def applied_scale(self): return self._applied_scale

    def detect(self, geometry: ParsedGeometry) -> list[Wall]:
        segs=list(geometry.wall_segments)

        if self.auto_scale and self.scale==1.0:
            self._applied_scale=infer_scale(geometry.bounds)
        else:
            self._applied_scale=self.scale

        if self._applied_scale!=1.0:
            s=self._applied_scale
            segs=[Segment(Point2D(sg.start.x*s,sg.start.y*s),
                          Point2D(sg.end.x*s,  sg.end.y*s),
                          sg.layer,sg.source_type) for sg in segs]
            if geometry.bounds:
                geometry.bounds={k:v*s for k,v in geometry.bounds.items()}

        segs=[s for s in segs if _len(s)>=MIN_WALL_LENGTH]

        # Raster only: merge broken fragments from Hough detection
        if self.is_raster:
            segs=merge_collinear_fragments(segs)

        # Try double-line pairing (external CAD DXFs)
        walls,unpaired=pair_double_lines(segs,self.default_height,self.default_thickness)

        # Single-line walls (our generator + most DXFs)
        for seg in unpaired:
            if _len(seg)>=MIN_WALL_LENGTH:
                walls.append(Wall(start=seg.start,end=seg.end,
                                  thickness=self.default_thickness,
                                  height=self.default_height,
                                  layer=seg.layer,paired=False,confidence=0.8))

        # Raster only: refine wall thickness from pixel mask
        if self.is_raster:
            wall_mask = getattr(geometry, '_wall_mask', None)
            ppm       = getattr(geometry, '_ppm', 0.0)
            if wall_mask is not None and ppm > 1.0:
                walls = measure_wall_thicknesses(walls, wall_mask, ppm)

        return walls
