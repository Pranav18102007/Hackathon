"""
Grid generation, C4 rotational symmetry, symmetry detection, and loop analysis
for kolam strokes. Built on top of the frozen shared model in core/model.py.
"""

import math
from typing import Dict, List, Optional, Tuple

from core.model import DotGrid, Stroke, Symmetry

# Number of decimal places used whenever we turn a floating-point (x, y) point
# into a hashable dict/set key. Rotated points are the result of float
# arithmetic, so two points that are "the same dot" on paper can differ by a
# tiny epsilon (e.g. 19.999999999998 vs 20.0). Rounding first makes point
# comparisons/lookups reliable.
_COORD_DECIMALS = 4


def _round_pt(p: Tuple[float, float], decimals: int = _COORD_DECIMALS) -> Tuple[float, float]:
    return (round(p[0], decimals), round(p[1], decimals))


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_dot_grid(rows: int, cols: int, spacing: float, layout: str = "square") -> DotGrid:
    """Build a DotGrid and populate its `dots` list with (x, y) positions.

    - "square":     plain rows x cols grid, dots at (c*spacing, r*spacing).
    - "rhombus":    the same square grid rotated 45 degrees, so the overall
                    silhouette becomes a diamond. Kolam artists often walk
                    such grids along the diagonals.
    - "triangular": alternating rows are offset by half a spacing unit and
                    rows are packed closer together (by a factor of sqrt(3)/2)
                    so every dot is equidistant from its neighbours, forming
                    equilateral triangles instead of squares.
    """
    dots: List[Tuple[float, float]] = []

    if layout == "square":
        for r in range(rows):
            for c in range(cols):
                dots.append((c * spacing, r * spacing))

    elif layout == "rhombus":
        # Standard 2D rotation by 45 degrees: (x, y) -> (x*cos45 - y*sin45, x*sin45 + y*cos45).
        # cos45 == sin45 == 1/sqrt(2), so this simplifies to:
        #   x' = (x - y) / sqrt(2)
        #   y' = (x + y) / sqrt(2)
        # Rotation preserves distances, so neighbouring dots stay `spacing`
        # apart -- only the overall shape (a diamond) changes, not the density.
        k = 1.0 / math.sqrt(2)
        for r in range(rows):
            for c in range(cols):
                x, y = c * spacing, r * spacing
                dots.append(((x - y) * k, (x + y) * k))

    elif layout == "triangular":
        # Equilateral-triangle lattice: shifting every other row by
        # spacing/2 and compressing row height to spacing*sqrt(3)/2 makes
        # each dot exactly `spacing` away from its 6 neighbours (instead of
        # the 4 neighbours you get in a square grid).
        row_height = spacing * math.sqrt(3) / 2
        for r in range(rows):
            offset = (spacing / 2) if (r % 2 == 1) else 0.0
            for c in range(cols):
                dots.append((c * spacing + offset, r * row_height))

    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    return DotGrid(rows=rows, cols=cols, spacing=spacing, layout=layout, dots=dots)


def _grid_center(grid: DotGrid) -> Tuple[float, float]:
    """Center of the grid's bounding box -- the pivot every rotation/mirror
    is performed around."""
    if grid.dots:
        xs = [p[0] for p in grid.dots]
        ys = [p[1] for p in grid.dots]
        return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    # Fallback if dots haven't been generated yet: assume a plain square grid.
    return ((grid.cols - 1) * grid.spacing / 2, (grid.rows - 1) * grid.spacing / 2)


# ---------------------------------------------------------------------------
# C4 symmetry (rotate by 90 degrees, 4 times, to fill the whole grid)
# ---------------------------------------------------------------------------

def _rotate_point_90k(point: Tuple[float, float], center: Tuple[float, float], k: int) -> Tuple[float, float]:
    """Rotate `point` around `center` by k * 90 degrees, counter-clockwise.

    Kolam symmetry only ever needs quarter-turns, so instead of calling
    cos()/sin() (which introduces tiny floating point errors even for "nice"
    angles like 90 degrees) we use the exact identity for a 90 degree
    rotation of a vector: (dx, dy) -> (-dy, dx). Applying that k times is the
    same as rotating by k * 90 degrees, and every step is just a sign flip +
    swap, so there is zero trig error -- the strokes line up exactly when
    the pieces are stitched back together into the C4 pattern.
    """
    cx, cy = center
    dx, dy = point[0] - cx, point[1] - cy
    for _ in range(k % 4):
        dx, dy = -dy, dx
    return (cx + dx, cy + dy)


def apply_c4_symmetry(strokes: List[Stroke], grid: DotGrid) -> List[Stroke]:
    """Take strokes drawn in a single quadrant and replicate them by
    rotating 0, 90, 180 and 270 degrees about the grid's center, returning
    all 4 copies as one flat list (the full C4-symmetric kolam).

    C4 refers to the cyclic group of order 4: the four rotations
    {0, 90, 180, 270} form a group under composition (rotating twice by 90
    gets you to 180, four times brings you back to the start). A kolam has
    C4 symmetry if it looks identical after any of those four rotations --
    which is exactly what replicating one quadrant guarantees by
    construction.
    """
    center = _grid_center(grid)
    full_strokes: List[Stroke] = []
    for k in range(4):  # k = 0 (0 deg), 1 (90 deg), 2 (180 deg), 3 (270 deg)
        for s in strokes:
            rotated_points = [_rotate_point_90k(p, center, k) for p in s.points]
            full_strokes.append(Stroke(points=rotated_points, closed=s.closed))
    return full_strokes


# ---------------------------------------------------------------------------
# Symmetry detection
# ---------------------------------------------------------------------------

def _all_points(strokes: List[Stroke]) -> "set[Tuple[float, float]]":
    pts = set()
    for s in strokes:
        for p in s.points:
            pts.add(_round_pt(p))
    return pts


def detect_symmetry(strokes: List[Stroke], grid: DotGrid) -> Symmetry:
    """Inspect the (x, y) points used by `strokes` and infer the kolam's
    symmetry, by checking whether the point cloud maps onto itself under
    each candidate rotation/reflection.

    The test is purely set-based: transform every point and see if the
    resulting set of points is identical (order doesn't matter) to the
    original set. If it is, the shape is invariant under that
    transformation, i.e. it has that symmetry.
    """
    center = _grid_center(grid)
    pts = _all_points(strokes)
    if not pts:
        return Symmetry(rotational=1, mirror_axes=0)

    # --- rotational symmetry -------------------------------------------------
    # Check the highest order first: if a shape is invariant under a 90
    # degree rotation (order 4) it is automatically invariant under 180
    # degrees too, so we only need to report the largest order that holds.
    rotational = 1
    for order, k in ((4, 1), (2, 2)):
        rotated = {_round_pt(_rotate_point_90k(p, center, k)) for p in pts}
        if rotated == pts:
            rotational = order
            break

    # --- mirror symmetry -------------------------------------------------
    # Up to 4 candidate axes through the center: vertical, horizontal, and
    # the two diagonals. Each reflection formula below mirrors a point
    # across that axis; mirror_axes counts how many axes the shape survives
    # unchanged under.
    cx, cy = center
    mirror_axes = 0

    vertical = {_round_pt((2 * cx - x, y)) for x, y in pts}           # flip across x = cx
    if vertical == pts:
        mirror_axes += 1

    horizontal = {_round_pt((x, 2 * cy - y)) for x, y in pts}         # flip across y = cy
    if horizontal == pts:
        mirror_axes += 1

    diag_main = {_round_pt((cx + (y - cy), cy + (x - cx))) for x, y in pts}   # flip across y - cy = x - cx
    if diag_main == pts:
        mirror_axes += 1

    diag_anti = {_round_pt((cx - (y - cy), cy - (x - cx))) for x, y in pts}   # flip across y - cy = -(x - cx)
    if diag_anti == pts:
        mirror_axes += 1

    return Symmetry(rotational=rotational, mirror_axes=mirror_axes)


# ---------------------------------------------------------------------------
# Loop analysis
# ---------------------------------------------------------------------------

class _UnionFind:
    """Tiny disjoint-set structure used to group strokes into connected
    components based on shared endpoints."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _endpoints(stroke: Stroke) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if not stroke.points:
        return None, None
    return _round_pt(stroke.points[0]), _round_pt(stroke.points[-1])


def count_loops(strokes: List[Stroke]) -> Tuple[int, bool]:
    """Return (loop_count, single_loop) by looking at which strokes are
    closed and how open strokes chain together end-to-end.

    The core idea, and the reason single_loop matters for a kolam: a
    traditional sikku kolam is meant to be drawn *without lifting the pen*
    and without retracing any line -- one continuous thread that winds
    around every dot and closes back on itself. That is exactly the
    definition of a graph cycle where every node has degree 2 (each point
    the thread passes through is entered once and left once):

      1. Treat every stroke's two endpoints as nodes in a graph, and the
         stroke itself as an edge connecting them (a stroke that is already
         `closed`, or whose start == end, is its own self-contained loop --
         a single node visited twice).
      2. Union-Find strokes into connected components based on shared
         endpoints: strokes that touch nose-to-tail belong to the same
         piece of thread.
      3. A component is a closed loop if, once assembled, every node in it
         has degree exactly 2 AND the number of edges (strokes) equals the
         number of distinct nodes. That combination is only possible for a
         single simple cycle -- degree 2 everywhere means no dead ends and
         no branching, and edges == nodes rules out multiple separate
         sub-cycles glued together.
      4. loop_count = how many components qualify as closed loops.
      5. single_loop = True only when the *entire* set of strokes forms
         exactly one connected component, and that component is a closed
         loop -- i.e. the whole kolam is one unbroken line, the traditional
         "single stroke" ideal.
    """
    n = len(strokes)
    if n == 0:
        return 0, False

    dsu = _UnionFind(n)
    endpoints = [_endpoints(s) for s in strokes]

    # Union any two strokes that share an endpoint (nose-to-tail connection).
    endpoint_to_strokes: Dict[Tuple[float, float], List[int]] = {}
    for i, (a, b) in enumerate(endpoints):
        for key in (a, b):
            if key is None:
                continue
            endpoint_to_strokes.setdefault(key, []).append(i)
    for idxs in endpoint_to_strokes.values():
        for other in idxs[1:]:
            dsu.union(idxs[0], other)

    # Group stroke indices by their connected component.
    components: Dict[int, List[int]] = {}
    for i in range(n):
        components.setdefault(dsu.find(i), []).append(i)

    loop_count = 0
    for idxs in components.values():
        degree: Dict[Tuple[float, float], int] = {}
        for i in idxs:
            a, b = endpoints[i]
            if a is None:
                continue
            if strokes[i].closed or a == b:
                # Already a self-contained loop: one node, visited twice.
                degree[a] = degree.get(a, 0) + 2
            else:
                degree[a] = degree.get(a, 0) + 1
                degree[b] = degree.get(b, 0) + 1

        num_nodes = len(degree)
        num_edges = len(idxs)
        is_closed_loop = num_nodes > 0 and num_edges == num_nodes and all(d == 2 for d in degree.values())
        if is_closed_loop:
            loop_count += 1

    single_loop = len(components) == 1 and loop_count == 1
    return loop_count, single_loop


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid = generate_dot_grid(5, 5, spacing=10.0, layout="square")
    print(f"Generated {grid.rows}x{grid.cols} {grid.layout} grid, {len(grid.dots)} dots")
    print(f"Grid center: {_grid_center(grid)}")

    # Two strokes drawn in a single quadrant: a curve from the dot directly
    # "north" of center, through a midpoint, to the dot directly "east" of
    # center. This is the kind of quarter-kolam an artist actually draws by
    # hand before the rest is filled in by symmetry.
    quadrant_strokes = [
        Stroke(points=[(20.0, 30.0), (25.0, 25.0)], closed=False),
        Stroke(points=[(25.0, 25.0), (30.0, 20.0)], closed=False),
    ]

    full_strokes = apply_c4_symmetry(quadrant_strokes, grid)
    print(f"\nC4 symmetry: {len(quadrant_strokes)} quadrant strokes -> {len(full_strokes)} total strokes")
    for i, s in enumerate(full_strokes):
        pts = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in s.points)
        print(f"  stroke {i}: closed={s.closed} points=[{pts}]")

    symmetry = detect_symmetry(full_strokes, grid)
    print(f"\nDetected symmetry: rotational=C{symmetry.rotational}, mirror_axes={symmetry.mirror_axes}")

    loop_count, single_loop = count_loops(full_strokes)
    print(f"\nLoop analysis: loop_count={loop_count}, single_loop={single_loop}")
