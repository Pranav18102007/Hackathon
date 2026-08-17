"""
Sikku kolam generation, built on the frozen shared model (core/model.py) and
the grid/symmetry/loop utilities in core/symmetry.py.
"""

from typing import List, Tuple

from core.model import DotGrid, Kolam, Stroke, Symmetry
from core.symmetry import apply_c4_symmetry, count_loops, detect_symmetry, generate_dot_grid

# Diamond "petal" size, as a fraction of dot spacing. Kept well under 0.5 so
# a dot's loop never reaches as far as its neighbour's dot -- that's what
# keeps the loops visually distinct and, more importantly, keeps their
# endpoints from ever accidentally coinciding with another loop's points.
_DIAMOND_RADIUS_FRACTION = 0.3


def _diamond_radius(spacing: float) -> float:
    return spacing * _DIAMOND_RADIUS_FRACTION


def _center(grid: DotGrid) -> Tuple[float, float]:
    xs = [p[0] for p in grid.dots]
    ys = [p[1] for p in grid.dots]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)


def _diamond_around_dot(dot: Tuple[float, float], radius: float) -> Stroke:
    """One weave primitive: a small diamond that loops AROUND a dot -- it
    never touches the dot itself, and never touches another dot -- tracing
    east, north, west, south and back to east.

    Loop closure here is a *construction guarantee*, not something we check
    after the fact: the stroke's point list explicitly repeats its first
    point as its last point, and `closed=True` is set directly. There is no
    code path that produces one of these diamonds without both of those
    being true, so every diamond this function returns is, by definition, a
    closed loop -- there's nothing to "get wrong" downstream.
    """
    x, y = dot
    east = (x + radius, y)
    north = (x, y + radius)
    west = (x - radius, y)
    south = (x, y - radius)
    return Stroke(points=[east, north, west, south, east], closed=True)


def _round_pt(p: Tuple[float, float], decimals: int = 4) -> Tuple[float, float]:
    return (round(p[0], decimals), round(p[1], decimals))


def _dedupe_strokes(strokes: List[Stroke]) -> List[Stroke]:
    """Drop strokes that trace the exact same set of points as one already
    kept.

    This matters because our "one quadrant" for C4 replication is chosen
    with inclusive bounds (x <= center_x and y <= center_y), which is the
    simplest way to describe "one quadrant" but double-counts dots that sit
    exactly on the center's axes (and, for odd-sized grids, the single dot
    exactly at the center). Those get rotated onto themselves or onto each
    other's positions, producing two identical diamonds stacked on top of
    one another. We compare by the *set* of rounded points rather than the
    ordered list, because a diamond's own 4-fold symmetry means the same
    physical diamond can come out of rotation with its points cyclically
    shifted (e.g. starting at "north" instead of "east") while still being
    the same loop.
    """
    seen = set()
    unique: List[Stroke] = []
    for s in strokes:
        key = frozenset(_round_pt(p) for p in s.points)
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


def _quadrant_dots(grid: DotGrid) -> List[Tuple[float, float]]:
    cx, cy = _center(grid)
    return [d for d in grid.dots if d[0] <= cx and d[1] <= cy]


def _build_diamond_kolam(grid: DotGrid, use_c4_shortcut: bool) -> List[Stroke]:
    radius = _diamond_radius(grid.spacing)
    if use_c4_shortcut:
        # Draw the primitive once per dot in a single quadrant, then let
        # apply_c4_symmetry() do the rest -- every diamond is individually
        # closed already, and rotating a closed stroke's points as a rigid
        # body can never separate its start point from its end point (a
        # geometric transform is applied point-by-point, so if start == end
        # going in, start == end coming out). Closure survives replication
        # for free.
        quadrant_strokes = [_diamond_around_dot(d, radius) for d in _quadrant_dots(grid)]
        return _dedupe_strokes(apply_c4_symmetry(quadrant_strokes, grid))
    # No quadrant shortcut available for this symmetry order -- build the
    # (still individually closed) loop directly for every dot.
    return [_diamond_around_dot(d, radius) for d in grid.dots]


def generate_sikku(rows: int, cols: int, spacing: float) -> Kolam:
    """Generate a sikku kolam: a square dot grid, with a closed loop woven
    around every dot, built from one quadrant and completed with C4
    rotational symmetry.
    """
    grid = generate_dot_grid(rows, cols, spacing, layout="square")
    strokes = _build_diamond_kolam(grid, use_c4_shortcut=True)

    symmetry = detect_symmetry(strokes, grid)
    loop_count, single_loop = count_loops(strokes)
    principles = {
        "grid": f"{rows}x{cols}",
        "symmetry": f"C{symmetry.rotational}",
        "loops": loop_count,
    }

    return Kolam(
        grid=grid,
        strokes=strokes,
        symmetry=symmetry,
        loop_count=loop_count,
        single_loop=single_loop,
        style="sikku",
        principles=principles,
    )


def regenerate_from_principles(kolam_partial: Kolam) -> Kolam:
    """The "recreation" step: takes a Kolam whose grid/symmetry/loop fields
    were filled in from analysis (e.g. the CV side detected a 7x7 grid with
    C4 symmetry from a photo) but has no strokes yet, and produces a full,
    clean Kolam with real stroke geometry.

    Any strokes already on `kolam_partial` are ignored on purpose -- the
    point of this function is to go from *principles* to strokes, not to
    edit strokes that were already there.
    """
    src_grid = kolam_partial.grid
    grid = src_grid if src_grid.dots else generate_dot_grid(
        src_grid.rows, src_grid.cols, src_grid.spacing, src_grid.layout
    )

    # We only have a dedicated quadrant-replication shortcut for C4; other
    # detected symmetry orders fall back to building the (still
    # closed-by-construction) loop primitive directly for every dot.
    use_c4_shortcut = kolam_partial.symmetry.rotational == 4
    strokes = _build_diamond_kolam(grid, use_c4_shortcut)

    symmetry = detect_symmetry(strokes, grid)
    loop_count, single_loop = count_loops(strokes)

    principles = dict(kolam_partial.principles)
    principles.update({
        "grid": f"{grid.rows}x{grid.cols}",
        "symmetry": f"C{symmetry.rotational}",
        "loops": loop_count,
    })

    return Kolam(
        grid=grid,
        strokes=strokes,
        symmetry=symmetry,
        loop_count=loop_count,
        single_loop=single_loop,
        style=kolam_partial.style or "sikku",
        principles=principles,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    kolam = generate_sikku(rows=7, cols=7, spacing=10.0)
    print(f"generate_sikku: {kolam.style} kolam on a {kolam.grid.rows}x{kolam.grid.cols} grid")
    print(f"  strokes: {len(kolam.strokes)} (all closed: {all(s.closed for s in kolam.strokes)})")
    print(f"  symmetry: rotational=C{kolam.symmetry.rotational} mirror_axes={kolam.symmetry.mirror_axes}")
    print(f"  loop_count={kolam.loop_count} single_loop={kolam.single_loop}")
    print(f"  principles: {kolam.principles}")

    print("\nregenerate_from_principles: recreating from CV-style detected principles only")
    detected = Kolam(
        grid=DotGrid(rows=7, cols=7, spacing=10.0),  # dims only -- as analysis would report, no dots yet
        strokes=[],                                   # CV hasn't produced clean strokes, only principles
        symmetry=Symmetry(rotational=4, mirror_axes=0),
        loop_count=0,
        single_loop=False,
        style="sikku",
        principles={"source": "cv-detection"},
    )
    recreated = regenerate_from_principles(detected)
    print(f"  strokes: {len(recreated.strokes)}")
    print(f"  principles: {recreated.principles}")
    print(f"  matches generate_sikku output: {len(recreated.strokes) == len(kolam.strokes)}")
