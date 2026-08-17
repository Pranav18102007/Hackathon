"""Kolam dot-grid detection from a clean, high-contrast raster image."""
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from core.model import DotGrid, Stroke, Symmetry, Kolam


def load_and_preprocess(path: str) -> np.ndarray:
    """Load an image and return a clean binary image (ink=255, background=0)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    denoised = cv2.medianBlur(binary, 3)
    kernel = np.ones((3, 3), np.uint8)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

    return denoised


def _cluster_1d(values: list) -> list:
    """Group nearby 1D coordinates into clusters, returning cluster centers.

    Splits at the largest jump in sorted gap sizes, which separates
    within-row/col jitter from the real row/col spacing on a clean grid.
    """
    values = sorted(values)
    if len(values) <= 1:
        return values

    diffs = np.diff(values)
    sorted_diffs = np.sort(diffs)
    gap_jumps = np.diff(sorted_diffs)

    if len(gap_jumps) == 0 or gap_jumps.max() == 0:
        threshold = sorted_diffs[0] / 2 if sorted_diffs[0] > 0 else 1.0
    else:
        split_idx = int(np.argmax(gap_jumps))
        threshold = (sorted_diffs[split_idx] + sorted_diffs[split_idx + 1]) / 2

    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def _estimate_spacing(centroids: list) -> float:
    """Median nearest-neighbor distance between dots."""
    if len(centroids) < 2:
        return 0.0
    pts = np.array(centroids)
    dists = []
    for i, p in enumerate(pts):
        d = np.linalg.norm(pts - p, axis=1)
        d[i] = np.inf
        dists.append(d.min())
    return float(np.median(dists))


def _infer_layout(row_centers: list, centroids: list, spacing: float) -> str:
    """Classify grid layout by checking for alternating row offsets."""
    if spacing == 0 or len(row_centers) < 2:
        return "square"

    rows = sorted(row_centers)
    offsets = []
    for r_prev, r_next in zip(rows, rows[1:]):
        prev_xs = sorted(x for x, y in centroids if abs(y - r_prev) < spacing / 2)
        next_xs = sorted(x for x, y in centroids if abs(y - r_next) < spacing / 2)
        if not prev_xs or not next_xs:
            continue
        offsets.append(abs(next_xs[0] - prev_xs[0]) % spacing)

    if not offsets:
        return "square"

    mean_offset = float(np.mean(offsets))
    if mean_offset < spacing * 0.2:
        return "square"
    if abs(mean_offset - spacing / 2) < spacing * 0.2:
        return "triangular"
    return "rhombus"


def detect_dots(binary_img: np.ndarray) -> DotGrid:
    """Find the pulli dot grid via contour/blob detection and infer its geometry."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.7:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        centroids.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))

    if not centroids:
        return DotGrid(rows=0, cols=0, spacing=0.0, layout="square", dots=[])

    row_centers = _cluster_1d([y for _, y in centroids])
    col_centers = _cluster_1d([x for x, _ in centroids])
    spacing = _estimate_spacing(centroids)
    layout = _infer_layout(row_centers, centroids, spacing)

    return DotGrid(
        rows=len(row_centers),
        cols=len(col_centers),
        spacing=spacing,
        layout=layout,
        dots=sorted(centroids, key=lambda p: (p[1], p[0])),
    )


# --- Stroke tracing --------------------------------------------------------
#
# trace_strokes() is PRIMARY: it skeletonizes the stroke-only mask and walks
# the resulting pixel graph into ordered per-stroke paths, closing strokes
# that form loops. It correctly separates strokes that don't touch each
# other, but a real crossing (two strokes overlapping at a pixel, 3+
# skeleton neighbors) is not disambiguated -- the walk just stops there,
# which fragments a stroke into pieces at that point. Curve bends can also
# spuriously look like 3-way junctions under 8-connectivity; the graph
# builder below prunes the redundant diagonal edge that usually causes that,
# but it is not a full fix for real intersections.
#
# trace_strokes_simple() is the FALLBACK named in the task: if trace_strokes()
# fragments too aggressively on a given image, this skips all endpoint/
# junction reasoning and just returns each connected skeleton component as
# one Stroke (points ordered left-to-right, not walked). It won't separate
# strokes that touch, but it always produces *something* usable for
# downstream symmetry/loop-count inference.

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _erase_dots(binary_img: np.ndarray, dot_grid: DotGrid) -> np.ndarray:
    """Blank out the dot blobs so skeletonization only sees stroke lines."""
    lines_only = binary_img.copy()
    radius = max(5, int(dot_grid.spacing * 0.18)) if dot_grid.spacing else 8
    for x, y in dot_grid.dots:
        cv2.circle(lines_only, (int(round(x)), int(round(y))), radius, 0, -1)
    return lines_only


def _skeleton_graph(skeleton: np.ndarray) -> dict:
    """8-connected adjacency dict over skeleton pixels, {(row, col): [(row, col), ...]}.

    Drops a diagonal edge when both of the orthogonal neighbors that would
    make it redundant are also present, so a smooth curved line doesn't
    register a spurious degree-3 junction at every bend.
    """
    points = set(map(tuple, np.argwhere(skeleton)))
    graph = {}
    for r, c in points:
        neighbors = []
        for dr, dc in _NEIGHBOR_OFFSETS:
            if dr != 0 and dc != 0 and (r + dr, c) in points and (r, c + dc) in points:
                continue
            if (r + dr, c + dc) in points:
                neighbors.append((r + dr, c + dc))
        graph[(r, c)] = neighbors
    return graph


def _walk_path(graph: dict, degree: dict, start: tuple, visited: set) -> list:
    """Walk an open chain from an endpoint until another endpoint or a junction."""
    path = [start]
    visited.add(start)
    current = start
    while True:
        neighbors = [n for n in graph[current] if n not in visited]
        if not neighbors:
            break
        nxt = neighbors[0]
        path.append(nxt)
        if degree[nxt] >= 3:
            break  # stop at a junction; leave it unvisited so other strokes can still reach it
        visited.add(nxt)
        current = nxt
    return path


def _walk_loop(graph: dict, start: tuple, visited: set) -> list:
    """Walk a closed ring starting and ending at `start`."""
    path = [start]
    visited.add(start)
    prev, current = None, start
    while True:
        neighbors = [n for n in graph[current] if n != prev]
        nxt = start if start in neighbors else next((n for n in neighbors if n not in visited), None)
        if nxt is None or nxt == start:
            break
        path.append(nxt)
        visited.add(nxt)
        prev, current = current, nxt
    return path


def trace_strokes(binary_img: np.ndarray, dot_grid: DotGrid) -> list:
    """PRIMARY: trace ordered Stroke paths from the skeletonized stroke lines."""
    lines_only = _erase_dots(binary_img, dot_grid)
    skeleton = skeletonize(lines_only > 0)

    graph = _skeleton_graph(skeleton)
    degree = {p: len(n) for p, n in graph.items()}
    visited = set()
    strokes = []

    endpoints = [p for p, d in degree.items() if d == 1]
    for start in endpoints:
        if start in visited:
            continue
        path = _walk_path(graph, degree, start, visited)
        if len(path) >= 3:
            strokes.append(Stroke(points=[(float(c), float(r)) for r, c in path], closed=False))

    for start in graph:
        if start in visited:
            continue
        path = _walk_loop(graph, start, visited)
        if len(path) >= 3:
            strokes.append(Stroke(points=[(float(c), float(r)) for r, c in path], closed=True))
        else:
            visited.add(start)

    return strokes


def trace_strokes_simple(binary_img: np.ndarray, dot_grid: DotGrid) -> list:
    """FALLBACK: one Stroke per connected skeleton component, no path walking."""
    lines_only = _erase_dots(binary_img, dot_grid)
    skeleton = skeletonize(lines_only > 0)

    num_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)

    strokes = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if len(xs) < 3:
            continue
        points = sorted(zip(xs.tolist(), ys.tolist()))
        strokes.append(Stroke(points=[(float(x), float(y)) for x, y in points], closed=False))

    return strokes


def _make_synthetic_kolam(path: str, rows: int = 5, cols: int = 5, spacing: int = 60, margin: int = 60) -> None:
    """Generate a clean black-on-white test kolam: a dot grid plus decorative strokes."""
    h = margin * 2 + spacing * (rows - 1)
    w = margin * 2 + spacing * (cols - 1)
    img = np.full((h, w), 255, dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            cv2.circle(img, (margin + c * spacing, margin + r * spacing), 6, 0, -1)

    # decorative bands weaving between dot rows, kept clear of the dots themselves
    for r in range(rows - 1):
        y = margin + r * spacing + spacing // 2
        pts = [(margin + c * spacing, y) for c in range(cols)]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], 0, 3)

    cv2.imwrite(path, img)


def _visualize(binary_img: np.ndarray, grid: DotGrid, out_path: str) -> None:
    overlay = cv2.cvtColor(255 - binary_img, cv2.COLOR_GRAY2BGR)
    for x, y in grid.dots:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 10, (0, 0, 255), 2)
    cv2.imwrite(out_path, overlay)


def _make_stroke_test_image(path: str, rows: int = 5, cols: int = 5, spacing: int = 60, canvas: int = 500) -> None:
    """Clean test kolam with one closed wavy loop and one open wavy stroke,
    on the same dot grid style as _make_synthetic_kolam, so both closed=True
    and closed=False get exercised.
    """
    img = np.full((canvas, canvas), 255, dtype=np.uint8)
    center = canvas // 2
    margin = center - (cols - 1) * spacing // 2

    dots = [(margin + c * spacing, margin + r * spacing) for r in range(rows) for c in range(cols)]
    for x, y in dots:
        cv2.circle(img, (x, y), 8, 0, -1)

    # closed loop: wavy ring fully outside the dot grid's convex hull
    loop_pts = []
    for deg in range(361):
        theta = np.radians(deg)
        r = 210 + 20 * np.sin(6 * theta)
        loop_pts.append((int(round(center + r * np.cos(theta))), int(round(center + r * np.sin(theta)))))
    cv2.polylines(img, [np.array(loop_pts, dtype=np.int32)], isClosed=True, color=0, thickness=3)

    # open stroke: wavy line through the middle dot row, dipping between dots without touching them
    mid_y = margin + (rows // 2) * spacing
    x0, x1 = margin, margin + (cols - 1) * spacing
    open_pts = [
        (x, int(round(mid_y + 25 * np.cos((x - margin) / spacing * np.pi))))
        for x in range(x0, x1 + 1)
    ]
    cv2.polylines(img, [np.array(open_pts, dtype=np.int32)], isClosed=False, color=0, thickness=3)

    cv2.imwrite(path, img)


_STROKE_PALETTE = [(0, 0, 255), (255, 0, 0), (0, 160, 0), (255, 140, 0), (200, 0, 200), (0, 180, 180)]


def _visualize_strokes(binary_img: np.ndarray, dot_grid: DotGrid, strokes: list, out_path: str) -> None:
    overlay = cv2.cvtColor(255 - binary_img, cv2.COLOR_GRAY2BGR)
    for x, y in dot_grid.dots:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 4, (0, 0, 0), -1)
    for i, stroke in enumerate(strokes):
        color = _STROKE_PALETTE[i % len(_STROKE_PALETTE)]
        pts = np.array([(int(round(x)), int(round(y))) for x, y in stroke.points], dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=stroke.closed, color=color, thickness=2)
        cv2.circle(overlay, tuple(pts[0]), 5, color, -1)
    cv2.imwrite(out_path, overlay)


if __name__ == "__main__":
    test_dir = Path("data")
    test_dir.mkdir(exist_ok=True)
    test_path = test_dir / "sample_kolam.png"
    out_path = test_dir / "sample_kolam_detected.png"

    if len(sys.argv) > 1:
        test_path = Path(sys.argv[1])
    else:
        _make_synthetic_kolam(str(test_path))

    binary = load_and_preprocess(str(test_path))
    grid = detect_dots(binary)

    print(f"Detected grid: {grid.rows} rows x {grid.cols} cols, spacing={grid.spacing:.1f}px, layout={grid.layout}")
    print(f"Total dots found: {len(grid.dots)}")

    _visualize(binary, grid, str(out_path))
    print(f"Overlay saved to {out_path}")

    # --- stroke tracing (needs an image with an actual loop, unlike the straight
    # decorative bands above, so it gets its own dedicated fixture) ---
    stroke_test_path = test_dir / "sample_kolam_strokes.png"
    stroke_out_path = test_dir / "sample_kolam_strokes_detected.png"
    _make_stroke_test_image(str(stroke_test_path))

    stroke_binary = load_and_preprocess(str(stroke_test_path))
    stroke_grid = detect_dots(stroke_binary)

    strokes = trace_strokes(stroke_binary, stroke_grid)
    closed_n = sum(s.closed for s in strokes)
    print(f"\n[trace_strokes] {len(strokes)} strokes ({closed_n} closed, {len(strokes) - closed_n} open)")
    for i, s in enumerate(strokes):
        print(f"  stroke {i}: {len(s.points)} pts, closed={s.closed}")
    _visualize_strokes(stroke_binary, stroke_grid, strokes, str(stroke_out_path))
    print(f"Stroke overlay saved to {stroke_out_path}")

    simple_strokes = trace_strokes_simple(stroke_binary, stroke_grid)
    print(f"[trace_strokes_simple fallback] {len(simple_strokes)} skeleton components")
