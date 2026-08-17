"""Kolam dot-grid detection from a clean, high-contrast raster image."""
import sys
from pathlib import Path

import cv2
import numpy as np

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
