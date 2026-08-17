"""End-to-end: raster image -> populated Kolam with extracted design principles."""
import sys
from pathlib import Path

from core.model import Kolam
from core.symmetry import detect_symmetry, count_loops
from analysis import detect


def analyze(path: str, debug: bool = False) -> Kolam:
    """Run the detect -> trace -> symmetry/loop pipeline and assemble a full Kolam."""
    if debug:
        print(f"[analyze] loading {path}")
    binary = detect.load_and_preprocess(path)

    if debug:
        print("[analyze] -> detect_dots")
    grid = detect.detect_dots(binary, debug=debug)

    if debug:
        print("[analyze] -> trace_strokes")
    strokes = detect.trace_strokes(binary, grid, debug=debug)

    symmetry = detect_symmetry(strokes, grid)
    loop_count, single_loop = count_loops(strokes)

    if debug:
        print(f"[analyze] detect_symmetry -> rotational=C{symmetry.rotational}, "
              f"mirror_axes={symmetry.mirror_axes}")
        print(f"[analyze] count_loops -> loop_count={loop_count}, single_loop={single_loop}")

    principles = {
        "grid_size": f"{grid.rows}x{grid.cols}",
        "layout": grid.layout,
        "spacing_px": round(grid.spacing, 1),
        "symmetry_group": f"C{symmetry.rotational}",
        "mirror_axes": symmetry.mirror_axes,
        "loop_count": loop_count,
        "single_loop": single_loop,
        "stroke_count": len(strokes),
    }

    # sikku kolams are traditionally drawn as one unbroken thread; that's
    # exactly what single_loop tests for, so it's a reasonable style guess.
    # This is the ONLY place style is ever set on the returned Kolam.
    style = "sikku" if single_loop else "pulli"
    if debug:
        print(f"[analyze] style: single_loop={single_loop} -> style='{style}' "
              f"(only set here, from single_loop, nowhere else)")

    return Kolam(
        grid=grid,
        strokes=strokes,
        symmetry=symmetry,
        loop_count=loop_count,
        single_loop=single_loop,
        style=style,
        principles=principles,
    )


def _print_principles(kolam: Kolam) -> None:
    p = kolam.principles
    print("Design principles:")
    print(f"  Grid:            {p['grid_size']} ({p['layout']}, spacing={p['spacing_px']}px)")
    print(f"  Symmetry group:  {p['symmetry_group']} (mirror axes: {p['mirror_axes']})")
    print(f"  Loop count:      {p['loop_count']}")
    print(f"  Single loop:     {'yes' if p['single_loop'] else 'no'}")
    print(f"  Strokes traced:  {p['stroke_count']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Reuse detect.py's own fixture: a dot grid, one closed loop, one open stroke.
        test_dir = Path("data")
        test_dir.mkdir(exist_ok=True)
        image_path = str(test_dir / "sample_kolam_strokes.png")
        detect._make_stroke_test_image(image_path)

    kolam = analyze(image_path)
    _print_principles(kolam)
