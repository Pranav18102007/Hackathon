from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class DotGrid:
    """The pulli (dot) grid a kolam is built on."""
    rows: int
    cols: int
    spacing: float
    layout: str = "square"          # "square" | "rhombus" | "triangular"
    dots: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class Stroke:
    """One continuous line/curve that weaves around dots."""
    points: List[Tuple[float, float]]
    closed: bool = False

@dataclass
class Symmetry:
    """Detected or enforced symmetry of the kolam."""
    rotational: int = 1             # 1,2,4 -> C1/C2/C4
    mirror_axes: int = 0

@dataclass
class Kolam:
    """Shared representation. CV writes it; generator writes it; renderer reads it."""
    grid: DotGrid
    strokes: List[Stroke]
    symmetry: Symmetry
    loop_count: int = 0
    single_loop: bool = False
    style: str = "sikku"            # "sikku" | "pulli"
    principles: dict = field(default_factory=dict)
