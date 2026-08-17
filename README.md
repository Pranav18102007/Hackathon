# Kolam Analysis & Generation

A pipeline for analyzing traditional South Indian Kolam (dot-grid) art and recreating/generating new designs from the extracted principles.

## Pipeline

1. **Detect** — take a photo/image of a kolam, recover its dot grid and stroke paths, and infer its symmetry and topological properties (`analysis/`).
2. **Represent** — store the result in a shared `Kolam` data model (`core/model.py`) that every other module reads and writes.
3. **Generate** — synthesize new kolams from the detected principles or from scratch (`core/`).
4. **Render** — draw kolams to SVG/raster output (`render/`).
5. **Serve** — expose the pipeline through a web UI and API (`web/`, `api/`).

## Project layout

| Folder      | Purpose                                              | Owner        |
|-------------|-------------------------------------------------------|--------------|
| `core/`     | Shared data model, generation logic                    | me           |
| `analysis/` | Computer vision: image -> Kolam extraction              | CV teammate  |
| `render/`   | Kolam -> SVG/image rendering                            | A            |
| `web/`      | Frontend                                               | A            |
| `api/`      | FastAPI backend endpoints                              | me           |
| `data/`     | Sample kolam images, fixtures                          | C            |
| `tests/`    | Test suite                                             | C            |
| `docs/`     | Documentation                                          | C            |

## Shared model

Every module reads/writes `Kolam` objects as defined in [`core/model.py`](core/model.py) — this is the contract between CV, generation, and rendering. Field names are frozen; do not rename without syncing across teammates.

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```
