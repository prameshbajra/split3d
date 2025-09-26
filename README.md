# split3d

## Overview
`split3d` provides a Python CLI for slicing watertight 3D meshes (3MF, STL, OBJ) into an axis-aligned grid of parts. The main entry point is `split_grid.py`, which orchestrates mesh loading, boolean slicing across multiple backends, and optional hole filling for downstream fabrication or analysis.

## Quick Start
1. Install dependencies with `uv sync`.
2. Explore CLI options via `uv run split_grid.py --help`.
3. Run a sample split against the included mesh:
   ```bash
   uv run split_grid.py nameofthefile.3mf --grid 3x2x1 --output-dir out --output-format 3mf --force-watertight
   ```
   Generated pieces land in `out/`, named by their grid coordinates (`*_x0_y1_z0.stl`).

## Development Notes
- Mesh operations rely on `trimesh`, with optional acceleration from `pyvista`, `cork`, or `igl`. Ensure any new dependencies are added to `pyproject.toml` and `uv.lock`.
- Temporary artefacts (e.g., `out/`) should be excluded from commits unless explicitly required.

## Sample Assets
`KathmanduSkyline.3mf` serves as a compact fixture for local validation. You can swap in custom meshes by pointing the CLI at your own file path.
