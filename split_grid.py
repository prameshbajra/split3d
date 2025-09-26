#!/usr/bin/env python3
"""Split a 3D mesh into an axis-aligned MxNxO grid of watertight pieces."""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import trimesh

_LOGGER = logging.getLogger("split3d")


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a 3MF or STL mesh into MxNxO pieces along the X, Y, and Z axes.",
    )
    parser.add_argument("input", type=Path, help="Path to the input .3mf or .stl file")
    parser.add_argument(
        "--grid",
        "-g",
        type=_parse_grid,
        required=True,
        metavar="MxNxO",
        help="Grid definition, e.g. 2x3x1",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("splits"),
        help="Directory to write the split meshes",
    )
    parser.add_argument(
        "--output-format",
        "-f",
        choices=("stl", "3mf", "obj"),
        default="stl",
        help="Export format for the pieces",
    )
    parser.add_argument(
        "--engine",
        choices=("igl", "cork", "scad", "blender", "pyvista", "autodetect"),
        default="autodetect",
        help="Boolean engine to use; autodetect tries igl->cork->scad->blender->pyvista.",
    )
    parser.add_argument(
        "--force-watertight",
        action="store_true",
        help="Attempt to fill small holes on the resulting pieces.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def _parse_grid(value: str) -> Tuple[int, int, int]:
    separators = ("x", "X")
    for sep in separators:
        if sep in value:
            parts = value.split(sep)
            break
    else:
        raise argparse.ArgumentTypeError("Grid must be in the form MxNxO (e.g. 2x2x1).")

    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Grid must contain exactly three factors.")

    try:
        grid = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Grid values must be integers.") from exc

    if any(v < 1 for v in grid):
        raise argparse.ArgumentTypeError("Grid values must be >= 1.")

    return grid  # type: ignore[return-value]


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", skip_materials=True, process=True)

    if isinstance(mesh, trimesh.Scene):
        # Merge all geometries in the scene into a single mesh.
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh type returned from loader: {type(mesh)!r}")

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    if not mesh.is_watertight:
        _LOGGER.warning("Input mesh is not watertight; resulting pieces may have open surfaces.")

    return mesh


def _boolean_intersection(
    mesh: trimesh.Trimesh,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    engines: Iterable[str],
) -> trimesh.Trimesh | None:
    extents = max_corner - min_corner
    center = (max_corner + min_corner) / 2.0
    transform = np.eye(4)
    transform[:3, 3] = center

    box = trimesh.creation.box(extents=extents, transform=transform)

    for engine in engines:
        if engine == "pyvista":
            piece = _pyvista_clip(mesh, min_corner=min_corner, max_corner=max_corner)
            if piece is not None:
                return piece
            continue
        try:
            result = trimesh.boolean.intersection([mesh, box], engine=None if engine == "autodetect" else engine)
        except BaseException as exc:  # noqa: BLE001
            _LOGGER.debug("Boolean intersection failed for engine %s: %s", engine, exc)
            continue

        if isinstance(result, trimesh.Scene):
            result = trimesh.util.concatenate(tuple(result.geometry.values()))

        if isinstance(result, trimesh.Trimesh) and result.faces.size:
            result.remove_unreferenced_vertices()
            return result

    return None


def _pyvista_clip(
    mesh: trimesh.Trimesh,
    *,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
) -> trimesh.Trimesh | None:
    try:
        import pyvista as pv
    except ImportError:  # pragma: no cover - dependency optional at runtime
        _LOGGER.debug("PyVista not installed; skipping pyvista engine.")
        return None

    def _apply_filter(dataset: pv.DataSet, method: str, /, *args, **kwargs) -> pv.DataSet:
        filter_fn = getattr(dataset, method)
        try:
            result = filter_fn(*args, **kwargs)
        except TypeError:
            if "inplace" in kwargs:
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != "inplace"}
                result = filter_fn(*args, **filtered_kwargs)
            else:
                raise
        return dataset if result is None else result

    if mesh.faces.size == 0:
        return None

    faces = np.hstack(
        (
            np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64),
        )
    ).ravel()

    poly = pv.PolyData(mesh.vertices.copy(), faces)
    bounds = (min_corner[0], max_corner[0], min_corner[1], max_corner[1], min_corner[2], max_corner[2])

    clipped = poly.clip_box(bounds=bounds, invert=False, merge_points=True)
    if clipped is None or clipped.n_cells == 0:
        return None

    clipped = _apply_filter(clipped, "triangulate")
    if clipped is None or clipped.n_cells == 0:
        return None

    try:
        capped = _apply_filter(clipped, "cap", inplace=False)
        if capped is not None and capped.n_cells:
            clipped = _apply_filter(capped, "triangulate")
    except AttributeError:
        _LOGGER.debug("PyVista version does not expose cap(); continuing without capping.")

    clipped = _apply_filter(clipped, "clean", inplace=False)
    if clipped is None or clipped.n_cells == 0:
        return None

    if not isinstance(clipped, pv.PolyData):
        clipped = _apply_filter(clipped, "extract_surface")
        if clipped is None or clipped.n_cells == 0:
            return None
        clipped = _apply_filter(clipped, "triangulate")
        if clipped is None or clipped.n_cells == 0:
            return None

    if clipped.faces is None or clipped.faces.size == 0:
        return None

    pv_faces = clipped.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    piece = trimesh.Trimesh(vertices=clipped.points.copy(), faces=pv_faces, process=True)
    piece.remove_unreferenced_vertices()

    if not piece.faces.size:
        return None

    return piece


def split_mesh(
    mesh: trimesh.Trimesh,
    grid: Tuple[int, int, int],
    engine: str,
    force_watertight: bool,
) -> Dict[Tuple[int, int, int], trimesh.Trimesh]:
    min_corner, max_corner = mesh.bounds
    xs = np.linspace(min_corner[0], max_corner[0], grid[0] + 1)
    ys = np.linspace(min_corner[1], max_corner[1], grid[1] + 1)
    zs = np.linspace(min_corner[2], max_corner[2], grid[2] + 1)

    engine_priority = {
        "autodetect": ("igl", "cork", "scad", "blender", "pyvista", "autodetect"),
        "igl": ("igl",),
        "cork": ("cork",),
        "scad": ("scad",),
        "blender": ("blender",),
        "pyvista": ("pyvista",),
    }[engine]

    pieces: Dict[Tuple[int, int, int], trimesh.Trimesh] = {}

    for ix, iy, iz in itertools.product(range(grid[0]), range(grid[1]), range(grid[2])):
        bounds_min = np.array([xs[ix], ys[iy], zs[iz]])
        bounds_max = np.array([xs[ix + 1], ys[iy + 1], zs[iz + 1]])

        piece = _boolean_intersection(mesh, bounds_min, bounds_max, engine_priority)
        if piece is None:
            _LOGGER.info("Skipping empty piece at (%d, %d, %d)", ix, iy, iz)
            continue

        if force_watertight:
            piece = piece.copy()
            piece.fill_holes()
            piece.remove_degenerate_faces()

        pieces[(ix, iy, iz)] = piece

    return pieces


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level))

    mesh = _load_mesh(args.input)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pieces = split_mesh(mesh, args.grid, engine=args.engine, force_watertight=args.force_watertight)

    if not pieces:
        _LOGGER.warning("No pieces were generated.")
        return 1

    base_name = args.input.stem

    for (ix, iy, iz), piece in pieces.items():
        target = output_dir / f"{base_name}_x{ix}_y{iy}_z{iz}.{args.output_format}" 
        piece.export(target)
        _LOGGER.info("Wrote %s", target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
