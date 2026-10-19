import re
from math import cos, pi, sin
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import ast_net.phasor as phr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgba
from tifffile import imread


def load_tif_dir(dir_path: str) -> Tuple[List[np.ndarray], List[str]]:
    """Read all .tif files from a given directory."""
    arrays: List[np.ndarray] = []
    filenames: List[str] = []
    base_dir = Path(dir_path)

    # Keep order deterministic so names[i] always maps to arrays[i].
    for file_path in sorted(base_dir.glob("*.tif")):
        arrays.append(imread(file_path))
        filenames.append(file_path.stem)

    return arrays, filenames


def add_point_label(ax: plt.Axes, label: str, x: float, y: float) -> None:
    ax.plot(x, y, "+", color="red", markersize=7)
    ax.text(x + 0.01, y + 0.01, label)


def get_universal_circle(points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    radius = 0.5
    center = (0.5, 0.0)
    x: List[float] = []
    y: List[float] = []
    for p in range(points + 1):
        theta = pi * p / points
        x.append(center[0] + radius * cos(theta))
        y.append(center[1] + radius * sin(theta))
    return np.asarray(x), np.asarray(y)


def _normalize_gs_points(arr: np.ndarray) -> np.ndarray:
    """Normalize any plausible ROI array shape into an (n, 2) G/S point cloud."""
    gs = np.asarray(arr)
    if gs.size == 0:
        return np.empty((0, 2), dtype=float)

    gs = np.atleast_2d(gs)
    if gs.shape[1] >= 2:
        gs = gs[:, :2]
    elif gs.shape[0] >= 2:
        gs = gs[:2, :].T
    else:
        return np.empty((0, 2), dtype=float)

    finite_mask = np.isfinite(gs).all(axis=1)
    return gs[finite_mask]


def _stack_result_points(roi_map: Mapping[int, np.ndarray]) -> np.ndarray:
    """Merge all ROI point clouds in one sample into a single (n, 2) array."""
    merged: List[np.ndarray] = []
    for _, arr in sorted(roi_map.items()):
        gs = _normalize_gs_points(arr)
        if gs.size > 0:
            merged.append(gs)

    if not merged:
        return np.empty((0, 2), dtype=float)
    return np.vstack(merged)


def _trailing_index(name: str) -> Optional[int]:
    """Return the integer at the end of `name` (e.g. 'foo_0012' -> 12), or None."""
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else None


def _draw_base(
    ax: plt.Axes,
    title: str,
    lifetime_points: Optional[Mapping[str, Tuple[float, float]]],
) -> None:
    """Draw the universal circle, reference points, and axis labels on `ax`."""
    ax.set_aspect("equal", adjustable="box")
    uc_x, uc_y = get_universal_circle()
    ax.plot(uc_x, uc_y, "k-", linewidth=1.2, label="Universal circle")

    if lifetime_points:
        for label, (x, y) in lifetime_points.items():
            add_point_label(ax, label, x, y)

    ax.set_xlabel("G")
    ax.set_ylabel("S")
    ax.set_title(title)


def plot_batch_gs(
    res: Sequence[Mapping[int, np.ndarray]],
    names: Sequence[str],
    lifetime_points: Optional[Mapping[str, Tuple[float, float]]] = None,
    cmap: Union[str, Colormap, None] = None,
    point_alpha: float = 0.15,
    point_size: float = 4.0,
    max_points_per_series: Optional[int] = 5000,
    label_centroids: bool = False,
    seed: int = 0,
) -> None:
    """Plot per-series G/S data twice, colored by each name's trailing index.

    Left: point clouds plus centroids. Right: centroids only.
    `cmap` may be a matplotlib colormap or its name. Defaults to red -> blue.

    To keep overlapping clouds readable, all points from all series are pooled
    and drawn in random order (so no series sits on top of the others), and each
    series is optionally subsampled to `max_points_per_series` points for
    display. Centroids are always computed from the full data.

    If `label_centroids` is True, each centroid in the right-hand (centroid-only)
    plot is annotated with its series name.
    """
    if len(res) != len(names):
        raise ValueError(
            f"Length mismatch: len(res)={len(res)} but len(names)={len(names)}."
        )
    fig, (ax_all, ax_cent) = plt.subplots(
        1, 2, figsize=(16, 6), constrained_layout=True
    )

    _draw_base(ax_all, "Batch ROI Phasor Plot", lifetime_points)
    _draw_base(ax_cent, "Batch ROI Centroids", lifetime_points)

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("red_blue", ["red", "blue"])
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    indices = [_trailing_index(n) for n in names]
    valid = [i for i in indices if i is not None]
    norm = Normalize(vmin=min(valid), vmax=max(valid)) if valid else Normalize(0, 1)

    any_points = False
    rng = np.random.default_rng(seed)
    cloud_xy: List[np.ndarray] = []
    cloud_rgba: List[np.ndarray] = []

    for idx, (series_name, roi_map) in enumerate(zip(names, res)):
        points = _stack_result_points(roi_map)
        if points.size == 0:
            continue

        any_points = True
        num = indices[idx]
        color = cmap(norm(num)) if num is not None else "gray"

        shown = points
        if max_points_per_series is not None and len(points) > max_points_per_series:
            keep = rng.choice(len(points), max_points_per_series, replace=False)
            shown = points[keep]
        cloud_xy.append(shown)
        cloud_rgba.append(np.tile(to_rgba(color), (len(shown), 1)))

        centroid = points.mean(axis=0)
        for ax in (ax_all, ax_cent):
            ax.scatter(
                centroid[0],
                centroid[1],
                s=80,
                marker="o",
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )

        # Series label next to the centroid, centroid pane only.
        if label_centroids:
            ax_cent.annotate(
                series_name,
                (centroid[0], centroid[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6,
                zorder=4,
            )

    if not any_points:
        raise ValueError("No plottable G/S points found in `res`.")

    # Draw all points in shuffled order so overlap isn't dominated by the last series.
    xy = np.vstack(cloud_xy)
    rgba = np.vstack(cloud_rgba)
    order = rng.permutation(len(xy))
    ax_all.scatter(
        xy[order, 0],
        xy[order, 1],
        s=point_size,
        c=rgba[order],
        alpha=point_alpha,
        linewidths=0,
        rasterized=True,
    )

    # Give the centroid plot the same view as the full plot for easy comparison.
    ax_cent.set_xlim(ax_all.get_xlim())
    ax_cent.set_ylim(ax_all.get_ylim())

    # The per-series legend is replaced by a colorbar; keep just the circle entry.
    ax_all.legend(loc="upper right")
    ax_cent.legend(loc="upper right")

    if valid:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_all, ax_cent], pad=0.02, fraction=0.03)
        cbar.set_label("Series index")

    plt.show()


if __name__ == "__main__":
    batch_data_path = "/home/edward/Documents/workspaces/research/projects/ast/data/20260507_live_elr_flim/tif/stbl2_b"
    cal_data_path = "/home/edward/Documents/workspaces/research/projects/ast/data/20260507_live_elr_flim/tif/chroma/chroma_740ex_457-50em_0000.tif"
    arrays, names = load_tif_dir(batch_data_path)
    res = phr.batch_segment_gs(arrays, imread(cal_data_path), 1.1, 12.5)
    plot_batch_gs(res, names, label_centroids=False)
