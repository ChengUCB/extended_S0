#!/usr/bin/env python3
"""Random-label baselines and cation chemical order in Li/Mg chloride mixtures.

For each saved frame, the random reference keeps the instantaneous atomic
positions, box, total cation--Cl neighbor graph, and exact cation counts.  The
central-cation labels are then randomized analytically.  Conditional on a Cl
having ``m`` cation neighbors, its number of Li/Mg neighbors follows the exact
hypergeometric distribution.  This removes the trivial changes caused by
composition, density, and the total local coordination environment.

The shell-resolved Warren--Cowley functions are evaluated with an exact
finite-size correction, P_random(Na|X) = N_Na / (N_cation - 1), so the random
label reference is identically zero.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.stats import hypergeom

from analyze_mgcl2_nacl_rdfs import (
    configure_matplotlib,
    ensure_orthorhombic,
    frame_indices,
)


CATEGORIES = ("0", "1", "2", "3+")


CATEGORY_LABELS = ("0", "1", "2", r"$\geq 3$")


COLUMN_WIDTH_IN = 105.0 / 25.4
LABEL_SIZE = 10
LETTER_SIZE = 11


CONNECTIVITY_COLORS = ("#1A4F9C", "#3FBFC0", "#F0913D", "#B2182B")

SERIES_COLORS = ("#0072B2", "#D55E00")


VIK_ANCHORS = (
    "#001261", "#021F69", "#022B71", "#023779", "#034481",
    "#055189", "#0C5E92", "#1C6D9C", "#307DA6", "#488DB2",
    "#619EBD", "#7AAEC8", "#94BED2", "#ADCDDD", "#C6DBE6",
    "#DEE6E9", "#ECE5E0", "#EEDBD0", "#E9CCBA", "#E3BCA5",
    "#DCAC90", "#D69D7C", "#CF8E68", "#C98056", "#C37243",
    "#BD6431", "#B3531F", "#A5400F", "#942F06", "#832106",
    "#741506", "#670A07", "#590008",
)
VIK_CMAP = mpl.colors.LinearSegmentedColormap.from_list("vik", VIK_ANCHORS, N=256)
ALPHA_CMAP = mpl.colormaps["RdBu_r"]


def panel_letter(axis, letter: str, dx: float = -26.0) -> None:
    """Bold panel letter outside the axes, above and to the left of its corner.

    ``dx`` is in points; the right-hand column needs a smaller offset so the
    letter stays inside the gap between the two columns.
    """
    axis.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(dx, 5.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=LETTER_SIZE,
        fontweight="bold",
        annotation_clip=False,
    )


def panel_label(
    axis,
    label: str,
    *,
    halo: bool = False,
    x: float = 0.5,
    y: float = 0.97,
    ha: str = "center",
    fontsize: float = LABEL_SIZE,
) -> None:
    """Short in-panel label used in place of an axes title."""
    text = axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha=ha,
        va="top",
        fontsize=fontsize,
        zorder=6,
    )
    if halo:
        text.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--li-input-dir",
        type=Path,
        default=Path("MD_inputs/NaCl-LiCl_MD"),
    )
    parser.add_argument(
        "--mg-input-dir",
        type=Path,
        default=Path("MD_inputs/SelectedMgCl2-NaCl_MD"),
    )
    parser.add_argument(
        "--li-cutoff-csv",
        type=Path,
        default=Path(
            "outputs/LiCl-NaCl_vs_MgCl2-NaCl_Cl_coordination/"
            "LiCl_NaCl_LiCl_first_shell_metrics.csv"
        ),
    )
    parser.add_argument(
        "--mg-cutoff-csv",
        type=Path,
        default=Path("outputs/SelectedMgCl2-NaCl_MD_RDF/mgcl_first_shell_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/LiCl-NaCl_vs_MgCl2-NaCl_random_SRO"),
    )
    parser.add_argument(
        "--replot-from-csv",
        action="store_true",
        help=(
            "Skip the trajectory analysis and rebuild the figures from the CSV "
            "files already written in --output-dir."
        ),
    )
    parser.add_argument(
        "--connectivity-csv",
        type=Path,
        default=None,
        help=(
            "Connectivity table for panels a/b when replotting "
            "(default: random_connectivity_baseline.csv in --output-dir). "
            "Use the criterion-control table to plot an alternative first-shell "
            "criterion."
        ),
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default=None,
        help="Row filter for a connectivity table that carries a 'criterion' column.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Where the replotted figures go (default: --output-dir).",
    )
    parser.add_argument("--discard-fraction", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument(
        "--window-ps",
        type=float,
        default=250.0,
        help="Analyse the last WINDOW_PS of each trajectory (0 disables the window).",
    )
    parser.add_argument("--frame-interval-ps", type=float, default=0.4)
    parser.add_argument(
        "--rdf-frames",
        type=int,
        default=60,
        help="Frames used to locate the M-Cl and Na-Cl first minima.",
    )
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--radial-max", type=float, default=12.0)
    parser.add_argument("--radial-bin-width", type=float, default=0.10)
    return parser.parse_args()


NEIGHBOUR_CRITERIA = ("M-Cl", "union")

DEFAULT_CRITERION = "union"


COLORBAR_LABEL_X = 5.0


def window_frames(n_frames: int, args: argparse.Namespace) -> np.ndarray:
    """Frames of the trailing analysis window, or the legacy subsample."""
    if args.window_ps and args.window_ps > 0.0:
        wanted = int(round(args.window_ps / args.frame_interval_ps))
        return np.arange(max(0, n_frames - wanted), n_frames, dtype=int)
    return frame_indices(n_frames, args.discard_fraction, args.max_frames)


def refine_extremum(centers: np.ndarray, values: np.ndarray, index: int) -> float:
    """Sub-bin position of an extremum from a three-point parabola."""
    if 0 < index < len(values) - 1:
        left, middle, right = values[index - 1], values[index], values[index + 1]
        curvature = left - 2.0 * middle + right
        if curvature > 0.0:
            shift = 0.5 * (left - right) / curvature
            return float(centers[index] + shift * (centers[1] - centers[0]))
    return float(centers[index])


def first_rdf_minimum(
    histogram: np.ndarray,
    edges: np.ndarray,
    n_pairs: float,
    volume: float,
    peak_window: tuple[float, float] = (1.8, 3.4),
    search_limit: float = 5.6,
) -> float:
    """First minimum of g(r) after its first peak, from a raw pair histogram."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_volume = 4.0 * np.pi * centers**2 * np.diff(edges)
    g = histogram / (n_pairs / volume * shell_volume)
    g = savgol_filter(g, 21, 2, mode="interp")
    peak_mask = (centers >= peak_window[0]) & (centers <= peak_window[1])
    peak_index = np.flatnonzero(peak_mask)[np.argmax(g[peak_mask])]
    tail = np.flatnonzero((centers > centers[peak_index]) & (centers <= search_limit))
    return refine_extremum(centers, g, int(tail[np.argmin(g[tail])]))


def measure_pair_cutoffs(
    trajectory,
    selected: np.ndarray,
    central_indices: np.ndarray,
    na_indices: np.ndarray,
    cl_indices: np.ndarray,
    n_frames: int,
) -> tuple[float, float]:
    """First minima of the M--Cl and Na--Cl radial distribution functions."""
    edges = np.arange(0.0, 6.0 + 1e-9, 0.02)
    histograms = {"central": np.zeros(len(edges) - 1), "na": np.zeros(len(edges) - 1)}
    picks = np.unique(
        selected[np.linspace(0, len(selected) - 1, n_frames).round().astype(int)]
    )
    volume = 0.0
    for frame_index in picks:
        atoms = trajectory[int(frame_index)]
        lengths = ensure_orthorhombic(atoms)
        volume += float(np.prod(lengths))
        positions = np.mod(atoms.positions, lengths)
        cl_tree = cKDTree(positions[cl_indices], boxsize=lengths)
        k_neighbours = min(96, len(cl_indices))
        for key, indices in (("central", central_indices), ("na", na_indices)):
            distances, _ = cl_tree.query(
                positions[indices], k=k_neighbours, distance_upper_bound=edges[-1]
            )
            histograms[key] += np.histogram(distances[np.isfinite(distances)], bins=edges)[0]
    volume /= len(picks)
    n_cl = len(cl_indices)
    return (
        first_rdf_minimum(
            histograms["central"], edges, len(picks) * len(central_indices) * n_cl, volume
        ),
        first_rdf_minimum(
            histograms["na"], edges, len(picks) * len(na_indices) * n_cl, volume
        ),
    )


def system_pair_cutoffs(
    trajectory_path: Path, central_cation: str, args: argparse.Namespace
) -> tuple[float, float]:
    """M--Cl and Na--Cl first minima for one system, on the analysis window."""
    trajectory = Trajectory(str(trajectory_path), "r")
    symbols = np.asarray(trajectory[0].get_chemical_symbols())
    return measure_pair_cutoffs(
        trajectory,
        window_frames(len(trajectory), args),
        np.flatnonzero(symbols == central_cation),
        np.flatnonzero(symbols == "Na"),
        np.flatnonzero(symbols == "Cl"),
        args.rdf_frames,
    )


def cutoff_map(path: Path, cutoff_column: str) -> dict[str, float]:
    table = pd.read_csv(path)
    required = {"system", cutoff_column}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    return {
        str(row["system"]): float(row[cutoff_column])
        for _, row in table.iterrows()
    }


def block_assignment(n_selected: int, n_blocks: int) -> np.ndarray:
    assignment = np.empty(n_selected, dtype=int)
    for block_id, positions in enumerate(
        np.array_split(np.arange(n_selected, dtype=int), n_blocks)
    ):
        assignment[positions] = block_id
    return assignment


def minimum_image(vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return vectors - lengths * np.rint(vectors / lengths)


def random_category_probabilities(
    total_cations: int,
    central_cations: int,
    total_degree: int,
) -> np.ndarray:
    """Exact P(n_X=0,1,2,>=3 | total cation degree)."""
    degree = int(total_degree)
    explicit = np.asarray(
        [hypergeom.pmf(n, total_cations, central_cations, degree) for n in range(3)],
        dtype=float,
    )
    tail = max(0.0, 1.0 - float(explicit.sum()))
    probabilities = np.concatenate((explicit, [tail]))
    probabilities /= probabilities.sum()
    return probabilities


def alpha_from_counts(
    unlike_counts: np.ndarray,
    like_counts: np.ndarray,
    random_na_fraction: float,
) -> np.ndarray:
    denominator = unlike_counts + 2.0 * like_counts
    alpha = np.full(denominator.shape, np.nan, dtype=float)
    valid = denominator > 0
    conditional_na = unlike_counts[valid] / denominator[valid]
    alpha[valid] = 1.0 - conditional_na / random_na_fraction
    return alpha


def smooth_valid(values: np.ndarray, window: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = values.copy()
    valid = np.isfinite(values)
    if np.count_nonzero(valid) >= window:
        locations = np.flatnonzero(valid)
        filled = np.interp(np.arange(len(values)), locations, values[valid])
        result = savgol_filter(filled, window, 2, mode="interp")
        result[~valid] = np.nan
    return result


def mean_absolute_alpha(alpha: np.ndarray, r: np.ndarray, lower: float, upper: float) -> float:
    mask = (r >= lower) & (r <= upper) & np.isfinite(alpha)
    if np.count_nonzero(mask) < 2:
        return np.nan
    selected_r = r[mask]
    return float(
        np.trapezoid(np.abs(alpha[mask]), selected_r)
        / (selected_r[-1] - selected_r[0])
    )


def analyze_trajectory(
    trajectory_path: Path,
    central_cation: str,
    mixture: str,
    cutoff: float,
    radial_edges: np.ndarray,
    args: argparse.Namespace,
    union_cutoff: float | None = None,
) -> dict:
    trajectory = Trajectory(str(trajectory_path), "r")
    first = trajectory[0]
    symbols = np.asarray(first.get_chemical_symbols())
    counts = Counter(symbols.tolist())
    central_indices = np.flatnonzero(symbols == central_cation)
    na_indices = np.flatnonzero(symbols == "Na")
    cl_indices = np.flatnonzero(symbols == "Cl")
    cation_indices = np.concatenate((central_indices, na_indices))
    n_central = len(central_indices)
    n_na = len(na_indices)
    n_cations = len(cation_indices)
    n_cl = len(cl_indices)
    x_nacl = n_na / n_cations
    random_na_fraction = n_na / (n_cations - 1)

    selected = window_frames(len(trajectory), args)
    assignment = block_assignment(len(selected), args.blocks)
    r_central_cl, r_na_cl = measure_pair_cutoffs(
        trajectory, selected, central_indices, na_indices, cl_indices, args.rdf_frames
    )


    radii = {
        "M-Cl": cutoff,
        "union": union_cutoff if union_cutoff else max(r_central_cl, r_na_cl),
    }
    n_radial = len(radial_edges) - 1
    md_counts_blocks = {
        name: np.zeros((args.blocks, 4), dtype=np.int64) for name in NEIGHBOUR_CRITERIA
    }
    random_counts_blocks = {
        name: np.zeros((args.blocks, 4), dtype=float) for name in NEIGHBOUR_CRITERIA
    }
    degree_sums = {name: 0.0 for name in NEIGHBOUR_CRITERIA}
    unlike_hist_blocks = np.zeros((args.blocks, n_radial), dtype=np.int64)
    like_hist_blocks = np.zeros((args.blocks, n_radial), dtype=np.int64)
    frames_per_block = np.zeros(args.blocks, dtype=np.int64)
    random_cache: dict[int, np.ndarray] = {}

    for selected_position, frame_index in enumerate(selected):
        block_id = int(assignment[selected_position])
        atoms = trajectory[int(frame_index)]
        lengths = ensure_orthorhombic(atoms)
        positions = np.mod(atoms.positions, lengths)
        central_positions = positions[central_indices]
        na_positions = positions[na_indices]
        cl_positions = positions[cl_indices]
        cation_positions = np.vstack((central_positions, na_positions))
        frames_per_block[block_id] += 1

        central_tree = cKDTree(central_positions, boxsize=lengths)
        cation_tree = cKDTree(cation_positions, boxsize=lengths)
        for name, radius in radii.items():
            actual_degree = np.asarray(
                central_tree.query_ball_point(cl_positions, radius, return_length=True),
                dtype=np.int16,
            )
            md_counts_blocks[name][block_id] += np.bincount(
                np.minimum(actual_degree, 3), minlength=4
            )
            total_degree = np.asarray(
                cation_tree.query_ball_point(cl_positions, radius, return_length=True),
                dtype=np.int16,
            )
            degree_sums[name] += float(total_degree.sum())
            for degree, multiplicity in enumerate(np.bincount(total_degree)):
                if not multiplicity:
                    continue
                if degree not in random_cache:
                    random_cache[degree] = random_category_probabilities(
                        n_cations,
                        n_central,
                        degree,
                    )
                random_counts_blocks[name][block_id] += multiplicity * random_cache[degree]

        pairs = cation_tree.query_pairs(args.radial_max, output_type="ndarray")
        if len(pairs):
            displacements = cation_positions[pairs[:, 1]] - cation_positions[pairs[:, 0]]
            distances = np.linalg.norm(minimum_image(displacements, lengths), axis=1)
            bins = np.searchsorted(radial_edges, distances, side="right") - 1
            in_range = (bins >= 0) & (bins < n_radial)
            left_central = pairs[:, 0] < n_central
            right_central = pairs[:, 1] < n_central
            unlike = in_range & (left_central != right_central)
            like = in_range & left_central & right_central
            if np.any(unlike):
                unlike_hist_blocks[block_id] += np.bincount(
                    bins[unlike], minlength=n_radial
                )
            if np.any(like):
                like_hist_blocks[block_id] += np.bincount(bins[like], minlength=n_radial)

    connectivity = {}
    for name in NEIGHBOUR_CRITERIA:
        md_counts = md_counts_blocks[name]
        random_counts = random_counts_blocks[name]
        md_block = md_counts / md_counts.sum(axis=1, keepdims=True)
        random_block = random_counts / random_counts.sum(axis=1, keepdims=True)
        md_probability = md_counts.sum(axis=0) / md_counts.sum()
        random_probability = random_counts.sum(axis=0) / random_counts.sum()
        connectivity[name] = {
            "cutoff_A": radii[name],
            "md_probability": md_probability,
            "md_probability_sd": np.std(md_block, axis=0, ddof=1),
            "random_probability": random_probability,
            "delta_probability": md_probability - random_probability,
            "delta_probability_sd": np.std(md_block - random_block, axis=0, ddof=1),
            "mean_cation_degree": degree_sums[name] / (len(selected) * n_cl),
        }

    alpha_blocks = np.vstack(
        [
            alpha_from_counts(
                unlike_hist_blocks[block_id],
                like_hist_blocks[block_id],
                random_na_fraction,
            )
            for block_id in range(args.blocks)
        ]
    )
    alpha = alpha_from_counts(
        unlike_hist_blocks.sum(axis=0),
        like_hist_blocks.sum(axis=0),
        random_na_fraction,
    )
    alpha_sd = np.nanstd(alpha_blocks, axis=0, ddof=1)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    strength = mean_absolute_alpha(alpha, radial_centers, 3.0, 10.0)
    tail_strength = mean_absolute_alpha(alpha, radial_centers, 6.0, 10.0)
    strength_blocks = np.asarray(
        [mean_absolute_alpha(row, radial_centers, 3.0, 10.0) for row in alpha_blocks]
    )
    tail_strength_blocks = np.asarray(
        [mean_absolute_alpha(row, radial_centers, 6.0, 10.0) for row in alpha_blocks]
    )

    result = {
        "mixture": mixture,
        "central_cation": central_cation,
        "system": trajectory_path.parent.name,
        "path": str(trajectory_path),
        "counts": dict(counts),
        "x_nacl": x_nacl,
        "production_cutoff_A": cutoff,
        "r_MCl_min_A": r_central_cl,
        "r_NaCl_min_A": r_na_cl,
        "connectivity": connectivity,
        "n_total_frames": len(trajectory),
        "n_sampled_frames": len(selected),
        "window_ps": len(selected) * args.frame_interval_ps,
        "frames_per_block": frames_per_block,
        "radial_centers": radial_centers,
        "alpha": alpha,
        "alpha_sd": alpha_sd,
        "alpha_blocks": alpha_blocks,
        "random_na_fraction": random_na_fraction,
        "sro_strength": strength,
        "sro_strength_sd": float(np.nanstd(strength_blocks, ddof=1)),
        "sro_tail_strength": tail_strength,
        "sro_tail_strength_sd": float(np.nanstd(tail_strength_blocks, ddof=1)),
    }

    result.update(connectivity[DEFAULT_CRITERION])
    return result


def plot_connectivity_panel(
    axis,
    results: list[dict],
    letter: str,
    label: str,
    letter_dx: float = -24.0,
    random_legend: bool = False,
) -> None:
    """One Cl-connectivity panel: MD curve plus its random-label reference."""
    x = np.asarray([result["x_nacl"] for result in results])
    for category_id, _category in enumerate(CATEGORIES):
        color = CONNECTIVITY_COLORS[category_id]
        axis.plot(
            x,
            [result["random_probability"][category_id] for result in results],
            color=color,
            linestyle=(0, (3.0, 1.8)),
            linewidth=0.9,
            alpha=0.7,
            zorder=1,
        )
        axis.errorbar(
            x,
            [result["md_probability"][category_id] for result in results],
            yerr=[result["md_probability_sd"][category_id] for result in results],
            color=color,
            linewidth=1.4,
            elinewidth=0.7,
            capsize=0.0,
            zorder=2,
        )
    panel_letter(axis, letter, dx=letter_dx)
    panel_label(axis, label)
    axis.set_xlabel(r"$x_{\rm NaCl}$")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks((0.0, 0.5, 1.0))
    axis.set_yticks((0.0, 0.5, 1.0))

    axis.tick_params(which="major", length=3.0, top=False, right=False)
    axis.tick_params(which="minor", length=1.7, top=False, right=False)
    if random_legend:

        axis.legend(
            [Line2D([], [], color="0.35", linestyle=(0, (3.0, 1.8)), linewidth=0.9)],
            ["random"],
            fontsize=8.0,
            loc="upper center",
            bbox_to_anchor=(0.65, 0.85),
            handlelength=1.7,
            handletextpad=0.4,
            borderpad=0.0,
            borderaxespad=0.0,
        )


def composition_edges(x: np.ndarray) -> np.ndarray:
    """Cell edges in composition; the outer cells reach the pure end members."""
    midpoint = 0.5 * (x[:-1] + x[1:])
    return np.concatenate(([0.0], midpoint, [1.0]))


def alpha_matrix(results: list[dict], radial_mask: np.ndarray) -> np.ndarray:
    return np.vstack([smooth_valid(result["alpha"])[radial_mask] for result in results])


def plot_summary(li_results: list[dict], mg_results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 4.0), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.012, h_pad=0.012, wspace=0.02, hspace=0.03)
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.0, 1.0, 0.055), wspace=0.06, hspace=0.02
    )
    ax_li = fig.add_subplot(grid[0, 0])
    ax_mg = fig.add_subplot(grid[0, 1], sharey=ax_li)
    cax_class = fig.add_subplot(grid[0, 2])
    ax_li_map = fig.add_subplot(grid[1, 0])
    ax_mg_map = fig.add_subplot(grid[1, 1], sharey=ax_li_map)
    cax_alpha = fig.add_subplot(grid[1, 2])

    plot_connectivity_panel(
        ax_li, li_results, "a", "LiCl\u2013NaCl", random_legend=True
    )
    plot_connectivity_panel(
        ax_mg, mg_results, "b", "MgCl$_2$\u2013NaCl", letter_dx=-10.0
    )

    ax_li.set_ylabel(r"$P(n_{\rm X})$")
    ax_mg.tick_params(labelleft=False)


    class_cmap = mpl.colors.ListedColormap(CONNECTIVITY_COLORS)
    class_norm = mpl.colors.BoundaryNorm(
        np.arange(len(CATEGORIES) + 1) - 0.5, class_cmap.N
    )
    class_bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=class_norm, cmap=class_cmap),
        cax=cax_class,
        ticks=np.arange(len(CATEGORIES)),
    )
    class_bar.ax.set_yticklabels(CATEGORY_LABELS)


    class_bar.ax.tick_params(which="major", length=2.5, width=0.8)
    class_bar.ax.minorticks_off()
    class_bar.outline.set_linewidth(0.8)
    class_bar.set_label(r"$n_{\rm X}$", fontsize=LABEL_SIZE)
    class_bar.ax.yaxis.set_label_coords(COLORBAR_LABEL_X, 0.5)

    r = li_results[0]["radial_centers"]
    radial_mask = (r >= 3.0) & (r <= 10.0)
    li_matrix = alpha_matrix(li_results, radial_mask)
    mg_matrix = alpha_matrix(mg_results, radial_mask)


    finite = np.concatenate(
        [matrix[np.isfinite(matrix)].ravel() for matrix in (li_matrix, mg_matrix)]
    )
    limit = max(0.15, float(np.abs(finite).max()))
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    meshes = []
    for axis, results, matrix, letter, letter_dx, label, pair in (
        (ax_li_map, li_results, li_matrix, "c", -24.0, r"$\alpha_{\rm LiNa}$", "Li\u2013Na"),
        (ax_mg_map, mg_results, mg_matrix, "d", -10.0, r"$\alpha_{\rm MgNa}$", "Mg\u2013Na"),
    ):
        x = np.asarray([result["x_nacl"] for result in results])
        r_selected = r[radial_mask]
        r_edges = np.concatenate((r_selected - 0.05, [r_selected[-1] + 0.05]))
        mesh = axis.pcolormesh(
            r_edges,
            composition_edges(x),
            matrix,
            cmap=ALPHA_CMAP,
            norm=norm,
            shading="flat",
            rasterized=True,
        )
        meshes.append(mesh)


        axis.contour(
            r_selected,
            x,
            gaussian_filter(matrix, sigma=(1.0, 1.6), mode="nearest"),
            levels=[0.0],
            colors="0.10",
            linewidths=0.9,
            linestyles=[(0, (0.6, 1.6))],
            alpha=0.55,
        )

        axis.set_xlabel(rf"$r_{{\rm {pair}}}$ [$\mathrm{{\AA}}$]")
        axis.set_xlim(r_edges[0], r_edges[-1])
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks((4, 6, 8, 10))
        axis.set_yticks((0.0, 0.5, 1.0))
        axis.tick_params(which="major", direction="out", length=3.0)
        axis.tick_params(which="minor", direction="out", length=1.7)
        panel_letter(axis, letter, dx=letter_dx)
        panel_label(
            axis, label, halo=True, x=0.04, y=0.97, ha="left",
            fontsize=LABEL_SIZE + 1,
        )
    ax_li_map.set_ylabel(r"$x_{\rm NaCl}$")
    ax_mg_map.tick_params(labelleft=False)
    colorbar = fig.colorbar(meshes[-1], cax=cax_alpha)
    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.tick_params(length=2.5, width=0.8)
    colorbar.set_ticks(mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    colorbar.set_label(r"$\alpha_{\rm XNa}(r)$", fontsize=LABEL_SIZE)
    colorbar.ax.yaxis.set_label_coords(COLORBAR_LABEL_X, 0.5)

    fig.savefig(output_dir / "random_baseline_and_warren_cowley_comparison.png", dpi=600)
    fig.savefig(output_dir / "random_baseline_and_warren_cowley_comparison.pdf")
    plt.close(fig)


def nearest_result(results: list[dict], target: float) -> dict:
    return min(results, key=lambda result: abs(result["x_nacl"] - target))


def plot_alpha_profiles(li_results: list[dict], mg_results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    targets = (0.25, 0.50, 0.75)


    fig, axes = plt.subplots(
        3,
        1,
        figsize=(COLUMN_WIDTH_IN, 5.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for panel_id, (axis, target) in enumerate(zip(axes, targets)):
        for series_id, (results, label, linestyle) in enumerate(
            (
                (li_results, r"$\alpha_{\rm LiNa}(r)$", "--"),
                (mg_results, r"$\alpha_{\rm MgNa}(r)$", "-"),
            )
        ):
            result = nearest_result(results, target)
            r = result["radial_centers"]
            alpha = smooth_valid(result["alpha"])
            sd = smooth_valid(result["alpha_sd"])
            color = SERIES_COLORS[series_id]
            axis.plot(r, alpha, color=color, linestyle=linestyle, linewidth=1.4, label=label)
            lower = np.clip(alpha - sd, -0.55, 0.40)
            upper = np.clip(alpha + sd, -0.55, 0.40)
            axis.fill_between(r, lower, upper, color=color, alpha=0.12)
        axis.axhline(0.0, color="0.55", linewidth=0.8, linestyle=(0, (2, 2)))
        axis.set_xlim(3.3, 10.0)
        axis.set_ylim(-0.55, 0.40)
        axis.set_yticks((-0.4, -0.2, 0.0, 0.2, 0.4))
        panel_letter(axis, chr(ord("a") + panel_id))
        panel_label(
            axis,
            rf"$x_{{\rm NaCl}}\approx {target:.2f}$",
            x=0.985,
            y=0.95,
            ha="right",
        )
    axes[-1].set_xlabel(r"$r$ [$\mathrm{\AA}$]")
    fig.supylabel("Warren\u2013Cowley $\\alpha(r)$", fontsize=LABEL_SIZE)
    axes[0].legend(fontsize=LABEL_SIZE, loc="lower right", borderpad=0.2)
    fig.savefig(output_dir / "alpha_LiNa_vs_MgNa_profiles.png", dpi=450)
    fig.savefig(output_dir / "alpha_LiNa_vs_MgNa_profiles.pdf")
    plt.close(fig)


def write_outputs(
    results: list[dict],
    args: argparse.Namespace,
    union_cutoffs: dict[str, tuple[float, float]] | None = None,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "random_connectivity_baseline.csv").open("w", newline="") as handle:
        fields = [
            "mixture",
            "central_cation",
            "system",
            "x_NaCl",
            "criterion",
            "cutoff_A",
            "mean_cation_degree",
            "n_cation_Cl",
            "P_MD",
            "P_MD_block_sd",
            "P_random",
            "delta_P",
            "delta_P_block_sd",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for name in NEIGHBOUR_CRITERIA:
                block = result["connectivity"][name]
                for category_id, category in enumerate(CATEGORIES):
                    writer.writerow(
                        {
                            "mixture": result["mixture"],
                            "central_cation": result["central_cation"],
                            "system": result["system"],
                            "x_NaCl": result["x_nacl"],
                            "criterion": name,
                            "cutoff_A": block["cutoff_A"],
                            "mean_cation_degree": block["mean_cation_degree"],
                            "n_cation_Cl": category,
                            "P_MD": block["md_probability"][category_id],
                            "P_MD_block_sd": block["md_probability_sd"][category_id],
                            "P_random": block["random_probability"][category_id],
                            "delta_P": block["delta_probability"][category_id],
                            "delta_P_block_sd": block["delta_probability_sd"][category_id],
                        }
                    )

    with (args.output_dir / "warren_cowley_comparison.csv").open("w", newline="") as handle:
        fields = [
            "mixture",
            "central_cation",
            "system",
            "x_NaCl",
            "r_A",
            "alpha_XNa",
            "alpha_block_sd",
            "random_reference",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for r, alpha, alpha_sd in zip(
                result["radial_centers"], result["alpha"], result["alpha_sd"]
            ):
                writer.writerow(
                    {
                        "mixture": result["mixture"],
                        "central_cation": result["central_cation"],
                        "system": result["system"],
                        "x_NaCl": result["x_nacl"],
                        "r_A": r,
                        "alpha_XNa": alpha,
                        "alpha_block_sd": alpha_sd,
                        "random_reference": 0.0,
                    }
                )

    with (args.output_dir / "sro_strength_summary.csv").open("w", newline="") as handle:
        fields = [
            "mixture",
            "central_cation",
            "system",
            "x_NaCl",
            "mean_abs_alpha_3_10A",
            "block_sd_3_10A",
            "mean_abs_alpha_6_10A",
            "block_sd_6_10A",
            "sampled_frames",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "mixture": result["mixture"],
                    "central_cation": result["central_cation"],
                    "system": result["system"],
                    "x_NaCl": result["x_nacl"],
                    "mean_abs_alpha_3_10A": result["sro_strength"],
                    "block_sd_3_10A": result["sro_strength_sd"],
                    "mean_abs_alpha_6_10A": result["sro_tail_strength"],
                    "block_sd_6_10A": result["sro_tail_strength_sd"],
                    "sampled_frames": result["n_sampled_frames"],
                }
            )

    metadata = {
        "random_connectivity_reference": (
            "For every frame, keep positions, box, total cation-Cl neighbor degree, "
            "and exact cation counts; randomize Li/Na or Mg/Na labels analytically. "
            "P(n_X|m) is the exact hypergeometric distribution."
        ),
        "delta_P": "P_MD - P_random",
        "warren_cowley": (
            "alpha_XNa(r) = 1 - P(Na|X,r) / [N_Na/(N_cation-1)]; the finite-size "
            "correction makes the random-label reference exactly zero."
        ),
        "overall_order_metric": "mean absolute alpha over 3-10 A",
        "long_range_metric": "mean absolute alpha over 6-10 A",
        "neighbour_criteria": {
            "M-Cl": "first minimum of the M-Cl RDF (species specific, SI control)",
            "union": (
                "max(first minimum of M-Cl, first minimum of Na-Cl); covers the "
                "first shell of both cation species, so the selected neighbours "
                "are not biased towards M"
            ),
        },
        "criterion_used_in_figures": DEFAULT_CRITERION,
        "union_cutoff_A": {
            mixture: {"mean": value, "composition_spread": spread}
            for mixture, (value, spread) in (union_cutoffs or {}).items()
        },
        "analysis_window_ps": args.window_ps,
        "frame_interval_ps": args.frame_interval_ps,
        "discard_fraction": args.discard_fraction,
        "max_frames_per_composition": args.max_frames,
        "time_blocks": args.blocks,
        "uncertainty": "sample standard deviation across matched contiguous trajectory blocks",
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def results_from_csv(
    output_dir: Path,
    connectivity_csv: Path | None = None,
    criterion: str | None = None,
) -> dict[str, list[dict]]:
    """Rebuild the per-system plotting records from the saved CSV tables."""
    connectivity = pd.read_csv(
        connectivity_csv or output_dir / "random_connectivity_baseline.csv"
    )
    if criterion is None and "criterion" in connectivity.columns:
        criterion = DEFAULT_CRITERION
    if criterion is not None:
        if "criterion" not in connectivity.columns:
            raise ValueError("--criterion needs a table with a 'criterion' column")
        connectivity = connectivity[connectivity["criterion"] == criterion]
        if connectivity.empty:
            raise ValueError(f"no rows for criterion {criterion!r}")
    connectivity["n_cation_Cl"] = connectivity["n_cation_Cl"].astype(str)
    radial = pd.read_csv(output_dir / "warren_cowley_comparison.csv")


    sd_column = (
        "P_MD_block_sd"
        if "P_MD_block_sd" in connectivity.columns
        else "delta_P_block_sd"
    )
    grouped: dict[str, list[dict]] = {}
    for mixture, block in connectivity.groupby("mixture", sort=False):
        results = []
        for system, rows in block.groupby("system", sort=False):
            ordered = rows.set_index("n_cation_Cl").loc[list(CATEGORIES)]
            profile = radial[
                (radial["mixture"] == mixture) & (radial["system"] == system)
            ].sort_values("r_A")
            results.append(
                {
                    "mixture": mixture,
                    "system": system,
                    "x_nacl": float(ordered["x_NaCl"].iloc[0]),
                    "md_probability": ordered["P_MD"].to_numpy(dtype=float),
                    "md_probability_sd": ordered[sd_column].to_numpy(dtype=float),
                    "random_probability": ordered["P_random"].to_numpy(dtype=float),
                    "radial_centers": profile["r_A"].to_numpy(dtype=float),
                    "alpha": profile["alpha_XNa"].to_numpy(dtype=float),
                    "alpha_sd": profile["alpha_block_sd"].to_numpy(dtype=float),
                }
            )
        results.sort(key=lambda result: result["x_nacl"])
        grouped[mixture] = results
    return grouped


def main() -> None:
    args = parse_args()
    if args.replot_from_csv:
        system_results = results_from_csv(
            args.output_dir, args.connectivity_csv, args.criterion
        )
        figure_dir = args.figure_dir or args.output_dir
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_summary(
            system_results["LiCl-NaCl"], system_results["MgCl2-NaCl"], figure_dir
        )
        plot_alpha_profiles(
            system_results["LiCl-NaCl"], system_results["MgCl2-NaCl"], figure_dir
        )
        return
    li_cutoffs = cutoff_map(args.li_cutoff_csv, "r_min_cutoff_A")
    mg_cutoffs = cutoff_map(args.mg_cutoff_csv, "r_min_A")
    radial_edges = np.arange(
        0.0,
        args.radial_max + 0.5 * args.radial_bin_width,
        args.radial_bin_width,
    )
    specifications = (
        (
            "LiCl-NaCl",
            "Li",
            sorted(args.li_input_dir.glob("li*/md.traj")),
            li_cutoffs,
        ),
        (
            "MgCl2-NaCl",
            "Mg",
            sorted(args.mg_input_dir.glob("cl*/md.traj")),
            mg_cutoffs,
        ),
    )
    all_results = []
    union_cutoffs: dict[str, tuple[float, float]] = {}
    system_results: dict[str, list[dict]] = {}
    for mixture, central_cation, paths, cutoffs in specifications:
        if not paths:
            raise FileNotFoundError(f"No trajectories found for {mixture}")
        print(f"{mixture}: measuring pair cutoffs", flush=True)
        measured = [
            system_pair_cutoffs(path, central_cation, args) for path in paths
        ]
        union_cutoff = float(np.mean([max(pair) for pair in measured]))
        union_spread = float(np.std([max(pair) for pair in measured], ddof=1))
        print(
            f"{mixture}: union cutoff {union_cutoff:.3f} +- {union_spread:.3f} A "
            f"(mean over {len(paths)} compositions)",
            flush=True,
        )
        union_cutoffs[mixture] = (union_cutoff, union_spread)
        results = []
        for path in paths:
            system = path.parent.name
            if system not in cutoffs:
                raise KeyError(f"No cutoff found for {system}")
            print(f"{mixture}: {system}", flush=True)
            result = analyze_trajectory(
                path,
                central_cation,
                mixture,
                cutoffs[system],
                radial_edges,
                args,
                union_cutoff=union_cutoff,
            )
            print(
                f"  x_NaCl={result['x_nacl']:.3f}, frames={result['n_sampled_frames']} "
                f"({result['window_ps']:.0f} ps), r(M-Cl)={result['r_MCl_min_A']:.2f}, "
                f"r(Na-Cl)={result['r_NaCl_min_A']:.2f}, "
                f"<|alpha|>3-10A={result['sro_strength']:.4f}",
                flush=True,
            )
            results.append(result)
        results.sort(key=lambda result: result["x_nacl"])
        system_results[mixture] = results
        all_results.extend(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(all_results, args, union_cutoffs)
    plot_summary(system_results["LiCl-NaCl"], system_results["MgCl2-NaCl"], args.output_dir)
    plot_alpha_profiles(
        system_results["LiCl-NaCl"],
        system_results["MgCl2-NaCl"],
        args.output_dir,
    )
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
