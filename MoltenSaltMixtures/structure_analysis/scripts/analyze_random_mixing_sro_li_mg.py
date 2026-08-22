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
from matplotlib.patches import Circle
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
# Tick labels of the class colorbar; CATEGORIES itself also keys the CSV output.
# The last class collects every Cl with three or more neighbours, so it is
# labelled ">=3": ">3" would read as four or more.
CATEGORY_LABELS = ("0", "1", "2", r"$\geq 3$")
# Figures are one half of the A4 page width (105 mm) with type no smaller than
# 10 pt, so every panel is sized for a single journal column.
COLUMN_WIDTH_IN = 105.0 / 25.4
LABEL_SIZE = 10
LETTER_SIZE = 11
X_AXIS_LABEL_SIZE = 11
X_TICK_LABEL_SIZE = 10
# Cl connectivity is an ordinal variable, so the four classes run cool to warm
# (n=0 blue, n>=3 red).  The two cool classes are separated in hue as well as in
# lightness, so 0 and 1 stay distinct in print and under deuteranopia.
CONNECTIVITY_COLORS = ("#1A4F9C", "#3FBFC0", "#F0913D", "#B2182B")
# Li- versus Mg-based mixtures in the alpha(r) comparison panels.
SERIES_COLORS = ("#0072B2", "#D55E00")
# "vik" from Crameri's scientific colour maps: perceptually uniform, symmetric
# about zero and colorblind safe.  Sampled every eighth level of the published
# 256-colour table so the script stays free of an extra dependency.
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
        default=Path("raw_data/LiCl-NaCl"),
    )
    parser.add_argument(
        "--mg-input-dir",
        type=Path,
        default=Path("raw_data/MgCl2-NaCl"),
    )
    parser.add_argument(
        "--li-cutoff-csv",
        type=Path,
        default=Path(
            "results/3_Cl_coordination/"
            "LiCl_NaCl_LiCl_first_shell_metrics.csv"
        ),
    )
    parser.add_argument(
        "--mg-cutoff-csv",
        type=Path,
        default=Path("results/1_MgCl2_RDF/mgcl_first_shell_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/4_random_SRO_200ps"),
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
        default=200.0,
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
    parser.add_argument("--rdf-max", type=float, default=8.0)
    parser.add_argument("--rdf-bin-width", type=float, default=0.02)
    parser.add_argument("--radial-max", type=float, default=12.0)
    parser.add_argument("--radial-bin-width", type=float, default=0.10)
    return parser.parse_args()


NEIGHBOUR_CRITERIA = ("M-Cl", "union")
# Panels a/b use the species-neutral radius; "M-Cl" is kept for the SI control.
DEFAULT_CRITERION = "union"
# Shared horizontal position of both colour-bar labels, in units of the bar's
# own width, so that the two labels line up despite unequal tick-label widths.
COLORBAR_LABEL_X = 5.0
# Coordination sketch: smaller type than the axes, and one colour per species.
SKETCH_SIZE = 7.5
CATION_COLOR = "#3B82F6"
ANION_COLOR = "#50c878"
SKETCH_BORDER_COLOR = "0.30"
# Official 256-level cmocean "dense" sequential scale for RDF panels a/b.
COMPOSITION_NORM = mpl.colors.Normalize(0.0, 1.0)
DENSE_COLORS = (
    "#e6f1f1", "#e4f0f0", "#e3efef", "#e1eeef", "#dfedee", "#ddeded",
    "#dceced", "#daebec", "#d8eaec", "#d7e9eb", "#d5e9eb", "#d3e8ea",
    "#d1e7ea", "#d0e6e9", "#cee5e9", "#cce4e8", "#cbe4e8", "#c9e3e8",
    "#c7e2e7", "#c6e1e7", "#c4e0e6", "#c2dfe6", "#c1dfe6", "#bfdee6",
    "#bedde5", "#bcdce5", "#badbe5", "#b9dae4", "#b7dae4", "#b6d9e4",
    "#b4d8e4", "#b2d7e4", "#b1d6e3", "#afd5e3", "#aed4e3", "#acd4e3",
    "#abd3e3", "#a9d2e3", "#a8d1e3", "#a6d0e3", "#a5cfe2", "#a3cee2",
    "#a2cee2", "#a0cde2", "#9fcce2", "#9ecbe2", "#9ccae2", "#9bc9e2",
    "#9ac8e2", "#98c7e2", "#97c6e2", "#96c5e2", "#94c5e2", "#93c4e2",
    "#92c3e2", "#90c2e2", "#8fc1e2", "#8ec0e2", "#8dbfe2", "#8cbee2",
    "#8abde3", "#89bce3", "#88bbe3", "#87bae3", "#86b9e3", "#85b8e3",
    "#84b7e3", "#83b6e3", "#82b5e3", "#81b4e3", "#80b3e3", "#7fb2e3",
    "#7fb1e4", "#7eb0e4", "#7dafe4", "#7caee4", "#7bade4", "#7bace4",
    "#7aabe4", "#79aae4", "#79a9e4", "#78a8e4", "#78a7e4", "#77a6e4",
    "#77a5e4", "#76a4e5", "#76a3e5", "#75a1e5", "#75a0e5", "#759fe5",
    "#759ee5", "#749de5", "#749ce4", "#749be4", "#749ae4", "#7498e4",
    "#7397e4", "#7396e4", "#7395e4", "#7394e4", "#7393e3", "#7391e3",
    "#7390e3", "#738fe3", "#738ee2", "#748de2", "#748be2", "#748ae2",
    "#7489e1", "#7488e1", "#7487e0", "#7485e0", "#7584df", "#7583df",
    "#7582de", "#7581de", "#757fdd", "#757edd", "#767ddc", "#767cdc",
    "#767bdb", "#7679da", "#7678da", "#7777d9", "#7776d8", "#7775d7",
    "#7773d7", "#7772d6", "#7871d5", "#7870d4", "#786fd3", "#786ed2",
    "#786cd2", "#786bd1", "#786ad0", "#7969cf", "#7968ce", "#7966cd",
    "#7965cc", "#7964cb", "#7963ca", "#7962c9", "#7961c8", "#7960c7",
    "#795ec5", "#795dc4", "#795cc3", "#795bc2", "#795ac1", "#7959c0",
    "#7958bf", "#7957bd", "#7956bc", "#7954bb", "#7953ba", "#7952b8",
    "#7951b7", "#7950b6", "#794fb5", "#784eb3", "#784db2", "#784cb1",
    "#784baf", "#784aae", "#7849ad", "#7748ab", "#7747aa", "#7746a9",
    "#7745a7", "#7743a6", "#7642a5", "#7641a3", "#7640a2", "#763fa0",
    "#753e9f", "#753d9d", "#753c9c", "#743b9b", "#743b99", "#743a98",
    "#733996", "#733895", "#733793", "#723692", "#723590", "#72348f",
    "#71338d", "#71328c", "#70318a", "#703088", "#6f2f87", "#6f2e85",
    "#6e2d84", "#6e2d82", "#6d2c81", "#6d2b7f", "#6c2a7e", "#6c297c",
    "#6b287a", "#6b2879", "#6a2777", "#6a2675", "#692574", "#682472",
    "#682471", "#67236f", "#67226d", "#66216c", "#65216a", "#652068",
    "#641f67", "#631f65", "#621e63", "#621d62", "#611d60", "#601c5e",
    "#5f1b5d", "#5f1b5b", "#5e1a59", "#5d1a58", "#5c1956", "#5b1954",
    "#5a1853", "#5a1851", "#591750", "#58174e", "#57164c", "#56164b",
    "#551649", "#541548", "#531546", "#521544", "#511443", "#501441",
    "#4f1440", "#4e133e", "#4d133d", "#4b133b", "#4a133a", "#491238",
    "#481237", "#471236", "#461234", "#451133", "#441132", "#421130",
    "#41112f", "#40102e", "#3f102d", "#3e102b", "#3c102a", "#3b0f29",
    "#3a0f28", "#390f27", "#380f25", "#360e24",
)
COMPOSITION_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "high_contrast_green_blue",
    (
        "#D9ED92",
        "#B5E48C",
        "#99D98C",
        "#76C893",
        "#52B69A",
        "#34A0A4",
        "#168AAD",
        "#1A759F",
        "#1E6091",
        "#184E77",
    ),
    N=256,
)
# One line style per cation--Cl pair, shared by both RDF panels.
PAIR_STYLES = {
    "Li-Cl": ("Li\u2013Cl", (0, (4.5, 1.4, 1.0, 1.4))),
    "Na-Cl": ("Na\u2013Cl", "-"),
    "Mg-Cl": ("Mg\u2013Cl", (0, (1.0, 1.6))),
}


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
    # "M-Cl" is the historical, species-specific radius; "union" also covers the
    # looser Na first shell, so the selected neighbours are not biased towards M.
    # The measured minima carry no composition trend beyond their own scatter, so
    # the union radius is one number per mixture rather than one per system.
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
    # Cl-centred partial RDFs, on their own fine grid: the 0.1 A bins used for
    # alpha(r) cannot resolve a first peak.
    rdf_edges = np.arange(0.0, args.rdf_max + 0.5 * args.rdf_bin_width, args.rdf_bin_width)
    n_rdf = len(rdf_edges) - 1
    rdf_hist = {"X-Cl": np.zeros(n_rdf), "Na-Cl": np.zeros(n_rdf)}
    volume_sum = 0.0
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

        volume_sum += float(np.prod(lengths))
        cl_tree = cKDTree(cl_positions, boxsize=lengths)
        for pair_name, tree in (("X-Cl", central_tree), ("Na-Cl", cKDTree(na_positions, boxsize=lengths))):
            records = cl_tree.sparse_distance_matrix(
                tree, args.rdf_max, output_type="ndarray"
            )
            if len(records):
                rdf_hist[pair_name] += np.histogram(records["v"], bins=rdf_edges)[0]

    # g_ab(r) = counts / [n_frames * N_a N_b / <V> * shell volume]
    mean_volume = volume_sum / len(selected)
    shell_volume = (4.0 * np.pi / 3.0) * (rdf_edges[1:] ** 3 - rdf_edges[:-1] ** 3)
    rdf_centers = 0.5 * (rdf_edges[:-1] + rdf_edges[1:])
    rdf_curves = {
        pair_name: rdf_hist[pair_name]
        / (len(selected) * n_cl * partner / mean_volume * shell_volume)
        for pair_name, partner in (("X-Cl", n_central), ("Na-Cl", n_na))
    }

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
        "rdf_centers": rdf_centers,
        "rdf_x_cl": rdf_curves["X-Cl"],
        "rdf_na_cl": rdf_curves["Na-Cl"],
        "mean_volume_A3": mean_volume,
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
    # The plotting helpers read the default criterion straight off the record.
    result.update(connectivity[DEFAULT_CRITERION])
    return result


def plot_connectivity_panel(
    axis,
    results: list[dict],
    letter: str,
    label: str,
    letter_dx: float = -24.0,
    random_legend: bool = False,
    label_x: float = 0.5,
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
    panel_label(axis, label, x=label_x)
    axis.set_xlabel(r"$x_{\rm NaCl}$", fontsize=X_AXIS_LABEL_SIZE)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks((0.0, 0.5, 1.0))
    axis.set_yticks((0.0, 0.5, 1.0))
    # Frame stays closed on all four sides, but ticks only on the labelled axes.
    axis.tick_params(which="major", length=3.0, top=False, right=False)
    axis.tick_params(which="minor", length=1.7, top=False, right=False)
    axis.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
    if random_legend:
        # Two-row in-panel key at centre-left: line above, label below.
        axis.plot(
            [0.15, 0.35],
            [0.53, 0.53],
            transform=axis.transAxes,
            color="0.35",
            linestyle=(0, (3.0, 1.8)),
            linewidth=0.9,
            zorder=5,
        )
        axis.text(
            0.25,
            0.48,
            "random",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=LABEL_SIZE,
            zorder=5,
        )


def coordination_sketch(axis) -> None:
    """Inset cartoon: one Cl with its first shell of cations, drawn generically."""
    # Right of the cutoff line, below the legend, above the g(r) tail.
    inset = axis.inset_axes([0.35, 0.25, 0.63, 0.70])
    inset.set_xlim(-1.85, 1.85)
    inset.set_ylim(-1.85, 1.85)
    inset.set_aspect("equal", adjustable="box")
    inset.set_anchor("NE")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.minorticks_off()
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(0.78)
    inset.add_patch(
        Circle(
            (0.0, 0.0),
            1.30,
            fill=False,
            edgecolor=SKETCH_BORDER_COLOR,
            linewidth=0.8,
            linestyle=(0, (2.2, 1.6)),
        )
    )
    # Four cations at irregular angles and radii -- a melt shell, not a lattice --
    # plus a fifth centred exactly on the cutoff circle, the borderline neighbour
    # that the first-shell criterion has to decide on.
    for angle, radius in (
        (78.0, 0.90),
        (168.0, 0.84),
        (250.0, 0.94),
        (330.0, 0.82),
        (18.0, 1.30),
    ):
        centre = radius * np.array(
            [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]
        )
        inset.add_patch(
            Circle(centre, 0.34, facecolor=CATION_COLOR, edgecolor="none")
        )
        inset.text(
            centre[0],
            centre[1],
            "M",
            ha="center",
            va="center_baseline",
            fontsize=SKETCH_SIZE,
            color="white",
        )
    inset.add_patch(
        Circle((0.0, 0.0), 0.40, facecolor=ANION_COLOR, edgecolor="none")
    )
    inset.text(
        0.0,
        0.0,
        "Cl",
        ha="center",
        va="center_baseline",
        fontsize=SKETCH_SIZE,
        color="0.15",
    )


def plot_rdf_panel(
    axis, results: list[dict], letter: str, cation: str, letter_dx: float = -24.0
) -> None:
    """Cl-centred partial RDFs, one curve pair per composition."""
    x_label, x_style = PAIR_STYLES[f"{cation}-Cl"]
    na_label, na_style = PAIR_STYLES["Na-Cl"]
    ordered_results = sorted(results, key=lambda item: item["x_nacl"])
    # Show ten evenly distributed compositions instead of all closely spaced
    # trajectories, avoiding visual weighting toward any x_NaCl interval.
    target_x = np.linspace(
        ordered_results[0]["x_nacl"], ordered_results[-1]["x_nacl"], 10
    )
    plot_results = []
    for target in target_x:
        result = min(ordered_results, key=lambda item: abs(item["x_nacl"] - target))
        if not plot_results or result["system"] != plot_results[-1]["system"]:
            plot_results.append(result)
    for result in plot_results:
        colour = COMPOSITION_CMAP(COMPOSITION_NORM(result["x_nacl"]))
        axis.plot(
            result["rdf_centers"],
            result["rdf_x_cl"],
            color=colour,
            linewidth=0.9,
            linestyle=x_style,
        )
        axis.plot(
            result["rdf_centers"],
            result["rdf_na_cl"],
            color=colour,
            linewidth=0.9,
            linestyle=na_style,
        )
    # Reported system cutoffs for the RDF panels.
    cutoff = {"Li": 4.18, "Mg": 4.20}.get(
        cation, float(results[0].get("cutoff_A", np.nan))
    )
    if np.isfinite(cutoff):
        line_top = 0.19 if cation == "Li" else 0.14
        arrow_depth = 0.022
        arrow_half_width = 0.075
        axis.plot(
            [cutoff, cutoff],
            [0.0, line_top - arrow_depth],
            transform=axis.get_xaxis_transform(),
            color=SKETCH_BORDER_COLOR,
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            dash_capstyle="butt",
            zorder=4,
        )
        axis.plot(
            [cutoff - arrow_half_width, cutoff, cutoff + arrow_half_width],
            [line_top - arrow_depth, line_top, line_top - arrow_depth],
            transform=axis.get_xaxis_transform(),
            color=SKETCH_BORDER_COLOR,
            linewidth=0.9,
            linestyle="-",
            solid_capstyle="butt",
            solid_joinstyle="miter",
            clip_on=False,
            zorder=5,
        )
        label_y = line_top + 0.025 if cation == "Li" else 0.16
        label_zorder = 20 if cation == "Li" else 5
        label_effects = (
            []
            if cation == "Li"
            else [pe.withStroke(linewidth=1.8, foreground="white")]
        )
        # Build the label around the equals sign so the cutoff arrow points
        # exactly at its centre in both panels.
        for text, offset_points, alignment in (
            (r"$r_c$", -5.0, "right"),
            (r"$=$", 0.0, "center"),
            (rf"${cutoff:.2f}\ \mathrm{{\AA}}$", 5.0, "left"),
        ):
            axis.annotate(
                text,
                xy=(cutoff, label_y),
                xycoords=axis.get_xaxis_transform(),
                xytext=(offset_points, 0.0),
                textcoords="offset points",
                ha=alignment,
                va="bottom",
                fontsize=LABEL_SIZE - 2.5,
                color="black",
                zorder=label_zorder,
                path_effects=label_effects,
                annotation_clip=False,
            )
    peak = max(
        float(np.nanmax(result[key]))
        for result in results
        for key in ("rdf_x_cl", "rdf_na_cl")
    )
    axis.set_xlim(1.6, 7.0)
    axis.set_ylim(0.0, 1.22 * peak)   # just enough headroom for the legend
    axis.set_xticks((2, 4, 6))
    axis.set_xlabel(r"$r$ [$\mathrm{\AA}$]", fontsize=X_AXIS_LABEL_SIZE)
    axis.tick_params(which="major", length=3.0, top=False, right=False)
    axis.tick_params(which="minor", length=1.7, top=False, right=False)
    axis.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
    axis.legend(
        [
            Line2D([], [], color="0.35", linewidth=0.9, linestyle=x_style),
            Line2D([], [], color="0.35", linewidth=0.9, linestyle=na_style),
        ],
        [x_label, na_label],
        fontsize=LABEL_SIZE - 1.5,
        loc="upper center",
        ncol=2,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.0,
        borderaxespad=0.3,
    )
    panel_letter(axis, letter, dx=letter_dx)


def composition_edges(x: np.ndarray) -> np.ndarray:
    """Cell edges in composition; the outer cells reach the pure end members."""
    midpoint = 0.5 * (x[:-1] + x[1:])
    return np.concatenate(([0.0], midpoint, [1.0]))


def alpha_matrix(results: list[dict], radial_mask: np.ndarray) -> np.ndarray:
    return np.vstack([smooth_valid(result["alpha"])[radial_mask] for result in results])


def plot_summary(li_results: list[dict], mg_results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 5.7), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.012, h_pad=0.012, wspace=0.02, hspace=0.03)
    # Row 0 is slightly taller because it carries the coordination sketch.
    grid = fig.add_gridspec(
        3, 3, width_ratios=(1.0, 1.0, 0.055), height_ratios=(1.06, 1.0, 1.0)
    )
    ax_li_rdf = fig.add_subplot(grid[0, 0])
    ax_mg_rdf = fig.add_subplot(grid[0, 1])
    cax_comp = fig.add_subplot(grid[0, 2])
    ax_li = fig.add_subplot(grid[1, 0])
    ax_mg = fig.add_subplot(grid[1, 1], sharey=ax_li)
    cax_class = fig.add_subplot(grid[1, 2])
    ax_li_map = fig.add_subplot(grid[2, 0])
    ax_mg_map = fig.add_subplot(grid[2, 1], sharey=ax_li_map)
    cax_alpha = fig.add_subplot(grid[2, 2])

    # Row 0: Cl-centred partial RDFs.  The y scales are not shared, since the
    # Mg--Cl peak is twice the Li--Cl one and would flatten the left panel.
    plot_rdf_panel(ax_li_rdf, li_results, "a", "Li")
    plot_rdf_panel(ax_mg_rdf, mg_results, "b", "Mg", letter_dx=-10.0)
    ax_li_rdf.set_ylabel(r"$g(r)$")
    coordination_sketch(ax_li_rdf)
    composition_bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=COMPOSITION_NORM, cmap=COMPOSITION_CMAP),
        cax=cax_comp,
        ticks=(0.0, 0.5, 1.0),
    )
    composition_bar.ax.set_yticklabels(("0", "0.5", "1"))
    composition_bar.ax.tick_params(which="major", length=2.5, width=0.8)
    composition_bar.ax.minorticks_off()
    composition_bar.outline.set_linewidth(0.8)
    composition_bar.set_label(r"$x_{\rm NaCl}$", fontsize=LABEL_SIZE)
    composition_bar.ax.yaxis.set_label_coords(COLORBAR_LABEL_X, 0.5)

    plot_connectivity_panel(
        ax_li,
        li_results,
        "c",
        "LiCl\u2013NaCl",
        random_legend=True,
        label_x=0.57,
    )
    plot_connectivity_panel(
        ax_mg, mg_results, "d", "MgCl$_2$\u2013NaCl", letter_dx=-10.0
    )
    # Fraction of Cl anions in each connectivity class, i.e. its probability.
    ax_li.set_ylabel(r"$P(n_{\rm M})$")
    ax_mg.tick_params(labelleft=False)

    # The four connectivity classes are ordinal, so they get a discrete colorbar
    # instead of a legend; X is the panel's substituting cation, Li in a and Mg in b.
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
    # One short tick at the centre of each colour block; the minor ticks that a
    # BoundaryNorm colorbar puts on the block edges are removed.
    class_bar.ax.tick_params(which="major", length=2.5, width=0.8)
    class_bar.ax.minorticks_off()
    class_bar.outline.set_linewidth(0.8)
    class_bar.set_label(r"$n_{\rm M}$", fontsize=LABEL_SIZE)
    class_bar.ax.yaxis.set_label_coords(COLORBAR_LABEL_X, 0.5)

    r = li_results[0]["radial_centers"]
    radial_mask = (r >= 3.0) & (r <= 10.0)
    li_matrix = alpha_matrix(li_results, radial_mask)
    mg_matrix = alpha_matrix(mg_results, radial_mask)
    # The scale covers the full range of both maps, so no cell is clipped and the
    # deepest features can be read off the colour bar directly.
    finite = np.concatenate(
        [matrix[np.isfinite(matrix)].ravel() for matrix in (li_matrix, mg_matrix)]
    )
    limit = max(0.15, float(np.abs(finite).max()))
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    meshes = []
    for axis, results, matrix, letter, letter_dx, label, pair in (
        (ax_li_map, li_results, li_matrix, "e", -24.0, r"$\alpha_{\rm LiNa}$", "Li\u2013Na"),
        (ax_mg_map, mg_results, mg_matrix, "f", -10.0, r"$\alpha_{\rm MgNa}$", "Mg\u2013Na"),
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
        # The zero line is drawn from a lightly smoothed copy so that bin-to-bin
        # noise does not turn the alpha = 0 boundary into a ragged staircase.
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
        # r is a cation-cation separation, so the pair is named on the axis.
        axis.set_xlabel(
            rf"$r_{{\rm {pair}}}$ [$\mathrm{{\AA}}$]",
            fontsize=X_AXIS_LABEL_SIZE,
        )
        axis.set_xlim(r_edges[0], r_edges[-1])
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks((4, 6, 8, 10))
        axis.set_yticks((0.0, 0.5, 1.0))
        axis.tick_params(which="major", direction="out", length=3.0)
        axis.tick_params(which="minor", direction="out", length=1.7)
        axis.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
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
    colorbar.set_label(r"$\alpha_{\rm MNa}(r)$", fontsize=LABEL_SIZE)
    colorbar.ax.yaxis.set_label_coords(COLORBAR_LABEL_X, 0.5)

    fig.savefig(output_dir / "random_baseline_and_warren_cowley_comparison.png", dpi=600)
    fig.savefig(output_dir / "random_baseline_and_warren_cowley_comparison.pdf")
    plt.close(fig)


def nearest_result(results: list[dict], target: float) -> dict:
    return min(results, key=lambda result: abs(result["x_nacl"] - target))


def plot_alpha_profiles(li_results: list[dict], mg_results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    targets = (0.25, 0.50, 0.75)
    # Stacked rather than side by side: three panels across 105 mm would leave
    # each one too narrow for 10 pt tick labels.
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

    with (args.output_dir / "cation_cl_rdfs.csv").open("w", newline="") as handle:
        fields = ["mixture", "system", "x_NaCl", "pair", "r_A", "g_r"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for pair_name, key in (("X-Cl", "rdf_x_cl"), ("Na-Cl", "rdf_na_cl")):
                for r, g in zip(result["rdf_centers"], result[key]):
                    writer.writerow(
                        {
                            "mixture": result["mixture"],
                            "system": result["system"],
                            "x_NaCl": result["x_nacl"],
                            "pair": pair_name,
                            "r_A": r,
                            "g_r": g,
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
            "P(n_M|m) is the exact hypergeometric distribution, with M = Li or Mg."
        ),
        "delta_P": "P_MD - P_random",
        "warren_cowley": (
            "alpha_MNa(r) = 1 - P(Na|M,r) / [N_Na/(N_cation-1)]; the finite-size "
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
    rdf_path = output_dir / "cation_cl_rdfs.csv"
    rdfs = pd.read_csv(rdf_path) if rdf_path.exists() else None
    # Older tables stored only the block spread of P_MD - P_random; the random
    # reference is essentially constant across blocks, so it is the same error bar.
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
            record = {}
            if rdfs is not None:
                block_rdf = rdfs[
                    (rdfs["mixture"] == mixture) & (rdfs["system"] == system)
                ]
                for pair_name, key in (("X-Cl", "rdf_x_cl"), ("Na-Cl", "rdf_na_cl")):
                    curve = block_rdf[block_rdf["pair"] == pair_name].sort_values("r_A")
                    record[key] = curve["g_r"].to_numpy(dtype=float)
                    record["rdf_centers"] = curve["r_A"].to_numpy(dtype=float)
            results.append(
                {
                    **record,
                    "mixture": mixture,
                    "system": system,
                    "x_nacl": float(ordered["x_NaCl"].iloc[0]),
                    "cutoff_A": float(ordered["cutoff_A"].iloc[0]),
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
