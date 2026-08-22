#!/usr/bin/env python3
"""Compute composition-resolved partial RDFs for molten NaCl-MgCl2 trajectories.

The input directories are expected to contain ASE ``md.traj`` files.  For each
composition, the script samples the equilibrated tail of the trajectory,
computes all six unique partial RDFs with periodic boundaries, and performs a
block analysis of the first Mg-Cl coordination shell.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase.io.trajectory import Trajectory
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree


PAIR_LABELS = [
    ("Mg", "Cl"),
    ("Na", "Cl"),
    ("Mg", "Mg"),
    ("Mg", "Na"),
    ("Na", "Na"),
    ("Cl", "Cl"),
]

PAIR_TITLES = {
    ("Mg", "Cl"): r"Mg$^{2+}$--Cl$^{-}$",
    ("Na", "Cl"): r"Na$^{+}$--Cl$^{-}$",
    ("Mg", "Mg"): r"Mg$^{2+}$--Mg$^{2+}$",
    ("Mg", "Na"): r"Mg$^{2+}$--Na$^{+}$",
    ("Na", "Na"): r"Na$^{+}$--Na$^{+}$",
    ("Cl", "Cl"): r"Cl$^{-}$--Cl$^{-}$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("raw_data/MgCl2-NaCl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/1_MgCl2_RDF"),
    )
    parser.add_argument("--r-max", type=float, default=8.0, help="RDF cutoff in angstrom")
    parser.add_argument("--bin-width", type=float, default=0.02, help="RDF bin width in angstrom")
    parser.add_argument(
        "--discard-fraction",
        type=float,
        default=0.25,
        help="Fraction of initial trajectory frames to discard",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of evenly spaced production frames per composition",
    )
    parser.add_argument("--blocks", type=int, default=5, help="Number of time blocks")
    return parser.parse_args()


def frame_indices(n_frames: int, discard_fraction: float, max_frames: int) -> np.ndarray:
    start = int(np.ceil(n_frames * discard_fraction))
    available = np.arange(start, n_frames, dtype=int)
    if len(available) <= max_frames:
        return available
    choice = np.linspace(0, len(available) - 1, max_frames)
    return available[np.unique(np.rint(choice).astype(int))]


def ensure_orthorhombic(atoms) -> np.ndarray:
    cell = np.asarray(atoms.cell.array, dtype=float)
    offdiag = cell - np.diag(np.diag(cell))
    if not np.all(atoms.pbc) or not np.allclose(offdiag, 0.0, atol=1e-8):
        raise ValueError("This fast RDF implementation requires a periodic orthorhombic cell")
    lengths = np.diag(cell)
    if np.any(lengths <= 0):
        raise ValueError("Invalid cell lengths")
    return lengths


def pair_lookup(symbols: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    species_to_code = {symbol: code for code, symbol in enumerate(("Mg", "Na", "Cl"))}
    codes = np.asarray([species_to_code[symbol] for symbol in symbols], dtype=np.int8)
    lookup = np.full((3, 3), -1, dtype=np.int8)
    for pair_id, (left, right) in enumerate(PAIR_LABELS):
        i, j = species_to_code[left], species_to_code[right]
        lookup[i, j] = lookup[j, i] = pair_id
    return lookup, species_to_code


def hist_for_frame(atoms, codes: np.ndarray, lookup: np.ndarray, edges: np.ndarray) -> np.ndarray:
    lengths = ensure_orthorhombic(atoms)
    positions = np.mod(atoms.get_positions(), lengths)
    pairs = cKDTree(positions, boxsize=lengths).query_pairs(edges[-1], output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((len(PAIR_LABELS), len(edges) - 1), dtype=np.int64)

    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    delta -= lengths * np.rint(delta / lengths)
    distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    bins = np.searchsorted(edges, distances, side="right") - 1
    valid = (bins >= 0) & (bins < len(edges) - 1)

    pair_ids = lookup[codes[pairs[:, 0]], codes[pairs[:, 1]]]
    valid &= pair_ids >= 0
    flat = pair_ids[valid].astype(np.int64) * (len(edges) - 1) + bins[valid]
    return np.bincount(
        flat,
        minlength=len(PAIR_LABELS) * (len(edges) - 1),
    ).reshape(len(PAIR_LABELS), len(edges) - 1)


def quadratic_extremum(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, float]:
    if index <= 0 or index >= len(y) - 1:
        return float(x[index]), float(y[index])
    denominator = y[index - 1] - 2.0 * y[index] + y[index + 1]
    if abs(denominator) < 1e-14:
        return float(x[index]), float(y[index])
    offset = 0.5 * (y[index - 1] - y[index + 1]) / denominator
    offset = float(np.clip(offset, -1.0, 1.0))
    dx = x[index + 1] - x[index]
    x_ext = x[index] + offset * dx
    y_ext = y[index] - 0.25 * (y[index - 1] - y[index + 1]) * offset
    return float(x_ext), float(y_ext)


def mgcl_metrics(
    centers: np.ndarray,
    rdf: np.ndarray,
    histogram: np.ndarray,
    n_frames: int,
    n_mg: int,
) -> dict[str, float]:
    window = min(11, len(rdf) if len(rdf) % 2 else len(rdf) - 1)
    smooth = savgol_filter(rdf, max(window, 5), 3, mode="interp")

    peak_mask = (centers >= 1.65) & (centers <= 3.00)
    peak_candidates = np.flatnonzero(peak_mask)
    peak_index = int(peak_candidates[np.argmax(smooth[peak_mask])])
    peak_r, peak_g = quadratic_extremum(centers, smooth, peak_index)

    minimum_mask = (centers >= peak_r + 0.35) & (centers <= 4.20)
    minimum_candidates = np.flatnonzero(minimum_mask)
    minimum_index = int(minimum_candidates[np.argmin(smooth[minimum_mask])])
    minimum_r, minimum_g = quadratic_extremum(centers, -smooth, minimum_index)
    minimum_g = -minimum_g

    shell_mask = centers <= minimum_r
    shell_counts = histogram[shell_mask].astype(float)
    shell_r = centers[shell_mask]
    total_shell_counts = shell_counts.sum()
    coordination = float(total_shell_counts / (n_frames * n_mg))
    shell_mean = float(np.dot(shell_counts, shell_r) / total_shell_counts)
    shell_std = float(
        np.sqrt(np.dot(shell_counts, (shell_r - shell_mean) ** 2) / total_shell_counts)
    )
    return {
        "r_peak_A": peak_r,
        "g_peak": peak_g,
        "r_min_A": minimum_r,
        "g_min": minimum_g,
        "coordination_number": coordination,
        "first_shell_mean_A": shell_mean,
        "first_shell_std_A": shell_std,
    }


def analyze_trajectory(
    path: Path,
    edges: np.ndarray,
    discard_fraction: float,
    max_frames: int,
    n_blocks: int,
) -> dict:
    trajectory = Trajectory(str(path), "r")
    first = trajectory[0]
    symbols = first.get_chemical_symbols()
    counts = Counter(symbols)
    indices = frame_indices(len(trajectory), discard_fraction, max_frames)
    blocks = np.array_split(np.arange(len(indices)), n_blocks)

    lookup, species_to_code = pair_lookup(symbols)
    codes = np.asarray([species_to_code[symbol] for symbol in symbols], dtype=np.int8)
    n_bins = len(edges) - 1
    hist_blocks = np.zeros((n_blocks, len(PAIR_LABELS), n_bins), dtype=np.int64)
    norm_blocks = np.zeros((n_blocks, len(PAIR_LABELS)), dtype=float)
    volume_blocks = np.zeros(n_blocks, dtype=float)
    frames_per_block = np.zeros(n_blocks, dtype=int)

    pair_factors = []
    for left, right in PAIR_LABELS:
        if left == right:
            pair_factors.append(counts[left] * (counts[left] - 1) / 2.0)
        else:
            pair_factors.append(counts[left] * counts[right])
    pair_factors = np.asarray(pair_factors, dtype=float)

    block_for_position = np.empty(len(indices), dtype=int)
    for block_id, positions in enumerate(blocks):
        block_for_position[positions] = block_id

    for selected_position, frame_index in enumerate(indices):
        block_id = int(block_for_position[selected_position])
        atoms = trajectory[int(frame_index)]
        if atoms.get_chemical_symbols() != symbols:
            raise ValueError(f"Species ordering changed in {path} at frame {frame_index}")
        volume = float(atoms.get_volume())
        hist_blocks[block_id] += hist_for_frame(atoms, codes, lookup, edges)
        norm_blocks[block_id] += pair_factors / volume
        volume_blocks[block_id] += volume
        frames_per_block[block_id] += 1

    shell_volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    block_rdfs = hist_blocks / (norm_blocks[:, :, None] * shell_volumes[None, None, :])
    hist_total = hist_blocks.sum(axis=0)
    norm_total = norm_blocks.sum(axis=0)
    rdf_total = hist_total / (norm_total[:, None] * shell_volumes[None, :])
    centers = 0.5 * (edges[:-1] + edges[1:])

    mgcl_id = PAIR_LABELS.index(("Mg", "Cl"))
    mean_cl_density = float(norm_total[mgcl_id] / (counts["Mg"] * len(indices)))
    full_metrics = mgcl_metrics(
        centers,
        rdf_total[mgcl_id],
        hist_total[mgcl_id],
        len(indices),
        counts["Mg"],
    )
    block_metrics = []
    for block_id in range(n_blocks):
        block_metrics.append(
            mgcl_metrics(
                centers,
                block_rdfs[block_id, mgcl_id],
                hist_blocks[block_id, mgcl_id],
                int(frames_per_block[block_id]),
                counts["Mg"],
            )
        )

    for key in tuple(full_metrics):
        values = np.asarray([entry[key] for entry in block_metrics], dtype=float)
        full_metrics[f"{key}_block_sd"] = float(values.std(ddof=1))

    n_cations = counts["Na"] + counts["Mg"]
    x_nacl = counts["Na"] / n_cations
    return {
        "name": path.parent.name,
        "path": str(path),
        "n_total_frames": len(trajectory),
        "sampled_indices": indices,
        "n_sampled_frames": len(indices),
        "counts": dict(counts),
        "x_nacl": x_nacl,
        "x_mgcl2": 1.0 - x_nacl,
        "mean_volume_A3": float(volume_blocks.sum() / frames_per_block.sum()),
        "mean_cl_density_A3": mean_cl_density,
        "centers": centers,
        "rdf": rdf_total,
        "rdf_block_sd": block_rdfs.std(axis=0, ddof=1),
        "mgcl_metrics": full_metrics,
    }


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def composition_color(x_nacl: float):
    return mpl.colormaps["viridis"](Normalize(0.0, 1.0)(x_nacl))


def plot_all_partial_rdfs(results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.5), sharex=True)
    norm = Normalize(0.0, 1.0)
    cmap = mpl.colormaps["viridis"]

    for pair_id, (pair, axis) in enumerate(zip(PAIR_LABELS, axes.flat)):
        for result in results:
            axis.plot(
                result["centers"],
                result["rdf"][pair_id],
                color=cmap(norm(result["x_nacl"])),
                linewidth=1.35 if pair == ("Mg", "Cl") else 1.05,
                alpha=0.95,
            )
        axis.axhline(1.0, color="0.55", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
        axis.set_title(PAIR_TITLES[pair])
        axis.set_xlim(1.5, 8.0)
        axis.set_ylim(bottom=0.0)
        axis.text(
            0.02,
            0.96,
            chr(ord("a") + pair_id),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=12,
        )
        if pair_id % 3 == 0:
            axis.set_ylabel(r"$g_{ij}(r)$")
        if pair_id >= 3:
            axis.set_xlabel(r"$r$ ($\mathrm{\AA}$)")

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.94, wspace=0.23, hspace=0.24)
    colorbar_axis = fig.add_axes((0.91, 0.18, 0.018, 0.66))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=colorbar_axis)
    cbar.set_label(r"$x_{\mathrm{NaCl}}=N_{\mathrm{Na}}/(N_{\mathrm{Na}}+N_{\mathrm{Mg}})$")
    fig.savefig(output_dir / "partial_rdfs_all_compositions.png", dpi=450)
    fig.savefig(output_dir / "partial_rdfs_all_compositions.pdf")
    plt.close(fig)


def metric_with_error(result: dict, key: str) -> tuple[float, float]:
    metrics = result["mgcl_metrics"]
    return metrics[key], metrics[f"{key}_block_sd"]


def plot_mgcl_focus(results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    norm = Normalize(0.0, 1.0)
    cmap = mpl.colormaps["viridis"]
    mgcl_id = PAIR_LABELS.index(("Mg", "Cl"))

    rdf_axis = axes[0, 0]
    for result in results:
        rdf_axis.plot(
            result["centers"],
            result["rdf"][mgcl_id],
            color=cmap(norm(result["x_nacl"])),
            linewidth=1.45,
        )
    rdf_axis.axhline(1.0, color="0.55", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    rdf_axis.set_xlim(1.65, 4.8)
    rdf_axis.set_ylim(bottom=0.0)
    rdf_axis.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    rdf_axis.set_ylabel(r"$g_{\mathrm{MgCl}}(r)$")
    rdf_axis.set_title(r"Mg$^{2+}$--Cl$^{-}$ RDF")

    x = np.asarray([result["x_nacl"] for result in results])
    metrics_to_plot = [
        ("r_peak_A", r"First-peak position ($\mathrm{\AA}$)", axes[0, 1]),
        ("first_shell_std_A", r"First-shell width, $\sigma_r$ ($\mathrm{\AA}$)", axes[1, 0]),
        ("coordination_number", r"Mg--Cl coordination number", axes[1, 1]),
    ]
    for key, ylabel, axis in metrics_to_plot:
        values, errors = zip(*(metric_with_error(result, key) for result in results))
        axis.errorbar(
            x,
            values,
            yerr=errors,
            color="0.15",
            ecolor="0.45",
            marker="o",
            markersize=4.5,
            linewidth=1.2,
            capsize=2.5,
        )
        axis.set_xlabel(r"$x_{\mathrm{NaCl}}$")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.02, 1.02)

    for panel_id, axis in enumerate(axes.flat):
        axis.text(
            0.02,
            0.96,
            chr(ord("a") + panel_id),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=12,
        )

    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.10, top=0.95, wspace=0.34, hspace=0.32)
    colorbar_axis = fig.add_axes((0.875, 0.59, 0.018, 0.28))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=colorbar_axis)
    cbar.set_label(r"$x_{\mathrm{NaCl}}$")
    fig.savefig(output_dir / "mgcl_rdf_composition_trends.png", dpi=450)
    fig.savefig(output_dir / "mgcl_rdf_composition_trends.pdf")
    plt.close(fig)


def plot_mgcl_shell_distributions(results: list[dict], output_dir: Path) -> None:
    """Compare density-weighted and cumulative Mg-Cl first-shell populations."""
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharex=True)
    norm = Normalize(0.0, 1.0)
    cmap = mpl.colormaps["viridis"]
    mgcl_id = PAIR_LABELS.index(("Mg", "Cl"))
    displayed_dndr_max = 0.0
    displayed_cumulative_max = 0.0

    for result in results:
        centers = result["centers"]
        dr = float(centers[1] - centers[0])
        edges = np.concatenate(([centers[0] - dr / 2.0], centers + dr / 2.0))
        shell_volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
        shell_population = (
            result["mean_cl_density_A3"] * result["rdf"][mgcl_id] * shell_volumes
        )
        color = cmap(norm(result["x_nacl"]))
        axes[0].plot(centers, shell_population / dr, color=color, linewidth=1.4)
        axes[1].plot(centers, np.cumsum(shell_population), color=color, linewidth=1.4)
        display_mask = centers <= 4.0
        displayed_dndr_max = max(
            displayed_dndr_max,
            float(np.max((shell_population / dr)[display_mask])),
        )
        displayed_cumulative_max = max(
            displayed_cumulative_max,
            float(np.max(np.cumsum(shell_population)[display_mask])),
        )

    axes[0].set_ylabel(r"$\mathrm{d}N_{\mathrm{Mg-Cl}}/\mathrm{d}r$ ($\mathrm{\AA}^{-1}$)")
    axes[1].set_ylabel(r"$N_{\mathrm{Mg-Cl}}(r)$")
    axes[0].set_ylim(0.0, 1.08 * displayed_dndr_max)
    axes[1].set_ylim(0.0, 1.08 * displayed_cumulative_max)
    for panel_id, axis in enumerate(axes):
        axis.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
        axis.set_xlim(1.65, 4.0)
        axis.set_ylim(bottom=0.0)
        axis.text(
            0.02,
            0.96,
            chr(ord("a") + panel_id),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=12,
        )

    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.17, top=0.95, wspace=0.34)
    colorbar_axis = fig.add_axes((0.875, 0.24, 0.018, 0.62))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=colorbar_axis)
    cbar.set_label(r"$x_{\mathrm{NaCl}}$")
    fig.savefig(output_dir / "mgcl_first_shell_distributions.png", dpi=450)
    fig.savefig(output_dir / "mgcl_first_shell_distributions.pdf")
    plt.close(fig)


def write_outputs(results: list[dict], output_dir: Path, settings: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "composition_summary.csv").open("w", newline="") as handle:
        fields = [
            "system",
            "N_Na",
            "N_Mg",
            "N_Cl",
            "x_NaCl",
            "x_MgCl2",
            "total_frames",
            "sampled_frames",
            "mean_volume_A3",
            "mean_cl_density_A3",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            counts = result["counts"]
            writer.writerow(
                {
                    "system": result["name"],
                    "N_Na": counts["Na"],
                    "N_Mg": counts["Mg"],
                    "N_Cl": counts["Cl"],
                    "x_NaCl": f"{result['x_nacl']:.9f}",
                    "x_MgCl2": f"{result['x_mgcl2']:.9f}",
                    "total_frames": result["n_total_frames"],
                    "sampled_frames": result["n_sampled_frames"],
                    "mean_volume_A3": f"{result['mean_volume_A3']:.6f}",
                    "mean_cl_density_A3": f"{result['mean_cl_density_A3']:.10f}",
                }
            )

    with (output_dir / "mgcl_first_shell_metrics.csv").open("w", newline="") as handle:
        metric_keys = [
            "r_peak_A",
            "g_peak",
            "r_min_A",
            "g_min",
            "coordination_number",
            "first_shell_mean_A",
            "first_shell_std_A",
        ]
        fields = ["system", "x_NaCl", "x_MgCl2"]
        for key in metric_keys:
            fields.extend((key, f"{key}_block_sd"))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "system": result["name"],
                "x_NaCl": f"{result['x_nacl']:.9f}",
                "x_MgCl2": f"{result['x_mgcl2']:.9f}",
            }
            for key in metric_keys:
                row[key] = f"{result['mgcl_metrics'][key]:.8f}"
                row[f"{key}_block_sd"] = f"{result['mgcl_metrics'][f'{key}_block_sd']:.8f}"
            writer.writerow(row)

    with (output_dir / "partial_rdf_curves.csv").open("w", newline="") as handle:
        fields = ["system", "x_NaCl", "pair", "r_A", "g_r", "g_r_block_sd"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for pair_id, pair in enumerate(PAIR_LABELS):
                pair_name = "-".join(pair)
                for r_value, g_value, g_sd in zip(
                    result["centers"],
                    result["rdf"][pair_id],
                    result["rdf_block_sd"][pair_id],
                ):
                    writer.writerow(
                        {
                            "system": result["name"],
                            "x_NaCl": f"{result['x_nacl']:.9f}",
                            "pair": pair_name,
                            "r_A": f"{r_value:.5f}",
                            "g_r": f"{g_value:.8f}",
                            "g_r_block_sd": f"{g_sd:.8f}",
                        }
                    )

    metadata = {
        "settings": settings,
        "pair_order": ["-".join(pair) for pair in PAIR_LABELS],
        "composition_definition": "x_NaCl = N_Na / (N_Na + N_Mg)",
        "uncertainty": "sample standard deviation across contiguous time blocks",
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_bins = int(np.ceil(args.r_max / args.bin_width))
    edges = np.linspace(0.0, args.r_max, n_bins + 1)
    trajectory_paths = sorted(args.input_dir.glob("cl*/md.traj"))
    if not trajectory_paths:
        raise FileNotFoundError(f"No cl*/md.traj files found below {args.input_dir}")

    results = []
    for path in trajectory_paths:
        print(f"Analyzing {path.parent.name} ...", flush=True)
        results.append(
            analyze_trajectory(
                path,
                edges,
                args.discard_fraction,
                args.max_frames,
                args.blocks,
            )
        )
    results.sort(key=lambda entry: entry["x_nacl"])

    settings = {
        "input_dir": str(args.input_dir),
        "r_max_A": args.r_max,
        "bin_width_A": args.bin_width,
        "discard_fraction": args.discard_fraction,
        "max_frames_per_composition": args.max_frames,
        "time_blocks": args.blocks,
    }
    write_outputs(results, args.output_dir, settings)
    plot_all_partial_rdfs(results, args.output_dir)
    plot_mgcl_focus(results, args.output_dir)
    plot_mgcl_shell_distributions(results, args.output_dir)
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
