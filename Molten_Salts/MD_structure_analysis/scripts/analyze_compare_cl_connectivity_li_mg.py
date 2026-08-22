#!/usr/bin/env python3
"""Compare Cl connectivity in LiCl-NaCl and MgCl2-NaCl trajectories.

For LiCl-NaCl, the Li-Cl neighbor cutoff is obtained independently at every
composition from the first minimum of the Li-Cl RDF.  The distribution of
``n_Li(Cl)`` is then evaluated using that cutoff.  Published MgCl2-NaCl
``n_Mg(Cl)`` results are read from the existing structure-analysis tables so
that both panels use exactly the same definition and block-error convention.
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
import pandas as pd
from ase.io.trajectory import Trajectory
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from analyze_mgcl2_nacl_rdfs import (
    configure_matplotlib,
    ensure_orthorhombic,
    frame_indices,
    quadratic_extremum,
)


CATEGORY_LABELS = ("0", "1", "2", "3+")
CONNECTIVITY_COLORS = ("#0047AB", "#FF8C00", "#00A000", "#D00070")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--li-input-dir",
        type=Path,
        default=Path("MD_inputs/NaCl-LiCl_MD"),
    )
    parser.add_argument(
        "--mg-coordination-csv",
        type=Path,
        default=Path(
            "outputs/SelectedMgCl2-NaCl_MD_structure/coordination_distributions.csv"
        ),
    )
    parser.add_argument(
        "--mg-summary-csv",
        type=Path,
        default=Path("outputs/SelectedMgCl2-NaCl_MD_structure/structural_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/LiCl-NaCl_vs_MgCl2-NaCl_Cl_coordination"),
    )
    parser.add_argument(
        "--random-baseline-csv",
        type=Path,
        default=Path(
            "outputs/LiCl-NaCl_vs_MgCl2-NaCl_random_SRO/"
            "random_connectivity_baseline.csv"
        ),
    )
    parser.add_argument("--discard-fraction", type=float, default=0.25)
    parser.add_argument("--rdf-frames", type=int, default=300)
    parser.add_argument("--connectivity-frames", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--r-max", type=float, default=4.2)
    parser.add_argument("--bin-width", type=float, default=0.02)
    return parser.parse_args()


def numeric_system_key(path: Path) -> int:
    return int(path.parent.name.removeprefix("li"))


def cross_distances(
    left_positions: np.ndarray,
    right_positions: np.ndarray,
    lengths: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    left = cKDTree(np.mod(left_positions, lengths), boxsize=lengths)
    right = cKDTree(np.mod(right_positions, lengths), boxsize=lengths)
    matrix = left.sparse_distance_matrix(right, cutoff, output_type="coo_matrix")
    return np.asarray(matrix.data, dtype=float)


def first_shell_metrics(centers: np.ndarray, rdf: np.ndarray) -> tuple[float, float, float, float]:
    window = min(11, len(rdf) if len(rdf) % 2 else len(rdf) - 1)
    smooth = savgol_filter(rdf, max(window, 5), 3, mode="interp")
    peak_candidates = np.flatnonzero((centers >= 1.55) & (centers <= 2.85))
    peak_index = int(peak_candidates[np.argmax(smooth[peak_candidates])])
    peak_r, peak_g = quadratic_extremum(centers, smooth, peak_index)
    minimum_candidates = np.flatnonzero(
        (centers >= peak_r + 0.35) & (centers <= 4.10)
    )
    minimum_index = int(minimum_candidates[np.argmin(smooth[minimum_candidates])])
    minimum_r, negative_minimum_g = quadratic_extremum(
        centers, -smooth, minimum_index
    )
    return peak_r, peak_g, minimum_r, -negative_minimum_g


def li_rdf_and_cutoff(
    trajectory_path: Path,
    args: argparse.Namespace,
    edges: np.ndarray,
) -> dict:
    trajectory = Trajectory(str(trajectory_path), "r")
    first = trajectory[0]
    symbols = np.asarray(first.get_chemical_symbols())
    counts = Counter(symbols.tolist())
    li_indices = np.flatnonzero(symbols == "Li")
    cl_indices = np.flatnonzero(symbols == "Cl")
    n_li, n_cl = len(li_indices), len(cl_indices)
    x_nacl = counts["Na"] / (counts["Na"] + counts["Li"])

    selected = frame_indices(len(trajectory), args.discard_fraction, args.rdf_frames)
    blocks = np.array_split(selected, args.blocks)
    hist_blocks = np.zeros((args.blocks, len(edges) - 1), dtype=np.int64)
    norm_blocks = np.zeros(args.blocks, dtype=float)

    for block_id, block_indices in enumerate(blocks):
        for frame_index in block_indices:
            atoms = trajectory[int(frame_index)]
            lengths = ensure_orthorhombic(atoms)
            distances = cross_distances(
                atoms.positions[li_indices], atoms.positions[cl_indices], lengths, edges[-1]
            )
            hist_blocks[block_id] += np.histogram(distances, bins=edges)[0]
            norm_blocks[block_id] += n_li * n_cl / atoms.get_volume()

    shell_volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    block_rdfs = hist_blocks / (norm_blocks[:, None] * shell_volumes[None, :])
    rdf = hist_blocks.sum(axis=0) / (norm_blocks.sum() * shell_volumes)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_r, peak_g, cutoff, minimum_g = first_shell_metrics(centers, rdf)

    block_cutoffs = []
    for block_rdf in block_rdfs:
        block_cutoffs.append(first_shell_metrics(centers, block_rdf)[2])

    return {
        "system": trajectory_path.parent.name,
        "path": str(trajectory_path),
        "counts": dict(counts),
        "x_nacl": x_nacl,
        "n_total_frames": len(trajectory),
        "n_rdf_frames": len(selected),
        "r": centers,
        "rdf": rdf,
        "rdf_sd": np.std(block_rdfs, axis=0, ddof=1),
        "r_peak_A": peak_r,
        "g_peak": peak_g,
        "cutoff_A": cutoff,
        "cutoff_block_sd_A": float(np.std(block_cutoffs, ddof=1)),
        "g_min": minimum_g,
    }


def li_cl_connectivity(result: dict, args: argparse.Namespace) -> None:
    trajectory = Trajectory(result["path"], "r")
    first = trajectory[0]
    symbols = np.asarray(first.get_chemical_symbols())
    li_indices = np.flatnonzero(symbols == "Li")
    cl_indices = np.flatnonzero(symbols == "Cl")
    selected = frame_indices(
        len(trajectory), args.discard_fraction, args.connectivity_frames
    )
    blocks = np.array_split(selected, args.blocks)
    counts_by_block = np.zeros((args.blocks, len(CATEGORY_LABELS)), dtype=np.int64)
    li_cl_bonds_by_block = np.zeros(args.blocks, dtype=np.int64)
    li_centers_by_block = np.zeros(args.blocks, dtype=np.int64)

    for block_id, block_indices in enumerate(blocks):
        for frame_index in block_indices:
            atoms = trajectory[int(frame_index)]
            lengths = ensure_orthorhombic(atoms)
            li_tree = cKDTree(
                np.mod(atoms.positions[li_indices], lengths), boxsize=lengths
            )
            connectivity = li_tree.query_ball_point(
                np.mod(atoms.positions[cl_indices], lengths),
                result["cutoff_A"],
                return_length=True,
            )
            categories = np.minimum(np.asarray(connectivity, dtype=np.int16), 3)
            counts_by_block[block_id] += np.bincount(categories, minlength=4)
            li_cl_bonds_by_block[block_id] += int(np.sum(connectivity))
            li_centers_by_block[block_id] += len(li_indices)

    block_totals = counts_by_block.sum(axis=1, keepdims=True)
    block_probability = counts_by_block / block_totals
    probability = counts_by_block.sum(axis=0) / counts_by_block.sum()
    li_cl_cn_blocks = li_cl_bonds_by_block / li_centers_by_block
    result["n_connectivity_frames"] = len(selected)
    result["cl_probability"] = probability
    result["cl_probability_sd"] = np.std(block_probability, axis=0, ddof=1)
    result["cl_probability_blocks"] = block_probability
    result["li_cl_coordination_number"] = float(
        li_cl_bonds_by_block.sum() / li_centers_by_block.sum()
    )
    result["li_cl_coordination_number_sd"] = float(
        np.std(li_cl_cn_blocks, ddof=1)
    )
    result["li_cl_coordination_number_blocks"] = li_cl_cn_blocks


def load_mg_results(coordination_csv: Path, summary_csv: Path) -> list[dict]:
    distributions = pd.read_csv(coordination_csv, dtype={"value": str})
    distributions = distributions[distributions["distribution"] == "n_Mg(Cl)"].copy()
    summary = pd.read_csv(summary_csv).set_index("system")
    results = []
    for system, group in distributions.groupby("system"):
        group = group.set_index("value").reindex(CATEGORY_LABELS)
        results.append(
            {
                "system": system,
                "x_nacl": float(group["x_NaCl"].iloc[0]),
                "cutoff_A": float(summary.loc[system, "cutoff_A"]),
                "cl_probability": group["probability"].to_numpy(float),
                "cl_probability_sd": group["block_sd"].to_numpy(float),
            }
        )
    return sorted(results, key=lambda result: result["x_nacl"])


def attach_random_reference(
    results: list[dict], mixture: str, baseline_csv: Path
) -> None:
    """Attach exact random-label P(n) values computed for the same trajectories."""
    table = pd.read_csv(baseline_csv, dtype={"n_cation_Cl": str})
    required = {"mixture", "system", "n_cation_Cl", "P_random"}
    if not required.issubset(table.columns):
        raise ValueError(f"{baseline_csv} must contain {sorted(required)}")
    table = table[table["mixture"] == mixture]
    for result in results:
        group = (
            table[table["system"] == result["system"]]
            .set_index("n_cation_Cl")
            .reindex(CATEGORY_LABELS)
        )
        if group["P_random"].isna().any():
            raise ValueError(
                f"Missing random reference for {mixture}/{result['system']}"
            )
        result["random_probability"] = group["P_random"].to_numpy(float)


def plot_panel(axis, results: list[dict], central_cation: str, title: str) -> None:
    x = np.asarray([result["x_nacl"] for result in results])
    markers = ("o", "s", "^", "D")
    for category_id, category in enumerate(CATEGORY_LABELS):
        values = [result["cl_probability"][category_id] for result in results]
        errors = [result["cl_probability_sd"][category_id] for result in results]
        random_values = [
            result["random_probability"][category_id] for result in results
        ]
        axis.plot(
            x,
            random_values,
            color=CONNECTIVITY_COLORS[category_id],
            linestyle=(0, (4.2, 2.2)),
            linewidth=1.35,
            alpha=0.95,
            zorder=1,
        )
        axis.errorbar(
            x,
            values,
            yerr=errors,
            color=CONNECTIVITY_COLORS[category_id],
            marker=markers[category_id],
            markersize=4.2,
            linewidth=1.25,
            capsize=2,
            zorder=2,
            label=rf"$n_{{\rm {central_cation}}}({{\rm Cl}})={category}$",
        )
    axis.set_title(title)
    axis.set_xlabel(r"$x_{\mathrm{NaCl}}$")
    axis.set_ylabel("Fraction of Cl")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            color="0.20",
            linestyle=(0, (4.2, 2.2)),
            linewidth=1.35,
            label="random mixing",
        )
    )
    labels.append("random mixing")
    axis.legend(handles, labels, fontsize=7.2, ncol=2, loc="upper center")


def plot_comparison(li_results: list[dict], mg_results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True, sharey=True)
    plot_panel(axes[0], li_results, "Li", r"LiCl--NaCl: Cl connectivity")
    plot_panel(axes[1], mg_results, "Mg", r"MgCl$_2$--NaCl: Cl connectivity")
    for panel_id, axis in enumerate(axes):
        axis.text(
            0.025,
            0.955,
            chr(ord("a") + panel_id),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "LiCl_NaCl_vs_MgCl2_NaCl_Cl_connectivity.png", dpi=450)
    fig.savefig(output_dir / "LiCl_NaCl_vs_MgCl2_NaCl_Cl_connectivity.pdf")
    plt.close(fig)


def plot_li_rdf_diagnostics(li_results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    targets = (0.05, 0.25, 0.50, 0.75, 0.95)
    selected = []
    for target in targets:
        selected.append(
            min(li_results, key=lambda result: abs(result["x_nacl"] - target))
        )
    cmap = mpl.colormaps["viridis"]
    for result in selected:
        color = cmap(result["x_nacl"])
        axes[0].plot(
            result["r"],
            result["rdf"],
            color=color,
            linewidth=1.3,
            label=rf"$x_{{\rm NaCl}}={result['x_nacl']:.2f}$",
        )
        axes[0].plot(
            result["cutoff_A"],
            np.interp(result["cutoff_A"], result["r"], result["rdf"]),
            marker="o",
            markersize=4,
            color=color,
            linestyle="none",
        )
    axes[0].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    axes[0].set_ylabel(r"$g_{\rm LiCl}(r)$")
    axes[0].set_xlim(1.5, 4.2)
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(fontsize=7)

    x = np.asarray([result["x_nacl"] for result in li_results])
    axes[1].plot(
        x,
        [result["r_peak_A"] for result in li_results],
        marker="o",
        markersize=4,
        linewidth=1.2,
        color=mpl.colormaps["tab10"](0),
        label="first-peak position",
    )
    axes[1].errorbar(
        x,
        [result["cutoff_A"] for result in li_results],
        yerr=[result["cutoff_block_sd_A"] for result in li_results],
        marker="s",
        markersize=4,
        linewidth=1.2,
        capsize=2,
        color=mpl.colormaps["tab10"](1),
        label="first-minimum cutoff",
    )
    axes[1].set_xlabel(r"$x_{\mathrm{NaCl}}$")
    axes[1].set_ylabel(r"Li--Cl distance ($\mathrm{\AA}$)")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].legend(fontsize=8)

    for panel_id, axis in enumerate(axes):
        axis.text(
            0.025,
            0.955,
            chr(ord("a") + panel_id),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "LiCl_NaCl_LiCl_RDF_cutoff_diagnostics.png", dpi=450)
    fig.savefig(output_dir / "LiCl_NaCl_LiCl_RDF_cutoff_diagnostics.pdf")
    plt.close(fig)


def write_outputs(
    li_results: list[dict],
    mg_results: list[dict],
    args: argparse.Namespace,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "LiCl_NaCl_LiCl_rdf_and_cutoffs.csv").open(
        "w", newline=""
    ) as handle:
        fields = [
            "system",
            "x_NaCl",
            "r_A",
            "g_LiCl",
            "g_LiCl_block_sd",
            "cutoff_A",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in li_results:
            for r, rdf, rdf_sd in zip(result["r"], result["rdf"], result["rdf_sd"]):
                writer.writerow(
                    {
                        "system": result["system"],
                        "x_NaCl": result["x_nacl"],
                        "r_A": r,
                        "g_LiCl": rdf,
                        "g_LiCl_block_sd": rdf_sd,
                        "cutoff_A": result["cutoff_A"],
                    }
                )

    with (args.output_dir / "LiCl_NaCl_LiCl_first_shell_metrics.csv").open(
        "w", newline=""
    ) as handle:
        fields = [
            "system",
            "x_NaCl",
            "N_Li",
            "N_Na",
            "N_Cl",
            "sampled_RDF_frames",
            "sampled_connectivity_frames",
            "r_peak_A",
            "g_peak",
            "r_min_cutoff_A",
            "r_min_cutoff_block_sd_A",
            "g_min",
            "Li_Cl_coordination_number",
            "Li_Cl_coordination_number_block_sd",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in li_results:
            counts = result["counts"]
            writer.writerow(
                {
                    "system": result["system"],
                    "x_NaCl": result["x_nacl"],
                    "N_Li": counts["Li"],
                    "N_Na": counts["Na"],
                    "N_Cl": counts["Cl"],
                    "sampled_RDF_frames": result["n_rdf_frames"],
                    "sampled_connectivity_frames": result[
                        "n_connectivity_frames"
                    ],
                    "r_peak_A": result["r_peak_A"],
                    "g_peak": result["g_peak"],
                    "r_min_cutoff_A": result["cutoff_A"],
                    "r_min_cutoff_block_sd_A": result["cutoff_block_sd_A"],
                    "g_min": result["g_min"],
                    "Li_Cl_coordination_number": result[
                        "li_cl_coordination_number"
                    ],
                    "Li_Cl_coordination_number_block_sd": result[
                        "li_cl_coordination_number_sd"
                    ],
                }
            )

    with (args.output_dir / "Li_vs_Mg_Cl_connectivity.csv").open(
        "w", newline=""
    ) as handle:
        fields = [
            "mixture",
            "central_cation",
            "system",
            "x_NaCl",
            "n_cation_Cl",
            "probability",
            "block_sd",
            "cutoff_A",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mixture, cation, results in (
            ("LiCl-NaCl", "Li", li_results),
            ("MgCl2-NaCl", "Mg", mg_results),
        ):
            for result in results:
                for category, probability, sd in zip(
                    CATEGORY_LABELS,
                    result["cl_probability"],
                    result["cl_probability_sd"],
                ):
                    writer.writerow(
                        {
                            "mixture": mixture,
                            "central_cation": cation,
                            "system": result["system"],
                            "x_NaCl": result["x_nacl"],
                            "n_cation_Cl": category,
                            "probability": probability,
                            "block_sd": sd,
                            "cutoff_A": result["cutoff_A"],
                        }
                    )

    metadata = {
        "li_input_dir": str(args.li_input_dir),
        "mg_coordination_source": str(args.mg_coordination_csv),
        "random_connectivity_source": str(args.random_baseline_csv),
        "composition": "x_NaCl = N_Na / (N_Na + N_Li or N_Mg)",
        "categories": {
            "0": "no Li/Mg within the corresponding first-shell cutoff",
            "1": "connected to one Li/Mg",
            "2": "shared by two Li/Mg",
            "3+": "shared by three or more Li/Mg",
        },
        "li_cutoff": "composition-specific first minimum of the Li-Cl RDF",
        "mg_cutoff": "composition-specific first minimum of the Mg-Cl RDF from the prior analysis",
        "discard_fraction": args.discard_fraction,
        "li_rdf_frames": args.rdf_frames,
        "li_connectivity_frames": args.connectivity_frames,
        "time_blocks": args.blocks,
        "uncertainty": "sample standard deviation across contiguous trajectory blocks",
        "random_reference": (
            "exact random relabeling at fixed instantaneous positions, total "
            "cation--Cl neighbor graph, and cation counts; plotted as dashed curves"
        ),
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    paths = sorted(args.li_input_dir.glob("li*/md.traj"), key=numeric_system_key)
    if not paths:
        raise FileNotFoundError(f"No li*/md.traj trajectories found in {args.li_input_dir}")
    edges = np.arange(0.0, args.r_max + args.bin_width * 0.5, args.bin_width)
    li_results = []
    for path in paths:
        print(f"Li-Cl RDF and cutoff: {path.parent.name}", flush=True)
        result = li_rdf_and_cutoff(path, args, edges)
        print(
            f"  x_NaCl={result['x_nacl']:.3f}, cutoff={result['cutoff_A']:.3f} A",
            flush=True,
        )
        li_results.append(result)
    li_results.sort(key=lambda result: result["x_nacl"])

    for result in li_results:
        print(f"Cl connectivity: {result['system']}", flush=True)
        li_cl_connectivity(result, args)

    mg_results = load_mg_results(args.mg_coordination_csv, args.mg_summary_csv)
    attach_random_reference(li_results, "LiCl-NaCl", args.random_baseline_csv)
    attach_random_reference(mg_results, "MgCl2-NaCl", args.random_baseline_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(li_results, mg_results, args)
    plot_comparison(li_results, mg_results, args.output_dir)
    plot_li_rdf_diagnostics(li_results, args.output_dir)
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
