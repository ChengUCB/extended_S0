#!/usr/bin/env python3
"""Local coordination, Mg-Cl network, and Mg/Na compensation diagnostics.

The Mg-Cl neighbor cutoff is composition-specific and taken from the first
minimum of the corresponding Mg-Cl RDF.  Uncertainties are sample standard
deviations across contiguous trajectory blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from analyze_mgcl2_nacl_rdfs import configure_matplotlib, ensure_orthorhombic, frame_indices


CN_MAX_EXPLICIT = 8
CL_CONNECTIVITY_LABELS = ("0", "1", "2", "3+")
MOTIF_CNS = (3, 4, 5, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("MD_inputs/SelectedMgCl2-NaCl_MD"),
    )
    parser.add_argument(
        "--rdf-metrics",
        type=Path,
        default=Path("outputs/SelectedMgCl2-NaCl_MD_RDF/mgcl_first_shell_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/SelectedMgCl2-NaCl_MD_structure"),
    )
    parser.add_argument("--discard-fraction", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--radial-max", type=float, default=12.0)
    parser.add_argument("--radial-bin-width", type=float, default=0.10)
    parser.add_argument("--angle-bin-width", type=float, default=2.0)
    parser.add_argument("--q-bin-width", type=float, default=0.02)
    return parser.parse_args()


class PeriodicUnionFind:
    """Union-find with integer image potentials for periodic wrapping tests."""

    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)
        self.weight = np.zeros((size, 3), dtype=np.int64)
        self.wrap = np.zeros((size, 3), dtype=bool)

    def find(self, node: int) -> int:
        parent = int(self.parent[node])
        if parent != node:
            root = self.find(parent)
            self.weight[node] += self.weight[parent]
            self.parent[node] = root
        return int(self.parent[node])

    def union(self, left: int, right: int, shift_right_minus_left: np.ndarray) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        weight_left = self.weight[left].copy()
        weight_right = self.weight[right].copy()
        shift = np.asarray(shift_right_minus_left, dtype=np.int64)

        if root_left == root_right:
            residual = (weight_right - weight_left) - shift
            self.wrap[root_left] |= residual != 0
            return

        root_right_minus_left = shift + weight_left - weight_right
        if self.rank[root_left] >= self.rank[root_right]:
            self.parent[root_right] = root_left
            self.weight[root_right] = root_right_minus_left
            self.wrap[root_left] |= self.wrap[root_right]
            if self.rank[root_left] == self.rank[root_right]:
                self.rank[root_left] += 1
        else:
            self.parent[root_left] = root_right
            self.weight[root_left] = -root_right_minus_left
            self.wrap[root_right] |= self.wrap[root_left]

    def component_summary(self) -> tuple[list[int], bool]:
        sizes: Counter[int] = Counter()
        for node in range(len(self.parent)):
            sizes[self.find(node)] += 1
        percolates = False
        for root in sizes:
            true_root = self.find(root)
            percolates |= bool(np.any(self.wrap[true_root]))
        return list(sizes.values()), percolates


def composition_cutoffs(path: Path) -> dict[str, float]:
    table = pd.read_csv(path)
    required = {"system", "r_min_A"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return {str(row.system): float(row.r_min_A) for row in table.itertuples()}


def block_assignment(n_selected: int, n_blocks: int) -> tuple[list[np.ndarray], np.ndarray]:
    blocks = np.array_split(np.arange(n_selected, dtype=int), n_blocks)
    assignment = np.empty(n_selected, dtype=int)
    for block_id, positions in enumerate(blocks):
        assignment[positions] = block_id
    return blocks, assignment


def minimum_image(vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return vectors - lengths * np.rint(vectors / lengths)


def angles_from_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 2:
        return np.empty(0, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-12
    vectors = vectors[valid]
    norms = norms[valid]
    if len(vectors) < 2:
        return np.empty(0, dtype=float)
    unit = vectors / norms[:, None]
    upper = np.triu_indices(len(unit), 1)
    cosines = np.clip((unit @ unit.T)[upper], -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


def tetrahedral_order(vectors: np.ndarray) -> float:
    """Standard q using the four nearest Cl neighbors."""
    if len(vectors) < 4:
        return float("nan")
    distances = np.linalg.norm(vectors, axis=1)
    nearest = vectors[np.argsort(distances)[:4]]
    unit = nearest / np.linalg.norm(nearest, axis=1)[:, None]
    upper = np.triu_indices(4, 1)
    cosines = np.clip((unit @ unit.T)[upper], -1.0, 1.0)
    return float(1.0 - (3.0 / 8.0) * np.sum((cosines + 1.0 / 3.0) ** 2))


def histogram_add(target: np.ndarray, values: np.ndarray, edges: np.ndarray) -> None:
    if len(values):
        target += np.histogram(values, bins=edges)[0]


def probability_from_blocks(hist_blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    totals = hist_blocks.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        block_probability = hist_blocks / totals
    total_hist = hist_blocks.sum(axis=0)
    probability = total_hist / total_hist.sum()
    sd = np.nanstd(block_probability, axis=0, ddof=1)
    return probability, sd, block_probability


def density_from_blocks(
    hist_blocks: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probability, sd_probability, _ = probability_from_blocks(hist_blocks)
    widths = np.diff(edges)
    return probability / widths, sd_probability / widths


def scalar_summary(values_by_block: list[list[float]]) -> tuple[float, float, np.ndarray]:
    block_means = np.asarray([
        np.nanmean(values) if np.any(np.isfinite(values)) else np.nan
        for values in values_by_block
    ], dtype=float)
    all_values = np.concatenate(
        [np.asarray(values, dtype=float) for values in values_by_block if len(values)]
    )
    finite_all = all_values[np.isfinite(all_values)]
    mean = float(np.mean(finite_all)) if len(finite_all) else np.nan
    finite_blocks = block_means[np.isfinite(block_means)]
    sd = float(np.std(finite_blocks, ddof=1)) if len(finite_blocks) >= 2 else np.nan
    return mean, sd, block_means


def analyze_composition(
    trajectory_path: Path,
    cutoff: float,
    args: argparse.Namespace,
    radial_edges: np.ndarray,
    angle_edges: np.ndarray,
    q_edges: np.ndarray,
) -> dict:
    trajectory = Trajectory(str(trajectory_path), "r")
    first = trajectory[0]
    symbols = np.asarray(first.get_chemical_symbols())
    counts = Counter(symbols.tolist())
    mg_global = np.flatnonzero(symbols == "Mg")
    na_global = np.flatnonzero(symbols == "Na")
    cl_global = np.flatnonzero(symbols == "Cl")
    n_mg, n_na, n_cl = len(mg_global), len(na_global), len(cl_global)
    x_nacl = n_na / (n_na + n_mg)

    selected = frame_indices(len(trajectory), args.discard_fraction, args.max_frames)
    _, assignment = block_assignment(len(selected), args.blocks)
    n_angle_bins = len(angle_edges) - 1
    n_q_bins = len(q_edges) - 1
    n_radial_bins = len(radial_edges) - 1

    cn_hist_blocks = np.zeros((args.blocks, CN_MAX_EXPLICIT + 2), dtype=np.int64)
    cn_sum_blocks = np.zeros(args.blocks, dtype=float)
    cn_count_blocks = np.zeros(args.blocks, dtype=np.int64)
    cl_connectivity_blocks = np.zeros((args.blocks, 4), dtype=np.int64)
    clmgcl_angle_blocks = np.zeros((args.blocks, n_angle_bins), dtype=np.int64)
    mgclmg_angle_blocks = np.zeros((args.blocks, n_angle_bins), dtype=np.int64)
    q_blocks = np.zeros((args.blocks, n_q_bins), dtype=np.int64)
    q_cn4_blocks = np.zeros((args.blocks, n_q_bins), dtype=np.int64)
    q_sum_blocks = np.zeros(args.blocks, dtype=float)
    q_count_blocks = np.zeros(args.blocks, dtype=np.int64)
    q_cn4_sum_blocks = np.zeros(args.blocks, dtype=float)
    q_cn4_count_blocks = np.zeros(args.blocks, dtype=np.int64)

    mgna_hist_blocks = np.zeros((args.blocks, n_radial_bins), dtype=np.int64)
    mgmg_hist_blocks = np.zeros((args.blocks, n_radial_bins), dtype=np.int64)
    mgna_norm_blocks = np.zeros(args.blocks, dtype=float)
    motif_hist_blocks = np.zeros((args.blocks, len(MOTIF_CNS), n_radial_bins), dtype=np.int64)
    motif_norm_blocks = np.zeros((args.blocks, len(MOTIF_CNS)), dtype=float)
    frames_per_block = np.zeros(args.blocks, dtype=np.int64)

    scalar_frames = {
        "corner_fraction": [[] for _ in range(args.blocks)],
        "edge_or_more_fraction": [[] for _ in range(args.blocks)],
        "face_or_more_fraction": [[] for _ in range(args.blocks)],
        "largest_cluster_fraction": [[] for _ in range(args.blocks)],
        "percolates": [[] for _ in range(args.blocks)],
    }
    cluster_counter: Counter[int] = Counter()
    sharing_counter: Counter[int] = Counter()

    for selected_position, frame_index in enumerate(selected):
        block_id = int(assignment[selected_position])
        atoms = trajectory[int(frame_index)]
        lengths = ensure_orthorhombic(atoms)
        positions = np.mod(atoms.get_positions(), lengths)
        mg_positions = positions[mg_global]
        na_positions = positions[na_global]
        cl_positions = positions[cl_global]
        volume = float(atoms.get_volume())
        frames_per_block[block_id] += 1

        tree_mg = cKDTree(mg_positions, boxsize=lengths)
        tree_cl = cKDTree(cl_positions, boxsize=lengths)
        mg_to_cl = tree_mg.query_ball_tree(tree_cl, cutoff)
        cn = np.fromiter((len(neighbors) for neighbors in mg_to_cl), dtype=np.int16, count=n_mg)
        cn_categories = np.minimum(cn, CN_MAX_EXPLICIT + 1)
        cn_hist_blocks[block_id] += np.bincount(
            cn_categories,
            minlength=CN_MAX_EXPLICIT + 2,
        )
        cn_sum_blocks[block_id] += float(cn.sum())
        cn_count_blocks[block_id] += n_mg

        cl_to_mg: list[list[int]] = [[] for _ in range(n_cl)]
        q_values = []
        q_cn4_values = []
        for mg_index, cl_neighbors in enumerate(mg_to_cl):
            if not cl_neighbors:
                continue
            for cl_index in cl_neighbors:
                cl_to_mg[cl_index].append(mg_index)
            vectors = minimum_image(cl_positions[cl_neighbors] - mg_positions[mg_index], lengths)
            histogram_add(clmgcl_angle_blocks[block_id], angles_from_vectors(vectors), angle_edges)
            if len(vectors) >= 4:
                q_value = tetrahedral_order(vectors)
                q_values.append(q_value)
                if len(vectors) == 4:
                    q_cn4_values.append(q_value)

        if q_values:
            q_array = np.asarray(q_values)
            histogram_add(q_blocks[block_id], q_array, q_edges)
            q_sum_blocks[block_id] += float(q_array.sum())
            q_count_blocks[block_id] += len(q_array)
        if q_cn4_values:
            q_cn4_array = np.asarray(q_cn4_values)
            histogram_add(q_cn4_blocks[block_id], q_cn4_array, q_edges)
            q_cn4_sum_blocks[block_id] += float(q_cn4_array.sum())
            q_cn4_count_blocks[block_id] += len(q_cn4_array)

        shared_cl_per_mg_pair: defaultdict[tuple[int, int], int] = defaultdict(int)
        cl_connectivity = np.fromiter(
            (len(neighbors) for neighbors in cl_to_mg),
            dtype=np.int16,
            count=n_cl,
        )
        cl_categories = np.minimum(cl_connectivity, 3)
        cl_connectivity_blocks[block_id] += np.bincount(cl_categories, minlength=4)
        for cl_index, mg_neighbors in enumerate(cl_to_mg):
            if len(mg_neighbors) >= 2:
                vectors = minimum_image(mg_positions[mg_neighbors] - cl_positions[cl_index], lengths)
                histogram_add(mgclmg_angle_blocks[block_id], angles_from_vectors(vectors), angle_edges)
                for left, right in combinations(sorted(mg_neighbors), 2):
                    shared_cl_per_mg_pair[(left, right)] += 1

        n_connected = len(shared_cl_per_mg_pair)
        if n_connected:
            sharing_values = np.fromiter(shared_cl_per_mg_pair.values(), dtype=np.int16)
            corner = float(np.count_nonzero(sharing_values == 1) / n_connected)
            edge_or_more = float(np.count_nonzero(sharing_values >= 2) / n_connected)
            face_or_more = float(np.count_nonzero(sharing_values >= 3) / n_connected)
            sharing_counter.update(sharing_values.tolist())
        else:


            corner = edge_or_more = face_or_more = np.nan
        scalar_frames["corner_fraction"][block_id].append(corner)
        scalar_frames["edge_or_more_fraction"][block_id].append(edge_or_more)
        scalar_frames["face_or_more_fraction"][block_id].append(face_or_more)

        union_find = PeriodicUnionFind(n_mg)
        for left, right in shared_cl_per_mg_pair:
            displacement = mg_positions[right] - mg_positions[left]
            shift = -np.rint(displacement / lengths).astype(np.int64)
            union_find.union(left, right, shift)
        cluster_sizes, percolates = union_find.component_summary()
        cluster_counter.update(cluster_sizes)
        largest_fraction = max(cluster_sizes) / n_mg
        scalar_frames["largest_cluster_fraction"][block_id].append(largest_fraction)
        scalar_frames["percolates"][block_id].append(float(percolates))

        cation_positions = np.vstack((mg_positions, na_positions))
        cation_pairs = cKDTree(cation_positions, boxsize=lengths).query_pairs(
            args.radial_max,
            output_type="ndarray",
        )
        if len(cation_pairs):
            displacement = cation_positions[cation_pairs[:, 1]] - cation_positions[cation_pairs[:, 0]]
            displacement = minimum_image(displacement, lengths)
            distances = np.linalg.norm(displacement, axis=1)
            bins = np.searchsorted(radial_edges, distances, side="right") - 1
            in_range = (bins >= 0) & (bins < n_radial_bins)
            left = cation_pairs[:, 0]
            right = cation_pairs[:, 1]
            mgmg_mask = in_range & (right < n_mg)
            mgna_mask = in_range & (left < n_mg) & (right >= n_mg)
            if np.any(mgmg_mask):
                mgmg_hist_blocks[block_id] += np.bincount(
                    bins[mgmg_mask],
                    minlength=n_radial_bins,
                )
            if np.any(mgna_mask):
                mgna_hist_blocks[block_id] += np.bincount(
                    bins[mgna_mask],
                    minlength=n_radial_bins,
                )
                central_mg = left[mgna_mask]
                central_cn = cn[central_mg]
                pair_bins = bins[mgna_mask]
                for motif_id, motif_cn in enumerate(MOTIF_CNS):
                    motif_mask = central_cn == motif_cn
                    if np.any(motif_mask):
                        motif_hist_blocks[block_id, motif_id] += np.bincount(
                            pair_bins[motif_mask],
                            minlength=n_radial_bins,
                        )

        mgna_norm_blocks[block_id] += n_mg * n_na / volume
        for motif_id, motif_cn in enumerate(MOTIF_CNS):
            motif_norm_blocks[block_id, motif_id] += np.count_nonzero(cn == motif_cn) * n_na / volume

    cn_total = cn_hist_blocks.sum(axis=0)
    cn_probability, cn_probability_sd, cn_probability_blocks = probability_from_blocks(cn_hist_blocks)
    mean_cn_blocks = cn_sum_blocks / cn_count_blocks
    mean_cn = float(cn_sum_blocks.sum() / cn_count_blocks.sum())
    mean_cn_sd = float(mean_cn_blocks.std(ddof=1))

    cl_probability, cl_probability_sd, cl_probability_blocks = probability_from_blocks(
        cl_connectivity_blocks
    )
    terminal_fraction = float(cl_probability[1])
    bridge_fraction = float(cl_probability[2:].sum())
    bound_fraction = float(1.0 - cl_probability[0])
    terminal_among_bound = terminal_fraction / bound_fraction if bound_fraction else np.nan
    bridge_among_bound = bridge_fraction / bound_fraction if bound_fraction else np.nan
    terminal_bound_blocks = cl_probability_blocks[:, 1] / (1.0 - cl_probability_blocks[:, 0])
    bridge_bound_blocks = cl_probability_blocks[:, 2:].sum(axis=1) / (
        1.0 - cl_probability_blocks[:, 0]
    )

    clmgcl_density, clmgcl_density_sd = density_from_blocks(clmgcl_angle_blocks, angle_edges)
    mgclmg_density, mgclmg_density_sd = density_from_blocks(mgclmg_angle_blocks, angle_edges)
    q_density, q_density_sd = density_from_blocks(q_blocks, q_edges)
    q_cn4_density, q_cn4_density_sd = density_from_blocks(q_cn4_blocks, q_edges)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])

    q_mean_blocks = q_sum_blocks / q_count_blocks
    q_cn4_mean_blocks = q_cn4_sum_blocks / q_cn4_count_blocks
    q_mean = float(q_sum_blocks.sum() / q_count_blocks.sum())
    q_mean_sd = float(np.nanstd(q_mean_blocks, ddof=1))
    q_cn4_mean = float(q_cn4_sum_blocks.sum() / q_cn4_count_blocks.sum())
    q_cn4_mean_sd = float(np.nanstd(q_cn4_mean_blocks, ddof=1))

    mgna_total = mgna_hist_blocks.sum(axis=0).astype(float)
    mgmg_total = mgmg_hist_blocks.sum(axis=0).astype(float)
    conditional_denominator = mgna_total + 2.0 * mgmg_total
    with np.errstate(invalid="ignore", divide="ignore"):
        p_na_given_mg = mgna_total / conditional_denominator
        alpha = 1.0 - p_na_given_mg / x_nacl
    alpha_blocks = np.full_like(mgna_hist_blocks, np.nan, dtype=float)
    for block_id in range(args.blocks):
        denominator = mgna_hist_blocks[block_id] + 2.0 * mgmg_hist_blocks[block_id]
        valid = denominator > 0
        alpha_blocks[block_id, valid] = (
            1.0 - (mgna_hist_blocks[block_id, valid] / denominator[valid]) / x_nacl
        )
    alpha_sd = np.nanstd(alpha_blocks, axis=0, ddof=1)

    shell_volumes = (4.0 * np.pi / 3.0) * (
        radial_edges[1:] ** 3 - radial_edges[:-1] ** 3
    )
    radial_widths = np.diff(radial_edges)
    n_sampled = len(selected)
    mgna_dndr = mgna_total / (n_sampled * n_mg * radial_widths)
    mgna_conditional_density = mgna_total / (n_sampled * n_mg * shell_volumes)
    mgna_g = mgna_total / (mgna_norm_blocks.sum() * shell_volumes)
    mgna_g_blocks = mgna_hist_blocks / (
        mgna_norm_blocks[:, None] * shell_volumes[None, :]
    )
    mgna_g_sd = np.nanstd(mgna_g_blocks, axis=0, ddof=1)

    motif_g = np.full((len(MOTIF_CNS), n_radial_bins), np.nan)
    motif_g_sd = np.full_like(motif_g, np.nan)
    for motif_id in range(len(MOTIF_CNS)):
        norm_total = motif_norm_blocks[:, motif_id].sum()
        if norm_total > 0:
            motif_g[motif_id] = motif_hist_blocks[:, motif_id].sum(axis=0) / (
                norm_total * shell_volumes
            )
            block_curves = np.full((args.blocks, n_radial_bins), np.nan)
            valid_blocks = motif_norm_blocks[:, motif_id] > 0
            block_curves[valid_blocks] = motif_hist_blocks[valid_blocks, motif_id] / (
                motif_norm_blocks[valid_blocks, motif_id, None] * shell_volumes[None, :]
            )
            motif_g_sd[motif_id] = np.nanstd(block_curves, axis=0, ddof=1)

    scalar_results = {}
    for name, values_by_block in scalar_frames.items():
        mean, sd, block_means = scalar_summary(values_by_block)
        scalar_results[name] = mean
        scalar_results[f"{name}_block_sd"] = sd
        scalar_results[f"{name}_block_means"] = block_means

    total_clusters = sum(cluster_counter.values())
    cluster_distribution = []
    for size, count in sorted(cluster_counter.items()):
        cluster_distribution.append(
            {
                "size": size,
                "size_fraction": size / n_mg,
                "mean_clusters_per_frame": count / n_sampled,
                "cluster_probability": count / total_clusters,
                "mg_weighted_probability": size * count / (n_sampled * n_mg),
            }
        )

    corner_mean, corner_sd, _ = scalar_summary(scalar_frames["corner_fraction"])
    edge_mean, edge_sd, _ = scalar_summary(scalar_frames["edge_or_more_fraction"])
    face_mean, face_sd, _ = scalar_summary(scalar_frames["face_or_more_fraction"])
    largest_mean, largest_sd, _ = scalar_summary(scalar_frames["largest_cluster_fraction"])
    percolation_probability, percolation_sd, _ = scalar_summary(scalar_frames["percolates"])

    return {
        "system": trajectory_path.parent.name,
        "path": str(trajectory_path),
        "counts": dict(counts),
        "x_nacl": x_nacl,
        "x_mgcl2": 1.0 - x_nacl,
        "cutoff_A": cutoff,
        "n_total_frames": len(trajectory),
        "n_sampled_frames": n_sampled,
        "mean_cn": mean_cn,
        "mean_cn_block_sd": mean_cn_sd,
        "cn_probability": cn_probability,
        "cn_probability_sd": cn_probability_sd,
        "cl_probability": cl_probability,
        "cl_probability_sd": cl_probability_sd,
        "terminal_fraction": terminal_fraction,
        "bridge_fraction": bridge_fraction,
        "terminal_among_bound": terminal_among_bound,
        "terminal_among_bound_block_sd": float(np.nanstd(terminal_bound_blocks, ddof=1)),
        "bridge_among_bound": bridge_among_bound,
        "bridge_among_bound_block_sd": float(np.nanstd(bridge_bound_blocks, ddof=1)),
        "angle_centers": angle_centers,
        "clmgcl_density": clmgcl_density,
        "clmgcl_density_sd": clmgcl_density_sd,
        "mgclmg_density": mgclmg_density,
        "mgclmg_density_sd": mgclmg_density_sd,
        "q_centers": q_centers,
        "q_density": q_density,
        "q_density_sd": q_density_sd,
        "q_cn4_density": q_cn4_density,
        "q_cn4_density_sd": q_cn4_density_sd,
        "q_mean": q_mean,
        "q_mean_block_sd": q_mean_sd,
        "q_cn4_mean": q_cn4_mean,
        "q_cn4_mean_block_sd": q_cn4_mean_sd,
        "corner_fraction": corner_mean,
        "corner_fraction_block_sd": corner_sd,
        "edge_or_more_fraction": edge_mean,
        "edge_or_more_fraction_block_sd": edge_sd,
        "face_or_more_fraction": face_mean,
        "face_or_more_fraction_block_sd": face_sd,
        "largest_cluster_fraction": largest_mean,
        "largest_cluster_fraction_block_sd": largest_sd,
        "percolation_probability": percolation_probability,
        "percolation_probability_block_sd": percolation_sd,
        "sharing_counter": dict(sharing_counter),
        "cluster_distribution": cluster_distribution,
        "radial_centers": radial_centers,
        "p_na_given_mg": p_na_given_mg,
        "alpha": alpha,
        "alpha_sd": alpha_sd,
        "mgna_dndr": mgna_dndr,
        "mgna_conditional_density": mgna_conditional_density,
        "mgna_g": mgna_g,
        "mgna_g_sd": mgna_g_sd,
        "motif_g": motif_g,
        "motif_g_sd": motif_g_sd,
    }


def selected_results(results: list[dict]) -> list[dict]:
    targets = (0.02, 0.33, 0.59, 0.70, 0.89)
    chosen = []
    used = set()
    for target in targets:
        index = int(np.argmin([abs(result["x_nacl"] - target) for result in results]))
        if index not in used:
            chosen.append(results[index])
            used.add(index)
    return chosen


def selected_compensation_results(results: list[dict]) -> list[dict]:
    """Avoid dilute endpoints where normalization by x_NaCl amplifies noise."""
    targets = (0.18, 0.33, 0.47, 0.59, 0.70, 0.89)
    chosen = []
    used = set()
    for target in targets:
        index = int(np.argmin([abs(result["x_nacl"] - target) for result in results]))
        if index not in used:
            chosen.append(results[index])
            used.add(index)
    return chosen


def add_panel_labels(axes) -> None:
    for panel_id, axis in enumerate(np.asarray(axes).flat):
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


def plot_local_coordination(results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8))
    x = np.asarray([result["x_nacl"] for result in results])
    cn_colors = mpl.colormaps["tab10"](np.arange(4))
    for offset, cn in enumerate(MOTIF_CNS):
        values = [result["cn_probability"][cn] for result in results]
        errors = [result["cn_probability_sd"][cn] for result in results]
        axes[0, 0].errorbar(
            x,
            values,
            yerr=errors,
            marker=("o", "s", "^", "D")[offset],
            markersize=4,
            linewidth=1.2,
            capsize=2,
            color=cn_colors[offset],
            label=f"CN = {cn}",
        )
    axes[0, 0].set_xlabel(r"$x_{\mathrm{NaCl}}$")
    axes[0, 0].set_ylabel("Probability")
    axes[0, 0].set_xlim(-0.02, 1.02)
    axes[0, 0].set_ylim(bottom=0.0)
    axes[0, 0].legend(ncol=2, fontsize=8)

    selected = selected_results(results)
    norm = Normalize(0.0, 1.0)
    cmap = mpl.colormaps["viridis"]
    for result in selected:
        color = cmap(norm(result["x_nacl"]))
        label = rf"$x_{{\rm NaCl}}={result['x_nacl']:.2f}$"
        axes[0, 1].plot(
            result["angle_centers"],
            result["clmgcl_density"],
            color=color,
            linewidth=1.35,
            label=label,
        )
        axes[1, 0].plot(
            result["q_centers"],
            result["q_density"],
            color=color,
            linewidth=1.35,
            label=label,
        )
    axes[0, 1].axvline(109.47, color="0.55", linewidth=0.8, linestyle=(0, (2, 2)))
    axes[0, 1].set_xlabel(r"Cl--Mg--Cl angle ($^\circ$)")
    axes[0, 1].set_ylabel("Probability density")
    axes[0, 1].set_xlim(45, 180)
    axes[0, 1].set_ylim(bottom=0.0)
    axes[0, 1].legend(fontsize=7, ncol=1)
    axes[1, 0].set_xlabel(r"Tetrahedral order $q_4$")
    axes[1, 0].set_ylabel("Probability density")
    axes[1, 0].set_xlim(-0.6, 1.0)
    axes[1, 0].set_ylim(bottom=0.0)

    q_all = [result["q_mean"] for result in results]
    q_all_err = [result["q_mean_block_sd"] for result in results]
    q_cn4 = [result["q_cn4_mean"] for result in results]
    q_cn4_err = [result["q_cn4_mean_block_sd"] for result in results]
    axes[1, 1].errorbar(
        x,
        q_all,
        yerr=q_all_err,
        color=mpl.colormaps["tab10"](0),
        marker="o",
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label=r"four nearest Cl, CN$\geq4$",
    )
    axes[1, 1].errorbar(
        x,
        q_cn4,
        yerr=q_cn4_err,
        color=mpl.colormaps["tab10"](1),
        marker="s",
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label="exact CN = 4",
    )
    axes[1, 1].set_xlabel(r"$x_{\mathrm{NaCl}}$")
    axes[1, 1].set_ylabel(r"Mean tetrahedral order $\langle q_4\rangle$")
    axes[1, 1].set_xlim(-0.02, 1.02)
    axes[1, 1].legend(fontsize=8)

    add_panel_labels(axes)
    fig.tight_layout()
    fig.savefig(output_dir / "local_coordination_diagnostics.png", dpi=450)
    fig.savefig(output_dir / "local_coordination_diagnostics.pdf")
    plt.close(fig)


def plot_network_topology(results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8))
    x = np.asarray([result["x_nacl"] for result in results])
    line_colors = mpl.colormaps["tab10"](np.arange(4))
    for category, label in enumerate(CL_CONNECTIVITY_LABELS):
        values = [result["cl_probability"][category] for result in results]
        errors = [result["cl_probability_sd"][category] for result in results]
        axes[0, 0].errorbar(
            x,
            values,
            yerr=errors,
            color=line_colors[category],
            marker=("o", "s", "^", "D")[category],
            markersize=4,
            linewidth=1.2,
            capsize=2,
            label=rf"$n_{{\rm Mg}}({{\rm Cl}})={label}$",
        )
    axes[0, 0].set_xlabel(r"$x_{\mathrm{NaCl}}$")
    axes[0, 0].set_ylabel("Fraction of Cl")
    axes[0, 0].set_xlim(-0.02, 1.02)
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].legend(fontsize=7, ncol=2)

    selected = selected_results(results)
    norm = Normalize(0.0, 1.0)
    cmap = mpl.colormaps["viridis"]
    for result in selected:
        axes[0, 1].plot(
            result["angle_centers"],
            result["mgclmg_density"],
            color=cmap(norm(result["x_nacl"])),
            linewidth=1.35,
            label=rf"$x_{{\rm NaCl}}={result['x_nacl']:.2f}$",
        )
    axes[0, 1].set_xlabel(r"Mg--Cl--Mg angle ($^\circ$)")
    axes[0, 1].set_ylabel("Probability density")
    axes[0, 1].set_xlim(45, 180)
    axes[0, 1].set_ylim(bottom=0.0)
    axes[0, 1].legend(fontsize=7)

    sharing_specs = [
        ("corner_fraction", "corner: 1 shared Cl", "o"),
        ("edge_or_more_fraction", r"edge+: $\geq2$ shared Cl", "s"),
        ("face_or_more_fraction", r"face+: $\geq3$ shared Cl", "^"),
    ]
    for series_id, (key, label, marker) in enumerate(sharing_specs):
        axes[1, 0].errorbar(
            x,
            [result[key] for result in results],
            yerr=[result[f"{key}_block_sd"] for result in results],
            color=mpl.colormaps["tab10"](series_id),
            marker=marker,
            markersize=4,
            linewidth=1.2,
            capsize=2,
            label=label,
        )
    axes[1, 0].set_xlabel(r"$x_{\mathrm{NaCl}}$")
    axes[1, 0].set_ylabel("Fraction of connected Mg pairs")
    axes[1, 0].set_xlim(-0.02, 1.02)
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].errorbar(
        x,
        [result["largest_cluster_fraction"] for result in results],
        yerr=[result["largest_cluster_fraction_block_sd"] for result in results],
        color=mpl.colormaps["tab10"](0),
        marker="o",
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label=r"largest cluster / $N_{\rm Mg}$",
    )
    axes[1, 1].errorbar(
        x,
        [result["percolation_probability"] for result in results],
        yerr=[result["percolation_probability_block_sd"] for result in results],
        color=mpl.colormaps["tab10"](3),
        marker="s",
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label="periodic wrapping probability",
    )
    axes[1, 1].set_xlabel(r"$x_{\mathrm{NaCl}}$")
    axes[1, 1].set_ylabel("Network connectivity")
    axes[1, 1].set_xlim(-0.02, 1.02)
    axes[1, 1].set_ylim(-0.02, 1.05)
    axes[1, 1].legend(fontsize=7)

    add_panel_labels(axes)
    fig.tight_layout()
    fig.savefig(output_dir / "mgcl_network_topology.png", dpi=450)
    fig.savefig(output_dir / "mgcl_network_topology.pdf")
    plt.close(fig)


def plot_cluster_distributions(results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axis = plt.subplots(figsize=(5.2, 3.8))
    norm = Normalize(0.0, 1.0)
    cmap = mpl.colormaps["viridis"]
    for result in selected_results(results):
        distribution = result["cluster_distribution"]
        x = np.asarray([entry["size_fraction"] for entry in distribution])
        y = np.asarray([entry["mg_weighted_probability"] for entry in distribution])
        valid = (x > 0) & (y > 0)
        x = x[valid]
        y = y[valid]
        if not len(x):
            continue
        lower = max(float(x.min()) * 0.95, 1.0e-5)
        bins = np.geomspace(lower, 1.0001, 32)
        probability, _ = np.histogram(x, bins=bins, weights=y)
        centers = np.sqrt(bins[:-1] * bins[1:])
        nonzero = probability > 0
        axis.plot(
            centers[nonzero],
            probability[nonzero],
            marker="o",
            markersize=3.0,
            linestyle="none",
            color=cmap(norm(result["x_nacl"])),
            label=rf"$x_{{\rm NaCl}}={result['x_nacl']:.2f}$",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Cluster size / $N_{\rm Mg}$")
    axis.set_ylabel("Mg-weighted probability")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "mgcl_cluster_size_distribution.png", dpi=450)
    fig.savefig(output_dir / "mgcl_cluster_size_distribution.pdf")
    plt.close(fig)


def smooth_valid(values: np.ndarray, window: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = values.copy()
    valid = np.isfinite(values)
    if np.count_nonzero(valid) >= window:
        indices = np.flatnonzero(valid)
        contiguous = np.interp(np.arange(len(values)), indices, values[valid])
        result = savgol_filter(contiguous, window, 2, mode="interp")
        result[~valid] = np.nan
    return result


def composition_edges(x: np.ndarray) -> np.ndarray:
    midpoints = 0.5 * (x[:-1] + x[1:])
    left = max(0.0, x[0] - (midpoints[0] - x[0]))
    right = min(1.0, x[-1] + (x[-1] - midpoints[-1]))
    return np.concatenate(([left], midpoints, [right]))


def plot_mgna_compensation(results: list[dict], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    x = np.asarray([result["x_nacl"] for result in results])
    r = results[0]["radial_centers"]
    radial_mask = (r >= 3.0) & (r <= 12.0)
    alpha_matrix = np.vstack([smooth_valid(result["alpha"])[radial_mask] for result in results])
    finite_alpha = np.abs(alpha_matrix[np.isfinite(alpha_matrix)])
    limit = float(np.quantile(finite_alpha, 0.96)) if len(finite_alpha) else 1.0
    limit = max(0.2, limit)
    mesh = axes[0, 0].pcolormesh(
        np.concatenate((r[radial_mask] - 0.05, [r[radial_mask][-1] + 0.05])),
        composition_edges(x),
        alpha_matrix,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        shading="flat",
    )
    axes[0, 0].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    axes[0, 0].set_ylabel(r"$x_{\mathrm{NaCl}}$")
    cbar = fig.colorbar(mesh, ax=axes[0, 0], pad=0.02)
    cbar.set_label(r"$\alpha_{\mathrm{MgNa}}(r)$")

    selected = selected_compensation_results(results)
    norm = Normalize(0.0, 1.0)
    cmap = mpl.colormaps["viridis"]
    for result in selected:
        color = cmap(norm(result["x_nacl"]))
        alpha = smooth_valid(result["alpha"])
        alpha_sd = smooth_valid(result["alpha_sd"])
        label = rf"$x_{{\rm NaCl}}={result['x_nacl']:.2f}$"
        axes[0, 1].plot(r, alpha, color=color, linewidth=1.3, label=label)
        axes[0, 1].fill_between(r, alpha - alpha_sd, alpha + alpha_sd, color=color, alpha=0.10)
        axes[1, 0].plot(r, result["mgna_dndr"], color=color, linewidth=1.3, label=label)
    axes[0, 1].axhline(0.0, color="0.55", linewidth=0.8, linestyle=(0, (2, 2)))
    axes[0, 1].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    axes[0, 1].set_ylabel(r"$\alpha_{\mathrm{MgNa}}(r)$")
    axes[0, 1].set_xlim(3.0, 12.0)
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    axes[1, 0].set_ylabel(r"$\mathrm{d}N_{\mathrm{Na}|\mathrm{Mg}}/\mathrm{d}r$ ($\mathrm{\AA}^{-1}$)")
    axes[1, 0].set_xlim(3.0, 12.0)
    axes[1, 0].set_ylim(bottom=0.0)

    motif_result = min(results, key=lambda result: abs(result["x_nacl"] - 0.59))
    motif_colors = mpl.colormaps["tab10"](np.arange(len(MOTIF_CNS)))
    for motif_id, motif_cn in enumerate(MOTIF_CNS):
        axes[1, 1].plot(
            r,
            motif_result["motif_g"][motif_id],
            color=motif_colors[motif_id],
            linewidth=1.3,
            label=f"MgCl$_{{{motif_cn}}}$",
        )
    axes[1, 1].axhline(1.0, color="0.55", linewidth=0.8, linestyle=(0, (2, 2)))
    axes[1, 1].set_xlabel(r"$r$ from Mg ($\mathrm{\AA}$)")
    axes[1, 1].set_ylabel(r"$g_{\mathrm{Na}|\mathrm{MgCl}_x}(r)$")
    axes[1, 1].set_xlim(3.0, 12.0)
    axes[1, 1].set_ylim(bottom=0.0)
    axes[1, 1].legend(fontsize=8, ncol=2, title=rf"$x_{{\rm NaCl}}={motif_result['x_nacl']:.2f}$")

    add_panel_labels(axes)
    fig.tight_layout()
    fig.savefig(output_dir / "mgna_compensation_diagnostics.png", dpi=450)
    fig.savefig(output_dir / "mgna_compensation_diagnostics.pdf")
    plt.close(fig)


def write_csv_outputs(results: list[dict], output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "system",
        "x_NaCl",
        "x_MgCl2",
        "N_Na",
        "N_Mg",
        "N_Cl",
        "cutoff_A",
        "sampled_frames",
        "mean_CN",
        "mean_CN_block_sd",
        "P_CN3",
        "P_CN4",
        "P_CN5",
        "P_CN6",
        "mean_q4",
        "mean_q4_block_sd",
        "mean_q4_exact_CN4",
        "mean_q4_exact_CN4_block_sd",
        "terminal_Cl_fraction",
        "bridging_Cl_fraction",
        "terminal_among_Mg_bound_Cl",
        "bridging_among_Mg_bound_Cl",
        "corner_sharing_fraction",
        "edge_or_more_sharing_fraction",
        "face_or_more_sharing_fraction",
        "largest_cluster_fraction",
        "percolation_probability",
    ]
    with (output_dir / "structural_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            counts = result["counts"]
            writer.writerow(
                {
                    "system": result["system"],
                    "x_NaCl": result["x_nacl"],
                    "x_MgCl2": result["x_mgcl2"],
                    "N_Na": counts["Na"],
                    "N_Mg": counts["Mg"],
                    "N_Cl": counts["Cl"],
                    "cutoff_A": result["cutoff_A"],
                    "sampled_frames": result["n_sampled_frames"],
                    "mean_CN": result["mean_cn"],
                    "mean_CN_block_sd": result["mean_cn_block_sd"],
                    "P_CN3": result["cn_probability"][3],
                    "P_CN4": result["cn_probability"][4],
                    "P_CN5": result["cn_probability"][5],
                    "P_CN6": result["cn_probability"][6],
                    "mean_q4": result["q_mean"],
                    "mean_q4_block_sd": result["q_mean_block_sd"],
                    "mean_q4_exact_CN4": result["q_cn4_mean"],
                    "mean_q4_exact_CN4_block_sd": result["q_cn4_mean_block_sd"],
                    "terminal_Cl_fraction": result["terminal_fraction"],
                    "bridging_Cl_fraction": result["bridge_fraction"],
                    "terminal_among_Mg_bound_Cl": result["terminal_among_bound"],
                    "bridging_among_Mg_bound_Cl": result["bridge_among_bound"],
                    "corner_sharing_fraction": result["corner_fraction"],
                    "edge_or_more_sharing_fraction": result["edge_or_more_fraction"],
                    "face_or_more_sharing_fraction": result["face_or_more_fraction"],
                    "largest_cluster_fraction": result["largest_cluster_fraction"],
                    "percolation_probability": result["percolation_probability"],
                }
            )

    with (output_dir / "coordination_distributions.csv").open("w", newline="") as handle:
        fields = ["system", "x_NaCl", "distribution", "value", "probability", "block_sd"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for cn, (probability, sd) in enumerate(
                zip(result["cn_probability"], result["cn_probability_sd"])
            ):
                label = f"{CN_MAX_EXPLICIT + 1}+" if cn == CN_MAX_EXPLICIT + 1 else str(cn)
                writer.writerow(
                    {
                        "system": result["system"],
                        "x_NaCl": result["x_nacl"],
                        "distribution": "Mg-Cl CN",
                        "value": label,
                        "probability": probability,
                        "block_sd": sd,
                    }
                )
            for label, probability, sd in zip(
                CL_CONNECTIVITY_LABELS,
                result["cl_probability"],
                result["cl_probability_sd"],
            ):
                writer.writerow(
                    {
                        "system": result["system"],
                        "x_NaCl": result["x_nacl"],
                        "distribution": "n_Mg(Cl)",
                        "value": label,
                        "probability": probability,
                        "block_sd": sd,
                    }
                )

    with (output_dir / "angle_and_tetrahedral_distributions.csv").open("w", newline="") as handle:
        fields = ["system", "x_NaCl", "distribution", "coordinate", "density", "block_sd"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            distributions = [
                ("Cl-Mg-Cl angle", result["angle_centers"], result["clmgcl_density"], result["clmgcl_density_sd"]),
                ("Mg-Cl-Mg angle", result["angle_centers"], result["mgclmg_density"], result["mgclmg_density_sd"]),
                ("q4 CN>=4", result["q_centers"], result["q_density"], result["q_density_sd"]),
                ("q4 exact CN4", result["q_centers"], result["q_cn4_density"], result["q_cn4_density_sd"]),
            ]
            for name, coordinate, density, sd in distributions:
                for x_value, y_value, error in zip(coordinate, density, sd):
                    writer.writerow(
                        {
                            "system": result["system"],
                            "x_NaCl": result["x_nacl"],
                            "distribution": name,
                            "coordinate": x_value,
                            "density": y_value,
                            "block_sd": error,
                        }
                    )

    with (output_dir / "cluster_size_distributions.csv").open("w", newline="") as handle:
        fields = [
            "system",
            "x_NaCl",
            "cluster_size",
            "cluster_size_fraction",
            "mean_clusters_per_frame",
            "cluster_probability",
            "Mg_weighted_probability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for entry in result["cluster_distribution"]:
                writer.writerow(
                    {
                        "system": result["system"],
                        "x_NaCl": result["x_nacl"],
                        "cluster_size": entry["size"],
                        "cluster_size_fraction": entry["size_fraction"],
                        "mean_clusters_per_frame": entry["mean_clusters_per_frame"],
                        "cluster_probability": entry["cluster_probability"],
                        "Mg_weighted_probability": entry["mg_weighted_probability"],
                    }
                )

    with (output_dir / "mgna_compensation_profiles.csv").open("w", newline="") as handle:
        fields = [
            "system",
            "x_NaCl",
            "r_A",
            "P_Na_given_Mg",
            "alpha_MgNa",
            "alpha_block_sd",
            "g_MgNa",
            "g_MgNa_block_sd",
            "dN_Na_given_Mg_dr_A-1",
            "conditional_Na_density_A-3",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for values in zip(
                result["radial_centers"],
                result["p_na_given_mg"],
                result["alpha"],
                result["alpha_sd"],
                result["mgna_g"],
                result["mgna_g_sd"],
                result["mgna_dndr"],
                result["mgna_conditional_density"],
            ):
                writer.writerow(
                    dict(
                        zip(
                            fields,
                            [result["system"], result["x_nacl"], *values],
                        )
                    )
                )

    with (output_dir / "motif_resolved_na_profiles.csv").open("w", newline="") as handle:
        fields = ["system", "x_NaCl", "Mg_CN", "r_A", "g_Na_given_motif", "block_sd"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for motif_id, motif_cn in enumerate(MOTIF_CNS):
                for r_value, g_value, sd in zip(
                    result["radial_centers"],
                    result["motif_g"][motif_id],
                    result["motif_g_sd"][motif_id],
                ):
                    writer.writerow(
                        {
                            "system": result["system"],
                            "x_NaCl": result["x_nacl"],
                            "Mg_CN": motif_cn,
                            "r_A": r_value,
                            "g_Na_given_motif": g_value,
                            "block_sd": sd,
                        }
                    )

    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cutoffs = composition_cutoffs(args.rdf_metrics)
    radial_edges = np.arange(
        0.0,
        args.radial_max + args.radial_bin_width * 0.5,
        args.radial_bin_width,
    )
    angle_edges = np.arange(0.0, 180.0 + args.angle_bin_width * 0.5, args.angle_bin_width)
    q_edges = np.arange(-2.0, 1.0 + args.q_bin_width * 0.5, args.q_bin_width)
    paths = sorted(args.input_dir.glob("cl*/md.traj"))
    if not paths:
        raise FileNotFoundError(f"No cl*/md.traj files found below {args.input_dir}")

    results = []
    for path in paths:
        system = path.parent.name
        if system not in cutoffs:
            raise KeyError(f"No Mg-Cl RDF cutoff for {system}")
        print(f"Analyzing {system} with Mg-Cl cutoff {cutoffs[system]:.3f} A ...", flush=True)
        results.append(
            analyze_composition(
                path,
                cutoffs[system],
                args,
                radial_edges,
                angle_edges,
                q_edges,
            )
        )
    results.sort(key=lambda result: result["x_nacl"])

    metadata = {
        "input_dir": str(args.input_dir),
        "rdf_cutoff_source": str(args.rdf_metrics),
        "composition": "x_NaCl = N_Na / (N_Na + N_Mg)",
        "discard_fraction": args.discard_fraction,
        "max_frames_per_composition": args.max_frames,
        "time_blocks": args.blocks,
        "tetrahedral_order": "q4 from four nearest Cl for Mg centers with CN >= 4",
        "terminal_Cl": "n_Mg(Cl) = 1",
        "bridging_Cl": "n_Mg(Cl) >= 2",
        "corner_sharing": "Mg pair shares exactly one Cl",
        "edge_or_more_sharing": "Mg pair shares at least two Cl",
        "percolation": "non-contractible Mg-network loop in the periodic cell",
        "warren_cowley": "alpha_MgNa(r) = 1 - P(Na|Mg,r)/x_NaCl; neighbors are cations in the radial shell",
        "uncertainty": "sample standard deviation across contiguous trajectory blocks",
    }
    write_csv_outputs(results, args.output_dir, metadata)
    plot_local_coordination(results, args.output_dir)
    plot_network_topology(results, args.output_dir)
    plot_cluster_distributions(results, args.output_dir)
    plot_mgna_compensation(results, args.output_dir)
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
