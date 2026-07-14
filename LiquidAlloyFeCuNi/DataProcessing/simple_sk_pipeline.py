#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from numba import njit, prange

def _parse_atoms_header(line):

    cols = line.strip().split()[2:]     # drop 'ITEM:' and 'ATOMS'

    def find(*names):
        for n in names:
            if n in cols:
                return cols.index(n), n
        return None, None

    # Accept both 'type' (numeric) and 'element' (symbol) as the species column
    ic, id_col_name = find('type', 'element')
    xc, _ = find('x')
    yc, _ = find('y')
    zc, _ = find('z')

    missing = [n for n, c in [('type/element', ic), ('x', xc), ('y', yc), ('z', zc)]
               if c is None]
    if missing:
        raise ValueError(
            f"LAMMPS ATOMS header missing required columns {missing}.\n"
            f"Got: {line.strip()!r}\n"
            f"Need at least: element/type x y z  "
        )

    id_kind    = 'element' if id_col_name == 'element' else 'type'

    return ic, id_kind, xc, yc, zc


def _read_lammpstrj_frame(f, type_to_elem):
    """Read one frame from an open LAMMPS dump file."""
    line = f.readline()
    if not line.strip():
        return None
    for _ in range(2):
        f.readline()                       # timestep value + "ITEM: NUMBER OF ATOMS"
    natoms = int(f.readline())
    f.readline()                           # ITEM: BOX BOUNDS ...
    cell = np.zeros(3)
    for i in range(3):
        lo, hi = map(float, f.readline().split()[:2])
        cell[i] = hi - lo
    ic, id_kind, xc, yc, zc = _parse_atoms_header(f.readline())
    elems = np.empty(natoms, dtype="U8")
    frac = np.zeros((natoms, 3))
    for i in range(natoms):
        parts = f.readline().split()
        if not parts:
            return None
        if id_kind == 'element':
            elems[i] = parts[ic]                          # already an element symbol
        else:
            elems[i] = type_to_elem.get(parts[ic], parts[ic])  # numeric → symbol
        x, y, z = float(parts[xc]), float(parts[yc]), float(parts[zc])
        frac[i] = (x / cell[0], y / cell[1], z / cell[2])
    return cell, elems, frac


def read_lammpstrj(path, type_to_elem):
    """Yield (cell, elems, frac_coords) for every frame in a LAMMPS dump file."""
    with open(path) as f:
        while True:
            frame = _read_lammpstrj_frame(f, type_to_elem)
            if frame is None:
                break
            # reads one frame into memory at a time 
            yield frame


def read_frames(path, type_to_elem=None):
    """Yield frames from a LAMMPS dump file."""
    if type_to_elem is None:
        raise ValueError("--type-map required for LAMMPS trajectories")
    yield from read_lammpstrj(Path(path), type_to_elem)

def make_kgrid(cell, bins):
    """
    Build the k-grid for an orthorhombic cell.

    Returns
    -------
    kgrid_real : (bins^3, 3) array
        k-vectors in 1/Å (no 2π).  These are written to the S(k) files.
    kgrid_2pi  : (bins^3, 3) array
        k-vectors as 2π × integer indices.  Used in the phase exp(-i k·q_frac).
    k_sq       : (bins^3,) array
        |k|² in (1/Å)².
    """
    idx = np.array(
        [[i, j, k]
         for i in range(bins)
         for j in range(bins)
         for k in range(bins)],
        dtype=float,
    )
    kgrid_real = idx * (1.0 / cell)[None, :]
    kgrid_2pi = idx * (2.0 * math.pi)
    k_sq = np.sum(kgrid_real**2, axis=1)
    return kgrid_real, kgrid_2pi, k_sq

@njit(parallel=True, fastmath=True)
def _ft_density(q_frac, kgrid_2pi):
    """Σ_j exp(-i kgrid_2pi[m] · q_frac[j]) for all k-points m."""
    nq = q_frac.shape[0]
    nk = kgrid_2pi.shape[0]
    ak = np.zeros(nk, dtype=np.complex128)
    for m in prange(nk):
        s = 0.0j
        kx = kgrid_2pi[m, 0]
        ky = kgrid_2pi[m, 1]
        kz = kgrid_2pi[m, 2]
        for j in range(nq):
            s += np.exp(-1j * (q_frac[j, 0]*kx + q_frac[j, 1]*ky + q_frac[j, 2]*kz))
        ak[m] = s
    return ak

def sk_one_frame(frac, elems, kgrid_2pi, elements):
    """
    Compute all partial S_αβ(k) for one frame.

    Parameters
    ----------
    frac       : (N, 3) fractional coordinates
    elems      : (N,)   element labels (strings)
    kgrid_2pi  : (M, 3) k-grid with 2π factor
    elements   : ordered list of element names

    Returns
    -------
    dict {(eA, eB): real array of length M}
    """
    fts, counts = {}, {}
    for e in elements:
        q = frac[elems == e]
        counts[e] = len(q)
        fts[e] = (
            _ft_density(q, kgrid_2pi)
            if len(q) > 0
            else np.zeros(len(kgrid_2pi), dtype=np.complex128)
        )

    sk = {}
    for eA, eB in combinations_with_replacement(elements, 2):
        nA, nB = counts[eA], counts[eB]
       
        if nA == 0 or nB == 0:
            sk[(eA, eB)] = np.full(len(kgrid_2pi), np.nan)
        elif eA == eB:
            sk[(eA, eB)] = (np.abs(fts[eA]) ** 2 / nA).real
        else:
            sk[(eA, eB)] = (fts[eA] * np.conj(fts[eB]) / math.sqrt(nA * nB)).real
    return sk

def _finalize_block_means(block_means_list):
    """
    Given a list of per-block mean dicts [{pair: array}, ...],
    return {pair: (overall_mean, std_error)}.
    """
    if not block_means_list:
        return {}
    pairs = list(block_means_list[0].keys())
    n = len(block_means_list)
    result = {}
    for pair in pairs:
        stack = np.stack([bm[pair] for bm in block_means_list])  # (n_blocks, n_k)
        mean = np.nanmean(stack, axis=0)
        error = np.nanstd(stack, axis=0, ddof=1) / math.sqrt(n) if n > 1 \
                else np.full_like(mean, np.nan)
        result[pair] = (mean, error)
    return result


def count_frames(path):
    """Fast frame count: scan for ITEM: TIMESTEP without parsing atom data."""
    path = Path(path)
    with open(path, "rb") as f:
        return sum(1 for line in f if line.startswith(b"ITEM: TIMESTEP"))
def write_sk_file(path, kgrid_real, k_sq, means, errors, pairLabels):
    """
    Write S(k) data to a plain-text file.

    Format:  k^2  S(k)  S(k)_error  (one mean/error column pair per element pair)
    k units: 1/Å (no 2π factor), consistent with existing workflow.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    row = "# k^2  "
    for label in pairLabels:
        row += f" {label} {label}_error"
    lines.append(row)
    for i in range(len(k_sq)):
        row = f"{k_sq[i]:.6e}"
        for mean, error in zip(means, errors):
            row += f" {mean[i]:.6e} {error[i]:.6e}"
        lines.append(row)
    path.write_text("\n".join(lines) + "\n")


def build_parser():
    p = argparse.ArgumentParser(
        description="Trajectory → S(k) pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--traj", type=Path,
                   help="single trajectory file (.lammpstrj)")
    g.add_argument("--traj-dir", type=Path,
                   help="batch: search this directory (and immediate subdirs) "
                        "for trajectory files")
    p.add_argument("--traj-name", default="lj.lammpstrj", metavar="NAME",
                   help="filename to look for inside --traj-dir subdirs "
                        "(default: lj.lammpstrj)")
    p.add_argument("--elements", nargs="+", required=True, metavar="ELEM",
                   help="element names in order, e.g. Fe Cu Ni  or  O I Cl Na")
    p.add_argument("--type-map", nargs="*", default=[], metavar="ID:ELEM",
                   help="LAMMPS type-id → element map, e.g. 1:Fe 2:Cu 3:Ni. "
                        "Not needed when the dump uses an 'element' column "
                        "(e.g. ITEM: ATOMS id element x y z). "
                        "If omitted for a 'type' column, uses --elements order "
                        "(1→first, 2→second, …).")
    p.add_argument("--n-splits", type=int, default=3, metavar="N",
                   help="number of trajectory splits (default: 3)")
    p.add_argument("--n-blocks", type=int, default=4, metavar="N",
                   help="blocks per split for block averaging (default: 4). "
                        "Increase if autocorrelation time is long.")
    p.add_argument("--k-bins", type=int, default=8, metavar="N",
                   help="k-grid bins per dimension (default: 8, gives 8³ k-points)")
    p.add_argument("--out-dir", type=Path, default=Path("sk_output"),
                   help="directory for S(k) files (default: sk_output)")
    p.add_argument("--label", default=None,
                   help="label prefix for output files (default: inferred from "
                        "element counts in the trajectory)")

    return p


def composition_label(counts, elements):
    """Build a label string from atom counts, e.g. 'Fe80-Cu6320-Ni1600'."""
    parts = [f"{e}{int(counts.get(e, 0))}" for e in elements]
    return "-".join(parts)


def process_trajectory(
    traj_path, elements, type_to_elem,
    n_splits, n_blocks, k_bins,
    out_dir,
    label=None,
):
    """
    Full pipeline for one trajectory file.

    Streams frames one at a time — never stores more than n_blocks block-mean
    arrays per split in memory (vs. all per-frame S(k) arrays previously).

    Writes one S(k) file per split (``{label}-{split}-allSk.dat``) plus a
    full-trajectory file (``{label}-allSk.dat``) into ``out_dir``. Returns None.
    """
    traj_path = Path(traj_path)

    print(f"  {traj_path.name}: counting frames...", end=" ", flush=True)
    n_frames = count_frames(traj_path)
    print(n_frames, flush=True)

    if n_frames == 0:
        print("    No frames found, skipping.")
        return []

    split_size   = n_frames // n_splits
    block_size   = max(1, split_size // n_blocks)
    full_block_size = max(1, n_frames // n_blocks)

    # Accumulators: running sum + count for each (split, block_within_split)
    # Indexed as split_acc[split_idx][block_idx] = {pair: sum_array} | None
    split_acc = [[None] * n_blocks for _ in range(n_splits)]
    full_acc  = [None] * n_blocks

    kgrid_real = kgrid_2pi = k_sq = None
    elem_counts = {e: 0 for e in elements}

    for frame_idx, (cell, elems, frac) in enumerate(
            read_frames(traj_path, type_to_elem)):

        if frame_idx == 0:
            kgrid_real, kgrid_2pi, k_sq = make_kgrid(cell, k_bins)
            for e in elements:
                elem_counts[e] = int(np.sum(elems == e))
        sk = sk_one_frame(frac, elems, kgrid_2pi, elements)

        split_idx = frame_idx // split_size
        # discard extra frames beyond the last full split
        if split_idx < n_splits:
            local       = frame_idx % split_size
            block_idx   = min(local // block_size, n_blocks - 1)
            acc = split_acc[split_idx][block_idx]
            if acc is None:
                split_acc[split_idx][block_idx] = {p: v.copy() for p, v in sk.items()}
                split_acc[split_idx][block_idx]["_n"] = 1
            else:
                for p, v in sk.items():
                    acc[p] += v
                acc["_n"] += 1

        # ── accumulate into full-trajectory block ─────────────────────────────
        fb_idx = min(frame_idx // full_block_size, n_blocks - 1)
        acc = full_acc[fb_idx]
        if acc is None:
            full_acc[fb_idx] = {p: v.copy() for p, v in sk.items()}
            full_acc[fb_idx]["_n"] = 1
        else:
            for p, v in sk.items():
                acc[p] += v
            acc["_n"] += 1

        if (frame_idx + 1) % 100 == 0:
            print(f"    {frame_idx + 1}/{n_frames} frames...", end="\r", flush=True)

    print(f"    {n_frames}/{n_frames} frames done        ", flush=True)

    if label is None:
        label = composition_label(elem_counts, elements)

    pairs = list(combinations_with_replacement(elements, 2))

    # ── convert accumulators → block means → mean + error ────────────────────
    def acc_to_block_means(acc_list):
        block_means = []
        for acc in acc_list:
            if acc is None:
                continue
            n = acc["_n"]
            block_means.append({p: acc[p] / n for p in pairs})
        return block_means

    def unpack(averaged):
        """Split {pair: (mean, error)} into parallel label/mean/error lists."""
        labels, means, errors = [], [], []
        for pair in pairs:
            mean, error = averaged[pair]
            labels.append("".join(pair))
            means.append(mean)
            errors.append(error)
        return labels, means, errors

    for s in range(n_splits):
        averaged = _finalize_block_means(acc_to_block_means(split_acc[s]))
        pairLabels, means, errors = unpack(averaged)
        write_sk_file(out_dir / f"{label}-{s + 1}-allSk.dat",
                      kgrid_real, k_sq, means, errors, pairLabels)

    averaged_full = _finalize_block_means(acc_to_block_means(full_acc))
    pairLabels, means, errors = unpack(averaged_full)
    write_sk_file(out_dir / f"{label}-allSk.dat",
                  kgrid_real, k_sq, means, errors, pairLabels)


def _collect_trajectories(args):
    """Return list of (traj_path, label) pairs."""
    trajs = []
    if args.traj:
        trajs.append((args.traj, args.label))
    else:
        base = args.traj_dir
        for f in sorted(base.glob("*.lammpstrj")):
            trajs.append((f, args.label))

        for subdir in sorted(p for p in base.iterdir() if p.is_dir()):
            candidate = subdir / args.traj_name
            if candidate.exists():
                trajs.append((candidate, args.label))
            else:
                for f in sorted(subdir.glob("*.lammpstrj")):
                    trajs.append((f, args.label))
    return trajs


def main():
    args = build_parser().parse_args()

    # Build type→element map
    type_to_elem = {}
    if args.type_map:
        for item in args.type_map:
            tid, elem = item.split(":", 1)
            type_to_elem[tid] = elem
    else:
        for i, e in enumerate(args.elements, 1):
            type_to_elem[str(i)] = e

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajs = _collect_trajectories(args)

    if not trajs:
        print("No trajectory files found.")
        return 1

    print(f"Found {len(trajs)} trajectory file(s).")
    print(f"Elements: {args.elements}  |  splits: {args.n_splits}  "
          f"|  blocks/split: {args.n_blocks}  |  k-bins: {args.k_bins}")

    for i, (traj_path, label) in enumerate(trajs, 1):
        print(f"\n[{i}/{len(trajs)}] {traj_path}")
        process_trajectory(
            traj_path, args.elements, type_to_elem,
            args.n_splits, args.n_blocks, args.k_bins,
            args.out_dir, label=label,
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
