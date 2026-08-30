#!/usr/bin/env python3
"""
Compute partial structure factors S(k) for molten MgCl2-NaCl straight from the
MD trajectories, and write them in the same .dat format as sk_output/.

  * three species (Mg, Cl, Na) instead of two, so six curves are written in the
    order MgMg, MgCl, MgNa, ClCl, ClNa, NaNa.
  * the merged "X" salt mode is not written.  The existing sk_output files do
    not have it, so adding it would change the column layout.
  * the trajectory is cut into blocks of a chosen length in ps rather than into
    a chosen number of pieces.

What it computes, for each frame:

    rho_A(k) = sum over atoms of species A of exp(-i k . r)
    S_AB(k)  = Re[ rho_A(k) rho_B(k)* ] / sqrt(N_A N_B)

The allowed k of a periodic box are k = h*b1 + k*b2 + l*b3 with integer h,k,l,
so the phase is just 2*pi times (h,k,l) dotted into the fractional coordinate.
That holds for any box shape.

k is reported WITHOUT the 2*pi factor, so k^2 comes out in cycles^2/Angstrom^2.
This matches the existing .dat files: the smallest non-zero k^2 there is
1/L^2 for the box edge L.

The k grid is h,k,l each running 0..k_bins-1, so k_bins=8 gives 512 rows,
which is what the existing .dat files have.

Output files, written into --out-dir:

    <tag>-allSk.dat      averaged over the whole time window
    <tag>-1-allSk.dat    first block
    <tag>-2-allSk.dat    second block, and so on

The error column in each file is the scatter between sub-blocks of that file
(the file is internally divided into --n-error-blocks pieces).

Run one system:

    python3 compute_sk.py --root "MD_inputs/SelectedMgCl2-NaCl_MD" \
                          --tags cl3232 --block-ps 50 --out-dir sk_recomputed

Run all of them by leaving out --tags.
"""

from __future__ import annotations

import argparse
import math
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from numba import njit, prange


SPECIES = ("Mg", "Cl", "Na")
PAIRS = list(combinations_with_replacement(SPECIES, 2))


def set_species(names: tuple[str, ...]) -> None:
    global SPECIES, PAIRS
    SPECIES = tuple(names)
    PAIRS = list(combinations_with_replacement(SPECIES, 2))


K_BINS = 8
FRAME_PS = 0.4
START_PS = 50.0
END_PS = 250.0
N_ERROR_BLOCKS = 5


@njit(parallel=True, fastmath=True)
def _ft_density(frac, kgrid_2pi):
    """Sum of exp(-i k . r) over the atoms of one species, for every k."""
    n_atoms = frac.shape[0]
    n_k = kgrid_2pi.shape[0]
    out = np.zeros(n_k, dtype=np.complex128)
    for m in prange(n_k):
        acc = 0.0j
        kx = kgrid_2pi[m, 0]
        ky = kgrid_2pi[m, 1]
        kz = kgrid_2pi[m, 2]
        for j in range(n_atoms):
            acc += np.exp(-1j * (frac[j, 0] * kx + frac[j, 1] * ky + frac[j, 2] * kz))
        out[m] = acc
    return out


def make_kgrid(bins: int):
    idx = np.array(
        [[i, j, k] for i in range(bins) for j in range(bins) for k in range(bins)],
        dtype=float,
    )
    return idx, idx * (2.0 * math.pi)


def sk_one_frame(frac, symbols, kgrid_2pi):
    """The six S_AB(k) curves for a single frame."""
    ft, counts = {}, {}
    for s in SPECIES:
        q = np.ascontiguousarray(frac[symbols == s])
        counts[s] = len(q)
        ft[s] = _ft_density(q, kgrid_2pi)
    out = {}
    for a, b in PAIRS:
        if a == b:
            out[(a, b)] = (np.abs(ft[a]) ** 2 / counts[a]).real
        else:
            out[(a, b)] = (ft[a] * np.conj(ft[b]) / math.sqrt(counts[a] * counts[b])).real
    return out, counts


def frame_iterator(path: Path, source: str, first: int, last: int):
    """Yield (fractional coords, reciprocal cell) for frames first..last-1."""
    if source == "traj":
        from ase.io.trajectory import Trajectory

        traj = Trajectory(str(path))
        for i in range(first, min(last, len(traj))):
            atoms = traj[i]
            yield (
                np.ascontiguousarray(atoms.get_scaled_positions(wrap=True)),
                np.array(atoms.cell.reciprocal()),
                np.array(atoms.get_chemical_symbols()),
            )
    else:
        from ase.io import iread

        for i, atoms in enumerate(iread(str(path), index=":")):
            if i < first:
                continue
            if i >= last:
                break
            yield (
                np.ascontiguousarray(atoms.get_scaled_positions(wrap=True)),
                np.array(atoms.cell.reciprocal()),
                np.array(atoms.get_chemical_symbols()),
            )


def average_group(frames, n_error_blocks: int):
    """Average a list of per-frame results and estimate an error from sub-blocks.

    frames is a list of (sk_dict, k_sq) pairs.  The reported value is the direct
    frame average, so unequal error-block sizes cannot bias it.  We split the
    frames into contiguous pieces only to estimate the error from the scatter
    of the piece averages.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("cannot average an empty frame group")
    if n_error_blocks < 1:
        raise ValueError("n_error_blocks must be at least 1")
    pieces = np.array_split(np.arange(n), min(n_error_blocks, n))

    piece_means = []
    piece_sizes = []
    for piece in pieces:
        if len(piece) == 0:
            continue
        piece_means.append(
            {p: np.mean([frames[i][0][p] for i in piece], axis=0) for p in PAIRS}
        )
        piece_sizes.append(len(piece))

    means, errors = {}, {}
    for p in PAIRS:
        stack = np.stack([pm[p] for pm in piece_means])
        means[p] = np.mean([frame[0][p] for frame in frames], axis=0)
        if len(stack) > 1:
            weights = np.asarray(piece_sizes, dtype=float)[:, None]
            residual_sum = np.sum(weights * (stack - means[p]) ** 2, axis=0)
            errors[p] = np.sqrt(
                residual_sum / ((len(stack) - 1) * np.sum(weights))
            )
        else:
            errors[p] = np.full(stack.shape[1], np.nan)
    k_sq = np.mean([frame[1] for frame in frames], axis=0)
    return k_sq, means, errors


def write_dat(path: Path, k_sq, means, errors):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["".join(p) for p in PAIRS]
    lines = ["# k^2  " + "  ".join(f"{l} {l}_error" for l in labels)]
    for i in range(len(k_sq)):
        row = f"{k_sq[i]:.6e}"
        for p in PAIRS:
            row += f" {means[p][i]:.6e} {errors[p][i]:.6e}"
        lines.append(row)
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def require_fresh_output_dir(path: Path) -> None:
    """Prevent stale numbered blocks from being mixed with a new run."""
    if not path.is_dir():
        return
    existing = sorted(path.glob("*-allSk.dat"))
    if existing:
        names = ", ".join(item.name for item in existing)
        raise FileExistsError(
            f"Refusing to mix or overwrite prior S(k) outputs in {path}: {names}. "
            "Use a new --out-dir or move the old files first."
        )


def count_frames(path: Path, source: str) -> int:
    if source == "traj":
        from ase.io.trajectory import Trajectory

        return len(Trajectory(str(path)))
    with path.open() as fh:
        return sum(1 for line in fh if line.startswith("Lattice") or "Lattice=" in line)


def process_one(tag: str, root: Path, out_dir: Path, source: str,
                start_ps: float, end_ps: float, frame_ps: float,
                block_ps: float, k_bins: int, n_error_blocks: int,
                n_blocks_req: int | None = None,
                last_ps: float | None = None) -> None:
    if frame_ps <= 0.0:
        raise ValueError("frame_ps must be positive")
    path = root / tag / ("md.traj" if source == "traj" else "npt_mace.xyz")
    require_fresh_output_dir(out_dir / tag)


    if last_ps:
        n_total = count_frames(path, source)
        total_ps = (n_total - 1) * frame_ps
        start_ps = max(total_ps - last_ps, 0.0)
        end_ps = 0.0
    first = int(round(start_ps / frame_ps))


    to_the_end = end_ps <= 0.0
    last = 1 << 30 if to_the_end else int(round(end_ps / frame_ps))


    frames_per_block = int(round(block_ps / frame_ps))
    if n_blocks_req is None and frames_per_block < 1:
        raise ValueError("block_ps must span at least one frame")

    kgrid_idx, kgrid_2pi = make_kgrid(k_bins)

    window = f"{start_ps:g}-end ps" if to_the_end else f"{start_ps:g}-{end_ps:g} ps"
    print(f"{tag}: from frame {first} ({window}), "
          f"blocks of {frames_per_block} frames ({block_ps:g} ps)", flush=True)

    collected = []
    t0 = time.time()
    for n, (frac, recip, symbols) in enumerate(
        frame_iterator(path, source, first, last)
    ):
        k_sq = np.sum((kgrid_idx @ recip) ** 2, axis=1)
        sk, counts = sk_one_frame(frac, symbols, kgrid_2pi)
        collected.append((sk, k_sq))
        if (n + 1) % 50 == 0:
            rate = (n + 1) / (time.time() - t0)
            print(f"  {n + 1} frames  ({rate:.1f} fps)", end="\r", flush=True)
    n_frames = len(collected)
    if n_frames == 0:
        print(f"  {tag}: no frames in the requested window, skipped")
        return
    print(f"  {n_frames} frames in {time.time() - t0:.0f} s"
          f"   composition: " + ", ".join(f"{s}={counts[s]}" for s in SPECIES),
          flush=True)

    if n_blocks_req is not None:
        if n_blocks_req < 1:
            raise ValueError("n_blocks must be at least 1")
        if n_blocks_req > n_frames:
            raise ValueError(
                f"n_blocks={n_blocks_req} exceeds the {n_frames} selected frames"
            )

    k_sq, means, errors = average_group(collected, n_error_blocks)
    write_dat(out_dir / tag / f"{tag}-allSk.dat", k_sq, means, errors)


    if n_blocks_req is not None:
        pieces = [p for p in np.array_split(np.arange(n_frames), n_blocks_req)]
        n_blocks = len(pieces)
        print(f"  splitting the window into {n_blocks} equal pieces of "
              f"{[len(p) for p in pieces]} frames "
              f"({[round(len(p) * frame_ps, 1) for p in pieces]} ps)")
    else:
        n_blocks = n_frames // frames_per_block
        pieces = [np.arange(b * frames_per_block, (b + 1) * frames_per_block)
                  for b in range(n_blocks)]
        if n_blocks * frames_per_block != n_frames:
            print(f"  note: {n_frames} frames is not a whole number of "
                  f"{frames_per_block}-frame blocks; the last "
                  f"{n_frames - n_blocks * frames_per_block} frames are dropped "
                  f"from the per-block files (they are still in the whole-window file)")
    for b in range(n_blocks):
        chunk = [collected[i] for i in pieces[b]]
        k_sq_b, means_b, errors_b = average_group(chunk, n_error_blocks)
        write_dat(out_dir / tag / f"{tag}-{b + 1}-allSk.dat", k_sq_b, means_b, errors_b)
    print(f"  wrote {n_blocks + 1} files to {out_dir / tag}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--species", type=str, default=",".join(SPECIES),
                   help="three species in output order, e.g. Li,Cl,Na")
    p.add_argument("--root", type=Path, required=True,
                   help="directory holding one sub-directory per system")
    p.add_argument("--tags", nargs="*", default=None,
                   help="which systems to do (default: all of them)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source", choices=("traj", "xyz"), default="traj",
                   help="md.traj is the same data as npt_mace.xyz but reads "
                        "faster and allows skipping straight to a frame")
    p.add_argument("--start-ps", type=float, default=START_PS)
    p.add_argument("--end-ps", type=float, default=END_PS,
                   help="end of the window in ps; pass 0 or a negative number "
                        "to run to the end of each trajectory")
    p.add_argument("--frame-ps", type=float, default=FRAME_PS)
    p.add_argument("--block-ps", type=float, default=50.0,
                   help="length of one block, in ps (ignored if --n-blocks given)")
    p.add_argument("--last-ps", type=float, default=None,
                   help="use the final N ps of each trajectory; overrides "
                        "--start-ps/--end-ps and gives every system the same "
                        "window length even though the runs differ in length")
    p.add_argument("--n-blocks", type=int, default=None,
                   help="instead of fixed-length blocks, cut the window into "
                        "this many equal pieces (matches sk_proc_traj.py)")
    p.add_argument("--k-bins", type=int, default=K_BINS)
    p.add_argument("--n-error-blocks", type=int, default=N_ERROR_BLOCKS)
    args = p.parse_args()
    set_species(tuple(x.strip() for x in args.species.split(",")))
    print(f"species order: {SPECIES}", flush=True)

    tags = args.tags or sorted(
        (d.name for d in args.root.iterdir()
         if d.is_dir() and not d.name.startswith(".")),
        key=lambda t: int(''.join(ch for ch in t if ch.isdigit())),
    )
    for tag in tags:
        process_one(tag, args.root, args.out_dir, args.source,
                    args.start_ps, args.end_ps, args.frame_ps,
                    args.block_ps, args.k_bins, args.n_error_blocks,
                    args.n_blocks, args.last_ps)


if __name__ == "__main__":
    main()
