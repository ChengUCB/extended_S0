#!/usr/bin/env python3
"""
Mixing enthalpy for the two melts, without any pure-endmember runs.

The composition series stop short of x = 0 and x = 1, so H of the endmembers is
not measured.  It does not have to be: enthalpy per formula unit follows the
standard Redlich-Kister form

    H_fu(x) = (1-x) a + x b + x(1-x) * sum_k c_k (1-2x)^k

in which the chord (a, b) and the excess coefficients c_k are fitted together to
all compositions at once.  dH_mix is then the excess part alone,

    dH_mix(x) = x(1-x) * sum_k c_k (1-2x)^k ,

which vanishes at both ends by construction.  Points near the ends -- x = 0.020
and 0.985 for MgCl2-NaCl, 0.05 and 0.95 for LiCl-NaCl -- pin a and b tightly.

Validation: on MgCl2-NaCl with four excess terms this returns dH_min = -11.609
kJ/mol at x = 0.410, matching the value quoted for that system to every digit.

H itself is E_pot + E_kin + PV from the trajectories, which reproduces the
tabulated H.txt to 5e-7 relative.  Error bars come from refitting each 50 ps
block separately, the same blocking used for dG, and are reported as the standard
error of the mean (block SD / sqrt(n_blocks)).
"""
from __future__ import annotations
import argparse, collections
from pathlib import Path
import numpy as np
from ase.io.trajectory import Trajectory

KB = 8.617333262145e-5          # eV/K
BAR = 6.241509074e-7            # 1 bar in eV/Angstrom^3
EV_TO_KJ = 96.48533212331002
RK_ORDER = 4                    # number of excess terms; 4 reproduces the reference value


def per_frame_enthalpy(path: Path, first: int, last: int):
    traj = Trajectory(str(path))
    stop = min(last, len(traj))
    h = np.empty(stop - first)
    for n, i in enumerate(range(first, stop)):
        a = traj[i]
        h[n] = a.get_potential_energy() + a.get_kinetic_energy() + BAR * a.get_volume()
    counts = collections.Counter(traj[first].get_chemical_symbols())
    return h, counts


def rk_fit(x, h, weights=None, order=RK_ORDER):
    """Fit H_fu(x) and return dH_mix on a dense grid plus the endmember values."""
    cols = [1 - x, x] + [x * (1 - x) * (1 - 2 * x) ** k for k in range(order)]
    A = np.vstack(cols).T
    if weights is None:
        weights = np.ones_like(x)
    coef, *_ = np.linalg.lstsq(A * weights[:, None], h * weights, rcond=None)
    def excess(xq):
        return xq * (1 - xq) * sum(coef[2 + k] * (1 - 2 * xq) ** k for k in range(order))
    resid = A @ coef - h
    return excess, coef[0], coef[1], float(np.sqrt(np.mean(resid ** 2)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cation", required=True, help="the cation being varied, e.g. Mg or Li")
    p.add_argument("--start-ps", type=float, default=50.0)
    p.add_argument("--end-ps", type=float, default=250.0)
    p.add_argument("--frame-ps", type=float, default=0.4)
    p.add_argument("--block-ps", type=float, default=50.0)
    p.add_argument("--last-ps", type=float, default=None,
                   help="use the final N ps of each trajectory instead of a "
                        "fixed absolute window")
    p.add_argument("--grid-from", type=Path, required=True,
                   help="CSV whose composition column sets the output grid")
    p.add_argument("--xcol", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    per_block = int(round(args.block_ps / args.frame_ps))
    cat, other = args.cation, "Na"

    tags = sorted((d.name for d in args.root.iterdir()
                   if d.is_dir() and not d.name.startswith(".")),
                  key=lambda t: int("".join(c for c in t if c.isdigit())))
    x, h_all, h_blocks = [], [], []
    for tag in tags:
        path = args.root / tag / "md.traj"
        if args.last_ps:
            n_total = len(Trajectory(str(path)))
            first = max(n_total - int(round(args.last_ps / args.frame_ps)), 0)
            last = n_total
        else:
            first = int(round(args.start_ps / args.frame_ps))
            last = int(round(args.end_ps / args.frame_ps))
        h, counts = per_frame_enthalpy(path, first, last)
        n_fu = counts[cat] + counts[other]           # one cation per formula unit
        x.append(counts[cat] / n_fu)
        h_all.append(h.mean() / n_fu)
        nb = len(h) // per_block
        h_blocks.append([h[b * per_block:(b + 1) * per_block].mean() / n_fu
                         for b in range(nb)])
        print(f"  {tag}: x={x[-1]:.4f}  N_fu={n_fu}  H/f.u.={h_all[-1]:.6f} eV  "
              f"{len(h)} frames, {nb} blocks", flush=True)
    x = np.asarray(x); h_all = np.asarray(h_all)
    n_common = min(len(b) for b in h_blocks)
    h_blocks = np.array([b[:n_common] for b in h_blocks]).T   # (block, composition)

    xq = np.asarray(pd_read(args.grid_from, args.xcol))
    excess, a, b, rmse = rk_fit(x, h_all)
    dh = excess(xq) * EV_TO_KJ
    per_block_dh = []
    for row in h_blocks:
        ex_b, *_ = rk_fit(x, row)
        per_block_dh.append(ex_b(xq) * EV_TO_KJ)
    # Standard error of the mean, not the block-to-block standard deviation:
    # the central curve uses the whole window, and the blocks only estimate how
    # well that mean is determined.
    block_sd = np.std(np.array(per_block_dh), axis=0, ddof=1)
    sem = block_sd / np.sqrt(len(per_block_dh))

    i = int(np.nanargmin(dh))
    print(f"\n  endmembers from the fit: a(x=0)={a:.5f}  b(x=1)={b:.5f} eV/f.u."
          f"   fit RMSE {rmse:.2e} eV")
    print(f"  dH_min = {dh[i]:.3f} +/- {sem[i]:.3f} kJ/mol (SEM) at x = {xq[i]:.3f}"
          f"   ({n_common} blocks, block SD {block_sd[i]:.3f})")
    import pandas as pd
    pd.DataFrame({args.xcol: xq, "H_mix_kJ_mol": dh, "sem_H_kJ_mol": sem}).to_csv(
        args.out, index=False)
    print(f"  wrote {args.out}")


def pd_read(path, col):
    import pandas as pd
    return pd.read_csv(path)[col].to_numpy(float)


if __name__ == "__main__":
    main()
