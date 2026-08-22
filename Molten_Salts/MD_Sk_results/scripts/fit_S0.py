#!/usr/bin/env python3
"""
Get S(0) for molten MgCl2-NaCl from the S(k) files in sk_output/.

Each file sk_output/<tag>/<tag>*-allSk.dat holds six curves measured from MD:
MgMg, MgCl, MgNa, ClCl, ClNa, NaNa, each as a function of k.  We want their
value at k = 0, but MD cannot reach k = 0 (the smallest k you can measure is
set by the box size).  So we fit a curve over the small-k points and read off
where it would land at k = 0.

The number we actually care about at the end is S0_NN, the "number-number"
combination.  It is related to how compressible the liquid is.

Put the six curves into a 3x3 table (a matrix) at each k, in the order
(Mg, Cl, Na).  The fitted shape is:

    S(k) = 1 / ( A  +  k^2 * L  +  charge_part * kappa^2 / k^2 )

read as matrix operations rather than plain division.  The three pieces:

  A            a constant.  This is the piece that survives at k = 0, and it
               is what we ultimately want.
  k^2 * L      a small correction that grows with k.  It only exists to let
               the curve bend a little so A is not distorted.
  kappa^2/k^2  a piece that blows up as k gets small.

Why the blow-up piece has to be there: a molten salt shields charge very well.
That forces the charge-charge part of S(k) to shrink to zero as k -> 0.  When
something shrinks to zero, its reciprocal blows up -- hence the 1/k^2.  The
"charge_part" factor makes sure the blow-up happens ONLY in the charge
combination and leaves everything else finite.

(An older version of this analysis dropped the 1/k^2 piece entirely -- the
"OZ" rows in S0_MgCl2NaCl_matrix_allsplits_*.csv.  Without it the fit has to
imitate the blow-up by making A enormous, which makes it numerically fragile;
about 10 of 76 rows never converged.  That is why this script does not offer
that option.)

Once A is fitted, S(0) is obtained by removing the charge direction (which is
pinned at zero) and keeping what is left.  A side effect: the reported S0_ZZ is
always zero, by construction.  Do not read anything into it -- it is only a
check that the code did what it was supposed to.

Two versions of the fit are written per row:

  "BC fixed kappa_D"   kappa is held at the value you can calculate directly
                       from the composition, volume and temperature.
  "BC fitted kappa_D"  kappa is left free for the fit to choose.

Use "BC fixed kappa_D".  It is stable, and it agrees with the old kcut=0.02
file on all 76 rows to 5 digits.  The free-kappa version is included for
comparison only, and the kappa it reports is not a real screening length -- it
is just soaking up curvature in the data.

Note that --kcut is a cut on k SQUARED, in units of cycles^2 per Angstrom^2.
The name is inherited from the old files and is misleading:

    --kcut 0.02   keeps points with k up to 0.141 cycles/A
    --kcut 0.01   keeps points with k up to 0.100 cycles/A

So halving this number does not halve the k range -- it shrinks it by about
1.41x.

The answer also depends on where you cut.  S0_NN moves by roughly 4% between
--kcut 0.02 and --kcut 0.01, which is much larger than the scatter between time
blocks (under 1%).  So the error bar you get from the four blocks is not the
whole uncertainty.  If you quote S(0) anywhere, run a few cuts and report the
spread.

To run:

    python3 fit_S0.py --kcut 0.02
    python3 fit_S0.py --kcut 0.01

Writes two files per run:
  S0_kcut<value>.csv           one row per composition / block / version
  S0_kcut<value>_summary.csv   one row per composition, averaged over blocks
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent

SK_ROOT = HERE / "sk_output"

TEMPERATURE_K = 1200.0


SPECIES = ("Mg", "Cl", "Na")
CHARGES = np.array([2.0, -1.0, 1.0])


E2_EV_A = 14.399645478425668
KB_EV_K = 8.617333262e-5


CYCLES2_TO_ANGULAR2 = (2.0 * np.pi) ** 2


START_FACTORS = (1.0, 0.75, 0.5, 0.3)


MAX_NFEV = 20000


IJ = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
COLUMNS: dict[str, int] = {}
PAIR_INDEX: dict[str, tuple[int, int]] = {}


def formula_label() -> str:
    """The varied salt's formula, e.g. MgCl2 or LiCl.

    The anion subscript is the cation charge, so it cannot be read off the
    species names alone.
    """
    q = int(round(abs(CHARGES[0]) / abs(CHARGES[1])))
    return SPECIES[0] + SPECIES[1] + ("" if q == 1 else str(q))


def set_species(names, charges) -> None:
    """Point the whole script at a different three-species system."""
    global SPECIES, CHARGES, COLUMNS, PAIR_INDEX
    SPECIES = tuple(names)
    CHARGES = np.asarray(charges, dtype=float)
    COLUMNS, PAIR_INDEX = {}, {}
    for n, (i, j) in enumerate(IJ):
        label = SPECIES[i] + SPECIES[j]
        COLUMNS[label] = 1 + 2 * n
        PAIR_INDEX[label] = (i, j)


set_species(SPECIES, CHARGES)


TRI = np.tril_indices(3)
N_TRI = 6


def load_sk(path: Path):
    """Read one .dat file.

    Returns the k^2 values, the 3x3 tables, how many atoms of each species
    there are, and the box edge length.
    """
    raw = np.loadtxt(path)
    k2 = raw[:, 0]


    zero_row = np.isclose(k2, 0.0)
    counts = np.array([raw[zero_row, COLUMNS[f"{s}{s}"]][0] for s in SPECIES])


    keep = k2 > 0.0
    k2 = k2[keep]
    raw = raw[keep]

    matrices = np.empty((len(k2), 3, 3))
    for name, (i, j) in PAIR_INDEX.items():
        matrices[:, i, j] = raw[:, COLUMNS[name]]
        matrices[:, j, i] = raw[:, COLUMNS[name]]


    box_l = 1.0 / math.sqrt(float(k2.min()))
    return k2, matrices, counts, box_l


def split_files(tag: str) -> dict[int, Path]:
    """The files for one composition.

    Block 0 is the average over the whole time window.  Blocks 1, 2, ... are
    shorter chunks of it, and their scatter is what the error bar comes from.
    However many numbered files are present is however many blocks we use, so
    this works both for the original 3-block data and for the 4-block data made
    with 50 ps blocks.
    """
    folder = SK_ROOT / tag
    files = {0: folder / f"{tag}-allSk.dat"}
    i = 1
    while (folder / f"{tag}-{i}-allSk.dat").is_file():
        files[i] = folder / f"{tag}-{i}-allSk.dat"
        i += 1
    return files


def charge_weights(counts: np.ndarray) -> np.ndarray:
    """The combination that picks out charge.

    The curves in the .dat file are scaled by sqrt(N_i * N_j), so the charge
    combination carries a sqrt(concentration) factor as well as the charge.
    """
    fractions = counts / counts.sum()
    return np.sqrt(fractions) * CHARGES


def number_weights(counts: np.ndarray) -> np.ndarray:
    """The combination that gives S0_NN."""
    return np.sqrt(counts / counts.sum())


def charge_part(w: np.ndarray) -> np.ndarray:
    """The factor that confines the 1/k^2 blow-up to the charge direction."""
    return np.outer(w, w) / float(w @ w)


def debye_kappa(counts: np.ndarray, box_l: float, temperature: float) -> float:
    """kappa calculated directly from composition, volume and temperature.

    No fitting involved.  This is the textbook Debye value.
    """
    volume = box_l**3
    sum_nq2 = float(counts @ CHARGES**2)
    return math.sqrt(
        4.0 * math.pi * E2_EV_A * sum_nq2 / (volume * KB_EV_K * temperature)
    )


def pack(matrix: np.ndarray) -> np.ndarray:
    """3x3 symmetric table -> the 6 independent numbers."""
    return matrix[TRI]


def unpack(values: np.ndarray) -> np.ndarray:
    """The 6 independent numbers -> 3x3 symmetric table."""
    m = np.zeros((3, 3))
    m[TRI] = values
    return m + np.tril(m, -1).T


def safe_inv(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix, rcond=1.0e-12)


def first_guess_for_A(k2: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """A rough starting value for A.

    Take the reciprocal of the measured tables, fit a straight line against
    k^2, and use the intercept.
    """
    inverses = np.stack([safe_inv(m) for m in matrices])
    design = np.column_stack((np.ones_like(k2), k2))
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(i + 1):
            coeff, *_ = np.linalg.lstsq(design, inverses[:, i, j], rcond=None)
            out[i, j] = out[j, i] = coeff[0]
    return out


def rescale(matrices: np.ndarray):
    """Put the six curves on a comparable footing.

    MgMg is around 0.8 while ClCl is around 0.06.  If we did not divide each
    curve by its own spread, the fit would chase MgMg and ignore the rest.
    """
    target = matrices[:, TRI[0], TRI[1]]
    scale = np.std(target, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    return target, scale


def s0_from_A(a_matrix: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Turn the fitted A into S(0).

    At k = 0 the charge direction is pinned to zero, so we strip it out and
    keep what remains.
    """
    a_inv = safe_inv(a_matrix)
    denom = float(w @ a_inv @ w)
    if not np.isfinite(denom) or abs(denom) < 1.0e-14:
        return np.full_like(a_matrix, np.nan)
    return a_inv - ((a_inv @ w[:, None]) @ (w[None, :] @ a_inv)) / denom


def fit_one(k2_angular, matrices, kappa_calc, w, free_kappa, kappa_start=None):
    """Run the fit once.

    kappa_calc    the calculated kappa.  Even when kappa is free, this stays
                  the reference point that kappa is nudged toward and bounded
                  around.
    free_kappa    True  -> the fit chooses kappa (13 unknowns)
                  False -> kappa is held at kappa_calc (12 unknowns)
    kappa_start   only changes where the optimiser begins.  It does NOT change
                  what is being minimised, which is the whole point: it means
                  the "cost" values from different starts mean the same thing
                  and can be compared.
    """
    target, scale = rescale(matrices)
    cp = charge_part(w)
    start = kappa_calc if kappa_start is None else kappa_start


    a0 = first_guess_for_A(k2_angular, matrices) - float(
        np.median(start**2 / k2_angular)
    ) * cp
    l0 = np.zeros((3, 3))
    log_kappa_calc = math.log(kappa_calc)

    def build(params):
        """Assemble A + k^2 L + charge_part * kappa^2/k^2 at every k."""
        a = unpack(params[:N_TRI])
        l = unpack(params[N_TRI: 2 * N_TRI])
        kappa = float(math.exp(params[2 * N_TRI])) if free_kappa else kappa_calc
        stack = (
            a[None, :, :]
            + k2_angular[:, None, None] * l[None, :, :]
            + (kappa**2 / k2_angular)[:, None, None] * cp[None, :, :]
        )
        return a, l, kappa, stack

    def residuals(params):
        """How far the model is from the data, plus two gentle nudges."""
        _, l, _, stack = build(params)
        try:
            model = np.linalg.inv(stack)
        except np.linalg.LinAlgError:

            return np.full(target.size + N_TRI + int(free_kappa), 1.0e8)

        misfit = (model[:, TRI[0], TRI[1]] - target) / scale


        nudges = [pack(l - l0) / 10000.0]


        if free_kappa:
            nudges.append(np.array([(params[-1] - log_kappa_calc) / 10.0]))


        return np.r_[misfit.ravel(), np.concatenate(nudges)]


    x0 = np.r_[pack(a0), pack(l0)]
    lo = np.full(len(x0), -np.inf)
    hi = np.full(len(x0), np.inf)
    if free_kappa:
        x0 = np.r_[x0, math.log(start)]

        lo = np.r_[lo, log_kappa_calc - math.log(10.0)]
        hi = np.r_[hi, log_kappa_calc + math.log(10.0)]

    result = least_squares(
        residuals,
        x0,
        bounds=(lo, hi),


        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=MAX_NFEV,
        ftol=1.0e-7,
        xtol=1.0e-7,
        gtol=1.0e-7,
    )

    a, l_matrix, kappa, _ = build(result.x)
    return {


        "A": a,
        "L": l_matrix,


        "converged": bool(result.status > 0),
        "status": int(result.status),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "kappa": float(kappa),
        "s0": s0_from_A(a, w),
        "rmse": float(np.sqrt(np.mean(residuals(result.x) ** 2))),
    }


def select_start(tries, number_weight, tol=0.20):
    """Reconcile the multi-start results.

    Normally every start lands in the same basin and the lowest cost is simply
    the best-converged one.  Occasionally a start lands somewhere with a very
    different S(0) but a slightly lower cost -- the low-k data cannot always
    distinguish the two, and bootstrapping the k-shells on one such case put the
    cost preference at only 1.5 sigma with a 95% interval spanning zero.  Taking
    the lower cost there means fitting noise in a direction the data does not
    constrain, so when the starts split into distinct basins the most-populated
    one is used instead and the row is marked.

    Returns (start factor, fit, number of basins, rule used, alternative).
    """
    def s0nn(o):
        return float(number_weight @ o["s0"] @ number_weight)

    basins: list[list] = []
    for f, o in tries:
        v = s0nn(o)
        for b in basins:
            if abs(v - s0nn(b[0][1])) <= tol * abs(s0nn(b[0][1])):
                b.append((f, o))
                break
        else:
            basins.append([(f, o)])

    cheapest = min(tries, key=lambda t: t[1]["cost"])
    if len(basins) == 1:
        return cheapest[0], cheapest[1], 1, "lowest-cost", None


    basins.sort(key=lambda b: (-len(b), min(o["cost"] for _, o in b)))
    factor, out = min(basins[0], key=lambda t: t[1]["cost"])
    alt = None
    if cheapest[1] is not out:
        alt = (s0nn(cheapest[1]), cheapest[1]["cost"])
    return factor, out, len(basins), "modal-basin", alt


def run(kcut: float) -> list[dict]:
    tags = sorted(
        (p.name for p in SK_ROOT.iterdir() if p.is_dir()), key=lambda t: int(''.join(ch for ch in t if ch.isdigit()))
    )
    rows = []
    for tag in tags:
        for split, path in split_files(tag).items():
            k2, matrices, counts, box_l = load_sk(path)


            keep = k2 <= kcut + 1.0e-14
            k2_fit = k2[keep] * CYCLES2_TO_ANGULAR2
            data = matrices[keep]

            w = charge_weights(counts)
            nw = number_weights(counts)
            kappa_calc = debye_kappa(counts, box_l, TEMPERATURE_K)
            n_mg, n_cl, n_na = (int(round(c)) for c in counts)

            for label, free_kappa in (
                ("BC fixed kappa_D", False),
                ("BC fitted kappa_D", True),
            ):
                if free_kappa:


                    tries = [
                        (f, fit_one(k2_fit, data, kappa_calc, w, True, f * kappa_calc))
                        for f in START_FACTORS
                    ]
                    good = [t for t in tries if t[1]["converged"]]
                    n_good = len(good)
                    factor, out, n_basins, rule, alt = select_start(good or tries, nw)
                else:

                    factor, out = 1.0, fit_one(k2_fit, data, kappa_calc, w, False)
                    n_good = int(out["converged"])
                    n_basins, rule, alt = 1, "single-fit", None

                s0 = out["s0"]
                rows.append({
                    "tag": tag,
                    f"x_{formula_label()}": n_mg / (n_mg + n_na),
                    "split": split,
                    "method": label,
                    "k2cut_cycles2_per_A2": kcut,
                    "k_max_cycles_per_A": math.sqrt(kcut),
                    f"n_{SPECIES[0]}": n_mg, f"n_{SPECIES[1]}": n_cl,
                    f"n_{SPECIES[2]}": n_na,
                    "box_L": box_l,
                    "n_fit_points": int(keep.sum()),

                    "fit_converged": int(out["converged"]),
                    "fit_status": out["status"],
                    "fit_nfev": out["nfev"],

                    "fit_cost": out["cost"],
                    "fit_rmse": out["rmse"],
                    "n_starts": len(START_FACTORS) if free_kappa else 1,
                    "n_starts_converged": n_good,
                    "best_start_factor": factor,


                    "n_basins": n_basins,
                    "selection_rule": rule,
                    "alt_S0_NN": alt[0] if alt else float("nan"),
                    "alt_cost": alt[1] if alt else float("nan"),
                    "kappa_calculated": kappa_calc,


                    "kappa_used": out["kappa"],
                    **{f"S0_{lab}": s0[i, j]
                       for lab, (i, j) in PAIR_INDEX.items()},

                    "S0_NN": float(nw @ s0 @ nw),

                    "S0_ZZ": float(w @ s0 @ w / (w @ w)),
                })
            print(f"{tag} split {split}: done", flush=True)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """One row per composition: the best value plus an error bar.

    Block 0 is the fit to the whole time window, so it uses all the data and is
    the best single estimate.  Blocks 1, 2, ... are shorter chunks of that same
    window, so they are not extra data -- their only job is to tell us how much
    the answer moves around, which is the error bar.

    So: the value comes from block 0, and the error bar is the scatter of
    blocks 1..N divided by sqrt(N).  The average of blocks 1..N is written out
    too, as "<col>_blockmean"; it should agree with block 0 within the error
    bar, and it is worth a look if it does not.

    The error bar covers statistical noise ONLY.  It does not cover the choice
    of where to cut in k, which moves S0_NN by more (see the note at the top).
    """
    out = []
    tags = sorted({r["tag"] for r in rows}, key=lambda t: int(''.join(ch for ch in t if ch.isdigit())))
    cols = ("S0_NN",) + tuple(f"S0_{lab}" for lab in PAIR_INDEX)
    for tag in tags:
        for label in ("BC fixed kappa_D", "BC fitted kappa_D"):
            sel = [r for r in rows if r["tag"] == tag and r["method"] == label]
            whole = next(r for r in sel if r["split"] == 0)
            blocks = [r for r in sel if r["split"] != 0]
            row = {
                "tag": tag,
                f"x_{formula_label()}": whole[f"x_{formula_label()}"],
                "method": label,
                "k2cut_cycles2_per_A2": whole["k2cut_cycles2_per_A2"],
                "n_blocks": len(blocks),
                "all_converged": int(all(r["fit_converged"] for r in sel)),
            }
            for col in cols:
                v = np.array([r[col] for r in blocks])
                row[col] = whole[col]
                row[col + "_err"] = (
                    float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
                )
                row[col + "_blockmean"] = float(v.mean()) if len(v) else float("nan")
            out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kcut", type=float, default=0.02,
        help="cut on k SQUARED, in cycles^2/A^2 (0.02 -> k up to 0.141 cycles/A)",
    )
    p.add_argument("--species", type=str, default=",".join(SPECIES),
                   help="three species in .dat column order, e.g. Li,Cl,Na")
    p.add_argument("--charges", type=str, default="2,-1,1",
                   help="their charges in the same order, e.g. 1,-1,1")
    p.add_argument("--sk-dir", type=Path, default=None,
                   help="directory of S(k) .dat files (default: sk_output)")
    p.add_argument("--label", type=str, default="",
                   help="extra text added to the output file names")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to write the CSVs (default: next to this script)")
    args = p.parse_args()

    set_species(tuple(x.strip() for x in args.species.split(",")),
                [float(x) for x in args.charges.split(",")])
    print(f"species {SPECIES}, charges {list(CHARGES)}")

    global SK_ROOT
    if args.sk_dir is not None:
        SK_ROOT = args.sk_dir.expanduser().resolve()
    print(f"reading S(k) from {SK_ROOT}")

    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else HERE)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = run(args.kcut)
    tag = f"{args.kcut:g}" + (f"_{args.label}" if args.label else "")
    detail = out_dir / f"S0_kcut{tag}.csv"
    summary = out_dir / f"S0_kcut{tag}_summary.csv"
    write_csv(detail, rows)
    write_csv(summary, summarise(rows))

    bad = [r for r in rows if not r["fit_converged"]]
    print(f"\nwrote {detail}   ({len(rows)} rows)")
    print(f"wrote {summary}   ({len(rows) // 8} compositions x 2 methods)")
    print(f"did not converge: {len(bad)} rows")
    for r in bad:
        print(f"   {r['tag']} split {r['split']} {r['method']} status={r['fit_status']}")


if __name__ == "__main__":
    main()
