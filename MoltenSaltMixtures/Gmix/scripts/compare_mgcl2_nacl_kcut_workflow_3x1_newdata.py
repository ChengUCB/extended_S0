"""Compare charge-neutral NaCl-MgCl2 workflows for two reciprocal-space cutoffs.

Same analysis as compare_mgcl2_nacl_kcut_workflow_3x1.py, re-pointed at the S(0)
values recomputed from the trajectories (all 19 compositions, 50 ps to the end of
each run, 50 ps blocks).  Three things had to change for that data:

  * the k-cut column is now called ``k2cut_cycles2_per_A2`` (the old name
    ``kcut`` hid the fact that the value is a cut on k SQUARED);
  * there are no longer exactly three blocks.  Each composition has 8-15 blocks,
    because the trajectories are of different lengths, so the block index is
    taken from the data instead of being hard-coded;
  * blocks are used only up to the largest index every composition has in
    common (N_common), since the per-block GPs are fitted across compositions.

Split 0 is still the whole-window fit and still supplies the central curve, so
the extra blocks only improve the error estimate, never the central values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from plot_mgcl2_nacl_workflow_3x1 import (
    TARGET_T_K,
    ebar,
    nist_experimental_thermodynamics,
)


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/Users/xiaoyuwang/Project/multi_S0_GP_test")
S0_DIR = Path("/Users/xiaoyuwang/Downloads/MoltenSaltData/MgCl2_NaCl")

# The two cutoffs to compare.  The first is drawn dashed/red, the second
# solid/blue.  Override on the command line, e.g. --cuts 0.02 0.015.
CUTS = (0.02, 0.01)

# Which S(0) tables to read.
#   "mine" - recomputed from the trajectories (19 compositions, 50 ps to the end
#            of each run, 8-15 blocks).  Cut column: k2cut_cycles2_per_A2.
#   "rs"   - the values RS circulated (19 compositions, 50-250 ps, 3 blocks).
#            Cut column: kcut.  Only 0.01 and 0.02 exist.
SOURCE = "mine"
# (filename template, k-cut column, convergence-flag column)
# Note on the RS convergence flag: it reads 1 on all 228 rows even though the
# same settings leave 10 rows unconverged when rerun, so it is not a reliable
# gate.  It is checked here only because the "mine" tables carry a real one.
SOURCE_SPEC = {
    "mine": ("S0_kcut{kcut:g}_full50ps.csv", "k2cut_cycles2_per_A2", "fit_converged"),
    "last200": ("S0_kcut{kcut:g}_last200ps.csv", "k2cut_cycles2_per_A2", "fit_converged"),
    "rs": ("S0_MgCl2NaCl_matrix_allsplits_kcut{kcut:g}.csv", "kcut", "fit_success"),
}


def input_path(kcut: float) -> Path:
    """Locate the S(0) table for one cut.

    My tables are filed per-cut under my_results/; the RS tables sit flat in the
    system directory.  Search rather than assume a layout.
    """
    name = SOURCE_SPEC[SOURCE][0].format(kcut=kcut)
    direct = S0_DIR / name
    if direct.is_file():
        return direct
    matches = sorted(S0_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {S0_DIR}")
    return matches[0]


INPUTS: dict[float, Path] = {}  # filled in by main() once CUTS is known
OUTPUT_DIR = HERE / "outputs" / "MgCl2_NaCl_charge_neutral_GP_newdata"


def output_pdf() -> Path:
    tag = ("_vs_".join(f"{k:g}" for k in reversed(CUTS)) if len(CUTS) > 1
           else f"{CUTS[0]:g}")
    suffix = "newdata" if SOURCE == "mine" else SOURCE
    return OUTPUT_DIR / f"MgCl2_NaCl_kcut{tag}_workflow_3x1_{suffix}.pdf"
CSV_METHOD = "BC fitted kappa_D"  # Internal CSV selector; not a physical label.

KB_IN_EV_PER_K = 8.617333262145e-5
KB_T_EV = KB_IN_EV_PER_K * TARGET_T_K
EV_TO_KJ_PER_MOL = 96.48533212331002
COMPONENTS = ["Mg", "Cl", "Na"]
INDEPENDENT = ["Mg"]
CHARGES = {"Mg": 2.0, "Cl": -1.0, "Na": 1.0}
COLUMN_MAP = {
    "x_Mg": "x_Mg",
    "x_Cl": "x_Cl",
    "x_Na": "x_Na",
    "S_MgMg": "S0_MgMg",
    "S_MgCl": "S0_MgCl",
    "S_MgNa": "S0_MgNa",
    "S_ClCl": "S0_ClCl",
    "S_ClNa": "S0_ClNa",
    "S_NaNa": "S0_NaNa",
}

sys.path[:0] = [
    str(SOURCE_ROOT / "GPR_grad"),
    str(SOURCE_ROOT / "S0_multi"),
]
from gpr_grad import GradientGP, RBFKernelFunction  # noqa: E402
from szero import prepare_gp_gradient_data  # noqa: E402


def load_input(path: Path, expected_kcut: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "x_MgCl2",
        "split",
        "method",
        SOURCE_SPEC[SOURCE][1],
        SOURCE_SPEC[SOURCE][2],
        "n_Mg",
        "n_Cl",
        "n_Na",
        "S0_MgMg",
        "S0_MgCl",
        "S0_MgNa",
        "S0_ClCl",
        "S0_ClNa",
        "S0_NaNa",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    frame = frame.loc[
        frame["method"].eq(CSV_METHOD)
        & np.isclose(frame[SOURCE_SPEC[SOURCE][1]], expected_kcut)
    ].copy()
    if frame.empty:
        raise ValueError(f"{path.name}: no selected BC rows at kcut={expected_kcut}.")
    if 0 not in set(frame["split"].unique()):
        raise ValueError(f"{path.name}: split 0 (whole-window fit) is missing.")
    if not frame[SOURCE_SPEC[SOURCE][2]].astype(bool).all():
        raise ValueError(f"{path.name}: at least one selected S(0) fit did not converge.")

    n_ions = frame["n_Mg"] + frame["n_Cl"] + frame["n_Na"]
    frame["x_Mg"] = frame["n_Mg"] / n_ions
    frame["x_Cl"] = frame["n_Cl"] / n_ions
    frame["x_Na"] = frame["n_Na"] / n_ions
    charge = 2.0 * frame["x_Mg"] - frame["x_Cl"] + frame["x_Na"]
    if np.max(np.abs(charge)) > 1e-12:
        raise ValueError(f"{path.name}: ionic compositions are not charge neutral.")
    return frame


def convert_split(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("x_MgCl2").reset_index(drop=True)
    gp_data = prepare_gp_gradient_data(
        frame,
        independent_components=INDEPENDENT,
        column_map=COLUMN_MAP,
        components=COMPONENTS,
        charges=CHARGES,
        excess=True,
        kb_t=KB_T_EV,
    )
    gamma = np.column_stack([gp_data[name][1][:, 0] for name in COMPONENTS])
    y = frame["x_MgCl2"].to_numpy(float)
    dx_mg_dy = 2.0 / (2.0 + y) ** 2
    dmu_mgcl2_dy = (gamma[:, 0] + 2.0 * gamma[:, 1]) * dx_mg_dy
    dmu_nacl_dy = (gamma[:, 2] + gamma[:, 1]) * dx_mg_dy
    return pd.DataFrame(
        {
            "x_MgCl2": y,
            "dmu_MgCl2_ex_dy": dmu_mgcl2_dy,
            "dmu_NaCl_ex_dy": dmu_nacl_dy,
            "Gex_curvature": dmu_mgcl2_dy - dmu_nacl_dy,
        }
    )


def trapz_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    matrix = np.zeros((len(x), len(x)), dtype=float)
    for i in range(1, len(x)):
        matrix[i] = matrix[i - 1]
        dx = x[i] - x[i - 1]
        matrix[i, i - 1] += 0.5 * dx
        matrix[i, i] += 0.5 * dx
    return matrix


def ideal_mixing(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    value = np.zeros_like(y)
    interior = (y > 0.0) & (y < 1.0)
    value[interior] = KB_T_EV * (
        y[interior] * np.log(y[interior])
        + (1.0 - y[interior]) * np.log(1.0 - y[interior])
    )
    return value


# Hyperparameters, set by hand rather than fitted (see the note below).
THETA_FIXED = 0.1        # RBF length scale, in mole-fraction units (dimensionless)
SIGMA_F = 0.0            # observation noise on the curvature values
SIGMA_G = 1e-4           # gradient-observation noise


def make_gp(theta: float, sigma_f: float, trainable_theta: bool) -> GradientGP:
    return GradientGP(
        kernel=RBFKernelFunction(
            theta=torch.tensor([theta]),
            trainable=trainable_theta,
        ),
        sigma_f=sigma_f,
        sigma_g=SIGMA_G,
        trainable_sigma_f=False,
        trainable_sigma_g=False,
        jitter=1e-8,
    )


# Why these are fixed and not fitted:
#
#   theta   Marginal-likelihood optimisation is well behaved for MgCl2-NaCl
#           (a clear interior maximum near 0.45, reached from every start) but
#           not for LiCl-NaCl, where the likelihood rises monotonically out to
#           theta = 100 because the block noise is as large as the composition
#           dependence.  A value reported from that optimisation would be an
#           artifact of the step budget, so theta is fixed for both systems and
#           the two panels stay comparable.
#
#   sigma_f Zero, i.e. the curvature values are treated as exact and the GP
#           interpolates them.  Only the jitter (1e-8) then regularises the
#           kernel, which takes cond(K) to about 1e7 -- one order of magnitude
#           of margin.  dG_min moves by under 0.006 kJ/mol relative to using the
#           measured block scatter as sigma_f, which is far inside the 0.03-0.05
#           statistical error, so the choice does not affect the reported
#           numbers.
#
#   sigma_g Has no effect with this call signature: the noise block it feeds is
#           sized by the number of gradient observations, and none are supplied
#           (fit is called with X_f and Y_f only).  Kept explicit for the record.


def fit_central_gp(
    central: pd.DataFrame, q_std: np.ndarray, n_steps: int = 400
) -> tuple[GradientGP, float, float, float]:
    y = central["x_MgCl2"].to_numpy(float)
    q = central["Gex_curvature"].to_numpy(float)
    offset = float(np.mean(q))
    # No marginal-likelihood optimisation: theta and sigma_f are fixed.  q_std
    # is still accepted so the call sites and the record of the measured block
    # scatter stay unchanged.
    theta, sigma_f = THETA_FIXED, SIGMA_F
    gp = make_gp(theta=theta, sigma_f=sigma_f, trainable_theta=False)
    gp.fit(X_f=torch.tensor(y[:, None]), Y_f=torch.tensor(q - offset))
    return gp, offset, theta, sigma_f


def fit_fixed_gp(
    frame: pd.DataFrame, theta: float, sigma_f: float
) -> tuple[GradientGP, float]:
    y = frame["x_MgCl2"].to_numpy(float)
    q = frame["Gex_curvature"].to_numpy(float)
    offset = float(np.mean(q))
    gp = make_gp(theta=theta, sigma_f=sigma_f, trainable_theta=False)
    gp.fit(X_f=torch.tensor(y[:, None]), Y_f=torch.tensor(q - offset))
    return gp, offset


def predict_and_integrate(
    gp: GradientGP,
    offset: float,
    y_grid: np.ndarray,
    return_posterior_std: bool = False,
) -> dict[str, np.ndarray]:
    prediction = gp.predict(
        torch.tensor(y_grid[:, None]), return_cov=return_posterior_std
    )
    if return_posterior_std:
        q_tensor, q_cov_tensor = prediction
        q_cov = q_cov_tensor.detach().cpu().numpy()
        q_cov = 0.5 * (q_cov + q_cov.T)
    else:
        q_tensor = prediction
        q_cov = None
    q = q_tensor.detach().cpu().numpy() + offset
    w = trapz_matrix(y_grid)
    ww = w @ w
    endpoint = np.zeros(len(y_grid))
    endpoint[-1] = 1.0
    boundary = np.eye(len(y_grid)) - np.outer(y_grid, endpoint)
    g_operator = boundary @ ww
    derivative_operator = w - np.outer(np.ones(len(y_grid)), endpoint @ ww)
    g_ex = g_operator @ q
    dg_ex = derivative_operator @ q
    result = {
        "q": q,
        "Gex": g_ex,
        "mu_MgCl2_ex": g_ex + (1.0 - y_grid) * dg_ex,
        "mu_NaCl_ex": g_ex - y_grid * dg_ex,
    }
    if q_cov is not None:
        mu_mg_operator = g_operator + (1.0 - y_grid)[:, None] * derivative_operator
        mu_na_operator = g_operator - y_grid[:, None] * derivative_operator

        def propagated_std(operator: np.ndarray) -> np.ndarray:
            covariance_times_operator = q_cov @ operator.T
            variance = np.einsum("ij,ji->i", operator, covariance_times_operator)
            return np.sqrt(np.clip(variance, 0.0, None))

        result.update(
            {
                "q_GP_std": np.sqrt(np.clip(np.diag(q_cov), 0.0, None)),
                "Gex_GP_std": propagated_std(g_operator),
                "mu_MgCl2_ex_GP_std": propagated_std(mu_mg_operator),
                "mu_NaCl_ex_GP_std": propagated_std(mu_na_operator),
            }
        )
    return result


def common_block_ids(raw: pd.DataFrame) -> list[int]:
    """Block indices that every composition has.

    The trajectories are of different lengths, so one composition may have 8
    blocks and another 15.  The per-block GPs are fitted across all compositions
    at once, so a block index is only usable if every composition has it.
    """
    per_tag = raw.loc[raw["split"] > 0].groupby("tag")["split"].max()
    n_common = int(per_tag.min())
    if n_common < 3:
        raise ValueError(f"only {n_common} blocks common to all compositions")
    return list(range(1, n_common + 1))


def process_cutoff(path: Path, kcut: float, y_grid: np.ndarray) -> dict:
    raw = load_input(path, kcut)
    block_ids = common_block_ids(raw)
    print(
        f"  kcut^2={kcut:g}: {raw['tag'].nunique()} compositions, "
        f"using blocks 1-{block_ids[-1]} "
        f"(dropping the tail of the longer runs so every composition "
        f"contributes to each block GP)"
    )
    splits = {
        split: convert_split(raw.loc[raw["split"].eq(split)])
        for split in [0] + block_ids
    }
    y_data = splits[0]["x_MgCl2"].to_numpy(float)
    for split in block_ids:
        if not np.allclose(splits[split]["x_MgCl2"], y_data, atol=1e-13):
            raise ValueError(f"Composition mismatch at kcut={kcut}, split={split}.")

    q_blocks = np.stack(
        [splits[split]["Gex_curvature"].to_numpy(float) for split in block_ids]
    )
    q_std = np.std(q_blocks, axis=0, ddof=1)
    integrand_blocks = np.stack(
        [
            y_data * splits[split]["dmu_MgCl2_ex_dy"].to_numpy(float)
            for split in block_ids
        ]
    )
    integrand = y_data * splits[0]["dmu_MgCl2_ex_dy"].to_numpy(float)
    integrand_std = np.std(integrand_blocks, axis=0, ddof=1)

    central_gp, central_offset, theta, sigma_f = fit_central_gp(splits[0], q_std)
    central_prediction = predict_and_integrate(
        central_gp,
        central_offset,
        y_grid,
        return_posterior_std=False,
    )

    block_predictions = []
    for split in block_ids:
        gp, offset = fit_fixed_gp(splits[split], theta, sigma_f)
        block_predictions.append(predict_and_integrate(gp, offset, y_grid))

    mean_keys = ("q", "Gex", "mu_MgCl2_ex", "mu_NaCl_ex")
    prediction_std = {
        key: np.std(
            np.stack([prediction[key] for prediction in block_predictions]),
            axis=0,
            ddof=1,
        )
        for key in mean_keys
    }
    return {
        "y_data": y_data,
        "integrand": integrand,
        "integrand_std": integrand_std,
        "central": central_prediction,
        "std": prediction_std,
        "theta": theta,
        "sigma_f": sigma_f,
        "n_blocks": len(block_ids),
        "n_compositions": int(raw["tag"].nunique()),
    }


def main() -> None:
    global CUTS, INPUTS, OUTPUT_PDF, SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=tuple(SOURCE_SPEC), default=SOURCE,
        help="which S(0) tables to read (default: mine)",
    )
    parser.add_argument(
        "--cuts", type=float, nargs="+", default=list(CUTS),
        metavar="KCUT",
        help="one or two k^2 cuts (default: 0.02 0.01)",
    )
    args = parser.parse_args()
    SOURCE = args.source
    CUTS = tuple(args.cuts)
    INPUTS = {kcut: input_path(kcut) for kcut in CUTS}
    OUTPUT_PDF = output_pdf()
    print(f"source={SOURCE}: k^2 cut(s) " + ", ".join(f"{c:g}" for c in CUTS))

    torch.set_default_dtype(torch.float64)
    np.random.seed(7)
    torch.manual_seed(7)

    y_grid = np.linspace(0.0, 1.0, 501)
    processed = {
        kcut: process_cutoff(path, kcut, y_grid)
        for kcut, path in INPUTS.items()
    }
    ideal_g = ideal_mixing(y_grid)
    interior = (y_grid > 0.0) & (y_grid < 1.0)
    ideal_mu_mg = np.full_like(y_grid, np.nan)
    ideal_mu_na = np.full_like(y_grid, np.nan)
    ideal_mu_mg[interior] = KB_T_EV * np.log(y_grid[interior])
    ideal_mu_na[interior] = KB_T_EV * np.log(1.0 - y_grid[interior])

    _st = [{"color": "#1764ab", "ls": "-", "marker": "o"},
           {"color": "#e12729", "ls": "--", "marker": "x"}]
    if len(CUTS) == 2:
        _st = [_st[1], _st[0]]
    styles = {c: _st[i] for i, c in enumerate(CUTS)}

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(3.45, 7.20),
        dpi=500,
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    ax1, ax2, ax3 = axes

    for kcut in CUTS:
        result = processed[kcut]
        style = styles[kcut]
        ebar(
            ax1,
            result["y_data"],
            result["integrand"] * EV_TO_KJ_PER_MOL,
            result["integrand_std"] * EV_TO_KJ_PER_MOL,
            color=style["color"],
            fmt=style["marker"],
            markerfacecolor="white" if style["marker"] == "o" else None,
            markersize=3.7,
            markeredgewidth=0.9,
            linestyle="none",
            clip_on=False,
        )
        integrand_gp = y_grid * (1.0 - y_grid) * result["central"]["q"]
        ax1.plot(
            y_grid,
            integrand_gp * EV_TO_KJ_PER_MOL,
            color=style["color"],
            ls=style["ls"],
            lw=1.25,
        )

    cutoff_handles = [
        Line2D(
            [0],
            [0],
            color=styles[kcut]["color"],
            ls=styles[kcut]["ls"],
            marker=styles[kcut]["marker"],
            markerfacecolor="white" if styles[kcut]["marker"] == "o" else None,
            lw=1.2,
            label=f"{kcut:g}",
        )
        for kcut in CUTS
    ]
    ax1.legend(
        handles=cutoff_handles,
        loc="lower center",
        frameon=False,
        ncol=2,
        title=r"$k_{cut}^2$ in $4\pi^2\,\AA^{-2}$",
        title_fontsize=6.5,
    )
    ax1.set_ylabel(
        r"${\partial \mu^{\rm ex}_{\rm MgCl_2}}/{\partial \ln x_{\rm MgCl_2}}$"
        "\n[kJ/mol]"
    )
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax2.plot(
        y_grid[interior],
        ideal_mu_mg[interior] * EV_TO_KJ_PER_MOL,
        color="gray",
        ls=(0, (1, 3)),
        alpha=0.55,
        lw=1.0,
    )
    ax2.plot(
        y_grid[interior],
        ideal_mu_na[interior] * EV_TO_KJ_PER_MOL,
        color="gray",
        ls=(0, (1, 3)),
        alpha=0.55,
        lw=1.0,
    )
    for kcut in CUTS:
        result = processed[kcut]
        style = styles[kcut]
        mu_mg = result["central"]["mu_MgCl2_ex"] + ideal_mu_mg
        mu_na = result["central"]["mu_NaCl_ex"] + ideal_mu_na
        ax2.plot(
            y_grid[interior],
            mu_mg[interior] * EV_TO_KJ_PER_MOL,
            color=style["color"],
            ls=style["ls"],
            lw=1.15,
        )
        ax2.plot(
            y_grid[interior],
            mu_na[interior] * EV_TO_KJ_PER_MOL,
            color=style["color"],
            ls=style["ls"],
            lw=1.15,
        )
        error_x = result["y_data"]
        ebar(
            ax2,
            error_x,
            np.interp(error_x, y_grid, mu_mg) * EV_TO_KJ_PER_MOL,
            np.interp(error_x, y_grid, result["std"]["mu_MgCl2_ex"])
            * EV_TO_KJ_PER_MOL,
            color=style["color"],
            fmt="p",
            fillstyle="none",
            markersize=3.7,
            linestyle="none",
            clip_on=False,
        )
        ebar(
            ax2,
            error_x,
            np.interp(error_x, y_grid, mu_na) * EV_TO_KJ_PER_MOL,
            np.interp(error_x, y_grid, result["std"]["mu_NaCl_ex"])
            * EV_TO_KJ_PER_MOL,
            color=style["color"],
            fmt="<",
            fillstyle="none",
            markersize=3.7,
            linestyle="none",
            clip_on=False,
        )

    species_handles = [
        Line2D([0], [0], color="black", marker="p", fillstyle="none", lw=1.0,
               label=r"$\mu_{\rm MgCl_2}$"),
        Line2D([0], [0], color="black", marker="<", fillstyle="none", lw=1.0,
               label=r"$\mu_{\rm NaCl}$"),
        Line2D([0], [0], color="gray", ls=(0, (1, 3)), lw=1.0, label="Ideal"),
    ]
    ax2.legend(
        handles=cutoff_handles + species_handles,
        loc="lower center",
        frameon=False,
        ncol=2,
        handlelength=1.8,
        title=r"$k_{cut}^2$ in $4\pi^2\,\AA^{-2}$",
        title_fontsize=6.5,
    )
    ax2.set_ylabel(r"$\mu$ [kJ/mol]")
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.set_ylim(bottom=-100.0, top=3.0)

    ax3.plot(
        y_grid,
        ideal_g * EV_TO_KJ_PER_MOL,
        color="gray",
        ls=(0, (1, 3)),
        alpha=0.70,
        lw=1.0,
        label="Ideal",
    )
    dense_rows = {"x_MgCl2": y_grid}
    for kcut in CUTS:
        result = processed[kcut]
        style = styles[kcut]
        gmix = result["central"]["Gex"] + ideal_g
        gmix_std = result["std"]["Gex"]
        label = rf"GP ({kcut:g})"
        ax3.plot(
            y_grid,
            gmix * EV_TO_KJ_PER_MOL,
            color=style["color"],
            ls=style["ls"],
            lw=1.3,
            label=label,
        )
        error_x = result["y_data"]
        ebar(
            ax3,
            error_x,
            np.interp(error_x, y_grid, gmix) * EV_TO_KJ_PER_MOL,
            np.interp(error_x, y_grid, gmix_std) * EV_TO_KJ_PER_MOL,
            color=style["color"],
            fmt=style["marker"],
            markerfacecolor="white" if style["marker"] == "o" else None,
            markersize=3.5,
            linestyle="none",
            clip_on=False,
        )
        prefix = f"kcut_{kcut:g}".replace(".", "p")
        dense_rows[f"{prefix}__Gmix_kJ_per_mol"] = gmix * EV_TO_KJ_PER_MOL
        dense_rows[f"{prefix}__Gmix_split_std_kJ_per_mol"] = (
            gmix_std * EV_TO_KJ_PER_MOL
        )
        dense_rows[f"{prefix}__integrand_GP_kJ_per_mol"] = (
            y_grid * (1.0 - y_grid) * result["central"]["q"]
            * EV_TO_KJ_PER_MOL
        )
        dense_rows[f"{prefix}__integrand_GP_split_std_kJ_per_mol"] = (
            y_grid * (1.0 - y_grid) * result["std"]["q"]
            * EV_TO_KJ_PER_MOL
        )
        dense_rows[f"{prefix}__mu_MgCl2_mix_kJ_per_mol"] = (
            result["central"]["mu_MgCl2_ex"] + ideal_mu_mg
        ) * EV_TO_KJ_PER_MOL
        dense_rows[f"{prefix}__mu_MgCl2_split_std_kJ_per_mol"] = (
            result["std"]["mu_MgCl2_ex"] * EV_TO_KJ_PER_MOL
        )
        dense_rows[f"{prefix}__mu_NaCl_mix_kJ_per_mol"] = (
            result["central"]["mu_NaCl_ex"] + ideal_mu_na
        ) * EV_TO_KJ_PER_MOL
        dense_rows[f"{prefix}__mu_NaCl_split_std_kJ_per_mol"] = (
            result["std"]["mu_NaCl_ex"] * EV_TO_KJ_PER_MOL
        )

    experimental = nist_experimental_thermodynamics(y_grid)
    experimental_styles = (
        ("Gmix_kJ_per_mol", r"Exp. $\Delta G$", "#111111", "-"),
        ("Hmix_kJ_per_mol", r"Exp. $\Delta H$", "#5c5c5c", "-."),
        (
            "minus_T_Smix_kJ_per_mol",
            r"Exp. $-T\Delta S$",
            "#e69f00",
            (0, (4, 1.5)),
        ),
    )
    for column, label, color, linestyle in experimental_styles:
        ax3.plot(
            y_grid,
            experimental[column],
            color=color,
            ls=linestyle,
            lw=1.15,
            label=label,
        )
        dense_rows[f"experimental__{column}"] = experimental[column].to_numpy()

    ax3.set_xlabel(r"$x_{\rm MgCl_2}$")
    ax3.set_ylabel(r"$\Delta G^{mix}$ [kJ/mol]")
    ax3.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax3.legend(
        loc="upper center",
        frameon=False,
        title=r"$k_{cut}^2$ in $4\pi^2\,\AA^{-2}$ (GP curves)",
        title_fontsize=5.9,
        handlelength=1.55,
        handletextpad=0.45,
        columnspacing=0.75,
        labelspacing=0.35,
        borderaxespad=0.25,
        fontsize=5.9,
        ncol=3,
    )

    for label, axis in zip("abc", axes):
        axis.set_xlim(0.0, 1.0)
        axis.yaxis.set_label_coords(-0.20, 0.5)
        axis.text(
            -0.20,
            1.03,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="bottom",
            ha="left",
            clip_on=False,
            fontsize=9,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_png = OUTPUT_PDF.with_suffix(".png")
    output_csv = OUTPUT_PDF.with_suffix(".csv")
    output_gp_band_pdf = OUTPUT_PDF.with_name(
        OUTPUT_PDF.stem + "_GP_errorband.pdf"
    )
    output_gp_band_png = output_gp_band_pdf.with_suffix(".png")
    pd.DataFrame(dense_rows).to_csv(output_csv, index=False)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.08, top=0.985, hspace=0.18)

    # Version 1: split-to-split variation shown as markers with error bars.
    fig.savefig(OUTPUT_PDF, dpi=500, bbox_inches="tight")
    fig.savefig(output_png, dpi=500, bbox_inches="tight")

    # Version 2: variation among the three split-GP predictions shown as
    # central GP mean +/- one sample standard deviation bands.
    # Remove split-errorbar containers while retaining all mean curves and legends.
    for axis in axes:
        for container in list(axis.containers):
            if isinstance(container, ErrorbarContainer):
                container.remove()

    for kcut in CUTS:
        result = processed[kcut]
        style = styles[kcut]
        color = style["color"]
        alpha = 0.15

        integrand_mean = y_grid * (1.0 - y_grid) * result["central"]["q"]
        integrand_std = (
            y_grid * (1.0 - y_grid) * result["std"]["q"]
        )
        ax1.fill_between(
            y_grid,
            (integrand_mean - integrand_std) * EV_TO_KJ_PER_MOL,
            (integrand_mean + integrand_std) * EV_TO_KJ_PER_MOL,
            color=color,
            alpha=alpha,
            linewidth=0,
        )
        ax1.plot(
            result["y_data"],
            result["integrand"] * EV_TO_KJ_PER_MOL,
            color=color,
            marker=style["marker"],
            markerfacecolor="white" if style["marker"] == "o" else None,
            markersize=3.7,
            markeredgewidth=0.9,
            linestyle="none",
            clip_on=False,
        )

        mu_mg = result["central"]["mu_MgCl2_ex"] + ideal_mu_mg
        mu_na = result["central"]["mu_NaCl_ex"] + ideal_mu_na
        mu_mg_std = result["std"]["mu_MgCl2_ex"]
        mu_na_std = result["std"]["mu_NaCl_ex"]
        ax2.fill_between(
            y_grid[interior],
            (mu_mg[interior] - mu_mg_std[interior]) * EV_TO_KJ_PER_MOL,
            (mu_mg[interior] + mu_mg_std[interior]) * EV_TO_KJ_PER_MOL,
            color=color,
            alpha=alpha,
            linewidth=0,
        )
        ax2.fill_between(
            y_grid[interior],
            (mu_na[interior] - mu_na_std[interior]) * EV_TO_KJ_PER_MOL,
            (mu_na[interior] + mu_na_std[interior]) * EV_TO_KJ_PER_MOL,
            color=color,
            alpha=alpha,
            linewidth=0,
        )

        gmix = result["central"]["Gex"] + ideal_g
        gmix_std = result["std"]["Gex"]
        ax3.fill_between(
            y_grid,
            (gmix - gmix_std) * EV_TO_KJ_PER_MOL,
            (gmix + gmix_std) * EV_TO_KJ_PER_MOL,
            color=color,
            alpha=alpha,
            linewidth=0,
        )

    ax3.text(
        0.98,
        0.04,
        rf"Shading: {processed[CUTS[0]]['n_blocks']}-block SD",
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        color="0.35",
    )
    ax1.set_ylim(-1.5, 31.5)
    fig.savefig(output_gp_band_pdf, dpi=500, bbox_inches="tight")
    fig.savefig(output_gp_band_png, dpi=500, bbox_inches="tight")
    plt.close(fig)

    print(OUTPUT_PDF)
    print(output_png)
    print(output_csv)
    print(output_gp_band_pdf)
    print(output_gp_band_png)
    for kcut in CUTS:
        print(
            f"kcut^2={kcut:g} (4pi^2 A^-2): "
            f"theta={processed[kcut]['theta']:.6f}, "
            f"sigma_f={processed[kcut]['sigma_f']:.6f} eV/f.u."
        )


if __name__ == "__main__":
    main()
