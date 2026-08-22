"""Compute LiCl-NaCl mixing thermodynamics from the retained k^2-cutoff 0.02 S(0) data.

The central curve uses the complete final 200 ps window. Four 50 ps blocks
provide the statistical uncertainty.
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

from plot_mgcl2_nacl_workflow_3x1 import TARGET_T_K, ebar


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SOURCE_ROOT = REPO_ROOT / "external" / "multi_S0_GP_test"
S0_DIR = REPO_ROOT / "MD_Sk_results"


CUTS = (0.02,)


SOURCE = "repo"


SOURCE_SPEC = {
    "repo": ("S0_k2cut{kcut:g}.csv", "k2cut_cycles2_per_A2", "fit_converged"),
    "mine": ("S0_kcut{kcut:g}_full50ps.csv", "k2cut_cycles2_per_A2", "fit_converged"),
    "rs": ("S0_NaClLiCl_matrix_allsplits_kcut{kcut:g}.csv", "kcut", "fit_success"),
    "last200": ("S0_kcut{kcut:g}_last200ps.csv", "k2cut_cycles2_per_A2", "fit_converged"),
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


INPUTS: dict[float, Path] = {}
OUTPUT_DIR = HERE / "outputs" / "LiCl_NaCl_charge_neutral_GP"


def output_pdf() -> Path:
    tag = "_vs_".join(f"{k:g}" for k in reversed(CUTS)) if len(CUTS) > 1 \
        else f"{CUTS[0]:g}"
    suffix = "newdata" if SOURCE == "mine" else SOURCE
    return OUTPUT_DIR / f"LiCl_NaCl_kcut{tag}_workflow_3x1_{suffix}.pdf"

KB_IN_EV_PER_K = 8.617333262145e-5
KB_T_EV = KB_IN_EV_PER_K * TARGET_T_K
EV_TO_KJ_PER_MOL = 96.48533212331002
COMPONENTS = ["Li", "Cl", "Na"]
INDEPENDENT = ["Li"]
CHARGES = {"Li": 1.0, "Cl": -1.0, "Na": 1.0}
COLUMN_MAP = {
    "x_Li": "x_Li",
    "x_Cl": "x_Cl",
    "x_Na": "x_Na",
    "S_LiLi": "S0_LiLi",
    "S_LiCl": "S0_LiCl",
    "S_LiNa": "S0_LiNa",
    "S_ClCl": "S0_ClCl",
    "S_ClNa": "S0_ClNa",
    "S_NaNa": "S0_NaNa",
}

sys.path[:0] = [
    str(SOURCE_ROOT / "GPR_grad"),
    str(SOURCE_ROOT / "S0_multi"),
]
from gpr_grad import GradientGP, RBFKernelFunction
from szero import prepare_gp_gradient_data


def load_input(path: Path, expected_kcut: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "system",
        "x_LiCl",
        "split",
        SOURCE_SPEC[SOURCE][1],
        SOURCE_SPEC[SOURCE][2],
        "n_Li",
        "n_Cl",
        "n_Na",
        "S0_LiLi",
        "S0_LiCl",
        "S0_LiNa",
        "S0_ClCl",
        "S0_ClNa",
        "S0_NaNa",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    frame = frame.loc[
        (frame["system"] == "LiCl-NaCl")
        & np.isclose(frame[SOURCE_SPEC[SOURCE][1]], expected_kcut)
    ].copy()
    if frame.empty:
        raise ValueError(f"{path.name}: no rows at kcut={expected_kcut}.")
    if 0 not in set(frame["split"].unique()):
        raise ValueError(f"{path.name}: split 0 (whole-window fit) is missing.")
    if not frame[SOURCE_SPEC[SOURCE][2]].astype(bool).all():
        raise ValueError(f"{path.name}: at least one selected S(0) fit did not converge.")

    n_ions = frame["n_Li"] + frame["n_Cl"] + frame["n_Na"]
    frame["x_Li"] = frame["n_Li"] / n_ions
    frame["x_Cl"] = frame["n_Cl"] / n_ions
    frame["x_Na"] = frame["n_Na"] / n_ions
    charge = frame["x_Li"] - frame["x_Cl"] + frame["x_Na"]
    if np.max(np.abs(charge)) > 1e-12:
        raise ValueError(f"{path.name}: ionic compositions are not charge neutral.")
    return frame


def convert_split(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("x_LiCl").reset_index(drop=True)
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
    y = frame["x_LiCl"].to_numpy(float)


    dx_li_dy = np.full_like(y, 0.5)

    dmu_licl_dy = (gamma[:, 0] + gamma[:, 1]) * dx_li_dy
    dmu_nacl_dy = (gamma[:, 2] + gamma[:, 1]) * dx_li_dy
    return pd.DataFrame(
        {
            "x_LiCl": y,
            "dmu_LiCl_ex_dy": dmu_licl_dy,
            "dmu_NaCl_ex_dy": dmu_nacl_dy,
            "Gex_curvature": dmu_licl_dy - dmu_nacl_dy,
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


THETA_FIXED = 0.1
SIGMA_F = 0.0
SIGMA_G = 1e-4


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


def fit_central_gp(
    central: pd.DataFrame, q_std: np.ndarray, n_steps: int = 400
) -> tuple[GradientGP, float, float, float]:
    y = central["x_LiCl"].to_numpy(float)
    q = central["Gex_curvature"].to_numpy(float)
    offset = float(np.mean(q))


    theta, sigma_f = THETA_FIXED, SIGMA_F
    gp = make_gp(theta=theta, sigma_f=sigma_f, trainable_theta=False)
    gp.fit(X_f=torch.tensor(y[:, None]), Y_f=torch.tensor(q - offset))
    return gp, offset, theta, sigma_f


def fit_fixed_gp(
    frame: pd.DataFrame, theta: float, sigma_f: float
) -> tuple[GradientGP, float]:
    y = frame["x_LiCl"].to_numpy(float)
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
        "mu_LiCl_ex": g_ex + (1.0 - y_grid) * dg_ex,
        "mu_NaCl_ex": g_ex - y_grid * dg_ex,
    }
    if q_cov is not None:
        mu_li_operator = g_operator + (1.0 - y_grid)[:, None] * derivative_operator
        mu_na_operator = g_operator - y_grid[:, None] * derivative_operator

        def propagated_std(operator: np.ndarray) -> np.ndarray:
            covariance_times_operator = q_cov @ operator.T
            variance = np.einsum("ij,ji->i", operator, covariance_times_operator)
            return np.sqrt(np.clip(variance, 0.0, None))

        result.update(
            {
                "q_GP_std": np.sqrt(np.clip(np.diag(q_cov), 0.0, None)),
                "Gex_GP_std": propagated_std(g_operator),
                "mu_LiCl_ex_GP_std": propagated_std(mu_li_operator),
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
    y_data = splits[0]["x_LiCl"].to_numpy(float)
    for split in block_ids:
        if not np.allclose(splits[split]["x_LiCl"], y_data, atol=1e-13):
            raise ValueError(f"Composition mismatch at kcut={kcut}, split={split}.")

    q_blocks = np.stack(
        [splits[split]["Gex_curvature"].to_numpy(float) for split in block_ids]
    )
    q_std = np.std(q_blocks, axis=0, ddof=1)
    integrand_blocks = np.stack(
        [
            y_data * splits[split]["dmu_LiCl_ex_dy"].to_numpy(float)
            for split in block_ids
        ]
    )
    integrand = y_data * splits[0]["dmu_LiCl_ex_dy"].to_numpy(float)
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

    mean_keys = ("q", "Gex", "mu_LiCl_ex", "mu_NaCl_ex")
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
        help="one or two k^2 cuts (default: 0.02 0.01).  With one cut the "
             "figure shows that cut alone.",
    )
    args = parser.parse_args()
    SOURCE = args.source
    if not 1 <= len(args.cuts) <= 2:
        parser.error("--cuts takes one or two values")
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
    ideal_mu_li = np.full_like(y_grid, np.nan)
    ideal_mu_na = np.full_like(y_grid, np.nan)
    ideal_mu_li[interior] = KB_T_EV * np.log(y_grid[interior])
    ideal_mu_na[interior] = KB_T_EV * np.log(1.0 - y_grid[interior])


    _styles = [{"color": "#1764ab", "ls": "-", "marker": "o"},
               {"color": "#e12729", "ls": "--", "marker": "x"}]
    if len(CUTS) == 2:
        _styles = [_styles[1], _styles[0]]
    styles = {c: _styles[i] for i, c in enumerate(CUTS)}

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
        loc="upper left",
        frameon=False,
        ncol=2,
        title=r"$k_{cut}^2$ in $4\pi^2\,\AA^{-2}$",
        title_fontsize=6.5,
    )
    ax1.set_ylabel(
        r"${\partial \mu^{\rm ex}_{\rm LiCl}}/{\partial \ln x_{\rm LiCl}}$"
        "\n[kJ/mol]"
    )
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax2.plot(
        y_grid[interior],
        ideal_mu_li[interior] * EV_TO_KJ_PER_MOL,
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
        mu_li = result["central"]["mu_LiCl_ex"] + ideal_mu_li
        mu_na = result["central"]["mu_NaCl_ex"] + ideal_mu_na
        ax2.plot(
            y_grid[interior],
            mu_li[interior] * EV_TO_KJ_PER_MOL,
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
            np.interp(error_x, y_grid, mu_li) * EV_TO_KJ_PER_MOL,
            np.interp(error_x, y_grid, result["std"]["mu_LiCl_ex"])
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
               label=r"$\mu_{\rm LiCl}$"),
        Line2D([0], [0], color="black", marker="<", fillstyle="none", lw=1.0,
               label=r"$\mu_{\rm NaCl}$"),
        Line2D([0], [0], color="gray", ls=(0, (1, 3)), lw=1.0, label="Ideal"),
    ]
    ax2.legend(
        handles=cutoff_handles + species_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncol=2,
        handlelength=1.8,
        title=r"$k_{cut}^2$ in $4\pi^2\,\AA^{-2}$",
        title_fontsize=6.5,
    )
    ax2.set_ylabel(r"$\mu$ [kJ/mol]")
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))


    _b = np.concatenate([
        (processed[c]["central"]["mu_LiCl_ex"] + ideal_mu_li)[interior]
        for c in CUTS
    ] + [
        (processed[c]["central"]["mu_NaCl_ex"] + ideal_mu_na)[interior]
        for c in CUTS
    ]) * EV_TO_KJ_PER_MOL
    ax2.set_ylim(bottom=1.06 * np.nanmin(_b), top=0.06 * abs(np.nanmin(_b)))

    ax3.plot(
        y_grid,
        ideal_g * EV_TO_KJ_PER_MOL,
        color="gray",
        ls=(0, (1, 3)),
        alpha=0.70,
        lw=1.0,
        label="Ideal",
    )
    dense_rows = {"x_LiCl": y_grid}
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
        dense_rows[f"{prefix}__mu_LiCl_mix_kJ_per_mol"] = (
            result["central"]["mu_LiCl_ex"] + ideal_mu_li
        ) * EV_TO_KJ_PER_MOL
        dense_rows[f"{prefix}__mu_LiCl_split_std_kJ_per_mol"] = (
            result["std"]["mu_LiCl_ex"] * EV_TO_KJ_PER_MOL
        )
        dense_rows[f"{prefix}__mu_NaCl_mix_kJ_per_mol"] = (
            result["central"]["mu_NaCl_ex"] + ideal_mu_na
        ) * EV_TO_KJ_PER_MOL
        dense_rows[f"{prefix}__mu_NaCl_split_std_kJ_per_mol"] = (
            result["std"]["mu_NaCl_ex"] * EV_TO_KJ_PER_MOL
        )


    ax3.text(
        0.02, 0.04,
        "no experimental assessment available for this system",
        transform=ax3.transAxes, ha="left", va="bottom",
        fontsize=5.9, color="0.45",
    )

    ax3.set_xlabel(r"$x_{\rm LiCl}$")
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
        ncol=2,
    )

    _c = ax3.get_ylim()
    ax3.set_ylim(_c[0] - 0.16 * (_c[1] - _c[0]), _c[1])

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


    fig.savefig(OUTPUT_PDF, dpi=500, bbox_inches="tight")
    fig.savefig(output_png, dpi=500, bbox_inches="tight")


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

        mu_li = result["central"]["mu_LiCl_ex"] + ideal_mu_li
        mu_na = result["central"]["mu_NaCl_ex"] + ideal_mu_na
        mu_li_std = result["std"]["mu_LiCl_ex"]
        mu_na_std = result["std"]["mu_NaCl_ex"]
        ax2.fill_between(
            y_grid[interior],
            (mu_li[interior] - mu_li_std[interior]) * EV_TO_KJ_PER_MOL,
            (mu_li[interior] + mu_li_std[interior]) * EV_TO_KJ_PER_MOL,
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
        0.11,
        rf"Shading: {processed[CUTS[0]]['n_blocks']}-block SD",
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        color="0.35",
    )


    _val = np.concatenate([processed[c]["integrand"] for c in CUTS])
    _err = np.concatenate([processed[c]["integrand_std"] for c in CUTS])
    _cap = np.minimum(_err, 3.0 * np.median(_err))
    _lo = (_val - _cap).min() * EV_TO_KJ_PER_MOL
    _hi = (_val + _cap).max() * EV_TO_KJ_PER_MOL
    _pad = 0.30 * (_hi - _lo)
    ax1.set_ylim(min(_lo - _pad, -0.5), _hi + _pad)
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
