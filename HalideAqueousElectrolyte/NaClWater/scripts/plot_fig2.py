#!/usr/bin/env python3
"""Figure 2: NaCl excess chemical potential from S0 workflows vs literature, plotted from pre-computed CSVs.

data/S0_mu_ex_NaCl.csv         S0 mu_ex curves (zero at the lowest S0 concentration), both k^2_cut values
data/reference_mu_ex_NaCl.csv  literature mu_ex, already aligned to the S0 curves at infinite dilution (DH limit)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repo root: scripts/, data/, figures/
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

KCUTS = [0.005, 0.01]

# Distinct hues kept clear of the workflow colors (plum-brown, mauve, blue, green, orange): gray, light green, light yellow, light purple; dict order = legend order (newest first).
LITERATURE_STYLES = {
    "Chattopadhyay et al. (2025)": {"marker": "^", "color": "#555555", "short_label": "Chattopadhyay 2025"},
    "Benavides et al. (2016)": {"marker": "s", "color": "#B08B32", "short_label": "Benavides 2016"},
    "Mester & Panagiotopoulos (2015)": {"marker": "D", "color": "#7650A3", "short_label": "Mester 2015"},
    "Moučka et al. (2013)": {"marker": "o", "color": "#3F8F7D", "short_label": "Moučka 2013"},
}
LITERATURE_ALPHA = 0.7

WORKFLOWS = [
    {
        "key": "individual_oz_direct_ox_numeric",
        "kind": "direct_numeric",
        "color": "#744A6C",
        "linestyle": "-",
        "short_label": "OZ 2-comp num.",
        "zorder": 4.5,  # points above the thick OZ 2-comp line
    },
    {
        "key": "individual_oz_direct_ox_gp",
        "kind": "direct_gp",
        "color": "#A97AAE",
        "linestyle": "-",
        "short_label": "OZ 2-comp",
        "lw": 1.25,  
        "zorder": 4,
    },
    {
        "key": "individual_oz_3comp_gp",
        "kind": "component_gp",
        "color": "#4C78A8",
        "linestyle": "--",
        "short_label": "OZ",
    },
    {
        "key": "matrix_oz_3comp_gp",
        "kind": "component_gp",
        "color": "#54A24B",
        "linestyle": "-.",
        "short_label": "matrix OZ",
    },
    {
        "key": "matrix_bc_3comp_gp_fixed_kappa",
        "kind": "component_gp",
        "color": "#F58518",
        "linestyle": "-",
        "short_label": "matrix BC",
    },
]


def plot(
    df: pd.DataFrame, output: Path, literature: pd.DataFrame | None = None, figsize: tuple[float, float] = (8.5 / 2.54, 2.75),
    scale: float = 1.0,
    show_kcut_legend: bool = True,
    legend_ncol: int = 2,
) -> None:
    """`scale` multiplies every font size, line width, marker size and tick size; `show_kcut_legend` toggles the k^2_cut legend.
    The S0 reference curve and its legend entry are drawn only if `df` has reference_csv rows."""
    plt.rcParams.update(
        {
            "font.size": scale * 10.5,
            "axes.labelsize": scale * 10.5,
            "xtick.labelsize": scale * 9.8,
            "ytick.labelsize": scale * 9.8,
            "legend.fontsize": scale * 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=figsize)

    styles = {workflow["key"]: workflow for workflow in WORKFLOWS}
    linestyle_by_kcut = {0.005: ":", 0.01: "-"}
    marker_by_kcut = {0.005: "s", 0.01: "o"}
    for key in [workflow["key"] for workflow in WORKFLOWS]:
        style = styles[key]
        for kcut in KCUTS:
            part = df[(df["workflow"] == key) & np.isclose(df["kcut"], kcut)].sort_values("c_NaCl_M")
            se = part.get("mu_ex_se_split1_to3_kJ_mol")
            has_se = se is not None and np.isfinite(se.to_numpy(dtype=float)).any()
            if has_se and style["kind"] != "direct_numeric":
                ax.fill_between(
                    part["c_NaCl_M"].to_numpy(dtype=float),
                    (part["mu_ex_kJ_mol"] - se).to_numpy(dtype=float),
                    (part["mu_ex_kJ_mol"] + se).to_numpy(dtype=float),
                    color=style["color"],
                    alpha=0.28 if kcut == 0.01 else 0.18,
                    lw=0.0,
                    zorder=1,
                )
            if style["kind"] == "direct_numeric":
                ax.errorbar(
                    part["c_NaCl_M"],
                    part["mu_ex_kJ_mol"],
                    yerr=se.fillna(0.0).to_numpy(dtype=float) if has_se else None,
                    marker=marker_by_kcut.get(kcut, "o"),
                    ms=scale * 1.5,
                    lw=scale * 0.5,
                    color=style["color"],
                    ls=linestyle_by_kcut.get(kcut, "-"),
                    elinewidth=scale * 0.4,
                    capsize=scale * 0.8,
                    alpha=1.0 if kcut == 0.01 else 0.6,
                    zorder=style.get("zorder", 3),
                )
            else:
                ax.plot(
                    part["c_NaCl_M"],
                    part["mu_ex_kJ_mol"],
                    lw=scale * style.get("lw", 0.82),
                    color=style["color"],
                    ls=linestyle_by_kcut.get(kcut, style["linestyle"]),
                    zorder=style.get("zorder", 3),
                )

    ref = df[df["workflow"] == "reference_csv"].sort_values("c_NaCl_M")
    if not ref.empty:
        ax.plot(
            ref["c_NaCl_M"],
            ref["mu_ex_kJ_mol"],
            color="black",
            ls=(0, (4.0, 1.5)),
            lw=scale * 0.9,
            zorder=5,
        )

    x_right = float(df["c_NaCl_M"].max())  # end axis where the S0 curves end
    y_range = ax.get_ylim()
    literature_refs = [] if literature is None else [r for r in LITERATURE_STYLES if r in set(literature["reference"])]
    literature_handles = []
    for reference in literature_refs:
        part = literature[literature["reference"] == reference]
        style = LITERATURE_STYLES[reference]
        container = ax.errorbar(
            part["c_NaCl_mol_per_L"],
            part["delta_mu_ex_kJ_mol"],
            yerr=part["mu_ex_err_kJ_per_mol"].fillna(0.0) if part["mu_ex_err_kJ_per_mol"].notna().any() else None,
            fmt=style["marker"],
            ms=scale * 3.1,
            mfc="white",
            mec=style["color"],
            mew=scale * 0.7,
            ecolor=style["color"],
            elinewidth=scale * 0.5,
            capsize=scale * 1.2,
            ls="none",
            alpha=LITERATURE_ALPHA,
            zorder=6,
        )
        literature_handles.append(container)

    ax.axhline(0.0, color="0.78", lw=scale * 0.35, zorder=0)
    ax.set_xlabel(r"$c_{\mathrm{NaCl}}$ [mol/L]", labelpad=scale * 1.5)
    ax.set_ylabel(r"$\mu_{\mathrm{NaCl}}^{\mathrm{ex}}$ [kJ/mol]", labelpad=scale * 1.5)
    ax.set_xlim(0.0, x_right)
    ax.set_ylim(y_range[0], y_range[1] + 0.8)
    ax.tick_params(direction="in", length=scale * 2.8, width=scale * 0.65, pad=scale * 2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(scale * 0.65)
    has_band = {
        key: np.isfinite(df.loc[df["workflow"] == key, "mu_ex_se_split1_to3_kJ_mol"].to_numpy(dtype=float)).any()
        if "mu_ex_se_split1_to3_kJ_mol" in df
        else False
        for key in styles
    }
    def workflow_handle(workflow: dict[str, object]):
        color = workflow["color"]
        if workflow["kind"] == "direct_numeric":
            # Legend-only marker with errorbar; NaN data draws nothing on the axes.
            return ax.errorbar(
                [np.nan],
                [np.nan],
                yerr=[np.nan] if has_band[workflow["key"]] else None,
                fmt="o",
                ms=scale * 2.0,
                color=color,
                elinewidth=scale * 0.4,
                capsize=scale * 0.8,
            )
        line = Line2D([0], [0], color=color, lw=scale * 1.0 * workflow.get("lw", 0.82) / 0.82)
        if has_band[workflow["key"]]:
            return (Patch(facecolor=color, alpha=0.28, lw=0.0), line)
        return line

    workflow_handles = [workflow_handle(workflow) for workflow in WORKFLOWS]
    workflow_labels = [workflow["short_label"] for workflow in WORKFLOWS]
    if not ref.empty:
        workflow_handles.insert(0, Line2D([0], [0], color="black", lw=scale * 0.9, ls=(0, (4.0, 1.5))))
        workflow_labels.insert(0, "S0 reference")
    workflow_handles.extend(literature_handles)
    workflow_labels.extend(LITERATURE_STYLES[reference]["short_label"] for reference in literature_refs)
    cutoff_handles = [
        Line2D([0], [0], color="0.25", lw=scale * 0.9, ls=linestyle_by_kcut[kcut], label=rf"$k^2_\mathrm{{cut}}={kcut:g}$")
        for kcut in KCUTS
    ]
    workflow_legend = ax.legend(
        workflow_handles,
        workflow_labels,
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
        frameon=False,
        loc="upper left",
        ncol=legend_ncol,
        handlelength=1.15,
        columnspacing=0.75,
        labelspacing=0.22,
        handletextpad=0.35,
        borderaxespad=0.15,
    )
    ax.add_artist(workflow_legend)
    if show_kcut_legend:
        ax.legend(
            handles=cutoff_handles,
            frameon=False,
            loc="lower right",
            handlelength=1.25,
            labelspacing=0.18,
            handletextpad=0.35,
            borderaxespad=0.18,
        )
    fig.tight_layout(pad=0.2)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    s0 = pd.read_csv(DATA_DIR / "S0_mu_ex_NaCl.csv")
    literature = pd.read_csv(DATA_DIR / "reference_mu_ex_NaCl.csv")
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "fig2.pdf"
    plot(s0, out, literature)
    print(f"wrote {out} and {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
