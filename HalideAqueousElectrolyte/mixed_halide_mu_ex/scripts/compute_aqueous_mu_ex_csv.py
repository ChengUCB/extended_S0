#!/usr/bin/env python3
"""Compute BC/GP excess salt chemical-potential surfaces from S0 CSV files."""

from __future__ import annotations

import os
import argparse
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/pycache")

ROOT = Path(__file__).resolve().parents[1]
SI_ROOT = ROOT.parent
MU_EX_DIR = ROOT / "data"

import numpy as np
import pandas as pd
import torch

from gpr_grad import GradientGP, RBFKernelFunction
from szero import prepare_gp_gradient_data

SYSTEMS = {
    "I": {
        "system": "NaClIWater",
        "label": "NaCl/NaI/water",
        "anion": "I",
        "x_coanion_max": 0.10,
        "csv": SI_ROOT / "NaClIWater" / "data" / "S0_NaClIWater_matrix_BC_s_loss_split0.csv",
        "output_prefix": "mu_ex_NaClIWater_BC_fixed",
    },
    "F": {
        "system": "NaClFWater",
        "label": "NaCl/NaF/water",
        "anion": "F",
        "x_coanion_max": 0.04,
        "csv": SI_ROOT / "NaClFWater" / "data" / "S0_NaClFWater_matrix_BC_s_loss_split0.csv",
        "output_prefix": "mu_ex_NaClFWater_BC_fixed",
    },
}

K_B_EV = 8.617333262e-5
TEMP_K = 298.15
KBT = K_B_EV * TEMP_K
R_KJ_PER_MOL_K = 0.00831446261815324
RT_KJ_PER_MOL = R_KJ_PER_MOL_K * TEMP_K
UNCERTAINTY_SPLITS = (1, 2, 3)
COMPONENT_BASE = ["O", "Na", "Cl"]
CHARGES_BASE = {"O": 0.0, "Na": 1.0, "Cl": -1.0}


def split_csv_path(path: Path, split: int) -> Path:
    return path.with_name(path.name.replace("_split0.csv", f"_split{split}.csv"))


def output_path(system_key: str, kcut: float) -> Path:
    return MU_EX_DIR / f"{SYSTEMS[system_key]['output_prefix']}_kcut{kcut:g}.csv"


def load_bc(system_key: str, kcut: float, csv_path: Path | None = None) -> pd.DataFrame:
    spec = SYSTEMS[system_key]
    df = pd.read_csv(csv_path or spec["csv"])
    df = df[
        (df["method"] == "BC S-loss, fixed kappa_D")
        & np.isclose(df["kcut"].to_numpy(float), kcut)
        & (df["fit_success"] == 1)
    ].copy()
    return df.sort_values(["x_Cl", f"x_{spec['anion']}"]).reset_index(drop=True)


def column_map(anion: str) -> dict[str, str]:
    return {
        "x_O": "x_O",
        "x_Na": "x_Na",
        "x_Cl": "x_Cl",
        f"x_{anion}": f"x_{anion}",
        "S_OO": "S0_OO",
        "S_ONa": "S0_ONa",
        "S_OCl": "S0_OCl",
        "S_NaNa": "S0_NaNa",
        "S_NaCl": "S0_NaCl",
        "S_ClCl": "S0_ClCl",
        f"S_Na{anion}": f"S0_Na{anion}",
        f"S_Cl{anion}": f"S0_Cl{anion}",
        f"S_O{anion}": f"S0_O{anion}",
        f"S_{anion}{anion}": f"S0_{anion}{anion}",
    }


def fit_mu_gp(df: pd.DataFrame, anion: str) -> dict[str, GradientGP]:
    components = [*COMPONENT_BASE, anion]
    charges = {**CHARGES_BASE, anion: -1.0}
    gp_grad_data = prepare_gp_gradient_data(
        df,
        independent_components=["Cl", anion],
        column_map=column_map(anion),
        components=components,
        species=components,
        temperature=TEMP_K,
        excess=True,
        charges=charges,
        as_torch=True,
    )

    gps: dict[str, GradientGP] = {}
    for species in components:
        x_g, g_g, grad_indices = gp_grad_data[species]
        kernel = RBFKernelFunction(theta=torch.tensor([0.05, 0.05]), trainable=False)
        gp = GradientGP(
            kernel=kernel,
            sigma_f=0.0,
            sigma_g=0.1,
            trainable_sigma_f=False,
            trainable_sigma_g=False,
            jitter=1e-8,
        )
        gp.fit(
            X_f=torch.tensor([[0.001, 0.001]]),
            Y_f=torch.tensor([0.0]),
            X_g=x_g,
            G_g=g_g,
            grad_indices=grad_indices,
        )
        gps[species] = gp
    return gps


def prediction_grid(x_max: float, y_max: float, step: float = 0.0025) -> torch.Tensor:
    xs = torch.arange(0.0, x_max + 0.5 * step, step)
    ys = torch.arange(0.0, y_max + 0.5 * step, step)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    return points[points.sum(dim=1) <= 0.115]


def predict_salt(gps: dict[str, GradientGP], points: torch.Tensor, salt: str) -> np.ndarray:
    mu_na, _ = gps["Na"].predict(points)
    if salt == "NaCl":
        mu_an, _ = gps["Cl"].predict(points)
    else:
        mu_an, _ = gps[salt[2:]].predict(points)
    return ((mu_na + mu_an) / KBT).detach().numpy()


def build_mu_ex_table(system_key: str, kcut: float, csv_path: Path | None = None, source_split: int = 0) -> pd.DataFrame:
    spec = SYSTEMS[system_key]
    anion = spec["anion"]
    source_csv = csv_path or spec["csv"]
    try:
        source_label = str(source_csv.resolve().relative_to(SI_ROOT))
    except ValueError:
        source_label = str(source_csv)
    df = load_bc(system_key, kcut, source_csv)
    gps = fit_mu_gp(df, anion)
    points = prediction_grid(0.10, spec["x_coanion_max"])
    xy = points.detach().numpy()

    rows = []
    for salt in ["NaCl", f"Na{anion}"]:
        mu_raw_kbt = predict_salt(gps, points, salt)
        reference = float(mu_raw_kbt[(xy[:, 0] == 0.0) & (xy[:, 1] == 0.0)][0])
        mu_kbt = mu_raw_kbt - reference
        for (x_cl, x_coanion), value, raw_value in zip(xy, mu_kbt, mu_raw_kbt):
            rows.append(
                {
                    "system": spec["system"],
                    "system_label": spec["label"],
                    "method": "BC S-loss, fixed kappa_D",
                    "kcut": kcut,
                    "temperature_K": TEMP_K,
                    "coanion": anion,
                    "salt": salt,
                    "quantity": f"neutral ion-pair {salt}: mu_Na + mu_{salt[2:]}",
                    "x_Cl": x_cl,
                    f"x_{anion}": x_coanion,
                    "x_coanion": x_coanion,
                    "mu_ex": value * RT_KJ_PER_MOL,
                    "mu_ex_units": "kJ/mol",
                    "mu_ex_kBT": value,
                    "mu_ex_eV": value * KBT,
                    "mu_ex_kJ_mol": value * RT_KJ_PER_MOL,
                    "mu_ex_raw_kBT": raw_value,
                    "mu_ex_reference_kBT": reference,
                    "reference_x_Cl": 0.0,
                    "reference_x_coanion": 0.0,
                    "source_s0_csv": source_label,
                    "source_split": source_split,
                }
            )
    return pd.DataFrame(rows)


def add_split_uncertainties(system_key: str, table: pd.DataFrame, kcut: float) -> pd.DataFrame:
    spec = SYSTEMS[system_key]
    split_tables = []
    for split in UNCERTAINTY_SPLITS:
        csv_path = split_csv_path(spec["csv"], split)
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} is missing; regenerate split {split} S0 values first")
        split_tables.append(build_mu_ex_table(system_key, kcut, csv_path=csv_path, source_split=split))

    replicates = pd.concat(split_tables, ignore_index=True)
    key_columns = ["system", "coanion", "salt", "x_Cl", "x_coanion"]
    stats = (
        replicates.groupby(key_columns, as_index=False)
        .agg(
            mu_ex_std_split1_to3=("mu_ex", lambda values: values.std(ddof=1)),
            mu_ex_se_split1_to3=("mu_ex", lambda values: values.std(ddof=1) / np.sqrt(values.count())),
            mu_ex_kBT_std_split1_to3=("mu_ex_kBT", lambda values: values.std(ddof=1)),
            mu_ex_kBT_se_split1_to3=("mu_ex_kBT", lambda values: values.std(ddof=1) / np.sqrt(values.count())),
            mu_ex_uncertainty_n_splits=("mu_ex", "count"),
        )
    )
    return table.merge(stats, on=key_columns, how="left", validate="one_to_one")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kcuts", type=float, nargs="+", default=[0.01])
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    for kcut in args.kcuts:
        for key in ["I", "F"]:
            table = build_mu_ex_table(key, kcut)
            table = add_split_uncertainties(key, table, kcut)
            output = output_path(key, kcut)
            table.to_csv(output, index=False)
            print(f"wrote {output} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
