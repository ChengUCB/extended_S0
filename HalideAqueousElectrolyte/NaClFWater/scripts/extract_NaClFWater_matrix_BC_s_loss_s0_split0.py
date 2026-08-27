#!/usr/bin/env python3
"""Extract NaClFWater split-0 S0 values with bare-Coulomb matrix S-loss fits."""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import extract_NaClFWater_matrix_DH_s_loss_s0_split0 as dh_common
import sk_s0_common as common


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Sk_files" if (SCRIPT_DIR / "Sk_files").is_dir() else SCRIPT_DIR
DEBYE_CSV = (
    SCRIPT_DIR / "x_ion_mu_ex_debye.csv"
    if (SCRIPT_DIR / "x_ion_mu_ex_debye.csv").exists()
    else SCRIPT_DIR.parent / "NaClWater-clean" / "x_ion_mu_ex_debye.csv"
)
K2_SCALE_BC = dh_common.K2_SCALE_DH
VALENCES = dh_common.VALENCES


@dataclass(frozen=True)
class BCFitResult:
    s0: np.ndarray
    kappa_fit: float
    n_fit: int
    rmse: float
    success: bool
    a_mat: np.ndarray
    l_mat: np.ndarray


def charge_projection_matrix(fractions: np.ndarray) -> np.ndarray:
    y = np.diag(np.sqrt(np.asarray(fractions, dtype=float)))
    w = y @ VALENCES
    denominator = float(w @ w)
    if denominator <= 0.0:
        return np.full((len(fractions), len(fractions)), np.nan)
    return np.outer(w, w) / denominator


def bare_coulomb_s0_limit(a_mat: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Return lim k->0 inv(A + Pz*kappa^2/k^2) using Sherman-Morrison."""
    y = np.diag(np.sqrt(np.asarray(fractions, dtype=float)))
    w = y @ VALENCES
    a_inv = dh_common.safe_inv(a_mat)
    numerator = (a_inv @ w[:, None]) @ (w[None, :] @ a_inv)
    denominator = float(w @ a_inv @ w)
    if not np.isfinite(denominator) or abs(denominator) < 1.0e-14:
        return np.full_like(a_mat, np.nan)
    return a_inv - numerator / denominator


def fit_matrix_bc_s_loss_result(
    k2: np.ndarray,
    sk_matrix: np.ndarray,
    mask: np.ndarray,
    fractions: np.ndarray,
    kappa_input: float,
    *,
    fit_kappa: bool = False,
    initial_a: np.ndarray | None = None,
    initial_l: np.ndarray | None = None,
    max_nfev: int = 800,
) -> BCFitResult:
    """Fit S(k)=inv(A+k_ang^2 L+Pz*kappa_D^2/k_ang^2) in S-space."""
    k_raw, target_matrix = dh_common.finite_fit_data(k2, sk_matrix, mask)
    n_comp = sk_matrix.shape[1]
    if len(k_raw) < 3:
        empty = np.full((n_comp, n_comp), np.nan)
        return BCFitResult(empty, kappa_input, len(k_raw), math.nan, False, empty, empty)

    bases, tri = dh_common.symmetric_bases(n_comp)
    n_tri = len(tri[0])
    target, scale = dh_common.scaled_lower_target(target_matrix, tri)
    k_scaled = k_raw * K2_SCALE_BC
    pz = charge_projection_matrix(fractions)
    lambda_input = kappa_input**2 / k_scaled

    if initial_a is None or initial_l is None:
        inv_initial = dh_common.inverse_space_a_initial(k_scaled, target_matrix)
        a0 = inv_initial - float(np.median(lambda_input)) * pz
        l0 = np.zeros_like(a0)
    else:
        a0 = np.asarray(initial_a, dtype=float)
        l0 = np.asarray(initial_l, dtype=float)
    log_kappa0 = math.log(kappa_input)
    initial = np.r_[dh_common.pack_lower(a0), dh_common.pack_lower(l0)]
    if fit_kappa:
        initial = np.r_[initial, log_kappa0]
    reg_scale = 10000.0
    n_reg = n_tri + (1 if fit_kappa else 0)
    bad_res = np.full(target.size + n_reg, 1.0e8, dtype=float)
    bad_jac = np.zeros((bad_res.size, initial.size), dtype=float)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        a_mat = dh_common.unpack_lower(params[:n_tri], n_comp)
        l_mat = dh_common.unpack_lower(params[n_tri : 2 * n_tri], n_comp)
        kappa = float(math.exp(params[-1])) if fit_kappa else kappa_input
        return a_mat, l_mat, kappa

    def residuals(params: np.ndarray) -> np.ndarray:
        a_mat, l_mat, kappa = unpack(params)
        lambda_c = kappa**2 / k_scaled
        mats = (
            a_mat[None, :, :]
            + k_scaled[:, None, None] * l_mat[None, :, :]
            + lambda_c[:, None, None] * pz[None, :, :]
        )
        pred, ok = dh_common.batched_inv(mats)
        if not ok:
            return bad_res.copy()
        data_res = (pred[:, tri[0], tri[1]] - target) / scale
        reg = dh_common.pack_lower(l_mat - l0) / reg_scale
        if fit_kappa:
            reg = np.r_[reg, (params[-1] - log_kappa0) / 10.0]
        return np.r_[data_res.ravel(), reg]

    def jacobian(params: np.ndarray) -> np.ndarray:
        a_mat, l_mat, kappa = unpack(params)
        lambda_c = kappa**2 / k_scaled
        mats = (
            a_mat[None, :, :]
            + k_scaled[:, None, None] * l_mat[None, :, :]
            + lambda_c[:, None, None] * pz[None, :, :]
        )
        pred, ok = dh_common.batched_inv(mats)
        if not ok:
            return bad_jac.copy()
        coeff_a = np.ones((len(k_raw), n_tri), dtype=float)
        coeff_l = np.repeat(k_scaled[:, None], n_tri, axis=1)
        jac_a = dh_common.s_jacobian_lower(pred, bases, tri, scale, coeff_a)
        jac_l = dh_common.s_jacobian_lower(pred, bases, tri, scale, coeff_l)
        blocks = [jac_a, jac_l]
        if fit_kappa:
            coeff_eta = 2.0 * lambda_c
            ds_eta = -np.einsum("kab,bc,kcd,k->kad", pred, pz, pred, coeff_eta, optimize=True)
            blocks.append((ds_eta[:, tri[0], tri[1]] / scale).reshape(len(k_raw), len(scale), 1))
        jac_data = np.concatenate(blocks, axis=2).reshape(target.size, initial.size)
        jac_reg = np.zeros((n_reg, initial.size), dtype=float)
        jac_reg[:n_tri, n_tri : 2 * n_tri] = np.eye(n_tri) / reg_scale
        if fit_kappa:
            jac_reg[-1, -1] = 0.1
        return np.vstack([jac_data, jac_reg])

    try:
        lower = np.full_like(initial, -np.inf)
        upper = np.full_like(initial, np.inf)
        if fit_kappa:
            lower[-1] = log_kappa0 - math.log(10.0)
            upper[-1] = log_kappa0 + math.log(10.0)
        result = least_squares(
            residuals,
            initial,
            jac=jacobian,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1.0e-6,
            xtol=1.0e-6,
            gtol=1.0e-6,
        )
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError("BC fit failed")
        a_mat, l_mat, kappa_fit = unpack(result.x)
        s0 = bare_coulomb_s0_limit(a_mat, fractions)
        ok = bool(np.isfinite(s0).all())
        return BCFitResult(
            s0,
            kappa_fit,
            len(k_raw),
            float(np.sqrt(np.mean(residuals(result.x) ** 2))),
            ok,
            a_mat,
            l_mat,
        )
    except Exception:
        empty = np.full((n_comp, n_comp), np.nan)
        return BCFitResult(empty, kappa_input, len(k_raw), math.nan, False, empty, empty)


def process_sk_file(payload: tuple[common.SkFile, tuple[float, ...], np.ndarray, np.ndarray, bool]) -> list[dict[str, object]]:
    item, kcuts, debye_x, debye_nm, include_fitted_kappa = payload
    rows = []
    column_index, data = common.read_allsk(item.path)
    k2 = np.asarray(data[:, column_index["k^2"]], dtype=float)
    allpair_matrix = common.assemble_matrix(data, column_index, common.ALLPAIR_MAP)
    kappa_input = dh_common.kappa_d_from_x_x(item.x_x, debye_x, debye_nm)

    for kcut in kcuts:
        mask = common.lowk_mask(k2, data, column_index, common.ALLPAIR_COMPONENTS, kcut)
        fixed = fit_matrix_bc_s_loss_result(
            k2,
            allpair_matrix,
            mask,
            item.fractions,
            kappa_input,
            fit_kappa=False,
        )
        results = [("BC S-loss, fixed kappa_D", fixed)]
        if include_fitted_kappa:
            fitted = fit_matrix_bc_s_loss_result(
                k2,
                allpair_matrix,
                mask,
                item.fractions,
                kappa_input,
                fit_kappa=True,
                initial_a=fixed.a_mat,
                initial_l=fixed.l_mat,
            )
            results.append(("BC S-loss, fitted kappa_D", fitted))

        for method, result in results:
            row = common.base_s0_row("matrix_BC_s_loss", method, item, kcut)
            row["kappa_D_input"] = kappa_input
            row["kappa_D_fit"] = result.kappa_fit
            row["n_fit_points"] = result.n_fit
            row["fit_rmse_scaled"] = result.rmse
            row["fit_success"] = int(result.success)
            common.set_allpair_fields(row, result.s0)
            rows.append(row)
    return rows


def build_s0_table(
    data_dir: Path,
    source: str,
    kcuts: tuple[float, ...],
    debye_csv: Path,
    n_procs: int = 4,
    include_fitted_kappa: bool = False,
    split: int = 0,
) -> object:
    debye_x, debye_nm = dh_common.read_debye_table(debye_csv)
    files = common.discover_split_files(data_dir, source, split=split)
    payloads = [(item, kcuts, debye_x, debye_nm, include_fitted_kappa) for item in files]
    if n_procs <= 1:
        nested_rows = []
        for file_index, payload in enumerate(payloads, start=1):
            print(f"{file_index:02d}/{len(files):02d} {payload[0].path.name}", flush=True)
            nested_rows.append(process_sk_file(payload))
    else:
        with ProcessPoolExecutor(max_workers=n_procs) as pool:
            nested_rows = []
            for file_index, file_rows in enumerate(pool.map(process_sk_file, payloads), start=1):
                print(f"{file_index:02d}/{len(files):02d} done", flush=True)
                nested_rows.append(file_rows)
    rows = [row for file_rows in nested_rows for row in file_rows]
    return common.ordered_s0_table(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--source", default="main")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--kcuts", type=common.parse_kcuts, default=common.DEFAULT_KCUTS)
    parser.add_argument("--debye-csv", type=Path, default=DEBYE_CSV)
    parser.add_argument("--n-procs", type=int, default=4, help="Number of file-level worker processes; use 1 for sequential.")
    parser.add_argument(
        "--include-fitted-kappa",
        action="store_true",
        help="Also write fitted-kappa BC rows, warm-started from the fixed-kappa fit. Default writes fixed-kappa rows only.",
    )
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "S0_NaClFWater_matrix_BC_s_loss_split0.csv")
    args = parser.parse_args()
    table = build_s0_table(
        args.data_dir.expanduser().resolve(),
        args.source,
        args.kcuts,
        args.debye_csv.expanduser().resolve(),
        args.n_procs,
        args.include_fitted_kappa,
        args.split,
    )
    common.write_s0_csv(table, args.output)
    print(f"wrote {args.output}")
    print(table.groupby(["method", "kcut"])["fit_success"].sum().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
