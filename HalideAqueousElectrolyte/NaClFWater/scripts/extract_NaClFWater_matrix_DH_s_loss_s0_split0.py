#!/usr/bin/env python3
"""Extract NaClFWater split-0 S0 values with matrix DH S-loss fits."""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import sk_s0_common as common


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Sk_files" if (SCRIPT_DIR / "Sk_files").is_dir() else SCRIPT_DIR
DEBYE_CSV = (
    SCRIPT_DIR / "x_ion_mu_ex_debye.csv"
    if (SCRIPT_DIR / "x_ion_mu_ex_debye.csv").exists()
    else SCRIPT_DIR.parent / "data" / "x_ion_mu_ex_debye.csv"
)
K2_SCALE_DH = (2.0 * np.pi) ** 2
VALENCES = np.array([0.0, 1.0, -1.0, -1.0])


@dataclass(frozen=True)
class DHFitResult:
    s0: np.ndarray
    kappa_fit: float
    n_fit: int
    rmse: float
    success: bool
    yhy: np.ndarray
    l_mat: np.ndarray

    def public_tuple(self) -> tuple[np.ndarray, float, int, float, bool]:
        return self.s0, self.kappa_fit, self.n_fit, self.rmse, self.success


def pack_lower(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.tril_indices(matrix.shape[0])]


def unpack_lower(values: np.ndarray, n_comp: int) -> np.ndarray:
    matrix = np.zeros((n_comp, n_comp), dtype=float)
    matrix[np.tril_indices(n_comp)] = values
    return matrix + np.tril(matrix, -1).T


def symmetric_bases(n_comp: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    tri = np.tril_indices(n_comp)
    bases = []
    for i, j in zip(*tri):
        basis = np.zeros((n_comp, n_comp), dtype=float)
        basis[i, j] = 1.0
        basis[j, i] = 1.0
        bases.append(basis)
    return np.asarray(bases), tri


def safe_inv(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix, rcond=1.0e-12)


def batched_inv(matrices: np.ndarray) -> tuple[np.ndarray, bool]:
    try:
        inv = np.linalg.inv(matrices)
        if not np.isfinite(inv).all():
            return inv, False
        return inv, True
    except np.linalg.LinAlgError:
        return np.full_like(matrices, np.nan), False


def inverse_space_a_initial(k: np.ndarray, sk_matrix: np.ndarray) -> np.ndarray:
    valid_k, invs = [], []
    for kk, matrix in zip(k, sk_matrix):
        inv = safe_inv(matrix)
        if np.isfinite(inv).all():
            valid_k.append(float(kk))
            invs.append(inv)
    if len(valid_k) < 3:
        return safe_inv(np.mean(sk_matrix, axis=0))
    design = np.column_stack([np.ones(len(valid_k)), np.asarray(valid_k)])
    invs = np.stack(invs, axis=0)
    n_comp = sk_matrix.shape[1]
    a0 = np.zeros((n_comp, n_comp), dtype=float)
    for i in range(n_comp):
        for j in range(i + 1):
            coeff, *_ = np.linalg.lstsq(design, invs[:, i, j], rcond=None)
            a0[i, j] = a0[j, i] = coeff[0]
    return a0


def finite_fit_data(k2: np.ndarray, sk_matrix: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k2[mask], dtype=float)
    target = np.asarray(sk_matrix[mask], dtype=float)
    finite = np.isfinite(k) & np.isfinite(target).all(axis=(1, 2))
    return k[finite], target[finite]


def scaled_lower_target(target: np.ndarray, tri: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    target_lower = target[:, tri[0], tri[1]]
    scale = np.std(target_lower, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    return target_lower, scale


def s_jacobian_lower(
    s_stack: np.ndarray,
    bases: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray],
    scale: np.ndarray,
    coeff: np.ndarray,
) -> np.ndarray:
    """Return d residual_lower / d params for dM/dp = coeff[k,p] * basis[p]."""
    ds = -np.einsum("kab,pbc,kcd,kp->kpad", s_stack, bases, s_stack, coeff, optimize=True)
    ds_lower = ds[:, :, tri[0], tri[1]]
    ds_lower = np.transpose(ds_lower, (0, 2, 1))
    return ds_lower / scale[None, :, None]


def fit_matrix_oz_s_loss(
    k2: np.ndarray,
    sk_matrix: np.ndarray,
    mask: np.ndarray,
    *,
    max_nfev: int = 1000,
) -> tuple[np.ndarray, int, float, bool]:
    """Neutral matrix OZ fallback: S(k)=inv(A+k^2 L), fitted in S-space."""
    k, target = finite_fit_data(k2, sk_matrix, mask)
    if len(k) < 3:
        return np.full(sk_matrix.shape[1:], np.nan), len(k), math.nan, False
    n_comp = target.shape[1]
    bases, tri = symmetric_bases(n_comp)
    n_tri = len(tri[0])
    target_lower, scale = scaled_lower_target(target, tri)
    a0 = inverse_space_a_initial(k, target)
    l0 = np.zeros_like(a0)
    initial = np.r_[pack_lower(a0), pack_lower(l0)]
    reg_scale = 10000.0
    bad_res = np.full(target_lower.size + n_tri, 1.0e8, dtype=float)
    bad_jac = np.zeros((bad_res.size, initial.size), dtype=float)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = unpack_lower(params[:n_tri], n_comp)
        l_mat = unpack_lower(params[n_tri:], n_comp)
        return a, l_mat

    def residuals(params: np.ndarray) -> np.ndarray:
        a, l_mat = unpack(params)
        mats = a[None, :, :] + k[:, None, None] * l_mat[None, :, :]
        pred, ok = batched_inv(mats)
        if not ok:
            return bad_res.copy()
        data_res = (pred[:, tri[0], tri[1]] - target_lower) / scale
        reg = pack_lower(l_mat - l0) / reg_scale
        return np.r_[data_res.ravel(), reg]

    def jacobian(params: np.ndarray) -> np.ndarray:
        a, l_mat = unpack(params)
        mats = a[None, :, :] + k[:, None, None] * l_mat[None, :, :]
        pred, ok = batched_inv(mats)
        if not ok:
            return bad_jac.copy()
        coeff_a = np.ones((len(k), n_tri), dtype=float)
        coeff_l = np.repeat(k[:, None], n_tri, axis=1)
        jac_a = s_jacobian_lower(pred, bases, tri, scale, coeff_a)
        jac_l = s_jacobian_lower(pred, bases, tri, scale, coeff_l)
        jac_data = np.concatenate([jac_a, jac_l], axis=2).reshape(target_lower.size, initial.size)
        jac_reg = np.zeros((n_tri, initial.size), dtype=float)
        jac_reg[:, n_tri:] = np.eye(n_tri) / reg_scale
        return np.vstack([jac_data, jac_reg])

    try:
        result = least_squares(
            residuals,
            initial,
            jac=jacobian,
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=max_nfev,
        )
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError("fit failed")
        a, _l_mat = unpack(result.x)
        s0 = safe_inv(a)
        return s0, len(k), float(np.sqrt(np.mean(residuals(result.x) ** 2))), bool(np.isfinite(s0).all())
    except Exception:
        return np.full((n_comp, n_comp), np.nan), len(k), math.nan, False


def read_debye_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    data = data[["x_ion", "debye_length_nm"]].apply(pd.to_numeric, errors="coerce").dropna()
    data = data.sort_values("x_ion")
    return data["x_ion"].to_numpy(dtype=float), data["debye_length_nm"].to_numpy(dtype=float)


def kappa_d_from_x_x(x_x: float, debye_x: np.ndarray, debye_nm: np.ndarray) -> float:
    lambda_nm = float(np.interp(x_x, debye_x, debye_nm))
    return 1.0 / (10.0 * lambda_nm)


def charge_mode_matrix(fractions: np.ndarray) -> np.ndarray:
    y = np.diag(np.sqrt(fractions))
    denominator = float(VALENCES @ np.diag(fractions) @ VALENCES)
    return y @ np.outer(VALENCES, VALENCES) @ y / denominator


def fit_matrix_dh_s_loss_result(
    k2: np.ndarray,
    sk_matrix: np.ndarray,
    mask: np.ndarray,
    fractions: np.ndarray,
    kappa_initial: float,
    *,
    fit_kappa: bool,
    initial_yhy: np.ndarray | None = None,
    initial_l: np.ndarray | None = None,
    max_nfev: int = 800,
) -> DHFitResult:
    """Fit DH S-loss model and retain YHY/L for fitted-kappa warm starts."""
    k, target_matrix = finite_fit_data(k2, sk_matrix, mask)
    n_comp = sk_matrix.shape[1]
    if len(k) < 3:
        empty = np.full((n_comp, n_comp), np.nan)
        return DHFitResult(empty, math.nan, len(k), math.nan, False, empty, empty)

    bases, tri = symmetric_bases(n_comp)
    n_tri = len(tri[0])
    target, scale = scaled_lower_target(target_matrix, tri)
    k_scaled = k * K2_SCALE_DH
    c_matrix = charge_mode_matrix(np.asarray(fractions, dtype=float))
    if initial_yhy is None or initial_l is None:
        a0 = inverse_space_a_initial(k_scaled, target_matrix)
        weight0 = kappa_initial**2 / (k_scaled + kappa_initial**2)
        yhy0 = a0 - float(np.mean(weight0)) * c_matrix
        l0 = np.zeros_like(yhy0)
    else:
        yhy0 = np.asarray(initial_yhy, dtype=float)
        l0 = np.asarray(initial_l, dtype=float)
    log_kappa0 = math.log(kappa_initial)
    initial = np.r_[pack_lower(yhy0), pack_lower(l0)]
    if fit_kappa:
        initial = np.r_[initial, log_kappa0]
    reg_scale = 10000.0
    n_reg = n_tri + (1 if fit_kappa else 0)
    bad_res = np.full(target.size + n_reg, 1.0e8, dtype=float)
    bad_jac = np.zeros((bad_res.size, initial.size), dtype=float)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        yhy = unpack_lower(params[:n_tri], n_comp)
        l_mat = unpack_lower(params[n_tri : 2 * n_tri], n_comp)
        kappa = float(math.exp(params[-1])) if fit_kappa else kappa_initial
        return yhy, l_mat, kappa

    def residuals(params: np.ndarray) -> np.ndarray:
        yhy, l_mat, kappa = unpack(params)
        kappa2 = kappa**2
        weight = kappa2 / (k_scaled + kappa2)
        mats = yhy[None, :, :] + k_scaled[:, None, None] * l_mat[None, :, :] + weight[:, None, None] * c_matrix[None, :, :]
        pred, ok = batched_inv(mats)
        if not ok:
            return bad_res.copy()
        data_res = (pred[:, tri[0], tri[1]] - target) / scale
        reg = pack_lower(l_mat - l0) / reg_scale
        if fit_kappa:
            reg = np.r_[reg, (params[-1] - log_kappa0) / 10.0]
        return np.r_[data_res.ravel(), reg]

    def jacobian(params: np.ndarray) -> np.ndarray:
        yhy, l_mat, kappa = unpack(params)
        kappa2 = kappa**2
        weight = kappa2 / (k_scaled + kappa2)
        mats = yhy[None, :, :] + k_scaled[:, None, None] * l_mat[None, :, :] + weight[:, None, None] * c_matrix[None, :, :]
        pred, ok = batched_inv(mats)
        if not ok:
            return bad_jac.copy()
        coeff_yhy = np.ones((len(k), n_tri), dtype=float)
        coeff_l = np.repeat(k_scaled[:, None], n_tri, axis=1)
        jac_yhy = s_jacobian_lower(pred, bases, tri, scale, coeff_yhy)
        jac_l = s_jacobian_lower(pred, bases, tri, scale, coeff_l)
        blocks = [jac_yhy, jac_l]
        if fit_kappa:
            dweight_deta = 2.0 * kappa2 * k_scaled / (k_scaled + kappa2) ** 2
            ds_eta = -np.einsum("kab,bc,kcd,k->kad", pred, c_matrix, pred, dweight_deta, optimize=True)
            jac_eta = (ds_eta[:, tri[0], tri[1]] / scale).reshape(len(k), len(scale), 1)
            blocks.append(jac_eta)
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
        if not np.isfinite(result.x).all():
            raise ValueError("fit failed")
        yhy, l_mat, kappa_fit = unpack(result.x)
        s0 = safe_inv(yhy + c_matrix)
        ok = bool(result.success and np.isfinite(s0).all())
        if not np.isfinite(s0).all():
            s0 = np.full((n_comp, n_comp), np.nan)
        return DHFitResult(
            s0,
            kappa_fit,
            len(k),
            float(np.sqrt(np.mean(residuals(result.x) ** 2))),
            ok,
            yhy,
            l_mat,
        )
    except Exception:
        empty = np.full((n_comp, n_comp), np.nan)
        return DHFitResult(empty, math.nan, len(k), math.nan, False, empty, empty)


def fit_matrix_dh_s_loss(
    k2: np.ndarray,
    sk_matrix: np.ndarray,
    mask: np.ndarray,
    fractions: np.ndarray,
    kappa_initial: float,
    *,
    fit_kappa: bool,
    initial_yhy: np.ndarray | None = None,
    initial_l: np.ndarray | None = None,
    max_nfev: int = 800,
) -> tuple[np.ndarray, float, int, float, bool]:
    """Fit S(k)=inv(YHY+k_ang^2 L+C*kappa^2/(k_ang^2+kappa^2))."""
    return fit_matrix_dh_s_loss_result(
        k2,
        sk_matrix,
        mask,
        fractions,
        kappa_initial,
        fit_kappa=fit_kappa,
        initial_yhy=initial_yhy,
        initial_l=initial_l,
        max_nfev=max_nfev,
    ).public_tuple()


def process_sk_file(payload: tuple[common.SkFile, tuple[float, ...], np.ndarray, np.ndarray]) -> list[dict[str, object]]:
    item, kcuts, debye_x, debye_nm = payload
    rows = []
    column_index, data = common.read_allsk(item.path)
    k2 = np.asarray(data[:, column_index["k^2"]], dtype=float)
    allpair_matrix = common.assemble_matrix(data, column_index, common.ALLPAIR_MAP)
    kappa_input = kappa_d_from_x_x(item.x_x, debye_x, debye_nm)
    for kcut in kcuts:
        mask = common.lowk_mask(k2, data, column_index, common.ALLPAIR_COMPONENTS, kcut)
        fixed = fit_matrix_dh_s_loss_result(
            k2,
            allpair_matrix,
            mask,
            item.fractions,
            kappa_input,
            fit_kappa=False,
        )
        fitted = fit_matrix_dh_s_loss_result(
            k2,
            allpair_matrix,
            mask,
            item.fractions,
            kappa_input,
            fit_kappa=True,
            initial_yhy=fixed.yhy,
            initial_l=fixed.l_mat,
        )
        for method, result in [("DH S-loss, fixed kappa_D", fixed), ("DH S-loss, fitted kappa_D", fitted)]:
            row = common.base_s0_row("matrix_DH_s_loss", method, item, kcut)
            row["kappa_D_input"] = kappa_input
            row["kappa_D_fit"] = result.kappa_fit
            row["n_fit_points"] = result.n_fit
            row["fit_rmse_scaled"] = result.rmse
            row["fit_success"] = int(result.success)
            common.set_allpair_fields(row, result.s0)
            rows.append(row)
    return rows


def build_s0_table(data_dir: Path, source: str, kcuts: tuple[float, ...], debye_csv: Path, n_procs: int = 4) -> object:
    debye_x, debye_nm = read_debye_table(debye_csv)
    files = common.discover_split0_files(data_dir, source)
    payloads = [(item, kcuts, debye_x, debye_nm) for item in files]
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
    parser.add_argument("--kcuts", type=common.parse_kcuts, default=common.DEFAULT_KCUTS)
    parser.add_argument("--debye-csv", type=Path, default=DEBYE_CSV)
    parser.add_argument("--n-procs", type=int, default=4, help="Number of file-level worker processes; use 1 for sequential.")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "S0_NaClFWater_matrix_DH_s_loss_split0.csv")
    args = parser.parse_args()
    table = build_s0_table(
        args.data_dir.expanduser().resolve(),
        args.source,
        args.kcuts,
        args.debye_csv.expanduser().resolve(),
        args.n_procs,
    )
    common.write_s0_csv(table, args.output)
    print(f"wrote {args.output}")
    print(table.groupby(["method", "kcut"])["fit_success"].sum().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
