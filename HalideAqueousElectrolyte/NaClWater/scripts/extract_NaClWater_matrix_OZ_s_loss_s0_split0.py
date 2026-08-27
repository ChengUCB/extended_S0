#!/usr/bin/env python3
"""Extract split-0 S0 values with matrix OZ S-loss fits."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import sk_s0_common as common


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Sk_files" if (SCRIPT_DIR / "Sk_files").is_dir() else SCRIPT_DIR


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
    """Fit S(k) = inv(A + k^2 L) by S-space residuals."""
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


def build_s0_table(data_dir: Path, source: str, kcuts: tuple[float, ...], split: int = 0) -> object:
    rows = []
    files = common.discover_split_files(data_dir, source, split)
    for file_index, item in enumerate(files, start=1):
        print(f"{file_index:02d}/{len(files):02d} {item.path.name}", flush=True)
        column_index, data = common.read_allsk(item.path)
        k2 = np.asarray(data[:, column_index["k^2"]], dtype=float)
        ox_matrix = common.assemble_matrix(data, column_index, common.OX_MAP)
        allpair_matrix = common.assemble_matrix(data, column_index, common.ALLPAIR_MAP)
        for kcut in kcuts:
            row = common.base_s0_row("matrix_OZ_s_loss", "direct O/X + reconstructed O/X from Na/Cl/O", item, kcut)

            direct_mask = common.lowk_mask(k2, data, column_index, common.OX_COMPONENTS, kcut)
            s0_direct, n_direct, rmse_direct, ok_direct = fit_matrix_oz_s_loss(
                k2,
                ox_matrix,
                direct_mask,
                max_nfev=2000,
            )
            common.set_direct_ox_fields(row, s0_direct)
            direct_oo = row["S0_OO"]

            allpair_mask = common.lowk_mask(k2, data, column_index, common.ALLPAIR_COMPONENTS, kcut)
            s0_allpair, n_allpair, rmse_allpair, ok_allpair = fit_matrix_oz_s_loss(
                k2,
                allpair_matrix,
                allpair_mask,
                max_nfev=500,
            )
            common.set_allpair_fields(row, item, s0_allpair)
            row["S0_OO"] = direct_oo

            row["n_fit_points"] = max(n_direct, n_allpair)
            row["fit_rmse_scaled"] = float(np.nanmean([rmse_direct, rmse_allpair]))
            row["fit_success"] = int(ok_direct and ok_allpair)
            rows.append(row)
    return common.ordered_s0_table(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--source", default="main")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--kcuts", type=common.parse_kcuts, default=common.DEFAULT_KCUTS)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "S0_NaClWater_matrix_OZ_s_loss_split0.csv")
    args = parser.parse_args()
    table = build_s0_table(args.data_dir.expanduser().resolve(), args.source, args.kcuts, args.split)
    common.write_s0_csv(table, args.output)
    print(f"wrote {args.output}")
    print(table.groupby(["method", "kcut"])["fit_success"].sum().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
