#!/usr/bin/env python3
"""Estimate S0 values from Fe-Cu-Ni allSk.dat files using a matrix OZ fit."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ELEMENTS = ("Fe", "Cu", "Ni")
FIT_COMPONENTS = ("FeFe", "FeCu", "CuCu", "FeNi", "CuNi", "NiNi")
COMPONENT_INDEX = {
    "FeFe": (0, 0),
    "FeCu": (1, 0),
    "CuCu": (1, 1),
    "FeNi": (2, 0),
    "CuNi": (2, 1),
    "NiNi": (2, 2),
}
SPLITS = (0, 1, 2, 3)
DAT_FILENAME_RE = re.compile(
    r"^Fe-(?P<fe>[^-]+)-Cu-(?P<cu>[^-]+)-Ni-(?P<ni>[^-]+)"
    r"(?:-(?P<split>\d+))?-allSk\.dat$"
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DAT_DIR = SCRIPT_DIR.parent / "Sk-FeCuNi"


@dataclass(frozen=True)
class DatFile:
    path: Path
    x_fe: float
    x_cu: float
    x_ni: float
    split: int

    @property
    def composition(self) -> tuple[float, float, float]:
        return self.x_fe, self.x_cu, self.x_ni


@dataclass(frozen=True)
class MatrixFitResult:
    s0: np.ndarray
    s0_error: np.ndarray
    n_fit_points: int
    n_skipped_matrices: int
    success: bool


def parse_dat_filename(path: Path) -> DatFile | None:
    match = DAT_FILENAME_RE.match(path.name)
    if match is None:
        return None

    return DatFile(
        path=path,
        x_fe=float(match.group("fe")),
        x_cu=float(match.group("cu")),
        x_ni=float(match.group("ni")),
        split=int(match.group("split") or 0),
    )


def load_allsk(path: Path) -> tuple[dict[str, int], np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline().strip()

    if not header_line.startswith("#"):
        raise ValueError(f"{path} does not start with a header line")

    header = header_line[1:].split()
    column_index = {name: index for index, name in enumerate(header)}
    missing = ["k^2"]
    for component in FIT_COMPONENTS:
        missing.extend((component, f"{component}_error"))
    missing = [name for name in missing if name not in column_index]
    if missing:
        raise ValueError(f"{path} is missing expected column(s): {', '.join(missing)}")

    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return column_index, data


def allsk_to_matrix(column_index: dict[str, int], data: np.ndarray) -> np.ndarray:
    sk_matrix = np.zeros((data.shape[0], len(ELEMENTS), len(ELEMENTS)), dtype=float)
    for component, (i, j) in COMPONENT_INDEX.items():
        values = data[:, column_index[component]]
        sk_matrix[:, i, j] = values
        sk_matrix[:, j, i] = values
    return sk_matrix


def nan_matrix_fit() -> MatrixFitResult:
    s0 = np.full((len(ELEMENTS), len(ELEMENTS)), math.nan, dtype=float)
    return MatrixFitResult(
        s0=s0,
        s0_error=s0.copy(),
        n_fit_points=0,
        n_skipped_matrices=0,
        success=False,
    )


def estimate_matrix_s0(
    k_squared: np.ndarray,
    sk_matrix: np.ndarray,
    kcut: float,
    min_points: int,
) -> MatrixFitResult:
    mask = (
        (k_squared > 0.0)
        & (k_squared <= kcut)
        & np.isfinite(k_squared)
        & np.isfinite(sk_matrix).all(axis=(1, 2))
    )

    selected_k = k_squared[mask]
    selected_s = sk_matrix[mask]
    if selected_k.size < min_points:
        return nan_matrix_fit()

    valid_k: list[float] = []
    inverse_matrices: list[np.ndarray] = []
    skipped = 0
    for kk, matrix in zip(selected_k, selected_s):
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            skipped += 1
            continue
        if not np.isfinite(inverse).all():
            skipped += 1
            continue
        valid_k.append(float(kk))
        inverse_matrices.append(inverse)

    if len(valid_k) < min_points:
        result = nan_matrix_fit()
        return MatrixFitResult(
            s0=result.s0,
            s0_error=result.s0_error,
            n_fit_points=len(valid_k),
            n_skipped_matrices=skipped,
            success=False,
        )

    x = np.asarray(valid_k, dtype=float)
    sinv = np.stack(inverse_matrices, axis=0)
    design = np.column_stack([np.ones_like(x), x])
    design_cov = np.linalg.pinv(design.T @ design)

    n_comp = len(ELEMENTS)
    a_matrix = np.zeros((n_comp, n_comp), dtype=float)
    l_matrix = np.zeros((n_comp, n_comp), dtype=float)
    a_variance = np.zeros((n_comp, n_comp), dtype=float)

    for component, (i, j) in COMPONENT_INDEX.items():
        del component
        y = sinv[:, i, j]
        coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coeff
        dof = max(1, x.size - design.shape[1])
        sigma2 = float(np.sum(residual * residual) / dof)
        coeff_cov = design_cov * sigma2

        a_matrix[i, j] = a_matrix[j, i] = coeff[0]
        l_matrix[i, j] = l_matrix[j, i] = coeff[1]
        a_variance[i, j] = a_variance[j, i] = max(float(coeff_cov[0, 0]), 0.0)

    del l_matrix
    try:
        s0 = np.linalg.inv(a_matrix)
    except np.linalg.LinAlgError:
        result = nan_matrix_fit()
        return MatrixFitResult(
            s0=result.s0,
            s0_error=result.s0_error,
            n_fit_points=x.size,
            n_skipped_matrices=skipped,
            success=False,
        )

    if not np.isfinite(s0).all():
        result = nan_matrix_fit()
        return MatrixFitResult(
            s0=result.s0,
            s0_error=result.s0_error,
            n_fit_points=x.size,
            n_skipped_matrices=skipped,
            success=False,
        )

    s0_variance = np.zeros_like(s0)
    for _, (a, b) in COMPONENT_INDEX.items():
        entry_variance = a_variance[a, b]
        if not math.isfinite(entry_variance):
            continue
        perturbation = np.zeros_like(s0)
        perturbation[a, b] = 1.0
        perturbation[b, a] = 1.0
        if a == b:
            perturbation[a, b] = 1.0

        derivative = -s0 @ perturbation @ s0
        s0_variance += derivative * derivative * entry_variance

    s0_error = np.sqrt(np.maximum(s0_variance, 0.0))
    return MatrixFitResult(
        s0=s0,
        s0_error=s0_error,
        n_fit_points=x.size,
        n_skipped_matrices=skipped,
        success=True,
    )


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.12g}"


def output_header() -> list[str]:
    header = ["row_id", "x_Fe", "x_Cu", "x_Ni"]
    for split in SPLITS:
        for component in FIT_COMPONENTS:
            header.extend(
                (
                    f"split{split}_{component}_S0",
                    f"split{split}_{component}_S0_error",
                )
            )
    return header


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Fe-Cu-Ni S0 matrices from split0 through split3 allSk.dat files "
            "using S(k) = [A + k^2 L]^-1."
        )
    )
    parser.add_argument(
        "dat_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_DAT_DIR,
        help="directory containing Fe-Cu-Ni split0/full-trajectory *allSk.dat files",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        help="directory containing Fe-Cu-Ni split1-split3 *allSk.dat files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("S0_results_FeCuNi_matrix.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--kcut",
        type=float,
        default=0.02,
        help="maximum k^2 value included in the matrix OZ fit",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="minimum number of invertible S(k) matrices required for one fit",
    )
    return parser


def composition_key(dat_file: DatFile) -> tuple[float, float, float]:
    return dat_file.x_fe, dat_file.x_cu, dat_file.x_ni


def gather_dat_files(dat_dir: Path, splits_dir: Path):
    by_composition: dict[tuple[float, float, float], dict[int, DatFile]] = {}
    skipped = []

    for directory in (dat_dir, splits_dir):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("Fe-*-Cu-*-Ni-*-allSk.dat")):
            parsed = parse_dat_filename(path)
            if parsed is None:
                continue
            if parsed.split not in SPLITS:
                skipped.append(path)
                continue

            key = composition_key(parsed)
            by_composition.setdefault(key, {})
            if parsed.split in by_composition[key]:
                existing = by_composition[key][parsed.split].path
                raise ValueError(
                    f"duplicate split{parsed.split} files for Fe={key[0]}, "
                    f"Cu={key[1]}, Ni={key[2]}: {existing} and {path}"
                )
            by_composition[key][parsed.split] = parsed

    return by_composition, skipped


def fit_one_file(dat_file: DatFile, args: argparse.Namespace) -> tuple[dict[str, str], MatrixFitResult]:
    column_index, data = load_allsk(dat_file.path)
    result = estimate_matrix_s0(
        data[:, column_index["k^2"]],
        allsk_to_matrix(column_index, data),
        kcut=args.kcut,
        min_points=args.min_points,
    )

    values = {}
    for component, (i, j) in COMPONENT_INDEX.items():
        values[f"split{dat_file.split}_{component}_S0"] = format_float(result.s0[i, j])
        values[f"split{dat_file.split}_{component}_S0_error"] = format_float(result.s0_error[i, j])
    return values, result


def nan_split_values(split: int) -> dict[str, str]:
    values = {}
    for component in FIT_COMPONENTS:
        values[f"split{split}_{component}_S0"] = "nan"
        values[f"split{split}_{component}_S0_error"] = "nan"
    return values


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.dat_dir = args.dat_dir.expanduser().resolve()
    if args.splits_dir is None:
        args.splits_dir = args.dat_dir / "splits"
    else:
        args.splits_dir = args.splits_dir.expanduser().resolve()
    args.output = args.output.expanduser()
    return args


def main() -> int:
    args = resolve_args(build_parser().parse_args())
    by_composition, skipped = gather_dat_files(args.dat_dir, args.splits_dir)

    rows = []
    fit_failures = 0
    missing_splits = 0
    skipped_matrices = 0
    min_fit_points = None
    max_fit_points = 0
    for row_id, key in enumerate(sorted(by_composition)):
        split_files = by_composition[key]
        split0_or_first = split_files.get(0) or split_files[sorted(split_files)[0]]
        x_fe, x_cu, x_ni = split0_or_first.composition
        row = {
            "row_id": str(row_id),
            "x_Fe": format_float(x_fe),
            "x_Cu": format_float(x_cu),
            "x_Ni": format_float(x_ni),
        }

        for split in SPLITS:
            dat_file = split_files.get(split)
            if dat_file is None:
                missing_splits += 1
                row.update(nan_split_values(split))
                continue

            values, result = fit_one_file(dat_file, args)
            row.update(values)
            if not result.success:
                fit_failures += 1
            skipped_matrices += result.n_skipped_matrices
            max_fit_points = max(max_fit_points, result.n_fit_points)
            if min_fit_points is None:
                min_fit_points = result.n_fit_points
            else:
                min_fit_points = min(min_fit_points, result.n_fit_points)

        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_header(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"input dat_dir: {args.dat_dir}")
    print(f"input splits_dir: {args.splits_dir}")
    if min_fit_points is not None:
        print(f"fit points per available split: min {min_fit_points}, max {max_fit_points}")
    if skipped:
        print(f"skipped {len(skipped)} non-split0-3 allSk.dat file(s)")
    if missing_splits:
        print(f"warning: {missing_splits} split slot(s) were missing and filled with nan")
    if skipped_matrices:
        print(f"warning: skipped {skipped_matrices} singular or invalid S(k) matrix row(s)")
    if fit_failures:
        print(f"warning: {fit_failures} matrix fit(s) returned nan")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
