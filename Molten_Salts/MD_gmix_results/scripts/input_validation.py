"""Strict validation helpers for canonical S(0) input tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def require_finite_numeric(
    frame: pd.DataFrame, columns: tuple[str, ...], path: Path
) -> None:
    """Convert selected columns to numeric values and reject gaps/infinities."""
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{path.name}: {column} must contain only finite numeric values."
            )
        frame[column] = converted


def require_integer_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    path: Path,
    *,
    nonnegative: bool,
) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{path.name}: {column} must contain integers.")
        if nonnegative and np.any(values < 0.0):
            raise ValueError(
                f"{path.name}: {column} must contain nonnegative counts."
            )


def require_converged(series: pd.Series, path: Path, column: str) -> None:
    """Parse explicit boolean/0/1 encodings without truth-casting strings."""
    if pd.api.types.is_bool_dtype(series.dtype):
        converged = series.to_numpy(dtype=bool)
    elif pd.api.types.is_numeric_dtype(series.dtype):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"{path.name}: {column} must contain only 0/1 values.")
        converged = values == 1.0
    else:
        normalized = series.astype("string").str.strip().str.lower()
        mapping = {"true": True, "false": False, "1": True, "0": False}
        if normalized.isna().any() or not normalized.isin(mapping).all():
            raise ValueError(
                f"{path.name}: {column} must contain only true/false or 0/1."
            )
        converged = normalized.map(mapping).to_numpy(dtype=bool)
    if not converged.all():
        raise ValueError(f"{path.name}: at least one selected S(0) fit did not converge.")


def require_count_fraction(
    frame: pd.DataFrame,
    path: Path,
    *,
    composition_column: str,
    solute_count_column: str,
    solvent_count_column: str,
) -> None:
    composition = frame[composition_column].to_numpy(dtype=float)
    if np.any((composition < 0.0) | (composition > 1.0)):
        raise ValueError(f"{path.name}: {composition_column} must be within [0, 1].")
    formula_count = (
        frame[solute_count_column].to_numpy(dtype=float)
        + frame[solvent_count_column].to_numpy(dtype=float)
    )
    if np.any(formula_count <= 0.0):
        raise ValueError(f"{path.name}: salt formula-unit count must be positive.")
    count_fraction = (
        frame[solute_count_column].to_numpy(dtype=float) / formula_count
    )
    if not np.allclose(composition, count_fraction, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{path.name}: {composition_column} is inconsistent with ion counts."
        )
