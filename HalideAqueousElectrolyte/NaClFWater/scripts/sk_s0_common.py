"""Common file I/O, preprocessing, and CSV helpers for NaClFWater split-0 S0 extraction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_KCUTS = (0.0025, 0.005, 0.01)
DAT_RE = re.compile(
    r"^O(?P<o>\d+)-Na(?P<na>\d+)-Cl(?P<cl>\d+)-F(?P<f>\d+)(?:-(?P<split>\d+))?-allSk\.dat$"
)

COMPONENTS = ("O", "Na", "Cl", "F")
ALLPAIR_COMPONENTS = ("OO", "ONa", "OCl", "OF", "NaNa", "NaCl", "NaF", "ClCl", "ClF", "FF")
ALLPAIR_MAP = {
    "OO": (0, 0),
    "ONa": (0, 1),
    "OCl": (0, 2),
    "OF": (0, 3),
    "NaNa": (1, 1),
    "NaCl": (1, 2),
    "NaF": (1, 3),
    "ClCl": (2, 2),
    "ClF": (2, 3),
    "FF": (3, 3),
}
S0_COLUMNS = tuple(f"S0_{component}" for component in ALLPAIR_COMPONENTS)
OUTPUT_COLUMNS = (
    "algorithm",
    "method",
    "kcut",
    "file",
    "split",
    "source",
    "n_O",
    "n_Na",
    "n_Cl",
    "n_F",
    "n_X",
    "x_O",
    "x_Na",
    "x_Cl",
    "x_F",
    "x_X",
    "x_F_anion",
    "n_fit_points",
    "fit_rmse_scaled",
    "fit_success",
    "kappa_D_input",
    "kappa_D_fit",
    *S0_COLUMNS,
)


@dataclass(frozen=True)
class SkFile:
    path: Path
    n_o: int
    n_na: int
    n_cl: int
    n_f: int
    split: int
    source: str = "main"

    @property
    def n_x(self) -> int:
        return self.n_na + self.n_cl + self.n_f

    @property
    def n_total(self) -> int:
        return self.n_o + self.n_x

    @property
    def fractions(self) -> np.ndarray:
        return np.array([self.n_o, self.n_na, self.n_cl, self.n_f], dtype=float) / self.n_total

    @property
    def x_o(self) -> float:
        return self.n_o / self.n_total

    @property
    def x_na(self) -> float:
        return self.n_na / self.n_total

    @property
    def x_cl(self) -> float:
        return self.n_cl / self.n_total

    @property
    def x_f(self) -> float:
        return self.n_f / self.n_total

    @property
    def x_x(self) -> float:
        return self.n_x / self.n_total

    @property
    def x_f_anion(self) -> float:
        denominator = self.n_cl + self.n_f
        return self.n_f / denominator if denominator else math.nan


def parse_kcuts(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_sk_file(path: Path, source: str = "main") -> SkFile | None:
    match = DAT_RE.match(path.name)
    if match is None:
        return None
    return SkFile(
        path=path,
        n_o=int(match.group("o")),
        n_na=int(match.group("na")),
        n_cl=int(match.group("cl")),
        n_f=int(match.group("f")),
        split=int(match.group("split") or 0),
        source=source,
    )


def discover_split_files(data_dir: Path, source: str = "main", split: int = 0) -> list[SkFile]:
    files = []
    seen: set[tuple[int, int, int, int]] = set()
    for path in sorted(data_dir.glob("O*-Na*-Cl*-F*-allSk.dat")):
        item = parse_sk_file(path, source)
        if item is None or item.split != split:
            continue
        key = (item.n_o, item.n_na, item.n_cl, item.n_f)
        if key in seen:
            raise ValueError(f"duplicate split_{split} composition {key} in {data_dir}")
        seen.add(key)
        files.append(item)
    if not files:
        raise FileNotFoundError(f"no split_{split} allSk.dat files in {data_dir}")
    return sorted(files, key=lambda item: (item.x_x, item.x_f_anion))


def discover_split0_files(data_dir: Path, source: str = "main") -> list[SkFile]:
    return discover_split_files(data_dir, source, split=0)


def read_allsk(path: Path) -> tuple[dict[str, int], np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
    if not header.startswith("#"):
        raise ValueError(f"{path} has no # header")
    column_index = {name: i for i, name in enumerate(header[1:].split())}
    if "k^2" not in column_index:
        raise ValueError(f"{path} is missing k^2 column")
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return column_index, data


def lowk_mask(k2: np.ndarray, data: np.ndarray, column_index: dict[str, int], components: tuple[str, ...], kcut: float) -> np.ndarray:
    mask = (k2 > 0.0) & (k2 <= kcut) & np.isfinite(k2)
    for component in components:
        if component not in column_index:
            return np.zeros_like(mask, dtype=bool)
        mask &= np.isfinite(data[:, column_index[component]])
        error_column = f"{component}_error"
        if error_column in column_index:
            errors = data[:, column_index[error_column]]
            mask &= np.isfinite(errors) & (errors > 0.0)
    return mask


def assemble_matrix(data: np.ndarray, column_index: dict[str, int], component_map: dict[str, tuple[int, int]]) -> np.ndarray:
    n_comp = max(max(indices) for indices in component_map.values()) + 1
    matrix = np.zeros((len(data), n_comp, n_comp), dtype=float)
    for component, (row, col) in component_map.items():
        values = np.asarray(data[:, column_index[component]], dtype=float)
        matrix[:, row, col] = values
        matrix[:, col, row] = values
    return matrix


def base_s0_row(algorithm: str, method: str, item: SkFile, kcut: float) -> dict[str, object]:
    row: dict[str, object] = {
        "algorithm": algorithm,
        "method": method,
        "kcut": kcut,
        "file": item.path.name,
        "split": item.split,
        "source": item.source,
        "n_O": item.n_o,
        "n_Na": item.n_na,
        "n_Cl": item.n_cl,
        "n_F": item.n_f,
        "n_X": item.n_x,
        "x_O": item.x_o,
        "x_Na": item.x_na,
        "x_Cl": item.x_cl,
        "x_F": item.x_f,
        "x_X": item.x_x,
        "x_F_anion": item.x_f_anion,
        "n_fit_points": 0,
        "fit_rmse_scaled": math.nan,
        "fit_success": 0,
        "kappa_D_input": math.nan,
        "kappa_D_fit": math.nan,
    }
    for column in S0_COLUMNS:
        row[column] = math.nan
    return row


def set_allpair_fields(row: dict[str, object], s0_matrix: np.ndarray) -> None:
    for component, (row_index, col_index) in ALLPAIR_MAP.items():
        row[f"S0_{component}"] = float(s0_matrix[row_index, col_index])


def ordered_s0_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_s0_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, columns=OUTPUT_COLUMNS)
