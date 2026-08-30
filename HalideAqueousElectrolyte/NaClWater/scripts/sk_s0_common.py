"""Common file I/O, preprocessing, and CSV helpers for S0 extraction."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_KCUTS = (0.0025, 0.005, 0.01)
DAT_RE = re.compile(r"^O(?P<o>\d+)-Na(?P<na>\d+)-Cl(?P<cl>\d+)(?:-(?P<split>\d+))?-allSk\.dat$")
TRACKED_SPLIT_CSV_RE = re.compile(r"_split(?P<split>[123])\.csv$")
LEGACY_SPLIT_CONTRACT_SCHEMA = "extended_S0.naclwater_split_provenance.v1"

OX_COMPONENTS = ("OO", "OX", "XX")
ALLPAIR_COMPONENTS = ("OO", "ONa", "OCl", "NaNa", "NaCl", "ClCl")
OX_MAP = {"OO": (0, 0), "OX": (0, 1), "XX": (1, 1)}
ALLPAIR_MAP = {
    "OO": (0, 0),
    "ONa": (0, 1),
    "OCl": (0, 2),
    "NaNa": (1, 1),
    "NaCl": (1, 2),
    "ClCl": (2, 2),
}
S0_COLUMNS = ("S0_OO", "S0_ONa", "S0_OCl", "S0_NaNa", "S0_NaCl", "S0_ClCl", "S0_OX", "S0_XX")
RECONSTRUCTED_COLUMNS = ("S0_reconstructed_OO", "S0_reconstructed_OX", "S0_reconstructed_XX")
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
    "n_X",
    "x_O",
    "x_Na",
    "x_Cl",
    "x_X",
    "n_fit_points",
    "fit_rmse_scaled",
    "fit_success",
    "kappa_D_input",
    "kappa_D_fit",
    *S0_COLUMNS,
    *RECONSTRUCTED_COLUMNS,
)


@dataclass(frozen=True)
class SkFile:
    path: Path
    n_o: int
    n_na: int
    n_cl: int
    split: int
    source: str = "main"

    @property
    def n_x(self) -> int:
        return self.n_na + self.n_cl

    @property
    def n_total(self) -> int:
        return self.n_o + self.n_x

    @property
    def fractions(self) -> np.ndarray:
        return np.array([self.n_o, self.n_na, self.n_cl], dtype=float) / self.n_total

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
    def x_x(self) -> float:
        return self.n_x / self.n_total


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
        split=int(match.group("split") or 0),
        source=source,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tracked_split_csv(
    path: Path, contract_path: Path | None = None
) -> pd.DataFrame:
    """Return the checksum-pinned legacy split CSV with canonical split metadata.

    The nine checked-in split1--3 CSVs historically store ``split = 0`` even
    though their filenames and source ``allSk.dat`` names identify the actual
    split.  Their bytes are retained as scientific source data.  This loader
    validates the tracked contract and corrects only the returned dataframe.
    """
    path = Path(path)
    contract_path = (
        path.parent / "S0_split_provenance.json"
        if contract_path is None
        else Path(contract_path)
    )
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read split provenance contract {contract_path}"
        ) from exc
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != LEGACY_SPLIT_CONTRACT_SCHEMA
        or not isinstance(contract.get("files"), dict)
    ):
        raise RuntimeError(f"Invalid split provenance contract {contract_path}")
    entry = contract["files"].get(path.name)
    required = {"sha256", "raw_split", "effective_split"}
    if not isinstance(entry, dict) or required.difference(entry):
        raise ValueError(f"{path.name} is not declared by {contract_path.name}")
    filename_match = TRACKED_SPLIT_CSV_RE.search(path.name)
    if filename_match is None:
        raise ValueError(f"{path.name} does not identify a canonical split")
    filename_split = int(filename_match.group("split"))
    raw_split = int(entry["raw_split"])
    effective_split = int(entry["effective_split"])
    if raw_split != 0 or effective_split != filename_split:
        raise RuntimeError(
            f"{path.name} split provenance conflicts with its filename"
        )
    if _sha256_file(path) != entry["sha256"]:
        raise RuntimeError(
            f"{path.name} checksum does not match {contract_path.name}; "
            "refusing to reinterpret split metadata"
        )

    frame = pd.read_csv(path)
    if "split" not in frame.columns or "file" not in frame.columns or frame.empty:
        raise ValueError(f"{path.name} does not satisfy the tracked S(0) CSV schema")
    raw_values = pd.to_numeric(frame["split"], errors="coerce")
    if raw_values.isna().any() or not raw_values.eq(raw_split).all():
        raise RuntimeError(f"{path.name} raw split values violate its provenance contract")
    source_suffix = f"-{effective_split}-allSk.dat"
    if not frame["file"].astype(str).str.endswith(source_suffix).all():
        raise RuntimeError(
            f"{path.name} source filenames do not identify split {effective_split}"
        )
    frame = frame.copy()
    frame["split"] = effective_split
    return frame


def discover_split_files(data_dir: Path, source: str = "main", split: int = 0) -> list[SkFile]:
    files = []
    seen: set[tuple[int, int, int]] = set()
    for path in sorted(data_dir.glob("O*-Na*-Cl*-allSk.dat")):
        item = parse_sk_file(path, source)
        if item is None or item.split != split:
            continue
        key = (item.n_o, item.n_na, item.n_cl)
        if key in seen:
            raise ValueError(f"duplicate split_{split} composition {key} in {data_dir}")
        seen.add(key)
        files.append(item)
    if not files:
        raise FileNotFoundError(f"no split_{split} allSk.dat files in {data_dir}")
    return sorted(files, key=lambda item: item.x_x)


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
    for component, (i, j) in component_map.items():
        values = np.asarray(data[:, column_index[component]], dtype=float)
        matrix[:, i, j] = values
        matrix[:, j, i] = values
    return matrix


def reconstruct_ox_xx(item: SkFile, s0_matrix: np.ndarray) -> tuple[float, float, float]:
    s0_oo = float(s0_matrix[0, 0])
    s0_ox = math.sqrt(item.x_na / item.x_x) * float(s0_matrix[0, 1]) + math.sqrt(item.x_cl / item.x_x) * float(s0_matrix[0, 2])
    s0_xx = (
        (item.x_na / item.x_x) * float(s0_matrix[1, 1])
        + (item.x_cl / item.x_x) * float(s0_matrix[2, 2])
        + 2.0 * math.sqrt(item.x_na * item.x_cl) / item.x_x * float(s0_matrix[1, 2])
    )
    return s0_oo, s0_ox, s0_xx


def reconstruct_ox_xx_from_components(item: SkFile, s0: dict[str, float]) -> tuple[float, float, float]:
    s0_ox = math.sqrt(item.x_na / item.x_x) * s0["ONa"] + math.sqrt(item.x_cl / item.x_x) * s0["OCl"]
    s0_xx = (
        (item.x_na / item.x_x) * s0["NaNa"]
        + (item.x_cl / item.x_x) * s0["ClCl"]
        + 2.0 * math.sqrt(item.x_na * item.x_cl) / item.x_x * s0["NaCl"]
    )
    return s0["OO"], s0_ox, s0_xx


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
        "n_X": item.n_x,
        "x_O": item.x_o,
        "x_Na": item.x_na,
        "x_Cl": item.x_cl,
        "x_X": item.x_x,
        "n_fit_points": 0,
        "fit_rmse_scaled": math.nan,
        "fit_success": 0,
        "kappa_D_input": math.nan,
        "kappa_D_fit": math.nan,
    }
    for column in (*S0_COLUMNS, *RECONSTRUCTED_COLUMNS):
        row[column] = math.nan
    return row


def set_direct_ox_fields(row: dict[str, object], s0_matrix: np.ndarray) -> None:
    row["S0_OO"] = float(s0_matrix[0, 0])
    row["S0_OX"] = float(s0_matrix[0, 1])
    row["S0_XX"] = float(s0_matrix[1, 1])
    row["S0_reconstructed_OO"] = row["S0_OO"]
    row["S0_reconstructed_OX"] = row["S0_OX"]
    row["S0_reconstructed_XX"] = row["S0_XX"]


def set_allpair_fields(row: dict[str, object], item: SkFile, s0_matrix: np.ndarray) -> None:
    for component, (i, j) in ALLPAIR_MAP.items():
        row[f"S0_{component}"] = float(s0_matrix[i, j])
    s0_oo, s0_ox, s0_xx = reconstruct_ox_xx(item, s0_matrix)
    row["S0_reconstructed_OO"] = s0_oo
    row["S0_reconstructed_OX"] = s0_ox
    row["S0_reconstructed_XX"] = s0_xx


def ordered_s0_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_s0_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, columns=OUTPUT_COLUMNS)
