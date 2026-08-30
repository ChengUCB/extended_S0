"""Provenance and fail-closed output helpers for the Gmix workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundled_method_contract(
    path: Path, bundled_csv: Path, allowed_methods: tuple[str, ...]
) -> str:
    """Label only byte-identical copies of the checksum-pinned legacy table."""
    sidecar = bundled_csv.with_suffix(".provenance.json")
    try:
        contract = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read bundled method contract {sidecar}; refusing to infer method."
        ) from exc
    required = {"schema", "csv", "csv_sha256", "method"}
    missing = (
        required.difference(contract) if isinstance(contract, dict) else required
    )
    if missing:
        raise RuntimeError(
            f"{sidecar.name}: invalid bundled method contract; missing {sorted(missing)}"
        )
    if (
        contract["schema"] != "extended_S0.s0_method_provenance.v1"
        or contract["csv"] != bundled_csv.name
        or contract["method"] not in allowed_methods
    ):
        raise RuntimeError(f"{sidecar.name}: invalid bundled method contract values")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != contract["csv_sha256"]:
        if path.resolve() == bundled_csv.resolve():
            raise RuntimeError(
                f"{bundled_csv.name}: checksum does not match {sidecar.name}; "
                "refusing to infer method."
            )
        raise ValueError(
            f"{path.name}: missing columns ['method']; its checksum is not the "
            "bundled legacy contract, so it must declare method explicitly."
        )
    return contract["method"]


def build_provenance(
    inputs: dict[float, Path], experimental_input: Path | None = None
) -> tuple[dict, str]:
    records = [
        {
            "kcut_cycles2_per_A2": float(kcut),
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for kcut, path in sorted(inputs.items())
    ]
    provenance = {"s0_inputs": records}
    if experimental_input is not None:
        provenance["experimental_input"] = {
            "path": str(experimental_input.resolve()),
            "sha256": sha256_file(experimental_input),
        }
    identity_payload = {
        "s0_inputs": [
            {
                "kcut_cycles2_per_A2": record["kcut_cycles2_per_A2"],
                "sha256": record["sha256"],
            }
            for record in records
        ]
    }
    if experimental_input is not None:
        identity_payload["experimental_input"] = {
            "sha256": provenance["experimental_input"]["sha256"]
        }
    canonical = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    identity = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return provenance, identity


def provenance_columns(
    provenance: dict, identity: str, fit_method: str, source: str
) -> dict[str, str]:
    columns = {
        "fit_method": fit_method,
        "input_identity_sha256": identity,
        "input_source": source,
    }
    for record in provenance["s0_inputs"]:
        prefix = f"kcut_{record['kcut_cycles2_per_A2']:g}".replace(".", "p")
        columns[f"{prefix}__input_path"] = record["path"]
        columns[f"{prefix}__input_sha256"] = record["sha256"]
    experimental = provenance.get("experimental_input")
    columns["experimental_input_path"] = (
        "" if experimental is None else experimental["path"]
    )
    columns["experimental_input_sha256"] = (
        "" if experimental is None else experimental["sha256"]
    )
    return columns


def require_provenance_unchanged(provenance: dict) -> None:
    records = list(provenance["s0_inputs"])
    experimental = provenance.get("experimental_input")
    if experimental is not None:
        records.append(experimental)
    for record in records:
        path = Path(record["path"])
        try:
            observed = sha256_file(path)
        except OSError as exc:
            raise RuntimeError(
                f"Input disappeared or became unreadable during analysis: {path}"
            ) from exc
        if observed != record["sha256"]:
            raise RuntimeError(
                f"Input changed during analysis; refusing to write results: {path}"
            )


def output_paths(output_pdf: Path) -> tuple[Path, ...]:
    gp_band_pdf = output_pdf.with_name(output_pdf.stem + "_GP_errorband.pdf")
    return (
        output_pdf,
        output_pdf.with_suffix(".png"),
        output_pdf.with_suffix(".csv"),
        gp_band_pdf,
        gp_band_pdf.with_suffix(".png"),
    )


def require_new_outputs(paths: tuple[Path, ...]) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing Gmix output(s): {names}. "
            "Move the old bundle or change the input/method."
        )
