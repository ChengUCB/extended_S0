#!/usr/bin/env python3
"""Run restart-safe NVT and isotropic NPT simulations with MACELES."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import warnings
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import mol
from mace.calculators import MACECalculator


RUN_METADATA_SCHEMA = "extended_S0.molten_npt_run.v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_xyz", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("npt_output"))
    parser.add_argument("--temperature", type=float, default=1200.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--nvt-steps", type=int, default=25_000)
    parser.add_argument("--npt-steps", type=int, default=1_000_000)
    parser.add_argument("--interval", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "temperature": args.temperature,
        "pressure_bar": args.pressure_bar,
        "timestep_fs": args.timestep_fs,
        "npt_steps": args.npt_steps,
        "interval": args.interval,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and > 0")
    if not np.isfinite(args.nvt_steps) or args.nvt_steps < 0:
        raise ValueError("--nvt-steps must be finite and >= 0")


def is_cubic(cell: np.ndarray, atol: float = 1e-7) -> bool:
    gram = np.asarray(cell, dtype=float) @ np.asarray(cell, dtype=float).T
    mean_length_sq = float(np.trace(gram) / 3.0)
    return bool(
        np.allclose(gram, np.eye(3) * mean_length_sq, rtol=1e-9, atol=atol)
    )


def require_cubic(atoms) -> None:
    if not is_cubic(atoms.cell.array):
        raise RuntimeError("MACELES NPT input cell must be cubic.")


def frame_count(path: Path) -> int:
    if not path.exists():
        if path.is_symlink():
            raise RuntimeError(
                f"Restart path {path} is a broken symlink; refusing to overwrite it."
            )
        return 0
    trajectory = None
    try:
        trajectory = Trajectory(path)
        count = len(trajectory)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read existing trajectory {path}; refusing to overwrite "
            "or append to this output directory."
        ) from exc
    finally:
        if trajectory is not None:
            trajectory.close()
    if count == 0:
        raise RuntimeError(
            f"Existing trajectory {path} contains no readable frames; refusing "
            "to overwrite this output directory."
        )
    return count


def require_clean_start(paths: tuple[Path, ...]) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            "No restart trajectory was found, but prior workflow outputs exist "
            f"({names}). Use a new --output-dir or preserve and inspect the old "
            "outputs before starting again."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"
    except Exception:
        return "unknown"


def build_run_metadata(
    args: argparse.Namespace, initial_xyz: Path, model: Path
) -> dict:
    protocol = (
        "nose_hoover_chain_nvt_then_isotropic_mtk_npt"
        if args.nvt_steps > 0
        else "isotropic_mtk_npt_only"
    )
    return {
        "schema": RUN_METADATA_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "initial_xyz_sha256": sha256_file(initial_xyz),
        "model_sha256": sha256_file(model),
        "temperature_K": float(args.temperature),
        "pressure_bar": float(args.pressure_bar),
        "timestep_fs": float(args.timestep_fs),
        "nvt_steps": int(args.nvt_steps),
        "protocol": protocol,
        "npt_steps": int(args.npt_steps),
        "interval": int(args.interval),
        "device": str(args.device),
        "calculator_default_dtype": "float32",
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": distribution_version("numpy"),
            "ase": distribution_version("ase"),
            "torch": distribution_version("torch"),
            "mace-torch": distribution_version("mace-torch"),
        },
    }


def validate_run_metadata(
    path: Path, expected: dict, *, required: bool
) -> bool:
    if not path.exists():
        if path.is_symlink():
            raise RuntimeError(
                f"Run metadata path {path} is a broken symlink; refusing to resume."
            )
        if required:
            raise RuntimeError(
                f"Missing {path.name}; refusing to resume this legacy output "
                "directory because its interval and physical conditions cannot "
                "be verified. Explicitly migrate it with verified metadata or "
                "use a new --output-dir."
            )
        return False
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read {path}; refusing to resume or overwrite this run."
        ) from exc
    if not isinstance(observed, dict):
        raise RuntimeError(f"{path} must contain a JSON object; refusing to resume.")
    mismatches = []
    for key, expected_value in expected.items():
        if key not in observed:
            mismatches.append(f"{key}: missing")
        elif observed[key] != expected_value:
            mismatches.append(
                f"{key}: stored={observed[key]!r}, requested={expected_value!r}"
            )
    if mismatches:
        raise RuntimeError(
            f"Run metadata mismatch in {path}; refusing to mix conditions: "
            + "; ".join(mismatches)
        )
    return True


def write_run_metadata(path: Path, metadata: dict) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Run metadata appeared concurrently at {path}; refusing to "
                "overwrite it."
            ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def density_g_cm3(atoms) -> float:
    return (atoms.get_masses().sum() / mol) / (atoms.get_volume() * 1e-24)


def append_line(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def ensure_header(path: Path, header: str) -> None:
    if not path.exists():
        append_line(path, header)


def main() -> None:
    args = parse_args()
    validate_args(args)
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    warnings.filterwarnings("ignore", message=".*torch.tensor_float32.*")

    initial_xyz = args.initial_xyz.expanduser().resolve()
    model = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not initial_xyz.is_file():
        raise FileNotFoundError(initial_xyz)
    if not model.is_file():
        raise FileNotFoundError(model)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = output_dir / "md.traj"
    xyz_path = output_dir / "npt_mace.xyz"
    volume_path = output_dir / "volume.log"
    pressure_components_path = output_dir / "pressure_components.log"
    md_log_path = output_dir / "md.log"
    pressure_log_path = output_dir / "pressure.log"
    density_path = output_dir / "density.txt"
    metadata_path = output_dir / "run_metadata.json"
    owned_outputs = (
        xyz_path,
        volume_path,
        pressure_components_path,
        md_log_path,
        pressure_log_path,
        density_path,
        output_dir / "nvt.log",
    )
    timestep = args.timestep_fs * units.fs
    tdamp = 100.0 * timestep
    pdamp = 1000.0 * timestep

    saved_frames = frame_count(trajectory_path)
    resuming = saved_frames > 0
    expected_metadata = build_run_metadata(args, initial_xyz, model)
    metadata_exists = True
    if resuming:
        validate_run_metadata(metadata_path, expected_metadata, required=True)
        step_offset = (saved_frames - 1) * args.interval
        atoms = read(trajectory_path, index=-1)
    else:
        metadata_exists = validate_run_metadata(
            metadata_path, expected_metadata, required=False
        )
        require_clean_start(owned_outputs)
        step_offset = 0
        atoms = read(initial_xyz)
        atoms.wrap()
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
        Stationary(atoms)
    calculator = MACECalculator(
        model_paths=[str(model)],
        device=args.device,
        default_dtype="float32",
    )
    atoms.calc = calculator
    require_cubic(atoms)
    if not resuming and not metadata_exists:
        write_run_metadata(metadata_path, expected_metadata)

    if not resuming and args.nvt_steps > 0:
        nvt = NoseHooverChainNVT(
            atoms,
            timestep=timestep,
            temperature_K=args.temperature,
            tdamp=tdamp,
            tchain=3,
        )
        nvt_logger = MDLogger(
            nvt,
            atoms,
            output_dir / "nvt.log",
            header=True,
            stress=False,
            peratom=True,
            mode="w",
        )
        nvt.attach(nvt_logger, interval=args.interval)
        nvt.run(args.nvt_steps)

    remaining = max(0, args.npt_steps - step_offset)
    if remaining == 0:
        print(f"NPT already complete: {step_offset} steps")
        return

    npt = IsotropicMTKNPT(
        atoms,
        timestep=timestep,
        temperature_K=args.temperature,
        pressure_au=args.pressure_bar * units.bar,
        tdamp=tdamp,
        pdamp=pdamp,
        tchain=3,
        pchain=3,
        tloop=1,
        ploop=1,
    )
    trajectory = Trajectory(
        trajectory_path,
        mode="a" if resuming else "w",
        atoms=atoms,
    )
    md_logger = MDLogger(
        npt,
        atoms,
        md_log_path,
        header=not resuming,
        stress=False,
        peratom=True,
        mode="a",
    )
    pressure_logger = MDLogger(
        npt,
        atoms,
        pressure_log_path,
        header=not resuming,
        stress=True,
        peratom=True,
        mode="a",
    )
    ensure_header(
        volume_path,
        "# step time_fs temperature_K volume_A3 density_g_cm3\n",
    )
    ensure_header(
        pressure_components_path,
        "# step time_fs temperature_K volume_A3 pressure_total_bar "
        "pressure_config_bar pressure_kinetic_bar\n",
    )

    def absolute_step() -> int:
        return step_offset + npt.nsteps

    def skip_restart_duplicate() -> bool:
        return resuming and npt.nsteps == 0

    def record_state() -> None:
        if skip_restart_duplicate():
            return
        step = absolute_step()
        time_fs = step * args.timestep_fs
        temperature = atoms.get_temperature()
        volume = atoms.get_volume()
        append_line(
            volume_path,
            f"{step:8d} {time_fs:14.3f} {temperature:12.3f} "
            f"{volume:18.8f} {density_g_cm3(atoms):14.8f}\n",
        )
        stress_total = atoms.get_stress(voigt=False, include_ideal_gas=True)
        stress_config = atoms.get_stress(voigt=False, include_ideal_gas=False)
        pressure_total = -np.trace(stress_total) / 3.0 / units.bar
        pressure_config = -np.trace(stress_config) / 3.0 / units.bar
        pressure_kinetic = pressure_total - pressure_config
        append_line(
            pressure_components_path,
            f"{step:8d} {time_fs:14.3f} {temperature:12.3f} "
            f"{volume:18.8f} {pressure_total:14.6f} "
            f"{pressure_config:14.6f} {pressure_kinetic:14.6f}\n",
        )
        append_line(density_path, f"{density_g_cm3(atoms):.10f}\n")
        trajectory.write(atoms)
        write(xyz_path, atoms, append=True)
        require_cubic(atoms)

    def log_md() -> None:
        if not skip_restart_duplicate():
            md_logger()
            pressure_logger()

    npt.attach(log_md, interval=args.interval)
    npt.attach(record_state, interval=args.interval)
    try:
        npt.run(remaining)
    finally:
        trajectory.close()


if __name__ == "__main__":
    main()
