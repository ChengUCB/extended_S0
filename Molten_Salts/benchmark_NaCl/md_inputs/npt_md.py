#!/usr/bin/env python3
"""Run a restart-safe molten-NaCl NPT benchmark with either MLIP model."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
from ase.units import mol
from mace.calculators import MACECalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_xyz", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direct-replica", action="store_true")
    parser.add_argument("--temperature", type=float, default=1200.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--nvt-steps", type=int, default=25_000)
    parser.add_argument("--npt-steps", type=int, default=1_000_000)
    parser.add_argument("--interval", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def is_cubic(cell: np.ndarray, atol: float = 1e-7) -> bool:
    cell = np.asarray(cell, dtype=float)
    gram = cell @ cell.T
    mean_length_sq = float(np.trace(gram) / 3.0)
    return bool(
        np.allclose(gram, np.eye(3) * mean_length_sq, rtol=1e-9, atol=atol)
    )


def require_cubic(atoms) -> None:
    if not is_cubic(atoms.cell.array):
        raise RuntimeError("The NPT benchmark requires a cubic cell.")


def frame_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(Trajectory(path))
    except Exception:
        return 0


def density_g_cm3(atoms) -> float:
    return (atoms.get_masses().sum() / mol) / (atoms.get_volume() * 1e-24)


def append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def ensure_header(path: Path, header: str) -> None:
    if not path.exists():
        append_line(path, header)


def reset_outputs(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
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
    timestep = args.timestep_fs * units.fs
    tdamp = 100.0 * timestep
    pdamp = 1000.0 * timestep

    calculator = MACECalculator(
        model_paths=[str(model)],
        device=args.device,
        default_dtype="float32",
    )

    saved_frames = frame_count(trajectory_path)
    resuming = saved_frames > 0
    if resuming:
        step_offset = (saved_frames - 1) * args.interval
        atoms = read(trajectory_path, index=-1)
    else:
        step_offset = 0
        reset_outputs(
            (
                trajectory_path,
                xyz_path,
                volume_path,
                pressure_components_path,
                md_log_path,
                pressure_log_path,
                density_path,
                output_dir / "opt.log",
                output_dir / "opt.xyz",
                output_dir / "nvt.log",
                output_dir / "nvt.traj",
                output_dir / "nvt.xyz",
            )
        )
        atoms = read(initial_xyz)
        atoms.wrap()
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
        Stationary(atoms)
    atoms.calc = calculator
    require_cubic(atoms)

    if not resuming and not args.direct_replica:
        optimizer = FIRE(atoms, logfile=output_dir / "opt.log")
        optimizer.run(fmax=args.fmax)
        write(output_dir / "opt.xyz", atoms)

        nvt_trajectory = Trajectory(output_dir / "nvt.traj", mode="w", atoms=atoms)
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

        def record_nvt() -> None:
            nvt_trajectory.write(atoms)
            write(output_dir / "nvt.xyz", atoms, append=True)

        nvt.attach(nvt_logger, interval=args.interval)
        nvt.attach(record_nvt, interval=args.interval)
        try:
            nvt.run(args.nvt_steps)
        finally:
            nvt_trajectory.close()

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

    def record_npt() -> None:
        if skip_restart_duplicate():
            return
        step = absolute_step()
        time_fs = step * args.timestep_fs
        temperature = atoms.get_temperature()
        volume = atoms.get_volume()
        density = density_g_cm3(atoms)
        append_line(
            volume_path,
            f"{step:8d} {time_fs:14.3f} {temperature:12.3f} "
            f"{volume:18.8f} {density:14.8f}\n",
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
        append_line(density_path, f"{density:.10f}\n")
        trajectory.write(atoms)
        write(xyz_path, atoms, append=True)
        require_cubic(atoms)

    def log_npt() -> None:
        if not skip_restart_duplicate():
            md_logger()
            pressure_logger()

    npt.attach(log_npt, interval=args.interval)
    npt.attach(record_npt, interval=args.interval)
    try:
        npt.run(remaining)
    finally:
        trajectory.close()


if __name__ == "__main__":
    main()
