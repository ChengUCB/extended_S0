#!/usr/bin/env python3
"""NPT production (isotropic, restart-safe).

On a fresh start, reads start.xyz
"""

from __future__ import annotations
import os
import warnings
from pathlib import Path
import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import mol
from mace.calculators import MACECalculator

warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")

TEMPERATURE = float(os.environ.get("MD_TEMP", 1200.0))  # set by the launcher; 1200 K default
PRESSURE_BAR = 1.0
TIMESTEP = 2.0 * units.fs
N_STEPS = 125_000
TRAJECTORY_INTERVAL = 200
LOG_INTERVAL = 200
PRESSURE_LOG_INTERVAL = 200
TDAMP = 100.0 * TIMESTEP
PDAMP = 1000.0 * TIMESTEP

MODEL_PATH = os.environ.get("MODEL_PATH", "salt-r8c.model")  # long-range LES model; set by launcher
INITIAL_XYZ = "start.xyz"            # 6400-atom mixed rocksalt crystal (build_crystal.py)

TRAJECTORY_FILE = "md.traj"
XYZ_FILE = "npt-mace.xyz"
VOLUME_LOG_FILE = "volume.log"
PRESSURE_COMPONENTS_FILE = "pressure_components.log"
MD_LOG_FILE = "md.log"
PRESSURE_LOG_FILE = "log_1200K_1bar_mace.log"
DENSITY_FILE = "density.txt"

STEP_OFFSET = 0  # absolute NPT step of the reloaded frame (set below on resume)


def is_cubic_cell(cell, atol=1e-7):
    cell = np.asarray(cell, dtype=float)
    gram = cell @ cell.T
    mean_length_sq = float(np.trace(gram) / 3.0)
    return np.allclose(gram, np.eye(3) * mean_length_sq, rtol=1e-9, atol=atol)


def assert_cubic_cell(atoms):
    if not is_cubic_cell(atoms.cell.array):
        raise RuntimeError("Cell is not cubic")


def print_energy(atoms):
    epot = atoms.get_potential_energy() / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)
    density = (atoms.get_masses().sum() / mol) / (atoms.get_volume() * 1e-24)
    print(f"Epot={epot:.6f}  Ekin={ekin:.6f}  T={atoms.get_temperature():.0f} K  "
          f"Density={density:.6f} g/cm^3")
    with open(DENSITY_FILE, "a") as f:
        f.write(f"{density}\n")


def log_volume(atoms, dyn):
    vol = atoms.get_volume()
    step = dyn.nsteps + STEP_OFFSET
    with Path(VOLUME_LOG_FILE).open("a") as f:
        f.write(f"{step:8d} {step * (TIMESTEP / units.fs):14.3f} "
                f"{atoms.get_temperature():12.3f} {vol:18.8f} {vol**(1/3):16.10f}\n")


def log_pressure_components(atoms, dyn):
    st = atoms.get_stress(voigt=False, include_ideal_gas=True)
    sc = atoms.get_stress(voigt=False, include_ideal_gas=False)
    pt = -np.trace(st) / 3.0 / units.bar
    pc = -np.trace(sc) / 3.0 / units.bar
    pk = pt - pc
    step = dyn.nsteps + STEP_OFFSET
    with Path(PRESSURE_COMPONENTS_FILE).open("a") as f:
        f.write(f"{step:8d} {step * (TIMESTEP / units.fs):14.3f} "
                f"{atoms.get_temperature():12.3f} {atoms.get_volume():18.8f} "
                f"{pt:14.6f} {pc:14.6f} {pk:14.6f}\n")


def ensure_header(filename, header):
    if not Path(filename).exists():
        with open(filename, "a") as f:
            f.write(header)


def n_frames(path):
    if not Path(path).exists():
        return 0
    try:
        return len(Trajectory(path))
    except Exception:
        return 0


def reset_file(path):
    p = Path(path)
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
DEVICE = os.environ.get("DEVICE", "cuda")
calculator = MACECalculator(model_paths=[MODEL_PATH], device=DEVICE, default_dtype="float32")

npt_frames = n_frames(TRAJECTORY_FILE)
resuming = npt_frames > 0

if resuming:
    STEP_OFFSET = (npt_frames - 1) * TRAJECTORY_INTERVAL
    remaining = max(0, N_STEPS - STEP_OFFSET)
    atoms = read(TRAJECTORY_FILE, index=-1)  # positions + cubic cell + velocities
    atoms.calc = calculator
    assert_cubic_cell(atoms)
    print(f"RESUME NPT from {TRAJECTORY_FILE}: {STEP_OFFSET} steps done, {remaining} remaining")
else:
    STEP_OFFSET = 0
    remaining = N_STEPS
    for fname in (VOLUME_LOG_FILE, PRESSURE_COMPONENTS_FILE, DENSITY_FILE,
                  MD_LOG_FILE, PRESSURE_LOG_FILE):
        reset_file(fname)

    atoms = read(INITIAL_XYZ)            # 6400-atom mixed rocksalt crystal
    assert_cubic_cell(atoms)
    atoms.wrap()
    atoms.calc = calculator

    density = (atoms.get_masses().sum() / mol) / (atoms.get_volume() * 1e-24)
    print(f"crystal seed: {len(atoms)} atoms, L={atoms.cell.lengths()[0]:.4f} A, "
          f"density={density:.4f} g/cm^3", flush=True)

    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE)
    Stationary(atoms)

print(f"NPT: T={TEMPERATURE} K, P={PRESSURE_BAR} bar, target steps={N_STEPS}")
ensure_header(PRESSURE_COMPONENTS_FILE,
              "# step time_fs temperature_K volume_A3 pressure_total_bar pressure_config_bar pressure_kinetic_bar\n")
ensure_header(VOLUME_LOG_FILE, "# step time_fs temperature_K volume_A3\n")

if remaining <= 0:
    print("NPT already complete; nothing to do.")
else:
    dyn = IsotropicMTKNPT(atoms, timestep=TIMESTEP, temperature_K=TEMPERATURE,
                           pressure_au=PRESSURE_BAR * units.bar,
                           tdamp=TDAMP, pdamp=PDAMP, tchain=3, pchain=3, tloop=1, ploop=1)

    def guard(fn):
        def _wrapped():
            if resuming and dyn.nsteps == 0:
                return
            fn()
        return _wrapped

    traj_writer = Trajectory(TRAJECTORY_FILE, mode="a" if resuming else "w", atoms=atoms)
    md_logger = MDLogger(dyn, atoms, MD_LOG_FILE, header=not resuming, stress=False, peratom=True, mode="a")
    pressure_logger = MDLogger(dyn, atoms, PRESSURE_LOG_FILE, header=not resuming, stress=True, peratom=True, mode="a")

    dyn.attach(guard(md_logger), interval=LOG_INTERVAL)
    dyn.attach(guard(pressure_logger), interval=PRESSURE_LOG_INTERVAL)
    dyn.attach(guard(lambda: log_pressure_components(atoms, dyn)), interval=PRESSURE_LOG_INTERVAL)
    dyn.attach(guard(lambda: log_volume(atoms, dyn)), interval=TRAJECTORY_INTERVAL)
    dyn.attach(guard(lambda: traj_writer.write(atoms)), interval=TRAJECTORY_INTERVAL)
    dyn.attach(guard(lambda: write(XYZ_FILE, atoms, append=True)), interval=TRAJECTORY_INTERVAL)
    dyn.attach(guard(lambda: print_energy(atoms)), interval=TRAJECTORY_INTERVAL)
    dyn.attach(lambda: assert_cubic_cell(atoms), interval=TRAJECTORY_INTERVAL)

    try:
        dyn.run(remaining)
    finally:
        traj_writer.close()
    print("NPT done.")
