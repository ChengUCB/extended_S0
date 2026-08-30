# MD inputs

All configurations and derived results in this repository use **MACELES**.

`initial_configurations.zip` contains one extended-XYZ file for each of the 38
compositions. Every file is the final frame extracted from its corresponding
`md.traj`, contains 6400 ions, and preserves the atomic species, positions,
periodic cell, and periodic-boundary flags. Calculator results and momenta are
not copied into these clean starting structures.

Archive layout:

```text
initial_configurations/
  LiCl_NaCl/li160.xyz ... li3040.xyz
  MgCl2_NaCl/cl3232.xyz ... cl4256.xyz
```

`scripts/npt_md.py` is a reusable, restart-safe MACELES workflow with defaults
matching the reported simulations: 1200 K, 1 bar, 2 fs, 25,000 NVT steps, and
1,000,000 isotropic MTK-NPT steps. Supply an extracted configuration and the
tracked MACELES model checkpoint:

```bash
python scripts/npt_md.py initial_configurations/LiCl_NaCl/li160.xyz \
  salt-MACELES.model --output-dir runs/li160
```

Temperature, pressure, timestep, reporting interval, and NPT step count must
be finite and positive; the NVT step count may be zero but not negative. These
domains are checked before the output directory is created.

`salt-MACELES.model` has SHA-256
`15dc639aae415aaf9f6a1e7b5f853836111313b4214dbdafcc1ad9553041a70b`.
The repository does not contain its original source/download metadata. The
checkpoint uses Python/PyTorch serialization and must therefore be treated as
trusted code: the hash checks identity, not authenticity or safety.

The root requirements file covers analysis, not this MD runtime. Install a
compatible ASE/PyTorch/MACE stack separately; the exact versions used to
generate the reported trajectories are not available in this repository.
Validate on a small system and record the resolved environment alongside
`run_metadata.json`.

The trajectory mirror is named `npt_mace.xyz`. If an existing `md.traj`
cannot be read, or an output directory contains workflow files but no restart
trajectory, the script stops without deleting or overwriting anything. Inspect
or move the old outputs before selecting a fresh output directory.

Before simulation starts, the script atomically writes `run_metadata.json`
with SHA-256 identities for the initial configuration and model plus the
temperature, pressure, timestep, NVT/NPT protocol and step counts, reporting
interval, device, calculator dtype, script SHA-256, and the Python, NumPy, ASE,
PyTorch, and mace-torch versions. Unavailable distribution metadata is recorded
explicitly as `not-installed` or `unknown`. A resume is allowed only when every
recorded field matches the requested run. Legacy output directories containing
`md.traj` but no metadata are rejected because their step offset and physical
conditions cannot be verified; migrate them only from a trusted run record, or
use a new `--output-dir`.

Resume starts a new dynamics segment from the last saved atomic configuration;
it does not restore Nose-Hoover-chain or barostat internal state. It is therefore
not a bitwise or exact dynamical checkpoint. Re-equilibrate after each restart
and exclude that interval before calculating equilibrium averages unless a
future workflow explicitly serializes and restores the complete integrator
state.

`scripts/compute_hmix.py` reads local trajectories for the enthalpy analysis.
Raw trajectories are intentionally not included.
