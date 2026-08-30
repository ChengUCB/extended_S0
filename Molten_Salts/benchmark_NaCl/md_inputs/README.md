# Molecular-dynamics inputs

This compact layout contains eight unique starting configurations, one
parameterized NPT script, and one SLURM template. The MACELES and MACE-SR
benchmark checkpoints are not included; supply trusted local files explicitly.

## Starting configurations

| Files | Use |
|---|---|
| `initial_configurations/0400.xyz` through `4800.xyz` | Shared by both models |
| `initial_configurations/6400_MACELES.xyz` | MACELES 6400-atom calculation |
| `initial_configurations/6400_MACE-SR.xyz` | MACE-SR 6400-atom calculation |

The shared sizes are 400, 800, 1600, 2700, 3200, and 4800 atoms. Their two
legacy model folders contained byte-identical starting structures. The two
6400-atom structures differ and are therefore both retained.

## Protocol selection

`npt_md.py` accepts an initial XYZ, a model checkpoint, and a distinct output
directory. Defaults reproduce the source inputs: 1200 K, 1 bar, 2 fs, and
1,000,000 NPT steps.

Temperature, pressure, timestep, reporting interval, NPT step count, and
optimization `fmax` must be finite and positive; the NVT step count may be zero
but not negative. Validation occurs before the output directory is created.

The 800- and 2700-atom configurations are direct replicas and enter NPT
directly. Pass `--direct-replica` for these two sizes. All other sizes first
run geometry optimization and 25,000 NVT steps, matching their source scripts.

Example for MACELES at 400 atoms:

```bash
python npt_md.py initial_configurations/0400.xyz /path/to/salt-r8c.model \
  --output-dir runs/MACELES/0400
```

Example for MACE-SR at 800 atoms:

```bash
python npt_md.py initial_configurations/0800.xyz \
  /path/to/SuperSalt-swa.model --output-dir runs/MACE-SR/0800 \
  --direct-replica
```

Run each model/size combination in a separate output directory. Existing
`md.traj` files are detected automatically and resumed without duplicating the
last saved frame. A corrupt or empty restart trajectory, or other workflow
outputs without `md.traj`, causes a fail-closed error: no old file is removed
or overwritten. The extended-XYZ trajectory mirror is `npt_mace.xyz`.

Each new output directory receives an atomically written `run_metadata.json`.
It records SHA-256 identities for the starting configuration and checkpoint;
the temperature, pressure, timestep, NVT/NPT step counts and interval; the
optimization/NVT/direct-replica protocol; `fmax` when applicable; device; and
calculator dtype. It also records the script SHA-256 and the Python, NumPy, ASE,
PyTorch, and mace-torch versions, using `not-installed` or `unknown` when
distribution metadata is unavailable. Resuming requires an exact match for
every recorded field. A legacy trajectory without metadata is not resumed
automatically: explicitly migrate it from verified run records or select a new
output directory.

Resume starts a new dynamics segment at the final saved atomic configuration;
it does not restore the Nose-Hoover-chain or barostat internal state and is not
a bitwise or exact dynamical checkpoint. Re-equilibrate after each restart and
exclude that interval from production analysis unless a future implementation
serializes and restores the complete integrator state.
