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
matching the production setup: 1200 K, 1 bar, 2 fs, 25,000 NVT steps, and
1,000,000 isotropic MTK-NPT steps. Supply an extracted configuration and the
local MACELES model checkpoint:

```bash
python scripts/npt_md.py initial_configurations/LiCl_NaCl/li160.xyz \
  /path/to/mace_les.model --output-dir runs/li160
```

`scripts/compute_hmix.py` reads local trajectories for the enthalpy analysis.
Raw trajectories and the model checkpoint are intentionally not included.
