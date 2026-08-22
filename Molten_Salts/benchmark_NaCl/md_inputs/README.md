# Molecular-dynamics inputs

This compact layout contains eight unique starting configurations, one
parameterized NPT script, and one SLURM template. The MACELES and MACE-SR
checkpoints remain in `../models/`.

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

The 800- and 2700-atom configurations are direct replicas and enter NPT
directly. Pass `--direct-replica` for these two sizes. All other sizes first
run geometry optimization and 25,000 NVT steps, matching their source scripts.

Example for MACELES at 400 atoms:

```bash
python npt_md.py initial_configurations/0400.xyz ../models/salt-r8c.model \
  --output-dir runs/MACELES/0400
```

Example for MACE-SR at 800 atoms:

```bash
python npt_md.py initial_configurations/0800.xyz \
  ../models/SuperSalt-swa.model --output-dir runs/MACE-SR/0800 \
  --direct-replica
```

Run each model/size combination in a separate output directory. Existing
`md.traj` files are detected automatically and resumed without duplicating the
last saved frame.
