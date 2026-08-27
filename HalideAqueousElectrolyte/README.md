# S0 Electrolyte SI Data

Cleaned data and minimal analysis scripts for the aqueous-electrolyte examples.

Assumptions:

- The GP and S0 helper repositories are already installed or on `PYTHONPATH`, so `gpr_grad` and `szero` can be imported.
- Raw trajectory-derived `Sk` files are not included. The fitted S0 CSV outputs are included.

## Contents

### `MD_Inputs/`

LAMMPS inputs used to generate the underlying trajectories.

- `npt.lmp`: NPT equilibration + production run (anneal 458.15 K -> 298.15 K, then
  298.15 K / 1 atm), dumping `dump.lammpstrj` with an `element` column.
- `interAll.pp`: charges, pair styles, and force-field parameters (SPC/E water,
  halides F/Cl/Br/I, alkali Li/Na/K/Rb/Cs) included by `npt.lmp`.
- `ExampleInputs/input_files_NaCl/O4080_Na11_Cl11.dat`
- `ExampleInputs/input_files_NaClI/O3040_Na480_Cl120_I360.dat`
- `ExampleInputs/input_files_NaClF/O3040_Na480_Cl120_F360.dat`

### `simple_sk_pipeline.py`

Trajectory -> S(k) processing script. Reads LAMMPS dump files, splits each
trajectory into `--n-splits` segments with block averaging, and writes the
partial structure factors `S_ab(k)`

### `NaClWater/`

- `data/S0_NaClWater_matrix_BC_s_loss_split*.csv`
- `data/S0_NaClWater_matrix_OZ_s_loss_split*.csv`
- `data/S0_NaClWater_individual_OZ_split*.csv`
- `data/mu_ex_NaClWater_S0_validation_split0.csv`
- `data/x_ion_mu_ex_debye.csv`
- `scripts/`: BC, OZ, and helper extraction scripts.
- `notebooks/test-gp.ipynb`: NaCl(aq) chemical-potential workflow.

### `NaClIWater/`

- `data/S0_NaClIWater_matrix_BC_s_loss_split*.csv`
- `scripts/`: BC extraction script and helper files.
- `notebooks/test-gp.ipynb`: NaCl/NaI/water chemical-potential workflow.

### `NaClFWater/`

- `data/S0_NaClFWater_matrix_BC_s_loss_split*.csv`
- `scripts/`: BC extraction script and helper files.

### `mixed_halide_mu_ex/`

- `data/mu_ex_NaClIWater_BC_fixed_kcut*.csv`
- `data/mu_ex_NaClFWater_BC_fixed_kcut*.csv`
- `scripts/compute_aqueous_mu_ex_csv.py`: recomputes the mixed-halide `mu^ex` CSVs from the bundled S0 CSVs.

## Conventions

- Temperature: 298.15 K.
- Salt `mu^ex` values are neutral ion-pair quantities:
  - NaCl: `mu_Na + mu_Cl`
  - NaI: `mu_Na + mu_I`
  - NaF: `mu_Na + mu_F`
- BC tables use the fixed-Debye-length protocol unless stated otherwise in the `method` column.
- The extractor filenames contain `split0` historically; use the `--split` argument to process other splits.

## Example

```bash
conda activate asap
# if needed: export PYTHONPATH=/path/to/GPR_grad:/path/to/S0_multi:$PYTHONPATH

cd mixed_halide_mu_ex
python scripts/compute_aqueous_mu_ex_csv.py --kcuts 0.005 0.01
```

Regenerate `Sk` files from a trajectory:

```bash
python simple_sk_pipeline.py \
    --traj-dir /path/to/trajectories --traj-name dump.lammpstrj \
    --elements O H Na Cl --n-splits 3 --n-blocks 4 --k-bins 8 \
    --out-dir sk_output
```

## Requirements

Beyond the shared GP + S0 requirements in the [top-level README](../README.md)
(Python 3.10+, NumPy, PyTorch, `gpr_grad`, `szero`):

| Item | Needs |
|---|---|
| `MD_Inputs/` | LAMMPS built with **KSPACE** (`kspace_style pppm`), **MOLECULE**, **RIGID** (`fix shake`) and **EXTRA-FIX** (`fix momentum`); `pair_style lj/cut/coul/long` |
| `simple_sk_pipeline.py` | `numpy`, `numba` |
| `*/scripts/extract_*.py` | `numpy`, `scipy`, `pandas` |
| `mixed_halide_mu_ex/` | shared requirements + `pandas` |
| `*/notebooks/test-gp.ipynb` | shared requirements + `pandas`, `matplotlib`, `mpltern`; `scipy` for NaClWater |
