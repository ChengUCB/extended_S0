# Liquid Fe–Cu–Ni: Gibbs free energy of mixing (Gmix)

Computes the Gibbs free energy of mixing of the liquid Fe–Cu–Ni ternary from
molecular dynamics. The route is:

```
MD (LAMMPS)  →  partial structure factors S_αβ(k)  →  S(k→0) matrix "S0"  →  Gmix (GP)
```

The long-wavelength limit `S0 = S(k→0)` encodes the concentration-fluctuation
curvature of the free energy. Fitting `S0` across the composition triangle and
integrating (via the Gaussian-process gradient-integration in
`../calcgp-master`) yields `Gmix`.

## Pipeline

### 1. Molecular dynamics — `MD/`
- `input.lmp` — LAMMPS template for one liquid alloy cell: 8000 atoms, EAM/alloy
  potential (`FeCuNi.eam.alloy`), NPT (`fix nph iso` + `fix temp/csvr`) at 3000 K
  and 1 bar. Composition placeholders `FRACA`/`FRACB`/`FRACC` are the Fe/Cu/Ni
  mole fractions. Equilibrates for 500 ps, then dumps `traj.lammpstrj`
  (id type x y z, every 1000 steps) over a 2 ns production run.
- `runall.sh` — stages one subdirectory per composition on the ternary grid,
  substituting the fractions into `input.lmp`. **This only sets up directories;
  it does not launch LAMMPS** — run/submit the jobs separately, e.g.
  `lmp -in input.lmp` inside each directory.

### 2. Structure factors — `DataProcessing/simple_sk_pipeline.py`
Streams a LAMMPS trajectory frame-by-frame and computes all partial structure
factors `S_αβ(k)` on a small-k grid, with block averaging and trajectory splits
for error bars. Writes one file per split (`{label}-{split}-allSk.dat`) plus a
full-trajectory file (`{label}-allSk.dat`). Columns: `k^2` followed by a
`mean error` pair per element pair (`FeFe`, `FeCu`, `FeNi`, `CuCu`, `CuNi`, `NiNi`).

```shell
python3 DataProcessing/simple_sk_pipeline.py \
    --traj MD/l-Fe-0.01-Cu-0.01-Ni-0.98/traj.lammpstrj \
    --elements Fe Cu Ni \
    --label Fe-0.01-Cu-0.01-Ni-0.98 \
    --out-dir Sk-FeCuNi
```

> **Label convention:** the extractor (step 3) parses composition from the
> filename and requires the dash form `Fe-<x>-Cu-<x>-Ni-<x>`. Always pass
> `--label` in that form. If `--label` is omitted, the auto-generated
> count-based label (e.g. `Fe80-Cu80-Ni7840`) will **not** be parsed and those
> files are silently skipped downstream.

### 3. S0 extraction — `DataProcessing/extract_FeCuNi_matrix_s0_all_splits.py`
Reads the `*-allSk.dat` files, and for each composition/split fits the matrix
Ornstein–Zernike form `S⁻¹(k) = A + k²·L` over small `k²`, then reports
`S0 = A⁻¹` with error bars propagated from the linear fit. Splits 1–3 come from
the per-split files; the full-trajectory file is treated as `split0`. Writes one
CSV row per composition (`x_Fe, x_Cu, x_Ni` and the six `S0`/`S0_error` entries
per split).

```shell
python3 DataProcessing/extract_FeCuNi_matrix_s0_all_splits.py Sk-FeCuNi \
    --kcut 0.005 \
    -o S0_results_FeCuNi_matrix_k0.005.csv
```

### 4. Gmix — `Fe-Cu-Ni-gp-clean.ipynb`
Loads the S0 CSV and fits/integrates it with a Gaussian process to produce the
Gibbs free energy of mixing across the ternary.

## Files
- `MD/` — LAMMPS input template, ternary sampler, EAM potential.
- `DataProcessing/simple_sk_pipeline.py` — trajectory → `S_αβ(k)`.
- `DataProcessing/extract_FeCuNi_matrix_s0_all_splits.py` — `S_αβ(k)` → `S0` CSV.
- `S0_results_FeCuNi_matrix_k0.005.csv` — extracted `S0` matrices (kcut = 0.005).
- `Fe-Cu-Ni-gp-clean.ipynb` — GP fit / integration to `Gmix`.

## Requirements
- LAMMPS with the `eam/alloy` pair style (step 1).
- Python with `numpy` and `numba` (steps 2–3).
- The `gpr` conda environment from `../calcgp-master/env.yml` for the notebook (step 4).
