# Mixing thermodynamics

This directory contains the mixing-enthalpy, S(0)-to-mixing-free-energy, and
two-panel plotting workflow for LiCl-NaCl and MgCl2-NaCl at 1200 K. All results
are labeled **MACELES**, and the production analysis uses only
`k^2_cut = 0.02 cycles^2 A^-2`.

Both system scripts read the single combined
`../MD_Sk_results/S0_k2cut0.02.csv` table and select their own `system` rows.
Its original CSV bytes do not contain a `method` column. The adjacent
`S0_k2cut0.02.provenance.json` pins the CSV SHA-256 and declares
`BC fitted kappa_D`, which is therefore the default without `--input`.

Canonical detailed tables freshly written by `fit_S0.py` contain both fit
variants and can be supplied directly. For this path the default is the
recommended `BC fixed kappa_D`:

```bash
python scripts/compute_gmix_mgcl2_nacl.py --cuts 0.02 \
  --input /path/to/S0_k2cut0.02.csv
```

Either default can be overridden explicitly, for example:

```bash
python scripts/compute_gmix_mgcl2_nacl.py --cuts 0.02 \
  --input /path/to/S0_k2cut0.02.csv \
  --fit-method "BC fitted kappa_D"
```

Both workflows print every resolved S(0) input path, the selected method, and
whether that method came from `--fit-method`, the canonical `--input` default,
or the no-`--input` default.
External and newly generated input tables must contain `method`; unlabeled
tables are rejected. The sole exception is a byte-identical copy of the bundled
production CSV whose content matches its tracked checksum sidecar, so no fit
identity is inferred from an arbitrary legacy table or from its file path.
When such a copy is passed explicitly with `--input`, also select
`--fit-method "BC fitted kappa_D"`; explicit canonical inputs otherwise default
to the newly generated fixed-kappa branch.

Output basenames include `fixed-kappa` or `fitted-kappa` and the first 12
characters of a relocation-stable SHA-256 identity calculated from every
ordered `(kcut, file-content SHA-256)` input. For MgCl2-NaCl, the content hash
of an optional experimental CSV also contributes to that identity. Absolute
paths do not affect the basename. The dense output CSV repeats the selected
method, source, full input identity, and each input's resolved path and full
SHA-256 on every row.
Before writing any CSV, PDF, or PNG, the workflow checks the complete five-file
bundle and stops if any target already exists; existing results are never
silently overwritten.

The workflows use the companion
[`ChengUCB/GPR_grad`](https://github.com/ChengUCB/GPR_grad) and
[`ChengUCB/S0_multi`](https://github.com/ChengUCB/S0_multi) Python packages.
Install PyTorch and place both cloned repository roots on `PYTHONPATH`; the
scripts now import `gpr_grad` and `szero` normally and report an actionable
error when they are unavailable. Plotting constants and error-bar behavior are
provided locally in `scripts/plot_helpers.py`.

No experimental thermodynamic curve is synthesized. The MgCl2-NaCl workflow
plots none by default. To overlay sourced measurements, pass
`--experimental-csv` with the columns `x_MgCl2`, `Gmix_kJ_per_mol`,
`Hmix_kJ_per_mol`, and `minus_T_Smix_kJ_per_mol`.
