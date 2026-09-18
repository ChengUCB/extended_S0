# Molten-salt molecular-dynamics data

This repository contains processed molecular-dynamics data for molten
LiCl-NaCl and MgCl2-NaCl mixtures at 1200 K and 1 bar. Every composition has
6400 ions. All trajectories and derived results in this release use
**MACELES**.

The distributed structure-factor and S(0) results use only the final 200 ps
and the cutoff `0 < k^2 <= 0.02 cycles^2 A^-2`
(`k <= 0.141421 cycles A^-1`). Results from other cutoffs are excluded.

## Repository layout

| Directory | Contents |
|---|---|
| `MD_inputs/` | Packaged final-frame configurations and reusable MD scripts |
| `MD_Sk_results/` | One combined S(k) CSV, one combined S(0) CSV, and analysis scripts |
| `MD_gmix_results/` | Mixing-thermodynamics notebook and scripts |
| `MD_structure_analysis/` | RDF, coordination, short-range-order data, figures, and scripts |

Raw trajectories are not included because of their size. The tracked
`MD_inputs/salt-MACELES.model` checkpoint has SHA-256
`15dc639aae415aaf9f6a1e7b5f853836111313b4214dbdafcc1ad9553041a70b`;
its original download/source record is not present, so the digest verifies
file identity but not provenance. Treat it as a trusted-code artifact and load
it only after accepting the repository source. Scripts otherwise use
command-line inputs or repository-relative paths; no machine-specific
trajectory path is committed.

## S(k) and S(0) policy

`MD_Sk_results/Sk_k2cut0.02.csv` is the only S(k) data file. It contains both
salt systems, all 19 compositions per system, the complete 200 ps window
(`block = 0`), and four 50 ps uncertainty blocks (`block = 1..4`).

`MD_Sk_results/S0_k2cut0.02.csv` is the only S(0) data file. It contains the
whole-window and block fits for both systems. The `system` column distinguishes
LiCl-NaCl from MgCl2-NaCl. Its original bytes do not contain a `method` column;
the adjacent checksum-pinned provenance sidecar identifies all rows as the
`BC fitted kappa_D` branch. Fresh `fit_S0.py` output contains both fixed- and
fitted-kappa rows with explicit method labels; the Gmix CLI documents its
input-aware defaults and rejects other unlabeled tables.

The partial structure factors use

`S_ab(k) = <Re[rho_a(k) rho_b*(k)]> / sqrt(N_a N_b)`.

The S(k) table uses `solute_cation = Li` or `Mg` and generic partial-factor
column names. Its `k^2` values use cycles-squared units without a `2*pi`
factor.

## Reproducibility

- `MD_inputs/scripts/npt_md.py` runs restart-safe MACELES NVT and NPT MD.
- `MD_inputs/scripts/compute_hmix.py` reads MD trajectories for enthalpy.
- `MD_Sk_results/scripts/compute_sk.py` generates partial S(k).
- `MD_Sk_results/scripts/fit_S0.py` fits S(0).
- `MD_gmix_results/` contains the downstream mixing-free-energy workflow.
- `MD_structure_analysis/README.md` documents the structural-analysis chain.

The root `requirements.txt` is the analysis/notebook environment; it does not
claim to reproduce the original MD runtime. Running the NPT scripts separately
requires ASE, PyTorch, and MACE with the APIs used by the scripts. The exact
versions used to generate the reported trajectories were not preserved, so
install them in an isolated environment, verify the checkpoint on a small
system, and record `pip freeze` with the run metadata. The Gmix scripts import
the public
companion packages `ChengUCB/GPR_grad` and `ChengUCB/S0_multi`.
