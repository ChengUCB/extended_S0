# Mixing thermodynamics

This directory contains the mixing-enthalpy, S(0)-to-mixing-free-energy, and
two-panel plotting workflow for LiCl-NaCl and MgCl2-NaCl at 1200 K. All results
are labeled **MACELES**, and the production analysis uses only
`k^2_cut = 0.02 cycles^2 A^-2`.

Both system scripts read the single combined
`../MD_Sk_results/S0_k2cut0.02.csv` table and select their own `system` rows.

The notebook's final plotting CSVs and external gradient-GP helper modules were
not present in the source folder, so they are not fabricated here. The
notebook is retained without embedded outputs as a provenance record and will
run after those inputs are supplied at its documented repository-relative
paths.
