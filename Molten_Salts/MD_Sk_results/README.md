# S(k) and S(0) results

Only the production cutoff `k^2_cut = 0.02 cycles^2 A^-2` is retained.

- `Sk_k2cut0.02.csv`: every retained partial S(k) value for both systems in one
  table; all rows satisfy `0 < k^2 <= 0.02`.
- `S0_k2cut0.02.csv`: all whole-window and four-block production S(0) fits for
  both systems in one table. The original CSV bytes are unchanged; the
  checksum-pinned `S0_k2cut0.02.provenance.json` contract identifies its rows
  as `method = BC fitted kappa_D`.
- `scripts/`: generation and fitting code.

`scripts/compute_sk.py --source xyz` reads the `npt_mace.xyz` name emitted by
both supplied NPT workflows. Whole-window S(k) values are direct frame means;
sub-blocks are used only for uncertainty estimation, so unequal sub-block
sizes do not reweight the central value or its block-size-aware standard error.
The script refuses an output tag directory containing prior `*-allSk.dat`
files, preventing stale block numbers from being mixed into a later fit.

For LiCl-NaCl, `solute_cation` is Li. For MgCl2-NaCl, it is Mg. In both result
tables, `block = 0` means the complete final 200 ps window and blocks 1-4 are
consecutive 50 ps segments used for uncertainty estimation. In the S(0) table,
the legacy field is named `split` and carries the same 0-4 indexing.

The single S(0) table keeps system-specific composition, count, and partial
S(0) columns so the downstream Li and Mg workflows can read the same file
without loss of precision. Fields that do not apply to a system are empty.
New `fit_S0.py` detailed outputs use the same `S0_k2cut<value>.csv` naming and
include explicit `system`, `solute_cation`, and `method` fields for both
`BC fixed kappa_D` and `BC fitted kappa_D`. With an explicit canonical
`--input`, the Gmix workflows default to the recommended fixed fit. Without
`--input`, they replay the tracked production table with its fitted label.
`--fit-method` overrides either default. External and newly generated tables
without `method` are rejected; only byte-identical copies of the bundled CSV
whose SHA-256 matches the tracked sidecar receive a label from that contract.
Before fitting or writing, `fit_S0.py` checks both its detailed and summary
targets, including broken symlinks, and refuses the entire output bundle if
either already exists.

The tracked label is recoverable from its fit diagnostics: all 190 rows use
four starting values (`n_starts = 4`) and have
`kappa_used != kappa_calculated`, which is the fitted-kappa branch. These rows
must not be interpreted as fixed-kappa results.
