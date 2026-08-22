# S(k) and S(0) results

Only the production cutoff `k^2_cut = 0.02 cycles^2 A^-2` is retained.

- `Sk_k2cut0.02.csv`: every retained partial S(k) value for both systems in one
  table; all rows satisfy `0 < k^2 <= 0.02`.
- `S0_k2cut0.02.csv`: all whole-window and four-block S(0) fits for both
  systems in one table.
- `scripts/`: generation and fitting code.

For LiCl-NaCl, `solute_cation` is Li. For MgCl2-NaCl, it is Mg. In both result
tables, `block = 0` means the complete final 200 ps window and blocks 1-4 are
consecutive 50 ps segments used for uncertainty estimation. In the S(0) table,
the legacy field is named `split` and carries the same 0-4 indexing.

The single S(0) table keeps system-specific composition, count, and partial
S(0) columns so the downstream Li and Mg workflows can read the same file
without loss of precision. Fields that do not apply to a system are empty.
