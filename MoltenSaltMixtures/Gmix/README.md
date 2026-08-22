# S(k) and S(0) for molten NaCl-MgCl2 and NaCl-LiCl

Structure factors, k->0 extrapolations, and the scripts that produced them.
All values here use the **fitted screening wavevector**; the fixed-kappa_D branch
is deliberately not included (see "Why fitted kappa" below).

19 compositions per system, 1200 K, 1 bar, 6400 ions.

    x_MgCl2 = N_Mg/(N_Mg+N_Na) = 0.020 - 0.985
    x_LiCl  = N_Li/(N_Li+N_Na) = 0.05  - 0.95

## Layout

    scripts/
      compute_sk.py     trajectories -> S(k)
      fit_S0.py         S(k) -> S(0)
      compare_*_workflow_3x1.py   S(0) -> excess chemical potentials -> dG_mix
      compute_hmix.py   enthalpies -> dH_mix

    <system>/sk_last200ps/<tag>/
      <tag>-allSk.dat       whole 200 ps window
      <tag>-{1..4}-allSk.dat   the four 50 ps blocks

    <system>/S0_fitted_kappa/
      S0_kcut{0.01,0.015,0.02}.csv           one row per composition and block
      S0_kcut{0.01,0.015,0.02}_summary.csv   one row per composition

## Analysis window

The **final 200 ps** of each trajectory, 500 configurations at 0.4 ps spacing,
split into four 50 ps blocks. Block 0 in the CSVs is the whole-window fit and is
the value to quote; blocks 1-4 exist to give the uncertainty. The summary files
report `<col>` from block 0 and `<col>_err` as the standard error of the mean,
block SD / sqrt(4).

The window is per-trajectory, so the absolute start differs by composition:
288-611 ps for NaCl-MgCl2 (runs are 488-811 ps) and 84-186 ps for NaCl-LiCl
(284-386 ps).

## S(k) file format

Twelve data columns after k^2, value and error for each pair in the order
MgMg MgCl MgNa ClCl ClNa NaNa (LiLi LiCl LiNa ClCl ClNa NaNa for the Li system).

    S_ab(k) = < Re[rho_a(k) rho_b*(k)] > / sqrt(N_a N_b)

512 rows: the integer triples 0 <= h,k,l <= 7. **k^2 is in cycles^2/Angstrom^2**,
i.e. without the 2*pi factor, so the smallest non-zero entry is 1/L^2. The fits
multiply by (2*pi)^2 internally.

One convention difference worth flagging: this k^2 column is the frame-by-frame
average <1/L^2>, which is what sk_proc_traj.py computes. The sk_output files
circulated earlier average the cell first, giving 1/<L>^2. The two differ by
3*sigma^2, about 6e-6 relative -- far too small to affect any fit, but it means
the two pipelines are not byte-identical.

## S(0) fit

    S(k) = [ A + k^2 L + P_Z kappa^2/k^2 ]^-1 ,   P_Z = w w^T/(w.w),  w_i = sqrt(x_i) q_i

fitted over 0 < k^2 <= k^2_cut on all six lower-triangle partials with a
component-scaled soft_l1 loss, then

    S(0) = A^-1 - A^-1 w w^T A^-1 / (w^T A^-1 w)

so S0_ZZ is exactly zero by construction (expect ~1e-17) and S0_NN is the
physical output. kappa is a free parameter, started from four values
(1.0, 0.75, 0.5, 0.3) x kappa_D with the objective pinned to kappa_D throughout,
and max_nfev = 20000.

Every row in every file converged (`fit_converged == 1`) and none required
arbitration between distinct solution branches (`n_basins == 1`).

## Why fitted kappa

Holding kappa at the analytic Debye value collapses individual blocks. On
li1760, the same 50 ps of trajectory gives S_LiLi(0) = 0.503 with kappa free and
**0.156** with kappa fixed at 7.93; block scatter over the four blocks goes from
2.6% to 42.8%. The fit is not broken in the fixed case -- it converges, S(0) is
positive semi-definite, the residuals are ordinary -- the constraint simply
forces the Coulomb term to a value that block's own data contradicts by up to a
factor 4.8, and the charge-mode projection then removes most of the Li-Li
variance.

More fundamentally, S_ZZ/(k^2/kappa_D^2) is about 2.7 even at the smallest
accessible k in both systems, so the fit window never reaches the
Stillinger-Lovett asymptote and kappa_D is not the right value to impose. The
fitted kappa comes out at 0.42-0.56 x kappa_D and should **not** be read as a
physical screening length.

## The k^2 cutoff is the dominant uncertainty

dG_mix at its minimum, in kJ/mol:

    k^2_cut     NaCl-MgCl2        NaCl-LiCl
    0.02      -16.14 +/- 0.04   -7.66 +/- 0.03
    0.015     -16.58 +/- 0.17   -7.89 +/- 0.20
    0.01      -17.42 +/- 0.51   -8.36 +/- 0.81

Monotonic, roughly 3% per step, no plateau. The spread (1.28 and 0.70 kJ/mol) is
about 30x the statistical error, and for NaCl-LiCl it is the same size as G^ex
itself (-0.71 kJ/mol at x = 0.5). Quoting the statistical error alone
substantially understates the uncertainty on any of these numbers.

## Reproducing

    python3 scripts/compute_sk.py --species Li,Cl,Na \
        --root <trajectory dir> --last-ps 200 --block-ps 50 \
        --out-dir LiCl_NaCl/sk_last200ps

    python3 scripts/fit_S0.py --kcut 0.02 --species Li,Cl,Na --charges 1,-1,1 \
        --sk-dir LiCl_NaCl/sk_last200ps --label last200ps --out-dir <out>

Drop `--species/--charges` for NaCl-MgCl2 (Mg,Cl,Na / 2,-1,1 are the defaults).
`--kcut` is a cut on k SQUARED in cycles^2/Angstrom^2: 0.02 keeps k up to
0.141 cycles/A, equivalently 0.02 x 4*pi^2 A^-2 in angular units.

Requires numpy, scipy, ase, numba. compute_sk.py runs at ~135 frames/s;
fit_S0.py takes 10-20 min per cutoff for 19 compositions.
