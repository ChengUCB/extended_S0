# Cation chemical order in LiCl-NaCl and MgCl2-NaCl melts

This directory contains the processed data, figures, and four-stage analysis
chain used for the 1200 K structural comparison. The underlying trajectories
and all results use **MACELES**. Raw trajectories are not included.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | RDF, coordination, connectivity, and random-mixing/SRO analysis |
| `data/1_MgCl2_RDF/` | Partial RDFs and Mg-Cl first-shell metrics |
| `data/2_MgCl2_structure/` | Mg-Cl coordination and network tables |
| `data/3_Cl_coordination/` | Li-Cl metrics and Li-vs-Mg Cl connectivity |
| `data/4_random_SRO/` | Random baselines and Warren-Cowley comparisons |
| `figures/` | Production and companion figures |

## Analysis chain

1. `analyze_mgcl2_nacl_rdfs.py`
2. `analyze_mgcl2_nacl_structure.py`
3. `analyze_compare_cl_connectivity_li_mg.py`
4. `analyze_random_mixing_sro_li_mg.py`

Stages 1-4 can be run in order after supplying the LiCl-NaCl and MgCl2-NaCl
trajectory roots. A figure-only replot can be made from the distributed tables:

```bash
cd MD_structure_analysis
PYTHONPATH=scripts python3 scripts/analyze_random_mixing_sro_li_mg.py \
  --replot-from-csv --output-dir data/4_random_SRO --figure-dir figures
```

## Method summary

- Analysis window: final 250 ps, sampled every 0.4 ps; the first 25% of that
  window is discarded and 200 frames per composition are analyzed.
- Uncertainty: sample standard deviation across five matched time blocks.
- Random reference: exact hypergeometric relabeling at fixed positions,
  simulation box, cation-Cl neighbor graph, and cation counts.
- Warren-Cowley function:
  `alpha_XNa(r) = 1 - P(Na|X,r) / [N_Na/(N_cation-1)]`.
- Neighbor criterion used in the production figure: the union of the first
  M-Cl and Na-Cl shells.

The structure-analysis window differs from the 200 ps thermodynamic window.
Rerun with a 200 ps window before directly combining those uncertainty
estimates in one analysis.
