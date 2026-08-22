# Molten-chloride structure analysis (200 ps)

Publication package for the LiCl--NaCl and MgCl2--NaCl structure analysis at
1200 K and 1 bar. It contains the analysis code, the numerical tables used in
the figure, a plotting-only entry point, and the final figure.

## Contents

| Path | Description |
|---|---|
| `scripts/analyze_mgcl2_nacl_rdfs.py` | Mg--Cl and Na--Cl RDFs and first-shell minima |
| `scripts/analyze_mgcl2_nacl_structure.py` | MgCl2--NaCl local-structure analysis |
| `scripts/analyze_compare_cl_connectivity_li_mg.py` | Li/Mg Cl-connectivity comparison and Li--Cl cutoff |
| `scripts/analyze_random_mixing_sro_li_mg.py` | Random-label connectivity and Warren--Cowley analysis |
| `scripts/plot_structure_analysis.py` | Plot-only entry point for the publication figure |
| `data/1_MgCl2_RDF/` | MgCl2--NaCl RDF tables and cutoff results |
| `data/2_MgCl2_structure/` | MgCl2--NaCl coordination tables |
| `data/3_Cl_coordination/` | Li--Cl cutoff and Li/Mg connectivity tables |
| `data/4_random_SRO_200ps/` | Complete numerical inputs for the final figure |
| `figures/structure_analysis.{pdf,png}` | Publication figure |

Raw trajectories are not included. The packaged CSV files are sufficient to
recreate the final figure.

## Replot from the packaged results

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/plot_structure_analysis.py
```

This writes `figures/structure_analysis.pdf` and
`figures/structure_analysis.png`.

## Full analysis from trajectories

Place trajectories under the following layout, or pass the corresponding CLI
paths explicitly:

```text
raw_data/
├── LiCl-NaCl/li*/md.traj
└── MgCl2-NaCl/cl*/md.traj
```

Run the stages in order:

```bash
python3 scripts/analyze_mgcl2_nacl_rdfs.py
python3 scripts/analyze_mgcl2_nacl_structure.py
python3 scripts/analyze_compare_cl_connectivity_li_mg.py
python3 scripts/analyze_random_mixing_sro_li_mg.py
python3 scripts/plot_structure_analysis.py \
  --data-dir results/4_random_SRO_200ps \
  --output-dir figures
```

The defaults in the publication copies use relative `raw_data/` and `results/`
paths and contain no machine-specific locations.

## Analysis definition

- Analysis window: final 200 ps of each trajectory.
- Maximum sampled frames per composition: 200 at 0.4 ps spacing.
- Uncertainty: sample standard deviation across five contiguous time blocks.
- NaCl fraction: `x_NaCl = N_Na / (N_Na + N_M)`, with M = Li or Mg.
- Connectivity: `n_M` is the number of Li or Mg neighbours of a Cl ion.
- Random reference: the instantaneous total cation--Cl degree and exact cation
  counts are retained; Li/Na or Mg/Na labels are randomized analytically with
  the exact hypergeometric distribution.
- Figure criterion: the union first shell, defined as the larger of the M--Cl
  and Na--Cl first minima, prevents species-biased neighbour selection.
- Warren--Cowley order: the finite-size random fraction is
  `N_Na / (N_cation - 1)`, for which random mixing gives zero exactly.

The precise run settings and cutoff values are recorded in each
`analysis_metadata.json` file.
