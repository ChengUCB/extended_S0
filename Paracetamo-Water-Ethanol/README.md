# Data — paracetamol / water / ethanol at 303.15 K

Four CSV files. Each holds exactly one table and reads with a single
`pd.read_csv(path, comment="#")`; the `#` lines carry the source attribution.

| file | rows | what it is |
|---|---|---|
| `S0_paracetamol_water_ethanol_303K.csv` | 262 | partial structure factors extrapolated to `k → 0`, one row per simulated composition |
| `solubility_exp_jouyban2008_nagai1984_303K.csv` | 11 | experimental solubility, molar basis |
| `solubility_exp_assis2021_303K.csv` | 5 | experimental solubility, molal and molar |
| `density_ethanol_water_303K_CRC.csv` | 21 | solvent density, for the mol/L → mol/kg conversion |

Only the first file enters the model. The other three are used solely for the
experimental points in Figure B(b).

## `S0_paracetamol_water_ethanol_303K.csv`

262 compositions from NPT MD at 303.15 K and 1 atm, 31 columns.

| column | meaning |
|---|---|
| `composition_key` | `P{n_para:05d}_W{n_water:05d}_E{n_ethanol:05d}`, unique per row |
| `n_para`, `n_water`, `n_ethanol` | molecule counts in the simulation cell |
| `x_para`, `x_water`, `x_ethanol` | mole fractions, summing to 1 |
| `S0_{NN,OO,CC,NO,NC,OC}` | the six partial `S(0)`, production value |
| `S0_*_split{1,2,3}` | the same six from each of three independent trajectory blocks |

Site labels: **N** = paracetamol (its nitrogen), **O** = water (its oxygen),
**C** = ethanol (the hydroxyl-adjacent carbon). Each molecule is represented by
that one site, which is exact in the `k → 0` limit that `S(0)` is taken in.

`S(0)` is the Ornstein–Zernike extrapolation of `S(k)` with a cutoff of
`k²_cut = 0.005 × 4π²/Å²`. The three `_split` sets are the whole finite-sampling
error term: every uncertainty band in the figures comes from resampling them.

Composition coverage: `x_para` from 5.000×10⁻⁴ to 0.1997; solute-free ethanol
mole fraction from 0 to 1 over 35 distinct values. The accessible `x_para` range
narrows towards water — in pure water (12 compositions) it reaches only 0.0065,
which is the boundary of the sampled wedge masked in Figure A(a). 236 cells hold
10,000 solvent molecules; the 26 most dilute water-rich ones hold ≈30,000.

## `solubility_exp_jouyban2008_nagai1984_303K.csv`

Paracetamol Form I solubility in ethanol–water at 303.15 K, mol/L, on a shared
grid of 11 solvent compositions.

| column | meaning |
|---|---|
| `volume fraction` | ethanol volume fraction of the solvent |
| `x_eth(solute free)` | ethanol mole fraction, paracetamol excluded |
| `sol(Jouyban) (mol/L)` | Jouyban et al., *Chem. Pharm. Bull.* **56**, 602 (2008) |
| `sol(Prakongpan) (mol/L)` | Nagai & Prakongpan, *Chem. Pharm. Bull.* **32**, 340 (1984) |

The `Prakongpan` column name is kept as the code's internal key; the citation is
Nagai & Prakongpan, and the figures label it `Nagai 1984`.

## `solubility_exp_assis2021_303K.csv`

Assis et al., *J. Mol. Liq.* **323**, 114617 (2021). Five solvent compositions,
indexed by solute-free ethanol weight percent, reported in both mol/kg and mol/L.
`x_para` is the saturated paracetamol mole fraction as published.

## `density_ethanol_water_303K_CRC.csv`

Ethanol–water solvent density at 30 °C, CRC Handbook table 15-33, at 5 wt%
intervals. Indexed by solute-free ethanol weight percent — the same axis the
analysis converts to — so the mol/L → mol/kg interpolation happens on the
table's own grid. The conversion `c / ρ_solvent` neglects the volume the
dissolved paracetamol occupies, matching the convention of the Assis mol/kg
column.

## Requirements

Beyond the shared GP + S0 requirements in the [top-level README](../README.md)
(Python 3.10+, NumPy, PyTorch, `gpr_grad`, `szero`):

| Item | Needs |
|---|---|
| both notebooks | shared requirements + `pandas`, `matplotlib`, `scipy` |
| `generate_md_inputs.py` | Python standard library only |
| running the MD itself | LAMMPS |

`gpr_grad` classes used here: `DirectionalGradientGP`, `InputWarpedKernelFunction`,
`GibbsKernelFunction`.

The production notebook's local helper modules are included in this directory.
For companion checkout variables and the validated notebook entry, see the
[top-level reproducibility guide](../REPRODUCIBILITY.md).

MD input generation is manifest-driven:

```bash
python generate_md_inputs.py --manifest /path/to/composition-manifest.json \
  --output-root /path/to/MD_inputs
```

The schema is `md_input_manifest.schema.json`. The bundled
`md_input_manifest.sample.json` is deliberately labeled as a non-production
smoke case and must not be used as scientific composition data.
