# Molten NaCl NPT benchmark

This benchmark contains compact molecular-dynamics inputs and processed
MACELES NPT structure-factor results for molten NaCl at 1200 K and 1 bar. The
input layout supports comparing **MACELES** with **MACE-SR** across seven
system sizes, but neither benchmark checkpoint is included.

This is a separate pure-NaCl system-size benchmark. Its final-500-ps analysis
and `k^2_cut = 0.01 cycles^2 A^-2` must not be combined with the mixture data
in `../MD_Sk_results/`, which use a 0.02 cutoff.

## Contents

| Directory | Contents |
|---|---|
| `md_inputs/` | Eight unique starting structures, one parameterized NPT script, and one SLURM template |
| `results/MACELES-NPT/` | One MACELES S(k) CSV and one MACELES S(0) CSV |

## Model identity

For MACE-SR, cite:

> C. Shen, S. Attarian, Y. Zhang, H. Zhang, M. Asta, I. Szlufarska, and
> D. Morgan, "SuperSalt: equivariant neural network force fields for
> multicomponent molten salts system," *Nature Communications* **16**, 7280
> (2025). https://doi.org/10.1038/s41467-025-62450-1


## NPT protocol

All calculations use 1200 K, 1 bar, a 2 fs timestep, 1,000,000 isotropic NPT
steps, a Nose-Hoover-chain thermostat, and a Martyna-Tobias-Klein barostat.
Configurations are written every 200 steps (0.4 ps).

The 800- and 2700-atom inputs are direct replicas and begin with NPT. The other
sizes first perform geometry optimization and 25,000 NVT steps, matching their
source workflows. `md_inputs/npt_md.py` represents both branches with the
`--direct-replica` option and accepts either model checkpoint.

See `md_inputs/README.md` for exact commands and restart behavior.
In particular, resumable runs require the per-directory `run_metadata.json`;
legacy trajectories without verifiable run conditions fail closed.

## Processed results

`results/MACELES-NPT/Sk_all_system_sizes_last500ps.csv` is the single S(k)
table. It contains binned number-density, charge-charge, and partial structure
factors for MACELES and all seven sizes.

`results/MACELES-NPT/S0_BC_all_system_sizes_k2cut0p01_last500ps.csv` is the
single S(0) table. It contains the coupled bare-Coulomb matrix fits

```text
S(k) = [A + k^2 L + Pz kappa^2/k^2]^-1
```

using the final 500 ps, `k^2_cut = 0.01 cycles^2 A^-2`, and five trajectory
blocks. No processing manifest or checksum sidecar is included; preserve the
CSV contents or add independently generated checksums before redistribution.
