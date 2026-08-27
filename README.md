Data repository for 
- "Chemical potentials from structure factors: I. Neutral multi-component mixtures"  — [arXiv:2608.08357](https://arxiv.org/abs/2608.08357)
- "Chemical potentials from structure factors: II. Charged multi-component mixtures"

This repository computes composition-dependent chemical potentials of multicomponent mixtures directly from NPT molecular dynamics (MD) simulations using the extended-S0 method.

## extended-S0 workflow: 
```
NPT MD  →  S_αβ(k)  →  S^0 = S(k→0)  →  Γ (chemical-potential derivatives)  →  ∫ → μ_i 
                       (OZ/BC fit)          (S0_multi)                       (GPR_grad)
```
1. Run NPT MD simulations over a grid of compositions.
2. Compute partial structure factors, Sαβ(k), from the trajectories.
3. Extrapolate S_αβ(k) to k→0 to obtain S_αβ^0 (the examples use the OZ matrix fit).
4. Calculate chemical-potential derivatives using the S0_multi package.
5. Integrate the chemical-potential derivatives using the GPR_grad package with one or more reference chemical potentials.
6. Optional: use GPR_grad's CUR point selection to identify additional compositions to simulate, as demonstrated for the Fe–Cu–Ni liquid alloy example.

## extended-S0 for charged systems 
For charged systems, we follow the same general workflow. However, step 3 is modified to use the Bare Coulomb (BC) fit, and charges need to be included when using the S0_multi package. 

## Table of contents

- [`LiquidAlloyFeCuNi/`](LiquidAlloyFeCuNi/) — MD inputs, simulated S0 data, and stationary GP regression models for the Fe-Cu-Ni system.
- [`Paracetamo-Water-Ethanol/`](Paracetamo-Water-Ethanol/) —  MD inputs, simulated S0 data, and non-stationary GP regression models for the paracetamol–water–ethanol system.
- [`Molten_Salts/`](Molten_Salts/) —  MD inputs, simulated S0 data, Bare Coulomb(BC) fits, GP regression models, and structural analysis
- [`HalideAqueousElectrolyte/`](HalideAqueousElectrolyte/) MD inputs, simulated S0 data, Bare Coulomb(BC) fits, and GP regression model


## 1.  Requirements

Requirements for S0_multi and GPR_grad

- Python 3.10+
- NumPy
- pandas
- PyTorch

The workflow relies on the following companion packages:
| Package | Role | Source |
| --- | --- | --- |
| **S0_multi** (`szero`) | Converts fitted `S0` values into the `Γ` matrix of chemical-potential derivatives. |[GitHub](https://github.com/ChengUCB/S0_multi)|
| **GPR_grad** (`gpr_grad`) | PyTorch Gaussian-process regression with function-value **and gradient** observations; includes CUR point selection. |[GitHub](https://github.com/ChengUCB/GPR_grad) |



