# Molten NaCl-MgCl2 and NaCl-LiCl mixtures

MACE-LES molecular dynamics of molten chloride mixtures at
1200 K

19 compositions per system, 1200 K, 1 bar, 6400 ions.

    x_MgCl2 = N_Mg/(N_Mg+N_Na) = 0.020 - 0.985
    x_LiCl  = N_Li/(N_Li+N_Na) = 0.05  - 0.95

## Repository layout

| Path | Contents |
|---|---|
| `md/` | The NPT production driver, the MACE model |
| `Gmix/` | S(k), S(0), and the thermodynamics that follow from them |
| `structure_analysis_production/` | Cation chemical order; the structural production figure |


---

# 1. `md/` -- NPT production

Everything needed to launch one composition. The trajectories consumed by both
analysis pipelines were produced with this script.

| File | Contents |
|---|---|
| `npt_md.py` | The NPT driver: isotropic MTK barostat, restart-safe |
| `salt-r8c.model` | The MACE long-range (LES) potential, 7.8 MB |
| `start.xyz` | 6400-atom mixed rocksalt seed, cubic, L = 54.7707 A |


## Settings

    temperature      1200 K, Nose-Hoover chain, tdamp = 100 dt = 200 fs
    pressure         1 bar, isotropic MTK, pdamp = 1000 dt = 2 ps
    timestep         2 fs
    steps            125,000  ->  250 ps per run
    output interval  200 steps  ->  0.4 ps per frame

---

# 2. `Gmix/` -- S(k), S(0) and thermodynamics

Structure factors, k->0 extrapolations, and the scripts that produced them.
All values here use the **fitted screening wavevector**



# 3. `structure_analysis/` -- cation chemical order

Everything needed to reproduce the structural analysis 




 
