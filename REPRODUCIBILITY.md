# Recalculation setup and required external inputs

This repository contains processed `S(0)` tables and example analyses. It does
not contain every MD trajectory, model checkpoint, or companion repository
needed to recalculate every result. The sections below explain what can be
checked locally and which external files must be supplied.

## 1. Local regression checks

Use Python 3.10 and install the repository requirements:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

These tests check split metadata, CSV and notebook path contracts, direct use of
the companion-package APIs, MD-input manifest checksums, restart handling, and
the trajectory-analysis regressions in `tests/`. They do not rerun the missing
MD trajectories or recalculate every reported result.

## 2. Analysis notebooks and companion repositories

Install the analysis dependencies with:

```bash
python -m pip install -r requirements.txt
```

The chemical-potential notebooks import two companion repositories directly;
neither package is copied into this repository:

```bash
git clone https://github.com/ChengUCB/S0_multi.git ../S0_multi
git clone https://github.com/ChengUCB/GPR_grad.git ../GPR_grad
export SZERO_DIR="$(cd ../S0_multi && pwd)"
export GPR_GRAD_DIR="$(cd ../GPR_grad && pwd)"
export PYTHONPATH="$SZERO_DIR:$GPR_GRAD_DIR${PYTHONPATH:+:$PYTHONPATH}"
```

Each variable must name the directory *containing* the `szero/` or `gpr_grad/`
package. Record both commits and the Python environment when recalculating an
analysis:

```bash
git -C "$SZERO_DIR" rev-parse HEAD
git -C "$GPR_GRAD_DIR" rev-parse HEAD
python -m pip freeze
```

The exact companion commits used for every reported calculation are not stored
in this repository. A new calculation should therefore record its own checkout
commits instead of claiming byte-for-byte identity with an earlier environment.
`Para-water-eth_Final.ipynb` calls
`szero.prepare_gp_gradient_data` and `gpr_grad.DirectionalGradientGP` directly.

`Para-water-eth_Final.ipynb` locates checked-in data relative to the clone and
gives setup instructions if a companion package is missing.
The NaClWater and NaClIWater `test-gp.ipynb` notebooks likewise locate their
bundled `data/` directories from the repository root. Other exploratory
notebooks are historical artifacts and may still require path adaptation on a
different workstation. Unverified stored outputs were cleared from the repaired
Paracetamol and NaClI notebooks; execute them after recording the companion
commits and environment as above.

## 3. Paracetamol MD-input generator

The compositions used for the reported MD calculations are external inputs.
Put them in a JSON file satisfying
`Paracetamo-Water-Ethanol/md_input_manifest.schema.json`, then run:

```bash
python Paracetamo-Water-Ethanol/generate_md_inputs.py \
  --manifest /path/to/composition-manifest.json \
  --output-root /path/to/MD_inputs
```

`md_input_manifest.sample.json` contains one small schema example. Its molecule
counts are illustrative and are not a replacement for the composition manifest
used in the reported calculations.

Every generated case records the composition, random seed, generator and
molecule-template SHA-256 values, and `input-config.dat` SHA-256 in
`case_info.json`. The output root also records the source manifest checksum.
An existing root must have the same source manifest and exact case-directory
set. If a root or case has different metadata, lacks a checksum, or its content
was modified, the generator stops with an actionable error instead of silently
reusing stale or orphaned input.

## 4. Inputs not bundled here

Regenerating the checked-in `S(0)` CSVs requires the corresponding trajectory
and `allSk.dat` inputs referenced by each extractor; those inputs are not all in
this repository. Running MD additionally requires the engine/model files named
by each example (for example LAMMPS or external MACE checkpoints). Supply those
files from their authoritative archive, record their source and SHA-256, and do
not treat the checked-in processed CSVs as raw-trajectory substitutes. The root
requirements cover the analysis scripts and local tests; they do not define a
complete MD software environment. Where the original ASE/MACE/PyTorch versions
are unavailable, use an isolated environment, validate on a small system, and
preserve the resolved package list with the run metadata.

## 5. Licensing status

No repository-level license file is currently present. Selecting a license is
a copyright-holder decision, so this reproducibility repair does not invent
one. Do not infer reuse permission from the presence of public source code;
maintainers should add the intended license explicitly.
