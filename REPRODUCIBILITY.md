# Reproducibility and external-input contract

This repository contains processed `S(0)` tables and example analyses. It does
not contain every MD trajectory, model checkpoint, or companion repository
needed to regenerate all scientific results from first principles. The setup
below distinguishes the checked-in replay path from those external inputs; no
missing scientific data are synthesized.

## 1. Lightweight checked-in verification

Use Python 3.10 and the pinned smoke environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-smoke.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

This gate checks split metadata, checked-in CSV/notebook path contracts,
Paracetamol helper imports and directional-observation dimensions, MD-input
manifest/checksum behavior, and the other lightweight regression tests in
`tests/`. GitHub Actions runs the same command.

## 2. Analysis notebooks and companion repositories

Install the analysis dependencies with:

```bash
python -m pip install -r requirements.txt
```

The chemical-potential notebooks also import two companion repositories that
are deliberately not vendored here:

```bash
git clone https://github.com/ChengUCB/S0_multi.git ../S0_multi
git clone https://github.com/ChengUCB/GPR_grad.git ../GPR_grad
export SZERO_DIR="$(cd ../S0_multi && pwd)"
export GPR_GRAD_DIR="$(cd ../GPR_grad && pwd)"
export PYTHONPATH="$SZERO_DIR:$GPR_GRAD_DIR${PYTHONPATH:+:$PYTHONPATH}"
```

Each variable must name the directory *containing* the `szero/` or `gpr_grad/`
package. For a publishable rerun, record both companion commits before running:

```bash
git -C "$SZERO_DIR" rev-parse HEAD
git -C "$GPR_GRAD_DIR" rev-parse HEAD
python -m pip freeze
```

The repository does not currently declare authoritative companion commits for
the published scientific runs, so your recorded commits remain part of the
external-input manifest. The CI compatibility test—not a claim about the
production provenance—pins `S0_multi` at
`db703fabf2aac05f38e821b1113b47fe46313089`. The same integration acceptance
test can use a local real `szero` checkout:

```bash
SZERO_DIR="$SZERO_DIR" python -m unittest \
  tests.test_reproducibility_contracts.ParacetamolHelperTests.test_real_szero_directional_contract -v
```

For the bundled Paracetamol table, the accepted dimensions are
`X_g=(490, 2)`, `G_g=(490,)`, and `dirs=(490, 2)`.

`Para-water-eth_Final.ipynb` locates checked-in data relative to the clone and
fails with clone/environment instructions if a companion package is missing.
The NaClWater and NaClIWater `test-gp.ipynb` notebooks likewise locate their
bundled `data/` directories from the repository root. Other exploratory
notebooks are historical artifacts and are not claimed by this smoke replay;
some still require path adaptation before use on another workstation.
Unverified historical outputs were cleared from the repaired Paracetamol and
NaClI notebooks; execute them only after recording the companion commits and
environment as above.

## 3. Paracetamol MD-input generator

Production compositions are external scientific inputs. Put them in a JSON
file satisfying
`Paracetamo-Water-Ethanol/md_input_manifest.schema.json`, then run:

```bash
python Paracetamo-Water-Ethanol/generate_md_inputs.py \
  --manifest /path/to/composition-manifest.json \
  --output-root /path/to/MD_inputs
```

`md_input_manifest.sample.json` is explicitly a one-case, non-production smoke
sample that demonstrates the JSON shape; its molecule counts are not a
replacement for the missing production composition manifest.

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
requirements are for analysis and smoke verification; they are not an MD lock.
Where the original ASE/MACE/PyTorch versions are unavailable, use an isolated
environment, validate on a small system, and preserve the resolved package list
with the run metadata rather than claiming the smoke environment reproduces MD.

## 5. Licensing status

No repository-level license file is currently present. Selecting a license is
a copyright-holder decision, so this reproducibility repair does not invent
one. Do not infer reuse permission from the presence of public source code;
maintainers should add the intended license explicitly.
