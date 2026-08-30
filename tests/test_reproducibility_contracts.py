from __future__ import annotations

import csv
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
PARA_DIR = REPO_ROOT / "Paracetamo-Water-Ethanol"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NACL_COMMON = load_module(
    "_test_naclwater_common",
    REPO_ROOT / "HalideAqueousElectrolyte/NaClWater/scripts/sk_s0_common.py",
)
PARA_INPUTS = load_module("_test_para_inputs", PARA_DIR / "generate_md_inputs.py")
PARA_DATA = load_module("_test_para_data", PARA_DIR / "diagnose_mu_ex_gp_para.py")
PARA_DIRECTIONS = load_module("_test_para_directions", PARA_DIR / "directional_gpr.py")


class NaClWaterSplitTests(unittest.TestCase):
    TRACKED_HASHES = {
        "S0_NaClWater_individual_OZ_split1.csv": "bde1f0c4a8955b8dd87ba451912af68afa5e04eac45a2eb4fa3d9bde413b5e34",
        "S0_NaClWater_individual_OZ_split2.csv": "9c8941f07867c391e2f5c4c42b1e634f571af56ed1bb969fba674e65a0fad7e7",
        "S0_NaClWater_individual_OZ_split3.csv": "1fced04f2de35e0595f94dbd6b6bae463b8f7e840ace94280fd55e16f59d461b",
        "S0_NaClWater_matrix_BC_s_loss_split1.csv": "b4d43a50a412aac62101026fb6e43adf644dc648b46e82b194e3d8d4c734288f",
        "S0_NaClWater_matrix_BC_s_loss_split2.csv": "ac375469bf70f9c676ac44d115c574f5a243d4cd681c26216fd36d6b646d6d74",
        "S0_NaClWater_matrix_BC_s_loss_split3.csv": "ca195b3adb6da90f32fd9c076e9f90f44328aad32eef55a7f0f7a7455ea9f626",
        "S0_NaClWater_matrix_OZ_s_loss_split1.csv": "8fb06d188355c77a5c2f47e6e174aea9ccdab3acddcc4f2d92eeea4fe183da4c",
        "S0_NaClWater_matrix_OZ_s_loss_split2.csv": "1984adf1189c5630a156c6469bce772d235d6cd0e190e3af728112246c917a37",
        "S0_NaClWater_matrix_OZ_s_loss_split3.csv": "a780673a8b5381f5f206c8866cda3af230f01b3fe5d329bcab1fd1244ee35d4d",
    }

    def test_base_row_propagates_parsed_split(self):
        item = NACL_COMMON.parse_sk_file(Path("O100-Na4-Cl4-3-allSk.dat"))
        self.assertIsNotNone(item)
        row = NACL_COMMON.base_s0_row("matrix_OZ", "test", item, 0.005)
        self.assertEqual(item.split, 3)
        self.assertEqual(row["split"], 3)

    def test_tracked_split_csv_bytes_and_canonical_metadata(self):
        data_dir = REPO_ROOT / "HalideAqueousElectrolyte/NaClWater/data"
        paths = sorted(data_dir.glob("S0_NaClWater_*_split[123].csv"))
        self.assertEqual(len(paths), 9)
        for path in paths:
            expected = int(path.stem.rsplit("split", 1)[1])
            with self.subTest(path=path.name), path.open(newline="", encoding="utf-8") as handle:
                original = path.read_bytes()
                self.assertEqual(hashlib.sha256(original).hexdigest(), self.TRACKED_HASHES[path.name])
                rows = list(csv.DictReader(handle))
                self.assertTrue(rows)
                self.assertEqual({row["split"] for row in rows}, {"0"})
                self.assertTrue(
                    all(f"-{expected}-allSk.dat" in row["file"] for row in rows),
                    "source filename must carry the same split",
                )
                canonical = NACL_COMMON.load_tracked_split_csv(path)
                self.assertEqual(set(canonical["split"]), {expected})
                self.assertEqual(path.read_bytes(), original)

    def test_tampered_tracked_split_csv_is_rejected(self):
        data_dir = REPO_ROOT / "HalideAqueousElectrolyte/NaClWater/data"
        source = data_dir / "S0_NaClWater_matrix_OZ_s_loss_split1.csv"
        contract = data_dir / "S0_split_provenance.json"
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / source.name
            copied.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                NACL_COMMON.load_tracked_split_csv(copied, contract)

    def test_split_sidecar_must_agree_with_csv_filename(self):
        data_dir = REPO_ROOT / "HalideAqueousElectrolyte/NaClWater/data"
        source = data_dir / "S0_NaClWater_matrix_OZ_s_loss_split1.csv"
        contract = json.loads(
            (data_dir / "S0_split_provenance.json").read_text(encoding="utf-8")
        )
        contract["files"][source.name]["effective_split"] = 2
        with tempfile.TemporaryDirectory() as temporary:
            contract_path = Path(temporary) / "S0_split_provenance.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "filename"):
                NACL_COMMON.load_tracked_split_csv(source, contract_path)


class NotebookPathTests(unittest.TestCase):
    @staticmethod
    def notebook_source(path: Path) -> str:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    def test_naclwater_uses_bundled_data_directory(self):
        source = self.notebook_source(
            REPO_ROOT / "HalideAqueousElectrolyte/NaClWater/notebooks/test-gp.ipynb"
        )
        self.assertIn('DATA_DIR / "x_ion_mu_ex_debye.csv"', source)
        self.assertIn("DATA_DIR / 'S0_NaClWater_matrix_OZ_s_loss_split0.csv'", source)
        self.assertTrue(
            (REPO_ROOT / "HalideAqueousElectrolyte/NaClWater/data/x_ion_mu_ex_debye.csv").is_file()
        )

    def test_nacli_explicitly_uses_charged_system_bc_data(self):
        notebook_path = REPO_ROOT / "HalideAqueousElectrolyte/NaClIWater/notebooks/test-gp.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("Bare Coulomb (BC)", source)
        self.assertIn("BC_CSV_PATH", source)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])
        csv_path = REPO_ROOT / (
            "HalideAqueousElectrolyte/NaClIWater/data/"
            "S0_NaClIWater_matrix_BC_s_loss_split0.csv"
        )
        self.assertTrue(csv_path.is_file())
        with csv_path.open(newline="", encoding="utf-8") as handle:
            first = next(csv.DictReader(handle))
        self.assertEqual(first["algorithm"], "matrix_BC_s_loss")
        self.assertIn("BC", first["method"])


class ParacetamolHelperTests(unittest.TestCase):
    def test_production_notebook_has_no_unverified_embedded_outputs(self):
        notebook = json.loads(
            (PARA_DIR / "Para-water-eth_Final.ipynb").read_text(encoding="utf-8")
        )
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])

    def test_bundled_s0_contract_and_direction_count(self):
        frame = PARA_DATA.load_s0()

        fake_szero = types.ModuleType("szero")

        def fake_prepare(subset, *, independent_components, **_kwargs):
            count = len(subset) * len(independent_components)
            return None, list(range(count)), None

        fake_szero.prepare_gp_gradient_data = fake_prepare
        with mock.patch.dict(sys.modules, {"szero": fake_szero}):
            points, gradients, directions = PARA_DIRECTIONS.prepare_para_directional_data(
                frame, PARA_DATA.COLUMN_MAP, PARA_DATA.KBT
            )

        self.assertEqual(len(frame), 262)
        self.assertEqual(points.shape, (490, 2))
        self.assertEqual(gradients.shape, (490,))
        self.assertEqual(directions.shape, (490, 2))
        self.assertEqual(
            {tuple(row) for row in directions},
            {(1.0, 0.0), (0.0, 1.0), (1.0, -1.0)},
        )

    @unittest.skipUnless(os.environ.get("SZERO_DIR"), "set SZERO_DIR for companion integration test")
    def test_real_szero_directional_contract(self):
        package_root = Path(os.environ["SZERO_DIR"])
        self.assertTrue((package_root / "szero").is_dir())
        sys.path.insert(0, str(package_root))
        try:
            points, gradients, directions = PARA_DIRECTIONS.prepare_para_directional_data(
                PARA_DATA.load_s0(), PARA_DATA.COLUMN_MAP, PARA_DATA.KBT
            )
        finally:
            sys.path.remove(str(package_root))
        self.assertEqual(points.shape, (490, 2))
        self.assertEqual(gradients.shape, (490,))
        self.assertEqual(directions.shape, (490, 2))


class MdInputManifestTests(unittest.TestCase):
    def test_empty_inline_configuration_fails(self):
        with self.assertRaisesRegex(ValueError, "--manifest"):
            PARA_INPUTS.build_levels({})

    def test_sample_manifest_loads_and_is_marked_nonproduction(self):
        sample = PARA_DIR / "md_input_manifest.sample.json"
        levels, source = PARA_INPUTS.load_composition_manifest(sample)
        self.assertEqual(set(levels), {"smoke_test_not_production"})
        self.assertEqual(source["sha256"], hashlib.sha256(sample.read_bytes()).hexdigest())

    def test_sample_manifest_builds_and_reuses_identical_input(self):
        sample = PARA_DIR / "md_input_manifest.sample.json"
        levels, source = PARA_INPUTS.load_composition_manifest(sample)
        source["kind"] = "file"
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            PARA_INPUTS.build_levels(
                levels, output_root=temporary, quiet=True, composition_manifest=source
            )
            case_dir = Path(temporary) / "smoke_test_not_production"
            original = (case_dir / "input-config.dat").read_bytes()
            first_info = json.loads((case_dir / "case_info.json").read_text(encoding="utf-8"))

            PARA_INPUTS.build_levels(
                levels, output_root=temporary, quiet=True, composition_manifest=source
            )
            second_info = json.loads((case_dir / "case_info.json").read_text(encoding="utf-8"))

            self.assertEqual((case_dir / "input-config.dat").read_bytes(), original)
            self.assertEqual(first_info, second_info)
            self.assertEqual(first_info["n_atoms_expected"], 23)
            self.assertEqual(first_info["input_sha256"], hashlib.sha256(original).hexdigest())

    def test_existing_input_requires_matching_metadata_and_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "case"
            case_dir.mkdir()
            data_path = case_dir / "input-config.dat"
            data_path.write_bytes(b"deterministic generated input\n")
            expected = {
                "level": "case",
                "n_para": 1,
                "n_water": 1,
                "n_ethanol": 0,
                "box_edge_angstrom": 40.0,
                "seed": 7,
                "n_atoms_expected": 23,
                "template_sha256": "template-checksum",
            }
            metadata = {
                **expected,
                "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            }
            PARA_INPUTS.write_json(case_dir / "case_info.json", metadata)

            self.assertEqual(
                PARA_INPUTS.validate_existing_case(case_dir, expected),
                metadata["input_sha256"],
            )
            with self.assertRaisesRegex(RuntimeError, "seed"):
                PARA_INPUTS.validate_existing_case(case_dir, {**expected, "seed": 8})
            data_path.write_bytes(b"modified input\n")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                PARA_INPUTS.validate_existing_case(case_dir, expected)

    def test_output_root_rejects_a_different_manifest(self):
        sample = PARA_DIR / "md_input_manifest.sample.json"
        levels, source = PARA_INPUTS.load_composition_manifest(sample)
        source["kind"] = "file"
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            PARA_INPUTS.build_levels(
                levels, output_root=temporary, quiet=True, composition_manifest=source
            )
            manifest_path = Path(temporary) / "manifest.json"
            original_manifest = manifest_path.read_bytes()
            original_input = (
                Path(temporary) / "smoke_test_not_production" / "input-config.dat"
            ).read_bytes()

            changed_source = {**source, "sha256": "different-manifest"}
            with self.assertRaisesRegex(RuntimeError, "composition_manifest"):
                PARA_INPUTS.build_levels(
                    levels,
                    output_root=temporary,
                    quiet=True,
                    composition_manifest=changed_source,
                )

            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            self.assertEqual(
                (Path(temporary) / "smoke_test_not_production" / "input-config.dat").read_bytes(),
                original_input,
            )

    def test_output_root_rejects_unmanifested_case_directory(self):
        sample = PARA_DIR / "md_input_manifest.sample.json"
        levels, source = PARA_INPUTS.load_composition_manifest(sample)
        source["kind"] = "file"
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            PARA_INPUTS.build_levels(
                levels, output_root=temporary, quiet=True, composition_manifest=source
            )
            (Path(temporary) / "orphan_case").mkdir()
            with self.assertRaisesRegex(RuntimeError, "case directories"):
                PARA_INPUTS.build_levels(
                    levels,
                    output_root=temporary,
                    quiet=True,
                    composition_manifest=source,
                )


if __name__ == "__main__":
    unittest.main()
