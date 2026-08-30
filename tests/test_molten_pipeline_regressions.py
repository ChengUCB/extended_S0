from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@contextmanager
def md_import_stubs():
    ase = types.ModuleType("ase")
    units = types.ModuleType("ase.units")
    units.mol = 1.0
    ase.units = units
    ase_io = types.ModuleType("ase.io")
    ase_io.read = ase_io.write = lambda *args, **kwargs: None
    trajectory = types.ModuleType("ase.io.trajectory")
    trajectory.Trajectory = object
    ase_md = types.ModuleType("ase.md")
    ase_md.MDLogger = object
    nose_hoover = types.ModuleType("ase.md.nose_hoover_chain")
    nose_hoover.IsotropicMTKNPT = object
    nose_hoover.NoseHooverChainNVT = object
    velocity = types.ModuleType("ase.md.velocitydistribution")
    velocity.MaxwellBoltzmannDistribution = lambda *args, **kwargs: None
    velocity.Stationary = lambda *args, **kwargs: None
    optimize = types.ModuleType("ase.optimize")
    optimize.FIRE = object
    mace = types.ModuleType("mace")
    calculators = types.ModuleType("mace.calculators")
    calculators.MACECalculator = object
    with mock.patch.dict(
        sys.modules,
        {
            "ase": ase,
            "ase.io": ase_io,
            "ase.io.trajectory": trajectory,
            "ase.md": ase_md,
            "ase.md.nose_hoover_chain": nose_hoover,
            "ase.md.velocitydistribution": velocity,
            "ase.optimize": optimize,
            "ase.units": units,
            "mace": mace,
            "mace.calculators": calculators,
        },
    ):
        yield


class RestartSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with md_import_stubs():
            cls.modules = (
                load_module(
                    "molten_npt",
                    "Molten_Salts/MD_inputs/scripts/npt_md.py",
                ),
                load_module(
                    "benchmark_npt",
                    "Molten_Salts/benchmark_NaCl/md_inputs/npt_md.py",
                ),
            )

    def test_corrupt_restart_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "md.traj"
            original = b"not an ASE trajectory\n"
            path.write_bytes(original)
            xyz = Path(tmp) / "npt_mace.xyz"
            xyz.write_text("old xyz\n", encoding="utf-8")
            for module in self.modules:
                with self.subTest(module=module.__name__):
                    with mock.patch.object(
                        module, "Trajectory", side_effect=OSError("corrupt")
                    ):
                        with self.assertRaisesRegex(RuntimeError, "refusing"):
                            module.frame_count(path)
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(xyz.read_text(encoding="utf-8"), "old xyz\n")

    def test_prior_output_without_restart_is_not_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "npt_mace.xyz"
            log = Path(tmp) / "md.log"
            xyz.write_text("old result\n", encoding="utf-8")
            log.write_text("old log\n", encoding="utf-8")
            for module in self.modules:
                with self.subTest(module=module.__name__):
                    with self.assertRaises(FileExistsError):
                        module.require_clean_start((xyz, log))
                    self.assertEqual(xyz.read_text(encoding="utf-8"), "old result\n")
                    self.assertEqual(log.read_text(encoding="utf-8"), "old log\n")


class RestartMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with md_import_stubs():
            cls.modules = (
                load_module(
                    "molten_npt_metadata",
                    "Molten_Salts/MD_inputs/scripts/npt_md.py",
                ),
                load_module(
                    "benchmark_npt_metadata",
                    "Molten_Salts/benchmark_NaCl/md_inputs/npt_md.py",
                ),
            )

    @staticmethod
    def args():
        return types.SimpleNamespace(
            temperature=1200.0,
            pressure_bar=1.0,
            timestep_fs=2.0,
            nvt_steps=25_000,
            npt_steps=1_000_000,
            interval=200,
            device="cuda",
            direct_replica=False,
            fmax=0.05,
        )

    def test_interval_and_model_mismatch_leave_old_outputs_untouched(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                with tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    initial_xyz = directory / "initial.xyz"
                    model = directory / "model.pt"
                    metadata_path = directory / "run_metadata.json"
                    old_log = directory / "md.log"
                    initial_xyz.write_bytes(b"initial configuration")
                    model.write_bytes(b"model version one")
                    old_log.write_bytes(b"old log must survive")

                    args = self.args()
                    metadata = module.build_run_metadata(args, initial_xyz, model)
                    module.write_run_metadata(metadata_path, metadata)
                    metadata_bytes = metadata_path.read_bytes()
                    with self.assertRaisesRegex(RuntimeError, "concurrently"):
                        module.write_run_metadata(metadata_path, metadata)

                    args.interval = 400
                    interval_request = module.build_run_metadata(
                        args, initial_xyz, model
                    )
                    with self.assertRaisesRegex(RuntimeError, "interval"):
                        module.validate_run_metadata(
                            metadata_path, interval_request, required=True
                        )

                    args.interval = 200
                    model.write_bytes(b"model version two")
                    model_request = module.build_run_metadata(args, initial_xyz, model)
                    with self.assertRaisesRegex(RuntimeError, "model_sha256"):
                        module.validate_run_metadata(
                            metadata_path, model_request, required=True
                        )

                    self.assertEqual(metadata_path.read_bytes(), metadata_bytes)
                    self.assertEqual(old_log.read_bytes(), b"old log must survive")
                    self.assertFalse(list(directory.glob(".run_metadata.json.*.tmp")))

    def test_legacy_resume_without_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "run_metadata.json"
            for module in self.modules:
                with self.subTest(module=module.__name__):
                    with self.assertRaisesRegex(RuntimeError, "legacy"):
                        module.validate_run_metadata(missing, {}, required=True)

    def test_metadata_pins_script_and_runtime_versions(self):
        required_versions = {"python", "numpy", "ase", "torch", "mace-torch"}
        for module in self.modules:
            with self.subTest(module=module.__name__):
                with tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    initial_xyz = directory / "initial.xyz"
                    model = directory / "model.pt"
                    metadata_path = directory / "run_metadata.json"
                    initial_xyz.write_bytes(b"initial configuration")
                    model.write_bytes(b"model")

                    metadata = module.build_run_metadata(
                        self.args(), initial_xyz, model
                    )
                    self.assertEqual(len(metadata["script_sha256"]), 64)
                    self.assertEqual(
                        set(metadata["runtime_versions"]), required_versions
                    )
                    self.assertTrue(
                        all(
                            isinstance(version, str) and version
                            for version in metadata["runtime_versions"].values()
                        )
                    )
                    module.write_run_metadata(metadata_path, metadata)

                    changed_script = dict(metadata)
                    changed_script["script_sha256"] = "0" * 64
                    with self.assertRaisesRegex(RuntimeError, "script_sha256"):
                        module.validate_run_metadata(
                            metadata_path, changed_script, required=True
                        )

                    changed_runtime = dict(metadata)
                    changed_runtime["runtime_versions"] = dict(
                        metadata["runtime_versions"], numpy="different"
                    )
                    with self.assertRaisesRegex(RuntimeError, "runtime_versions"):
                        module.validate_run_metadata(
                            metadata_path, changed_runtime, required=True
                        )

    def test_invalid_step_controls_fail_before_run_setup(self):
        invalid_values = (
            ("interval", -200, "interval"),
            ("interval", 0, "interval"),
            ("npt_steps", -1, "npt-steps"),
            ("npt_steps", 0, "npt-steps"),
            ("nvt_steps", -1, "nvt-steps"),
            ("temperature", 0.0, "temperature"),
            ("pressure_bar", -1.0, "pressure-bar"),
            ("timestep_fs", 0.0, "timestep-fs"),
        )
        for module in self.modules:
            for field, value, message in invalid_values:
                with self.subTest(module=module.__name__, field=field):
                    args = self.args()
                    setattr(args, field, value)
                    with self.assertRaisesRegex(ValueError, message):
                        module.validate_args(args)

        args = self.args()
        args.fmax = 0.0
        with self.assertRaisesRegex(ValueError, "fmax"):
            self.modules[1].validate_args(args)


class ComputeSkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(
            "compute_sk_regression",
            "Molten_Salts/MD_Sk_results/scripts/compute_sk.py",
        )

    def test_central_value_is_direct_frame_mean(self):
        values = [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
        frames = []
        for value in values:
            sk = {pair: np.array([value]) for pair in self.module.PAIRS}
            frames.append((sk, np.array([value])))

        k_sq, means, _ = self.module.average_group(frames, n_error_blocks=5)

        expected = np.mean(values)
        self.assertAlmostEqual(float(k_sq[0]), expected)
        for pair in self.module.PAIRS:
            self.assertAlmostEqual(float(means[pair][0]), expected)

    def test_unequal_error_blocks_use_frame_count_weights(self):
        values = [0.0, 0.0, 0.0, 10.0, 10.0]
        frames = []
        for value in values:
            sk = {pair: np.array([value]) for pair in self.module.PAIRS}
            frames.append((sk, np.array([value])))

        _, _, errors = self.module.average_group(frames, n_error_blocks=2)

        block_means = np.array([0.0, 10.0])
        block_sizes = np.array([3.0, 2.0])
        frame_mean = np.mean(values)
        expected = np.sqrt(
            np.sum(block_sizes * (block_means - frame_mean) ** 2)
            / ((len(block_means) - 1) * np.sum(block_sizes))
        )
        for pair in self.module.PAIRS:
            self.assertAlmostEqual(float(errors[pair][0]), expected)

    def test_equal_error_blocks_match_previous_sem(self):
        values = [0.0, 0.0, 10.0, 10.0]
        frames = []
        for value in values:
            sk = {pair: np.array([value]) for pair in self.module.PAIRS}
            frames.append((sk, np.array([value])))

        _, _, errors = self.module.average_group(frames, n_error_blocks=2)
        expected = np.std([0.0, 10.0], ddof=1) / np.sqrt(2.0)
        for pair in self.module.PAIRS:
            self.assertAlmostEqual(float(errors[pair][0]), expected)

    def test_empty_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.module.average_group([], n_error_blocks=5)

    def test_empty_xyz_window_uses_canonical_name_and_skips(self):
        seen = []

        def no_frames(path, source, first, last):
            seen.append((path, source))
            return iter(())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            out = Path(tmp) / "out"
            with mock.patch.object(self.module, "frame_iterator", side_effect=no_frames):
                self.module.process_one(
                    "tag1", root, out, "xyz", 0.0, 1.0, 0.4, 0.8, 2, 2
                )
            self.assertEqual(seen[0][0].name, "npt_mace.xyz")
            self.assertFalse(out.exists())

    def test_stale_numbered_output_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            out = Path(tmp) / "out"
            target = out / "tag1"
            target.mkdir(parents=True)
            stale = target / "tag1-4-allSk.dat"
            stale.write_bytes(b"old block\n")

            with self.assertRaisesRegex(FileExistsError, "Refusing"):
                self.module.process_one(
                    "tag1", root, out, "xyz", 0.0, 1.0, 0.4, 0.8, 2, 2
                )
            self.assertEqual(stale.read_bytes(), b"old block\n")
            self.assertFalse((target / "tag1-allSk.dat").exists())


class FitToGmixContractTests(unittest.TestCase):
    TRACKED_S0 = ROOT / "Molten_Salts/MD_Sk_results/S0_k2cut0.02.csv"

    @classmethod
    def setUpClass(cls):
        cls.fit = load_module(
            "fit_s0_contract",
            "Molten_Salts/MD_Sk_results/scripts/fit_S0.py",
        )
        scripts = ROOT / "Molten_Salts/MD_gmix_results/scripts"
        sys.path.insert(0, str(scripts))
        try:
            cls.li = load_module(
                "gmix_li_contract",
                "Molten_Salts/MD_gmix_results/scripts/compute_gmix_licl_nacl.py",
            )
            cls.mg = load_module(
                "gmix_mg_contract",
                "Molten_Salts/MD_gmix_results/scripts/compute_gmix_mgcl2_nacl.py",
            )
        finally:
            sys.path.remove(str(scripts))

    @staticmethod
    def fake_fit(*args, **kwargs):
        return {
            "A": np.eye(3),
            "L": np.zeros((3, 3)),
            "converged": True,
            "status": 1,
            "nfev": 1,
            "cost": 0.0,
            "kappa": 1.0,
            "s0": np.eye(3),
            "rmse": 0.0,
        }

    def make_rows(self, species, charges, tag, counts):
        self.fit.set_species(species, charges)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / tag).mkdir()
            with (
                mock.patch.object(self.fit, "SK_ROOT", root),
                mock.patch.object(
                    self.fit,
                    "split_files",
                    return_value={0: root / tag / f"{tag}-allSk.dat"},
                ),
                mock.patch.object(
                    self.fit,
                    "load_sk",
                    return_value=(
                        np.array([0.001, 0.002]),
                        np.stack((np.eye(3), np.eye(3))),
                        np.asarray(counts, dtype=float),
                        20.0,
                    ),
                ),
                mock.patch.object(self.fit, "fit_one", side_effect=self.fake_fit),
            ):
                return self.fit.run(0.02)

    def assert_direct_contract(self, rows, loader, expected_system):
        fixed = [row for row in rows if row["method"] == loader.FIXED_FIT_METHOD]
        self.assertTrue(fixed)
        self.assertEqual({row["system"] for row in rows}, {expected_system})
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "S0_k2cut0.02.csv"
            self.fit.write_csv(csv_path, rows)
            loaded = loader.load_input(
                csv_path, 0.02, fit_method=loader.FIXED_FIT_METHOD
            )
        self.assertEqual(len(loaded), len(fixed))
        self.assertEqual(set(loaded["method"]), {loader.FIXED_FIT_METHOD})

    def test_fit_method_defaults_depend_only_on_explicit_input(self):
        for loader in (self.li, self.mg):
            with self.subTest(loader=loader.__name__):
                self.assertEqual(
                    loader.resolve_fit_method(None, False), loader.FITTED_FIT_METHOD
                )
                self.assertEqual(
                    loader.resolve_fit_method(None, True), loader.FIXED_FIT_METHOD
                )
                self.assertEqual(
                    loader.resolve_fit_method(loader.FITTED_FIT_METHOD, True),
                    loader.FITTED_FIT_METHOD,
                )

    def test_bundled_fitted_table_is_explicitly_labeled_and_readable(self):
        tracked = pd.read_csv(self.TRACKED_S0)
        self.assertEqual(len(tracked), 190)
        self.assertNotIn("method", tracked.columns)
        self.assertEqual(set(tracked["n_starts"]), {4})
        self.assertTrue((tracked["kappa_used"] != tracked["kappa_calculated"]).all())
        self.assertEqual(
            self.li.load_bundled_method_contract(
                self.TRACKED_S0,
                self.li.BUNDLED_S0,
                self.li.FIT_METHODS,
            ),
            self.li.FITTED_FIT_METHOD,
        )

        for loader, system in (
            (self.li, "LiCl-NaCl"),
            (self.mg, "MgCl2-NaCl"),
        ):
            with self.subTest(loader=loader.__name__):
                loaded = loader.load_input(self.TRACKED_S0, 0.02)
                self.assertEqual(set(loaded["system"]), {system})
                self.assertEqual(set(loaded["method"]), {loader.FITTED_FIT_METHOD})
                self.assertEqual(len(loaded), 95)

    def test_legacy_table_without_method_is_rejected(self):
        legacy = pd.read_csv(self.TRACKED_S0).drop(
            columns="method", errors="ignore"
        )
        legacy.loc[0, "fit_cost"] += 1e-9
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy-no-method.csv"
            legacy.to_csv(path, index=False)
            for loader in (self.li, self.mg):
                with self.subTest(loader=loader.__name__):
                    with self.assertRaisesRegex(ValueError, "missing columns.*method"):
                        loader.load_input(path, 0.02)

    def test_byte_identical_bundled_copy_uses_checksum_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "copied-S0.csv"
            copied.write_bytes(self.TRACKED_S0.read_bytes())
            for loader, system in (
                (self.li, "LiCl-NaCl"),
                (self.mg, "MgCl2-NaCl"),
            ):
                with self.subTest(loader=loader.__name__):
                    loaded = loader.load_input(copied, 0.02)
                    self.assertEqual(set(loaded["system"]), {system})
                    self.assertEqual(
                        set(loaded["method"]), {loader.FITTED_FIT_METHOD}
                    )

    def assert_mutated_input_rejected(
        self, loader, system, column, value, message
    ):
        frame = pd.read_csv(self.TRACKED_S0)
        frame["method"] = loader.FITTED_FIT_METHOD
        row = frame.index[frame["system"].eq(system)][0]
        frame[column] = frame[column].astype(object)
        frame.loc[row, column] = value
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid-S0.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, message):
                loader.load_input(path, 0.02)

    def test_string_false_fit_status_is_not_truth_cast(self):
        for loader, system in (
            (self.li, "LiCl-NaCl"),
            (self.mg, "MgCl2-NaCl"),
        ):
            with self.subTest(loader=loader.__name__):
                self.assert_mutated_input_rejected(
                    loader,
                    system,
                    "fit_converged",
                    "False",
                    "did not converge",
                )

    def test_fractional_ion_count_is_rejected(self):
        for loader, system, count_column in (
            (self.li, "LiCl-NaCl", "n_Li"),
            (self.mg, "MgCl2-NaCl", "n_Mg"),
        ):
            with self.subTest(loader=loader.__name__):
                self.assert_mutated_input_rejected(
                    loader, system, count_column, 1.5, "must contain integers"
                )

    def test_composition_must_match_ion_counts(self):
        for loader, system, composition_column in (
            (self.li, "LiCl-NaCl", "x_LiCl"),
            (self.mg, "MgCl2-NaCl", "x_MgCl2"),
        ):
            with self.subTest(loader=loader.__name__):
                self.assert_mutated_input_rejected(
                    loader,
                    system,
                    composition_column,
                    0.123456789,
                    "inconsistent with ion counts",
                )

    def test_nonfinite_partial_s0_is_rejected(self):
        for loader, system, s0_column in (
            (self.li, "LiCl-NaCl", "S0_LiLi"),
            (self.mg, "MgCl2-NaCl", "S0_MgMg"),
        ):
            with self.subTest(loader=loader.__name__):
                self.assert_mutated_input_rejected(
                    loader, system, s0_column, np.inf, "finite numeric values"
                )

    def test_licl_output_loads_without_schema_conversion(self):
        rows = self.make_rows(
            ("Li", "Cl", "Na"), (1.0, -1.0, 1.0), "li160", (1, 2, 1)
        )
        self.assert_direct_contract(rows, self.li, "LiCl-NaCl")

    def test_mgcl2_output_loads_without_schema_conversion(self):
        rows = self.make_rows(
            ("Mg", "Cl", "Na"), (2.0, -1.0, 1.0), "cl3232", (1, 3, 1)
        )
        self.assert_direct_contract(rows, self.mg, "MgCl2-NaCl")

    def test_fit_output_bundle_preserves_either_existing_target(self):
        for existing_name in ("detail", "summary"):
            with self.subTest(existing=existing_name):
                with tempfile.TemporaryDirectory() as tmp:
                    detail = Path(tmp) / "S0_k2cut0.02.csv"
                    summary = Path(tmp) / "S0_k2cut0.02_summary.csv"
                    existing = detail if existing_name == "detail" else summary
                    absent = summary if existing_name == "detail" else detail
                    existing.write_bytes(b"old result must survive")

                    with self.assertRaisesRegex(FileExistsError, "Refusing"):
                        self.fit.write_output_bundle(
                            detail,
                            summary,
                            [{"value": 1}],
                            [{"value": 2}],
                        )

                    self.assertEqual(existing.read_bytes(), b"old result must survive")
                    self.assertFalse(absent.exists())

    def test_fit_output_bundle_rejects_broken_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            detail = Path(tmp) / "S0_k2cut0.02.csv"
            summary = Path(tmp) / "S0_k2cut0.02_summary.csv"
            detail.symlink_to(Path(tmp) / "missing-target.csv")
            with self.assertRaisesRegex(FileExistsError, "Refusing"):
                self.fit.write_output_bundle(
                    detail,
                    summary,
                    [{"value": 1}],
                    [{"value": 2}],
                )
            self.assertTrue(detail.is_symlink())
            self.assertFalse(summary.exists())

    def test_write_csv_uses_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.csv"
            path.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                self.fit.write_csv(path, [{"value": 1}])
            self.assertEqual(path.read_bytes(), b"old")

    def test_experimental_csv_requires_sourced_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experimental.csv"
            path.write_text("x_MgCl2,Gmix_kJ_per_mol\n0,-1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing experimental columns"):
                self.mg.load_experimental_csv(path)


class GmixOutputProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = ROOT / "Molten_Salts/MD_gmix_results/scripts"
        sys.path.insert(0, str(scripts))
        try:
            cls.workflow = load_module(
                "gmix_workflow_io",
                "Molten_Salts/MD_gmix_results/scripts/workflow_io.py",
            )
            cls.li = load_module(
                "gmix_li_output",
                "Molten_Salts/MD_gmix_results/scripts/compute_gmix_licl_nacl.py",
            )
            cls.mg = load_module(
                "gmix_mg_output",
                "Molten_Salts/MD_gmix_results/scripts/compute_gmix_mgcl2_nacl.py",
            )
        finally:
            sys.path.remove(str(scripts))

    def test_method_and_all_input_files_are_encoded_in_output_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = directory / "first.csv"
            second = directory / "second.csv"
            first.write_bytes(b"first S0 input")
            second.write_bytes(b"second S0 input")
            _, one_identity = self.workflow.build_provenance({0.02: first})
            _, two_identity = self.workflow.build_provenance(
                {0.02: first, 0.01: second}
            )
            self.assertNotEqual(one_identity, two_identity)

            for loader in (self.li, self.mg):
                with self.subTest(loader=loader.__name__):
                    with (
                        mock.patch.object(loader, "CUTS", (0.02,)),
                        mock.patch.object(loader, "SOURCE", "repo"),
                        mock.patch.object(
                            loader, "FIT_METHOD", loader.FIXED_FIT_METHOD
                        ),
                    ):
                        fixed = loader.output_pdf(one_identity)
                    with mock.patch.object(
                        loader, "FIT_METHOD", loader.FITTED_FIT_METHOD
                    ):
                        fitted = loader.output_pdf(one_identity)
                    self.assertIn("fixed-kappa", fixed.name)
                    self.assertIn("fitted-kappa", fitted.name)
                    self.assertIn(one_identity[:12], fixed.name)
                    self.assertNotEqual(fixed, fitted)

    def test_dense_provenance_has_full_hashes_paths_and_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            s0_input = directory / "S0.csv"
            experimental = directory / "experimental.csv"
            s0_input.write_bytes(b"S0")
            experimental.write_bytes(b"experiment")
            provenance, identity = self.workflow.build_provenance(
                {0.02: s0_input}, experimental
            )
            columns = self.workflow.provenance_columns(
                provenance, identity, "BC fixed kappa_D", "repo"
            )
            self.assertEqual(columns["fit_method"], "BC fixed kappa_D")
            self.assertEqual(columns["input_identity_sha256"], identity)
            self.assertEqual(
                columns["kcut_0p02__input_sha256"],
                self.workflow.sha256_file(s0_input),
            )
            self.assertEqual(
                columns["experimental_input_sha256"],
                self.workflow.sha256_file(experimental),
            )
            self.assertEqual(
                columns["kcut_0p02__input_path"], str(s0_input.resolve())
            )

    def test_input_identity_is_stable_across_checkout_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "checkout-a" / "S0.csv"
            second = Path(tmp) / "checkout-b" / "renamed.csv"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"same scientific input")
            second.write_bytes(b"same scientific input")
            first_provenance, first_identity = self.workflow.build_provenance(
                {0.02: first}
            )
            second_provenance, second_identity = self.workflow.build_provenance(
                {0.02: second}
            )
            self.assertEqual(first_identity, second_identity)
            self.assertNotEqual(
                first_provenance["s0_inputs"][0]["path"],
                second_provenance["s0_inputs"][0]["path"],
            )

    def test_existing_bundle_target_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_pdf = Path(tmp) / "result.pdf"
            targets = self.workflow.output_paths(output_pdf)
            existing_csv = targets[2]
            existing_csv.write_bytes(b"old result")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                self.workflow.require_new_outputs(targets)
            self.assertEqual(existing_csv.read_bytes(), b"old result")
            self.assertFalse(targets[0].exists())

    def test_input_changed_after_hashing_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            s0_input = Path(tmp) / "S0.csv"
            s0_input.write_bytes(b"original")
            provenance, _ = self.workflow.build_provenance({0.02: s0_input})
            s0_input.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed during analysis"):
                self.workflow.require_provenance_unchanged(provenance)


if __name__ == "__main__":
    unittest.main()
