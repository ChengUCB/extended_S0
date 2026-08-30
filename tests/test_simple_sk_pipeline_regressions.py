from __future__ import annotations

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINES = (
    REPO_ROOT / "LiquidAlloyFeCuNi/DataProcessing/simple_sk_pipeline.py",
    REPO_ROOT / "HalideAqueousElectrolyte/simple_sk_pipeline.py",
)


def _load_pipeline(path: Path):
    module_name = "_test_" + "_".join(path.relative_to(REPO_ROOT).parts[:-1])
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SimpleSkPipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipelines = [(path, _load_pipeline(path)) for path in PIPELINES]

    @staticmethod
    def _run_pipeline(module, frames, out_dir, *, n_splits, n_blocks, label="case"):
        def fake_sk(frac, elems, kgrid_2pi, elements):
            values = np.full(len(kgrid_2pi), frac[0, 0], dtype=float)
            return {(elements[0], elements[0]): values}

        with (
            mock.patch.object(module, "count_frames", return_value=len(frames)),
            mock.patch.object(module, "read_frames", return_value=iter(frames)),
            mock.patch.object(module, "sk_one_frame", side_effect=fake_sk),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            module.process_trajectory(
                Path("synthetic.lammpstrj"),
                ["X"],
                {"1": "X"},
                n_splits,
                n_blocks,
                2,
                out_dir,
                label=label,
            )

    def test_variable_cell_uses_mean_physical_k_squared(self):
        frames = [
            (np.full(3, 10.0), np.array(["X"]), np.array([[0.25, 0.0, 0.0]])),
            (np.full(3, 20.0), np.array(["X"]), np.array([[0.25, 0.0, 0.0]])),
        ]
        expected_k_sq = ((1.0 / 10.0) ** 2 + (1.0 / 20.0) ** 2) / 2.0

        for path, module in self.pipelines:
            with self.subTest(pipeline=path), tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                self._run_pipeline(
                    module, frames, out_dir, n_splits=1, n_blocks=2
                )
                data = np.loadtxt(out_dir / "case-allSk.dat")
                self.assertAlmostEqual(data[1, 0], expected_k_sq, places=9)

    def test_triclinic_lammps_box_is_rejected(self):
        trajectory = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp pp
0 10 1
0 10 0
0 10 0
ITEM: ATOMS id type x y z
1 1 1 1 1
"""
        for path, module in self.pipelines:
            with self.subTest(pipeline=path), self.assertRaisesRegex(
                ValueError, "Triclinic"
            ):
                module._read_lammpstrj_frame(io.StringIO(trajectory), {"1": "X"})

    def test_orthorhombic_nonzero_origin_is_subtracted(self):
        trajectory = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
-5 5
10 30
2 42
ITEM: ATOMS id type x y z
1 1 0 20 22
"""
        for path, module in self.pipelines:
            with self.subTest(pipeline=path):
                cell, elements, fractional = module._read_lammpstrj_frame(
                    io.StringIO(trajectory), {"1": "X"}
                )
                np.testing.assert_allclose(cell, [10.0, 20.0, 40.0])
                np.testing.assert_array_equal(elements, ["X"])
                np.testing.assert_allclose(fractional, [[0.5, 0.5, 0.5]])

    def test_unequal_blocks_are_weighted_by_frame_count(self):
        frame_values = (0.0, 0.0, 0.0, 10.0, 10.0)
        frames = [
            (np.full(3, 10.0), np.array(["X"]), np.array([[value, 0.0, 0.0]]))
            for value in frame_values
        ]

        for path, module in self.pipelines:
            with self.subTest(pipeline=path), tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                self._run_pipeline(
                    module, frames, out_dir, n_splits=1, n_blocks=2
                )
                data = np.loadtxt(out_dir / "case-allSk.dat")
                np.testing.assert_allclose(data[:, 1], np.mean(frame_values))

    def test_equal_block_standard_error_is_backward_compatible(self):
        pair = ("X", "X")
        block_means = [
            {pair: np.array([value], dtype=float)} for value in (1.0, 3.0, 5.0)
        ]
        expected_error = np.std((1.0, 3.0, 5.0), ddof=1) / np.sqrt(3.0)

        for path, module in self.pipelines:
            with self.subTest(pipeline=path):
                result = module._finalize_block_means(block_means, [2, 2, 2])
                self.assertAlmostEqual(result[pair][0][0], 3.0)
                self.assertAlmostEqual(result[pair][1][0], expected_error)

    def test_more_splits_than_frames_produces_nonempty_splits(self):
        frames = [
            (np.full(3, 10.0), np.array(["X"]), np.array([[value, 0.0, 0.0]]))
            for value in (1.0, 2.0)
        ]

        for path, module in self.pipelines:
            with self.subTest(pipeline=path), tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                self._run_pipeline(
                    module, frames, out_dir, n_splits=4, n_blocks=4
                )
                self.assertTrue((out_dir / "case-1-allSk.dat").is_file())
                self.assertTrue((out_dir / "case-2-allSk.dat").is_file())
                self.assertFalse((out_dir / "case-3-allSk.dat").exists())
                self.assertTrue((out_dir / "case-allSk.dat").is_file())

    def test_existing_split_output_is_preserved_and_rejected(self):
        frames = [
            (np.full(3, 10.0), np.array(["X"]), np.array([[1.0, 0.0, 0.0]]))
        ]

        for path, module in self.pipelines:
            with self.subTest(pipeline=path), tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                stale = out_dir / "case-4-allSk.dat"
                stale.write_bytes(b"old block\n")
                with self.assertRaisesRegex(FileExistsError, "Refusing"):
                    self._run_pipeline(
                        module, frames, out_dir, n_splits=1, n_blocks=1
                    )
                self.assertEqual(stale.read_bytes(), b"old block\n")
                self.assertFalse((out_dir / "case-allSk.dat").exists())

    def test_inferred_label_also_rejects_stale_split_output(self):
        frames = [
            (np.full(3, 10.0), np.array(["X"]), np.array([[1.0, 0.0, 0.0]]))
        ]

        for path, module in self.pipelines:
            with self.subTest(pipeline=path), tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                stale = out_dir / "X1-4-allSk.dat"
                stale.write_bytes(b"old inferred block\n")
                with self.assertRaisesRegex(FileExistsError, "X1"):
                    self._run_pipeline(
                        module,
                        frames,
                        out_dir,
                        n_splits=1,
                        n_blocks=1,
                        label=None,
                    )
                self.assertEqual(stale.read_bytes(), b"old inferred block\n")
                self.assertFalse((out_dir / "X1-allSk.dat").exists())


if __name__ == "__main__":
    unittest.main()
