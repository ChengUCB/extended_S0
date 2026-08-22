#!/usr/bin/env python3
"""Recreate the publication figure from the packaged analysis tables."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from analyze_random_mixing_sro_li_mg import plot_summary, results_from_csv


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PACKAGE_ROOT / "data" / "4_random_SRO_200ps",
        help="Directory containing the packaged CSV analysis tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "figures",
        help="Destination for structure_analysis.pdf and .png.",
    )
    parser.add_argument(
        "--criterion",
        default="union",
        choices=("union", "M-Cl"),
        help="First-shell criterion stored in the connectivity table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = results_from_csv(args.data_dir, criterion=args.criterion)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="structure-analysis-") as tmp_name:
        temporary_dir = Path(tmp_name)
        plot_summary(
            grouped["LiCl-NaCl"],
            grouped["MgCl2-NaCl"],
            temporary_dir,
        )
        source_base = temporary_dir / "random_baseline_and_warren_cowley_comparison"
        for suffix in (".pdf", ".png"):
            shutil.copy2(
                source_base.with_suffix(suffix),
                args.output_dir / f"structure_analysis{suffix}",
            )

    print(f"Wrote {args.output_dir / 'structure_analysis.pdf'}")
    print(f"Wrote {args.output_dir / 'structure_analysis.png'}")


if __name__ == "__main__":
    main()
