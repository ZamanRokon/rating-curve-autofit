"""Installed-package command-line smoke tests."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


class CommandLineTests(unittest.TestCase):
    def command(self, *arguments, timeout=90):
        return subprocess.run(
            [sys.executable, "-m", "ratingcurve_autofit", *arguments],
            capture_output=True, text=True, timeout=timeout,
        )

    def test_help_exposes_both_workflows(self):
        result = self.command("--help")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("additive", result.stdout)
        self.assertIn("validated", result.stdout)

    def test_additive_cli_writes_finite_rating_with_bootstrap_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic.csv"
            stage = np.linspace(1.0, 8.0, 24)
            pd.DataFrame({"wl": stage, "discharge": 3.0 * (stage + 1.0)**1.7}).to_csv(path, index=False)
            process = self.command("additive", str(path), "--out", str(root / "results"),
                                   "--max-segments", "1", "--bootstrap", "0")
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            outputs = list((root / "results").glob("*"))
            self.assertEqual(len(outputs), 1)
            out = outputs[0]
            for name in ["best_model.json", "best_parameters.csv", "rating_table.csv", "report.md",
                         "equation.txt", "model_comparison.csv", "input_quality.json"]:
                self.assertTrue((out / name).is_file(), name)
                self.assertGreater((out / name).stat().st_size, 0, name)
            rating = pd.read_csv(out / "rating_table.csv")
            numeric = rating.select_dtypes(include=["number"])
            self.assertGreater(len(numeric), 0)
            self.assertTrue(np.isfinite(numeric.to_numpy()).all())
            quality = json.loads((out / "input_quality.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["rows_used"], 24)


if __name__ == "__main__":
    unittest.main()
