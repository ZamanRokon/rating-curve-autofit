"""Behavioral checks for reusable stage-discharge fitting and input loading.

Run with: python -m unittest discover -s tests -v
"""

import json
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from ratingcurve_autofit.core import fit_curve, predict_curve
from ratingcurve_autofit.io import load_inputs
from ratingcurve_autofit.validated import (
    Settings, calibration, load_model, make_folds, metrics, nested_validation,
    predict_pipeline, prediction_table, rf_fit, save_model, select_pipeline,
)


class CurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = np.linspace(1.0, 7.0, 65)
        cls.q = 2.5 * (cls.h + 0.7) ** 1.9
        cls.single = fit_curve(cls.h, cls.q, segments=1)

    def test_single_power_curve_recovers_known_discharge(self):
        predicted = predict_curve(self.single, self.h)
        np.testing.assert_allclose(predicted, self.q, rtol=3e-3, atol=0.03)

    def test_stage_units_and_datum_do_not_change_discharge(self):
        transformed_h = self.h * 100.0 + 1000.0
        transformed = fit_curve(transformed_h, self.q, segments=1)
        np.testing.assert_allclose(
            predict_curve(transformed, transformed_h),
            predict_curve(self.single, self.h),
            rtol=3e-3,
            atol=0.03,
        )

    def test_discharge_unit_changes_preserve_each_fitting_objective(self):
        q = self.q * (1.0 + 0.04 * np.sin(self.h * 2.0))
        for loss in ("linear", "log1p"):
            with self.subTest(loss=loss):
                original = fit_curve(self.h, q, loss=loss)
                rescaled = fit_curve(self.h, q * 1000.0, loss=loss)
                np.testing.assert_allclose(
                    predict_curve(rescaled, self.h) / 1000.0,
                    predict_curve(original, self.h), rtol=1e-4, atol=0.003,
                )

    def test_zero_discharge_is_supported_by_log_objective(self):
        h = np.linspace(0.0, 5.0, 31)
        q = 4.0 * h**1.5
        model = fit_curve(h, q, loss="log1p")
        prediction = predict_curve(model, h)
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(prediction, q, rtol=3e-3, atol=0.03)

    def test_two_regimes_join_with_continuous_slope(self):
        h = np.linspace(1.0, 11.0, 81)
        threshold = 6.0
        low = 3.0 * (h + 1.0) ** 1.6
        at_threshold = 3.0 * (threshold + 1.0) ** 1.6
        slope = 3.0 * 1.6 * (threshold + 1.0) ** 0.6
        excess = np.maximum(h - threshold, 0.0)
        high = at_threshold + slope * excess + 2.0 * excess**2
        q = np.where(h <= threshold, low, high)
        model = fit_curve(
            h, q, segments=2, shape="convex", fixed_threshold=threshold
        )
        join = model["threshold"]
        eps = np.ptp(h) * 1e-5
        q_left, q_join, q_right = predict_curve(
            model, np.array([join - eps, join, join + eps])
        )
        left_slope = (q_join - q_left) / eps
        right_slope = (q_right - q_join) / eps
        self.assertGreater(left_slope, 0.0)
        self.assertAlmostEqual(left_slope / right_slope, 1.0, delta=1e-3)
        # The assembled curve must retain convexity across the actual join.
        grid_q = predict_curve(model, np.linspace(h.min(), h.max(), 501))
        self.assertGreaterEqual(float(np.diff(grid_q).min()), -1e-9)
        self.assertGreaterEqual(float(np.diff(grid_q, n=2).min()), -1e-7)

    def test_too_few_high_regime_measurements_is_reported(self):
        h = np.linspace(1.0, 10.0, 30)
        with self.assertRaises(ValueError):
            fit_curve(h, h**2, segments=2, fixed_threshold=8.0, min_regime=12)

    def test_empty_below_control_and_invalid_prediction_stages(self):
        empty = predict_curve(self.single, np.empty((0,)))
        self.assertEqual(empty.shape, (0,))
        stage = np.array([
            self.single["zero_stage"] - 10.0,
            self.single["zero_stage"],
            np.nan,
            np.inf,
        ])
        predicted = predict_curve(self.single, stage)
        np.testing.assert_array_equal(predicted[:2], np.array([0.0, 0.0]))
        self.assertTrue(np.isnan(predicted[2:]).all())

    def test_json_roundtrip_preserves_predictions(self):
        restored = json.loads(json.dumps(self.single, allow_nan=False))
        query = np.linspace(0.0, 8.0, 100)
        np.testing.assert_allclose(
            predict_curve(restored, query), predict_curve(self.single, query),
            rtol=0.0, atol=0.0,
        )

    def test_invalid_training_discharge_is_not_silently_fitted(self):
        for invalid in (-1.0, np.nan, np.inf):
            q = self.q.copy()
            q[10] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fit_curve(self.h, q)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    @staticmethod
    def observations(n=36):
        stage = np.linspace(1.0, 10.0, n)
        return pd.DataFrame({
            "Date": pd.NaT, "WL": stage, "Q": 2.5 * (stage + 0.7) ** 1.9,
            "source_row": np.arange(n) + 2, "dWL": np.nan,
        })

    def test_validation_keeps_whole_dates_and_years_together(self):
        frame = self.observations()
        cases = {
            "date": np.repeat(pd.date_range("2020-01-01", periods=18), 2),
            "year": np.repeat(pd.to_datetime(["2018-04-01", "2019-04-01", "2020-04-01"]), 12),
        }
        for grouping, dates in cases.items():
            with self.subTest(grouping=grouping):
                frame["Date"] = dates
                folds, strategy = make_folds(frame, n_folds=3)
                self.assertIn(grouping, strategy)
                assignments = np.zeros(len(frame), dtype=int)
                for train, test in folds:
                    self.assertFalse(set(train).intersection(test))
                    train_groups = frame.iloc[train].Date
                    test_groups = frame.iloc[test].Date
                    if grouping == "year":
                        train_groups, test_groups = train_groups.dt.year, test_groups.dt.year
                    self.assertFalse(set(train_groups).intersection(test_groups))
                    assignments[test] += 1
                np.testing.assert_array_equal(assignments, np.ones(len(frame), dtype=int))

    def test_metrics_refuses_to_score_partial_predictions(self):
        with self.assertRaises(ValueError):
            metrics([2.0, 4.0, 6.0], [2.0, np.nan, 5.0])
        with self.assertRaises(ValueError):
            metrics([2.0, 4.0, 6.0], [2.0, 4.0, np.inf])
        with self.assertRaises(ValueError):
            metrics([2.0, 4.0, 6.0], [2.0])
        with self.assertRaises(ValueError):
            metrics([2.0, 4.0, 6.0], [[2.0], [4.0], [6.0]])

    def test_normalized_log_score_is_invariant_to_discharge_units(self):
        measured = np.array([0.0, 1.0, 10.0, 20.0, 100.0])
        predicted = np.array([0.5, 1.2, 9.0, 25.0, 80.0])
        original = metrics(measured, predicted)
        rescaled = metrics(measured * 1000.0, predicted * 1000.0)
        for key in ("RMSLE", "NSE", "KGE"):
            self.assertAlmostEqual(original[key], rescaled[key], places=12)
        self.assertAlmostEqual(original["RMSE"], rescaled["RMSE"] / 1000.0, places=12)

    def test_candidate_with_failed_folds_is_not_scored_as_eligible(self):
        frame = self.observations()
        settings = Settings(max_segments=2, threshold=6.0, min_regime=10)
        pipeline, comparison, _ = select_pipeline(frame, settings, folds_count=3)
        failed_candidates = [row for row in comparison if row["Segments"] == 2]
        self.assertEqual(len(failed_candidates), 2)
        for row in failed_candidates:
            self.assertEqual(row["Status"], "ineligible")
            self.assertNotIn("RMSE", row)
            self.assertGreater(row["N"], 0)  # Some folds succeeded, but not all.
            self.assertLess(row["N"], len(frame))
            self.assertTrue(row["Failure"])
        self.assertEqual(pipeline["base"]["segments"], 1)

    def test_nested_validation_predicts_every_row_once(self):
        frame = self.observations()
        settings = Settings(max_segments=1, folds=3, inner_folds=3)
        with redirect_stdout(io.StringIO()):
            oof, base_oof, fold_ids, records, _ = nested_validation(frame, settings)
        self.assertEqual(len(oof), len(frame))
        self.assertTrue(np.isfinite(oof).all())
        self.assertTrue(np.isfinite(base_oof).all())
        self.assertEqual(set(fold_ids), {1, 2, 3})
        self.assertEqual(sum(row["Test_N"] for row in records), len(frame))
        self.assertEqual(metrics(frame.Q, oof)["N"], len(frame))

    def test_prediction_bands_are_unavailable_outside_observed_stages(self):
        observed = self.observations()
        base = fit_curve(observed.WL, observed.Q)
        pipeline = {"base": base, "alpha": 0.0, "rf": None, "candidate": "1regime_linear"}
        cal = calibration(observed, predict_curve(base, observed.WL) * 1.05, 0.9)
        query = pd.DataFrame({"Date": pd.NaT, "WL": [0.5, 5.0, 10.5], "dWL": np.nan})
        result = prediction_table(query, pipeline, cal, observed)
        self.assertTrue(np.isfinite(result.Q_Estimate).all())
        self.assertTrue(result.loc[[0, 2], ["Q_Lower", "Q_Upper"]].isna().all().all())
        self.assertTrue(np.isfinite(result.loc[1, ["Q_Lower", "Q_Upper"]].to_numpy(float)).all())
        self.assertEqual(result.Stage_extrapolation.tolist(), [
            "below_measured_range", "within_measured_range", "above_measured_range"
        ])
        self.assertTrue(result.loc[[0, 2], "Interval_status"].eq("unavailable_extrapolation").all())

    def test_saved_complete_stage_pipeline_predicts_identically(self):
        observed = self.observations()
        pipeline = {
            "base": fit_curve(observed.WL, observed.Q), "alpha": 0.0,
            "rf": None, "candidate": "1regime_linear",
        }
        before = predict_pipeline(pipeline, observed)
        cal = calibration(observed, before, 0.9)
        path = self.root / "model.json"
        save_model(path, pipeline, cal, {"station": "synthetic"})
        restored = load_model(path)
        np.testing.assert_allclose(predict_pipeline(restored, observed), before, atol=0.0, rtol=0.0)
        self.assertEqual(restored["metadata"]["station"], "synthetic")
        self.assertIsNone(restored["rf"])

    @unittest.skipUnless(importlib.util.find_spec("sklearn") is not None, "optional scikit-learn unavailable")
    def test_saved_residual_rf_pipeline_predicts_identically(self):
        observed = self.observations(n=60)
        observed["dWL"] = np.sin(np.arange(len(observed)) * 0.4)
        observed["Q"] += 8.0 * observed["dWL"]
        base = fit_curve(observed.WL, observed.Q)
        pipeline = {
            "base": base, "alpha": 0.5, "rf": rf_fit(observed, base, seed=42),
            "candidate": "1regime_linear_residualRF_0.5",
        }
        before = predict_pipeline(pipeline, observed)
        self.assertGreater(float(np.max(np.abs(before - predict_curve(base, observed.WL)))), 0.01)
        path = self.root / "model.json"
        save_model(path, pipeline, calibration(observed, before, 0.9), {})
        restored = load_model(path)
        np.testing.assert_allclose(predict_pipeline(restored, observed), before, atol=1e-10, rtol=1e-12)
        self.assertIsNotNone(restored["rf"])
        query = pd.DataFrame({"Date": pd.NaT, "WL": [6.0, 6.0, 6.0],
                              "dWL": [np.nan, 100.0, 0.0]})
        cal = {"relative_radius": 0.1, "q_reference": 1.0,
               "base_curve": {"relative_radius": 0.4, "q_reference": 2.0}}
        table = prediction_table(query, restored, cal, observed)
        # Unsupported daily changes fall back to both the stage curve and its band.
        np.testing.assert_allclose(table.Q_Estimate.iloc[:2], table.Q_RatingCurve.iloc[:2])
        np.testing.assert_allclose(
            table.Q_Upper.iloc[:2] - table.Q_Estimate.iloc[:2],
            0.4 * (table.Q_Estimate.iloc[:2] + 2.0),
        )
        self.assertAlmostEqual(
            table.Q_Upper.iloc[2] - table.Q_Estimate.iloc[2],
            0.1 * (table.Q_Estimate.iloc[2] + 1.0),
        )

    def cli(self, observations, *extra):
        path = self.root / "measurements.csv"
        observations.to_csv(path, index=False)
        command = [sys.executable, "-m", "ratingcurve_autofit", "validated",
                   str(path), "--max-segments", "1", "--folds", "3", "--inner-folds", "3",
                   "--out", str(self.root / "results"), *extra]
        return subprocess.run(command, capture_output=True, text=True, timeout=60)

    def test_cli_twenty_measurements_without_dates_produces_reviewable_outputs(self):
        observed = self.observations(n=20)[["WL", "Q"]]
        process = self.cli(observed)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        outputs = list((self.root / "results").glob("*"))
        self.assertEqual(len(outputs), 1)
        result = outputs[0]
        for name in ["model.json", "rating_curve.png", "validation_diagnostics.png",
                     "report.md", "equation.txt", "rating_table.csv", "model_comparison.csv",
                     "measurement_predictions.csv", "validation_scores.csv", "validation_folds.csv"]:
            self.assertTrue((result / name).is_file(), name)
            self.assertGreater((result / name).stat().st_size, 0, name)
        measured = pd.read_csv(result / "measurement_predictions.csv")
        self.assertEqual(len(measured), 20)
        self.assertTrue(np.isfinite(measured.Q_NestedCV).all())
        restored = load_model(result / "model.json")
        np.testing.assert_allclose(predict_pipeline(restored, measured), measured.Q_Estimate, rtol=1e-12)

    def test_cli_reports_insufficient_data_before_creating_outputs(self):
        process = self.cli(self.observations(n=19)[["WL", "Q"]])
        self.assertEqual(process.returncode, 2)
        self.assertIn("At least 20", process.stderr)
        self.assertFalse((self.root / "results").exists())

    @unittest.skipUnless(importlib.util.find_spec("sklearn") is not None, "optional scikit-learn unavailable")
    def test_cli_compares_residual_rf_and_produces_complete_daily_predictions(self):
        dates = pd.date_range("2020-01-01", periods=61)
        stage = 6.0 + 2.0 * np.sin(np.arange(61) * 0.35)
        daily = pd.DataFrame({"Date": dates, "WL": stage})
        daily_path = self.root / "daily.csv"
        daily.to_csv(daily_path, index=False)
        observed = pd.DataFrame({
            "Date": dates[1:], "WL": stage[1:],
            "Q": 3.0 * stage[1:]**2 + 8.0 * np.diff(stage),
        })
        process = self.cli(observed, "--daily", str(daily_path), "--rf")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = next((self.root / "results").glob("*"))
        comparison = pd.read_csv(result / "model_comparison.csv")
        residual_candidates = comparison[comparison.Alpha > 0]
        self.assertGreater(len(residual_candidates), 0)
        self.assertTrue(residual_candidates.Status.eq("eligible").all())
        self.assertTrue(residual_candidates.N.eq(len(observed)).all())
        self.assertTrue(np.isfinite(residual_candidates.RMSE).all())
        daily_result = pd.read_csv(result / "daily_discharge_calculated.csv")
        self.assertEqual(len(daily_result), 61)
        self.assertTrue(np.isfinite(daily_result.Q_Estimate).all())
        measured_result = pd.read_csv(result / "measurement_predictions.csv")
        self.assertTrue(np.isfinite(measured_result.Q_NestedCV).all())
        restored = load_model(result / "model.json")
        np.testing.assert_allclose(predict_pipeline(restored, daily_result), daily_result.Q_Estimate,
                                   rtol=1e-12, atol=1e-10)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def csv(self, name, data):
        path = self.root / name
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    def test_input_aliases_and_explicit_column_overrides(self):
        aliases = self.csv("aliases.csv", {
            "Date": ["2020-01-01", "2020-01-02"],
            "Water Level": [2.0, 3.0],
            "Discharge": [10.0, 20.0],
        })
        result = load_inputs(aliases)
        np.testing.assert_allclose(result["observations"]["WL"], [2.0, 3.0])
        np.testing.assert_allclose(result["observations"]["Q"], [10.0, 20.0])
        custom = self.csv("custom.csv", {
            "when": ["2020-01-01", "2020-01-02"],
            "sensor_A": [2.0, 3.0],
            "flow_A": [10.0, 20.0],
        })
        result = load_inputs(
            custom, date_col="when", stage_col="sensor_A", discharge_col="flow_A"
        )
        np.testing.assert_allclose(result["observations"]["WL"], [2.0, 3.0])
        np.testing.assert_allclose(result["observations"]["Q"], [10.0, 20.0])

    def test_ambiguous_stage_columns_require_explicit_selection(self):
        q_path = self.csv("ambiguous.csv", {
            "date": ["2020-01-01", "2020-01-02"],
            "WL": [2.0, 3.0],
            "stage": [12.0, 13.0],
            "Q": [10.0, 20.0],
        })
        with self.assertRaises(ValueError):
            load_inputs(q_path)
        result = load_inputs(q_path, stage_col="stage")
        np.testing.assert_allclose(result["observations"]["WL"], [12.0, 13.0])

    def test_zero_discharge_kept_and_bad_rows_are_auditable(self):
        q_path = self.csv("invalid.csv", {
            "Date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "WL": [2.0, 3.0, 4.0, np.nan],
            "Q": [0.0, -1.0, np.nan, 20.0],
        })
        result = load_inputs(q_path)
        self.assertEqual(len(result["observations"]), 1)
        self.assertEqual(result["observations"]["Q"].iloc[0], 0.0)
        rejected = result["rejected_observations"]
        self.assertEqual(len(rejected), 3)
        self.assertIn("source_row", rejected.columns)
        self.assertIn("rejection_reason", rejected.columns)
        self.assertTrue(rejected["rejection_reason"].str.len().gt(0).all())

    def test_sparse_observations_get_daily_changes_with_gap_protection(self):
        q_path = self.csv("q.csv", {
            "date": ["2020-01-03", "2020-01-07"],
            "wl": [14.0, 13.0],
            "discharge": [100.0, 80.0],
        })
        daily_path = self.csv("daily.csv", {
            "Date": ["2020-01-07", "2020-01-01", "2020-01-03", "2020-01-05", "2020-01-06", "2020-01-02"],
            "WL": [13.0, 10.0, 14.0, 12.0, 11.0, 11.0],
        })
        result = load_inputs(q_path, daily_path)
        np.testing.assert_allclose(result["observations"]["dWL"], [3.0, 2.0])
        daily = result["daily"].set_index("Date")
        self.assertTrue(np.isnan(daily.loc[pd.Timestamp("2020-01-01"), "dWL"]))
        self.assertTrue(np.isnan(daily.loc[pd.Timestamp("2020-01-05"), "dWL"]))
        self.assertEqual(daily.loc[pd.Timestamp("2020-01-06"), "dWL"], -1.0)

    def test_observations_without_daily_input_do_not_invent_daily_change(self):
        q_path = self.csv("q.csv", {
            "date": ["2020-01-01", "2020-01-15"],
            "wl": [2.0, 7.0], "Q": [10.0, 20.0],
        })
        result = load_inputs(q_path)
        self.assertTrue(result["observations"]["dWL"].isna().all())

    def test_daily_mean_stage_does_not_replace_gauged_stage(self):
        q_path = self.csv("q.csv", {
            "date": ["2020-01-02"], "wl": [2.5], "Q": [10.0]
        })
        daily_path = self.csv("daily.csv", {
            "date": ["2020-01-01", "2020-01-02"], "wl": [2.0, 3.0]
        })
        result = load_inputs(q_path, daily_path)
        self.assertEqual(result["observations"]["WL"].iloc[0], 2.5)
        self.assertEqual(result["observations"]["daily_WL"].iloc[0], 3.0)
        self.assertEqual(result["observations"]["dWL"].iloc[0], 1.0)

    def test_explicit_date_format_controls_ambiguous_dates(self):
        q_path = self.csv("q.csv", {
            "date": ["02/01/2020", "bad date"], "wl": [2.0, 3.0], "Q": [10.0, 20.0]
        })
        result = load_inputs(q_path, date_format="%d/%m/%Y")
        self.assertEqual(result["observations"]["Date"].iloc[0], pd.Timestamp("2020-01-02"))
        self.assertEqual(len(result["rejected_observations"]), 1)

    def test_conflicting_daily_duplicates_are_rejected(self):
        q_path = self.csv("q.csv", {"WL": [2.0, 3.0], "Q": [10.0, 20.0]})
        daily_path = self.csv("daily.csv", {
            "Date": ["2020-01-01", "2020-01-01"], "WL": [2.0, 3.0]
        })
        with self.assertRaises(ValueError):
            load_inputs(q_path, daily_path)

    def test_equal_daily_duplicates_do_not_corrupt_daily_changes(self):
        q_path = self.csv("q.csv", {
            "Date": ["2020-01-02"], "WL": [3.0], "Q": [20.0]
        })
        daily_path = self.csv("daily.csv", {
            "Date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "WL": [2.0, 2.0, 3.0],
        })
        result = load_inputs(q_path, daily_path)
        self.assertEqual(len(result["daily"]), 2)
        self.assertEqual(result["observations"]["dWL"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
