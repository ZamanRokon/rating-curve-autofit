"""Behavioral checks for the additive power-law workflow."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ratingcurve_autofit import additive


class AdditiveTests(unittest.TestCase):
    @staticmethod
    def data(n=36):
        stage = np.linspace(1.0, 8.0, n)
        return pd.DataFrame({"wl": stage, "discharge": 3.0 * (stage + 1.0)**1.7})

    def test_additive_prediction_activates_terms_at_their_thresholds(self):
        stage = np.array([-2.0, -1.0, 1.0, 4.0, 5.0, 8.0])
        params = np.array([-1.0, 0.0, 2.0, 4.0, np.log10(3.0), 1.2])
        expected = np.maximum(stage + 1.0, 0.0)**2 + 3.0 * np.maximum(stage - 4.0, 0.0)**1.2
        predicted = additive.predict_discharge(params, stage, 2)
        np.testing.assert_allclose(predicted, expected)
        self.assertTrue((np.diff(predicted) >= 0).all())
        self.assertEqual(additive.predict_discharge(params, np.array([]), 2).size, 0)

    def test_extra_terms_need_observations_above_their_activation(self):
        stage = np.arange(1.0, 31.0)
        supported, active_counts = additive.parameter_support(
            np.array([0.0, 0.0, 2.0, 25.0, 0.0, 2.0]), stage, 2
        )
        self.assertFalse(supported)
        self.assertEqual(active_counts, [5])
        supported, active_counts = additive.parameter_support(
            np.array([0.0, 0.0, 2.0, 15.0, 0.0, 2.0]), stage, 2
        )
        self.assertTrue(supported)
        self.assertEqual(active_counts, [15])

    def test_selection_prefers_supported_simple_comparable_candidate(self):
        simple = {"success": True, "empirically_supported": True, "cv_rmse_log10": 0.102,
                  "n_segments": 1, "bic": 12.0}
        complex_fit = {"success": True, "empirically_supported": True, "cv_rmse_log10": 0.100,
                       "n_segments": 2, "bic": 10.0}
        unsupported = {"success": True, "empirically_supported": False, "cv_rmse_log10": 0.001,
                       "n_segments": 3, "bic": -100.0}
        selected, _ = additive.choose_best_model([unsupported, complex_fit, simple])
        self.assertIs(selected, simple)

    def test_repeated_dates_never_straddle_validation_folds(self):
        data = self.data()
        data["date"] = np.repeat(pd.date_range("2020-01-01", periods=18), 2)
        folds, method = additive.validation_folds(data, 5)
        self.assertIn("date", method)
        visits = np.zeros(len(data), dtype=int)
        for test in folds:
            train = np.setdiff1d(np.arange(len(data)), test)
            self.assertFalse(set(data.iloc[train].date).intersection(data.iloc[test].date))
            visits[test] += 1
        np.testing.assert_array_equal(visits, np.ones(len(data), dtype=int))

    def test_cross_validation_does_not_initialize_from_full_data_fit(self):
        data = self.data()
        true_params = np.array([-1.0, np.log10(3.0), 1.7])
        whole_data_fit = {"n_segments": 1, "params": np.array([-50.0, -30.0, 0.1])}
        with patch.object(additive, "fit_model", return_value={
            "success": True, "empirically_supported": True, "params": true_params,
        }) as fitter:
            result = additive.cross_validate_model(data, whole_data_fit, 3)
        self.assertEqual(result["n"], len(data))
        self.assertLess(result["rmse_log10"], 1e-10)
        self.assertEqual(fitter.call_count, 3)
        for call in fitter.call_args_list:
            self.assertLess(len(call.args[0]), len(data))
            self.assertIsNone(call.kwargs.get("start_params"))
            self.assertTrue(call.kwargs.get("global_search", True))

    def test_failed_fold_cannot_produce_optimistic_partial_score(self):
        data = self.data()
        fit = {"n_segments": 1, "params": np.array([-1.0, np.log10(3.0), 1.7])}
        success = {"success": True, "empirically_supported": True, "params": fit["params"]}
        with patch.object(additive, "fit_model", side_effect=[{"success": False}, success, success]):
            result = additive.cross_validate_model(data, fit, 3)
        self.assertFalse(np.isfinite(result["rmse_log10"]))
        self.assertLess(result["n"], len(data))

    def test_csv_nonfinite_values_are_excluded_with_quality_counts(self):
        data = self.data(n=24)
        data = pd.concat([data, pd.DataFrame({
            "wl": [np.inf, 3.0, np.nan], "discharge": [5.0, np.inf, 5.0]
        })], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            data.to_csv(path, index=False)
            cleaned, quality = additive.read_input_csv(path)
        self.assertEqual(len(cleaned), 24)
        self.assertEqual(quality["rows_removed"], 3)
        self.assertTrue(np.isfinite(cleaned[["wl", "discharge"]].to_numpy()).all())

    def test_direct_fit_rejects_nonfinite_training_values(self):
        data = self.data()
        stage = data.wl.to_numpy().copy()
        stage[0] = np.inf
        with self.assertRaises(ValueError):
            additive.fit_model(stage, data.discharge.to_numpy(), 1, global_search=False)


if __name__ == "__main__":
    unittest.main()
