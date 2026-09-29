"""End-to-end checks for the editable script and direct Python entry point."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ratingcurve_autofit import fit_rating_curve
from ratingcurve_autofit.validated import Settings, load_model, predict_pipeline


def measurements(path):
    stage = np.linspace(1.0, 7.0, 24)
    pd.DataFrame({"wl": stage, "discharge": 3 * (stage + 1)**1.7}).to_csv(path, index=False)


def test_python_fit_writes_daily_estimates_and_preserves_extrapolation_flags(tmp_path):
    source = tmp_path / "measurements.csv"
    measurements(source)
    original = source.read_bytes()
    daily = tmp_path / "daily.csv"
    pd.DataFrame({"date": ["01/02/2020", "02/02/2020", "03/02/2020"],
                  "wl": [0.5, 3.0, 8.0]}).to_csv(daily, index=False)

    result = fit_rating_curve(
        source, daily_stages=daily, output_folder=tmp_path / "results",
        date_format="%d/%m/%Y", stage_unit="ft", discharge_unit="ft3/s",
        settings=Settings(max_segments=1, folds=3, inner_folds=3),
    )

    assert result.is_absolute()
    assert source.read_bytes() == original
    model = load_model(result / "model.json")
    assert model["metadata"]["stage_unit"] == "ft"
    assert model["metadata"]["discharge_unit"] == "ft3/s"
    predictions = pd.read_csv(result / "daily_discharge_calculated.csv")
    assert predictions.Date.iloc[0] == "2020-02-01"
    assert predictions.Stage_extrapolation.tolist() == [
        "below_measured_range", "within_measured_range", "above_measured_range",
    ]
    assert predictions.Q_Lower.isna().tolist() == [True, False, True]
    assert np.isfinite(predictions.Q_Estimate).all()
    np.testing.assert_allclose(predict_pipeline(model, predictions), predictions.Q_Estimate, rtol=1e-12)
    measured = pd.read_csv(result / "measurement_predictions.csv")
    assert np.isfinite(measured.Q_NestedCV).all()
    for filename in ["report.md", "equation.txt", "rating_curve.png", "rating_table.csv"]:
        assert (result / filename).stat().st_size > 0


def test_editable_script_uses_its_folder_when_editor_working_directory_differs(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "run_rating_curve.py"
    spec = importlib.util.spec_from_file_location("starter", script)
    starter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(starter)
    # Loading the script must not start an analysis. Edit its ordinary variables.
    starter.HERE = tmp_path
    starter.MEASUREMENTS = "station.csv"
    starter.DAILY_STAGES = None
    starter.OUTPUT_FOLDER = "results"
    measurements(tmp_path / "station.csv")
    elsewhere = tmp_path / "editor_working_directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    first = starter.main()
    report = (first / "report.md").read_bytes()
    second = starter.main()

    assert first.parent == tmp_path / "results"
    assert second != first
    assert (first / "report.md").read_bytes() == report
    assert not (elsewhere / "results").exists()
    assert not (second / "daily_discharge_calculated.csv").exists()
    model = json.loads((second / "model.json").read_text(encoding="utf-8"))
    assert model["metadata"]["settings"]["max_segments"] == 2
    assert model["metadata"]["settings"]["rf"] is False


@pytest.mark.parametrize("settings, message", [
    (Settings(folds=1), "folds"),
    (Settings(coverage=1), "coverage"),
    (Settings(max_segments=1, threshold=2), "threshold"),
    (Settings(max_segments=3), "max_segments"),
    (Settings(shape="unknown"), "shape"),
    (Settings(selection_metric="unknown"), "selection_metric"),
])
def test_python_call_rejects_invalid_settings_before_reading_data(tmp_path, settings, message):
    with pytest.raises(ValueError, match=message):
        fit_rating_curve(tmp_path / "missing.csv", settings=settings)
