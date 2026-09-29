"""End-to-end checks for the editable script and direct Python entry point."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ratingcurve_autofit import additive, fit_rating_curve
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


def load_starter():
    script = Path(__file__).resolve().parents[2] / "run_rating_curve.py"
    spec = importlib.util.spec_from_file_location("starter", script)
    starter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(starter)
    return starter


@pytest.mark.parametrize("method", ["validated", "additive"])
def test_station_routing_dates_units_and_previous_runs(tmp_path, monkeypatch, method):
    starter = load_starter()
    starter.HERE = tmp_path
    starter.STATION = "Station B"
    starter.METHOD = method
    starter.MAX_SEGMENTS = 1
    starter.BOOTSTRAP_SAMPLES = 0
    starter.STAGE_UNIT = "ft"
    starter.DISCHARGE_UNIT = "ft3/s"
    starter.DATE_FORMAT = "%d/%m/%Y"
    starter.DAILY_STAGES = "daily_stage.csv"
    station = tmp_path / "input" / "Station B"
    station.mkdir(parents=True)
    measurements(station / "measurements.csv")
    observed = pd.read_csv(station / "measurements.csv")
    observed["date"] = pd.date_range("2020-02-01", periods=24).strftime("%d/%m/%Y")
    observed.to_csv(station / "measurements.csv", index=False)
    original = (station / "measurements.csv").read_bytes()
    pd.DataFrame({"date": ["01/02/2020", "02/02/2020"], "wl": [2., 3.]}).to_csv(
        station / "daily_stage.csv", index=False,
    )
    # Another station exists but is never read or written by this run.
    other = tmp_path / "input" / "Station A"
    other.mkdir()
    (other / "measurements.csv").write_text("not this station", encoding="utf-8")
    elsewhere = tmp_path / "editor_working_directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    first = starter.main()
    report = (first / "report.md").read_bytes()
    second = starter.main()

    assert first.parent == tmp_path / "output" / "Station B" / method
    assert second != first
    assert (first / "report.md").read_bytes() == report
    assert not (elsewhere / "output").exists()
    assert not (tmp_path / "output" / "Station A").exists()
    assert (station / "measurements.csv").read_bytes() == original
    assert (other / "measurements.csv").read_text(encoding="utf-8") == "not this station"
    assert (second / "daily_discharge_calculated.csv").exists() == (method == "validated")
    if method == "validated":
        model = json.loads((second / "model.json").read_text(encoding="utf-8"))
        assert model["metadata"]["settings"]["rf"] is False
        assert model["metadata"]["stage_unit"] == "ft"
        cleaned = pd.read_csv(second / "cleaned_measurements.csv")
        assert cleaned.Date.iloc[0] == "2020-02-01"
        np.testing.assert_allclose(cleaned.WL, observed.wl)
        assert (second / "rating_curve.png").stat().st_size > 0
    else:
        model = json.loads((second / "best_model.json").read_text(encoding="utf-8"))
        assert model["stage_unit"] == "ft"
        assert model["discharge_unit"] == "ft3/s"
        metadata = json.loads((second / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["settings"]["date_format"] == "%d/%m/%Y"
        cleaned = pd.read_csv(second / "cleaned_data.csv")
        assert cleaned.date.iloc[0] == "2020-02-01"
        np.testing.assert_allclose(cleaned.wl, observed.wl)
        rating = pd.read_csv(second / "rating_table.csv")
        assert "stage_ft" in rating and "discharge_median_ft3/s" in rating
        assert np.isfinite(rating.select_dtypes(include="number")).all().all()
        assert additive.STAGE_UNIT == "m"  # Per-run labels do not alter module defaults.
        assert (second / "plots" / "rating_curve.png").stat().st_size > 0


@pytest.mark.parametrize("station", ["", "..", "../outside", "a/b", "a\\b", "D:\\outside"])
def test_station_must_be_one_folder_under_input(tmp_path, station):
    starter = load_starter()
    starter.HERE, starter.STATION = tmp_path, station
    with pytest.raises(ValueError, match="STATION"):
        starter.main()
    assert not (tmp_path / "output").exists()


def test_script_reports_missing_station_without_traceback(tmp_path):
    script = Path(__file__).resolve().parents[2] / "run_rating_curve.py"
    copied_script = tmp_path / script.name
    copied_script.write_bytes(script.read_bytes())
    process = subprocess.run([sys.executable, "-B", str(copied_script)],
                             cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert "measurements.csv" in process.stderr
    assert "Traceback" not in process.stderr


def test_daily_file_must_belong_to_selected_station(tmp_path):
    starter = load_starter()
    starter.HERE = tmp_path
    station = tmp_path / "input" / "test"
    station.mkdir(parents=True)
    measurements(station / "measurements.csv")
    starter.DAILY_STAGES = "../another_station/daily.csv"
    with pytest.raises(ValueError, match="DAILY_STAGES"):
        starter.main()
    assert not (tmp_path / "output").exists()


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
