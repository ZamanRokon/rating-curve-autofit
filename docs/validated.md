# Smooth rating with nested validation

The `validated` backend compares one- and two-regime curves, evaluates the whole selection procedure on omitted periods, and optionally calculates discharge for a daily stage series. Its stage-only curve is nondecreasing and has matching discharge and slope at the transition. Random-forest residual correction is optional and **off by default**.

## Run

```sh
rating-curve validated examples/measurements.csv --daily examples/daily_stage.csv --date-format "%Y-%m-%d" --out results/validated
```

Equivalent commands are `python -m ratingcurve_autofit validated ...` and, from the repository root, `python universal_rating_curve.py ...`.

Without daily stages, omit `--daily`; fitting, nested validation, the rating table, and reports still run. Add `--shape convex` when upward curvature is physically justified. Convexity permits a straight branch at exponent one and does not guarantee a visibly curved result.

Each run creates a timestamped directory under `--out`, or under `universal_rating_results` beside the measurement CSV when `--out` is omitted. The complete path is printed. Input files are read only. Nested fitting can take several minutes.

## Input requirements

Paired measurements need stage and discharge. Dates are recommended; an optional constant `DL` column supplies a danger-level reference in the plot only.

```csv
date,wl,discharge
2020-01-01,1.20,8.40
2020-01-15,1.55,15.70
2020-02-01,2.10,33.50
```

The optional daily file needs date and stage:

```csv
date,wl
2020-01-01,1.20
2020-01-02,1.25
2020-01-03,1.18
```

These illustrate the schema; they are too short for fitting. At least **20 valid measurements** must remain after cleaning, with varying discharge, some positive discharge, and five distinct stages. Viable nested training groups are also required: 20 rows alone do not guarantee usable splits. Two-regime candidates require at least 12 observations and three distinct stages on each side of the transition in each applicable training fold by default. Ineligible candidates are reported while eligible single-regime candidates remain available.

Use one station, consistent units, and a consistent stage datum. Finite negative stages can be valid relative to a datum. Negative discharge is rejected; zero discharge is retained. The estimated zero-flow stage stays below the minimum measured stage, so the model cannot identify an internal dry cutoff where several different measured stages all have exactly zero flow.

Numeric fields use a decimal point. Convert decimal commas or thousands separators before fitting. Unit options change labels only; no units are converted.

### Columns and dates

Common headers are recognized without regard to case or punctuation, including `WL`, `water_level`, `stage`, `h`, `Q`, `discharge`, `flow`, `date`, and `timestamp`. Common unit suffixes such as `Stage (m)` are recognized. Comma, semicolon, tab, and pipe delimiters are inferred. Additional metadata columns are allowed.

If aliases are ambiguous or headers are custom, select columns explicitly:

```sh
rating-curve validated measured.csv --daily daily.csv --date-col "Sample date" --stage-col "Gauged height" --discharge-col "Measured flow" --daily-date-col "Day" --daily-stage-col "Daily height" --date-format "%d/%m/%Y"
```

Specify a date convention when dates are ambiguous. `03/04/2020` is 4 March with `%m/%d/%Y` and 3 April with `%d/%m/%Y`. `--dayfirst` requests day-first parsing; otherwise the default is month-first. A supplied `--date-format` applies to both files. Standardize their date formats first if they differ. Dates are reduced to the local calendar day; subdaily times are not model inputs.

If no measurement date column exists, validation uses input-row blocks and cannot establish temporal independence. If a date column exists, invalid or missing dates are rejected rather than treated as undated measurements.

### Cleaning and daily stage change

Rejected rows retain source fields, `source_row`, and `rejection_reason` in output CSVs. Invalid/nonfinite numbers, negative discharge, and unparseable supplied dates are rejected. Exact duplicate date/stage/discharge measurements are collapsed with a note. Different measurements on the same date remain valid and stay together in validation folds.

Daily rows are sorted by date. Identical duplicate date/stage pairs are collapsed. Conflicting stages for one date stop the run: resolve the conflict explicitly instead of relying on automatic averaging.

Daily `dWL` is today's stage minus the **previous calendar day's** stage. The first day and a day following a gap have missing `dWL`. Missing days are neither interpolated nor filled with zero. Measurements receive this daily feature through an exact calendar-date join, so training and daily prediction share the same definition.

The gauged stage remains the measured stage. The report counts matched dates whose gauged stage differs from the daily value by more than 0.001 stage units. Review datum, timing, and daily-mean versus instantaneous differences. For nonlinear ratings, discharge evaluated at daily average stage is not necessarily true daily average discharge.

## Fit and nested validation

The workflow compares one- and two-regime curves with linear squared error and normalized `log1p` squared error. It estimates stage/discharge normalization within each training set. Final selection uses held-out RMSE by default; `--selection-metric RMSLE` uses a normalized score with `log1p(Q / scale)`, where scale is the scored observations' 90th-percentile discharge. This score differs from unscaled conventional RMSLE. Candidates within 2% of the best score prefer fewer regimes and then no RF.

Transition candidates come from training-stage quantiles. A danger level never determines the transition. `--threshold VALUE` fixes the transition for eligible two-regime candidates when hydraulic information justifies it; the workflow can still select one regime. `--max-segments 1` restricts the model family to one regime. The [method comparison](methods.md) gives the equations and shape assumptions.

Outer folds hold out contiguous year groups when multiple years exist, otherwise date groups, with input-row blocks reserved for undated data. Inner splits choose the model, transition, loss, and optional RF weight without seeing the outer group. Default outer/inner counts are five/three. Same-day measurements stay together. After evaluation, selection and fitting repeat using all accepted observations for the saved model.

Training can include observations after a held-out period. The reported evaluation measures reconstruction across omitted periods, **not future-only forecasting**. Review results by year and for the upper stage quartile as well as the overall score.

## Optional residual forest

Install the extra and request candidate RF corrections:

```sh
python -m pip install -e ".[rf]"
rating-curve validated examples/measurements.csv --daily examples/daily_stage.csv --rf --out results/validated_rf
```

The forest learns measured discharge minus the fitted base curve using `WL` and daily `dWL`. Each applicable training split needs at least 24 rows with usable `dWL` and 70% feature coverage. Correction weights 0, 0.25, 0.5, and 1 are compared inside model selection. Requesting RF does not force its selection; ineligible candidates are recorded.

`Q_RatingCurve` remains the static, shape-constrained rating. `Q_Estimate` includes any selected correction and does **not** inherit monotonicity or convexity. The correction falls back to zero for missing features or features outside training support and fades near the gauged stage boundaries. The rating-curve plot shows the static curve.

## Options

Run `rating-curve validated --help` for the full command reference.

| Option | Default | Purpose |
|---|---|---|
| `--daily PATH` | None | Daily date/stage CSV. Required for `--rf`. |
| `--out PATH` | Beside measurements | Parent directory for timestamped results. |
| `--shape {monotone,convex}` | `monotone` | Stage-only shape constraint. |
| `--max-segments {1,2}` | `2` | Largest candidate regime count. |
| `--min-regime N` | `12` | Minimum observations per regime in each training fit. |
| `--threshold VALUE` | Estimated | Fixed transition for two-regime candidates. |
| `--folds N` / `--inner-folds N` | `5` / `3` | Outer evaluation and inner selection folds. |
| `--selection-metric {RMSE,RMSLE}` | `RMSE` | Inner selection metric. |
| `--rf` | Off | Compare residual RF corrections. |
| `--seed N` | `42` | Random seed for RF fitting. |
| `--coverage VALUE` | `0.90` | Empirical band target, greater than 0.5 and below 1. |
| `--stage-unit TEXT` / `--discharge-unit TEXT` | `m` / `m3/s` | Output labels; no conversion. |
| `--date-format FORMAT` / `--dayfirst` | Month-first parsing | Date convention for both files. |
| `--date-col`, `--stage-col`, `--discharge-col` | Detected | Measurement column overrides. |
| `--daily-date-col`, `--daily-stage-col` | Detected | Daily column overrides. |

## Outputs and uncertainty

| File | Contents |
|---|---|
| `report.md` | Selected model, equation, validation, and data/fit notes. Read first. |
| `rating_curve.png` | Measurements, static rating, empirical band, and extrapolation shading. |
| `daily_discharge_calculated.csv`, `daily_discharge.png` | Daily estimates and flags; created with a daily input. |
| `rating_table.csv` | 400-stage lookup table spanning measured and supplied daily stages. |
| `equation.txt` | Static stage-only equation. |
| `validation_scores.csv` | Nested held-out overall and upper-stage-quartile scores. |
| `validation_folds.csv`, `validation_by_year.csv` | Held-out scores by fold and, when dated, year. |
| `validation_diagnostics.png` | Held-out predictions and residuals. |
| `model_comparison.csv` | Final inner selection scores and candidate failure reasons, not outer performance. |
| `measurement_predictions.csv` | Full-fit and nested held-out predictions and residuals. |
| `cleaned_measurements.csv` | Accepted measurements and matched daily features. |
| `rejected_measurements.csv`, `rejected_daily_rows.csv` | Excluded source records and reasons. |
| `model.json` | Parameters, settings, calibration, input hashes, and metadata. |
| `residual_rf.joblib` | Created only for a selected RF; keep beside its `model.json`. |

`Q_NestedCV` on measurement rows is the prediction to use for evaluating held-out errors. `Q_RatingCurve` is the static rating; `Q_Estimate` is the full selected prediction.

`Q_Lower` and `Q_Upper` are empirical bands calibrated from scaled absolute nested-validation errors. Their default target is 90%. Coverage is approximate, has not been independently validated, and is not guaranteed under temporal change. The static rating and RF fallback rows use separately calibrated base-curve errors. Bounds are blank outside the measured stage range; discharge estimates there are extrapolations.

Review `Stage_extrapolation`, `Interval_status`, `dWL_available`, `Predictor`, and `Date_outside_gauging_period` where present. `Local_observation_count` counts measurements within 5% of the measured stage span. It indicates local coverage, not independently calculated uncertainty.

## Reuse a saved model

The CLI fits a new model. To reuse fitted parameters without refitting, replace the placeholder below with a completed run's `model.json`:

```python
import numpy as np
import pandas as pd
from ratingcurve_autofit.core import predict_curve
from ratingcurve_autofit.validated import load_model, predict_pipeline

model = load_model("results/your_validated_run/model.json")
stage = np.linspace(model["metadata"]["stage_min"], model["metadata"]["stage_max"], 3)
new = pd.DataFrame({"WL": stage, "dWL": np.nan})
new["Q_RatingCurve"] = predict_curve(model["base"], new["WL"].to_numpy())
new["Q_Estimate"] = predict_pipeline(model, new)
new["Stage_extrapolation"] = ~new["WL"].between(
    model["metadata"]["stage_min"], model["metadata"]["stage_max"]
)
new.to_csv("new_predictions.csv", index=False)
```

This supplies no daily change, so even selected RF models use the static rating. To apply a selected residual model, supply `dWL` from consecutive calendar-day stages in the training units and convention, preserving missing differences. This low-level example evaluates predictions; it does not recreate the CLI's calibrated interval columns.

Load only trusted `joblib` files because they contain serialized Python objects. An additive `best_model.json` has a different schema and is not accepted by this loader.
