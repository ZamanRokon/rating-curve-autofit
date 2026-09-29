# Technical manual

This manual describes the station-folder workflow and the code in this
repository. There are two fitting methods: `additive` and `validated` (a smooth
rating with nested validation). The optional random forest corrects residuals
within validated; it is not a third independent rating method.
Both methods can convert daily stages to discharge after fitting paired
stage–discharge measurements.

## Contents

1. [Files and folders](#files-and-folders)
2. [Install and select Python](#install-and-select-python)
3. [Run the supplied test station](#run-the-supplied-test-station)
4. [Add another station](#add-another-station)
5. [Script settings](#script-settings)
6. [Input data and cleaning](#input-data-and-cleaning)
7. [Additive method](#additive-method)
8. [Smooth validated method](#smooth-validated-method)
9. [Daily discharge and random forest](#daily-discharge-and-random-forest)
10. [Read and assess results](#read-and-assess-results)
11. [Python functions and saved models](#python-functions-and-saved-models)
12. [Automated checks](#automated-checks)
13. [Troubleshooting](#troubleshooting)
14. [Editing and GitHub](#editing-and-github)
15. [References and credit](#references-and-credit)

## Files and folders

```text
rating-curve-autofit/
  docs/technical_manual.md
  input/
    test/measurements.csv
    test/daily_stage.csv
    Feni_Ramgarh/measurements.csv       # Your station, when added
    Feni_Ramgarh/daily_stage.csv        # Optional
  output/
    test/
    Feni_Ramgarh/                       # Created when this station runs
      validated/measurements_<time>/
      additive/measurements_<time>/
  src/
    ratingcurve_autofit/                # Calculation code
    tests/                             # Automated checks
  run_rating_curve.py
  requirements.txt
  README.md
  LICENSE
```

Only `run_rating_curve.py` is the user-facing analysis script. Paths start
beside it, including when Spyder or VS Code uses a different working directory.
Each run gets a timestamped folder. Rerunning a station or switching methods
preserves earlier results. The script runs one selected station at a time;
adding a folder does not automatically run it.

The hidden `.git` stores version history and the GitHub connection. `.gitignore`
keeps local data, results and caches out of ordinary commits. `.github` contains
automated checks. Keep those files. `output/test/.gitkeep` retains that empty
folder in Git.

This layout runs source files directly. Install the libraries in
`requirements.txt`; do not install the repository as a package. Earlier
launchers, command-line subcommands and package-building files have been removed.
`MANIFEST.in` selected extra files for a Python source distribution; it was not
part of either numerical method. This workflow needs neither it nor
`pip install -e .` nor a wheel build.

On the maintained PC the project is `D:\rating-curve-autofit`. Previous layout
files and backups are outside it in
`D:\rating-curve-autofit-support\previous-layout`. Historical example results
remain in `output/test/previous_runs`; their recorded input paths describe
those original runs and were not rewritten.

## Install and select Python

Use Python **3.10 or later**. Open a terminal in the repository folder:

```powershell
cd D:\rating-curve-autofit
python -m pip install -r requirements.txt
```

Required libraries: NumPy, pandas, SciPy and Matplotlib. Optional random forest
needs scikit-learn and joblib. Automated checks need pytest.

### Existing environment on the maintained PC

The prepared interpreter is:

```text
D:\rating-curve-autofit-support\environment\Scripts\python.exe
```

Run it directly from PowerShell; activation is unnecessary:

```powershell
& "D:\rating-curve-autofit-support\environment\Scripts\python.exe" run_rating_curve.py
```

In Spyder or VS Code, select that interpreter, open the script and press Run.
To identify the editor's Python, run `import sys` and `print(sys.executable)`
in its console. Libraries installed into another environment will not be
available to the selected interpreter.

### New installation on another PC

Use an existing scientific Python environment, or create one outside the project.
From the repository folder:

```sh
python -m venv ../rating-curve-environment
```

Windows PowerShell:

```powershell
& "..\rating-curve-environment\Scripts\python.exe" -m pip install -r requirements.txt
& "..\rating-curve-environment\Scripts\python.exe" run_rating_curve.py
```

macOS/Linux:

```sh
../rating-curve-environment/bin/python -m pip install -r requirements.txt
../rating-curve-environment/bin/python run_rating_curve.py
```

The remaining examples use `python` for readability. Substitute your selected
interpreter in every install, run and test command. On Windows, `py` may be used
instead when it selects the intended Python installation.

## Run the supplied test station

### Smooth method

The script starts with:

```python
STATION = "test"
METHOD = "validated"
DAILY_STAGES = None
MAX_SEGMENTS = None
```

Run:

```sh
python run_rating_curve.py
```

It reads `input/test/measurements.csv`, prints validation progress and creates
a run under `output/test/validated/`. Open `report.md`, `rating_curve.png` and
`rating_table.csv` there first.

For daily discharge, change `DAILY_STAGES = "daily_stage.csv"`. This reads
`input/test/daily_stage.csv` and additionally writes
`daily_discharge_calculated.csv` and `daily_discharge.png` in the run folder.
For a trial restricted to one regime, set `MAX_SEGMENTS = 1`; `None` restores
the method's normal candidate range.

### Additive method

In the same script set:

```python
METHOD = "additive"
DAILY_STAGES = "daily_stage.csv"
MAX_SEGMENTS = 1
BOOTSTRAP_SAMPLES = 0
```

Run the same command. This quick trial writes to `output/test/additive/`, with
the rating plot at `plots/rating_curve.png`. Bootstrap 0 deliberately omits
uncertainty intervals. For the normal additive analysis, set `MAX_SEGMENTS = None`
and `BOOTSTRAP_SAMPLES = 200`: eligible one-/two-/three-term models are compared,
then the selected term count is bootstrapped. This can take several minutes.

This also writes `daily_discharge_calculated.csv` and `daily_discharge.png`
directly in the additive run folder. The CSV gives median and bias-corrected
mean discharge, with flags for stages outside the measured range. Daily output
has no uncertainty bounds, including when bootstrap is enabled; bootstrap
intervals are in the measured-range rating table. Keep `DAILY_STAGES` set when
switching methods to use the same daily input, or set it to `None` to skip daily
predictions with either method.

### What the example represents

These are synthetic data, not field measurements: 90 consecutive daily stages
starting 1 January 2020 and 60 measurement dates sampled from days 1–89. For
integer day `t = 0, ..., 89`:

```text
h(t) = 4.5 + 1.7 sin(2 pi t / 23) + 0.4 cos(2 pi t / 9)
Q(t) = 6 (h(t) - 1)^1.7 exp(epsilon)
epsilon ~ Normal(0, 0.06^2)
```

The NumPy generator seed is `20260907`. Date selection occurs before discharge
noise is sampled. Numbers are rounded to eight decimals. To regenerate the
example CSVs, deliberately overwriting those two files:

```sh
python src/tests/generate_example_data.py
```

A successful example verifies software operation, not real-station accuracy.

## Add another station

1. Create a folder such as `input/Feni_Ramgarh/`.
2. Save paired measurements there as **`measurements.csv`**.
3. Optionally add `daily_stage.csv` or another named daily CSV.
4. Set `STATION = "Feni_Ramgarh"`, choose `METHOD`, and check dates/units.
5. Run the script. Results go under `output/Feni_Ramgarh/<method>/`.

The station setting is a folder name, not an absolute path. Spaces and ordinary
Unicode names work. Do not use slashes or `..`. Every station uses the filename
`measurements.csv`; files are not searched for automatically. The optional daily
filename must also refer to a file in that station's folder.

Change `STATION` to process another folder. Input files are read without
modification. Use one station, stage datum and consistent rating period per
analysis; a folder name alone does not separate hydraulic changes or datum shifts.

## Script settings

These are ordinary assignments in [`run_rating_curve.py`](../run_rating_curve.py).
Strings need quotes; `None`, `True` and `False` do not.

| Setting | Default | Meaning |
|---|---|---|
| `STATION` | `"test"` | Folder under `input/`; also names its output folder. |
| `METHOD` | `"validated"` | `"validated"` or `"additive"`. |
| `DAILY_STAGES` | `None` | Both methods: daily filename in the station folder, e.g. `"daily_stage.csv"`; `None` skips daily predictions. |
| `DATE_FORMAT` | `"%Y-%m-%d"` | Date convention for both CSVs. |
| `STAGE_UNIT` | `"m"` | Label only; no number conversion. |
| `DISCHARGE_UNIT` | `"m3/s"` | Label only; no number conversion. |
| `MAX_SEGMENTS` | `None` | Default maximum: 2 for validated, 3 for additive. A maximum, not a forced count. Use 1 for a single branch/term. |
| `BOOTSTRAP_SAMPLES` | `200` | Additive refits; use a nonnegative integer. 0 omits intervals. |
| `SHAPE` | `"monotone"` | Validated: nondecreasing; `"convex"` additionally imposes upward curvature or straight branches. |
| `USE_RANDOM_FOREST` | `False` | Validated: compare residual forests; needs daily stages and optional libraries. |

Date examples: `%Y-%m-%d` means `2020-01-31`, `%d/%m/%Y` means `31/01/2020`,
and `%m/%d/%Y` means `01/31/2020`. Use one convention in both files. `None`
requests general date interpretation, which can misinterpret ambiguous dates.
Advanced fold counts, fixed transitions and coverage settings are covered in
[Python functions](#python-functions-and-saved-models).

## Input data and cleaning

### Measurements

Use a comma-separated CSV with these headers:

```csv
date,wl,discharge
2020-01-01,1.20,8.40
2020-01-15,1.55,15.70
2020-02-01,2.10,33.50
```

These three rows illustrate the format and are insufficient for fitting. `wl`
is water level and `discharge` is measured flow. Dates are recommended but the
whole date column may be omitted. Use decimal points without thousands
separators. Export Excel worksheets as CSV; `.xlsx` files are not read directly.
Finite negative stages can be valid relative to the station datum.

| Requirement/treatment | Additive | Validated |
|---|---|---|
| Minimum usable observations | 20/40/60 for 1/2/3 terms. | At least 20 plus viable nested validation groups. |
| Distinct stages | At least 4. | At least 5; discharge must vary and include positive values. |
| Extra-regime support | At least 10 observations above each later activation stage. | By default 12 observations and 3 distinct stages on each side in applicable training folds. |
| Zero discharge | Excluded because fitting uses log discharge. | Retained. |
| Invalid numbers | Nonfinite stage/discharge and nonpositive discharge excluded. | Nonfinite stage/discharge and negative discharge excluded. |
| Invalid supplied dates | Row retained; incomplete dates cause stage-stratified validation. | Row rejected with a reported reason. |
| Repeated measurements | Retained and counted. | Exact duplicate date/stage/discharge records collapsed with a note. |
| No date column | Stage-stratified validation. | Input-row blocks; temporal independence is unknown. |

Counts alone do not establish a good rating. In validated, many observations
on a few dates/years can leave insufficient training data even above 20 total.
Neither support rule identifies a physical control. Review cleaning counts and
stage coverage.

Additive measurements need exact `wl`, `discharge` and optional `date` headers. Validated
also recognizes aliases such as `stage`, `WL`, `Q`, `flow` and `timestamp`, and
common delimiters. Use standard comma-separated headers for files used by both
methods. Ambiguous aliases need renamed headers or explicit column selection.
Validated measurement input and both methods' daily inputs reserve `source_row`
and `rejection_reason` for audit output.

### Daily stages

```csv
date,wl
2020-01-01,1.20
2020-01-02,1.25
2020-01-03,1.18
```

Both methods use the same daily input checks. Dates and finite stages are
required, and `DATE_FORMAT` applies to daily dates too. Rows are sorted by date.
Identical duplicates are collapsed; conflicting stages for one date stop the
analysis. Missing days are not filled or interpolated. Invalid rows appear in
`rejected_daily_rows.csv` with the original row number and reason; if no usable
daily rows remain, the run stops before fitting. Recognized daily header aliases
and delimiters are shared by both methods, but `date,wl` is the simplest format.

Daily rows do not supply additional discharge measurements. The additive fit
uses only the paired measurement file. Validated also fits paired measurements;
its optional random forest can use daily stage changes as additional predictors.
In validated, a constant optional `DL` measurement column supplies a plot
reference only, not a fitted transition.

## Additive method

Implementation: [`additive.py`](../src/ratingcurve_autofit/additive.py).
For stage h:

```text
Q_median(h) = sum over j = 1..K of alpha_j * max(h - h_j, 0)^beta_j
K = 1, 2 or 3; alpha_j > 0 and beta_j > 0
```

The first zero-flow stage is below the measured range; later activation stages
are ordered. Positive terms make the curve nondecreasing. Slopes need not match
at activation stages, and exponents below one can give steep right-hand slopes.
Convexity is not imposed.

The optimizer searches globally and refines locally to minimize squared
`log10(Q)` residuals. Eligible term counts are evaluated by five-fold
cross-validation: date blocks when all dates are usable, otherwise
stage-stratified folds. Same-date observations stay together. Each fold is
fitted using its own training observations. The simplest supported candidate
within 3% of the best usable CV log RMSE is preferred. If usable candidate CV
is unavailable, selection falls back to BIC and records that limitation.
AIC, AICc and BIC are also exported as diagnostics.

These candidate scores select the model; they are not an independent evaluation
of the whole selection process. Check validation sample counts and failures.

### Median, mean and uncertainty

Under the lognormal error assumption the equation estimates conditional median
discharge. A separate table column applies the mean correction:

```text
Q_mean = Q_median * exp(0.5 * (ln(10) * sigma_log10Q)^2)
```

This assumes constant log-error variance and is not an independently fitted
mean-discharge model. Record which quantity you use.

Bootstrap resamples observation pairs and refits the selected term count.
Default 90% pointwise curve intervals describe fitted-median variability;
prediction intervals additionally include simulated lognormal residual noise.
This does not repeat term-count selection or preserve temporal dependence.
Check successful refit counts: few successful fits or changing residual spread
weaken interpretation. Bootstrap 0 produces no intervals. The 100-point rating
table is confined to the measured stage range. Optional daily output evaluates
the selected equation at each supplied stage and flags extrapolations. It
contains median/mean estimates only; bootstrap intervals are not transferred
or extrapolated from the rating table to daily rows.

## Smooth validated method

Implementation: [`validated.py`](../src/ratingcurve_autofit/validated.py) and
[`core.py`](../src/ratingcurve_autofit/core.py). The name describes validation,
not certification of a station rating.

A single regime uses a power law. For two regimes, let b be a zero-flow stage
below the measured range, t > b the transition, and a, pL, pH positive:

```text
QL(h) = a * max(h - b, 0)^pL

For h > t:
QH(h) = QL(t) * [1 + (pL/pH) * (((h-b)/(t-b))^pH - 1)]
```

Branches have matching value and first derivative at t; second derivatives may
differ. `monotone` uses exponent bounds 0.3–5. `convex` uses 1–5, permitting a
straight branch at exponent one. The common zero-flow stage is below the measured
range, so this model cannot identify an internal dry cutoff with different
measured stages all having exactly zero flow.

Eligible one-/two-regime models are compared using linear squared error and
normalized `log1p` squared error. Stage/discharge normalization is learned
within each training subset. Transition candidates come from training-stage
quantiles. They describe changes in curve shape, not automatically established
hydraulic controls. A fixed transition may be supplied through `Settings` when
station information supports it.

### Nested validation

Outer folds hold out contiguous year groups when multiple years exist;
otherwise they hold out date groups. Undated inputs use row blocks. Inner
validation selects the regime count, transition, loss and optional forest
weight without seeing the outer test group. Outer/inner defaults are five/three
folds. Same-date measurements never cross a train/test boundary.

Inner selection uses RMSE by default. Candidates within 2% of the best score
prefer fewer regimes and then no residual forest. Final selection and fitting
using all accepted measurements produce the saved model after outer evaluation.

`validation_scores.csv` reports outer held-out performance;
`model_comparison.csv` reports final inner selection. Training can include
observations after a held-out period, so these scores evaluate reconstruction
across omitted periods, not future-only forecasting.

### Empirical error bands

Absolute nested held-out errors are scaled by predicted discharge plus a
reference flow and calibrated to the target coverage (90% by default).
Base-curve errors separately calibrate the static rating and forest-fallback
rows. Coverage is approximate, not independently verified or guaranteed under
temporal change. Bounds are blank outside the measured stage range; estimates
there are still reported and flagged as extrapolations.

## Daily discharge and random forest

Set these in `run_rating_curve.py`:

```python
STATION = "test"                  # Or your station folder name
METHOD = "additive"               # Or "validated"
DAILY_STAGES = "daily_stage.csv"
```

Run the same script. Both methods read `input/<station>/measurements.csv` and
the selected daily file, then write `daily_discharge_calculated.csv` and
`daily_discharge.png` to `output/<station>/<method>/<run>/`. Daily predictions
need no random forest. `DAILY_STAGES = None` skips them.

| Daily output | Additive | Validated |
|---|---|---|
| Discharge columns | `Q_Median`, `Q_Mean_bias_corrected` | `Q_RatingCurve`, `Q_Estimate` |
| Uncertainty | No daily bounds; bootstrap intervals remain in `rating_table.csv`. | `Q_Lower`, `Q_Upper`: empirical error band within measured stages. |
| Range flags | `Stage_extrapolation` | `Stage_extrapolation` and additional support/interval flags. |
| Role of daily data | Prediction stages only; fitting, selection and bootstrap are unchanged. | Prediction stages; optional daily changes also support residual-forest fitting. |

For additive, `Date`, `WL` and `source_row` identify the date, supplied stage
and original CSV line. Discharge columns use `DISCHARGE_UNIT`; `WL` uses
`STAGE_UNIT`. The mean column applies the same lognormal correction as the
rating table. At or below the fitted zero-flow stage, the equation returns zero;
this is a model extrapolation, not confirmation of a dry river. The daily plot
shows both estimates and marks out-of-range stages; missing days break its lines.

Both methods label stages `below_measured_range`, `within_measured_range`, or
`above_measured_range`; the measured minimum and maximum count as within range.
An extrapolated estimate is still reported, but needs separate hydraulic
assessment. Additive's rating table stays within measured stages. Validated's
table spans measured and supplied daily stages, with blank bounds outside the
measured range.

Applying a nonlinear rating to daily mean stage does not generally produce
daily mean discharge. Additive's bias correction estimates a conditional mean
at the supplied stage; it does not reconstruct within-day stage variability.

### Optional random forest (validated only)

For optional residual correction, install:

```sh
python -m pip install scikit-learn joblib
```

Then set:

```python
METHOD = "validated"
DAILY_STAGES = "daily_stage.csv"
USE_RANDOM_FOREST = True
```

Daily `dWL` is today's stage minus the previous **calendar day's** stage. It is
missing on the first day and after gaps. Measurement dates join to these daily
features, but gauged stages are not replaced by daily stages. The report counts
matched stage differences exceeding 0.001 stage units.

The forest learns residual discharge from stage and daily `dWL`. Each applicable
training split needs at least 24 usable `dWL` rows and 70% feature coverage.
Weights 0, 0.25, 0.5 and 1 are compared within selection. Enabling the option does
not force a forest into the result. Missing/unsupported features fall back to
the static curve; corrections fade near measured boundaries.

`Q_RatingCurve` is the static shape-constrained rating. `Q_Estimate` includes any
selected forest correction and does not inherit the static curve's shape
guarantees. The rating plot shows the static curve. Q(daily mean stage) is not
necessarily daily mean discharge for a nonlinear rating.

## Read and assess results

Start with the **report**, **rating plot** and **rating table** in the printed
run folder. Review omitted observations, low-/high-stage coverage and residuals
before adopting a rating. Static stage–discharge relations do not resolve all
backwater, reversing/tidal flow, hysteresis or changing controls.

### Additive files

| File | Meaning |
|---|---|
| `report.md` | Equation, selection, scores, data notes and uncertainty summary. |
| `plots/rating_curve.png` | Measurements, median curve and any bootstrap bands. |
| `rating_table.csv` | 100 measured-range stages, median/mean flow and available intervals. |
| `daily_discharge_calculated.csv`, `daily_discharge.png` | Daily median/mean estimates and range flags when daily input is supplied; no daily bounds. |
| `rejected_daily_rows.csv` | Invalid daily source records and reasons, when daily input is supplied; may contain headers only. |
| `equation.txt` | Median-discharge equation. |
| `input_quality.json` | Accepted/excluded measurement counts, repeats and data ranges; daily counts and cleaning notes when supplied. |
| `cleaned_data.csv` | Retained observations. |
| `best_model.json` | Term count, parameters, equation, measured range and diagnostics. |
| `best_parameters.csv` | Parameters and available bootstrap bounds. |
| `model_comparison.csv` | Candidate support, information criteria, CV scores and failures. |
| `fitted_values_and_residuals.csv` | Median/mean predictions, residuals and outlier flags. |
| `run_metadata.json` | Measurement and optional daily input hashes, library versions and settings including dates and units. |
| Other `plots/` files | Log rating, residual plots, QQ plot, comparison and available time diagnostics. |

Default lookup columns are `stage_m`, `discharge_median_m3/s` and
`discharge_mean_bias_corrected_m3/s`. Unit-label changes also change these suffixes.
Available bootstrap columns are `curve_lower_90`, `curve_upper_90`,
`prediction_lower_90` and `prediction_upper_90`. Outlier flags do not automatically
remove observations.

### Validated files

| File | Meaning |
|---|---|
| `report.md` | Equation, selection, validation and data/fit notes. |
| `rating_curve.png` | Static rating, observations, empirical band and extrapolation shading. |
| `rating_table.csv` | 400-stage lookup; daily stages can extend its range. |
| `equation.txt` | Static stage-only equation. |
| `daily_discharge_calculated.csv`, `daily_discharge.png` | Daily results when daily input is supplied. |
| `validation_scores.csv` | Outer held-out overall and upper-stage-quartile scores. |
| `validation_folds.csv`, `validation_by_year.csv` | Scores by fold and, when dated, year. |
| `validation_diagnostics.png` | Held-out predictions and residuals. |
| `model_comparison.csv` | Final inner selection results and candidate failures. |
| `measurement_predictions.csv` | Full-fit/held-out predictions, residuals and fold IDs. |
| `cleaned_measurements.csv` | Accepted observations and matched daily features. |
| `rejected_measurements.csv`, `rejected_daily_rows.csv` | Excluded source records and reasons. An empty file can indicate no rejected rows. |
| `model.json` | Model, calibration, settings, input hashes and metadata. |
| `residual_rf.joblib` | Selected forest, when present; keep beside `model.json`. |

Key columns: `WL` (stage), `Q_RatingCurve` (static rating), `Q_Estimate` (selected
prediction), `Q_Lower`/`Q_Upper` (empirical band), and `Q_NestedCV` (held-out
measurement prediction). Use `Q_NestedCV` to evaluate held-out errors.

Check `Stage_extrapolation`, `Interval_status`, `Predictor`, `dWL_available` and
`Date_outside_gauging_period` where present. `Local_observation_count` counts
observations within 5% of the measured stage span; it is not a separate
uncertainty estimate.

### Comparing scores

RMSE and MAE are errors in discharge units; lower is better. Bias uses predicted
minus observed flow. NSE compares squared errors against the observed-mean
baseline; a negative NSE means that baseline is better on the evaluated set.
KGE combines correlation, relative spread and relative mean. Some metrics are
unavailable for constant or otherwise degenerate subsets.

Additive selects using RMSE in `log10(Q)`. Validated's optional RMSLE uses
`log1p(Q / scale)`, with scale based on the evaluated observations' 90th-percentile
discharge. Neither score is directly comparable to RMSE in flow units. A rigorous
method comparison needs the same observations, outer splits, target and metric,
with all selection within each training split. The two default reports alone
do not provide that controlled comparison.

## Python functions and saved models

Ordinary use should stay in the single entry script. For additional analysis,
start Python from the repository folder and make `src` importable:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("src").resolve()))
```

For an advanced validated fit:

```python
from ratingcurve_autofit.validated import Settings, fit_rating_curve

result = fit_rating_curve(
    "input/test/measurements.csv",
    daily_stages="input/test/daily_stage.csv",
    output_folder="output/test/validated",
    date_format="%Y-%m-%d",
    settings=Settings(max_segments=1, folds=5, inner_folds=3, coverage=0.90),
)
print(result)
```

Other `Settings` fields are `shape`, `min_regime`, `threshold`, `selection_metric`,
`complexity_tolerance`, `rf` and `seed`. A fixed threshold requires
`max_segments=2`; selection can still choose one regime. Direct calls accept
custom headers through `date_col`, `stage_col`, `discharge_col`,
`daily_date_col` and `daily_stage_col`.

For additive:

```python
from ratingcurve_autofit.additive import run

result = run(
    "input/test/measurements.csv",
    output_root="output/test/additive",
    daily_stages="input/test/daily_stage.csv",  # Optional; omit to skip daily predictions.
    max_segments=1, bootstrap_samples=0,
    date_format="%Y-%m-%d", stage_unit="m", discharge_unit="m3/s",
)
```

Both return the run folder as a `Path`. These direct calls use paths relative
to the current working directory; they do not infer station folders. This
differs from the root script's anchored station paths. Numerical functions
remain under `src/ratingcurve_autofit/`; output schemas are method-specific.

### Reuse a validated model without refitting

After the import-path setup above, replace the example run name with your own:

```python
import numpy as np
import pandas as pd
from ratingcurve_autofit.validated import load_model, predict_pipeline

run_folder = Path("output/test/validated/measurements_YOUR_TIMESTAMP")
model = load_model(run_folder / "model.json")
stages = np.linspace(model["metadata"]["stage_min"], model["metadata"]["stage_max"], 10)
query = pd.DataFrame({"WL": stages, "dWL": np.nan})
query["Q_Estimate"] = predict_pipeline(model, query)
query.to_csv(run_folder / "reused_predictions.csv", index=False)
```

Missing `dWL` uses the static rating even if a forest was selected. Applying a
forest needs daily changes defined exactly as in training. This short example
does not recreate interval/flag columns. Only load trusted joblib models, because
they deserialize Python objects. Keep inputs in the model's units and datum.

### Reuse an additive model without refitting

```python
import json
import numpy as np
from ratingcurve_autofit.additive import parameter_names, predict_discharge

run_folder = Path("output/test/additive/measurements_YOUR_TIMESTAMP")
saved = json.loads((run_folder / "best_model.json").read_text())
k = saved["best_segments"]
parameters = np.array([saved["parameters"][name] for name in parameter_names(k)])
low, high = saved["observed_stage_range"]
stages = np.linspace(low, high, 10)
q_median = predict_discharge(parameters, stages, k)
print(q_median)
```

This returns median discharge. Stay within measured stages unless a separate
hydraulic assessment supports extrapolation. An additive `best_model.json`
cannot be read by the validated model loader.

## Automated checks

Running the test station produces demonstration results. Automated checks
verify software properties and regressions. These are different uses of test.
From the project folder:

```sh
python -m pip install pytest
python -B -m pytest src/tests -q -p no:cacheprovider
```

`-B` prevents new bytecode caches during checks; `-p no:cacheprovider` avoids a
pytest cache folder. `src/tests/conftest.py` makes the source importable without
package installation. Most tests use isolated temporary input/output folders
and leave station inputs unchanged.

To include all forest checks, also install:

```sh
python -m pip install scikit-learn joblib
```

Without these libraries the optional forest tests are skipped. Checks cover
curve shape/continuity, date-group separation, training-only selection,
nonfinite inputs, repeated dates, daily gaps, saved-model predictions, station
routing, units/dates, and preservation of earlier results. Additive daily checks
also cover median/mean calculation, extrapolation flags, rejected daily rows,
and unchanged fitting/selection/bootstrap when daily input is supplied.

For syntax and undefined-name checks:

```sh
python -m pip install ruff
python -m ruff check --no-cache --select E9,F63,F7,F82 src run_rating_curve.py
```

GitHub's workflow is configured to check Windows and Linux, including Python
3.10 and 3.12. Passing tests does not establish a station's hydraulic validity
or guarantee empirical band coverage.

## Troubleshooting

| Symptom | Action |
|---|---|
| Python not found | Select/install Python 3.10+ or use the prepared interpreter path above. |
| `ModuleNotFoundError` | Install requirements in the interpreter used by your editor/terminal. |
| Missing station measurements | Check `STATION` and put `measurements.csv` directly in `input/<station>/`. |
| Missing daily file | Set only its filename in `DAILY_STAGES`; put it in the selected station folder. |
| Missing/ambiguous headers | Prefer `date,wl,discharge`; rename duplicates and export proper CSV. |
| Too few accepted observations | Check numbers, discharge signs and `DATE_FORMAT`; review cleaning diagnostics. |
| Too few rows in validation groups | Add observations across suitable dates/periods. Same-date rows are not split to manufacture independence. |
| No eligible two-regime model | Review support on both sides in training folds. `MAX_SEGMENTS = 1` explicitly tests a simpler family. |
| Conflicting daily duplicates | Resolve differing stages in the input; automatic averaging is not performed. |
| Blank/missing bounds | Validated omits bands beyond measured stages. Additive daily output has no bounds; its rating table has intervals only when bootstrap fits succeed. |
| No daily output | For either method, set `DAILY_STAGES` to the daily filename, rerun, and open the new run folder. `None` skips daily predictions. |
| Forest not selected | Inspect `model_comparison.csv`: feature coverage may be insufficient or the correction may not improve scores. |
| Long runtime | Multi-regime/term optimization, nested fitting and bootstrap need many fits. For a limited trial use one regime/term and additive bootstrap 0. |
| Cannot write output | Check permissions/free space and close applications locking files. |

For an issue report, record settings, interpreter/library versions, headers,
date format, full error text and expected behavior. Use a small synthetic
reproduction when practical.

## Editing and GitHub

Edit the root script for station settings; edit `src/` for calculation changes.
After code changes, run the checks and a relevant example. Preserve the
median/mean, inner-selection/outer-evaluation, and curve/prediction-interval
distinctions when modifying calculations or reports.

An analysis saves results locally and does not push. To publish an intentional
script edit from the project folder:

```sh
git status
git diff
git add run_rating_curve.py
git commit -m "Update station workflow settings"
git push origin master
```

Stage other edited source/documentation paths explicitly as needed. Station
folders other than the supplied `input/test` and all generated results are
ignored by default. Routine source-code pushes therefore omit local station
data. Ignoring does not untrack data that were previously committed.

## References and credit

The original [MIT license](../LICENSE) is retained. This is an independent
implementation, not affiliated with or endorsed by USACE-RMC, IWR, ERDC-CHL or
BaRatin-tools. Neither method implements Bayesian BaRatin inference or
automatically establishes hydraulic controls. The equations and behavior above
describe this code; references provide background and do not imply reproduction
of another program's complete model or uncertainty calculations.

Original additive-method attribution is retained:

- [RMC-BestFit technical reference](https://github.com/USACE-RMC/RMC-BestFit/blob/main/docs/technical-reference/analysis/rating-curve.md).
- [RMC-BestFit repository](https://github.com/USACE-RMC/RMC-BestFit) and [software page](https://www.rmc.usace.army.mil/Software/RMC-BestFit/).
- [BaRatin engine](https://github.com/BaRatin-tools/BaRatin) and [rating-curve source](https://github.com/BaRatin-tools/BaRatin/blob/main/src/RatingCurve_tools.f90).
- Le Coz, J., Renard, B., Bonnifait, L., Branger, F., and Le Boursicaud, R. (2014). Combining hydraulic knowledge and uncertain gaugings in the estimation of hydrometric rating curves: A Bayesian approach. Journal of Hydrology.
- Rantz, S. E., et al. (1982). Measurement and computation of streamflow, Volume 2: Computation of discharge. USGS Water-Supply Paper 2175.
- Kennedy, E. J. (1984). Discharge ratings at gaging stations. USGS Techniques of Water-Resources Investigations, Book 3, Chapter A10.

Other background and implementation references retained from earlier documentation:

- [USGS rating-curve background](https://thodson-usgs.github.io/ratingcurve/meta/background.html).
- [WMO hydrological monitoring guidance](https://wmo.int/media/magazine-article/5-essential-elements-of-hydrological-monitoring-programme).
- [scikit-learn nested validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- [scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

Dependencies retain their own licenses. Earlier attribution recorded RMC-BestFit
as 0BSD and BaRatin as GPL-3.0; consult their source licenses when reusing code.
Referencing a published method is distinct from copying source. Future copied
or translated implementation should retain its origin and applicable license.
