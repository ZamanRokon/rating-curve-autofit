# Additive power-law method

The `additive` backend preserves the repository's empirical one- to three-term rating family and its log-discharge fitting objective. It includes data-support checks, cross-validation selection with a simplicity preference, and optional bootstrap uncertainty.

## Run

From an installed checkout:

```sh
rating-curve additive examples/measurements.csv --out results/additive
```

Equivalent commands are `python -m ratingcurve_autofit additive ...` and, from the repository root, `python rating_curve_autofit.py ...`.

A smaller run without bootstrap:

```sh
rating-curve additive examples/measurements.csv --max-segments 1 --bootstrap 0 --out results/additive
```

Inputs are read without modification. The command prints the timestamped result directory. The default output root is `rating_curve_results` relative to the working directory.

## Input

The default comma-separated schema uses exact `wl` and `discharge` column names. `date` is optional:

```csv
date,wl,discharge
2020-01-01,1.20,8.40
2020-01-15,1.55,15.70
2020-02-01,2.10,33.50
```

This is a schema illustration. At least 20 valid paired observations and four distinct water levels are required for the one-term candidate. Two and three terms require at least 40 and 60 observations, respectively. Later activated terms need at least 10 observations above each activation stage; these checks do not prove that distinct hydraulic controls have been identified.

The log objective needs **positive discharge**. Rows with unusable numeric stage/discharge or nonpositive discharge are excluded; data quality counts are saved. Repeated observations are retained. ISO dates (`YYYY-MM-DD`) are recommended to avoid ambiguous date parsing. Incomplete dates make date-blocked validation unavailable.

For custom headers, rename columns to the schema above before fitting. For automated input detection, explicit column/date overrides, rejected-row reports, zero-discharge retention, or daily-series matching, use the [validated backend](validated.md).

## Options

| Option | Default | Purpose |
|---|---|---|
| `--out PATH` | `rating_curve_results` | Parent directory for timestamped results. |
| `--max-segments {1,2,3}` | `3` | Maximum additive term count; data-support rules may reduce it. |
| `--bootstrap N` | `200` | Number of bootstrap resamples. `0` disables uncertainty calculations. |
| `--show-plots` | Off | Display figures as well as saving them. |

The default validation uses five folds, a 3% simplicity tolerance, and 90% bootstrap intervals. See the backend's settings and saved report when reproducing an analysis. `rating-curve additive --help` lists the supported CLI options.

## Fit and selection

Each term has a positive coefficient and exponent. The first zero-flow stage is constrained below the minimum measured stage, and additional activation stages are ordered. The fitter searches globally and refines locally to minimize squared `log10` discharge residuals.

Eligible candidate curves are evaluated with date-blocked CV when complete dates are available, otherwise stage-stratified CV. Same-date observations remain together in dated folds. Each fold is fitted from its training data. The method prefers the simplest candidate within 3% of the smallest CV log RMSE. BIC is the fallback when usable candidate CV is unavailable; AIC, AICc, and BIC are also exported as diagnostics.

These candidate scores also drive model selection and are not a nested estimate of the complete selection procedure. They should not be reported as independent test performance. Check the reported validation strategy and evaluated sample count before interpreting results.

The rating is nondecreasing but does not constrain transition slopes or curvature. Activation exponents below one can create steep slope behavior; inspect the plotted curve and fitted parameters. See the [equation comparison](methods.md) for the mathematical distinction from the smooth backend.

## Median, mean, and bootstrap bands

The fitted equation returns a conditional median under the lognormal error model. The table also contains a mean discharge obtained using the fitted constant log-error variance. Choose the output consistent with your analysis and record that choice.

The bootstrap resamples observation pairs and refits the selected term count. It produces pointwise percentile intervals for the fitted curve and for predictions after adding simulated lognormal residual noise. It does not propagate term-count selection uncertainty or preserve temporal dependence. Parameter intervals use the same successful fits.

`--bootstrap 0` omits these intervals. With bootstrap enabled, check how many refits succeeded; few successful draws or nonconstant log-error spread weaken the interval interpretation.

## Outputs

| File | Contents |
|---|---|
| `report.md` | Selection rule, scores, assumptions, warnings, and uncertainty summary. |
| `input_quality.json` | Row counts, exclusions, repeated observations, and stage/discharge ranges. |
| `run_metadata.json` | Input hash, package/runtime versions, and fitting settings. |
| `cleaned_data.csv` | Accepted observations. |
| `best_model.json` | Selected term count, named parameters, median equation, range, error scale, and fit metadata. |
| `best_parameters.csv` | Parameter estimates and available bootstrap bounds. |
| `equation.txt` | Median rating equation. |
| `model_comparison.csv` | Candidate support, information criteria, selection CV scores, and fit diagnostics. |
| `fitted_values_and_residuals.csv` | Median and corrected-mean predictions, residuals, and potential-outlier flags. |
| `rating_table.csv` | 100 stages within the measured range, median/mean discharge, and available intervals. |
| `plots/` | Linear/log rating curves, residual plots, model comparison, and available temporal diagnostics. |

Outlier flags are diagnostics; the fitter does not automatically remove those observations. Investigate measurement conditions before excluding data.

## Python use

The same workflow is available programmatically:

```python
from ratingcurve_autofit.additive import run

result_dir = run(
    "examples/measurements.csv",
    output_root="results/additive",
    max_segments=1,
    bootstrap_samples=0,
)
print(result_dir)
```

To evaluate parameters from a saved `best_model.json`:

```python
import json
from pathlib import Path

import numpy as np
from ratingcurve_autofit.additive import parameter_names, predict_discharge

saved = json.loads(Path("results/your_additive_run/best_model.json").read_text())
k = saved["best_segments"]
params = np.array([saved["parameters"][key] for key in parameter_names(k)])
low, high = saved["observed_stage_range"]
stage = np.linspace(low, high, 3)  # Replace with requested stages in the same units.
if np.any((stage < low) | (stage > high)):
    raise ValueError("Requested stages exceed the observed rating range.")
q_median = predict_discharge(params, stage, k)
print(q_median)
```

Use stage values in the model's units and datum. The numerical prediction function can evaluate other stages, but the CLI deliberately confines its rating table to observations' range.
