# Rating Curve Autofit

Fit stage–discharge rating curves from CSV measurements, compare candidate curves, and export equations, diagnostics, uncertainty estimates, and rating tables. Two methods live in one Python package so their assumptions and results remain explicit.

| Method | What it does | Choose it when |
|---|---|---|
| **`additive`** | Fits one to three additive power-law terms in log discharge; selects with cross-validation and estimates bootstrap intervals. | You want the repository's original empirical method, positive-flow fitting, and separate median/mean rating tables. |
| **`validated`** | Fits one or two regimes with matching value and slope, evaluates the full selection procedure with nested validation, and optionally predicts a daily series. | You want daily discharge estimates, explicit shape constraints, zero-flow handling, and complete-pipeline validation. |

Both methods are empirical tools for one station and a consistent stage datum. They do not automatically establish hydraulic controls or resolve backwater, reversing flow, changing controls, or hysteresis. The `validated` name describes its validation workflow; it is not a certification of any station's rating. See the [method comparison](docs/methods.md).

## Install

Requires **Python 3.10 or later**. From a terminal:

```sh
git clone https://github.com/ZamanRokon/rating-curve-autofit.git
cd rating-curve-autofit
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Then install from the checkout:

```sh
python -m pip install -e .
```

If your shell cannot find `rating-curve`, use `python -m ratingcurve_autofit` with the same arguments. The package is installed from this repository; these instructions do not assume a PyPI release.

## Try both methods

The included [measurement](examples/measurements.csv) and [daily stage](examples/daily_stage.csv) files are **synthetic**, provided to exercise the software. They are not field observations or evidence of accuracy at a real station.

Start with a smaller additive run:

```sh
rating-curve additive examples/measurements.csv --max-segments 1 --bootstrap 0 --out results/additive
```

Compare up to three additive terms and request the default 200 bootstrap resamples:

```sh
rating-curve additive examples/measurements.csv --out results/additive
```

Fit the smooth method and calculate discharge for daily stages:

```sh
rating-curve validated examples/measurements.csv --daily examples/daily_stage.csv --date-format "%Y-%m-%d" --out results/validated
```

To require upward curvature, add `--shape convex` to the `validated` command. Convexity allows a straight branch when its exponent is one. The default `monotone` constraint allows either curvature while keeping the stage-only rating nondecreasing.

Each run writes a timestamped subfolder and prints its location. Read that run's `report.md` first. Bootstrap and nested fits can take several minutes; the first command disables bootstrap for a quick check, so it does not produce uncertainty intervals.

## Use your data

A common CSV format works with both methods:

```csv
date,wl,discharge
2020-01-01,1.20,8.40
2020-01-15,1.55,15.70
2020-02-01,2.10,33.50
```

This illustrates the schema only: a run needs **at least 20 usable observations**, and more may be necessary for validation and multiple regimes. Dates are recommended. Use one set of stage and discharge units throughout; unit options change labels and do not convert numbers.

For daily prediction with `validated`, supply a second file:

```csv
date,wl
2020-01-01,1.20
2020-01-02,1.25
2020-01-03,1.18
```

Replace the example paths in the commands with your files. The `validated` loader recognizes common header aliases and provides explicit column and date-format overrides. The additive workflow uses the `wl`, `discharge`, and optional `date` schema above.

| Guide | Contents |
|---|---|
| [Method comparison](docs/methods.md) | Equations, selection, shape, uncertainty, and fair comparisons. |
| [Additive method](docs/additive.md) | Input requirements, options, bootstrap outputs, and Python use. |
| [Validated method](docs/validated.md) | Input mapping, daily features, nested validation, optional RF, and saved models. |

## Optional random forest

The `validated` method uses the stage-only curve by default. To let its inner validation compare random-forest residual corrections:

```sh
python -m pip install -e ".[rf]"
rating-curve validated examples/measurements.csv --daily examples/daily_stage.csv --rf --out results/validated_rf
```

`--rf` requires a daily stage file and sufficient usable daily stage changes in the training folds. It does not force selection of the forest. A selected correction varies with daily stage change and does not inherit the static curve's monotonicity or convexity guarantee.

## Compatibility and development

The original source-checkout command remains available:

```sh
python rating_curve_autofit.py examples/measurements.csv --max-segments 1 --bootstrap 0
```

The smooth method also has a source launcher, `universal_rating_curve.py`. Installation is recommended for imports and the unified command. New Python code should import from `ratingcurve_autofit`, rather than rely on the old standalone module layout.

For development:

```sh
python -m pip install -e ".[dev,rf]"
python -m pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [CHANGELOG.md](CHANGELOG.md) for changes. Keep station data and generated result folders outside commits; the public examples are synthetic.

## License and credit

The repository retains its [MIT license](LICENSE) and the original additive method's attribution. It is an independent project, not affiliated with or endorsed by USACE-RMC, IWR, ERDC-CHL, or BaRatin-tools. Published hydraulic methods inform the work; neither backend is a Bayesian BaRatin implementation. References and implementation provenance are recorded in [SOURCES_AND_CREDIT.md](SOURCES_AND_CREDIT.md).
