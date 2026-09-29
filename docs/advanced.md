# Advanced use

For the normal edit-and-run workflow, use [`run_rating_curve.py`](../run_rating_curve.py)
and the [README](../README.md). This page covers Python imports, alternative
fitting settings and command-line automation.

## Import the fitting function

Install the package from the repository folder:

```sh
python -m pip install -e .
```

Then call the same function used by the starter script:

```python
from ratingcurve_autofit import fit_rating_curve

result = fit_rating_curve(
    "examples/measurements.csv",
    daily_stages="examples/daily_stage.csv",  # Omit when no daily file is needed.
    output_folder="results",
    date_format="%Y-%m-%d",
)
print(result)  # Path to this run's plots, tables and report.
```

Unlike the starter script, paths in your own Python code are relative to the
current working directory. Absolute paths also work. No command-line parser
or configuration file is needed.

## Change fitting choices

The default is the smooth method, comparing one/two regimes with five outer
and three inner validation folds, monotone shape and no random forest. For a
single-regime analysis, pass the existing settings object:

```python
from ratingcurve_autofit import fit_rating_curve
from ratingcurve_autofit.validated import Settings

result = fit_rating_curve(
    "examples/measurements.csv",
    output_folder="results/single_regime",
    settings=Settings(max_segments=1),
)
```

Other choices include `shape="convex"`, `coverage=0.90`, and a hydraulically
justified `threshold`. Read the [smooth method guide](validated.md) before
changing their assumptions. Custom CSV headers can be passed as `stage_col`,
`discharge_col`, `date_col`, `daily_stage_col` and `daily_date_col`.

## Use the original additive method

This method fits one to three additive power laws in log discharge. It uses
positive discharge measurements and distinguishes median from corrected-mean
discharge. It is not interchangeable with the smooth method.

```python
from ratingcurve_autofit.additive import run

result = run(
    "examples/measurements.csv",
    output_root="results/additive",
    max_segments=1,
    bootstrap_samples=0,
)
```

This is a quick example with uncertainty disabled. The method's defaults
compare up to three terms and request 200 bootstrap resamples. See the
[additive guide](additive.md) and [method comparison](methods.md).

## Optional command line

The installed commands remain available for batch work:

```sh
rating-curve validated examples/measurements.csv --daily examples/daily_stage.csv --out results/validated
rating-curve additive examples/measurements.csv --max-segments 1 --bootstrap 0 --out results/additive
```

Use `python -m ratingcurve_autofit` in place of `rating-curve` if needed.
Add `--help` after either method name for its options. All command-line
arguments are defined in `src/ratingcurve_autofit/cli.py`.

For optional residual random forest:

```sh
python -m pip install -e ".[rf]"
rating-curve validated examples/measurements.csv --daily examples/daily_stage.csv --rf --out results/validated_rf
```

The forest is compared during validation; it is not forced into the result.
It needs daily stage changes and does not preserve the static rating's shape
guarantees. See [residual forest details](validated.md#optional-residual-forest).

## Earlier source-checkout commands

The earlier root launchers, import shims, duplicate requirements file and sample
are now in [`legacy/`](../legacy/README.md). Update an old command such as
`python rating_curve_autofit.py ...` to `python legacy/rating_curve_autofit.py ...`,
or use the installed `rating-curve additive ...` command. Update the smooth
launcher path to `legacy/universal_rating_curve.py`.

The new starter explicitly uses the smooth method. Its outputs are different
from the original additive script's outputs. Fitting algorithms and existing
installed command options remain the same. For programmatic smooth fits, use
`fit_rating_curve(...)` instead of constructing parser arguments for the old
internal `validated.run(args)` function.
