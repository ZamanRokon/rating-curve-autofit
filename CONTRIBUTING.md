# Contributing

Changes should keep the two rating methods explicit, reproducible, and usable without private station data.

## Set up

Use Python 3.10 or later, clone the repository, and activate a virtual environment as described in the [README](README.md). Install development and optional RF dependencies:

```sh
python -m pip install -e ".[dev,rf]"
python -m pytest
```

The Python import package is `ratingcurve_autofit`; the distribution is `rating-curve-autofit`. Implementation lives under `src/ratingcurve_autofit`. Root scripts provide source-checkout compatibility, and `examples/` contains synthetic inputs.

## Make a change

1. Create a branch and keep changes focused on a stated problem.
2. Update method documentation when equations, defaults, validation, output fields, or input acceptance change.
3. Add a meaningful test for changed numerical behavior or an identified regression. Prefer invariants such as continuity, monotonicity, finite outputs, group isolation, and save/load equivalence over optimizer-specific parameter snapshots.
4. Run the tests and the relevant CLI workflow. Use a temporary result directory or an ignored `results/` subfolder.
5. Open a pull request with the behavior change, its rationale, and the checks performed. Explain any migration or result-interpretation change.

For a quick additive smoke run:

```sh
rating-curve additive examples/measurements.csv --max-segments 1 --bootstrap 0 --out results/smoke_additive
```

For the validated workflow:

```sh
rating-curve validated examples/measurements.csv --daily examples/daily_stage.csv --max-segments 1 --out results/smoke_validated
```

A smoke run demonstrates that the workflow completes; it is not a station accuracy benchmark. Full bootstrap and nested fits may take longer than unit tests.

## Scientific changes

Preserve the meaning of the two backends' outputs. In particular, distinguish additive median discharge from corrected mean discharge, inner model-selection scores from outer performance estimates, and uncertainty bands from independently established coverage.

Any feature derived from daily stages must have the same definition in training and prediction. Data preprocessing and parameter selection that learn from measurements belong inside the relevant training folds. Tests should catch leakage when validation behavior changes.

Add references for new methods and identify any reused implementation in [SOURCES_AND_CREDIT.md](SOURCES_AND_CREDIT.md). Keep applicable licenses and attribution with copied material. Preserve the repository's existing [LICENSE](LICENSE).

## Data and issue reports

Do not commit private station measurements, local paths, credentials, generated model files, or result folders. Share a small synthetic or appropriately licensed reproduction when reporting a bug. Include the command, Python/package versions, column names and date convention, relevant error text, and expected behavior.

For proposed changes to a station rating, describe the measurement coverage and hydraulic evidence separately from software correctness. Code review cannot establish a rating's validity for every hydraulic setting.
