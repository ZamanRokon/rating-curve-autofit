# Changelog

## 0.2.0 — Unreleased

### Added

- One installable package and `rating-curve` command with explicit `additive` and `validated` subcommands.
- Smooth one- or two-regime curves with matching discharge and slope, optional convexity, training-only transition selection, and nested validation.
- Validated CSV detection and overrides, rejected-row reports, daily discharge calculation, and extrapolation flags.
- Optional daily-stage residual random forest, disabled by default and selected inside the validation workflow.
- Saved validated models, empirical error bands, and complete-pipeline validation reports.
- Synthetic example inputs, contributor guidance, method comparison, package tests, and continuous integration.

### Fixed

- Additive CV keeps same-date rows together, fits each fold from its training data, and requires complete held-out coverage for a usable selection score.
- Nonfinite additive input values are excluded, and timestamped result directories are created without overwriting an existing run.
- Run provenance records package/runtime versions and input hashes.

### Preserved

- The existing additive method: one to three positive power-law terms, log-discharge fitting, support checks, candidate cross-validation, simplicity preference, BIC fallback, and bootstrap intervals.
- The original `rating_curve_autofit.py` command for source-checkout use.
- The repository's MIT license and source attribution.

### Migration notes

- The existing root launcher continues to run the additive method. Use `rating-curve additive` for its installed command, or `rating-curve validated` for the new method.
- Import new code from `ratingcurve_autofit.additive`, `ratingcurve_autofit.core`, or `ratingcurve_autofit.validated`.
- The backends retain distinct model JSON and output schemas. An additive `best_model.json` is not a validated `model.json` and cannot be loaded as one.
- Neither command automatically chooses between the two backends; choose and record the method explicitly.
