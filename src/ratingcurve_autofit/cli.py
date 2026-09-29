"""Optional command-line interface. All argument parsing lives here."""
from __future__ import annotations

import argparse

from . import __version__
from .validated import Settings, fit_rating_curve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rating-curve",
        description="Fit stage-discharge curves from CSV measurements.",
        epilog="For an editable Python script, start with run_rating_curve.py in the repository.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    methods = parser.add_subparsers(dest="method")

    additive = methods.add_parser("additive", help="Original additive power laws with bootstrap intervals")
    additive.add_argument("csv", help="CSV containing wl, discharge and optional date")
    additive.add_argument("--out", default="rating_curve_results", help="Parent output folder")
    additive.add_argument("--max-segments", type=int, default=3, choices=(1, 2, 3))
    additive.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples; 0 disables intervals")
    additive.add_argument("--show-plots", action="store_true", help="Display plots as well as saving them")

    defaults = Settings()
    validated = methods.add_parser("validated", help="Smooth rating, nested validation and optional daily estimates")
    validated.add_argument("observations", help="CSV with paired water levels and measured discharge")
    validated.add_argument("--daily", help="Optional daily water-level CSV")
    validated.add_argument("--out", help="Parent output folder; each run creates a unique subfolder")
    for flag in ["date-col", "stage-col", "discharge-col", "daily-date-col", "daily-stage-col", "date-format"]:
        validated.add_argument("--" + flag)
    validated.add_argument("--dayfirst", action="store_true", help="Interpret ambiguous dates as day/month/year")
    validated.add_argument("--shape", choices=["monotone", "convex"], default=defaults.shape)
    validated.add_argument("--max-segments", type=int, choices=[1, 2], default=defaults.max_segments)
    validated.add_argument("--min-regime", type=int, default=defaults.min_regime)
    validated.add_argument("--threshold", type=float, help="Hydraulically justified fixed transition")
    validated.add_argument("--folds", type=int, default=defaults.folds)
    validated.add_argument("--inner-folds", type=int, default=defaults.inner_folds)
    validated.add_argument("--selection-metric", choices=["RMSE", "RMSLE"], default=defaults.selection_metric)
    validated.add_argument("--rf", action="store_true", help="Compare optional residual forests; requires daily stages")
    validated.add_argument("--seed", type=int, default=defaults.seed)
    validated.add_argument("--coverage", type=float, default=defaults.coverage)
    validated.add_argument("--stage-unit", default="m", help="Label only; does not convert units")
    validated.add_argument("--discharge-unit", default="m3/s", help="Label only; does not convert units")

    args = parser.parse_args(argv)
    if args.method is None:
        parser.print_help()
        return
    try:
        if args.method == "additive":
            from .additive import run

            run(args.csv, args.out, args.max_segments, args.bootstrap, args.show_plots)
        else:
            settings = Settings(
                shape=args.shape, max_segments=args.max_segments, min_regime=args.min_regime,
                threshold=args.threshold, folds=args.folds, inner_folds=args.inner_folds,
                selection_metric=args.selection_metric, rf=args.rf, seed=args.seed, coverage=args.coverage,
            )
            fit_rating_curve(
                args.observations, daily_stages=args.daily, output_folder=args.out,
                stage_unit=args.stage_unit, discharge_unit=args.discharge_unit,
                date_col=args.date_col, stage_col=args.stage_col, discharge_col=args.discharge_col,
                daily_date_col=args.daily_date_col, daily_stage_col=args.daily_stage_col,
                date_format=args.date_format, dayfirst=args.dayfirst, settings=settings,
            )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
