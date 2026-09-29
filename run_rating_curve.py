"""Choose a station and method below, then run this file. See docs/technical_manual.md."""

# ---- CHOOSE YOUR STATION AND METHOD ----
STATION = "test"                  # Folder name under input/, e.g. "Feni_Ramgarh"
METHOD = "validated"              # "validated" (smooth) or "additive"
DAILY_STAGES = None                # Both methods: "daily_stage.csv", or None

# Each station folder must contain measurements.csv with date, wl, discharge.
DATE_FORMAT = "%Y-%m-%d"           # YYYY-MM-DD; use "%d/%m/%Y" for DD/MM/YYYY
STAGE_UNIT = "m"                   # Labels only; values are not converted.
DISCHARGE_UNIT = "m3/s"

# ---- OPTIONAL FITTING CHOICES ----
MAX_SEGMENTS = None                # None: validated compares 1-2, additive 1-3.
BOOTSTRAP_SAMPLES = 200            # Additive only; 0 gives a quick run without intervals.
SHAPE = "monotone"                 # Validated only: "monotone" or "convex".
USE_RANDOM_FOREST = False          # Validated only; needs daily stages and scikit-learn.

# ---- RUN THE ANALYSIS ----
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))


def main():
    # Use one station folder, even when an editor runs from another directory.
    if not STATION or STATION in {".", ".."} or any(c in STATION for c in "/\\:"):
        raise ValueError("STATION must be one folder name under input/, such as 'Feni_Ramgarh'.")
    if METHOD not in {"validated", "additive"}:
        raise ValueError("METHOD must be 'validated' or 'additive'.")
    station_folder = HERE / "input" / STATION
    measurements = station_folder / "measurements.csv"
    output_folder = HERE / "output" / STATION / METHOD
    if not measurements.is_file():
        raise FileNotFoundError(f"Put measurements.csv in this station folder: {station_folder}")
    daily = None
    if DAILY_STAGES is not None:
        if not DAILY_STAGES or DAILY_STAGES in {".", ".."} or any(c in DAILY_STAGES for c in "/\\:"):
            raise ValueError("DAILY_STAGES must be a filename in the selected station folder, or None.")
        daily = station_folder / DAILY_STAGES
        if not daily.is_file():
            raise FileNotFoundError(f"Daily stage file not found: {daily}")
    print(f"Station: {STATION} | Method: {METHOD}", flush=True)

    if METHOD == "validated":
        from ratingcurve_autofit.validated import Settings, fit_rating_curve

        result = fit_rating_curve(
            measurements, daily_stages=daily, output_folder=output_folder,
            date_format=DATE_FORMAT, stage_unit=STAGE_UNIT, discharge_unit=DISCHARGE_UNIT,
            settings=Settings(
                max_segments=2 if MAX_SEGMENTS is None else MAX_SEGMENTS,
                shape=SHAPE, rf=USE_RANDOM_FOREST,
            ),
        )
        plot = result / "rating_curve.png"
    else:
        from ratingcurve_autofit.additive import run

        result = run(
            measurements, output_root=output_folder, daily_stages=daily,
            max_segments=3 if MAX_SEGMENTS is None else MAX_SEGMENTS,
            bootstrap_samples=BOOTSTRAP_SAMPLES, date_format=DATE_FORMAT,
            stage_unit=STAGE_UNIT, discharge_unit=DISCHARGE_UNIT,
        )
        plot = result / "plots" / "rating_curve.png"

    print(f"\nRead the report: {result / 'report.md'}")
    print(f"Rating plot: {plot}")
    print(f"Rating table: {result / 'rating_table.csv'}")
    if daily is not None:
        print(f"Daily discharge: {result / 'daily_discharge_calculated.csv'}")
        print(f"Daily plot: {result / 'daily_discharge.png'}")
    return result


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Could not complete the rating curve: {exc}") from None
    except ImportError as exc:
        raise SystemExit(
            f"A required Python library could not be imported: {exc}\n"
            "Install the libraries with: python -m pip install -r requirements.txt"
        ) from None
