"""Edit the settings below, then run this file in Python, Spyder or VS Code.

First install the libraries: python -m pip install -r requirements.txt
Then run: python run_rating_curve.py
Uses the smooth rating method with nested validation; see docs/validated.md.
"""

# ---- EDIT THESE SETTINGS ----
MEASUREMENTS = "examples/measurements.csv"  # CSV columns: date, wl, discharge
DAILY_STAGES = None  # Optional: "examples/daily_stage.csv" (date, wl)
OUTPUT_FOLDER = "results"
DATE_FORMAT = "%Y-%m-%d"  # YYYY-MM-DD; for DD/MM/YYYY use "%d/%m/%Y"
STAGE_UNIT = "m"  # Labels only; the program does not convert units.
DISCHARGE_UNIT = "m3/s"

# ---- RUN THE ANALYSIS ----
from pathlib import Path
import sys

# Relative paths start beside this script, even when an editor runs it elsewhere.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))


def main():
    from ratingcurve_autofit import fit_rating_curve

    result = fit_rating_curve(
        HERE / Path(MEASUREMENTS).expanduser(),
        daily_stages=HERE / Path(DAILY_STAGES).expanduser() if DAILY_STAGES is not None else None,
        output_folder=HERE / Path(OUTPUT_FOLDER).expanduser(),
        date_format=DATE_FORMAT,
        stage_unit=STAGE_UNIT,
        discharge_unit=DISCHARGE_UNIT,
    )
    print(f"\nStart with the report: {result / 'report.md'}")
    print(f"Rating plot: {result / 'rating_curve.png'}")
    print(f"Rating table: {result / 'rating_table.csv'}")
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
