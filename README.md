# Rating Curve Autofit

Turn measured water levels and discharge into a rating curve, a lookup table,
and a short report. You only need to edit a few settings in
[`run_rating_curve.py`](run_rating_curve.py), then run it.

## 1. Install once

Use **Python 3.10 or later**. Download this repository using **Code → Download
ZIP** on GitHub and extract it, or clone it with Git. Open a terminal in the
extracted repository folder and install the four required libraries:

```sh
python -m pip install -r requirements.txt
```

If you use Spyder or VS Code, install into the same Python environment your
editor uses. See [setup help](docs/setup.md) if Python or a library is not found.

## 2. Try the example

Run the script from that folder:

```sh
python run_rating_curve.py
```

You can also open `run_rating_curve.py` in Spyder or VS Code and press **Run**.
Keep it in the repository folder. The supplied data are synthetic, so you can
try the complete workflow before preparing station data. Fitting and validation
can take several minutes; progress is printed for each validation period.

The script prints the location of a new folder under `results/`. Open:

| File | What to use it for |
|---|---|
| `report.md` | Read the fitted equation, validation results and data notes. |
| `rating_curve.png` | Check the fitted curve against your measurements. |
| `rating_table.csv` | Look up discharge (`Q_RatingCurve`) for water level (`WL`). |
| `equation.txt` | Copy the fitted stage–discharge equation. |

Each run gets its own folder. Existing results and input files are preserved.
Additional diagnostic files are explained in the [method guide](docs/validated.md#outputs-and-uncertainty).

## 3. Use your measurements

Save a comma-separated CSV with these column names:

```csv
date,wl,discharge
2020-01-01,1.20,8.40
2020-01-15,1.55,15.70
2020-02-01,2.10,33.50
```

Here `wl` is water level and `discharge` is measured flow. Dates are recommended
but may be omitted. The three rows above show the format only: provide at least
**20 usable measurements**, with varying discharge and at least five distinct
water levels. Validation may require more data, particularly when measurements
are concentrated on a few dates. Use one station and one stage datum.

Open `run_rating_curve.py` and edit the settings at the top:

```python
MEASUREMENTS = "my_measurements.csv"
DAILY_STAGES = None
OUTPUT_FOLDER = "results"
DATE_FORMAT = "%Y-%m-%d"
STAGE_UNIT = "m"
DISCHARGE_UNIT = "m3/s"
```

Place `my_measurements.csv` beside the script, or use a full path such as
`"C:/Hydrology/Station_A/measurements.csv"`. Relative paths start from the script's
folder, including when you run it from an editor. For dates like `31/01/2020`,
change `DATE_FORMAT` to `"%d/%m/%Y"`. Units are labels; values are **not converted**.
Use a decimal point for numbers. Then run the same script again.

## Optional: calculate daily discharge

Prepare a second CSV containing daily water levels:

```csv
date,wl
2020-01-01,1.20
2020-01-02,1.25
2020-01-03,1.18
```

Set `DAILY_STAGES = "my_daily_stages.csv"` and run again. To try the supplied
example, use `DAILY_STAGES = "examples/daily_stage.csv"`. Both files must use
the same date format and stage datum. The results now also include
`daily_discharge_calculated.csv` and `daily_discharge.png`; `Q_Estimate` contains
the discharge estimates. Inspect `Stage_extrapolation` before using them.

## What the script fits

The starter uses the **smooth rating method**, named `validated` in the package.
It compares one- and two-regime power-law curves, checks predictions on omitted
periods, and saves an empirical error band. Random forest is off. The original
**additive method** remains available through the [advanced guide](docs/advanced.md).
The two methods have different equations and uncertainty calculations.

Review the plot, rejected observations and validation results before adopting
a station rating. Estimates outside the measured stage range are flagged and
have no error bounds. The bands do not guarantee coverage, and the method does
not resolve backwater, hysteresis or changing controls. Discharge calculated
from daily mean stage is not necessarily daily mean discharge.

## Where things live

```text
run_rating_curve.py       Start here: edit settings and run
requirements.txt         Libraries to install
examples/                Synthetic measurements and daily stages
docs/                    Setup, advanced use and method explanations
src/ratingcurve_autofit/  Fitting, input loading and output code
tests/                   Checks for the calculations and workflows
legacy/                  Earlier launchers and sample files
results/                 Your generated output (created when you run)
```

For more detail: [setup help](docs/setup.md), [advanced Python and command-line
use](docs/advanced.md), [method comparison](docs/methods.md),
[contributing](CONTRIBUTING.md), and [changes](CHANGELOG.md).

## License and credit

[MIT license](LICENSE). Independent project, not affiliated with or endorsed by
USACE-RMC, IWR, ERDC-CHL or BaRatin-tools. Neither method is a Bayesian BaRatin
implementation. See [sources and credit](SOURCES_AND_CREDIT.md).
