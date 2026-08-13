# Rating Curve Autofit

A simple standalone Python script for fitting stage-discharge rating curves from a CSV file containing date, water level, and discharge observations.

The script reads paired observations, fits 1-, 2-, and 3-segment power-law rating curves, selects the best model using BIC, and writes tables, plots, model parameters, equations, and a short report into an output folder.

This project is an independent educational/research automation script. It is not affiliated with or endorsed by USACE-RMC, IWR, ERDC-CHL, or BaRatin-tools.

## Input CSV

Default column names:

```csv
date,wl,discharge
2025-01-01,1.00,12.4
2025-01-02,1.20,18.0
```

You can change the column names at the top of `rating_curve_autofit.py`:

```python
DATE_COLUMN = "date"
STAGE_COLUMN = "wl"
DISCHARGE_COLUMN = "discharge"
```

## Run

Install requirements:

```powershell
pip install -r requirements.txt
```

Run with the sample file:

```powershell
python rating_curve_autofit.py sample_rating_data.csv
```

Run with your own CSV:

```powershell
python rating_curve_autofit.py your_data.csv
```

Optional output folder:

```powershell
python rating_curve_autofit.py your_data.csv --out rating_curve_results
```

## Model

The script uses a power-law stage-discharge relationship.

Single segment:

```text
Q = alpha1 * (h - h1)^beta1
```

Two segment addition mode:

```text
Q = alpha1 * (h - h1)^beta1 + alpha2 * (h - h2)^beta2 * I(h > h2)
```

Three segment addition mode:

```text
Q = alpha1 * (h - h1)^beta1 + alpha2 * (h - h2)^beta2 * I(h > h2) + alpha3 * (h - h3)^beta3 * I(h > h3)
```

Errors are fit in log10 discharge space.

## Outputs

Each run creates a timestamped output folder with:

- `cleaned_data.csv`
- `best_parameters.csv`
- `best_model.json`
- `equation.txt`
- `model_comparison.csv`
- `fitted_values_and_residuals.csv`
- `rating_table.csv`
- `report.md`
- `plots/rating_curve.png`
- `plots/rating_curve_log_scale.png`
- `plots/residuals_vs_stage.png`
- `plots/residual_histogram.png`
- `plots/residual_qq_plot.png`
- `plots/model_comparison.png`

## Main Sources And Credit

This script is informed by published rating-curve methods and public documentation:

- RMC-BestFit GitHub repository: https://github.com/USACE-RMC/RMC-BestFit
- RMC-BestFit rating-curve technical reference: https://github.com/USACE-RMC/RMC-BestFit/blob/main/docs/technical-reference/analysis/rating-curve.md
- RMC-BestFit software page: https://www.rmc.usace.army.mil/Software/RMC-BestFit/
- BaRatin computational engine: https://github.com/BaRatin-tools/BaRatin
- BaRatin rating-curve Fortran source: https://github.com/BaRatin-tools/BaRatin/blob/main/src/RatingCurve_tools.f90

See `SOURCES_AND_CREDIT.md` for more notes.

## License Note

This repository is licensed under the MIT License.

RMC-BestFit is published under the Zero-Clause BSD license. BaRatin is published under GPL-3.0. If this script is changed to directly copy or translate BaRatin GPL-3.0 source code, the derived work should be distributed under GPL-3.0-compatible terms.
