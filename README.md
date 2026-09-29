# Rating Curve Autofit

Fit stage–discharge rating curves for individual stations using either the
**smooth/validated** or **additive** method. One Python script produces plots,
equations, rating tables, daily discharge estimates and diagnostic reports.

1. Put your station's `measurements.csv` in `input/<station>/`.
2. Open `run_rating_curve.py` and set `STATION` and `METHOD`.
3. Install the libraries and run:

```sh
python -m pip install -r requirements.txt
python run_rating_curve.py
```

Results go to `output/<station>/<method>/<run>/`. Previous runs are kept.
Start with `STATION = "test"` to use the supplied synthetic data. For daily
discharge with **either method**, put a `date,wl` CSV in that station's folder
and set `DAILY_STAGES = "daily_stage.csv"` (or your filename). Each run adds
`daily_discharge_calculated.csv` and `daily_discharge.png`. Both methods fit
the paired measurements; the daily stages supply prediction inputs.

| Folder | Contents |
|---|---|
| `docs/` | [Technical manual](docs/technical_manual.md): setup, both methods, tests, outputs and troubleshooting. |
| `input/test/` | Synthetic CSVs. Add other station folders beside `test/`. |
| `output/test/` | Example results; other stations get their own output folders. |
| `src/` | Calculation code and automated checks. |

Requires Python 3.10+. Unit labels do not convert values. Review validation,
measurement coverage and extrapolation before adopting a station rating.

[MIT license](LICENSE). Sources and original attribution are in the
[manual](docs/technical_manual.md#references-and-credit).
