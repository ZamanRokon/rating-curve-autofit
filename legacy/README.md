# Earlier source-checkout files

Start new analyses with [`run_rating_curve.py`](../run_rating_curve.py).
These older files were moved here to keep one starting script at the top level.
Their fitting methods have not changed.

From the repository folder, old additive commands now use:

```sh
python legacy/rating_curve_autofit.py examples/measurements.csv --max-segments 1 --bootstrap 0
```

The smooth launcher is `legacy/universal_rating_curve.py`. The old
`rating_curve_core.py` and `rating_curve_io.py` import shims are also here;
update Python imports to `ratingcurve_autofit.core` and `ratingcurve_autofit.io`
after installing with `python -m pip install -e .`.

Installed `rating-curve additive ...` and `rating-curve validated ...` commands
continue to work. The old sample and requirements file are retained here;
new users should use `examples/measurements.csv` and the root `requirements.txt`.
