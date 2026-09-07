# Synthetic demonstration data

These files are generated examples, not measurements from a real station. They
contain no field observations.

- `daily_stage.csv`: 90 consecutive daily stages, starting 1 January 2020.
- `measurements.csv`: 60 dates sampled without replacement from days 1–89,
  with paired stage and synthetic discharge.

For integer day `t = 0, ..., 89`, stage in metres is

```text
h(t) = 4.5 + 1.7 sin(2 pi t / 23) + 0.4 cos(2 pi t / 9)
```

Discharge in cubic metres per second is generated from

```text
Q(t) = 6 (h(t) - 1)^1.7 exp(epsilon)
epsilon ~ Normal(0, 0.06^2)
```

The NumPy `default_rng` seed is **20260907**. Date selection uses the generator
before discharge noise is sampled. CSV numeric values are rounded to eight
decimal places. The paired stages are taken from the same daily series so the
validated workflow can calculate consistent daily stage changes.

Regenerate both files from the repository root:

```bash
python examples/generate_examples.py
```

After installing the package, run either workflow:

```bash
python -m ratingcurve_autofit additive examples/measurements.csv --max-segments 1 --bootstrap 0
python -m ratingcurve_autofit validated examples/measurements.csv --daily examples/daily_stage.csv --max-segments 1 --folds 3 --inner-folds 3
```

This simple single-control relationship demonstrates the input format and
software behavior. It is not a realistic hydrological benchmark or evidence of
accuracy on field data.
