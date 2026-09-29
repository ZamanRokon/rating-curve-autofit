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

After installing the libraries, try the starter from the repository root:

```bash
python run_rating_curve.py
```

Set `DAILY_STAGES = "examples/daily_stage.csv"` in the script to also calculate
daily discharge. See [advanced use](../docs/advanced.md) for the additive method.

This simple single-control relationship demonstrates the input format and
software behavior. It is not a realistic hydrological benchmark or evidence of
accuracy on field data.
