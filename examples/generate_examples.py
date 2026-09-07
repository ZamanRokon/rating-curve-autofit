"""Regenerate the entirely synthetic, reproducible demonstration CSV files."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(20260907)
    day = np.arange(90)
    dates = pd.date_range("2020-01-01", periods=len(day))
    stage = 4.5 + 1.7 * np.sin(2 * np.pi * day / 23) + 0.4 * np.cos(2 * np.pi * day / 9)
    selected = np.sort(rng.choice(np.arange(1, 90), size=60, replace=False))
    measured_q = 6.0 * (stage[selected] - 1.0)**1.7 * np.exp(rng.normal(0.0, 0.06, len(selected)))
    pd.DataFrame({"date": dates, "wl": stage}).to_csv(
        root / "daily_stage.csv", index=False, float_format="%.8f"
    )
    pd.DataFrame({"date": dates[selected], "wl": stage[selected], "discharge": measured_q}).to_csv(
        root / "measurements.csv", index=False, float_format="%.8f"
    )


if __name__ == "__main__":
    main()
