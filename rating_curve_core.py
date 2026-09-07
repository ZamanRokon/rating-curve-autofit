"""Compatibility launcher for ratingcurve_autofit.core."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ratingcurve_autofit import core as _implementation

if __name__ == "__main__":
    raise SystemExit("Use rating-curve additive or rating-curve validated.")
else:
    sys.modules[__name__] = _implementation
