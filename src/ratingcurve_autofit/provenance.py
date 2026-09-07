"""Reproducibility metadata without credentials or machine-specific settings."""
from importlib.metadata import PackageNotFoundError, version
import math
import platform

from . import __version__


def runtime_versions(include_rf=False):
    versions = {"rating-curve-autofit": __version__, "python": platform.python_version()}
    dependencies = ["numpy", "pandas", "scipy", "matplotlib"]
    if include_rf:
        dependencies.extend(["scikit-learn", "joblib"])
    for name in dependencies:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def json_safe(value):
    """Use JSON null for unavailable numeric diagnostics instead of NaN/Infinity."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
