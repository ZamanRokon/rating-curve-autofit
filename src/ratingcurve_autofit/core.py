"""Portable, shape-constrained stage-discharge rating curves.

The two-regime curve has a common zero-flow stage and an anchored high-flow
branch. Its value and first derivative agree exactly at the transition. Both
branches are monotone; ``shape='convex'`` additionally enforces upward curvature.
All fitting takes place in normalized coordinates so changing stage datum or
measurement units does not change the fitted relationship.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares


def _evaluate_normalized(
    parameters: np.ndarray,
    stage: np.ndarray,
    transition: float | None,
) -> np.ndarray:
    """Evaluate nonnegative discharge in normalized coordinates."""
    log_a, head_shift, low_exponent = parameters[:3]
    head = np.maximum(stage + head_shift, 0.0)
    answer = np.exp(log_a) * head**low_exponent
    if transition is not None:
        high_exponent = parameters[3]
        high = stage > transition
        transition_head = transition + head_shift
        transition_q = np.exp(log_a) * transition_head**low_exponent
        # expm1/log1p preserve precision immediately above the transition.
        growth = np.expm1(
            high_exponent * np.log1p((stage[high] - transition) / transition_head)
        )
        answer[high] = transition_q * (1.0 + low_exponent / high_exponent * growth)
    return answer


def _model_parameters(model: dict[str, Any]) -> np.ndarray:
    parameters = model["params"]
    result = [parameters["log_a"], parameters["head_shift"], parameters["low_exponent"]]
    if model["segments"] == 2:
        result.append(parameters["high_exponent"])
    return np.asarray(result, dtype=float)


def predict_curve(model: dict[str, Any], h: Any) -> np.ndarray:
    """Predict discharge, preserving shape; missing/nonfinite stages yield NaN.

    Stages at or below the estimated zero-flow stage yield zero. Extrapolation
    above the calibration range is mathematical, not hydraulic validation.
    """
    stage = np.asarray(h, dtype=float)
    flattened = stage.reshape(-1)
    output = np.full(flattened.shape, np.nan, dtype=float)
    usable = np.isfinite(flattened)
    normalized = (flattened[usable] - model["stage_min"]) / model["stage_scale"]
    transition = None
    if model["segments"] == 2:
        transition = (model["threshold"] - model["stage_min"]) / model["stage_scale"]
    output[usable] = model["discharge_scale"] * _evaluate_normalized(
        _model_parameters(model), normalized, transition
    )
    return output.reshape(stage.shape)


def derivative_curve(model: dict[str, Any], h: Any) -> np.ndarray:
    """Return dQ/dh; choose zero derivative at/below the zero-flow stage."""
    stage = np.asarray(h, dtype=float)
    flattened = stage.reshape(-1)
    result = np.full(flattened.shape, np.nan, dtype=float)
    usable = np.isfinite(flattened)
    normalized = (flattened[usable] - model["stage_min"]) / model["stage_scale"]
    log_a, head_shift, low_exponent = _model_parameters(model)[:3]
    head = normalized + head_shift
    derivative = np.zeros(normalized.shape, dtype=float)
    positive = head > 0.0
    derivative[positive] = np.exp(log_a) * low_exponent * head[positive] ** (low_exponent - 1)
    if model["segments"] == 2:
        transition = (model["threshold"] - model["stage_min"]) / model["stage_scale"]
        high = normalized > transition
        transition_head = transition + head_shift
        high_exponent = model["params"]["high_exponent"]
        derivative[high] = (
            np.exp(log_a)
            * low_exponent
            * transition_head ** (low_exponent - 1)
            * (head[high] / transition_head) ** (high_exponent - 1)
        )
    result[usable] = derivative * model["discharge_scale"] / model["stage_scale"]
    return result.reshape(stage.shape)


def _transition_candidates(
    stage: np.ndarray,
    min_regime: int,
    fixed_threshold: float | None,
) -> list[float]:
    if fixed_threshold is not None:
        if not np.isfinite(fixed_threshold):
            raise ValueError("The fixed threshold must be finite.")
        candidates = [float(fixed_threshold)]
    else:
        # Quantiles and regime counts use these training observations only.
        # Include the count-feasible tails as well as broad central candidates.
        # Retaining 0.9 where feasible helps identify uncommon high-flow changes.
        # At q=min_regime/n, linear interpolation falls between the m-th and
        # (m+1)-th sorted observations, giving exactly m low rows without ties.
        if stage.size < 2 * min_regime:
            raise ValueError(
                f"A two-regime fit needs at least {2 * min_regime} observations "
                f"to place {min_regime} measurements in each regime."
            )
        tail = min_regime / stage.size
        quantiles = np.unique(
            np.clip(np.r_[tail, np.linspace(0.1, 0.9, 7), 1.0 - tail], tail, 1.0 - tail)
        )
        candidates = np.unique(np.quantile(stage, quantiles)).tolist()
    eligible = [
        float(value)
        for value in candidates
        if np.count_nonzero(stage <= value) >= min_regime
        and np.count_nonzero(stage > value) >= min_regime
        and np.unique(stage[stage <= value]).size >= 3
        and np.unique(stage[stage > value]).size >= 3
    ]
    if not eligible:
        qualifier = "The fixed threshold" if fixed_threshold is not None else "A two-regime fit"
        raise ValueError(
            f"{qualifier} needs at least {min_regime} observations and three distinct "
            "stage values on each side. Use one regime, supply more measurements, "
            "or choose a feasible threshold/min_regime."
        )
    return eligible


def fit_curve(
    h: Any,
    q: Any,
    segments: int = 1,
    loss: str = "linear",
    shape: str = "monotone",
    min_regime: int = 12,
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    """Fit a rating curve and return a JSON-serializable model dictionary.

    ``loss='linear'`` minimizes squared discharge error without high-flow weights.
    ``loss='log1p'`` minimizes squared log1p(Q / discharge_scale) error, retaining
    zero measurements and invariance to discharge units. This is a calibration
    fit only; select models and assess performance on independent data.

    ``segments=2`` searches up to nine training-stage quantiles, including
    count-feasible tails, unless a physical ``fixed_threshold`` is provided.
    ``shape='monotone'`` permits either curvature;
    ``shape='convex'`` constrains both exponents to at least one.
    """
    stage = np.asarray(h, dtype=float)
    discharge = np.asarray(q, dtype=float)
    if stage.ndim != 1 or discharge.ndim != 1 or stage.size != discharge.size:
        raise ValueError("Stage and discharge must be one-dimensional arrays of equal length.")
    if stage.size < 6:
        raise ValueError("At least six paired stage-discharge measurements are required.")
    if not np.all(np.isfinite(stage)) or not np.all(np.isfinite(discharge)):
        raise ValueError("Training stage/discharge contain missing or nonfinite values; clean them first.")
    if np.any(discharge < 0):
        raise ValueError("Negative discharge is unsupported by this nonnegative rating model.")
    if np.unique(stage).size < 3 or np.ptp(stage) <= 0:
        raise ValueError("A rating curve needs at least three distinct stage values.")
    if np.ptp(discharge) <= 0:
        raise ValueError("Discharge is constant; a stage-discharge relationship cannot be identified.")
    if segments not in (1, 2):
        raise ValueError("segments must be 1 or 2.")
    if loss not in ("linear", "log1p"):
        raise ValueError("loss must be 'linear' or 'log1p'.")
    if shape not in ("monotone", "convex"):
        raise ValueError("shape must be 'monotone' or 'convex'.")
    if not isinstance(min_regime, (int, np.integer)) or min_regime < 3:
        raise ValueError("min_regime must be an integer of at least 3.")
    if segments == 1 and fixed_threshold is not None:
        raise ValueError("A fixed threshold is only applicable to a two-regime fit.")

    stage_min = float(np.min(stage))
    stage_scale = float(np.ptp(stage))
    discharge_scale = float(np.quantile(discharge, 0.9))
    if discharge_scale <= 0:
        discharge_scale = float(np.max(discharge))
    normalized_stage = (stage - stage_min) / stage_scale
    normalized_q = discharge / discharge_scale
    physical_thresholds: list[float | None] = [None]
    if segments == 2:
        physical_thresholds = _transition_candidates(stage, min_regime, fixed_threshold)

    minimum_exponent = 1.0 if shape == "convex" else 0.3
    # Finite bounds keep ill-identified dry-stage offsets from diverging.
    lower = np.array([-40.0, 1e-6, minimum_exponent] + ([minimum_exponent] if segments == 2 else []))
    upper = np.array([10.0, 100.0, 5.0] + ([5.0] if segments == 2 else []))
    names = ["log_a", "head_shift", "low_exponent"] + (["high_exponent"] if segments == 2 else [])
    target = np.log1p(normalized_q) if loss == "log1p" else normalized_q
    best_result = None
    best_threshold = None
    best_objective = math.inf
    total_evaluations = 0
    unsuccessful_attempts = 0
    attempts = 0
    errors: list[str] = []

    for threshold in physical_thresholds:
        normalized_threshold = None if threshold is None else (threshold - stage_min) / stage_scale

        def residual(parameters: np.ndarray) -> np.ndarray:
            prediction = _evaluate_normalized(parameters, normalized_stage, normalized_threshold)
            transformed = np.log1p(prediction) if loss == "log1p" else prediction
            return transformed - target

        starts = [(0.03, 1.5), (0.25, 2.5), (1.0, 1.05)]
        for head_shift, exponent in starts:
            amplitude = max(float(np.max(normalized_q)), 1e-8) / (1.0 + head_shift) ** exponent
            initial = np.array([math.log(amplitude), head_shift, exponent] + ([exponent] if segments == 2 else []))
            initial = np.clip(initial, lower + 1e-8, upper - 1e-8)
            attempts += 1
            try:
                fit = least_squares(
                    residual,
                    initial,
                    bounds=(lower, upper),
                    max_nfev=800,
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                    x_scale="jac",
                )
            except (ValueError, FloatingPointError, OverflowError) as exc:
                unsuccessful_attempts += 1
                errors.append(f"threshold={threshold!r}: {type(exc).__name__}: {exc}")
                continue
            total_evaluations += int(fit.nfev)
            if not fit.success or not np.all(np.isfinite(fit.fun)):
                unsuccessful_attempts += 1
                errors.append(f"threshold={threshold!r}: {fit.message}")
                continue
            objective = float(np.mean(fit.fun**2))
            if objective < best_objective:
                best_result = fit
                best_threshold = threshold
                best_objective = objective

    if best_result is None:
        explanation = "; ".join(errors[:3])
        raise RuntimeError(f"All {attempts} curve-fitting attempts failed. {explanation}")

    parameters = {name: float(value) for name, value in zip(names, best_result.x)}
    if segments == 1:
        parameters["high_exponent"] = None
    warnings: list[str] = []
    if np.any(discharge == 0):
        warnings.append(
            "Zero-discharge observations were retained. The zero-flow stage is "
            "constrained below the minimum measured stage; this model cannot "
            "represent a dry-stage cutoff inside the calibration range."
        )
    at_bounds: list[str] = []
    for index, name in enumerate(names):
        value = best_result.x[index]
        tolerance = 1e-4 * max(1.0, abs(value))
        if value - lower[index] <= tolerance:
            at_bounds.append(f"{name}:lower")
        elif upper[index] - value <= tolerance:
            at_bounds.append(f"{name}:upper")
    if at_bounds:
        warnings.append("Fitted parameters reached their bounds: " + ", ".join(at_bounds) + ".")
    if unsuccessful_attempts:
        warnings.append(
            f"{unsuccessful_attempts} of {attempts} optimization starts failed; "
            "the best successfully converged fit was retained."
        )
    model: dict[str, Any] = {
        "model_type": "continuous_power_rating_curve",
        "model_version": 1,
        "segments": int(segments),
        "loss": loss,
        "shape": shape,
        "threshold": None if best_threshold is None else float(best_threshold),
        "stage_min": stage_min,
        "stage_max": float(np.max(stage)),
        "stage_scale": stage_scale,
        "discharge_scale": discharge_scale,
        "zero_stage": stage_min - stage_scale * parameters["head_shift"],
        "params": parameters,
        "success": True,
        "warnings": warnings,
        "fit": {
            "success": True,
            "message": str(best_result.message),
            "n_observations": int(stage.size),
            "n_parameters": int(best_result.x.size + (segments == 2 and fixed_threshold is None)),
            "objective": best_objective,
            "nfev": int(best_result.nfev),
            "total_nfev": total_evaluations,
            "optimization_attempts": attempts,
            "unsuccessful_attempts": unsuccessful_attempts,
            "optimization_failures": errors,
            "parameters_at_bounds": at_bounds,
            "threshold_candidates": [float(t) for t in physical_thresholds if t is not None],
            "threshold_was_fixed": fixed_threshold is not None,
            "parameter_bounds": {name: [float(lo), float(hi)] for name, lo, hi in zip(names, lower, upper)},
        },
    }
    fitted = predict_curve(model, stage)
    model["fit"]["rmse"] = float(np.sqrt(np.mean((fitted - discharge) ** 2)))
    model["fit"]["mae"] = float(np.mean(np.abs(fitted - discharge)))
    if best_threshold is not None:
        model["transition_discharge"] = float(predict_curve(model, best_threshold))
        model["transition_slope"] = float(derivative_curve(model, best_threshold))
        model["fit"]["low_count"] = int(np.count_nonzero(stage <= best_threshold))
        model["fit"]["high_count"] = int(np.count_nonzero(stage > best_threshold))
    else:
        model["transition_discharge"] = None
        model["transition_slope"] = None
    return model


def equation_text(model: dict[str, Any]) -> str:
    """Readable dimensional equations, with h in the input stage units."""
    parameters = model["params"]
    exponent = parameters["low_exponent"]
    coefficient = model["discharge_scale"] * math.exp(parameters["log_a"]) / model["stage_scale"] ** exponent
    zero_stage = model["zero_stage"]
    low = f"Q_L(h) = {coefficient:.6g} * max(h - ({zero_stage:.6g}), 0)^{exponent:.6g}"
    if model["segments"] == 1:
        return low
    threshold = model["threshold"]
    high_exponent = parameters["high_exponent"]
    transition_head = threshold - zero_stage
    multiplier = model["transition_slope"] * transition_head / high_exponent
    high = (
        f"Q_H(h) = {model['transition_discharge']:.6g} + {multiplier:.6g} * "
        f"[((h - ({zero_stage:.6g})) / {transition_head:.6g})^{high_exponent:.6g} - 1]"
    )
    return low + f", h <= {threshold:.6g}\n" + high + f", h > {threshold:.6g}"
