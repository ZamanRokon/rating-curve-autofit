"""
Empirical stage-discharge rating-curve autofit script.
Required CSV columns by default: wl, discharge
Optional CSV column:
    date
Run:
    python rating_curve_autofit.py input.csv
Or edit the USER SETTINGS block and run:
    python rating_curve_autofit.py

The script fits one to three additive power-law segments, applies empirical
support checks, evaluates blocked cross-validation, prefers the simplest model
with comparable predictive accuracy, and calculates bootstrap uncertainty.
Rating tables are restricted to the observed water-level range.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import probplot


# USER SETTINGS
INPUT_CSV = ""  # Example: r"C:\path\to\input.csv"; leave blank to use command line.
OUTPUT_ROOT = "rating_curve_results"

DATE_COLUMN = "date"  # Optional. Time diagnostics are skipped when unavailable.
STAGE_COLUMN = "wl"
DISCHARGE_COLUMN = "discharge"
STAGE_UNIT = "m"
DISCHARGE_UNIT = "m3/s"

MAX_SEGMENTS = 3
MIN_TOTAL_OBSERVATIONS = {1: 20, 2: 40, 3: 60}
MIN_ACTIVE_OBSERVATIONS = 10
CV_FOLDS = 5
SIMPLICITY_TOLERANCE = 0.03
BOOTSTRAP_SAMPLES = 200
CONFIDENCE_LEVEL = 0.90
RATING_TABLE_POINTS = 100
RANDOM_SEED = 12345
GLOBAL_MAXITER = 800
FAST_MAXITER = 1200


# MODEL FUNCTIONS
def parameter_names(n_segments: int) -> list[str]:
    names: list[str] = []
    for i in range(1, n_segments + 1):
        if i == 1:
            names.extend(["h1_zero_flow_stage", "log10_alpha1", "beta1"])
        else:
            names.extend([f"h{i}_activation_stage", f"log10_alpha{i}", f"beta{i}"])
    return names


def predict_discharge(params: np.ndarray, stage: np.ndarray, n_segments: int) -> np.ndarray:
    """Empirical additive power-law curve; each term activates above its threshold."""
    stage = np.asarray(stage, dtype=float)
    q = np.zeros_like(stage, dtype=float)
    for seg in range(n_segments):
        h0 = params[3 * seg]
        alpha = 10.0 ** params[3 * seg + 1]
        beta = params[3 * seg + 2]
        q += alpha * np.power(np.maximum(stage - h0, 0.0), beta)
    return q


def make_bounds(stage: np.ndarray, discharge: np.ndarray, n_segments: int) -> list[tuple[float, float]]:
    s_min = float(np.min(stage))
    s_max = float(np.max(stage))
    s_range = max(s_max - s_min, 1e-6)
    n = len(stage)
    q_min = float(np.min(discharge))
    q_max = float(np.max(discharge))
    log_alpha_low = max(-10.0, math.log10(max(q_min, 1e-12)) - 5.0)
    log_alpha_high = min(10.0, math.log10(max(q_max, 1e-12)) + 5.0)
    if log_alpha_low >= log_alpha_high:
        log_alpha_low, log_alpha_high = -10.0, 10.0

    bounds: list[tuple[float, float]] = []
    ordered = np.sort(stage)
    for seg in range(n_segments):
        if seg == 0:
            h_bounds = (s_min - 2.0 * s_range, s_min - 1e-8 * s_range)
        else:
            low_rank = min(seg * MIN_ACTIVE_OBSERVATIONS, n - 2)
            high_rank = max(low_rank + 1, n - (n_segments - seg) * MIN_ACTIVE_OBSERVATIONS - 1)
            high_rank = min(max(high_rank, low_rank + 1), n - 1)
            h_bounds = (float(ordered[low_rank]), float(ordered[high_rank]))
            if h_bounds[0] >= h_bounds[1]:
                h_bounds = (s_min + 0.05 * s_range, s_max - 0.05 * s_range)
        bounds.extend([h_bounds, (log_alpha_low, log_alpha_high), (0.05, 5.0)])
    return bounds


def invalid_params(params: np.ndarray, stage: np.ndarray, n_segments: int) -> bool:
    if len(params) != 3 * n_segments or not np.all(np.isfinite(params)):
        return True
    h_values = [params[3 * i] for i in range(n_segments)]
    betas = [params[3 * i + 2] for i in range(n_segments)]
    if any(beta <= 0.0 or beta > 8.0 for beta in betas):
        return True
    if h_values[0] >= float(np.min(stage)):
        return True
    return any(h_values[i] >= h_values[i + 1] for i in range(len(h_values) - 1))


def log_residuals(params: np.ndarray, stage: np.ndarray, discharge: np.ndarray, n_segments: int) -> np.ndarray:
    q_pred = predict_discharge(params, stage, n_segments)
    if np.any(q_pred <= 0.0) or np.any(~np.isfinite(q_pred)):
        return np.full_like(discharge, 1e9, dtype=float)
    return np.log10(discharge) - np.log10(q_pred)


def sse_objective(params: np.ndarray, stage: np.ndarray, discharge: np.ndarray, n_segments: int) -> float:
    if invalid_params(params, stage, n_segments):
        return 1e30
    residuals = log_residuals(params, stage, discharge, n_segments)
    return 1e30 if np.any(~np.isfinite(residuals)) else float(np.sum(residuals**2))


def initial_parameters(bounds: list[tuple[float, float]], n_segments: int) -> np.ndarray:
    params = np.array([(low + high) / 2.0 for low, high in bounds], dtype=float)
    if n_segments > 1:
        h = sorted(params[3 * seg] for seg in range(1, n_segments))
        for seg, value in enumerate(h, start=1):
            low, high = bounds[3 * seg]
            params[3 * seg] = np.clip(value, low, high)
    return params


def parameter_support(params: np.ndarray, stage: np.ndarray, n_segments: int) -> tuple[bool, list[int]]:
    active_counts = [int(np.sum(stage > params[3 * seg])) for seg in range(1, n_segments)]
    return all(count >= MIN_ACTIVE_OBSERVATIONS for count in active_counts), active_counts


def parameters_near_bounds(params: np.ndarray, bounds: list[tuple[float, float]]) -> list[str]:
    hits = []
    for name, value, (low, high) in zip(parameter_names(len(params) // 3), params, bounds):
        tolerance = max((high - low) * 0.005, 1e-10)
        if value - low <= tolerance or high - value <= tolerance:
            hits.append(name)
    return hits


def linear_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    score = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "pbias_percent": float(100.0 * np.sum(error) / np.sum(observed)),
        "r2": score,
        "nse": score,
    }


def fit_model(
    stage: np.ndarray,
    discharge: np.ndarray,
    n_segments: int,
    seed_offset: int = 0,
    start_params: np.ndarray | None = None,
    global_search: bool = True,
) -> dict:
    bounds = make_bounds(stage, discharge, n_segments)
    messages: list[str] = []
    if global_search:
        result_global = differential_evolution(
            sse_objective,
            bounds=bounds,
            args=(stage, discharge, n_segments),
            seed=RANDOM_SEED + 1009 * n_segments + seed_offset,
            polish=False,
            updating="immediate",
            workers=1,
            maxiter=GLOBAL_MAXITER,
            tol=1e-8,
        )
        x0 = result_global.x
        messages.append(str(result_global.message))
    else:
        x0 = np.asarray(start_params, dtype=float).copy() if start_params is not None else initial_parameters(bounds, n_segments)
        x0 = np.array([np.clip(value, low, high) for value, (low, high) in zip(x0, bounds)])
        if invalid_params(x0, stage, n_segments):
            x0 = initial_parameters(bounds, n_segments)

    result_local = minimize(
        sse_objective,
        x0=x0,
        args=(stage, discharge, n_segments),
        method="Powell",
        bounds=bounds,
        options={"maxiter": FAST_MAXITER, "ftol": 1e-10, "xtol": 1e-8},
    )
    messages.append(str(result_local.message))
    candidates = [(result_local.fun, result_local.x)]
    if global_search:
        candidates.append((result_global.fun, result_global.x))
    objective, params = min(candidates, key=lambda item: item[0])
    params = np.asarray(params, dtype=float)
    valid = bool(np.isfinite(objective) and objective < 1e20 and not invalid_params(params, stage, n_segments))
    if not valid:
        return {
            "n_segments": n_segments,
            "params": params,
            "param_names": parameter_names(n_segments),
            "success": False,
            "optimizer_message": " | ".join(messages),
            "bic": float("inf"),
            "aic": float("inf"),
            "aicc": float("inf"),
            "cv_rmse_log10": float("inf"),
        }

    residuals = log_residuals(params, stage, discharge, n_segments)
    sse = float(np.sum(residuals**2))
    n = len(discharge)
    sigma = math.sqrt(max(sse / n, 1e-30))
    sigma_ln = math.log(10.0) * sigma
    exact_cv = math.sqrt(math.exp(sigma_ln**2) - 1.0)
    log_likelihood = -0.5 * n * math.log(2.0 * math.pi) - n * math.log(sigma) - sse / (2.0 * sigma**2)
    k = 3 * n_segments + 1
    aic = 2.0 * k - 2.0 * log_likelihood
    aicc = aic + 2.0 * k * (k + 1) / (n - k - 1) if n > k + 1 else float("inf")
    bic = k * math.log(n) - 2.0 * log_likelihood
    q_median = predict_discharge(params, stage, n_segments)
    q_mean = q_median * math.exp(0.5 * sigma_ln**2)
    metrics = linear_metrics(discharge, q_median)
    supported, active_counts = parameter_support(params, stage, n_segments)

    return {
        "n_segments": n_segments,
        "params": params,
        "param_names": parameter_names(n_segments),
        "sigma_log10": sigma,
        "sigma_ln": sigma_ln,
        "cv_exact": exact_cv,
        "sse_log10": sse,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "rmse_log10": math.sqrt(sse / n),
        "residuals": residuals,
        "fitted_median": q_median,
        "fitted_mean": q_mean,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "bias": metrics["bias"],
        "pbias_percent": metrics["pbias_percent"],
        "r2": metrics["r2"],
        "nse": metrics["nse"],
        "active_counts": active_counts,
        "empirically_supported": supported,
        "near_bounds": parameters_near_bounds(params, bounds),
        "success": valid,
        "optimizer_converged": bool(result_local.success),
        "optimizer_message": " | ".join(messages),
        "cv_rmse_log10": float("inf"),
        "cv_mae_log10": float("inf"),
        "cv_rmse": float("inf"),
        "cv_mae": float("inf"),
        "cv_bias": float("nan"),
        "cv_n": 0,
    }


def equation_text(fit: dict, digits: int = 6) -> str:
    parts: list[str] = []
    for seg in range(fit["n_segments"]):
        h0 = fit["params"][3 * seg]
        alpha = 10.0 ** fit["params"][3 * seg + 1]
        beta = fit["params"][3 * seg + 2]
        parts.append(f"{alpha:.{digits}g} * max(h - {h0:.{digits}g}, 0)^{beta:.{digits}g}")
    return "Q_median(h) = " + " + ".join(parts)


# VALIDATION AND UNCERTAINTY
def validation_folds(data: pd.DataFrame, folds: int) -> tuple[list[np.ndarray], str]:
    folds = min(max(2, folds), len(data))
    complete_dates = DATE_COLUMN in data and data[DATE_COLUMN].notna().all()
    if complete_dates and data[DATE_COLUMN].nunique() >= folds:
        order = np.argsort(data[DATE_COLUMN].to_numpy())
        return [part for part in np.array_split(order, folds) if len(part)], "date-blocked"
    order = np.argsort(data[STAGE_COLUMN].to_numpy(dtype=float))
    return [order[i::folds] for i in range(folds) if len(order[i::folds])], "stage-stratified"


def cross_validate_model(data: pd.DataFrame, fit: dict, folds: int) -> dict:
    stage = data[STAGE_COLUMN].to_numpy(dtype=float)
    discharge = data[DISCHARGE_COLUMN].to_numpy(dtype=float)
    test_folds, method = validation_folds(data, folds)
    observed_all: list[np.ndarray] = []
    predicted_all: list[np.ndarray] = []
    for fold_number, test_idx in enumerate(test_folds):
        train_mask = np.ones(len(data), dtype=bool)
        train_mask[test_idx] = False
        if np.sum(train_mask) < max(6, 3 * fit["n_segments"] + 2):
            continue
        fold_fit = fit_model(
            stage[train_mask],
            discharge[train_mask],
            fit["n_segments"],
            seed_offset=5000 + fold_number,
            start_params=fit["params"],
            global_search=False,
        )
        if not fold_fit["success"]:
            continue
        predicted = predict_discharge(fold_fit["params"], stage[test_idx], fit["n_segments"])
        valid = np.isfinite(predicted) & (predicted > 0.0)
        if np.any(valid):
            observed_all.append(discharge[test_idx][valid])
            predicted_all.append(predicted[valid])
    if not observed_all:
        return {"method": method, "n": 0, "rmse_log10": float("inf")}
    observed = np.concatenate(observed_all)
    predicted = np.concatenate(predicted_all)
    log_error = np.log10(observed) - np.log10(predicted)
    metrics = linear_metrics(observed, predicted)
    return {
        "method": method,
        "n": int(len(observed)),
        "rmse_log10": float(np.sqrt(np.mean(log_error**2))),
        "mae_log10": float(np.mean(np.abs(log_error))),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "bias": metrics["bias"],
    }


def choose_best_model(fits: list[dict]) -> tuple[dict, str]:
    eligible = [fit for fit in fits if fit["success"] and fit["empirically_supported"] and np.isfinite(fit["cv_rmse_log10"])]
    if eligible:
        minimum_cv = min(fit["cv_rmse_log10"] for fit in eligible)
        comparable = [fit for fit in eligible if fit["cv_rmse_log10"] <= minimum_cv * (1.0 + SIMPLICITY_TOLERANCE)]
        best = min(comparable, key=lambda item: (item["n_segments"], item["bic"]))
        rule = f"lowest cross-validated log RMSE, preferring the simpler model within {100 * SIMPLICITY_TOLERANCE:.1f}%"
        return best, rule
    eligible = [fit for fit in fits if fit["success"] and fit["empirically_supported"]]
    if eligible:
        return min(eligible, key=lambda item: item["bic"]), "minimum BIC because cross-validation was unavailable"
    raise RuntimeError("No valid, empirically supported rating-curve model was fitted.")


def bootstrap_intervals(data: pd.DataFrame, best: dict, grid: np.ndarray, samples: int) -> dict:
    if samples <= 0:
        return {"successful": 0}
    rng = np.random.default_rng(RANDOM_SEED + 90000)
    stage = data[STAGE_COLUMN].to_numpy(dtype=float)
    discharge = data[DISCHARGE_COLUMN].to_numpy(dtype=float)
    curve_draws: list[np.ndarray] = []
    prediction_draws: list[np.ndarray] = []
    parameter_draws: list[np.ndarray] = []
    for sample in range(samples):
        index = rng.integers(0, len(data), len(data))
        boot_fit = fit_model(
            stage[index],
            discharge[index],
            best["n_segments"],
            seed_offset=10000 + sample,
            start_params=best["params"],
            global_search=False,
        )
        if not boot_fit["success"]:
            continue
        curve = predict_discharge(boot_fit["params"], grid, best["n_segments"])
        if np.any(~np.isfinite(curve)) or np.any(curve <= 0.0):
            continue
        curve_draws.append(curve)
        parameter_draws.append(boot_fit["params"])
        noise = rng.normal(0.0, boot_fit["sigma_log10"], len(grid))
        prediction_draws.append(curve * np.power(10.0, noise))
    if not curve_draws:
        return {"successful": 0}
    lower = 100.0 * (1.0 - CONFIDENCE_LEVEL) / 2.0
    upper = 100.0 - lower
    curves = np.asarray(curve_draws)
    predictions = np.asarray(prediction_draws)
    parameters = np.asarray(parameter_draws)
    return {
        "successful": int(len(curves)),
        "curve_lower": np.percentile(curves, lower, axis=0),
        "curve_median": np.percentile(curves, 50.0, axis=0),
        "curve_upper": np.percentile(curves, upper, axis=0),
        "prediction_lower": np.percentile(predictions, lower, axis=0),
        "prediction_upper": np.percentile(predictions, upper, axis=0),
        "parameter_lower": np.percentile(parameters, lower, axis=0),
        "parameter_median": np.percentile(parameters, 50.0, axis=0),
        "parameter_upper": np.percentile(parameters, upper, axis=0),
    }


# INPUT / OUTPUT
def read_input_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    required = [STAGE_COLUMN, DISCHARGE_COLUMN]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found columns: {list(raw.columns)}")
    columns = required + ([DATE_COLUMN] if DATE_COLUMN in raw.columns else [])
    out = raw[columns].copy()
    stage_numeric = pd.to_numeric(out[STAGE_COLUMN], errors="coerce")
    discharge_numeric = pd.to_numeric(out[DISCHARGE_COLUMN], errors="coerce")
    invalid_stage = int(stage_numeric.isna().sum())
    invalid_discharge = int(discharge_numeric.isna().sum())
    nonpositive_discharge = int(((discharge_numeric <= 0.0) & discharge_numeric.notna()).sum())
    out[STAGE_COLUMN] = stage_numeric
    out[DISCHARGE_COLUMN] = discharge_numeric
    valid = stage_numeric.notna() & discharge_numeric.notna() & (discharge_numeric > 0.0)
    out = out.loc[valid].copy()
    invalid_dates = 0
    if DATE_COLUMN in out.columns:
        parsed_dates = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
        invalid_dates = int(parsed_dates.isna().sum())
        out[DATE_COLUMN] = parsed_dates
        if parsed_dates.notna().all():
            out = out.sort_values(DATE_COLUMN)
    out = out.reset_index(drop=True)
    duplicate_pairs = int(out.duplicated([STAGE_COLUMN, DISCHARGE_COLUMN]).sum())
    duplicate_dates = int(out[DATE_COLUMN].duplicated().sum()) if DATE_COLUMN in out.columns else 0
    quality = {
        "rows_read": int(len(raw)),
        "rows_used": int(len(out)),
        "rows_removed": int(len(raw) - len(out)),
        "invalid_stage_rows": invalid_stage,
        "invalid_discharge_rows": invalid_discharge,
        "nonpositive_discharge_rows": nonpositive_discharge,
        "date_column_available": DATE_COLUMN in out.columns,
        "invalid_date_rows_retained": invalid_dates,
        "repeated_stage_discharge_pairs_retained": duplicate_pairs,
        "duplicate_dates_retained": duplicate_dates,
        "stage_min": float(out[STAGE_COLUMN].min()) if len(out) else None,
        "stage_max": float(out[STAGE_COLUMN].max()) if len(out) else None,
        "discharge_min": float(out[DISCHARGE_COLUMN].min()) if len(out) else None,
        "discharge_max": float(out[DISCHARGE_COLUMN].max()) if len(out) else None,
    }
    if len(out) < MIN_TOTAL_OBSERVATIONS[1]:
        raise ValueError(f"At least {MIN_TOTAL_OBSERVATIONS[1]} valid paired observations are required.")
    if out[STAGE_COLUMN].nunique() < 4:
        raise ValueError("At least four unique water-level values are required.")
    return out, quality


def candidate_segment_counts(n: int, requested_max: int) -> list[int]:
    requested_max = min(max(1, requested_max), 3)
    return [segments for segments in range(1, requested_max + 1) if n >= MIN_TOTAL_OBSERVATIONS[segments]]


def diagnostic_warnings(data: pd.DataFrame, best: dict, bootstrap: dict, requested_bootstrap: int) -> list[str]:
    notes: list[str] = []
    if best["near_bounds"]:
        notes.append("Best-model parameters near search bounds: " + ", ".join(best["near_bounds"]) + ".")
    if not best["optimizer_converged"]:
        notes.append("The local optimizer did not formally converge; inspect the fitted curve and parameter bounds.")
    if requested_bootstrap > 0 and bootstrap.get("successful", 0) < max(30, int(0.5 * requested_bootstrap)):
        notes.append("Few bootstrap fits succeeded; uncertainty intervals may be unstable.")
    fitted_log = np.log10(best["fitted_median"])
    abs_residual = np.abs(best["residuals"])
    if len(data) >= 10 and np.std(fitted_log) > 0.0 and np.std(abs_residual) > 0.0:
        hetero_corr = float(np.corrcoef(fitted_log, abs_residual)[0, 1])
        best["absolute_residual_logq_correlation"] = hetero_corr
        if abs(hetero_corr) >= 0.30:
            notes.append("Residual spread changes with discharge; the constant log-error assumption may be weak.")
    if DATE_COLUMN not in data.columns or data[DATE_COLUMN].isna().any():
        notes.append("Complete dates were unavailable, so chronological shift diagnostics and date-blocked validation were not possible.")
    notes.append("The curve is empirical; breakpoint causes, backwater and flood-wave hysteresis cannot be confirmed from paired stage-discharge data alone.")
    notes.append("Do not extrapolate the rating beyond the observed water-level range without additional measurements.")
    return notes


def save_tables(
    out_dir: Path,
    data: pd.DataFrame,
    quality: dict,
    fits: list[dict],
    best: dict,
    selection_rule: str,
    grid: np.ndarray,
    bootstrap: dict,
    notes: list[str],
    requested_bootstrap: int,
) -> None:
    data.to_csv(out_dir / "cleaned_data.csv", index=False)
    (out_dir / "input_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    comparison_rows = []
    for fit in fits:
        comparison_rows.append(
            {
                "segments": fit["n_segments"],
                "empirically_supported": fit["empirically_supported"],
                "active_observations_above_thresholds": ";".join(map(str, fit["active_counts"])),
                "aic": fit["aic"],
                "aicc": fit["aicc"],
                "bic": fit["bic"],
                "rmse_log10_in_sample": fit["rmse_log10"],
                "rmse_log10_cross_validated": fit["cv_rmse_log10"],
                "mae_log10_cross_validated": fit["cv_mae_log10"],
                "rmse_discharge_in_sample": fit["rmse"],
                "mae_discharge_in_sample": fit["mae"],
                "bias_discharge_in_sample": fit["bias"],
                "pbias_percent_in_sample": fit["pbias_percent"],
                "r2_in_sample": fit["r2"],
                "nse_in_sample": fit["nse"],
                "sigma_log10": fit["sigma_log10"],
                "cv_exact_fraction": fit["cv_exact"],
                "success": fit["success"],
                "optimizer_converged": fit["optimizer_converged"],
                "parameters_near_bounds": ";".join(fit["near_bounds"]),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(out_dir / "model_comparison.csv", index=False)
    parameter_rows = []
    has_intervals = bootstrap.get("successful", 0) > 0
    for index, (name, value) in enumerate(zip(best["param_names"], best["params"])):
        row = {"parameter": name, "estimate": value}
        if has_intervals:
            row.update(
                {
                    "bootstrap_lower": bootstrap["parameter_lower"][index],
                    "bootstrap_median": bootstrap["parameter_median"][index],
                    "bootstrap_upper": bootstrap["parameter_upper"][index],
                }
            )
        parameter_rows.append(row)
    pd.DataFrame(parameter_rows).to_csv(out_dir / "best_parameters.csv", index=False)
    fitted = data.copy()
    fitted["fitted_discharge_median"] = best["fitted_median"]
    fitted["fitted_discharge_mean_bias_corrected"] = best["fitted_mean"]
    fitted["residual_log10"] = best["residuals"]
    fitted["error"] = best["fitted_median"] - fitted[DISCHARGE_COLUMN]
    fitted["absolute_error"] = np.abs(fitted["error"])
    fitted["percent_error"] = 100.0 * fitted["error"] / fitted[DISCHARGE_COLUMN]
    fitted["symmetric_percent_error"] = 200.0 * fitted["error"] / (
        np.abs(best["fitted_median"]) + np.abs(fitted[DISCHARGE_COLUMN])
    )
    fitted["standardized_log_residual"] = best["residuals"] / max(best["sigma_log10"], 1e-12)
    fitted["potential_outlier_abs_z_gt_3"] = np.abs(fitted["standardized_log_residual"]) > 3.0
    fitted.to_csv(out_dir / "fitted_values_and_residuals.csv", index=False)
    median_curve = predict_discharge(best["params"], grid, best["n_segments"])
    mean_curve = median_curve * math.exp(0.5 * best["sigma_ln"] ** 2)
    rating = pd.DataFrame(
        {
            f"stage_{STAGE_UNIT}": grid,
            f"discharge_median_{DISCHARGE_UNIT}": median_curve,
            f"discharge_mean_bias_corrected_{DISCHARGE_UNIT}": mean_curve,
            "within_observed_stage_range": True,
        }
    )
    if has_intervals:
        level = int(round(CONFIDENCE_LEVEL * 100))
        rating[f"curve_lower_{level}"] = bootstrap["curve_lower"]
        rating[f"curve_upper_{level}"] = bootstrap["curve_upper"]
        rating[f"prediction_lower_{level}"] = bootstrap["prediction_lower"]
        rating[f"prediction_upper_{level}"] = bootstrap["prediction_upper"]
    rating.to_csv(out_dir / "rating_table.csv", index=False)
    best_json = {
        "model_description": "Empirical additive power-law stage-discharge rating curve",
        "best_segments": best["n_segments"],
        "selection_rule": selection_rule,
        "equation_returns": "conditional median discharge",
        "equation": equation_text(best),
        "observed_stage_range": [float(grid.min()), float(grid.max())],
        "stage_unit": STAGE_UNIT,
        "discharge_unit": DISCHARGE_UNIT,
        "parameters": {name: float(value) for name, value in zip(best["param_names"], best["params"])},
        "sigma_log10": float(best["sigma_log10"]),
        "cv_exact_fraction": float(best["cv_exact"]),
        "aicc": float(best["aicc"]),
        "bic": float(best["bic"]),
        "rmse_log10_cross_validated": float(best["cv_rmse_log10"]),
        "bootstrap_requested": requested_bootstrap,
        "bootstrap_successful": int(bootstrap.get("successful", 0)),
        "warnings": notes,
    }
    (out_dir / "best_model.json").write_text(json.dumps(best_json, indent=2), encoding="utf-8")
    (out_dir / "equation.txt").write_text(equation_text(best) + "\n", encoding="utf-8")


def finish_plot(path: Path, show_plots: bool) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()


def save_plots(
    out_dir: Path,
    data: pd.DataFrame,
    fits: list[dict],
    best: dict,
    grid: np.ndarray,
    bootstrap: dict,
    show_plots: bool,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    stage = data[STAGE_COLUMN].to_numpy(dtype=float)
    discharge = data[DISCHARGE_COLUMN].to_numpy(dtype=float)
    q_grid = predict_discharge(best["params"], grid, best["n_segments"])
    plt.figure(figsize=(9, 6))
    if bootstrap.get("successful", 0) > 0:
        level = int(round(CONFIDENCE_LEVEL * 100))
        plt.fill_between(grid, bootstrap["prediction_lower"], bootstrap["prediction_upper"], color="#9ecae1", alpha=0.25, label=f"{level}% prediction interval")
        plt.fill_between(grid, bootstrap["curve_lower"], bootstrap["curve_upper"], color="#3182bd", alpha=0.25, label=f"{level}% curve interval")
    plt.scatter(stage, discharge, s=28, color="#1f77b4", alpha=0.75, label="Observed")
    plt.plot(grid, q_grid, linewidth=2.6, color="#d62728", label=f"Selected: {best['n_segments']} segment(s)")
    plt.xlabel(f"Water level ({STAGE_UNIT})")
    plt.ylabel(f"Discharge ({DISCHARGE_UNIT})")
    plt.title("Empirical Stage-Discharge Rating Curve")
    plt.legend()
    plt.grid(True, alpha=0.25)
    finish_plot(plots_dir / "rating_curve.png", show_plots)
    plt.figure(figsize=(9, 6))
    if bootstrap.get("successful", 0) > 0:
        plt.fill_between(grid, bootstrap["prediction_lower"], bootstrap["prediction_upper"], color="#9ecae1", alpha=0.25)
        plt.fill_between(grid, bootstrap["curve_lower"], bootstrap["curve_upper"], color="#3182bd", alpha=0.25)
    plt.scatter(stage, discharge, s=28, color="#1f77b4", alpha=0.75, label="Observed")
    plt.plot(grid, q_grid, linewidth=2.6, color="#d62728", label="Selected curve")
    plt.yscale("log")
    plt.xlabel(f"Water level ({STAGE_UNIT})")
    plt.ylabel(f"Discharge ({DISCHARGE_UNIT}, log scale)")
    plt.title("Rating Curve on Log-Discharge Scale")
    plt.legend()
    plt.grid(True, alpha=0.25, which="both")
    finish_plot(plots_dir / "rating_curve_log_scale.png", show_plots)
    if DATE_COLUMN in data.columns and data[DATE_COLUMN].notna().all():
        plt.figure(figsize=(9, 6))
        date_number = data[DATE_COLUMN].map(pd.Timestamp.toordinal)
        points = plt.scatter(stage, discharge, c=date_number, cmap="viridis", s=34, alpha=0.85)
        plt.plot(grid, q_grid, color="black", linewidth=2.0, label="Selected curve")
        colorbar = plt.colorbar(points)
        ticks = np.linspace(date_number.min(), date_number.max(), min(5, data[DATE_COLUMN].nunique()))
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([pd.Timestamp.fromordinal(int(value)).strftime("%Y-%m-%d") for value in ticks])
        colorbar.set_label("Observation date")
        plt.xlabel(f"Water level ({STAGE_UNIT})")
        plt.ylabel(f"Discharge ({DISCHARGE_UNIT})")
        plt.title("Rating Observations Coloured by Date")
        plt.legend()
        plt.grid(True, alpha=0.25)
        finish_plot(plots_dir / "rating_curve_by_date.png", show_plots)
    residuals = best["residuals"]
    plt.figure(figsize=(9, 5))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.scatter(stage, residuals, s=28, color="#2ca02c", alpha=0.75)
    plt.xlabel(f"Water level ({STAGE_UNIT})")
    plt.ylabel("log10(Q observed) - log10(Q fitted)")
    plt.title("Residuals vs Water Level")
    plt.grid(True, alpha=0.25)
    finish_plot(plots_dir / "residuals_vs_stage.png", show_plots)
    plt.figure(figsize=(9, 5))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.scatter(best["fitted_median"], residuals, s=28, color="#ff7f0e", alpha=0.75)
    plt.xscale("log")
    plt.xlabel(f"Fitted median discharge ({DISCHARGE_UNIT}, log scale)")
    plt.ylabel("Log10 residual")
    plt.title("Residuals vs Fitted Discharge")
    plt.grid(True, alpha=0.25, which="both")
    finish_plot(plots_dir / "residuals_vs_fitted.png", show_plots)
    if DATE_COLUMN in data.columns and data[DATE_COLUMN].notna().all():
        plt.figure(figsize=(10, 5))
        plt.axhline(0.0, color="black", linewidth=1)
        plt.scatter(data[DATE_COLUMN], residuals, s=28, color="#17becf", alpha=0.8)
        plt.xlabel("Date")
        plt.ylabel("Log10 residual")
        plt.title("Residuals over Time")
        plt.grid(True, alpha=0.25)
        finish_plot(plots_dir / "residuals_vs_date.png", show_plots)
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=min(25, max(8, len(residuals) // 4)), color="#9467bd", alpha=0.75, edgecolor="white")
    plt.xlabel("Residual in log10 discharge")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    finish_plot(plots_dir / "residual_histogram.png", show_plots)
    plt.figure(figsize=(6, 6))
    probplot(residuals, dist="norm", plot=plt)
    plt.title("Residual Q-Q Plot")
    finish_plot(plots_dir / "residual_qq_plot.png", show_plots)
    comparison = pd.DataFrame({"segments": [fit["n_segments"] for fit in fits], "CV RMSE log10": [fit["cv_rmse_log10"] for fit in fits]})
    ax = comparison.plot(x="segments", y="CV RMSE log10", marker="o", figsize=(8, 5), legend=False)
    ax.set_xlabel("Number of segments")
    ax.set_ylabel("Cross-validated RMSE in log10 discharge")
    ax.set_title("Predictive Model Comparison")
    ax.grid(True, alpha=0.25)
    finish_plot(plots_dir / "model_comparison.png", show_plots)


def save_report(
    out_dir: Path,
    input_path: Path,
    data: pd.DataFrame,
    quality: dict,
    fits: list[dict],
    best: dict,
    selection_rule: str,
    bootstrap: dict,
    notes: list[str],
    requested_bootstrap: int,
) -> None:
    lines = [
        "# Empirical Rating Curve Autofit Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Input CSV: `{input_path}`",
        f"Valid paired observations used: {len(data)} of {quality['rows_read']}",
        f"Observed water-level range: {data[STAGE_COLUMN].min():.6g} to {data[STAGE_COLUMN].max():.6g} {STAGE_UNIT}",
        f"Observed discharge range: {data[DISCHARGE_COLUMN].min():.6g} to {data[DISCHARGE_COLUMN].max():.6g} {DISCHARGE_UNIT}",
        "",
        "## Selected Model",
        "",
        f"Selection rule: {selection_rule}.",
        f"Selected number of segments: **{best['n_segments']}**",
        f"Cross-validation method: {best['cv_method']}",
        "The equation estimates conditional median discharge, not mean discharge.",
        "",
        "```text",
        equation_text(best),
        "```",
        "",
        "## Model Comparison",
        "",
        "| Segments | Supported | AICc | BIC | CV RMSE log10 | In-sample RMSE | In-sample MAE | PBIAS |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fit in fits:
        lines.append(
            f"| {fit['n_segments']} | {fit['empirically_supported']} | {fit['aicc']:.3f} | {fit['bic']:.3f} | "
            f"{fit['cv_rmse_log10']:.6f} | {fit['rmse']:.6g} | {fit['mae']:.6g} | {fit['pbias_percent']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Error and Uncertainty",
            "",
            f"Residual standard deviation: {best['sigma_log10']:.6g} log10 units",
            f"Exact lognormal coefficient of variation: {100.0 * best['cv_exact']:.3f}%",
            f"Successful bootstrap fits: {bootstrap.get('successful', 0)} of {requested_bootstrap}",
            f"Reported bootstrap interval: {100.0 * CONFIDENCE_LEVEL:.0f}%",
            "",
            "The rating table contains the median curve and lognormal bias-corrected mean. Curve intervals describe fitted-curve uncertainty; prediction intervals also include residual scatter.",
            "",
            "## Data Screening",
            "",
            f"Rows removed because stage/discharge was invalid or discharge was nonpositive: {quality['rows_removed']}",
            f"Repeated stage-discharge pairs retained: {quality['repeated_stage_discharge_pairs_retained']}",
            "Potential outliers are flagged in the residual table but are not removed automatically.",
            "",
            "## Warnings and Limitations",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `input_quality.json`",
            "- `cleaned_data.csv`",
            "- `best_parameters.csv`",
            "- `best_model.json`",
            "- `equation.txt`",
            "- `model_comparison.csv`",
            "- `fitted_values_and_residuals.csv`",
            "- `rating_table.csv`",
            "- `plots/`",
            "",
            "## Method and Credit",
            "",
            "This script fits empirical additive power-law stage-discharge curves from paired observations. It is inspired by published hydraulic-control rating-curve methodology, but it is not a BaRatin implementation and does not infer the physical cause of controls.",
            "",
            "References:",
            "",
            "- RMC-BestFit rating-curve technical reference: https://github.com/USACE-RMC/RMC-BestFit/blob/main/docs/technical-reference/analysis/rating-curve.md",
            "- BaRatin method overview: https://baratin-tools.github.io/en/",
            "- Le Coz et al. (2014), Journal of Hydrology, Bayesian rating-curve method.",
            "- Rantz et al. (1982), USGS Water-Supply Paper 2175.",
            "- Kennedy (1984), USGS discharge ratings at gaging stations.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# WORKFLOW
def run(
    input_csv: str | Path = INPUT_CSV,
    output_root: str | Path = OUTPUT_ROOT,
    max_segments: int = MAX_SEGMENTS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    show_plots: bool = False,
) -> Path:
    requested_bootstrap = max(0, int(bootstrap_samples))
    if not input_csv:
        raise ValueError("Provide an input CSV path or set INPUT_CSV in USER SETTINGS.")
    input_path = Path(input_csv).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    data, quality = read_input_csv(input_path)
    stage = data[STAGE_COLUMN].to_numpy(dtype=float)
    discharge = data[DISCHARGE_COLUMN].to_numpy(dtype=float)
    fits = [fit_model(stage, discharge, segments) for segments in candidate_segment_counts(len(data), max_segments)]
    fits = [fit for fit in fits if fit["success"]]
    if not fits:
        raise RuntimeError("All candidate models failed during optimization.")
    for fit in fits:
        cv = cross_validate_model(data, fit, CV_FOLDS)
        fit["cv_method"] = cv["method"]
        fit["cv_rmse_log10"] = cv["rmse_log10"]
        fit["cv_mae_log10"] = cv.get("mae_log10", float("inf"))
        fit["cv_rmse"] = cv.get("rmse", float("inf"))
        fit["cv_mae"] = cv.get("mae", float("inf"))
        fit["cv_bias"] = cv.get("bias", float("nan"))
        fit["cv_n"] = cv["n"]
    best, selection_rule = choose_best_model(fits)
    grid = np.linspace(float(stage.min()), float(stage.max()), RATING_TABLE_POINTS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bootstrap = bootstrap_intervals(data, best, grid, requested_bootstrap)
    notes = diagnostic_warnings(data, best, bootstrap, requested_bootstrap)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root).expanduser().resolve() / f"{input_path.stem}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tables(out_dir, data, quality, fits, best, selection_rule, grid, bootstrap, notes, requested_bootstrap)
    save_plots(out_dir, data, fits, best, grid, bootstrap, show_plots)
    save_report(out_dir, input_path, data, quality, fits, best, selection_rule, bootstrap, notes, requested_bootstrap)
    print("")
    print("Empirical rating-curve autofit complete.")
    print(f"Output folder: {out_dir}")
    print(f"Selected model: {best['n_segments']} segment(s)")
    print(f"Selection: {selection_rule}")
    print(equation_text(best))
    print(f"Bootstrap fits: {bootstrap.get('successful', 0)} / {requested_bootstrap}")
    print("")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit an empirical rating curve from paired water level and discharge.")
    parser.add_argument("csv", nargs="?", default=INPUT_CSV, help="CSV containing wl and discharge; date is optional.")
    parser.add_argument("--out", default=OUTPUT_ROOT, help="Output root folder.")
    parser.add_argument("--max-segments", type=int, default=MAX_SEGMENTS, choices=(1, 2, 3))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_SAMPLES, help="Bootstrap resamples; use 0 to disable.")
    parser.add_argument("--show-plots", action="store_true", help="Display plots as well as saving them.")
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    args = parser.parse_args(argv)
    if not args.csv:
        raise SystemExit("Provide an input CSV path or edit INPUT_CSV at the top of the script.")
    run(args.csv, args.out, args.max_segments, args.bootstrap, args.show_plots)


if __name__ == "__main__":
    main()
