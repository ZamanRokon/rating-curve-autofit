"""
Simple rating-curve autofit script.

Input CSV columns by default:
    date, wl, discharge

Run:
    python rating_curve_autofit.py input.csv

Or edit the USER SETTINGS block below and run:
    python rating_curve_autofit.py

Outputs are written to a folder containing cleaned data, fitted values,
model-comparison metrics, equations, plots, and a markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import probplot


# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_CSV = ""  # Example: r"C:\path\to\input.csv"; leave blank to use command line.
OUTPUT_ROOT = "rating_curve_results"

DATE_COLUMN = "date"
STAGE_COLUMN = "wl"
DISCHARGE_COLUMN = "discharge"

MAX_SEGMENTS = 3
RATING_TABLE_POINTS = 100
RANDOM_SEED = 12345


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================


def parameter_names(n_segments: int) -> list[str]:
    names: list[str] = []
    for i in range(1, n_segments + 1):
        if i == 1:
            names.extend(["h1_zero_flow_stage", "log10_alpha1", "beta1"])
        else:
            names.extend([f"h{i}_activation_stage", f"log10_alpha{i}", f"beta{i}"])
    return names


def predict_discharge(params: np.ndarray, stage: np.ndarray, n_segments: int) -> np.ndarray:
    """BaRatin-style addition-mode power-law rating curve."""
    stage = np.asarray(stage, dtype=float)
    q = np.zeros_like(stage, dtype=float)

    for seg in range(n_segments):
        h0 = params[3 * seg]
        alpha = 10.0 ** params[3 * seg + 1]
        beta = params[3 * seg + 2]

        if seg == 0:
            active = stage > h0
        else:
            active = stage > h0

        depth = np.zeros_like(stage, dtype=float)
        depth[active] = stage[active] - h0
        q[active] += alpha * np.power(depth[active], beta)

    return q


def invalid_params(params: np.ndarray, stage: np.ndarray, n_segments: int) -> bool:
    h_values = [params[3 * i] for i in range(n_segments)]
    betas = [params[3 * i + 2] for i in range(n_segments)]

    if not all(np.isfinite(params)):
        return True
    if any(beta <= 0.0 or beta > 8.0 for beta in betas):
        return True
    if h_values[0] >= np.min(stage):
        return True
    if any(h_values[i] >= h_values[i + 1] for i in range(len(h_values) - 1)):
        return True
    return False


def log_residuals(params: np.ndarray, stage: np.ndarray, discharge: np.ndarray, n_segments: int) -> np.ndarray:
    q_pred = predict_discharge(params, stage, n_segments)
    if np.any(q_pred <= 0.0) or np.any(~np.isfinite(q_pred)):
        return np.full_like(discharge, 1e9, dtype=float)
    return np.log10(discharge) - np.log10(q_pred)


def sse_objective(params: np.ndarray, stage: np.ndarray, discharge: np.ndarray, n_segments: int) -> float:
    if invalid_params(params, stage, n_segments):
        return 1e30
    residuals = log_residuals(params, stage, discharge, n_segments)
    if np.any(~np.isfinite(residuals)):
        return 1e30
    return float(np.sum(residuals * residuals))


def make_bounds(stage: np.ndarray, discharge: np.ndarray, n_segments: int) -> list[tuple[float, float]]:
    s_min = float(np.min(stage))
    s_max = float(np.max(stage))
    s_range = max(s_max - s_min, 1e-6)

    q_min = float(np.min(discharge))
    q_max = float(np.max(discharge))
    approx_log_alpha_min = math.log10(max(q_min, 1e-12)) - 5.0
    approx_log_alpha_max = math.log10(max(q_max, 1e-12)) + 5.0

    bounds: list[tuple[float, float]] = []
    for seg in range(n_segments):
        if seg == 0:
            h_bounds = (s_min - 2.0 * s_range, s_min - 1e-6)
        elif seg == 1:
            h_bounds = (s_min + 0.10 * s_range, s_min + 0.90 * s_range)
        else:
            h_bounds = (s_min + 0.35 * s_range, s_min + 0.98 * s_range)

        bounds.extend(
            [
                h_bounds,
                (max(-10.0, approx_log_alpha_min), min(10.0, approx_log_alpha_max)),
                (0.05, 5.0),
            ]
        )

    return bounds


def fit_model(stage: np.ndarray, discharge: np.ndarray, n_segments: int) -> dict:
    bounds = make_bounds(stage, discharge, n_segments)

    result_global = differential_evolution(
        sse_objective,
        bounds=bounds,
        args=(stage, discharge, n_segments),
        seed=RANDOM_SEED + n_segments,
        polish=False,
        updating="immediate",
        workers=1,
        maxiter=1200,
        tol=1e-8,
    )

    result_local = minimize(
        sse_objective,
        x0=result_global.x,
        args=(stage, discharge, n_segments),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-12},
    )

    params = result_local.x if result_local.fun <= result_global.fun else result_global.x
    sse = sse_objective(params, stage, discharge, n_segments)
    residuals = log_residuals(params, stage, discharge, n_segments)
    n = len(discharge)

    sigma = math.sqrt(max(sse / n, 1e-30))
    log_likelihood = -0.5 * n * math.log(2.0 * math.pi) - n * math.log(sigma) - sse / (2.0 * sigma * sigma)

    k = 3 * n_segments + 1  # fitted curve parameters plus sigma
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = k * math.log(n) - 2.0 * log_likelihood
    rmse_log10 = math.sqrt(sse / n)
    cv_approx = math.log(10.0) * sigma

    q_pred = predict_discharge(params, stage, n_segments)
    ss_res_linear = float(np.sum((discharge - q_pred) ** 2))
    ss_tot_linear = float(np.sum((discharge - np.mean(discharge)) ** 2))
    r2_linear = 1.0 - ss_res_linear / ss_tot_linear if ss_tot_linear > 0 else float("nan")

    return {
        "n_segments": n_segments,
        "params": params,
        "param_names": parameter_names(n_segments),
        "sigma_log10": sigma,
        "cv_approx": cv_approx,
        "sse_log10": sse,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "rmse_log10": rmse_log10,
        "r2_linear": r2_linear,
        "residuals": residuals,
        "fitted": q_pred,
        "success": bool(result_local.success or result_global.success),
        "optimizer_message": str(result_local.message),
    }


def equation_text(fit: dict, digits: int = 6) -> str:
    params = fit["params"]
    n_segments = fit["n_segments"]
    parts: list[str] = []

    for seg in range(n_segments):
        h0 = params[3 * seg]
        alpha = 10.0 ** params[3 * seg + 1]
        beta = params[3 * seg + 2]
        term = f"{alpha:.{digits}g} * (h - {h0:.{digits}g})^{beta:.{digits}g}"
        if seg > 0:
            term += f" * I(h > {h0:.{digits}g})"
        parts.append(term)

    return "Q(h) = " + " + ".join(parts)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


def read_input_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [DATE_COLUMN, STAGE_COLUMN, DISCHARGE_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found columns: {list(df.columns)}")

    out = df[required].copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
    out[STAGE_COLUMN] = pd.to_numeric(out[STAGE_COLUMN], errors="coerce")
    out[DISCHARGE_COLUMN] = pd.to_numeric(out[DISCHARGE_COLUMN], errors="coerce")
    out = out.dropna()
    out = out[np.isfinite(out[STAGE_COLUMN]) & np.isfinite(out[DISCHARGE_COLUMN])]
    out = out[out[DISCHARGE_COLUMN] > 0.0]
    out = out.sort_values(DATE_COLUMN).reset_index(drop=True)

    if len(out) < 10:
        raise ValueError("At least 10 valid paired observations are required.")

    return out


def choose_best_model(fits: list[dict]) -> dict:
    # BIC is the primary default because it penalizes unnecessary extra segments strongly.
    return min(fits, key=lambda item: item["bic"])


def save_tables(out_dir: Path, data: pd.DataFrame, fits: list[dict], best: dict) -> None:
    data.to_csv(out_dir / "cleaned_data.csv", index=False)

    comparison_rows = []
    for fit in fits:
        comparison_rows.append(
            {
                "segments": fit["n_segments"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "rmse_log10": fit["rmse_log10"],
                "sigma_log10": fit["sigma_log10"],
                "cv_approx_fraction": fit["cv_approx"],
                "cv_approx_percent": 100.0 * fit["cv_approx"],
                "r2_linear": fit["r2_linear"],
                "log_likelihood": fit["log_likelihood"],
                "sse_log10": fit["sse_log10"],
                "success": fit["success"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(out_dir / "model_comparison.csv", index=False)

    param_rows = []
    for name, value in zip(best["param_names"], best["params"]):
        param_rows.append({"parameter": name, "value": value})
    param_rows.append({"parameter": "sigma_log10", "value": best["sigma_log10"]})
    param_rows.append({"parameter": "cv_approx_fraction", "value": best["cv_approx"]})
    pd.DataFrame(param_rows).to_csv(out_dir / "best_parameters.csv", index=False)

    fitted_df = data.copy()
    fitted_df["fitted_discharge"] = best["fitted"]
    fitted_df["residual_log10"] = best["residuals"]
    fitted_df["percent_error"] = 100.0 * (fitted_df["fitted_discharge"] - fitted_df[DISCHARGE_COLUMN]) / fitted_df[DISCHARGE_COLUMN]
    fitted_df.to_csv(out_dir / "fitted_values_and_residuals.csv", index=False)

    stages = np.linspace(float(data[STAGE_COLUMN].min()), float(data[STAGE_COLUMN].max()), RATING_TABLE_POINTS)
    rating_df = pd.DataFrame(
        {
            "stage": stages,
            "discharge": predict_discharge(best["params"], stages, best["n_segments"]),
        }
    )
    rating_df.to_csv(out_dir / "rating_table.csv", index=False)

    best_json = {
        "best_segments": best["n_segments"],
        "selection_rule": "minimum BIC",
        "equation": equation_text(best),
        "parameters": {name: float(value) for name, value in zip(best["param_names"], best["params"])},
        "sigma_log10": float(best["sigma_log10"]),
        "cv_approx_fraction": float(best["cv_approx"]),
        "aic": float(best["aic"]),
        "bic": float(best["bic"]),
        "rmse_log10": float(best["rmse_log10"]),
        "r2_linear": float(best["r2_linear"]),
    }
    (out_dir / "best_model.json").write_text(json.dumps(best_json, indent=2), encoding="utf-8")
    (out_dir / "equation.txt").write_text(equation_text(best) + "\n", encoding="utf-8")


def save_plots(out_dir: Path, data: pd.DataFrame, fits: list[dict], best: dict) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    stage = data[STAGE_COLUMN].to_numpy(dtype=float)
    discharge = data[DISCHARGE_COLUMN].to_numpy(dtype=float)
    grid = np.linspace(float(np.min(stage)), float(np.max(stage)), 300)

    plt.figure(figsize=(9, 6))
    plt.scatter(stage, discharge, s=28, color="#1f77b4", alpha=0.75, label="Observed")
    for fit in fits:
        q_grid = predict_discharge(fit["params"], grid, fit["n_segments"])
        lw = 2.8 if fit["n_segments"] == best["n_segments"] else 1.3
        alpha = 1.0 if fit["n_segments"] == best["n_segments"] else 0.55
        plt.plot(grid, q_grid, linewidth=lw, alpha=alpha, label=f"{fit['n_segments']} segment")
    plt.xlabel("Stage / water level")
    plt.ylabel("Discharge")
    plt.title("Rating Curve Fit")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "rating_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(stage, discharge, s=28, color="#1f77b4", alpha=0.75, label="Observed")
    q_grid = predict_discharge(best["params"], grid, best["n_segments"])
    plt.plot(grid, q_grid, linewidth=2.8, color="#d62728", label=f"Best: {best['n_segments']} segment")
    plt.yscale("log")
    plt.xlabel("Stage / water level")
    plt.ylabel("Discharge, log scale")
    plt.title("Rating Curve Fit, Log Discharge Scale")
    plt.legend()
    plt.grid(True, alpha=0.25, which="both")
    plt.tight_layout()
    plt.savefig(plots_dir / "rating_curve_log_scale.png", dpi=180)
    plt.close()

    residuals = best["residuals"]

    plt.figure(figsize=(9, 5))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.scatter(stage, residuals, s=28, color="#2ca02c", alpha=0.75)
    plt.xlabel("Stage / water level")
    plt.ylabel("Residual: log10(Q observed) - log10(Q fitted)")
    plt.title("Residuals vs Stage")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "residuals_vs_stage.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=min(25, max(8, len(residuals) // 4)), color="#9467bd", alpha=0.75, edgecolor="white")
    plt.xlabel("Residual in log10 discharge")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    plt.tight_layout()
    plt.savefig(plots_dir / "residual_histogram.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    probplot(residuals, dist="norm", plot=plt)
    plt.title("Residual Q-Q Plot")
    plt.tight_layout()
    plt.savefig(plots_dir / "residual_qq_plot.png", dpi=180)
    plt.close()

    comparison = pd.DataFrame(
        {
            "segments": [fit["n_segments"] for fit in fits],
            "AIC": [fit["aic"] for fit in fits],
            "BIC": [fit["bic"] for fit in fits],
        }
    )
    ax = comparison.plot(x="segments", y=["AIC", "BIC"], marker="o", figsize=(8, 5))
    ax.set_xlabel("Number of segments")
    ax.set_ylabel("Information criterion")
    ax.set_title("Model Comparison")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison.png", dpi=180)
    plt.close()


def save_report(out_dir: Path, input_path: Path, data: pd.DataFrame, fits: list[dict], best: dict) -> None:
    lines: list[str] = []
    lines.append("# Rating Curve Autofit Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Input CSV: `{input_path}`")
    lines.append(f"Valid observations used: {len(data)}")
    lines.append("")
    lines.append("## Best Model")
    lines.append("")
    lines.append(f"Selection rule: minimum BIC")
    lines.append(f"Best number of segments: **{best['n_segments']}**")
    lines.append("")
    lines.append("```text")
    lines.append(equation_text(best))
    lines.append("```")
    lines.append("")
    lines.append("## Best Parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---:|")
    for name, value in zip(best["param_names"], best["params"]):
        lines.append(f"| {name} | {value:.10g} |")
    lines.append(f"| sigma_log10 | {best['sigma_log10']:.10g} |")
    lines.append(f"| approximate CV | {100.0 * best['cv_approx']:.3f}% |")
    lines.append("")
    lines.append("## Model Comparison")
    lines.append("")
    lines.append("| Segments | AIC | BIC | RMSE log10 | Approx CV | R2 linear |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for fit in fits:
        lines.append(
            f"| {fit['n_segments']} | {fit['aic']:.4f} | {fit['bic']:.4f} | "
            f"{fit['rmse_log10']:.6f} | {100.0 * fit['cv_approx']:.3f}% | {fit['r2_linear']:.6f} |"
        )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `cleaned_data.csv`")
    lines.append("- `best_parameters.csv`")
    lines.append("- `best_model.json`")
    lines.append("- `equation.txt`")
    lines.append("- `model_comparison.csv`")
    lines.append("- `fitted_values_and_residuals.csv`")
    lines.append("- `rating_table.csv`")
    lines.append("- `plots/`")
    lines.append("")
    lines.append("## Method And Credit")
    lines.append("")
    lines.append("This independent script fits stage-discharge rating curves using a power-law")
    lines.append("addition-mode formulation inspired by published rating-curve methodology,")
    lines.append("RMC-BestFit technical documentation, and the BaRatin matrix-of-controls framework.")
    lines.append("It is not affiliated with or endorsed by USACE-RMC, IWR, ERDC-CHL, or BaRatin-tools.")
    lines.append("")
    lines.append("Primary sources to cite:")
    lines.append("")
    lines.append("- RMC-BestFit technical reference: https://github.com/USACE-RMC/RMC-BestFit/blob/main/docs/technical-reference/analysis/rating-curve.md")
    lines.append("- RMC-BestFit software page: https://www.rmc.usace.army.mil/Software/RMC-BestFit/")
    lines.append("- RMC-BestFit repository and 0BSD license: https://github.com/USACE-RMC/RMC-BestFit")
    lines.append("- BaRatin computational engine and GPL-3.0 license: https://github.com/BaRatin-tools/BaRatin")
    lines.append("- Le Coz et al. (2014), Journal of Hydrology, BaRatin Bayesian rating-curve method.")
    lines.append("- Rantz et al. (1982), USGS Water-Supply Paper 2175.")
    lines.append("- Kennedy (1984), USGS discharge ratings at gaging stations.")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple CSV-to-rating-curve autofit script.")
    parser.add_argument("csv", nargs="?", default=INPUT_CSV, help="Input CSV containing date, wl, discharge columns.")
    parser.add_argument("--out", default=OUTPUT_ROOT, help="Output root folder.")
    args = parser.parse_args()

    if not args.csv:
        raise SystemExit("Provide an input CSV path, or edit INPUT_CSV at the top of this script.")

    input_path = Path(args.csv).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    data = read_input_csv(input_path)
    stage = data[STAGE_COLUMN].to_numpy(dtype=float)
    discharge = data[DISCHARGE_COLUMN].to_numpy(dtype=float)

    max_segments = min(MAX_SEGMENTS, max(1, len(data) // 10), 3)
    fits = [fit_model(stage, discharge, n_segments) for n_segments in range(1, max_segments + 1)]
    best = choose_best_model(fits)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out).expanduser().resolve() / f"{input_path.stem}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_tables(out_dir, data, fits, best)
    save_plots(out_dir, data, fits, best)
    save_report(out_dir, input_path, data, fits, best)

    print("")
    print("Rating curve autofit complete.")
    print(f"Output folder: {out_dir}")
    print(f"Best model: {best['n_segments']} segment(s), selected by minimum BIC")
    print(equation_text(best))
    print("")


if __name__ == "__main__":
    main()
