"""Reusable CSV rating-curve fitting, nested validation and daily prediction.

Run: rating-curve validated measured.csv --daily daily_stage.csv
See docs/validated.md for settings, assumptions and output definitions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .core import fit_curve, predict_curve, equation_text
from .io import load_inputs
from .provenance import runtime_versions

VERSION = '1.0.0'


@dataclass
class Settings:
    shape: str = 'monotone'
    max_segments: int = 2
    min_regime: int = 12
    threshold: float | None = None
    folds: int = 5
    inner_folds: int = 3
    selection_metric: str = 'RMSE'
    complexity_tolerance: float = .02
    rf: bool = False
    seed: int = 42
    coverage: float = .90


def metrics(observed, predicted):
    y, p = np.asarray(observed, float), np.asarray(predicted, float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape:
        raise ValueError('Observed and predicted scores require one-dimensional arrays of identical length.')
    if np.any(y < 0) or np.any(p < 0):
        raise ValueError('Rating-curve scores require nonnegative observations and predictions.')
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.all():
        raise ValueError('Scoring requires a finite prediction for every evaluation row.')
    if not len(y):
        return {'N': 0, **{k: None for k in ['RMSE', 'MAE', 'Bias', 'NSE', 'KGE', 'RMSLE']}}
    e = p-y
    denom = np.sum((y-y.mean())**2)
    kge = None
    if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 and y.mean() > 0:
        r = np.corrcoef(y, p)[0, 1]
        kge = float(1-np.sqrt((r-1)**2+(np.std(p)/np.std(y)-1)**2+(p.mean()/y.mean()-1)**2))
    log_scale = float(np.quantile(y, .9))
    if log_scale <= 0:
        log_scale = max(float(np.max(y)), float(np.max(p)), np.finfo(float).tiny)
    return {'N': len(y), 'RMSE': float(np.sqrt(np.mean(e**2))),
            'MAE': float(np.mean(np.abs(e))), 'Bias': float(e.mean()),
            'NSE': float(1-np.sum(e**2)/denom) if denom > 0 else None,
            'KGE': kge, 'RMSLE': float(np.sqrt(np.mean((np.log1p(p/log_scale)-np.log1p(y/log_scale))**2)))}


def make_folds(frame, n_folds=5, minimum_train=8):
    """Hold out contiguous year groups, then date groups, then input-row blocks.

    This estimates reconstruction across held-out periods, not future-only skill.
    Same-date observations never straddle train and test sets.
    """
    if len(frame) < minimum_train+2:
        raise ValueError(f'Need at least {minimum_train+2} valid observations for validation.')
    dates = pd.to_datetime(frame['Date'])
    if dates.notna().all() and dates.dt.year.nunique() >= 2:
        groups = dates.dt.year.to_numpy()
        strategy = 'contiguous year groups'
    elif dates.notna().all() and dates.nunique() >= 2:
        groups = dates.to_numpy()
        strategy = 'contiguous date groups'
    elif dates.isna().all():
        groups = np.arange(len(frame))
        strategy = 'input-row blocks (no dates; temporal independence unknown)'
    else:
        raise ValueError('Partial/missing dates cannot be mixed with dated validation records.')
    unique = np.unique(groups)
    n = min(n_folds, len(unique))
    if n < 2:
        raise ValueError('Need at least two distinct date groups for validation.')
    folds = []
    for block in np.array_split(unique, n):
        test = np.flatnonzero(np.isin(groups, block))
        train = np.flatnonzero(~np.isin(groups, block))
        if len(train) < minimum_train:
            raise ValueError('Too few training rows in a date/year group split. Add observations or use more folds; dates are never split across groups.')
        folds.append((train, test))
    return folds, strategy


def rf_fit(frame, base, seed):
    """Optional true residual learner. Fit only on training-fold data."""
    from sklearn.ensemble import RandomForestRegressor
    good = np.isfinite(frame['dWL'].to_numpy(float))
    if good.sum() < 24 or good.mean() < .7:
        raise ValueError('Residual RF needs >=24 training rows and >=70% coverage of daily dWL.')
    x = frame.loc[good, ['WL', 'dWL']].to_numpy(float)
    target = frame.loc[good, 'Q'].to_numpy(float)-predict_curve(base, x[:, 0])
    rf = RandomForestRegressor(n_estimators=160, max_depth=5,
                               min_samples_leaf=max(4, int(good.sum()*.04)),
                               n_jobs=-1, random_state=seed)
    rf.fit(x, target)
    return {'estimator': rf, 'h_min': float(x[:, 0].min()), 'h_max': float(x[:, 0].max()),
            'd_min': float(x[:, 1].min()), 'd_max': float(x[:, 1].max())}


def residual_correction(rf, frame):
    result = np.zeros(len(frame), float)
    if rf is None or not len(frame):
        return result
    x = frame[['WL', 'dWL']].to_numpy(float)
    good = (np.isfinite(x).all(axis=1) & (x[:, 0] >= rf['h_min']) &
            (x[:, 0] <= rf['h_max']) & (x[:, 1] >= rf['d_min']) & (x[:, 1] <= rf['d_max']))
    if good.any():
        # Fade to zero near gauged stage boundaries to avoid an edge jump.
        span = max(rf['h_max']-rf['h_min'], np.finfo(float).eps)
        taper = np.clip(np.minimum(x[good, 0]-rf['h_min'], rf['h_max']-x[good, 0])/(.1*span), 0, 1)
        result[good] = rf['estimator'].predict(x[good])*taper
    return result


def predict_pipeline(pipeline, frame):
    base = predict_curve(pipeline['base'], frame['WL'].to_numpy(float))
    pred = np.maximum(base+pipeline['alpha']*residual_correction(pipeline.get('rf'), frame), 0)
    if not np.isfinite(pred).all():
        raise ValueError('Prediction overflow/nonfinite value; check stage units and extrapolation range.')
    return pred


def select_pipeline(frame, settings, folds_count):
    folds, strategy = make_folds(frame, folds_count)
    rows, predictions = [], {}
    candidates = [(s, loss) for s in range(1, settings.max_segments+1) for loss in ['linear', 'log1p']]
    for segments, loss in candidates:
        key = f'{segments}regime_{loss}'
        oof, correction = np.full(len(frame), np.nan), np.zeros(len(frame))
        failure, rf_failure = [], []
        for fold_no, (train, test) in enumerate(folds, 1):
            tr, te = frame.iloc[train], frame.iloc[test]
            try:
                base = fit_curve(tr.WL.to_numpy(), tr.Q.to_numpy(), segments=segments,
                                 loss=loss, shape=settings.shape, min_regime=settings.min_regime,
                                 fixed_threshold=settings.threshold if segments == 2 else None)
                oof[test] = predict_curve(base, te.WL.to_numpy())
                if not np.isfinite(oof[test]).all():
                    raise ValueError('nonfinite validation prediction')
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                failure.append(f'fold {fold_no}: {exc}')
                continue
            if settings.rf:
                try:
                    rf = rf_fit(tr, base, settings.seed)
                    correction[test] = residual_correction(rf, te)
                except (ValueError, RuntimeError) as exc:
                    rf_failure.append(f'fold {fold_no}: {exc}')
        if failure:
            rows.append({'Candidate': key, 'Segments': segments, 'Loss': loss, 'Alpha': 0.,
                         'Status': 'ineligible', 'Failure': '; '.join(failure),
                         'N': int(np.isfinite(oof).sum())})
            continue
        alphas = [0., .25, .5, 1.] if settings.rf and not rf_failure else [0.]
        for alpha in alphas:
            name = key if alpha == 0 else f'{key}_residualRF_{alpha:g}'
            pred = np.maximum(oof+alpha*correction, 0)
            row = {'Candidate': name, 'Segments': segments, 'Loss': loss, 'Alpha': alpha,
                   'Status': 'eligible', 'Failure': '', **metrics(frame.Q, pred)}
            rows.append(row)
            predictions[name] = pred
        if rf_failure:
            rows.append({'Candidate': key+'_residualRF', 'Segments': segments, 'Loss': loss,
                         'Alpha': None, 'Status': 'ineligible', 'Failure': '; '.join(rf_failure), 'N': 0})
    eligible = [r for r in rows if r['Status'] == 'eligible']
    if not eligible:
        raise ValueError('No candidate passed all validation folds: '+str(rows))
    best_score = min(r[settings.selection_metric] for r in eligible)
    close = [r for r in eligible if r[settings.selection_metric] <= best_score*(1+settings.complexity_tolerance)+1e-12]
    chosen = min(close, key=lambda r: (r['Segments'], r['Alpha'] > 0, r[settings.selection_metric]))
    # Refit selected pipeline with the entire training set. No outer test rows here.
    base = fit_curve(frame.WL.to_numpy(), frame.Q.to_numpy(), segments=chosen['Segments'],
                     loss=chosen['Loss'], shape=settings.shape, min_regime=settings.min_regime,
                     fixed_threshold=settings.threshold if chosen['Segments'] == 2 else None)
    rf = rf_fit(frame, base, settings.seed) if chosen['Alpha'] > 0 else None
    return {'base': base, 'alpha': chosen['Alpha'], 'rf': rf, 'candidate': chosen['Candidate']}, rows, strategy


def nested_validation(frame, settings):
    folds, strategy = make_folds(frame, settings.folds, minimum_train=12)
    oof, base_oof = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    fold_ids = np.zeros(len(frame), int)
    records = []
    for fold_no, (train, test) in enumerate(folds, 1):
        print(f'Validating held-out period {fold_no}/{len(folds)} ...', flush=True)
        pipeline, _, inner_strategy = select_pipeline(frame.iloc[train].reset_index(drop=True), settings, settings.inner_folds)
        oof[test] = predict_pipeline(pipeline, frame.iloc[test])
        base_oof[test] = predict_curve(pipeline['base'], frame.iloc[test].WL.to_numpy())
        fold_ids[test] = fold_no
        records.append({'Fold': fold_no, 'Training_N': len(train), 'Test_N': len(test),
                        'Selected': pipeline['candidate'], 'Threshold': pipeline['base']['threshold'],
                        'Test_start': str(frame.iloc[test].Date.min()), 'Test_end': str(frame.iloc[test].Date.max()),
                        'Inner_strategy': inner_strategy, **metrics(frame.iloc[test].Q, oof[test])})
    metrics(frame.Q, oof)  # fail instead of silently dropping missing/failed folds
    return oof, base_oof, fold_ids, records, strategy


def calibration(frame, oof, coverage):
    qref = max(float(np.median(frame.Q[frame.Q > 0]))*.1, float(frame.Q.max())*1e-6)
    residual = np.abs(frame.Q.to_numpy()-oof)/(oof+qref)
    radius = float(np.quantile(residual, coverage, method='higher'))
    return {'coverage_target': coverage, 'relative_radius': radius, 'q_reference': qref,
            'method': 'Empirical absolute scaled errors from nested out-of-fold predictions; not a guaranteed coverage interval.',
            'n': len(oof)}


def prediction_table(frame, pipeline, cal, observed):
    result = frame.copy()
    h = result.WL.to_numpy(float)
    result['Q_RatingCurve'] = predict_curve(pipeline['base'], h)
    result['Q_Estimate'] = predict_pipeline(pipeline, result)
    low, high = float(observed.WL.min()), float(observed.WL.max())
    result['Stage_extrapolation'] = np.where(h < low, 'below_measured_range', np.where(h > high, 'above_measured_range', 'within_measured_range'))
    result['dWL_available'] = np.isfinite(result.dWL)
    span = max(high-low, np.finfo(float).eps)
    hs = np.sort(observed.WL.to_numpy(float))
    result['Local_observation_count'] = np.searchsorted(hs, h+.05*span, side='right')-np.searchsorted(hs, h-.05*span, side='left')
    base_cal = cal.get('base_curve', cal)
    radius = np.full(len(result), base_cal['relative_radius'])
    qref = np.full(len(result), base_cal['q_reference'])
    if pipeline['alpha']:
        rf = pipeline['rf']
        valid_rf = np.isfinite(result.dWL) & result.WL.between(rf['h_min'], rf['h_max']) & result.dWL.between(rf['d_min'], rf['d_max'])
        radius[valid_rf] = cal['relative_radius']
        qref[valid_rf] = cal['q_reference']
    half = radius*(result.Q_Estimate+qref)
    result['Q_Lower'] = np.maximum(0, result.Q_Estimate-half)
    result['Q_Upper'] = result.Q_Estimate+half
    outside = (h < low) | (h > high)
    result.loc[outside, ['Q_Lower', 'Q_Upper']] = np.nan
    result['Interval_status'] = np.where(outside, 'unavailable_extrapolation', 'empirical_error_band')
    if observed.Date.notna().all() and result.Date.notna().all():
        result['Date_outside_gauging_period'] = (result.Date < observed.Date.min()) | (result.Date > observed.Date.max())
    if pipeline['alpha']:
        rf = pipeline['rf']
        result['Predictor'] = np.where(valid_rf, 'rating_plus_residual_RF', 'rating_only_RF_inputs_unavailable_or_outside_support')
    else:
        result['Predictor'] = 'rating_curve'
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_model(path, pipeline, cal, metadata):
    payload = {'version': VERSION, 'base': pipeline['base'], 'alpha': pipeline['alpha'],
               'candidate': pipeline['candidate'], 'calibration': cal, 'metadata': metadata, 'rf_file': None}
    if pipeline['rf'] is not None:
        import joblib
        rf_path = path.with_name('residual_rf.joblib')
        joblib.dump(pipeline['rf'], rf_path)
        payload['rf_file'] = rf_path.name
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding='utf-8')


def load_model(path):
    """Load a model created by this program. Load only trusted RF joblib files."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['rf'] = None
    if payload.get('rf_file'):
        import joblib
        payload['rf'] = joblib.load(path.parent/payload['rf_file'])
    return payload


def plots(out, observed, daily, rating, oof, pipeline, stage_unit, discharge_unit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.fill_between(rating.WL, rating.Q_Lower, rating.Q_Upper, color='#159a9c', alpha=.16,
                    label='Empirical error band from held-out predictions')
    ax.scatter(observed.WL, observed.Q, color='#64748b', s=22, alpha=.6, label='Measured discharge', zorder=3)
    ax.plot(rating.WL, rating.Q_RatingCurve, color='#007a78', lw=2.2, label='Selected stage-only rating')
    threshold = pipeline['base']['threshold']
    if threshold is not None:
        ax.axvline(threshold, ls='--', color='#8064a2', label=f'Fitted transition: {threshold:.3g}')
    if 'DL' in observed and observed.DL.notna().any():
        dl = observed.DL.dropna().unique()
        if len(dl) == 1:
            ax.axvline(float(dl[0]), ls=':', color='#b7791f', label=f'DL: {dl[0]:g} (reference)')
    xmin, xmax = observed.WL.min(), observed.WL.max()
    shaded = False
    for a, b in [(rating.WL.min(), xmin), (xmax, rating.WL.max())]:
        if b > a:
            ax.axvspan(a, b, color='#e2e8f0', alpha=.7,
                       label='Outside measured stage range' if not shaded else None)
            shaded = True
    ymax = max(float(observed.Q.max())*1.15, float(rating.Q_RatingCurve.max())*1.08)
    ax.set(xlabel=f'Water level ({stage_unit})', ylabel=f'Discharge ({discharge_unit})',
           title=f'Rating curve with {pipeline["base"]["segments"]} regime'+('s' if pipeline['base']['segments'] > 1 else ''), ylim=(0, ymax))
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='upper left', fontsize=8)
    fig.savefig(out/'rating_curve.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(observed.Q, oof, s=22, color='#007a78', alpha=.6)
    lim = max(observed.Q.max(), oof.max())*1.03
    axes[0].plot([0, lim], [0, lim], color='#475569', ls='--')
    axes[0].set(xlabel=f'Measured discharge ({discharge_unit})', ylabel=f'Held-out prediction ({discharge_unit})', title='Nested validation')
    residual = observed.Q.to_numpy()-oof
    dates = observed.Date
    x = dates if dates.notna().all() else observed.WL
    points = axes[1].scatter(x, residual, c=observed.WL, cmap='viridis', s=22, alpha=.8)
    fig.colorbar(points, ax=axes[1], label=f'Water level ({stage_unit})', shrink=.8)
    axes[1].axhline(0, color='#475569', ls='--')
    axes[1].set(xlabel='Date' if dates.notna().all() else f'Water level ({stage_unit})', ylabel=f'Measured minus predicted ({discharge_unit})', title='Held-out residuals')
    fig.savefig(out/'validation_diagnostics.png', dpi=180)
    plt.close(fig)
    if daily is not None and len(daily):
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        ax.plot(daily.Date, daily.Q_Estimate, color='#007a78', lw=.8, label='Estimated daily discharge')
        ax.scatter(observed.Date, observed.Q, color='#475569', s=10, alpha=.6, label='Measurements')
        ax.set(xlabel='Date', ylabel=f'Discharge ({discharge_unit})', title='Daily discharge estimates')
        ax.legend(fontsize=8)
        fig.savefig(out/'daily_discharge.png', dpi=180)
        plt.close(fig)


def run(args):
    settings = Settings(shape=args.shape, max_segments=args.max_segments, min_regime=args.min_regime,
                        threshold=args.threshold, folds=args.folds, inner_folds=args.inner_folds,
                        selection_metric=args.selection_metric, rf=args.rf, seed=args.seed, coverage=args.coverage)
    if args.rf:
        try:
            import sklearn  # noqa: F401
        except ImportError as exc:
            raise ValueError('Optional RF requires scikit-learn. Install it or run without --rf.') from exc
    loaded = load_inputs(args.observations, args.daily, date_col=args.date_col,
                         stage_col=args.stage_col, discharge_col=args.discharge_col,
                         daily_date_col=args.daily_date_col, daily_stage_col=args.daily_stage_col,
                         date_format=args.date_format, dayfirst=args.dayfirst)
    obs, daily = loaded['observations'], loaded['daily']
    obs = obs.reset_index(drop=True)
    if len(obs) < 20:
        raise ValueError(f'Only {len(obs)} valid measurements remain. At least 20 are required for nested model selection; more are needed for two regimes.')
    if obs.Q.max() <= 0 or obs.Q.nunique() < 2 or obs.WL.nunique() < 5:
        raise ValueError('Need varying discharge with positive values and at least five distinct water levels.')
    if args.rf and daily is None:
        raise ValueError('--rf requires a daily WL file so training and prediction share the same daily dWL feature.')
    if daily is not None and daily.empty:
        raise ValueError('The daily CSV has no valid rows after parsing dates and water levels. Check column mapping and date format.')
    print(f'Loaded {len(obs)} discharge measurements'+(f' and {len(daily)} daily stages.' if daily is not None else '.'), flush=True)
    oof, base_oof, fold_ids, fold_rows, strategy = nested_validation(obs, settings)
    print('Selecting and fitting the final model using all observations ...', flush=True)
    pipeline, comparison, _ = select_pipeline(obs, settings, settings.inner_folds)
    cal = calibration(obs, oof, settings.coverage)
    cal['base_curve'] = calibration(obs, base_oof, settings.coverage)
    notes = list(loaded['notes'])
    notes.extend(pipeline['base'].get('warnings', []))
    if pipeline['alpha']:
        notes.append('The optional residual RF is a time-varying predictor; monotonicity is guaranteed only for the stage-only rating. Its plot excludes RF corrections.')
    if obs.Date.isna().all():
        notes.append('No measurement dates supplied: validation uses file-row blocks and cannot diagnose rating changes over time.')
    notes.append('This is a single-site, nonnegative-discharge rating. Tidal/reversing flow or changing backwater controls may require another model or separate rating periods.')
    score_rows = []
    top = obs.WL >= obs.WL.quantile(.75)
    for label, mask in [('All', np.ones(len(obs), bool)), ('Upper_stage_quartile', top.to_numpy())]:
        score_rows.append({'Evaluation': 'Nested_CV', 'Subset': label, **metrics(obs.Q[mask], oof[mask])})
    full_score = score_rows[0]
    if full_score['NSE'] is not None and full_score['NSE'] < .5:
        notes.append(f'Limited held-out skill: overall NSE={full_score["NSE"]:.3f}. Review temporal shifts and measurement quality before relying on estimates.')
    if score_rows[1]['NSE'] is not None and score_rows[1]['NSE'] < 0:
        notes.append('Upper-stage held-out NSE is negative; high-flow estimates need particular review.')
    out = Path(args.out or Path(args.observations).parent/'universal_rating_results')
    out = out/(Path(args.observations).stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    clean = prediction_table(obs, pipeline, cal, obs)
    clean['Q_NestedCV'] = oof
    clean['Q_Base_NestedCV'] = base_oof
    clean['CV_fold'] = fold_ids
    clean['CV_residual'] = obs.Q.to_numpy()-oof
    clean.to_csv(out/'measurement_predictions.csv', index=False)
    obs.to_csv(out/'cleaned_measurements.csv', index=False)
    loaded['rejected_observations'].to_csv(out/'rejected_measurements.csv', index=False)
    loaded['rejected_daily'].to_csv(out/'rejected_daily_rows.csv', index=False)
    pd.DataFrame(comparison).to_csv(out/'model_comparison.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out/'validation_folds.csv', index=False)
    pd.DataFrame(score_rows).to_csv(out/'validation_scores.csv', index=False)
    by_year = []
    if obs.Date.notna().all():
        for year in sorted(obs.Date.dt.year.unique()):
            mask = obs.Date.dt.year == year
            by_year.append({'Year': int(year), **metrics(obs.Q[mask], oof[mask])})
        pd.DataFrame(by_year).to_csv(out/'validation_by_year.csv', index=False)
    daily_out = prediction_table(daily, pipeline, cal, obs) if daily is not None else None
    if daily_out is not None:
        daily_out.to_csv(out/'daily_discharge_calculated.csv', index=False)
        count = int((daily_out.Stage_extrapolation != 'within_measured_range').sum())
        notes.append(f'{count} daily rows lie outside measured stage range; their error-band bounds are unavailable.')
    hmin = min(obs.WL.min(), daily.WL.min()) if daily is not None else obs.WL.min()
    hmax = max(obs.WL.max(), daily.WL.max()) if daily is not None else obs.WL.max()
    grid = pd.DataFrame({'Date': pd.NaT, 'WL': np.linspace(hmin, hmax, 400), 'dWL': np.nan})
    rating = prediction_table(grid, pipeline, cal, obs)
    rating.to_csv(out/'rating_table.csv', index=False)
    paths = [Path(args.observations)]+([Path(args.daily)] if args.daily else [])
    metadata = {'settings': asdict(settings), 'input_columns': loaded['columns'],
                'inputs': [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
                'stage_unit': args.stage_unit, 'discharge_unit': args.discharge_unit,
                'stage_min': float(obs.WL.min()), 'stage_max': float(obs.WL.max()),
                'validation_strategy': strategy, 'notes': notes,
                'versions': runtime_versions(include_rf=settings.rf)}
    save_model(out/'model.json', pipeline, cal, metadata)
    # Prove the serialized complete pipeline predicts identically.
    restored = load_model(out/'model.json')
    if not np.allclose(predict_pipeline(restored, obs), clean.Q_Estimate, rtol=1e-12, atol=1e-12):
        raise RuntimeError('Saved model round-trip prediction verification failed.')
    equation = equation_text(pipeline['base'])
    (out/'equation.txt').write_text(equation, encoding='utf-8')
    plots(out, obs, daily_out, rating, oof, pipeline, args.stage_unit, args.discharge_unit)
    report = [f'# Rating curve: {Path(args.observations).stem}', '',
              f'Selected model: **{pipeline["candidate"]}**. Shape constraint: {settings.shape}.', '',
              f'{len(obs)} valid measurements. Measured stage range: {obs.WL.min():.6g} to {obs.WL.max():.6g} {args.stage_unit}.', '',
              '## Equation', '', '```text', equation, '```', '', '## Validation', '',
              f'Nested validation uses {strategy}. Every outer test period is excluded from model, threshold, loss and optional RF-weight selection. This evaluates reconstruction across periods, not future-only forecasting.', '',
              f'Overall held-out RMSE: {full_score["RMSE"]:.3f} {args.discharge_unit}; MAE: {full_score["MAE"]:.3f}; NSE: {full_score["NSE"]}.', '',
              '`model_comparison.csv` contains inner model-selection scores. Use `validation_scores.csv` for complete-pipeline held-out performance. R² is omitted because its residual definition duplicates NSE. RMSLE uses log1p(Q / scale), where scale is the 90th-percentile observed discharge in the scored set, so changing discharge units does not change the score.', '',
              '## Uncertainty and predictions', '',
              f'Q_Lower/Q_Upper are an empirical {settings.coverage:.0%} error band calibrated from scaled, absolute nested-CV errors. Coverage is approximate, has not been independently validated, and is not guaranteed under temporal change. Bounds are left blank outside measured stage range.', '',
              'Q_RatingCurve is the monotone stage-only curve. Q_Estimate additionally uses the selected residual RF, if any. Missing/out-of-support RF features fall back to the rating curve. The stage-only table and RF fallback rows use errors from base-only outer predictions to calibrate their bands. Local_observation_count counts measurements within 5% of the measured stage span.', '',
              'If WL is a daily average, Q(WL) is the discharge evaluated at that average stage; it is not necessarily the true daily mean discharge. Instantaneous/subdaily stage is preferable when within-day variability is large.', '',
              '## Notes', '', *['- '+str(n) for n in notes], '',
              '## Method references', '',
              '- [USGS rating-curve background](https://thodson-usgs.github.io/ratingcurve/meta/background.html)',
              '- [WMO hydrological monitoring guidance](https://wmo.int/media/magazine-article/5-essential-elements-of-hydrological-monitoring-programme)',
              '- [scikit-learn nested validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)', '']
    (out/'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'Selected: {pipeline["candidate"]} | nested RMSE={full_score["RMSE"]:.3f} | NSE={full_score["NSE"]}', flush=True)
    print(f'Output: {out.resolve()}', flush=True)
    return out


def parser():
    p = argparse.ArgumentParser(prog='rating-curve validated', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('observations', help='CSV with paired water levels and measured discharge')
    p.add_argument('--daily', help='Optional daily water-level CSV')
    p.add_argument('--out', help='Parent output directory; each run creates a unique subfolder')
    for flag in ['date-col', 'stage-col', 'discharge-col', 'daily-date-col', 'daily-stage-col', 'date-format']:
        p.add_argument('--'+flag)
    p.add_argument('--dayfirst', action='store_true', help='Interpret ambiguous slash dates as day/month/year')
    p.add_argument('--shape', choices=['monotone', 'convex'], default='monotone')
    p.add_argument('--max-segments', type=int, choices=[1, 2], default=2)
    p.add_argument('--min-regime', type=int, default=12)
    p.add_argument('--threshold', type=float, help='Optional hydraulically justified fixed transition; DL is never used automatically')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--inner-folds', type=int, default=3)
    p.add_argument('--selection-metric', choices=['RMSE', 'RMSLE'], default='RMSE')
    p.add_argument('--rf', action='store_true', help='Compare optional RF residual corrections using daily dWL; requires scikit-learn')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--coverage', type=float, default=.90)
    p.add_argument('--stage-unit', default='m', help='Plot label only; does not convert units')
    p.add_argument('--discharge-unit', default='m3/s', help='Plot label only; does not convert units')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.folds < 2 or args.inner_folds < 2 or args.min_regime < 5:
        p.error('folds and inner-folds must be >=2; min-regime must be >=5')
    if not .5 < args.coverage < 1:
        p.error('coverage must be between 0.5 and 1 (exclusive)')
    if args.threshold is not None and args.max_segments != 2:
        p.error('--threshold requires --max-segments 2')
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        p.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
