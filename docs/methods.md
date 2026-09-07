# Comparing the two methods

The backends solve related problems with different model families, fitting losses, and uncertainty procedures. Keep their method names with exported results. Running both does not automatically make their reported scores directly comparable.

## At a glance

| Property | `additive` | `validated` |
|---|---|---|
| Curve family | Sum of 1–3 activated power laws. | 1–2 power-law regimes with a common zero-flow stage and a smooth transition. |
| Fitting loss | Squared residuals in `log10(Q)`. | Compares squared discharge loss and normalized `log1p` loss. |
| Discharge data | Strictly positive discharge; nonpositive values are excluded. | Nonnegative discharge; zeros retained, negative values rejected. |
| Minimum data | 20, 40, or 60 observations for 1, 2, or 3 terms. At least 10 above each later activation stage. | At least 20 observations plus viable validation splits; two regimes need at least 12 observations on each side in each applicable training fold by default. |
| Default shape | Monotone; curvature and slope at activation stages are unconstrained. | Monotone with matching first derivative at the transition. Optional `--shape convex`. |
| Selection | Lowest cross-validated log RMSE, preferring fewer terms within 3%; BIC fallback if CV is unavailable. | Inner held-out RMSE by default, preferring fewer regimes and no RF within 2%; optional normalized RMSLE selection. |
| Performance estimate | Candidate CV scores also used to select the model; not a nested estimate of selection performance. | Outer held-out predictions evaluate model, threshold, loss, and optional RF-weight selection. |
| Splits | Date blocks when dates are usable; otherwise stage-stratified folds. | Contiguous year groups, then date groups; input-row blocks only without dates. Same-day rows stay together. |
| Uncertainty | Pairs-bootstrap curve intervals and prediction intervals with simulated constant lognormal error. | Empirical error bands calibrated from scaled absolute nested-validation errors. |
| Rating table | Measured stage range only. | Measured plus supplied daily stage range; extrapolations flagged and error-band bounds omitted there. |
| Daily predictions | No daily CSV workflow. | Optional daily CSV, consistent daily stage changes, flags, and saved pipeline. |
| RF correction | None. | Optional residual learner, off by default. |

## Additive equation

For stage $h$, the static rating is

$$
Q_{\mathrm{median}}(h)=\sum_{j=1}^{K}\alpha_j\max(h-h_j,0)^{\beta_j},
\qquad K\in\{1,2,3\},\quad \alpha_j,\beta_j>0.
$$

The first zero-flow stage is below the minimum gauged stage. Later activation stages are ordered. Positive terms make the function nondecreasing, but an exponent below one can produce a steep or singular right derivative where a term activates. The fit does not require upward curvature or matching slopes at those stages.

The log-error formulation reports the equation as a conditional **median** rating. Under its constant-variance, zero-mean normal log-error assumption, the bias-corrected mean is

$$
Q_{\mathrm{mean}}=Q_{\mathrm{median}}
\exp\left[\tfrac12(\ln(10)\,\sigma_{\log_{10}Q})^2\right].
$$

That conversion depends on the assumed error distribution. It is not an independently calibrated mean-flow model. Exported in-sample and selection scores use the median curve.

## Smooth two-regime equation

For the `validated` backend, let $b$ be a zero-flow stage below the gauged range, $t>b$ a transition, and $a,p_L,p_H>0$. The lower branch is

$$
Q_L(h)=a\max(h-b,0)^{p_L}.
$$

Above the transition, the anchored branch is

$$
Q_H(h)=Q_L(t)\left[1+\frac{p_L}{p_H}
\left\{\left(\frac{h-b}{t-b}\right)^{p_H}-1\right\}\right].
$$

Both branches give $Q_L(t)$ at the transition and share the derivative $a p_L(t-b)^{p_L-1}$. The second derivative may change at the transition. The exponent bounds are 0.3–5 for `monotone` and 1–5 for `convex`.

An exponent above one gives upward curvature; one gives a straight branch; an exponent below one gives downward curvature. Upward curvature is a modeling constraint to justify using station information, not a universal requirement for every river. A danger-level column is a plotting reference, not evidence of a hydraulic transition.

Training-stage and discharge normalization are fitted separately in every training set. Loss selection compares linear squared error and squared `log1p(Q / scale)` error. The resulting estimate is not explicitly a lognormal conditional median or an analytically bias-corrected mean; interpret it with the selected loss and held-out residuals.

## What the validation supports

The additive method uses candidate CV to select term count. Its reported candidate score is not an independent outer estimate after that selection. BIC and AIC/AICc remain available as diagnostics; BIC is used for selection only when eligible candidates lack usable CV.

The validated method repeats all candidate selection inside each outer training set, then predicts the omitted outer group. Final parameters are fitted with all accepted observations after this evaluation. `model_comparison.csv` holds final inner selection scores; `validation_scores.csv` holds the outer performance estimate.

Both approaches may train on dates after an omitted period. They evaluate reconstruction across missing periods, **not future-only forecasting**. Neither cross-validation strategy establishes performance after an unobserved channel change.

To compare backends scientifically, use the same cleaned positive-flow observations, the same outer splits, and the same target and scoring metric. Fit and select each backend entirely inside those training splits. Comparing additive `log10` CV RMSE to validated discharge RMSE is not meaningful, and selecting whichever of two published scores looks better adds another selection step.

## What the intervals mean

For additive fits, bootstrap resampling refits the selected term count; it does not rerun model-count selection. Its curve interval describes variation in that fitted median curve under observation resampling. Its prediction interval adds sampled lognormal residual noise. Pairs resampling treats observations as exchangeable and does not preserve temporal dependence. Check the number of successful bootstrap fits and the error assumptions.

For validated fits, `Q_Lower` and `Q_Upper` are empirical bands with a default 90% target. They are calibrated from outer prediction errors, with separate base-curve calibration for the static rating and RF fallback rows. Coverage has not been independently validated, is not guaranteed under temporal change, and is unavailable beyond measured stages. These bands are not bootstrap parameter confidence intervals or Bayesian credible intervals.

## Common interpretation limits

- Use consistent units and a single stage datum. Changing a unit label does not transform the input numbers.
- Sparse high-stage observations limit what either method can learn; inspect their local coverage and held-out errors.
- A fitted transition is empirical evidence of a change in curve shape, not identification of its physical cause.
- A static stage rating cannot capture all backwater, tidal/reversing flow, control shifts, or flood-wave hysteresis. Separate rating periods or additional hydraulic measurements may be needed.
- Discharge evaluated at daily average stage is generally not the daily average discharge of a nonlinear rating.
- Synthetic examples validate software behavior, not hydrological accuracy at a real station.

See [sources and credit](../SOURCES_AND_CREDIT.md) for the rating-curve and validation references.
