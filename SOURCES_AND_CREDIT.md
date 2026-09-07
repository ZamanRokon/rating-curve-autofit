# Sources and credit

Rating Curve Autofit is an independent Python project for empirical stage–discharge fitting. The repository's original additive implementation is retained as the `additive` backend; the smooth rating and daily-prediction workflow is provided as `validated`.

This project is not affiliated with or endorsed by USACE-RMC, IWR, ERDC-CHL, or BaRatin-tools. Neither backend implements Bayesian BaRatin inference or automatically identifies physical hydraulic controls.

## Original additive method

The original repository credited these published methods and public references. That attribution is retained:

- [RMC-BestFit rating-curve technical reference](https://github.com/USACE-RMC/RMC-BestFit/blob/main/docs/technical-reference/analysis/rating-curve.md).
- [RMC-BestFit repository](https://github.com/USACE-RMC/RMC-BestFit).
- [RMC-BestFit software page](https://www.rmc.usace.army.mil/Software/RMC-BestFit/).
- [BaRatin computational engine](https://github.com/BaRatin-tools/BaRatin).
- [BaRatin rating-curve Fortran source](https://github.com/BaRatin-tools/BaRatin/blob/main/src/RatingCurve_tools.f90).
- Le Coz, J., Renard, B., Bonnifait, L., Branger, F., and Le Boursicaud, R. (2014). *Combining hydraulic knowledge and uncertain gaugings in the estimation of hydrometric rating curves: A Bayesian approach.* Journal of Hydrology.
- Rantz, S. E., et al. (1982). *Measurement and computation of streamflow, Volume 2: Computation of discharge.* USGS Water-Supply Paper 2175.
- Kennedy, E. J. (1984). *Discharge ratings at gaging stations.* USGS Techniques of Water-Resources Investigations, Book 3, Chapter A10.

The references provide hydraulic and statistical context. The presence of a reference does not mean this software reproduces its full model, uncertainty treatment, or validation procedure.

## Smooth rating and validation workflow

The `validated` implementation uses a locally implemented anchored power-law equation whose value and first derivative match at the transition. Its [equation and assumptions](docs/methods.md) are documented explicitly. Its residual forest is optional and uses scikit-learn; its error bands are empirical calibration bands, not a reproduction of BaRatin uncertainty.

Relevant background and implementation references:

- [USGS rating-curve background](https://thodson-usgs.github.io/ratingcurve/meta/background.html): stage–discharge power laws and hydraulic context.
- [WMO hydrological monitoring guidance](https://wmo.int/media/magazine-article/5-essential-elements-of-hydrological-monitoring-programme): measurement quality and hydraulic information in monitoring.
- [scikit-learn nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): separation of parameter selection from performance assessment.
- [scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html): the optional residual learner.
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html): bounded nonlinear least-squares fitting.

## Software and data

The package uses NumPy, pandas, SciPy, and Matplotlib. The optional RF extra uses scikit-learn and joblib. Each dependency retains its own license.

Files under `examples/` are synthetic demonstrations. They do not contain the station measurements used during development. Generated outputs and private station data are not part of the package's scientific reference material.

## License and provenance

The repository retains its original [MIT license](LICENSE). The original attribution identified RMC-BestFit as 0BSD and BaRatin as GPL-3.0; consult those projects' own license files for their terms. Referencing their published equations is distinct from copying their source. Any contribution that copies or translates external implementation must record its origin and address the applicable license before inclusion.
