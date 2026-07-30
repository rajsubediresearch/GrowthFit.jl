# GrowthFit.jl

[![CI](https://github.com/rajsubediresearch/GrowthFit.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/rajsubediresearch/GrowthFit.jl/actions/workflows/CI.yml)
[![DOI](https://zenodo.org/badge/1313363249.svg)](https://doi.org/10.5281/zenodo.21696876)

Phenomenological growth-model fitting, uncertainty quantification, and
short-term forecasting for epidemic incidence data in Julia.

A Julia port of the R and MATLAB toolboxes of the same design, with the
numerical fixes from those ports carried across: SLSQP rather than plain
gradient descent, data-scaled bounds on the carrying capacity `K`, and a
sanity guard on implausibly-scaled trajectories under negative-binomial
error.

## Installation

```julia
using Pkg
Pkg.add("GrowthFit")
```

## Models

Seven nested phenomenological growth models, fitted to cumulative incidence
and reported as weekly incidence. `x` is cumulative cases, `r` the growth
rate, `p` the deceleration-of-growth parameter, `a` a shape parameter, and
`K` the final epidemic size.

| `flag`  | Model                             | ODE                        | Free parameters |
| ------- | --------------------------------- | -------------------------- | --------------- |
| `:exp`  | Exponential Growth Model (EXP)    | `dx = r·x`                 | `r`             |
| `:ggm`  | Generalized Growth Model (GGM)    | `dx = r·x^p`               | `r, p`          |
| `:lm`   | Logistic Model (LM)               | `dx = r·x·(1 − x/K)`       | `r, K`          |
| `:glm`  | Generalized Logistic Model (GLM)  | `dx = r·x^p·(1 − x/K)`     | `r, p, K`       |
| `:rich` | Richards Model (RICH)             | `dx = r·x·(1 − (x/K)^a)`   | `r, a, K`       |
| `:grm`  | Generalized Richards Model (GRM)  | `dx = r·x^p·(1 − (x/K)^a)` | `r, p, a, K`    |
| `:gom`  | Gompertz Model (GOM)              | `dx = r·x·exp(−a·t)`       | `r, a`          |

Note that GLM here is the Generalized *Logistic* Model, not the generalized
linear model of `GLM.jl`.

Error structures: `:normal` (least squares), `:poisson`, `:nb1`
(negative binomial with `Var = μ + α·μ`).

## Usage

```julia
using GrowthFit

timevect = collect(1.0:40.0)
ydata    = weekly_incidence          # your data

# Optional pre-screen: is this model even identifiable on this window?
screen_all_models(timevect, ydata)

# Point estimate
res = fit_growth_model(:grm, timevect, ydata; dist = :nb1)

# Parametric-bootstrap uncertainty + 3-step-ahead forecast
boot = run_bootstrap(:grm, timevect, ydata, res;
                     dist = :nb1, M = 300, forecast_horizon = 3, seed = 20260726)

# Fit/forecast panel with parameter histograms
plt = plot_fit_and_forecast(boot, timevect, ydata, ydata_all, timevect_all;
                            forecast_horizon = 3)
savefig(plt, "fit_forecast.png")
```

See [`examples/run_example.jl`](examples/run_example.jl) for a complete
worked example on measles incidence data, including performance metrics
(MAE, MSE, 95% PI coverage, WIS) and CSV output.

To run it:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/run_example.jl
```

The bootstrap is multithreaded; start Julia with `-t 8` (or set
`JULIA_NUM_THREADS`) to use it.

## Reproducibility

`run_bootstrap` derives one random stream per replicate index from a single
master `seed`, so results depend only on that seed and **not** on
`Threads.nthreads()`. Pass an explicit `seed` when reporting intervals.
With `seed = nothing` (the default) a master seed is drawn from the global
RNG, so an upstream `Random.seed!` still pins the whole run.

`fit_growth_model` does not yet take a `seed`; its random restarts draw from
the global RNG, so call `Random.seed!` before it if you need the point
estimate pinned to the last digit. The run-to-run variation is at optimizer
convergence tolerance (~1e-8), not at any scientifically meaningful scale.

## API

**Fitting** — `fit_growth_model`, `solve_incidence`, `negloglik`, `FitResult`

**Uncertainty** — `run_bootstrap`, `simulate_noisy_curve`, `BootstrapResult`,
`compute_calibration_performance`, `compute_forecast_performance`

**Identifiability** — `screen_identifiability`, `screen_all_models`,
`print_identifiability_summary`

**Plotting** — `plot_fit_and_forecast`

## Identifiability caveat

`screen_identifiability` is a *local* diagnostic: it evaluates parameter
sensitivity around a single parameter point, not a global property of the
model. A collinearity index below ~20 (Omlin & Reichert 2001) suggests the
free trajectory parameters can be distinguished from the shape of the data.
Above that, treat individual parameter estimates with caution even when the
fit looks good.

## Known limitations

**The identifiability screen is local.** See the caveat above — it describes
the likelihood surface near your fit, not globally.

**`fit_growth_model` has no `seed` argument.** Its random restarts draw from
the global RNG, so call `Random.seed!` beforehand if you need the point
estimate pinned to the last digit. The variation without that is at
optimizer convergence tolerance (~1e-8). `run_bootstrap` does take a `seed`
and is fully reproducible.

**These are single-wave models.** Multi-wave epidemics, mid-outbreak
interventions, and reporting artifacts all break the assumption that one
growth curve describes the series.

## Least-squares convergence (fixed in v0.2.0)

Kept here because the diagnosis is worth not repeating.

Through v0.1.1, `dist = :normal` fits were unreliable. SLSQP would return
its starting point unchanged while reporting `Success`, and extra restarts
did not help. On the bundled Jalisco measles example it reached SSE 783,855;
the MATLAB implementation this toolbox is ported from reaches 135,363 on the
same problem with a 25-point MultiStart search.

The cause is SLSQP's line search failing when the gradient is large relative
to the feasible region. At the stall point the scaled gradient components
are of order 1e6 against an objective of order 1e5 — nowhere near
stationary. The optimizer simply cannot take a step.

Ruled out by testing, each of which made matters *worse* rather than better:
log-space reparameterization of `r` and `K`; narrowing `Kmax_mult`; adding
restarts. Automatic differentiation was never implicated — ForwardDiff
gradients agree with central differences to ~8 significant digits.

v0.2.0 polishes the search result with a derivative-free COBYLA pass, from
both the best SLSQP point and the warm start, keeping the lowest objective.
Polishing from the SLSQP point alone is not sufficient: COBYLA is a local
method and cannot cross basins, which gets SSE only to 125,037. With both
starts it reaches 83,440 — better than either previous implementation.
Pass `polish = false` to recover the earlier behaviour.

Poisson and negative-binomial fits were never affected and are unchanged:
their log-scale objectives are far better conditioned across `r` and `K`.
This is asserted in the test suite rather than assumed — a fit with and
without polish must agree.

One substantive point falls out of this. Under least squares the residuals
near the peak dominate, so the fit drives `p` to 1 (no sub-exponential
deceleration). Under NB1 the relative errors matter, so the early low-count
weeks pull `p` down to ~0.84. The error structure decides which part of the
curve the fit prioritizes, and `p` is the parameter most sensitive to that
choice — another reason `:nb1` is the appropriate default for count data.
## Citation

If you use this package, please cite it via its DOI:

> Subedi, R. GrowthFit.jl: Phenomenological growth-model fitting,
> uncertainty quantification, and short-term forecasting for epidemic
> incidence data. https://doi.org/10.5281/zenodo.21696876

The DOI above always resolves to the most recent release. See
[`CITATION.cff`](CITATION.cff), or use GitHub's "Cite this repository"
button, for a formatted entry.

## License

MIT
