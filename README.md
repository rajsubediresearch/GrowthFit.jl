# GrowthFit.jl

[![CI](https://github.com/rajsubediresearch/GrowthFit.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/rajsubediresearch/GrowthFit.jl/actions/workflows/CI.yml)

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
Pkg.add(url = "https://github.com/rajsubediresearch/GrowthFit.jl")
```

Once registered in the General registry, this becomes `Pkg.add("GrowthFit")`.

## Models

| `flag`  | Name | ODE                        | Free parameters |
| ------- | ---- | -------------------------- | --------------- |
| `:exp`  | EXP  | `dx = r·x`                 | `r`             |
| `:ggm`  | GGM  | `dx = r·x^p`               | `r, p`          |
| `:glm`  | GLM  | `dx = r·x^p·(1 − x/K)`     | `r, p, K`       |
| `:grm`  | GRM  | `dx = r·x^p·(1 − (x/K)^a)` | `r, p, a, K`    |
| `:lm`   | LM   | `dx = r·x·(1 − x/K)`       | `r, K`          |
| `:rich` | RICH | `dx = r·x·(1 − (x/K)^a)`   | `r, a, K`       |
| `:gom`  | GOM  | `dx = r·x·exp(−a·t)`       | `r, a`          |

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
                     dist = :nb1, M = 300, forecast_horizon = 3)

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

## License

MIT
