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

Make a folder for your analysis, open a terminal **in that folder**, and start
Julia with the current directory as its project:

```
cd path/to/my_analysis
julia --project=.
```

Any terminal works — Terminal on macOS, bash or zsh on Linux, Command Prompt
or PowerShell on Windows. Two Windows notes: use backslashes in paths
(`cd E:\Julia\my_analysis`), and in Command Prompt you must switch drive
first by typing the drive letter on its own line (`E:`) before `cd`.

If `julia` is not recognised, it is not on your PATH. The
[juliaup installer](https://julialang.org/install/) sets this up; restart the
terminal after installing.

Then, at the `julia>` prompt:

```julia
using Pkg
Pkg.add("GrowthFit")
```

That is all. The `--project=.` flag tells Julia to use the folder you are in,
so it writes a `Project.toml` and `Manifest.toml` there recording exactly
which package versions your analysis used.

Start Julia the same way every time you return to that folder, and everything
is already installed:

```bash
cd path/to/my_analysis
julia --project=.
```

To run a script in that folder:

```bash
julia --project=. my_analysis.jl
```

<details>
<summary>Why <code>--project=.</code> rather than just installing?</summary>

Running `Pkg.add("GrowthFit")` without a project puts it in Julia's default
environment, which is shared by everything you have ever installed. The
resolver then has to satisfy all of it at once, and it may pin a dependency
to an older version to keep something unrelated happy — leaving you with a
version combination nobody tested.

A per-analysis project gives the resolver a clean problem, and the
`Manifest.toml` it writes is what makes the analysis reproducible months
later: `Pkg.instantiate()` in that folder restores the exact versions.

If you ever do hit a resolution failure, a throwaway environment isolates
whether your other packages are the cause:

```julia
using Pkg
Pkg.activate(mktempdir())
Pkg.add("GrowthFit")
```

</details>

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
(negative binomial with `Var = μ + α·μ`). For epidemic case counts `:nb1` is
the appropriate default — counts are almost always overdispersed relative to
Poisson.

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

The bootstrap is multithreaded. Start Julia with `-t 8` (or however many
cores you have) to use it — `julia --project=. -t 8`. Results are identical
however many threads you use; see Reproducibility below.

### Running the worked example

[`examples/run_example.jl`](examples/run_example.jl) fits, bootstraps,
forecasts and scores a real measles series, writing plots and CSVs to
`output/`. It is the fastest way to see what the package does end to end.

**1. Get the repository.** Either clone it:

```
git clone https://github.com/rajsubediresearch/GrowthFit.jl.git
cd GrowthFit.jl
```

or download the ZIP from the green **Code** button on GitHub, extract it, and
`cd` into the extracted folder.

**2. Start Julia from inside that folder**, using the examples environment.
The `-t 8` gives the bootstrap eight threads; use a number no higher than
your core count, or leave it off entirely.

```
julia --project=examples -t 8
```

**3. At the `julia>` prompt**, install the dependencies and run it:

```julia
using Pkg
Pkg.develop(path = ".")     # use the GrowthFit source in this clone
Pkg.instantiate()           # install what the example needs

include("examples/run_example.jl")
```

The first `instantiate` downloads and compiles the dependencies, which takes
a few minutes. After that, starting up takes seconds.

Everything from step 3 onward happens at the Julia prompt, so it is identical
on every operating system and shell.

**4. Look at `output/`** for the fit and forecast plot, parameter estimates
with confidence intervals, performance metrics, and the saved fit object.

To change the model, error structure, dataset or forecast horizon, edit the
variables at the top of `examples/run_example.jl` and re-run step 3's
`include` line — no need to restart Julia.

## Reproducibility

`run_bootstrap` derives one random stream per replicate index from a single
master `seed`, so results depend only on that seed and **not** on
`Threads.nthreads()`. Pass an explicit `seed` when reporting intervals. With
`seed = nothing` (the default) a master seed is drawn from the global RNG, so
an upstream `Random.seed!` still pins the whole run.

`fit_growth_model` does not take a `seed`; its random restarts draw from the
global RNG, so call `Random.seed!` before it if you need the point estimate
pinned to the last digit. The run-to-run variation is at optimizer
convergence tolerance (~1e-8), not at any scientifically meaningful scale.

## API

**Fitting** — `fit_growth_model`, `solve_incidence`, `negloglik`, `FitResult`

**Uncertainty** — `run_bootstrap`, `simulate_noisy_curve`, `BootstrapResult`,
`compute_calibration_performance`, `compute_forecast_performance`

**Identifiability** — `screen_identifiability`, `screen_all_models`,
`print_identifiability_summary`

**Plotting** — `plot_fit_and_forecast`

## Known limitations

**The identifiability screen is local.** `screen_identifiability` evaluates
parameter sensitivity around a single parameter point, not as a global
property of the model. A collinearity index below ~20 (Omlin & Reichert,
2001) suggests the free trajectory parameters can be distinguished from the
shape of the data. Above that, treat individual parameter estimates with
caution even when the fit looks good — and remember the diagnostic describes
the surface near your fit, not everywhere.

**`fit_growth_model` has no `seed` argument.** See Reproducibility above.

**These are single-wave models.** Multi-wave epidemics, mid-outbreak
interventions, and reporting artifacts all break the assumption that one
growth curve describes the series.

## Least-squares convergence (fixed in v0.2.0)

Kept here because the diagnosis is worth not repeating.

Through v0.1.1, `dist = :normal` fits were unreliable. SLSQP would return its
starting point unchanged while reporting `Success`, and extra restarts did
not help. On the bundled Jalisco measles example it reached SSE 783,855; the
MATLAB implementation this toolbox is ported from reaches 135,363 on the same
problem with a 25-point MultiStart search.

The cause is SLSQP's line search failing when the gradient is large relative
to the feasible region. At the stall point the scaled gradient components are
of order 1e6 against an objective of order 1e5 — nowhere near stationary. The
optimizer simply cannot take a step.

Ruled out by testing, each of which made matters *worse* rather than better:
log-space reparameterization of `r` and `K`; narrowing `Kmax_mult`; adding
restarts. Automatic differentiation was never implicated — ForwardDiff
gradients agree with central differences to ~8 significant digits.

v0.2.0 polishes the search result with a derivative-free COBYLA pass, from
both the best SLSQP point and the warm start, keeping the lowest objective.
Polishing from the SLSQP point alone is not sufficient: COBYLA is a local
method and cannot cross basins, which gets SSE only to 125,037. With both
starts it reaches 83,440 — better than either previous implementation. Pass
`polish = false` to recover the earlier behaviour.

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
