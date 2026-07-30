# Running the examples

`run_example.jl` fits one growth model to one dataset, bootstraps it,
forecasts, scores the forecast, and writes everything to `output/`.
`edit_plot.jl` re-draws a saved plot without re-fitting.

These scripts have their own environment (`examples/Project.toml`) because
they need CSV, DataFrames, and JLD2 for I/O, which the package itself does
not depend on.

## Setup

From the repository root, once:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then, to run:

```bash
julia --project=examples examples/run_example.jl
```

The bootstrap is multithreaded — add `-t 8` before `--project` to use it.
Results are identical either way; see "Reproducibility" below.

If you change the package's dependencies, re-run the setup line with
`Pkg.resolve()` added, or the examples environment will still be looking
for the old ones.

## What to edit

Near the top of `run_example.jl`:

| variable | meaning |
| --- | --- |
| `dataset_name` | short label used in every output filename |
| `datafile` | path to a 2-column tab-separated file (time, incidence; no header) |
| `calibration_period` | number of time points used to fit |
| `forecast_horizon` | weeks ahead to forecast; `0` fits only |
| `flag` | which growth model (see below) |
| `dist` | error structure (see below) |
| `M` | bootstrap replicates (default 300; more is smoother but slower) |
| `bootstrap_seed` | master seed for the bootstrap; `nothing` draws a fresh one each run |

For `forecast_performance` to be computed against real held-out data, the
file needs at least `calibration_period + forecast_horizon` rows.

### Growth models (`flag`)

| flag | Model | ODE | Free parameters |
| --- | --- | --- | --- |
| `:exp` | Exponential Growth | `dx = r·x` | `r` |
| `:ggm` | Generalized Growth | `dx = r·x^p` | `r, p` |
| `:lm` | Logistic | `dx = r·x·(1 − x/K)` | `r, K` |
| `:glm` | Generalized Logistic | `dx = r·x^p·(1 − x/K)` | `r, p, K` |
| `:rich` | Richards | `dx = r·x·(1 − (x/K)^a)` | `r, a, K` |
| `:grm` | Generalized Richards | `dx = r·x^p·(1 − (x/K)^a)` | `r, p, a, K` |
| `:gom` | Gompertz | `dx = r·x·exp(−a·t)` | `r, a` |

### Error structures (`dist`)

| dist | Meaning | Variance |
| --- | --- | --- |
| `:normal` | least squares | constant |
| `:poisson` | Poisson | `Var = μ` |
| `:nb1` | negative binomial | `Var = μ + α·μ` |

`:nb1` is usually the right default for real case counts, which are almost
always overdispersed relative to Poisson.

GRM — and to a lesser extent RICH — can be practically unidentifiable on
some datasets: the fit looks fine but the individual parameters are
trading off against each other. Symptoms are an implausibly large `K` and
wide or multimodal parameter histograms. Sanity-check against GLM, LM, or
RICH, and see the identifiability screen below.

## Reproducibility

`run_bootstrap` derives one random stream per replicate index from
`bootstrap_seed`, so results depend only on that seed — **not** on
`Threads.nthreads()`. Pass an explicit seed whenever you report an
interval.

`fit_growth_model` does not yet take a seed; its random restarts draw from
the global RNG. `run_example.jl` calls `Random.seed!(123)` at the top,
which pins them. The run-to-run variation without that is at optimizer
convergence tolerance (~1e-8), not anything meaningful.

## Output files

Written to `output/`, named `<Dataset>_<Model>_<ErrorStructure>_...`:

| file | contents |
| --- | --- |
| `*_fit_forecast.png` | parameter histograms + fit/forecast band |
| `*_parameters.csv` | medians and 95% CIs for r, p, a, K |
| `*_AICc.csv` | AICc, negative log-likelihood, parameter count |
| `*_calibration_performance.csv` | MAE, MSE, 95% PI coverage, WIS over the calibration window |
| `*_forecast_performance.csv` | same metrics per forecast step against held-out data (only if `forecast_horizon > 0` and the data extends far enough) |
| `*_predicted_calibration.csv` | fitted median and interval per calibration point |
| `*_predicted_forecast.csv` | forecast median and interval per step |
| `*_fit_object.jld2` | the full saved objects, for re-plotting without re-fitting |

## Re-drawing a plot (`edit_plot.jl`)

Loads a saved `*_fit_object.jld2` and re-renders with different fonts,
sizes, or colours. No re-fitting, so it is fast.

1. Run `run_example.jl` first so the `.jld2` exists
2. Edit the settings under "PART 1: WHAT TO CHANGE" (`jld2_path`, fonts,
   colours, dimensions, export format)
3. `julia --project=examples examples/edit_plot.jl`
4. Look at `output/..._edited.png`
5. Change a setting and re-run — no need to restart Julia

## Practical identifiability screen

Checks whether a model's free parameters can be distinguished from the
shape of the data, before you spend time on a full bootstrap. Cheap — no
optimizer involved.

```julia
using GrowthFit, DelimitedFiles

data = readdlm("examples/data/jalisco_measles.txt", '\t', Float64)
timevect = data[1:40, 1]        # match your calibration_period
ydata    = data[1:40, 2]

screen_all_models(timevect, ydata)
```

Prints a collinearity index per model. Below ~20 is considered
identifiable (Omlin & Reichert 2001); higher means the parameters are
trading off.

A model can be flagged here and still give a perfectly good curve and
forecast — the warning is about interpreting the individual parameter
values, not about the fit. GRM is the one most often flagged.

For one model in detail:

```julia
params0 = (r = 0.5, p = 0.9, a = 1.0, K = min(sum(ydata), 1e5), I0 = ydata[1])
screen_identifiability(:grm, timevect, params0; debug = true)
```

`debug = true` prints the full sensitivity matrix and every pairwise
collinearity value, so you can see which pair is the problem. If the
model's overall collinearity is much higher than its worst pairwise value,
three or more parameters are trading off jointly rather than any single
pair — common for GRM.

Note the diagnostic is **local**: it evaluates sensitivities around one
point in parameter space, so it describes the surface near that point, not
globally.

## Troubleshooting

**"Package GrowthFit does not have X in its dependencies"** — the package's
dependency list changed since this environment's manifest was written:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.resolve(); Pkg.instantiate()'
```

**"Package GrowthFit not found"** — the `dev` link is missing. Same command.

**A fit fails to converge** — the script prints the full Julia error rather
than just "failed to converge". Read the first few lines of the stack
trace.

**Everything is slow the first time** — Julia compiles on first call. The
second run of the same code is far faster.

**Stuck precompilation cache** — delete `~/.julia/compiled` (Windows:
`C:\Users\<you>\.julia\compiled`) and let it rebuild. Slow but reliable.
