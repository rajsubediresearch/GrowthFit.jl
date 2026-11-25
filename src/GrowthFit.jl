"""
    GrowthFit

Phenomenological growth-model fitting, uncertainty quantification, and
short-term forecasting for epidemic incidence data.

Seven nested growth models (EXP, GGM, GLM, GRM, LM, RICH, GOM) are fitted by
maximum likelihood under normal, Poisson, or NB1 error structures, with
parametric-bootstrap uncertainty and a local practical-identifiability screen.

# Quick start

```julia
using GrowthFit

timevect = collect(1.0:40.0)
ydata    = # weekly incidence

res  = fit_growth_model(:grm, timevect, ydata; dist = :nb1)
boot = run_bootstrap(:grm, timevect, ydata, res;
                     dist = :nb1, M = 300, forecast_horizon = 3, seed = 2026)
plt  = plot_fit_and_forecast(boot, timevect, ydata, ydata_all, timevect_all;
                             forecast_horizon = 3)
```

Or, to fit and score the model week after week as the data arrived rather
than once over the whole series:

```julia
rw = rolling_windows(:grm, timevect, ydata;
                     windowsize = 20, forecast_horizon = 4,
                     dist = :nb1, M = 300, seed = 2026)
parameter_trajectory(rw)   # parameters, with CIs, per window
skill_trajectory(rw)       # AICc and forecast skill, per window
```

See `examples/run_example.jl` for a complete worked example.
"""
module GrowthFit

# Submodules. Order matters: each `include` below relies on the modules
# already brought in above it (they refer to each other as `..GrowthModels`
# etc., which resolves to `GrowthFit.GrowthModels` from inside this module).
include("GrowthModels.jl")
include("FitGrowthModel.jl")
include("BootstrapUncertainty.jl")
include("RollingWindows.jl")
include("Identifiability.jl")
include("PlotGrowthFit.jl")

using .GrowthModels
using .FitGrowthModel
using .BootstrapUncertainty
using .RollingWindows
using .Identifiability
using .PlotGrowthFit

# Re-export the user-facing API at the top level so callers only need
# `using GrowthFit` rather than reaching into submodules.
export
    # GrowthModels
    growth_rhs!, MODEL_NPARAMS, MODEL_NAMES,
    # FitGrowthModel
    solve_incidence, negloglik, fit_growth_model, FitResult,
    # BootstrapUncertainty
    BootstrapResult, simulate_noisy_curve, run_bootstrap,
    compute_calibration_performance, compute_forecast_performance,
    # RollingWindows
    rolling_windows, RollingWindowResult, WindowFit, compute_aicc,
    parameter_trajectory, skill_trajectory, forecast_skill_by_step,
    # Identifiability
    screen_identifiability, print_identifiability_summary, screen_all_models,
    # PlotGrowthFit
    plot_fit_and_forecast

end # module GrowthFit
