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
boot = run_bootstrap(res, timevect, ydata; n_boot = 200, forecast_horizon = 3)
plt  = plot_fit_and_forecast(boot, timevect, ydata, ydata_all, timevect_all;
                             forecast_horizon = 3)
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
include("Identifiability.jl")
include("PlotGrowthFit.jl")

using .GrowthModels
using .FitGrowthModel
using .BootstrapUncertainty
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
    # Identifiability
    screen_identifiability, print_identifiability_summary, screen_all_models,
    # PlotGrowthFit
    plot_fit_and_forecast

end # module GrowthFit
