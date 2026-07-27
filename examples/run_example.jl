"""
run_example.jl -- worked example: fit GRM to the JALISCO measles data under
NB1 error, quantify uncertainty via bootstrap, plot, and save outputs.

simple model fits, their
uncertainties, simple plots, forecasts with bands, error structure choice,
and saved outputs (CSV + a saved object for later re-plotting/editing).
tested with few datasets, more testing in progress

Setup (one-time):
    using Pkg
    Pkg.add(["DifferentialEquations", "Optimization", "OptimizationNLopt",
             "SpecialFunctions", "Distributions", "Plots", "CSV",
             "DataFrames", "JLD2", "ADTypes", "ForwardDiff"])
"""

using DelimitedFiles, CSV, DataFrames, JLD2, Random, Plots, Statistics

using GrowthFit
# Bind the submodule names too: this script uses qualified calls such as
# GrowthModels.MODEL_NAMES and PlotGrowthFit.ci95, and those names are not
# exported (ci95 is internal), so `using GrowthFit` alone does not bring
# them into scope.
using GrowthFit: GrowthModels, FitGrowthModel, BootstrapUncertainty, PlotGrowthFit

Random.seed!(123)

# Master seed for the bootstrap. Because run_bootstrap derives one stream per
# replicate index from this value, the intervals below are reproducible on any
# machine and at any Threads.nthreads() -- set it to `nothing` to draw a fresh
# seed from the global RNG on each run instead.
bootstrap_seed = 20260726

# =============================================================================
# 1. Load data
# =============================================================================
dataset_name = "JALISCO MEXICO"
datafile = joinpath(@__DIR__, "data", "JALISCO_2025-08-18_2026-06-29-trimmed.txt")

data = readdlm(datafile, '\t', Float64)
timevect_all = data[:, 1]
ydata_all = data[:, 2]

calibration_period = 40
forecast_horizon = 3

timevect = timevect_all[1:calibration_period]
ydata = ydata_all[1:calibration_period]

# =============================================================================
# 2. Point-estimate fit (SLSQP, data-scaled K bounds, NB1 error structure)
# =============================================================================
#-------------------------------------------------------
#GROWTH MODEL SYMBOLS (the `flag` variable)
#-------------------------------------------------------
#
 # flag     Name    ODE                                  Free parameters
  #------   ----    ----------------------------------    ---------------
  #:exp     EXP     dx = r*x                              r
  #:ggm     GGM     dx = r*x^p                             r, p
  #:glm     GLM     dx = r*x^p*(1 - x/K)                   r, p, K
  #:grm     GRM     dx = r*x^p*(1 - (x/K)^a)                r, p, a, K
  #:lm      LM      dx = r*x*(1 - x/K)                     r, K
  #:rich    RICH    dx = r*x*(1 - (x/K)^a)                  r, a, K
  #:gom     GOM     dx = r*x*exp(-a*t)   (no K term)        r, a
#
 # Example:  flag = :grm     (Generalized Richards Model)

flag = :grm

#dist       Meaning                          Variance assumption
 # --------   -----------------------------    -----------------------
  #:normal    Normal / least-squares            constant variance
  #:poisson   Poisson                           Var = mean
  #:nb1       Negative binomial (NB1)            Var = mean + alpha*mean
   #          (overdispersed count data —
    #          usually the best default for
     #         real epidemic case counts)

dist = :nb1

# Single source of truth for every output filename below -- dataset + model +
# error structure, same convention as the MATLAB GrowthPredict toolbox' run_tag, so
# different runs never collide/overwrite each other in output/.
model_str = GrowthModels.MODEL_NAMES[flag]
run_tag = "$(dataset_name)_$(model_str)_$(dist)"

println("Fitting $model_str under $dist error structure...")
point_fit = FitGrowthModel.fit_growth_model(flag, timevect, ydata; dist=dist, n_restarts=10)

if !point_fit.converged
    error("Point-estimate fit failed to converge -- check bounds/data before proceeding.")
end

println("Best fit: r=$(round(point_fit.r,digits=4))  p=$(round(point_fit.p_exp,digits=4))  ",
        "a=$(round(point_fit.a,digits=4))  K=$(round(point_fit.K,digits=1))  ",
        "alpha=$(round(point_fit.alpha,digits=4))")

# =============================================================================
# 3. Bootstrap uncertainty
# =============================================================================
println("Running bootstrap (M=300)...")
M = 300
boot = BootstrapUncertainty.run_bootstrap(flag, timevect, ydata, point_fit;
                     dist=dist, M=M, forecast_horizon=forecast_horizon,
                     n_restarts_boot=1, seed=bootstrap_seed)

println("Successful replicates: $(boot.n_success)/$M")

function summarize(v)
    m, lo, hi = PlotGrowthFit.ci95(v)
    return (median=m, lb=lo, ub=hi)
end

r_s = summarize(boot.params[:, 1])
p_s = summarize(boot.params[:, 2])
a_s = summarize(boot.params[:, 3])
K_s = summarize(boot.params[:, 4])

println("Parameter estimates:")
println("  r = $(round(r_s.median,digits=2)) (95% CI: $(round(r_s.lb,digits=2)), $(round(r_s.ub,digits=2)))")
println("  p = $(round(p_s.median,digits=2)) (95% CI: $(round(p_s.lb,digits=2)), $(round(p_s.ub,digits=2)))")
println("  a = $(round(a_s.median,digits=2)) (95% CI: $(round(a_s.lb,digits=2)), $(round(a_s.ub,digits=2)))")
println("  K = $(round(K_s.median,digits=1)) (95% CI: $(round(K_s.lb,digits=1)), $(round(K_s.ub,digits=1)))")

# =============================================================================
# 4. Plot
# =============================================================================
mkpath("output")
title_str = model_str
fig = PlotGrowthFit.plot_fit_and_forecast(boot, timevect, ydata, ydata_all, timevect_all;
                            forecast_horizon=forecast_horizon, title_str=title_str)
plot_path = joinpath("output", "$(run_tag)_fit_forecast.png")
Plots.savefig(fig, plot_path)
println("Plot saved to: $plot_path")

# =============================================================================
# 5. Save outputs: CSV summary + full object (for later re-plotting/editing,
#    same purpose as the R version's saved .rds files)
#
# CSVs keep full precision (rounding is cosmetic, only applied in the plot
# titles above) -- easy to round for display later without losing information
# now.
# =============================================================================
summary_df = DataFrame(
    parameter = ["r", "p", "a", "K"],
    median = [r_s.median, p_s.median, a_s.median, K_s.median],
    lb95 = [r_s.lb, p_s.lb, a_s.lb, K_s.lb],
    ub95 = [r_s.ub, p_s.ub, a_s.ub, K_s.ub],
)
csv_path = joinpath("output", "$(run_tag)_parameters.csv")
CSV.write(csv_path, summary_df)
println("Parameter summary saved to: $csv_path")

# Calibration performance (MAE, MSE, coverage, WIS) -- computed purely from
# the calibration window, so this is available regardless of whether
# forecast_horizon > 0.
calib_perf = BootstrapUncertainty.compute_calibration_performance(boot, ydata)
calib_df = DataFrame(model = [model_str], dist = [string(dist)],
                     MAE = [calib_perf.MAE], MSE = [calib_perf.MSE],
                     Coverage_95PI = [calib_perf.Coverage95PI], WIS = [calib_perf.WIS])
calib_path = joinpath("output", "$(run_tag)_calibration_performance.csv")
CSV.write(calib_path, calib_df)
println("Calibration performance saved to: $calib_path")
println("  MAE=$(round(calib_perf.MAE,digits=2))  Coverage=$(round(calib_perf.Coverage95PI,digits=1))%  WIS=$(round(calib_perf.WIS,digits=2))")

# Forecast performance (one row per horizon step) -- requires ydata_all to
# extend forecast_horizon weeks beyond the calibration window (held-out
# ground truth). Returns nothing (with a warning) if forecast_horizon==0 or
# there isn't enough held-out data, e.g. for a real-time forecast where the
# future hasn't happened yet.
fc_perf = BootstrapUncertainty.compute_forecast_performance(boot, ydata_all, calibration_period, forecast_horizon)
if fc_perf !== nothing
    fc_df = DataFrame(fc_perf)
    fc_path = joinpath("output", "$(run_tag)_forecast_performance.csv")
    CSV.write(fc_path, fc_df)
    println("Forecast performance saved to: $fc_path")
    for row in fc_perf
        println("  step $(row.horizon_step): true=$(row.y_true) pred=$(round(row.y_pred,digits=1)) ",
                "MAE=$(round(row.MAE,digits=2)) WIS=$(round(row.WIS,digits=2)) ",
                "95%PI=[$(round(row.LB95,digits=1)),$(round(row.UB95,digits=1))] covered=$(row.Coverage_95PI==100.0)")
    end
else
    println("Forecast performance not computed (forecast_horizon=0 or insufficient held-out data).")
end

# AICc (MLE formula, matching getAICc()'s method!=0 branch in the R/MATLAB
# toolboxes: AICc = -2*logL + 2*k + 2k(k+1)/(n-k-1), logL = -objective here
# since point_fit.objective is already the NLL)
n_obs = length(ydata)
n_params = GrowthModels.MODEL_NPARAMS[flag] + (dist == :nb1 ? 1 : 0)  # +1 for alpha under NB1
aicc = 2 * point_fit.objective + 2 * n_params + (2 * n_params * (n_params + 1)) / (n_obs - n_params - 1)
aicc_df = DataFrame(model = [model_str], dist = [string(dist)], AICc = [aicc],
                    numparams = [n_params], nll = [point_fit.objective])
aicc_path = joinpath("output", "$(run_tag)_AICc.csv")
CSV.write(aicc_path, aicc_df)
println("AICc saved to: $aicc_path  (AICc=$(round(aicc,digits=2)))")

jld2_path = joinpath("output", "$(run_tag)_fit_object.jld2")
@save jld2_path point_fit boot timevect ydata timevect_all ydata_all forecast_horizon
println("Full fit object saved to: $jld2_path")
println("  (reload later with: @load \"$jld2_path\" point_fit boot timevect ydata timevect_all ydata_all forecast_horizon)")

# =============================================================================
# 6. Save predicted values + quantiles (calibration period and forecast
#    period), matching the R/MATLAB toolboxes' QuantilesCalibration /
#    QuantilesForecastingPerformance CSVs.
# =============================================================================
alphaquantiles = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250,
                  0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600,
                  0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950,
                  0.975, 0.990]
quant_colnames = ["q" * string(round(a * 100, digits=1)) for a in alphaquantiles]

function quantile_row(col::AbstractVector)
    finite_col = filter(isfinite, col)
    return [quantile(finite_col, a) for a in alphaquantiles]
end

# Calibration: one row per calibration time point, with the OBSERVED value
# alongside the predicted median and full quantile grid (quantiles drawn
# from the noisy PI-band source, same as compute_calibration_performance()).
n_cal = length(timevect)
calib_rows = [quantile_row(boot.forecast_noisy[t, :]) for t in 1:n_cal]
predicted_calib_df = DataFrame(
    time = timevect,
    observed = ydata,
    predicted_median = [median(filter(isfinite, boot.fit_curves[t, :])) for t in 1:n_cal],
)
for (j, name) in enumerate(quant_colnames)
    predicted_calib_df[!, name] = [calib_rows[t][j] for t in 1:n_cal]
end
predicted_calib_path = joinpath("output", "$(run_tag)_predicted_calibration.csv")
CSV.write(predicted_calib_path, predicted_calib_df)
println("Predicted calibration values saved to: $predicted_calib_path")

# Forecast: one row per forecast-horizon time point, with predicted median +
# full quantile grid. Includes the OBSERVED (held-out) value if ydata_all
# extends far enough to have it; otherwise that column is filled with
# `missing` (real-time forecast, ground truth not yet available).
if forecast_horizon > 0
    timevect_forecast = vcat(timevect, timevect[end] .+ (1:forecast_horizon))
    fc_rows = [quantile_row(boot.forecast_noisy[t, :]) for t in (n_cal+1):(n_cal+forecast_horizon)]
    fc_median = [median(filter(isfinite, boot.forecast_curves[t, :])) for t in (n_cal+1):(n_cal+forecast_horizon)]
    has_holdout = length(ydata_all) >= n_cal + forecast_horizon
    observed_fc = has_holdout ? ydata_all[(n_cal+1):(n_cal+forecast_horizon)] :
                                fill(missing, forecast_horizon)

    predicted_forecast_df = DataFrame(
        time = timevect_forecast[(n_cal+1):(n_cal+forecast_horizon)],
        horizon_step = 1:forecast_horizon,
        observed = observed_fc,
        predicted_median = fc_median,
    )
    for (j, name) in enumerate(quant_colnames)
        predicted_forecast_df[!, name] = [fc_rows[h][j] for h in 1:forecast_horizon]
    end
    predicted_forecast_path = joinpath("output", "$(run_tag)_predicted_forecast.csv")
    CSV.write(predicted_forecast_path, predicted_forecast_df)
    println("Predicted forecast values saved to: $predicted_forecast_path")
    if !has_holdout
        println("  (no held-out data available yet -- 'observed' column is empty)")
    end
else
    println("Predicted forecast values not saved (forecast_horizon=0).")
end
