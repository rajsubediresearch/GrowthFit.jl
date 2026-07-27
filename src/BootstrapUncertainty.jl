"""
BootstrapUncertainty.jl -- parametric bootstrap for parameter/forecast
uncertainty. Each replicate resamples data around the point-estimate fit
(matching the chosen error structure) and refits via `fit_growth_model`,
warm-started from the original point estimate. This is the approach that
was validated against MATLAB's toolbox for the hardest case tested
(GRM under NB1 error) -- unlike MCMC with a plain random-walk sampler, each
replicate gets its own independent gradient-aware search, so there's no
random-walk-into-a-flat-region failure mode.
"""
module BootstrapUncertainty

using Random, Distributions, Statistics

using ..FitGrowthModel: solve_incidence, fit_growth_model, FitResult

export BootstrapResult, simulate_noisy_curve, run_bootstrap,
       compute_calibration_performance, compute_forecast_performance

"""
    simulate_noisy_curve(mean_curve, dist, alpha)

Simulate one noisy observation realization around `mean_curve` (a vector of
weekly incidence means) under the chosen error structure. Used both to build
bootstrap resampled datasets and (with more replicates) to build forecast
prediction-interval bands.
"""
function simulate_noisy_curve(mean_curve::AbstractVector, dist::Symbol, alpha::Float64,
                              rng=Random.default_rng())
    n = length(mean_curve)
    out = similar(mean_curve)
    out[1] = mean_curve[1]
    for t in 2:n
        mu = max(mean_curve[t], 1e-6)
        if dist == :normal
            out[t] = max(rand(rng, Normal(mu, alpha)), 0.0)
        elseif dist == :poisson
            out[t] = rand(rng, Poisson(mu))
        elseif dist == :nb1
            # Var = mu + alpha*mu  =>  standard NB(r, p) reparameterization
            var_ = mu + alpha * mu
            p_ = mu / var_
            r_ = mu * p_ / (1 - p_)
            out[t] = rand(rng, NegativeBinomial(r_, p_))
        else
            error("Unknown error structure: $dist")
        end
    end
    return out
end

struct BootstrapResult
    flag::Symbol
    dist::Symbol
    params::Matrix{Float64}          # M x 5: r, p_exp, a, K, alpha
    fit_curves::Matrix{Float64}      # length(timevect) x M, calibration period, clean
    forecast_curves::Matrix{Float64} # length(timevect_forecast) x M, clean (no obs noise)
    forecast_noisy::Matrix{Float64}  # length(timevect_forecast) x (M*20), WITH obs noise (for PI bands)
    n_success::Int
end

"""
    run_bootstrap(flag, timevect, ydata, point_fit; dist=:nb1, M=300,
                  forecast_horizon=0, n_restarts_boot=1, noise_reps=20,
                  seed=nothing, rng=Random.default_rng())

Run `M` bootstrap replicates. `point_fit` is a `FitResult` from
`fit_growth_model` on the original data -- each replicate is warm-started
from it. `n_restarts_boot` extra random restarts per replicate (kept small;
SLSQP reliably finds the right basin from a good warm start, so heavy
multistart per replicate is usually unnecessary overhead -- see project
history where this was identified as a real speedup opportunity that wasn't
tested in the R version).

## Reproducibility

Replicate `j` draws from its own `Xoshiro` stream, derived deterministically
from a single master seed. This makes results depend only on that seed --
NOT on `Threads.nthreads()`.

That independence is the point of the change. Previously each replicate used
the per-task RNG that `Threads.@threads` hands out, which is thread-SAFE and
reproducible at a fixed thread count, but `@threads` chunks the loop by
`nthreads()`, so the streams land on different replicate indices when the
thread count changes. Measured on the Jalisco GRM/NB1 fit (M=50, same seed):
the 95% interval for `r` was [0.583, 1.145] on 4 threads and [0.586, 1.084]
on 8. Reproducible within a run configuration, not across one.

`seed` controls the master seed:
  * `seed = <Int>` -- fully explicit; the same value always gives the same
    replicates, on any machine and any thread count. Prefer this when
    reporting intervals.
  * `seed = nothing` (default) -- one master seed is drawn from `rng`
    serially before the loop, so an upstream `Random.seed!(...)` still makes
    the whole run reproducible, and successive calls in one session differ
    as you would expect.

`rng` is only ever touched on the main task, before any replicate starts, so
passing an explicit RNG object is now safe. (It previously was not: an
`Xoshiro` passed here would have been shared mutable state across threads.
The default `Random.default_rng()` was safe because it resolves to the
per-task RNG.)
"""
function run_bootstrap(flag::Symbol, timevect::AbstractVector, ydata::AbstractVector,
                       point_fit::FitResult; dist::Symbol=:nb1, M::Int=300,
                       forecast_horizon::Int=0, n_restarts_boot::Int=1, noise_reps::Int=20,
                       seed::Union{Nothing,Integer}=nothing,
                       rng=Random.default_rng())
    I0 = ydata[1]
    n_cal = length(timevect)
    timevect_forecast = forecast_horizon > 0 ?
        vcat(timevect, timevect[end] .+ (1:forecast_horizon)) : timevect
    n_fc = length(timevect_forecast)

    params = zeros(M, 5)
    fit_curves = zeros(n_cal, M)
    forecast_curves = zeros(n_fc, M)
    forecast_noisy_list = Vector{Matrix{Float64}}(undef, M)
    # Per-replicate success flags rather than a shared counter: `n_success += 1`
    # inside Threads.@threads is a read-modify-write on one Int from several
    # threads, so increments can be lost. Count after the loop instead.
    success = falses(M)

    # One master seed, drawn serially on the main task, then one independent
    # stream per replicate index. Replicate j always gets stream j, however
    # @threads happens to chunk the loop.
    master = seed === nothing ? rand(rng, UInt64) : UInt64(seed)
    rngs = [Random.Xoshiro(hash((master, j))) for j in 1:M]

    Threads.@threads for j in 1:M
        rng_j = rngs[j]
        boot_data = simulate_noisy_curve(point_fit.fitcurve, dist, point_fit.alpha, rng_j)

        best = fit_growth_model(flag, timevect, boot_data;
                                dist=dist, alpha0=point_fit.alpha,
                                n_restarts=n_restarts_boot, rng=rng_j)

        # Fallback to the point estimate if this replicate failed to converge,
        # so one bad replicate doesn't propagate NaNs through the whole matrix
        # (mirrors the R version's fallback-to-point-estimate behavior).
        use = best.converged && isfinite(best.objective) ? best : point_fit

        params[j, :] = [use.r, use.p_exp, use.a, use.K, use.alpha]

        curve_full = solve_incidence(flag, use.r, use.p_exp, use.a, use.K, I0, timevect_forecast)
        if curve_full === nothing || any(!isfinite, curve_full)
            curve_full = fill(NaN, n_fc)
        end
        forecast_curves[:, j] = curve_full
        fit_curves[:, j] = curve_full[1:n_cal]

        noisy = zeros(n_fc, noise_reps)
        for k in 1:noise_reps
            noisy[:, k] = any(!isfinite, curve_full) ? fill(NaN, n_fc) :
                          simulate_noisy_curve(curve_full, dist, use.alpha, rng_j)
        end
        forecast_noisy_list[j] = noisy

        success[j] = use.converged
    end

    n_success = count(success)
    forecast_noisy = hcat(forecast_noisy_list...)

    return BootstrapResult(flag, dist, params, fit_curves, forecast_curves,
                           forecast_noisy, n_success)
end

"""
    compute_calibration_performance(result, ydata)

Calibration-period performance metrics: MAE, MSE, 95% PI coverage, and WIS
(weighted interval score), computed purely from the calibration window --
independent of whether a forecast horizon was used. Matches the R/MATLAB
toolboxes' calibration metrics (the `*CS`/`WISC` outputs in
computeforecastperformance.m / computeWIS.m), so this is available even for
fit-only runs with no forecasting step.

`result.fit_curves` (clean, no observation noise) provides the median
prediction; `result.forecast_noisy[1:n_cal, :]` (with observation noise)
provides the prediction-interval quantiles, same split as the R version.
"""
function compute_calibration_performance(result::BootstrapResult, ydata::AbstractVector)
    n_cal = length(ydata)
    fit_cal = result.fit_curves[1:n_cal, :]
    noisy_cal = result.forecast_noisy[1:n_cal, :]

    y_median = [median(filter(isfinite, fit_cal[t, :])) for t in 1:n_cal]
    mae = mean(abs.(ydata .- y_median))
    mse = mean((ydata .- y_median) .^ 2)

    lb = [max(quantile(filter(isfinite, noisy_cal[t, :]), 0.025), 0.0) for t in 1:n_cal]
    ub = [max(quantile(filter(isfinite, noisy_cal[t, :]), 0.975), 0.0) for t in 1:n_cal]
    coverage = 100 * mean((ydata .>= lb) .& (ydata .<= ub))

    # Weighted Interval Score (WIS), same alpha levels as the R/MATLAB
    # toolboxes' computeWIS.m (0.02, 0.05, 0.1:0.1:0.9)
    alphas = vcat(0.02, 0.05, collect(0.1:0.1:0.9))
    K = length(alphas)
    w0 = 0.5
    wis_per_t = zeros(n_cal)
    for t in 1:n_cal
        y = ydata[t]
        col = filter(isfinite, noisy_cal[t, :])
        s = 0.0
        for alpha in alphas
            w_k = alpha / 2
            l = max(quantile(col, alpha / 2), 0.0)
            u = max(quantile(col, 1 - alpha / 2), 0.0)
            is_score = (u - l) + (2 / alpha) * (l - y) * (y < l) + (2 / alpha) * (y - u) * (y > u)
            s += w_k * is_score
        end
        m = quantile(col, 0.5)
        wis_per_t[t] = (1 / (K + 0.5)) * (w0 * abs(y - m) + s)
    end
    wis = mean(wis_per_t)

    return (MAE=mae, MSE=mse, Coverage95PI=coverage, WIS=wis)
end

"""
    compute_forecast_performance(result, ydata_all, n_cal, forecast_horizon)

Forecast-period performance metrics, ONE ROW PER FORECAST STEP (matches the
R/MATLAB toolboxes' MAEFS/MSEFS/PIFS/WISFS -- per-horizon-step granularity,
not a single collapsed number). Requires `ydata_all` (the full observed
series, including the held-out weeks beyond the calibration window) to have
at least `n_cal + forecast_horizon` entries; returns `nothing` with a
warning if it doesn't, mirroring the R version's length check.

This is the piece that was missing from the initial Julia port: bootstrap
already PRODUCES a forecast (`result.forecast_curves`/`forecast_noisy`
extend `forecast_horizon` steps past the calibration window), but nothing
previously compared that forecast against the actual observed values -- so
no forecast-performance CSV was ever generated, independent of how short
the horizon was.
"""
function compute_forecast_performance(result::BootstrapResult, ydata_all::AbstractVector,
                                      n_cal::Int, forecast_horizon::Int)
    if forecast_horizon <= 0
        return nothing
    end
    if length(ydata_all) < n_cal + forecast_horizon
        @warn "compute_forecast_performance: ydata_all has $(length(ydata_all)) points, " *
              "need at least $(n_cal + forecast_horizon) to evaluate forecast_horizon=$forecast_horizon. " *
              "Returning nothing."
        return nothing
    end

    rows = NamedTuple[]
    for h in 1:forecast_horizon
        t_idx = n_cal + h
        y_true = ydata_all[t_idx]

        pred_col = filter(isfinite, result.forecast_curves[t_idx, :])
        y_pred = median(pred_col)
        mae = abs(y_true - y_pred)
        mse = (y_true - y_pred)^2

        noisy_col = filter(isfinite, result.forecast_noisy[t_idx, :])
        lb = max(quantile(noisy_col, 0.025), 0.0)
        ub = max(quantile(noisy_col, 0.975), 0.0)
        covered = (y_true >= lb) && (y_true <= ub)

        alphas = vcat(0.02, 0.05, collect(0.1:0.1:0.9))
        K = length(alphas)
        w0 = 0.5
        s = 0.0
        for alpha in alphas
            w_k = alpha / 2
            l = max(quantile(noisy_col, alpha / 2), 0.0)
            u = max(quantile(noisy_col, 1 - alpha / 2), 0.0)
            is_score = (u - l) + (2 / alpha) * (l - y_true) * (y_true < l) +
                       (2 / alpha) * (y_true - u) * (y_true > u)
            s += w_k * is_score
        end
        m = quantile(noisy_col, 0.5)
        wis = (1 / (K + 0.5)) * (w0 * abs(y_true - m) + s)

        push!(rows, (horizon_step=h, y_true=y_true, y_pred=y_pred, MAE=mae, MSE=mse,
                     Coverage_95PI=covered ? 100.0 : 0.0, LB95=lb, UB95=ub, WIS=wis))
    end
    return rows
end

end # module
