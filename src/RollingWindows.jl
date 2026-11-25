"""
RollingWindows.jl -- sequential rolling-window fitting, uncertainty and
short-term forecasting, ported from the rolling-window analysis in the
MATLAB GrowthPredict toolbox.

The question a rolling-window run answers is not "what are the parameters of
this epidemic" but "how would this model have performed had I run it week
after week as the data arrived". Each window is an independent fit to a
fixed-length slice of the series, with its own initial condition, its own
bootstrap, and its own forecast scored against whatever was observed next.
Two things fall out of that:

  * parameter trajectories -- does `r` drift, does `p` stabilize once the
    peak is inside the window;
  * forecast skill versus window -- whether the model was usable in real
    time, and at which point in the outbreak it stopped being so.

Unlike the MATLAB implementation this takes a window *size* and an optional
subset of window *indices*, not a pair of start/end row numbers. Window `i`
covers observations `i : i+windowsize-1`, so window 1 is the earliest
window and `length(ydata) - windowsize + 1` is the last one that fits.

Nothing here writes files. A run returns a `RollingWindowResult`; the
accessors `parameter_trajectory`, `skill_trajectory` and
`forecast_skill_by_step` return column tables (NamedTuples of equal-length
vectors) that go straight into `DataFrame(...)`, `CSV.write` or a plot,
without this package taking a dependency on any of them.
"""
module RollingWindows

using Random, Statistics

using ..GrowthModels: MODEL_NPARAMS, MODEL_NAMES
using ..FitGrowthModel: FitResult, fit_growth_model
using ..BootstrapUncertainty: BootstrapResult, run_bootstrap,
                              compute_calibration_performance,
                              compute_forecast_performance

export WindowFit, RollingWindowResult, rolling_windows, compute_aicc,
       parameter_trajectory, skill_trajectory, forecast_skill_by_step

# ---------------------------------------------------------------------------
# AICc
# ---------------------------------------------------------------------------

"""
    compute_aicc(fit, dist; n_obs = length(fit.fitcurve))

Small-sample-corrected Akaike information criterion for a single
`FitResult`, matching `getAICc()` in the R/MATLAB toolboxes:

    AICc = -2*logL + 2k + 2k(k+1)/(n - k - 1)

`k` counts the model's free trajectory parameters plus one for any estimated
dispersion: `+1` for `alpha` under `:nb1`, `+1` for the residual variance
under `:normal`, nothing extra under `:poisson` (whose variance is fixed to
the mean).

The two likelihood branches differ because `FitResult.objective` is not the
same quantity under every error structure:

  * `:poisson`, `:nb1` -- the objective *is* the negative log-likelihood, so
    `-2logL = 2 * objective`.
  * `:normal` -- the objective is the residual sum of squares. The Gaussian
    log-likelihood concentrated over its unknown variance gives
    `-2logL = n*log(SSE/n) + n*(1 + log 2π)`; the constant is dropped here,
    as in `getAICc`'s least-squares branch. **Absolute values under
    `:normal` are therefore not comparable with values under `:poisson` or
    `:nb1`** -- only differences within one error structure and one `n` mean
    anything, which is all AICc is for.

Returns `Inf` for a non-converged fit, and for a window too short to apply
the correction at all (`n <= k + 1`), so a model that cannot be scored
sorts last rather than silently winning.

Note that `examples/run_example.jl` through v0.2.2 applied the
negative-log-likelihood branch unconditionally, which overstates AICc under
`dist = :normal`. Use this function instead.
"""
function compute_aicc(fit::FitResult, dist::Symbol;
                      n_obs::Integer = length(fit.fitcurve))
    n = Int(n_obs)
    k_traj = MODEL_NPARAMS[fit.flag]
    k = if dist == :normal
        k_traj + 1          # + residual variance
    elseif dist == :nb1
        k_traj + 1          # + overdispersion alpha
    elseif dist == :poisson
        k_traj
    else
        error("Unknown error structure: $dist")
    end

    (n - k - 1) <= 0 && return Inf
    correction = 2 * k + (2 * k * (k + 1)) / (n - k - 1)

    if dist == :normal
        sse = fit.objective
        (!isfinite(sse) || sse <= 0) && return Inf
        return n * log(sse / n) + correction
    else
        nll = fit.objective
        !isfinite(nll) && return Inf
        return 2 * nll + correction
    end
end

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

"""
    WindowFit

One window of a rolling-window run.

  * `window` -- the window index, as passed in `windows` (window `i` covers
    observations `i : i+windowsize-1`).
  * `index_range` -- those observation indices, into the original `ydata`.
  * `t_start`, `t_end` -- the first and last time value in the window.
  * `fit` -- the `FitResult` for this window alone.
  * `boot` -- the `BootstrapResult` for this window alone.
  * `aicc` -- `compute_aicc(fit, dist)`, so comparable across windows of
    equal length.
  * `calibration` -- `compute_calibration_performance` on the window's own
    observations: how well the model describes the data it was fitted to.
  * `forecast` -- per-horizon-step forecast metrics, or `nothing` when the
    window ends at the last observation so there is nothing to score
    against. Where the horizon runs only partly past the end of the series,
    the steps that *are* observed are scored and the rest omitted, so
    `length(forecast)` can be less than `forecast_horizon`; each row carries
    its own `horizon_step`.
"""
struct WindowFit
    window::Int
    index_range::UnitRange{Int}
    t_start::Float64
    t_end::Float64
    fit::FitResult
    boot::BootstrapResult
    aicc::Float64
    calibration::NamedTuple
    forecast::Union{Nothing,Vector{NamedTuple}}
end

"""
    RollingWindowResult

The result of a `rolling_windows` run: run settings plus a vector of
`WindowFit`. Supports `length`, indexing and iteration, so

```julia
for w in rw
    println(w.t_end, "  ", w.fit.r)
end
```

works, as does `rw[3]` and `rw[end]`. For anything tabular, prefer the
accessors `parameter_trajectory`, `skill_trajectory` and
`forecast_skill_by_step`.

`seed` is the master seed actually used, whether it was passed in or drawn
from the RNG -- record it and the whole run replays.
"""
struct RollingWindowResult
    flag::Symbol
    dist::Symbol
    windowsize::Int
    forecast_horizon::Int
    M::Int
    seed::UInt64
    windows::Vector{WindowFit}
end

Base.length(rw::RollingWindowResult) = length(rw.windows)
Base.isempty(rw::RollingWindowResult) = isempty(rw.windows)
Base.getindex(rw::RollingWindowResult, i) = rw.windows[i]
Base.firstindex(rw::RollingWindowResult) = firstindex(rw.windows)
Base.lastindex(rw::RollingWindowResult) = lastindex(rw.windows)
Base.eachindex(rw::RollingWindowResult) = eachindex(rw.windows)
Base.eltype(::Type{RollingWindowResult}) = WindowFit
Base.iterate(rw::RollingWindowResult, state = 1) =
    state > length(rw.windows) ? nothing : (rw.windows[state], state + 1)

function Base.show(io::IO, w::WindowFit)
    print(io, "WindowFit(", w.window, ", t=", w.t_start, "..", w.t_end,
          ", AICc=", round(w.aicc; digits = 1), ")")
end

function Base.show(io::IO, ::MIME"text/plain", rw::RollingWindowResult)
    nw = length(rw)
    println(io, "RollingWindowResult: ", get(MODEL_NAMES, rw.flag, String(rw.flag)),
            " under ", rw.dist)
    if nw == 0
        print(io, "  no windows")
        return
    end
    println(io, "  ", nw, " window", nw == 1 ? "" : "s", " of ", rw.windowsize,
            " observations, t = ", rw.windows[1].t_start, " .. ",
            rw.windows[end].t_end)
    println(io, "  forecast horizon ", rw.forecast_horizon, ", M = ", rw.M,
            " bootstrap replicates, seed = ", rw.seed)

    aiccs = [w.aicc for w in rw.windows]
    if any(isfinite, aiccs)
        imin = argmin(aiccs)
        println(io, "  AICc  min ", round(aiccs[imin]; digits = 1),
                " (window ", rw.windows[imin].window, ")")
    end

    cal_wis = [w.calibration.WIS for w in rw.windows]
    print(io, "  WIS   calibration mean ",
          round(mean(filter(isfinite, cal_wis)); digits = 2))
    scored = [w for w in rw.windows if w.forecast !== nothing && !isempty(w.forecast)]
    if isempty(scored)
        print(io, "   (no window had held-out data to score)")
    else
        fc_wis = [mean(row.WIS for row in w.forecast) for w in scored]
        print(io, "   forecast mean ", round(mean(fc_wis); digits = 2),
              " over ", length(scored), " scored window",
              length(scored) == 1 ? "" : "s")
    end
end

Base.show(io::IO, rw::RollingWindowResult) =
    print(io, "RollingWindowResult(", rw.flag, ", ", rw.dist, ", ",
          length(rw), " windows of ", rw.windowsize, ")")

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

"""
    rolling_windows(flag, timevect, ydata; windowsize = 20,
                    forecast_horizon = 0, windows = nothing, dist = :nb1,
                    M = 300, seed = nothing, verbose = true, ...)

Fit `flag` independently to each rolling window of `windowsize`
observations, bootstrap each fit, and score each window's forecast against
the observations that follow it.

Window `i` covers observations `i : i+windowsize-1`. By default every window
that fits is run, `1 : length(ydata) - windowsize + 1`; pass `windows` (any
range or vector of indices, e.g. `windows = 1:5` or `windows = [1, 6, 11]`)
to run a subset. Each window is fitted to its own slice with its own initial
condition `ydata[i]` and its own time values, exactly as if that slice were
the whole series -- which is the point, since that is the information a
forecaster would have had at the time.

```julia
rw = rolling_windows(:grm, timevect, ydata;
                     windowsize = 20, forecast_horizon = 4,
                     dist = :nb1, M = 300, seed = 2026)

rw                              # summary
rw[3].fit.r                     # third window's growth rate
parameter_trajectory(rw)        # r, p, a, K, alpha with CIs, per window
skill_trajectory(rw)            # AICc and MAE/WIS/coverage, per window
forecast_skill_by_step(rw)      # one row per window x horizon step
```

# Keyword arguments

`windowsize`, `forecast_horizon`, `windows`, `dist`, `M` and `seed` as
above. `alpha0`, `n_restarts`, `Kmax_mult` and `polish` are passed through
to `fit_growth_model`; `n_restarts_boot` and `noise_reps` to
`run_bootstrap`. `verbose = true` prints one line per window as it finishes,
which a long run needs -- pass `false` for silence.

# Cost

A run is `length(windows)` complete fit-plus-bootstrap analyses. At the
defaults that is 13 windows x 300 replicates on a 32-point series, so budget
accordingly and start with a small `M` and `windows = 1:3` to size it.

Windows are run serially and the bootstrap inside each one is threaded, so
`julia -t 8` already parallelizes the expensive part. Nesting threads over
windows as well would oversubscribe without helping.

# Reproducibility

Each window draws its restarts and its bootstrap from streams derived from
one master `seed` *and the window index*, so window `i` is identical whether
you ran `windows = 1:20` or `windows = i:i`. Re-running a single suspicious
window reproduces it exactly rather than approximately, which is what makes
a rolling-window run debuggable at all. As in `run_bootstrap`,
`seed = nothing` draws a master seed from `rng` on the main task, so an
upstream `Random.seed!` still pins the run; the master seed used is recorded
in `RollingWindowResult.seed` either way.

This also means `fit_growth_model`'s restarts are pinned here, which they
are not when you call it directly.
"""
function rolling_windows(flag::Symbol, timevect::AbstractVector, ydata::AbstractVector;
                         windowsize::Int = 20,
                         forecast_horizon::Int = 0,
                         windows = nothing,
                         dist::Symbol = :nb1,
                         M::Int = 300,
                         alpha0::Float64 = 1.0,
                         n_restarts::Int = 5,
                         n_restarts_boot::Int = 1,
                         noise_reps::Int = 20,
                         Kmax_mult::Float64 = 1000.0,
                         polish::Bool = true,
                         seed::Union{Nothing,Integer} = nothing,
                         rng = Random.default_rng(),
                         verbose::Bool = true)
    haskey(MODEL_NPARAMS, flag) ||
        error("Unknown model flag: $flag. One of $(sort(collect(keys(MODEL_NPARAMS)))).")
    dist in (:normal, :poisson, :nb1) ||
        error("Unknown error structure: $dist. One of :normal, :poisson, :nb1.")

    n = length(ydata)
    length(timevect) == n ||
        error("timevect has $(length(timevect)) entries but ydata has $n; they must match.")
    forecast_horizon >= 0 || error("forecast_horizon must be >= 0, got $forecast_horizon.")
    M >= 1 || error("M must be >= 1, got $M.")

    # A window needs more observations than the model has parameters, and
    # strictly more than k+1 for the AICc correction to exist at all. Catch
    # that here rather than handing back a window full of Inf.
    k_min = MODEL_NPARAMS[flag] + (dist == :poisson ? 0 : 1)
    windowsize >= k_min + 2 ||
        error("windowsize = $windowsize is too short for $(MODEL_NAMES[flag]) under " *
              "$dist, which estimates $k_min parameters; need at least $(k_min + 2).")
    windowsize <= n ||
        error("windowsize = $windowsize exceeds the $n observations available.")

    n_windows = n - windowsize + 1
    win_list = windows === nothing ? collect(1:n_windows) : collect(Int, windows)
    isempty(win_list) && error("`windows` is empty; nothing to run.")
    bad = filter(i -> i < 1 || i > n_windows, win_list)
    isempty(bad) ||
        error("window index/indices $bad out of range: with windowsize = $windowsize " *
              "and $n observations the valid windows are 1:$n_windows.")

    # One master seed drawn on the main task, then one stream per (window,
    # purpose). Hashing the window index in is what makes a window
    # independent of which other windows were run alongside it.
    master = seed === nothing ? rand(rng, UInt64) : UInt64(seed)

    tv = collect(float.(timevect))
    yv = collect(float.(ydata))

    fits = Vector{WindowFit}(undef, length(win_list))

    for (slot, i) in enumerate(win_list)
        idx = i:(i + windowsize - 1)
        tw = tv[idx]
        yw = yv[idx]

        fit = fit_growth_model(flag, tw, yw;
                               dist = dist, alpha0 = alpha0,
                               n_restarts = n_restarts, Kmax_mult = Kmax_mult,
                               polish = polish,
                               rng = Random.Xoshiro(hash((master, i, :fit))))

        boot = run_bootstrap(flag, tw, yw, fit;
                             dist = dist, M = M,
                             forecast_horizon = forecast_horizon,
                             n_restarts_boot = n_restarts_boot,
                             noise_reps = noise_reps,
                             seed = hash((master, i, :boot)))

        cal = compute_calibration_performance(boot, yw)
        aicc = compute_aicc(fit, dist; n_obs = windowsize)

        # Score only the horizon steps that were actually observed. Checking
        # here rather than letting compute_forecast_performance refuse keeps
        # the late windows -- the ones forecasting past the end of the
        # series, which is the realistic case -- from each emitting a
        # warning about data that was never going to exist.
        n_avail = n - last(idx)
        h_score = min(forecast_horizon, n_avail)
        fc = h_score > 0 ?
            compute_forecast_performance(boot, yv[i:end], windowsize, h_score) : nothing

        fits[slot] = WindowFit(i, idx, tw[1], tw[end], fit, boot, aicc, cal, fc)

        if verbose
            fc_note = fc === nothing || isempty(fc) ? "unscored" :
                      "fcWIS=" * string(round(mean(row.WIS for row in fc); digits = 2))
            println("window ", lpad(i, 3), " of ", lpad(n_windows, 3),
                    "  t=", tw[1], "..", tw[end],
                    "  AICc=", round(aicc; digits = 1),
                    "  calWIS=", round(cal.WIS; digits = 2),
                    "  ", fc_note,
                    "  boot ", boot.n_success, "/", M)
        end
    end

    return RollingWindowResult(flag, dist, windowsize, forecast_horizon, M, master, fits)
end

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

const _PARAM_COLS = (:r, :p, :a, :K, :alpha)

"""
    parameter_trajectory(rw; q = (0.025, 0.975))

Bootstrap median and interval for every parameter, one row per window --
the table behind a "does `r` drift as the window slides" plot.

Returns a column table with `window`, `t_start`, `t_end`, and then `r`,
`r_lo`, `r_hi`, `p`, `p_lo`, ..., `alpha`, `alpha_lo`, `alpha_hi`, taken
from each window's `boot.params` over its finite replicates. `q` sets the
interval quantiles.

All five parameters are always present so the table's shape does not depend
on the model: entries the model holds fixed (`p` for `:lm`, `a` for `:glm`,
`alpha` for anything but `:nb1`) come back at their fixed value with a
zero-width interval. See `MODEL_NPARAMS` for which are free.

```julia
using DataFrames
df = DataFrame(parameter_trajectory(rw))
```
"""
function parameter_trajectory(rw::RollingWindowResult; q = (0.025, 0.975))
    nw = length(rw)
    window  = [w.window  for w in rw.windows]
    t_start = [w.t_start for w in rw.windows]
    t_end   = [w.t_end   for w in rw.windows]

    med = Dict{Symbol,Vector{Float64}}()
    lo  = Dict{Symbol,Vector{Float64}}()
    hi  = Dict{Symbol,Vector{Float64}}()
    for (j, nm) in enumerate(_PARAM_COLS)
        m = fill(NaN, nw); l = fill(NaN, nw); h = fill(NaN, nw)
        for (s, w) in enumerate(rw.windows)
            col = filter(isfinite, view(w.boot.params, :, j))
            isempty(col) && continue
            m[s] = median(col)
            l[s] = quantile(col, q[1])
            h[s] = quantile(col, q[2])
        end
        med[nm] = m; lo[nm] = l; hi[nm] = h
    end

    return (window = window, t_start = t_start, t_end = t_end,
            r = med[:r], r_lo = lo[:r], r_hi = hi[:r],
            p = med[:p], p_lo = lo[:p], p_hi = hi[:p],
            a = med[:a], a_lo = lo[:a], a_hi = hi[:a],
            K = med[:K], K_lo = lo[:K], K_hi = hi[:K],
            alpha = med[:alpha], alpha_lo = lo[:alpha], alpha_hi = hi[:alpha])
end

"""
    skill_trajectory(rw)

One row per window: AICc, bootstrap convergence count, the calibration
metrics, and the forecast metrics averaged over the horizon steps that were
scored -- the table behind a "when did this model stop forecasting usefully"
plot.

Columns: `window`, `t_start`, `t_end`, `n_success`, `aicc`, `objective`,
then `MAE_cal`, `MSE_cal`, `Coverage95PI_cal`, `WIS_cal`, then `MAE_fc`,
`MSE_fc`, `Coverage95PI_fc`, `WIS_fc`, `steps_scored`.

The `_fc` columns are means across horizon steps and are `NaN` for a window
with nothing to score (`steps_scored == 0`), which the late windows always
are. Averaging a 1-step and a 4-step-ahead error into one number loses the
horizon dependence -- `forecast_skill_by_step` keeps it.
"""
function skill_trajectory(rw::RollingWindowResult)
    nw = length(rw)
    window  = [w.window  for w in rw.windows]
    t_start = [w.t_start for w in rw.windows]
    t_end   = [w.t_end   for w in rw.windows]
    n_success = [w.boot.n_success for w in rw.windows]
    aicc = [w.aicc for w in rw.windows]
    objective = [w.fit.objective for w in rw.windows]

    MAE_cal = [Float64(w.calibration.MAE) for w in rw.windows]
    MSE_cal = [Float64(w.calibration.MSE) for w in rw.windows]
    COV_cal = [Float64(w.calibration.Coverage95PI) for w in rw.windows]
    WIS_cal = [Float64(w.calibration.WIS) for w in rw.windows]

    MAE_fc = fill(NaN, nw); MSE_fc = fill(NaN, nw)
    COV_fc = fill(NaN, nw); WIS_fc = fill(NaN, nw)
    steps  = zeros(Int, nw)
    for (s, w) in enumerate(rw.windows)
        rows = w.forecast
        (rows === nothing || isempty(rows)) && continue
        steps[s]  = length(rows)
        MAE_fc[s] = mean(row.MAE for row in rows)
        MSE_fc[s] = mean(row.MSE for row in rows)
        COV_fc[s] = mean(row.Coverage_95PI for row in rows)
        WIS_fc[s] = mean(row.WIS for row in rows)
    end

    return (window = window, t_start = t_start, t_end = t_end,
            n_success = n_success, aicc = aicc, objective = objective,
            MAE_cal = MAE_cal, MSE_cal = MSE_cal,
            Coverage95PI_cal = COV_cal, WIS_cal = WIS_cal,
            MAE_fc = MAE_fc, MSE_fc = MSE_fc,
            Coverage95PI_fc = COV_fc, WIS_fc = WIS_fc,
            steps_scored = steps)
end

"""
    forecast_skill_by_step(rw)

Long-format forecast metrics: one row per (window, horizon step), stacked
across every window that had held-out data to score. This is the granularity
the R/MATLAB toolboxes report (`MAEFS`/`WISFS` and relatives) and the one
that answers "how fast does skill decay with horizon", which a per-window
average cannot.

Columns: `window`, `t_end` (last observation the window was fitted to),
`horizon_step`, `y_true`, `y_pred`, `MAE`, `MSE`, `Coverage_95PI`, `LB95`,
`UB95`, `WIS`.

Returns empty vectors when no window could be scored -- `forecast_horizon`
was 0, or every window ran to the end of the series.

```julia
# mean WIS by horizon step, across windows
t = forecast_skill_by_step(rw)
[(h, mean(t.WIS[t.horizon_step .== h])) for h in sort(unique(t.horizon_step))]
```
"""
function forecast_skill_by_step(rw::RollingWindowResult)
    window = Int[]; t_end = Float64[]; horizon_step = Int[]
    y_true = Float64[]; y_pred = Float64[]
    MAE = Float64[]; MSE = Float64[]; COV = Float64[]
    LB95 = Float64[]; UB95 = Float64[]; WIS = Float64[]

    for w in rw.windows
        rows = w.forecast
        (rows === nothing || isempty(rows)) && continue
        for row in rows
            push!(window, w.window)
            push!(t_end, w.t_end)
            push!(horizon_step, row.horizon_step)
            push!(y_true, Float64(row.y_true))
            push!(y_pred, Float64(row.y_pred))
            push!(MAE, Float64(row.MAE))
            push!(MSE, Float64(row.MSE))
            push!(COV, Float64(row.Coverage_95PI))
            push!(LB95, Float64(row.LB95))
            push!(UB95, Float64(row.UB95))
            push!(WIS, Float64(row.WIS))
        end
    end

    return (window = window, t_end = t_end, horizon_step = horizon_step,
            y_true = y_true, y_pred = y_pred,
            MAE = MAE, MSE = MSE, Coverage_95PI = COV,
            LB95 = LB95, UB95 = UB95, WIS = WIS)
end

end # module
