"""
FitGrowthModel.jl -- point-estimate fitting via NLopt's SLSQP (the same
algorithm family as MATLAB's fmincon+SQP, chosen specifically because plain
gradient descent / L-BFGS-B was found to converge to a materially worse
local optimum on collinear surfaces like GRM under negative-binomial error --
see project history). Supports normal (LSQ), Poisson, and NB1 error
structures via a single `dist` argument.
"""
module FitGrowthModel

# Only Tsit5 is used, so depend on that solver package directly rather than
# on the DifferentialEquations meta-package. That pulled in every solver in
# the stack (Rosenbrock, BDF, SDIRK, Verner, NonlinearSolve, LinearSolve,
# MKL) for one explicit RK method, which cost roughly fifteen minutes of
# precompilation on a cold install and exposed users to version conflicts
# between sibling OrdinaryDiffEq packages that this package never touches.
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve
# ForwardDiff must be LOADED, not merely declared as a dependency:
# `AutoForwardDiff()` below is only a marker type from ADTypes, and
# DifferentiationInterface supplies the actual implementation through a
# package extension that activates when ForwardDiff is in the session.
# This used to happen by accident, because `using DifferentialEquations`
# pulled ForwardDiff in transitively. With the narrower solver dependency
# it no longer does, and every solve fails with a MethodError on
# `_prepare_pushforward_aux`. (Aqua's stale-deps check flags the same
# problem from the other direction: a [deps] entry that is never imported.)
import ForwardDiff
using Optimization, OptimizationNLopt
using SpecialFunctions: loggamma
using Random
using ADTypes: AutoForwardDiff

using ..GrowthModels: growth_rhs!, MODEL_NPARAMS

export solve_incidence, negloglik, fit_growth_model, FitResult

"""
    solve_incidence(flag, r, p_exp, a, K, I0, timevect; reltol=1e-8, abstol=1e-8)

Solve the growth ODE for the given parameters and return WEEKLY INCIDENCE
(not cumulative) at each time in `timevect`. Returns `nothing` if the solve
fails or produces non-finite output, so callers can treat that as an invalid
parameter region (mirrors R's `solve_ode()` returning NA).
"""
function solve_incidence(flag, r, p_exp, a, K, I0, timevect; reltol=1e-8, abstol=1e-8)
    tspan = (timevect[1], timevect[end])
    params = (flag, r, p_exp, a, K)
    # Promote the initial condition to match the parameter element type (Dual
    # during AD, Float64 otherwise) so OrdinaryDiffEq's dual-number
    # propagation works correctly through the solve.
    T = typeof(r)
    u0 = T[max(I0, 0.01)]
    prob = ODEProblem(growth_rhs!, u0, tspan, params)
    sol = try
        solve(prob, Tsit5(); saveat=timevect, reltol=reltol, abstol=abstol)
    catch e
        @debug "solve_incidence: ODE solve threw" exception=(e, catch_backtrace())
        return nothing
    end
    if sol.retcode !== :Success && sol.retcode !== :Default && Symbol(sol.retcode) !== :Success
        return nothing
    end
    cumulative = [u[1] for u in sol.u]
    if length(cumulative) != length(timevect) || any(!isfinite, cumulative)
        return nothing
    end
    incidence = abs.(vcat(cumulative[1], diff(cumulative)))
    return incidence
end

"""
    negloglik(yfit, ydata, dist, alpha)

Negative log-likelihood for the fitted incidence `yfit` against observed
`ydata`, under the chosen error structure:
  dist = :normal  -- sum of squared errors (minimizing SSE == maximizing a
                     Gaussian likelihood with constant variance)
  dist = :poisson -- Poisson NLL
  dist = :nb1     -- Negative binomial, Var = mean + alpha*mean (NB1),
                     using the log-gamma form (numerically identical to the
                     explicit sum-of-logs loop used in the R/MATLAB code,
                     but vectorized and stable via `loggamma`)

Includes the same numerical sanity guard used in the R fix: rejects
implausibly-scaled `yfit` (more than 50x the max observed value) up front,
since a technically-finite but wildly-scaled trajectory can otherwise
produce a spuriously "good" NLL under NB1 and mislead the optimizer into a
flat, unidentifiable region (this was the root cause of the original R/GRM
bug this whole toolbox redesign is meant to avoid repeating).
"""
function negloglik(yfit::AbstractVector, ydata::AbstractVector, dist::Symbol, alpha::Real)
    if any(!isfinite, yfit)
        return 1e10
    end
    max_plausible = 50 * max(maximum(ydata), 1.0)
    if maximum(yfit) > max_plausible
        return 1e10
    end
    yfit = max.(yfit, 0.001)

    if dist == :normal
        return sum((ydata .- yfit) .^ 2)
    elseif dist == :poisson
        return -sum(ydata .* log.(yfit) .- yfit)
    elseif dist == :nb1
        c = 1.0 / alpha
        # sum_{j=0}^{y-1} log(j + c*yfit) = loggamma(y + c*yfit) - loggamma(c*yfit)
        s = 0.0
        for i in eachindex(ydata)
            y = ydata[i]
            if y > 0
                s += loggamma(y + c * yfit[i]) - loggamma(c * yfit[i])
            end
            s += y * log(alpha) - (y + c * yfit[i]) * log(1 + alpha)
        end
        return -s
    else
        error("Unknown error structure: $dist")
    end
end

struct FitResult
    flag::Symbol
    r::Float64
    p_exp::Float64
    a::Float64
    K::Float64
    I0::Float64
    alpha::Float64
    fitcurve::Vector{Float64}
    objective::Float64
    converged::Bool
end

"""
    fit_growth_model(flag, timevect, ydata; dist=:nb1, alpha0=1.0,
                      n_restarts=5, Kmax_mult=1000.0, polish=true)

Fit a single growth model to `ydata` via SLSQP (NLopt) with `n_restarts`
random restarts (plus one warm start from a simple heuristic guess), keeping
the best (lowest-objective) result. Bounds on K are scaled to the data
(`Kmax_mult * sum(ydata)`), not a fixed huge constant -- this is the fix that
kept the optimizer from wandering into a flat, unidentifiable-K region on
collinear models like GRM (same fix as in the R version, ported directly).

After the multistart search, the best point is polished with a
derivative-free COBYLA pass and the lower objective is kept (`polish=true`,
the default). This matters most under `dist=:normal`, where SLSQP can stall
and return its starting point; see the comment at the polish step. Pass
`polish=false` to recover pre-v0.2.0 behaviour.

Returns a `FitResult`.
"""
function fit_growth_model(flag::Symbol, timevect::AbstractVector, ydata::AbstractVector;
                           dist::Symbol=:nb1, alpha0::Float64=1.0,
                           n_restarts::Int=5, Kmax_mult::Float64=1000.0,
                           polish::Bool=true,
                           rng=Random.default_rng())
    I0 = ydata[1]
    data_sum = sum(ydata)
    data_max = maximum(ydata)

    # Bounds: [r, p_exp, a, K, alpha]. Unused entries for a given flag are
    # still bounded (kept fixed at 1 via LB==UB) so the optimizer's
    # parameter vector length stays constant across models.
    Kmax = max(Kmax_mult * data_sum, 1000.0)
    rlb, rub = 1e-4, 20.0

    bounds = Dict(
        :exp  => (lb=[rlb, 1.0, 1.0, 1.0],       ub=[rub, 1.0, 1.0, 1.0]),
        :ggm  => (lb=[rlb, 0.01, 1.0, 1.0],      ub=[rub, 1.0, 1.0, 1.0]),
        :glm  => (lb=[rlb, 0.01, 1.0, 1.0],      ub=[rub, 1.0, 1.0, Kmax]),
        :grm  => (lb=[rlb, 0.01, 0.0, 1.0],      ub=[rub, 1.0, 10.0, Kmax]),
        :lm   => (lb=[rlb, 1.0, 1.0, 20.0],      ub=[rub, 1.0, 1.0, Kmax]),
        :rich => (lb=[rlb, 1.0, 0.0, 1.0],       ub=[rub, 1.0, 10.0, Kmax]),
        :gom  => (lb=[1e-4, 1.0, 0.0, 1.0],      ub=[20.0, 1.0, 10.0, 1.0]),
    )
    lb4, ub4 = bounds[flag].lb, bounds[flag].ub
    # alpha bound appended as a 5th dimension (only meaningful for dist=:nb1)
    lb = vcat(lb4, dist == :nb1 ? 1e-8 : 1.0)
    ub = vcat(ub4, dist == :nb1 ? 1e3  : 1.0)

    function objective(z, _p)
        r, p_exp, a, K, alpha = z
        yfit = solve_incidence(flag, r, p_exp, a, K, I0, timevect)
        yfit === nothing && return 1e10
        return negloglik(yfit, ydata, dist, alpha)
    end

    optf = OptimizationFunction(objective, AutoForwardDiff())
    best = nothing
    best_obj = Inf
    first_error = nothing

    # Warm-start guess + n_restarts random draws within bounds
    guesses = Vector{Vector{Float64}}()
    push!(guesses, [max(rlb, 0.1), 0.9, 1.0, min(data_sum, Kmax), alpha0])
    for _ in 1:n_restarts
        push!(guesses, lb .+ rand(rng, length(lb)) .* (ub .- lb))
    end

    for z0 in guesses
        z0c = clamp.(z0, lb, ub)
        prob = OptimizationProblem(optf, z0c, nothing; lb=lb, ub=ub)
        sol = try
            solve(prob, NLopt.LD_SLSQP(); maxiters=2000, xtol_rel=1e-8, ftol_rel=1e-10)
        catch e
            if first_error === nothing
                first_error = e
            end
            nothing
        end
        if sol !== nothing && isfinite(sol.objective) && sol.objective < best_obj
            best_obj = sol.objective
            best = sol.u
        end
    end

    # ---- Derivative-free polish (added in v0.2.0) ------------------------
    # SLSQP's line search fails when the gradient is large relative to the
    # feasible region: it returns its starting point unchanged while
    # reporting Success, and extra restarts do not rescue it. Under
    # dist=:normal on the bundled Jalisco example this produced SSE 783,855,
    # against 135,363 from the reference MATLAB implementation on the same
    # problem.
    #
    # COBYLA is derivative-free and unaffected. Running it once from the best
    # point SLSQP found, and keeping whichever objective is lower, recovers
    # SSE 83,448 there — better than either the Julia or the MATLAB
    # least-squares fit. It is a no-op wherever SLSQP already converged:
    # started from the fitted NB1 optimum on the same data, COBYLA returns
    # the identical point to four significant figures.
    #
    # Ruled out as alternative fixes, all of which made things worse:
    # log-space reparameterization of r and K, narrowing Kmax_mult, and more
    # restarts. Automatic differentiation is not implicated — ForwardDiff
    # gradients agree with central differences to ~8 significant digits.
    #
    # Cost is one extra solve, roughly 10% of a 10-restart fit.
    # COBYLA is a local method: it descends from where it is started and
    # cannot cross between basins. Polishing only the best SLSQP point is
    # therefore not enough when SLSQP stalled somewhere unhelpful — on the
    # Jalisco example under :normal that alone takes SSE from 783,855 to
    # 125,037, an improvement but still short of the 83,448 reachable from
    # the warm start. So polish from both the best SLSQP result and the
    # data-informed warm start, and keep the lowest objective of the three.
    # Cost is two extra solves, roughly 20% of a 10-restart fit.
    if polish
        polish_starts = Vector{Vector{Float64}}()
        best === nothing || push!(polish_starts, copy(best))
        warm_c = clamp.(guesses[1], lb, ub)
        if best === nothing || !isapprox(warm_c, best; rtol=1e-8)
            push!(polish_starts, warm_c)
        end

        for z_start in polish_starts
            prob_polish = OptimizationProblem(optf, copy(z_start), nothing; lb=lb, ub=ub)
            sol_polish = try
                solve(prob_polish, NLopt.LN_COBYLA(); maxiters=20_000)
            catch e
                if first_error === nothing
                    first_error = e
                end
                nothing
            end
            if sol_polish !== nothing && isfinite(sol_polish.objective) &&
               sol_polish.objective < best_obj
                best_obj = sol_polish.objective
                best = sol_polish.u
            end
        end
    end

    if best === nothing
        if first_error !== nothing
            @warn "All optimizer attempts failed for flag=$flag. First exception was:" exception=(first_error, catch_backtrace())
        end
        return FitResult(flag, NaN, NaN, NaN, NaN, I0, alpha0, fill(NaN, length(timevect)), Inf, false)
    end

    r, p_exp, a, K, alpha = best
    fitcurve = solve_incidence(flag, r, p_exp, a, K, I0, timevect)
    return FitResult(flag, r, p_exp, a, K, I0, alpha,
                     fitcurve === nothing ? fill(NaN, length(timevect)) : fitcurve,
                     best_obj, true)
end

end # module
