using GrowthFit
using Test
using Random
using Aqua

@testset "GrowthFit.jl" begin

    @testset "Aqua quality assurance" begin
        # Mechanical package hygiene: method ambiguities, unbound type
        # parameters, undefined exports, [extras]/[targets] consistency,
        # stale deps, [compat] coverage, type piracy, persistent tasks.
        #
        # undocumented_names is advisory only (it flags exported symbols
        # without docstrings) and is not an AutoMerge requirement, so it is
        # disabled here to keep the signal clean. deps_compat IS a hard
        # registration requirement -- if it fails, a [deps] entry is missing
        # its [compat] bound.
        Aqua.test_all(GrowthFit; undocumented_names = false)
    end

    @testset "model bookkeeping" begin
        for flag in (:exp, :ggm, :glm, :grm, :lm, :rich, :gom)
            @test haskey(MODEL_NPARAMS, flag)
            @test haskey(MODEL_NAMES, flag)
            @test MODEL_NPARAMS[flag] >= 1
        end
        # Nesting: each model has at least as many free parameters as the
        # models it generalizes.
        @test MODEL_NPARAMS[:grm] > MODEL_NPARAMS[:glm] > MODEL_NPARAMS[:ggm] > MODEL_NPARAMS[:exp]
    end

    @testset "growth_rhs!" begin
        du = zeros(1)
        # exponential: du = r*x
        growth_rhs!(du, [10.0], (:exp, 0.2, 1.0, 1.0, 1.0), 0.0)
        @test du[1] ≈ 2.0
        # logistic at carrying capacity has zero growth
        growth_rhs!(du, [100.0], (:lm, 0.2, 1.0, 1.0, 100.0), 0.0)
        @test isapprox(du[1], 0.0; atol = 1e-10)
        # unknown flag is an error, not silent garbage
        @test_throws ErrorException growth_rhs!(du, [1.0], (:nope, 0.2, 1.0, 1.0, 1.0), 0.0)
    end

    @testset "solve_incidence" begin
        timevect = collect(1.0:20.0)
        inc = solve_incidence(:lm, 0.4, 1.0, 1.0, 5000.0, 5.0, timevect)
        @test inc !== nothing
        @test length(inc) == length(timevect)
        @test all(isfinite, inc)
        @test all(>=(0), inc)
        # Logistic incidence is single-peaked: it should rise then fall.
        @test argmax(inc) > 1
        @test argmax(inc) < length(inc)
    end

    @testset "negloglik" begin
        ydata = [5.0, 12.0, 30.0, 55.0, 70.0]
        # A perfect fit scores better than a poor one under every error model.
        for dist in (:normal, :poisson, :nb1)
            @test negloglik(ydata, ydata, dist, 1.0) <
                  negloglik(ydata .* 2, ydata, dist, 1.0)
        end
        # SSE is exact for the normal case
        @test negloglik(ydata, ydata, :normal, 1.0) ≈ 0.0
        # Wildly-scaled trajectories are rejected by the sanity guard
        @test negloglik(ydata .* 1000, ydata, :nb1, 1.0) == 1e10
        @test negloglik(fill(NaN, 5), ydata, :normal, 1.0) == 1e10
        @test_throws ErrorException negloglik(ydata, ydata, :bogus, 1.0)
    end

    @testset "fit_growth_model recovers known parameters" begin
        Random.seed!(20260726)
        # Simulate noiseless logistic incidence, then refit it.
        timevect = collect(1.0:30.0)
        r_true, K_true, I0 = 0.35, 4000.0, 5.0
        ytrue = solve_incidence(:lm, r_true, 1.0, 1.0, K_true, I0, timevect)

        # Well-scaled search range. This is the control: it establishes that
        # the likelihood, the ODE solve, and the AD path are all correct.
        res = fit_growth_model(:lm, timevect, ytrue;
                               dist = :normal, n_restarts = 30, Kmax_mult = 2.0)
        @test res isa FitResult
        @test res.converged
        @test res.flag === :lm
        @test isfinite(res.objective)
        @test isapprox(res.r, r_true; rtol = 0.15)
        @test isapprox(res.K, K_true; rtol = 0.20)
        @test length(res.fitcurve) == length(timevect)
        # Noiseless data ⇒ the global SSE minimum is exactly zero.
        @test res.objective < 1e-3 * sum(abs2, ytrue)
    end

    @testset "default multistart on wide K bounds (known weakness)" begin
        # KNOWN ISSUE, noiseless data only. With the default Kmax_mult=1000,
        # SLSQP returns its own starting point unchanged and reports
        # Success. Two compounding causes, both confirmed:
        #
        #   1. Gradient scaling. At the warm start the gradient is
        #      df/dr = -8.0e5 against df/dK = -0.14, a ratio of 5.7e6. The
        #      first QP step is essentially pure r, clips at its upper
        #      bound, and lands in the region where negloglik's plausibility
        #      guard fires.
        #   2. That guard returns a flat constant (1e10), so the line search
        #      backtracks against a plateau with no gradient, exhausts its
        #      budget, and stops where it started.
        #
        # Verified NOT to be the cause: ForwardDiff is correct (gradients
        # agree with central differences to ~8 significant digits), and
        # derivative-free COBYLA recovers r and K to ~1e-11 from the same
        # start with the same bounds.
        #
        # Uniform restarts do not rescue it: with Kmax_mult=1000 the true
        # K sits in the bottom 0.1% of the sampled range.
        #
        # Real (noisy) fits are unaffected — the saved Jalisco GRM/NB1 fit
        # recovers r=0.793, p=0.840, a=1.346, K=7341, nowhere near its warm
        # start. Candidate fixes (log-space reparameterization of r/K/alpha,
        # a smooth infeasibility penalty, retcode gating) are tracked
        # separately and must be validated against the MATLAB reference
        # before adoption.
        Random.seed!(20260726)
        timevect = collect(1.0:30.0)
        ytrue = solve_incidence(:lm, 0.35, 1.0, 1.0, 4000.0, 5.0, timevect)
        res = fit_growth_model(:lm, timevect, ytrue; dist = :normal, n_restarts = 8)
        @test_broken isapprox(res.r, 0.35; rtol = 0.15)
    end

    @testset "bootstrap" begin
        Random.seed!(1234)
        timevect = collect(1.0:25.0)
        ydata = solve_incidence(:lm, 0.35, 1.0, 1.0, 3000.0, 5.0, timevect)

        curve = simulate_noisy_curve(ydata, :poisson, 1.0)
        @test length(curve) == length(ydata)
        @test all(>=(0), curve)

        res = fit_growth_model(:lm, timevect, ydata; dist = :normal, n_restarts = 3)
        boot = run_bootstrap(:lm, timevect, ydata, res;
                             dist = :normal, M = 10, forecast_horizon = 2)
        @test boot isa BootstrapResult
        @test boot.flag === :lm
        @test size(boot.params, 2) == 5
        @test size(boot.fit_curves, 1) == length(timevect)
        @test size(boot.forecast_curves, 1) == length(timevect) + 2
        @test boot.n_success > 0
    end

    @testset "identifiability screen" begin
        timevect = collect(1.0:30.0)
        # EXP has a single free parameter, so collinearity is not applicable.
        out_exp = screen_identifiability(:exp, timevect,
                                         (r = 0.3, p = 1.0, a = 1.0, K = 1.0, I0 = 5.0))
        @test length(out_exp.free_names) == MODEL_NPARAMS[:exp]
        @test ismissing(out_exp.identifiable)
        @test isnan(out_exp.collinearity)

        # GRM has four free parameters; the screen should return a finite
        # collinearity index and name the worst-correlated pair.
        out_grm = screen_identifiability(:grm, timevect,
                                         (r = 0.3, p = 0.9, a = 1.0, K = 3000.0, I0 = 5.0))
        @test length(out_grm.free_names) == MODEL_NPARAMS[:grm]
        @test isfinite(out_grm.collinearity)
        @test out_grm.collinearity >= 1.0
        @test out_grm.identifiable isa Bool
    end

end
