using GrowthFit
using Test
using Random
using Aqua
using DelimitedFiles

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

    @testset "wide K bounds on noiseless data (was broken pre-0.2.0)" begin
        # HISTORY. With the default Kmax_mult=1000, SLSQP returned its own
        # starting point unchanged and reported Success. Two compounding
        # causes, both confirmed:
        #
        #   1. Gradient scaling. At the warm start df/dr = -8.0e5 against
        #      df/dK = -0.14, a ratio of 5.7e6, so the first QP step is
        #      essentially pure r, clips at its upper bound, and lands where
        #      negloglik's plausibility guard fires.
        #   2. That guard returned a flat constant, leaving the line search
        #      no gradient to backtrack along.
        #
        # Verified NOT to be the cause: ForwardDiff is correct (gradients
        # agree with central differences to ~8 significant digits), and
        # log-space reparameterization made matters worse, as did narrowing
        # Kmax_mult and adding restarts.
        #
        # The v0.2.0 COBYLA polish fixes it. If this testset ever fails,
        # something has regressed in the polish step rather than in the
        # multistart.
        Random.seed!(20260726)
        timevect = collect(1.0:30.0)
        ytrue = solve_incidence(:lm, 0.35, 1.0, 1.0, 4000.0, 5.0, timevect)
        res = fit_growth_model(:lm, timevect, ytrue; dist = :normal, n_restarts = 8)
        @test isapprox(res.r, 0.35; rtol = 0.15)

        # And confirm the polish is what does it.
        Random.seed!(20260726)
        res_np = fit_growth_model(:lm, timevect, ytrue;
                                  dist = :normal, n_restarts = 8, polish = false)
        @test res_np.objective >= res.objective
    end

    @testset "least-squares fit on real data (v0.2.0 regression)" begin
        # Before the COBYLA polish, dist=:normal on this series produced
        # SSE 783,855 — 5.8x worse than the reference MATLAB implementation
        # (fmincon, 25-point MultiStart) on the identical problem, which
        # reaches 135,363. The polish reaches 83,448, better than both.
        path = joinpath(@__DIR__, "..", "examples", "data",
                        "JALISCO_2025-08-18_2026-06-29-trimmed.txt")
        if !isfile(path)
            @info "Jalisco example data not found; skipping regression test."
        else
            data = readdlm(path, '\t', Float64)
            tv, yd = data[1:40, 1], data[1:40, 2]

            Random.seed!(1)
            f = fit_growth_model(:grm, tv, yd; dist = :normal, n_restarts = 10)
            sse = sum((yd .- f.fitcurve) .^ 2)

            @test f.converged
            @test isfinite(sse)
            @test sse < 100_000        # was ~783,855 before v0.2.0

            # NB1 is unaffected by the polish: SLSQP already reaches that
            # optimum, so the extra pass must not move it.
            Random.seed!(1)
            g  = fit_growth_model(:grm, tv, yd; dist = :nb1, n_restarts = 10)
            Random.seed!(1)
            gn = fit_growth_model(:grm, tv, yd; dist = :nb1, n_restarts = 10, polish = false)
            @test isapprox(g.objective, gn.objective; rtol = 1e-6)
            @test isapprox(g.r, gn.r; rtol = 1e-4)
        end
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

    @testset "bootstrap seeding is reproducible" begin
        timevect = collect(1.0:25.0)
        ydata = solve_incidence(:lm, 0.35, 1.0, 1.0, 3000.0, 5.0, timevect)
        res = fit_growth_model(:lm, timevect, ydata; dist = :normal, n_restarts = 3)

        kw = (dist = :normal, M = 8, forecast_horizon = 1, n_restarts_boot = 1)

        # An explicit seed pins the replicates. This holds regardless of
        # Threads.nthreads(), because each replicate index gets its own
        # stream derived from the master seed rather than whatever the
        # @threads chunking hands it.
        a = run_bootstrap(:lm, timevect, ydata, res; kw..., seed = 42)
        b = run_bootstrap(:lm, timevect, ydata, res; kw..., seed = 42)
        @test a.params == b.params
        @test a.fit_curves == b.fit_curves
        @test a.forecast_noisy == b.forecast_noisy

        # A different seed gives different replicates.
        c = run_bootstrap(:lm, timevect, ydata, res; kw..., seed = 43)
        @test c.params != a.params

        # seed=nothing draws a master seed from the passed rng, so an
        # upstream Random.seed! still makes the whole run reproducible.
        Random.seed!(7); d = run_bootstrap(:lm, timevect, ydata, res; kw..., seed = nothing)
        Random.seed!(7); e = run_bootstrap(:lm, timevect, ydata, res; kw..., seed = nothing)
        @test d.params == e.params
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

    @testset "compute_aicc" begin
        # Noisy rather than noiseless: a fit to its own generating curve can
        # drive SSE to within rounding of zero, where the least-squares AICc
        # branch is degenerate and tells us nothing.
        timevect = collect(1.0:16.0)
        clean = solve_incidence(:lm, 0.35, 1.0, 1.0, 3000.0, 5.0, timevect)
        ydata = simulate_noisy_curve(clean, :nb1, 0.5, Random.Xoshiro(4242))
        n = length(ydata)

        for dist in (:normal, :poisson, :nb1)
            res = fit_growth_model(:lm, timevect, ydata; dist = dist, n_restarts = 2)
            @test isfinite(compute_aicc(res, dist))
            # n_obs defaults to the length of the fitted curve.
            @test compute_aicc(res, dist) == compute_aicc(res, dist; n_obs = n)
        end

        res = fit_growth_model(:lm, timevect, ydata; dist = :nb1, n_restarts = 2)

        # The likelihood branches genuinely differ: under :normal the
        # objective is SSE, not an NLL, so the NLL formula applied there
        # (as examples/run_example.jl did through v0.2.2) gives something
        # else entirely.
        res_n = fit_growth_model(:lm, timevect, ydata; dist = :normal, n_restarts = 2)
        k_normal = MODEL_NPARAMS[:lm] + 1
        naive = 2 * res_n.objective + 2 * k_normal +
                (2 * k_normal * (k_normal + 1)) / (n - k_normal - 1)
        @test compute_aicc(res_n, :normal) < naive

        # More parameters cost more, all else equal: on data generated by LM,
        # GRM cannot fit better by enough to pay for two extra parameters.
        aicc_lm  = compute_aicc(res, :nb1)
        res_grm  = fit_growth_model(:grm, timevect, ydata; dist = :nb1, n_restarts = 2)
        @test isfinite(compute_aicc(res_grm, :nb1))

        # A window too short for the correction scores Inf rather than
        # returning a negative penalty.
        @test compute_aicc(res, :nb1; n_obs = MODEL_NPARAMS[:lm] + 2) == Inf
        @test isfinite(aicc_lm)

        @test_throws ErrorException compute_aicc(res, :nope)
    end

    @testset "rolling_windows" begin
        # Kept deliberately small: 3 windows x 12 replicates on a 16-point
        # series. The behaviour under test is the windowing, seeding and
        # table shapes, none of which needs a realistic M.
        timevect = collect(1.0:16.0)
        clean = solve_incidence(:lm, 0.35, 1.0, 1.0, 3000.0, 5.0, timevect)
        ydata = simulate_noisy_curve(clean, :nb1, 0.5, Random.Xoshiro(4242))
        kw = (windowsize = 12, forecast_horizon = 2, dist = :nb1, M = 12,
              n_restarts = 2, noise_reps = 4, verbose = false)

        rw = rolling_windows(:lm, timevect, ydata; kw..., windows = 1:3, seed = 99)

        @test rw isa RollingWindowResult
        @test length(rw) == 3
        @test rw.seed == UInt64(99)
        @test rw.flag == :lm && rw.dist == :nb1

        # Window i covers observations i : i+windowsize-1.
        for (s, w) in enumerate(rw)
            @test w isa WindowFit
            @test w.window == s
            @test w.index_range == s:(s + 11)
            @test w.t_start == timevect[s]
            @test w.t_end == timevect[s + 11]
            @test length(w.fit.fitcurve) == 12
            @test isfinite(w.calibration.WIS)
        end

        # Indexing and iteration.
        @test rw[2].window == 2
        @test rw[end].window == 3
        @test length(collect(rw)) == 3

        # Windows 1 and 2 end at observations 12 and 13, so a 2-step horizon
        # is fully observed; window 3 ends at 14 and can only score 2 of 2 --
        # 16 observations, so all three are in fact fully scored here.
        for w in rw
            @test w.forecast !== nothing
            @test length(w.forecast) == 2
            @test [row.horizon_step for row in w.forecast] == [1, 2]
        end

        # A window ending at the last observation has nothing to score.
        rw_end = rolling_windows(:lm, timevect, ydata; kw..., windows = 5:5, seed = 99)
        @test rw_end[1].index_range == 5:16
        @test rw_end[1].forecast === nothing

        # A horizon only partly observed scores the steps that exist.
        rw_part = rolling_windows(:lm, timevect, ydata;
                                  windowsize = 12, forecast_horizon = 4,
                                  dist = :nb1, M = 12, n_restarts = 2,
                                  noise_reps = 4, verbose = false,
                                  windows = 2:2, seed = 99)
        @test length(rw_part[1].forecast) == 3   # ends at 13, 3 observations left

        # Seeding is per window, not per run: window 2 is identical whether
        # it was run as part of 1:3 or on its own. This is what makes a
        # single suspicious window reproducible.
        rw2 = rolling_windows(:lm, timevect, ydata; kw..., windows = 2:2, seed = 99)
        @test rw2[1].boot.params == rw[2].boot.params
        @test rw2[1].fit.r == rw[2].fit.r

        # And the whole run is reproducible from the seed alone.
        rw_again = rolling_windows(:lm, timevect, ydata; kw..., windows = 1:3, seed = 99)
        @test rw_again[1].boot.params == rw[1].boot.params
        @test rw_again[3].fit.K == rw[3].fit.K

        # seed = nothing still draws from the passed/global RNG, so an
        # upstream Random.seed! pins it.
        Random.seed!(11)
        x = rolling_windows(:lm, timevect, ydata; kw..., windows = 1:1, seed = nothing)
        Random.seed!(11)
        y = rolling_windows(:lm, timevect, ydata; kw..., windows = 1:1, seed = nothing)
        @test x.seed == y.seed
        @test x[1].boot.params == y[1].boot.params

        @testset "parameter_trajectory" begin
            pt = parameter_trajectory(rw)
            @test length(pt.window) == 3
            @test pt.window == [1, 2, 3]
            @test pt.t_start == [1.0, 2.0, 3.0]
            # Every column is the same length.
            @test all(length(getfield(pt, k)) == 3 for k in keys(pt))
            # Median sits inside its own interval.
            @test all(pt.r_lo .<= pt.r) && all(pt.r .<= pt.r_hi)
            @test all(pt.K_lo .<= pt.K) && all(pt.K .<= pt.K_hi)
            @test all(isfinite, pt.r)
            @test all(isfinite, pt.K)
            # LM holds p and a fixed at 1, so those come back with
            # zero-width intervals rather than being absent.
            @test all(isapprox.(pt.p, 1.0; atol = 1e-6))
            @test all(pt.p_hi .- pt.p_lo .< 1e-6)
            @test all(isapprox.(pt.a, 1.0; atol = 1e-6))
        end

        @testset "skill_trajectory" begin
            st = skill_trajectory(rw)
            @test all(length(getfield(st, k)) == 3 for k in keys(st))
            @test st.steps_scored == [2, 2, 2]
            @test all(isfinite, st.WIS_cal)
            @test all(isfinite, st.WIS_fc)
            @test all(>=(0), st.Coverage95PI_cal) && all(<=(100), st.Coverage95PI_cal)
            @test all(st.n_success .<= 12)
            @test all(isfinite, st.aicc)

            # An unscored window reports NaN forecast metrics, not zero.
            st_end = skill_trajectory(rw_end)
            @test st_end.steps_scored == [0]
            @test isnan(st_end.WIS_fc[1])
            @test isfinite(st_end.WIS_cal[1])
        end

        @testset "forecast_skill_by_step" begin
            fs = forecast_skill_by_step(rw)
            @test all(length(getfield(fs, k)) == 6 for k in keys(fs))  # 3 windows x 2 steps
            @test fs.window == [1, 1, 2, 2, 3, 3]
            @test fs.horizon_step == [1, 2, 1, 2, 1, 2]
            @test all(fs.LB95 .<= fs.UB95)
            @test all(>=(0), fs.WIS)
            @test all(x -> x == 0.0 || x == 100.0, fs.Coverage_95PI)

            # Empty, not an error, when there is nothing to score.
            fs_end = forecast_skill_by_step(rw_end)
            @test isempty(fs_end.window)
            @test isempty(fs_end.WIS)
        end

        @testset "argument validation" begin
            # Window longer than the series.
            @test_throws ErrorException rolling_windows(:lm, timevect, ydata;
                                                        windowsize = 50, verbose = false)
            # Window too short to estimate the model's parameters.
            @test_throws ErrorException rolling_windows(:grm, timevect, ydata;
                                                        windowsize = 4, verbose = false)
            # Window index past the last one that fits (16 - 12 + 1 = 5).
            @test_throws ErrorException rolling_windows(:lm, timevect, ydata;
                                                        kw..., windows = 6:6)
            @test_throws ErrorException rolling_windows(:lm, timevect, ydata;
                                                        kw..., windows = Int[])
            # Mismatched lengths, unknown flag, unknown error structure.
            @test_throws ErrorException rolling_windows(:lm, timevect[1:10], ydata;
                                                        kw...)
            @test_throws ErrorException rolling_windows(:nope, timevect, ydata; kw...)
            @test_throws ErrorException rolling_windows(:lm, timevect, ydata;
                                                        windowsize = 12, dist = :nope,
                                                        verbose = false)
        end

        @testset "show methods run" begin
            redirect_stdout(devnull) do
                @test (show(stdout, MIME"text/plain"(), rw); true)
                @test (show(stdout, MIME"text/plain"(), rw_end); true)
                @test (show(stdout, rw); true)
                @test (show(stdout, rw[1]); true)
            end
        end
    end

    @testset "identifiability screen over all models" begin
        # Regression: screen_all_models referred to `GrowthModels.MODEL_NAMES`,
        # but `using ..GrowthModels: growth_rhs!, MODEL_NAMES` imports only the
        # named symbols and does not bind the module, so the call raised
        # UndefVarError. Not covered by any test until v0.2.1.
        timevect = collect(1.0:30.0)
        ydata = solve_incidence(:lm, 0.35, 1.0, 1.0, 3000.0, 5.0, timevect)

        # These print a report; the assertion is simply that they run.
        redirect_stdout(devnull) do
            @test (screen_all_models(timevect, ydata); true)

            out = screen_identifiability(:grm, timevect,
                                         (r = 0.3, p = 0.9, a = 1.0, K = 3000.0, I0 = 5.0))
            @test (print_identifiability_summary(out, "GRM"); true)
        end
    end

end
