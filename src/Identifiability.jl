"""
Identifiability.jl -- practical (structural) identifiability screening via
local sensitivity + collinearity, direct port of the validated R
identifiability.R logic from this project's earlier work.

Screens whether the FREE TRAJECTORY PARAMETERS (r, p, a, K -- never the
dispersion parameter alpha, which has zero effect on the mean trajectory
and would corrupt the collinearity number if included) can be jointly
distinguished from the shape of the data, given a candidate parameter
point. This is a LOCAL diagnostic: it evaluates sensitivity around ONE
parameter point (e.g. an initial guess or a fitted point estimate), not a
global property of the model.

Rule of thumb (Omlin & Reichert 2001, same threshold used in the R
version): collinearity < ~20 is considered identifiable; higher values
indicate the parameter combination is practically unidentifiable from this
data window -- i.e. many different parameter combinations would produce
nearly the same trajectory, so don't over-interpret individual parameter
values even if the FIT looks good.

No fitting required -- can be run on an initial guess BEFORE spending time
fitting, as a cheap pre-screen (this was the original motivation for the
R version too: "which models are even worth fitting on this data window").
"""
module Identifiability

using LinearAlgebra, Statistics

using ..GrowthModels: growth_rhs!, MODEL_NAMES
using ..FitGrowthModel: solve_incidence

export screen_identifiability, print_identifiability_summary, screen_all_models

# Which of (r, p_exp, a, K) are actually free (not fixed) for a given model
# flag -- same nesting logic as get_bounds() in the R/MATLAB toolboxes, and
# as GrowthModels.MODEL_NPARAMS.
const FREE_PARAM_INDICES = Dict(
    :exp  => [1],           # r
    :ggm  => [1, 2],        # r, p
    :glm  => [1, 2, 4],     # r, p, K
    :grm  => [1, 2, 3, 4],  # r, p, a, K
    :lm   => [1, 4],        # r, K
    :rich => [1, 3, 4],     # r, a, K
    :gom  => [1, 3],        # r, a
)
const PARAM_LABELS = ["r", "p", "a", "K"]

"""
    screen_identifiability(flag, timevect, params0; threshold=20.0,
                           rel_step=1e-4)

Evaluate local trajectory-parameter identifiability at `params0 = (r, p,
a, K, I0)` for the given growth model. Returns a NamedTuple:

    (free_names, collinearity, identifiable, worst_pair, note)

`worst_pair` names the two parameters with the highest pairwise
collinearity (the ones most likely trading off against each other), or
`nothing` if fewer than 2 free parameters.
"""
function screen_identifiability(flag::Symbol, timevect::AbstractVector,
                                params0::NamedTuple; threshold::Float64=20.0,
                                rel_step::Float64=1e-4, debug::Bool=false)
    free_idx = FREE_PARAM_INDICES[flag]
    free_names = PARAM_LABELS[free_idx]
    n_free = length(free_idx)

    if n_free < 2
        return (free_names=free_names, collinearity=NaN, identifiable=missing,
                worst_pair=nothing,
                note="Fewer than 2 free trajectory parameters; collinearity screening not applicable.")
    end

    p0_full = [params0.r, params0.p, params0.a, params0.K]
    I0 = params0.I0

    base_curve = solve_incidence(flag, p0_full[1], p0_full[2], p0_full[3], p0_full[4], I0, timevect)
    if base_curve === nothing || any(!isfinite, base_curve)
        return (free_names=free_names, collinearity=NaN, identifiable=missing,
                worst_pair=nothing,
                note="ODE solve failed at params0; cannot compute sensitivity.")
    end

    n_t = length(timevect)
    S = zeros(n_t, n_free)  # sensitivity matrix: d(trajectory)/d(param_i), scaled

    zero_sens = String[]
    for (col, idx) in enumerate(free_idx)
        p_perturbed = copy(p0_full)
        h = rel_step * max(abs(p0_full[idx]), 1e-6)
        p_perturbed[idx] += h
        curve_h = solve_incidence(flag, p_perturbed[1], p_perturbed[2], p_perturbed[3],
                                  p_perturbed[4], I0, timevect)
        if curve_h === nothing || any(!isfinite, curve_h)
            push!(zero_sens, PARAM_LABELS[idx])
            continue
        end
        # Scaled sensitivity (dy/dp * p / y_typical), matching FME::sensFun's
        # default scaling so the collinearity index is comparable to the R
        # version's numbers.
        y_typical = max(mean(abs.(base_curve)), 1e-6)
        S[:, col] = ((curve_h .- base_curve) ./ h) .* (p0_full[idx] / y_typical)
    end

    if !isempty(zero_sens)
        keep_cols = [i for i in 1:n_free if !(free_names[i] in zero_sens)]
        if length(keep_cols) < 2
            return (free_names=free_names, collinearity=NaN, identifiable=missing,
                    worst_pair=nothing,
                    note="Parameter(s) $(join(zero_sens, ", ")) had zero sensitivity; " *
                         "fewer than 2 non-degenerate free parameters remain.")
        end
        S = S[:, keep_cols]
        free_names = free_names[keep_cols]
        n_free = length(keep_cols)
    end

    # Collinearity index (Omlin & Reichert 2001): normalize each column to
    # unit length, then collinearity = 1/sqrt(smallest eigenvalue of the
    # normalized S'S). A small eigenvalue means some linear combination of
    # columns is nearly zero -- i.e. those parameters trade off against each
    # other with almost no effect on the trajectory.
    function collinearity_index(Smat::AbstractMatrix)
        norms = [norm(Smat[:, j]) for j in 1:size(Smat, 2)]
        if any(n -> n < 1e-12, norms)
            return Inf
        end
        Snorm = Smat ./ reshape(norms, 1, :)
        M = Snorm' * Snorm
        eigvals_M = eigvals(Symmetric(M))
        min_eig = max(minimum(eigvals_M), 1e-12)
        return 1 / sqrt(min_eig)
    end

    full_collin = collinearity_index(S)

    if debug
        println("DEBUG [$flag]: free_names=$free_names")
        println("DEBUG [$flag]: S = ")
        display(S)
        println()
        println("DEBUG [$flag]: full_collin=$full_collin")
    end

    # Worst pair: highest collinearity among all 2-column subsets
    worst_pair = nothing
    if n_free > 2
        worst_val = -Inf
        for i in 1:n_free, j in (i+1):n_free
            c = collinearity_index(S[:, [i, j]])
            if debug
                println("DEBUG [$flag]: pair ($(free_names[i]),$(free_names[j])) collinearity=$c")
            end
            if c > worst_val
                worst_val = c
                worst_pair = (names=(free_names[i], free_names[j]), collinearity=c)
            end
        end
    elseif n_free == 2
        worst_pair = (names=(free_names[1], free_names[2]), collinearity=full_collin)
    end

    identifiable = full_collin < threshold
    note = isempty(zero_sens) ? nothing :
        "Parameter(s) $(join(zero_sens, ", ")) excluded (zero sensitivity to trajectory)."

    return (free_names=free_names, collinearity=full_collin, identifiable=identifiable,
            worst_pair=worst_pair, note=note)
end

"""
    print_identifiability_summary(result, model_name; threshold=20.0)

Print a compact, human-readable summary of a `screen_identifiability(...)`
result.
"""
function print_identifiability_summary(result, model_name::String; threshold::Float64=20.0)
    println("\n--- Identifiability screen ($model_name) ---")
    if ismissing(result.identifiable)
        println("  ", something(result.note, "Collinearity not available."))
        return nothing
    end
    println("Free trajectory parameters: ", join(result.free_names, ", "))
    status = result.identifiable ? "IDENTIFIABLE" : "NOT RELIABLY IDENTIFIABLE"
    println("Full-set collinearity: $(round(result.collinearity, digits=1))  ($status, threshold=$threshold)")
    if result.worst_pair !== nothing
        wp = result.worst_pair
        println("Most collinear pair: $(wp.names[1]) & $(wp.names[2])  (collinearity=$(round(wp.collinearity, digits=1)))")
    end
    if result.note !== nothing
        println("Note: ", result.note)
    end
    if !result.identifiable
        println("  -> Treat parameter values for this fit with caution: this data window")
        println("     cannot jointly pin down this parameter combination. Consider a")
        println("     simpler model, a longer calibration window, or report only the")
        println("     identifiable subset/combinations.")
    end
    return nothing
end

"""
    screen_all_models(timevect, ydata; flags=[:exp,:ggm,:glm,:grm,:lm,:rich,:gom],
                      threshold=20.0)

Cheap pre-fit screen across all (or a chosen subset of) growth models,
using a simple heuristic initial guess for each (no optimization needed --
this is meant to run BEFORE spending time on a full fit/bootstrap).
Prints a summary table and returns a Vector of NamedTuples, one per model.
"""
function screen_all_models(timevect::AbstractVector, ydata::AbstractVector;
                           flags::Vector{Symbol}=[:exp, :ggm, :glm, :grm, :lm, :rich, :gom],
                           threshold::Float64=20.0)
    I0 = ydata[1]
    data_sum = sum(ydata)
    results = NamedTuple[]

    println("\n=== Identifiability pre-screen (trajectory parameters only) ===")
    for flag in flags
        n_free = length(FREE_PARAM_INDICES[flag])
        if n_free > length(ydata)
            continue
        end
        # Simple heuristic initial guess (same spirit as initialParams() in
        # the R/MATLAB toolboxes) -- no fitting, just a plausible starting
        # point for the sensitivity evaluation.
        params0 = (r=0.5, p=0.9, a=1.0, K=min(data_sum, 1e5), I0=I0)
        result = screen_identifiability(flag, timevect, params0; threshold=threshold)
        model_str = GrowthModels.MODEL_NAMES[flag]
        push!(results, merge((model=model_str, flag=flag), result))

        status = ismissing(result.identifiable) ? "n/a" :
                 (result.identifiable ? "OK" : "FLAGGED")
        collin_str = isnan(result.collinearity) ? "  NA" : lpad(round(result.collinearity, digits=1), 6)
        println("  $(rpad(model_str,5))  collinearity=$collin_str  [$status]")
    end
    println("\nRule of thumb: collinearity < $threshold = identifiable.")
    println("This is a LOCAL screen at a heuristic starting point, not the fitted")
    println("optimum -- treat it as a quick filter before fitting, not a final verdict.")

    return results
end

end # module
