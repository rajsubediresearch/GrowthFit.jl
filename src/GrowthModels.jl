"""
GrowthModels.jl -- ODE definitions for the seven standard phenomenological
growth models, matching the flag convention used throughout this project:

    flag = :exp   dx = r*x                          (unbounded exponential)
    flag = :ggm   dx = r*x^p                         (generalized growth model)
    flag = :glm   dx = r*x^p*(1 - x/K)               (generalized logistic)
    flag = :grm   dx = r*x^p*(1 - (x/K)^a)           (generalized Richards)
    flag = :lm    dx = r*x*(1 - x/K)                 (logistic)
    flag = :rich  dx = r*x*(1 - (x/K)^a)             (Richards)
    flag = :gom   dx = r*x*exp(-a*t)                 (Gompertz, no K term)

Parameters are always passed as a single NamedTuple-like vector
p = (r, p_exp, a, K) -- unused entries for a given flag are simply ignored,
same spirit as the R/MATLAB toolboxes where e.g. GLM ignores `a`.
"""
module GrowthModels

export growth_rhs!, MODEL_NPARAMS, MODEL_NAMES

# Right-hand side for DifferentialEquations.jl's in-place ODE interface.
# u[1] = cumulative incidence. params = (flag, r, p_exp, a, K).
function growth_rhs!(du, u, params, t)
    flag, r, p_exp, a, K = params
    x = max(u[1], 1e-10)  # guard against a solver step dipping below 0

    if flag == :exp
        du[1] = r * x
    elseif flag == :ggm
        du[1] = r * x^p_exp
    elseif flag == :glm
        du[1] = r * x^p_exp * (1 - x / K)
    elseif flag == :grm
        du[1] = r * x^p_exp * (1 - (x / K)^a)
    elseif flag == :lm
        du[1] = r * x * (1 - x / K)
    elseif flag == :rich
        du[1] = r * x * (1 - (x / K)^a)
    elseif flag == :gom
        du[1] = r * x * exp(-a * t)
    else
        error("Unknown growth model flag: $flag")
    end
    return nothing
end

# Number of free trajectory parameters per model (excludes I0, matches
# get_nparams() in the R/MATLAB toolboxes)
const MODEL_NPARAMS = Dict(
    :exp  => 1,  # r
    :ggm  => 2,  # r, p
    :glm  => 3,  # r, p, K
    :grm  => 4,  # r, p, a, K
    :lm   => 2,  # r, K
    :rich => 3,  # r, a, K
    :gom  => 2,  # r, a
)

const MODEL_NAMES = Dict(
    :exp  => "EXP",
    :ggm  => "GGM",
    :glm  => "GLM",
    :grm  => "GRM",
    :lm   => "LM",
    :rich => "RICH",
    :gom  => "GOM",
)

end # module
