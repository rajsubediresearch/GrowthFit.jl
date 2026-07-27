"""
PlotGrowthFit.jl -- fit/forecast plot matching the MATLAB toolbox convention
confirmed during the R port: cyan spaghetti drawn from the OBSERVATION-NOISE
matrix (full calibration+forecast range), with clean parameter-only fit
curves drawn in gray on top for the calibration period only. Also produces
parameter histograms with 95% CI lines, degenerate (fixed) parameters shown
as a flat spike rather than omitted -- matches the MATLAB panel style.
"""
module PlotGrowthFit

using Plots, Statistics

using ..GrowthModels
using ..BootstrapUncertainty: BootstrapResult

export plot_fit_and_forecast

function ci95(v::AbstractVector)
    v = filter(isfinite, v)
    isempty(v) && return (NaN, NaN, NaN)
    return (median(v), quantile(v, 0.025), quantile(v, 0.975))
end

"""
    plot_fit_and_forecast(result, timevect, ydata, data_all, timevect_all;
                          forecast_horizon=0, title_str="",
                          title_fontsize=14, hist_title_fontsize=8,
                          axis_fontsize=10, tick_fontsize=8,
                          hist_color=:darkblue, ci_color=:red,
                          boot_color=RGBA(0,1,1,0.08), fit_color=RGBA(0.4,0.4,0.4,0.3),
                          median_color=:red, pi_color=:red, point_color=:blue,
                          figsize=(1400,900))

Build the combined parameter-histogram + fit/forecast panel. Returns the
Plots.jl figure object (save separately with `savefig`, or keep the object
itself -- e.g. via JLD2 -- for later re-editing, same purpose as the R
version's habit of saving RDS objects alongside PNGs).

All appearance options have keyword defaults matching the toolbox's
original look, so existing calls (e.g. in run_example.jl) are unaffected --
these exist so a separate, simple "edit and re-plot" script can override
them without needing to touch this file.
"""
function plot_fit_and_forecast(result::BootstrapResult, timevect::AbstractVector,
                               ydata::AbstractVector, data_all::AbstractVector,
                               timevect_all::AbstractVector;
                               forecast_horizon::Int=0, title_str::String="",
                               title_fontsize::Int=14, hist_title_fontsize::Int=8,
                               axis_fontsize::Int=10, tick_fontsize::Int=8,
                               hist_color=:darkblue, ci_color=:red,
                               boot_color=RGBA(0,1,1,0.08), fit_color=RGBA(0.4,0.4,0.4,0.3),
                               median_color=:red, pi_color=:red, point_color=:blue,
                               point_size::Real=3, line_width::Real=2,
                               figsize::Tuple{Int,Int}=(1400,900))
    param_names = ["r", "p", "a", "K"]
    hist_plots = Vector{Any}(undef, 4)
    for (i, name) in enumerate(param_names)
        vals = result.params[:, i]
        med, lo, hi = ci95(vals)
        is_fixed = (maximum(vals) - minimum(vals)) < 1e-8 * max(abs(med), 1)
        if name == "K"
            label_val = string(round(Int, med))
            lo_str, hi_str = string(round(Int, lo)), string(round(Int, hi))
        else
            label_val = string(round(med, digits=3))
            lo_str, hi_str = string(round(lo, digits=3)), string(round(hi, digits=3))
        end
        title_i = "$name=$label_val (95% CI:$lo_str,$hi_str)"

        if is_fixed
            hist_plots[i] = bar([med], [1.0]; bar_width=max(abs(med), 1)*0.05,
                                title=title_i, xlabel=name, ylabel="Frequency",
                                legend=false, titlefontsize=hist_title_fontsize,
                                guidefontsize=axis_fontsize, tickfontsize=tick_fontsize,
                                color=hist_color)
        else
            hist_plots[i] = histogram(vals; bins=10, title=title_i, xlabel=name,
                                      ylabel="Frequency", legend=false,
                                      titlefontsize=hist_title_fontsize,
                                      guidefontsize=axis_fontsize, tickfontsize=tick_fontsize,
                                      color=hist_color, linecolor=hist_color)
            vline!(hist_plots[i], [lo, hi]; color=ci_color, linestyle=:dash, linewidth=line_width)
        end
    end

    n_cal = length(timevect)
    timevect_forecast = forecast_horizon > 0 ?
        vcat(timevect, timevect[end] .+ (1:forecast_horizon)) : timevect

    fitplot = plot(; title=title_str, xlabel="Time", ylabel="Cases", legend=false,
                   titlefontsize=title_fontsize, guidefontsize=axis_fontsize,
                   tickfontsize=tick_fontsize)

    # Cyan spaghetti: noisy forecast realizations, FULL range (matches
    # MATLAB's plot(timevect2, forecast_model12, 'c') convention confirmed
    # against a real MATLAB screenshot during the R port).
    for j in 1:size(result.forecast_noisy, 2)
        col = result.forecast_noisy[:, j]
        any(!isfinite, col) && continue
        plot!(fitplot, timevect_forecast, col; color=boot_color, linewidth=0.5)
    end

    # 95% PI band from the same noisy matrix
    lb = [quantile(filter(isfinite, result.forecast_noisy[t, :]), 0.025) for t in 1:length(timevect_forecast)]
    ub = [quantile(filter(isfinite, result.forecast_noisy[t, :]), 0.975) for t in 1:length(timevect_forecast)]
    med = [median(filter(isfinite, result.forecast_noisy[t, :])) for t in 1:length(timevect_forecast)]
    plot!(fitplot, timevect_forecast, lb; color=pi_color, linestyle=:dash, linewidth=line_width)
    plot!(fitplot, timevect_forecast, ub; color=pi_color, linestyle=:dash, linewidth=line_width)
    plot!(fitplot, timevect_forecast, med; color=median_color, linewidth=line_width)

    # Gray clean fit curves, calibration period only, drawn ON TOP of cyan
    for j in 1:size(result.fit_curves, 2)
        col = result.fit_curves[:, j]
        any(!isfinite, col) && continue
        plot!(fitplot, timevect, col; color=fit_color, linewidth=0.5)
    end

    # Data points
    scatter!(fitplot, timevect_all, data_all; color=point_color, markershape=:circle,
            markersize=point_size, markerstrokewidth=1.5)

    if forecast_horizon > 0
        vline!(fitplot, [timevect[end]]; color=:black, linestyle=:dash, linewidth=1.5)
    end

    layout = @layout [a b c d; e{0.8h}]
    combined = plot(hist_plots..., fitplot; layout=layout, size=figsize)
    return combined
end

end # module
