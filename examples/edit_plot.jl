"""
edit_plot.jl -- reload a saved fit (.jld2 file from run_example.jl) and
re-draw the plot with different fonts, sizes, or colors, then save as a
PNG or PDF.

HOW TO USE:
  1. Run run_example.jl first (or any script that saves a *_fit_object.jld2
     file to output/) -- you need that file to exist before running this.
  2. Edit the settings in "PART 1: WHAT TO CHANGE" below.
  3. In the Julia REPL (same folder, same "using" packages already
     installed):

        include("edit_plot.jl")

  4. Repeat: change a setting, re-run `include("edit_plot.jl")`, look at
     the new PNG/PDF in output/. No need to re-fit the model -- this only
     re-draws the plot from the numbers that are already saved.

No new packages needed beyond what run_example.jl already required.
"""

using JLD2, Plots

using GrowthFit
# Qualified calls below (GrowthModels.MODEL_NAMES, PlotGrowthFit.plot_fit_and_forecast)
# need the submodule names bound explicitly.
using GrowthFit: GrowthModels, PlotGrowthFit


# =============================================================================
# PART 1: WHAT TO CHANGE
# =============================================================================

# Which saved fit to load and re-plot (must already exist in output/ --
# run run_example.jl first if you haven't).
jld2_path = joinpath("output", "JALISCO MEXICO_GRM_nb1_fit_object.jld2")

# Plot title (shown above the fit/forecast panel). Leave as "" to use the
# model name automatically.
title_str = "GRM Model fit and 3 week ahead forecast for Jalisco"

# --- Font sizes (bigger number = bigger text) ---
title_fontsize       = 14   # main plot title
hist_title_fontsize  = 8    # histogram titles (r=..., p=..., etc.)
axis_fontsize        = 10   # axis labels ("Time", "Cases", "Frequency")
tick_fontsize        = 8    # the numbers along each axis

# --- Colors --- (see https://docs.juliaplots.org/latest/generated/colorschemes/
# for named color options, or use RGBA(red, green, blue, alpha) for custom,
# each 0.0-1.0)
hist_color   = :darkblue          # histogram bar fill/border
ci_color     = :red               # dashed 95% CI lines on histograms
boot_color   = RGBA(0, 1, 1, 0.08)   # bootstrap spaghetti (cyan, semi-transparent)
fit_color    = RGBA(0.4, 0.4, 0.4, 0.3)  # gray calibration-fit lines
median_color = :red               # median forecast line
pi_color     = :red               # dashed 95% prediction interval lines
point_color  = :blue              # observed data points

# --- Sizes ---
point_size = 3     # observed data point marker size
line_width = 2      # CI/PI line thickness
figsize    = (1400, 900)   # overall figure size in pixels (width, height)

# --- Output ---
export_format = "pdf"   # "png" or "pdf"
export_suffix = "_edited"   # added to the filename so you don't overwrite
                             # the original plot from run_example.jl


# =============================================================================
# PART 2: RELOAD AND RE-PLOT (you shouldn't need to change anything below)
# =============================================================================

if !isfile(jld2_path)
    error("Could not find $jld2_path -- run run_example.jl first to create it.")
end

@load jld2_path point_fit boot timevect ydata timevect_all ydata_all forecast_horizon

model_str = GrowthModels.MODEL_NAMES[point_fit.flag]
title_final = isempty(title_str) ? model_str : title_str

fig = PlotGrowthFit.plot_fit_and_forecast(
    boot, timevect, ydata, ydata_all, timevect_all;
    forecast_horizon = forecast_horizon,
    title_str = title_final,
    title_fontsize = title_fontsize,
    hist_title_fontsize = hist_title_fontsize,
    axis_fontsize = axis_fontsize,
    tick_fontsize = tick_fontsize,
    hist_color = hist_color,
    ci_color = ci_color,
    boot_color = boot_color,
    fit_color = fit_color,
    median_color = median_color,
    pi_color = pi_color,
    point_color = point_color,
    point_size = point_size,
    line_width = line_width,
    figsize = figsize,
)

base_name = splitext(basename(jld2_path))[1]
base_name = replace(base_name, "_fit_object" => "")
out_path = joinpath("output", "$(base_name)$(export_suffix).$(export_format)")
Plots.savefig(fig, out_path)

println("Edited plot saved to: $out_path")
