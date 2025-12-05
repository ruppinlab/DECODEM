#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:44:31 2024

@author: dhrubas2
"""

import os, sys
if sys.platform == "darwin":                                                   # mac
    _mpath_ = "/Users/dhrubas2/OneDrive - National Institutes of Health/miscellaneous/py/"
    _wpath_ = "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/analysis_final/"
elif sys.platform == "linux":                                                  # biowulf
    _mpath_ = "/home/dhrubas2/vivid/"
    _wpath_ = "/data/Lab_ruppin/projects/TME_contribution_project/analysis/analysis_final/"

os.chdir(_wpath_)                                                              # current path
if _mpath_ not in sys.path:
    sys.path.append(_mpath_)                                                   # to load miscellaneous

import numpy as np, pandas as pd, pickle, string
import matplotlib.pyplot as plt, seaborn as sns
from math import floor
from itertools import product
from scipy.stats import mannwhitneyu
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn.metrics import RocCurveDisplay
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from warnings import filterwarnings


#%% functions.

## load model predictions from saved pickle.
def read_pkl_data(data_path, mode = "resp"):
    with open(data_path, mode = "rb") as file:
        data_obj  = pickle.load(file)
        
    match mode.lower():
        case "resp":
            y_test    = data_obj["label"]
            y_pred    = data_obj["pred"]
            th_test   = data_obj["th"]
            perf_test = data_obj["perf"]
            outs      = (y_test, y_pred, th_test, perf_test)
        case "surv":
            y_test    = data_obj["label"]
            y_pred    = data_obj["pred"]
            th_test   = data_obj["th"]
            clin_test = data_obj["clin"]
            outs      = (y_test, y_pred, th_test, clin_test)
    del data_obj
    
    return outs


## add p-values to boxplots / violin plots.
def add_stat(ax, stats, data, x, y, yloc = 0.05, lines = True, lw = 2, 
             align = True, fontdict = None):
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    for pxc_, mdl_ in enumerate(data[x].unique()):
        px = pxc_ + np.array([-0.2, 0.2]) / 1.6
        if align:                                                              # align multiple p-values to the same vertical line
            py = data[y].max() + np.array([yloc - 0.025, yloc])
        else:
            py = data.groupby(x).max().loc[mdl_, y] + np.array(
                [yloc - 0.025, yloc])
        
        if lines:                                                              # plot bounding lines
            ax.plot([px[0], px[0], px[1], px[1]], 
                    [py[0], py[1], py[1], py[0]],    
                    linewidth = lw, color = "#000000");
        ax.text(x = pxc_, y = py.max(), s = stats.loc[mdl_, "annot"],          # print p-value
                ha = "center", va = "bottom", color = "#000000", 
                **fontdict["label"]);
    
    ax.set_xlim([-0.5, len(stats) - 0.5]);                                     # restore the original xlims
    
    return ax


## make a group of violin plots.
def make_violinplot(data, x, y, hue, ax, orient = "v", stats = None, 
                    dodge = True, split = False, fill = True, order = None, 
                    hue_order = None, statloc = 0.35, statline = False, 
                    inner = "box", colors = None, xlabel = None, ylabel = None, 
                    title = None, legend_out = True, legend_title = None, 
                    legend_vert = True, fontdict = None):
    
    ## plot parameters.
    if colors is None:
        colors = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                  "#FFC72C", "#708090", "#A9A9A9", "#000000"]
    
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
        
    lineprop  = {"linestyle": "-", "linewidth": 2, "edgecolor": colors[-1]}
    boxprop   = {"box_width": 6, "whis_width": 2, "color": colors[-1]}
    innerprop = boxprop
    if inner.lower() != "box":
        innerprop = {"linestyle": "-", "linewidth": 1.5, "color": colors[-1]}
    
    
    ## main plot.
    sns.violinplot(
        data = data, x = x, y = y, hue = hue, width = 0.8, orient = orient, 
        dodge = dodge, gap = 0.08, order = order, hue_order = hue_order, 
        inner = inner, inner_kws = innerprop, split = split, fill = fill, 
        palette = colors[:data[hue].nunique()], saturation = 0.8, 
        bw_adjust = 0.8, # cut = 1, 
        density_norm = "area", **lineprop, ax = ax)
    
    if stats is not None:
        ax = add_stat(data = data, stats = stats, x = x, y = y, align = True, 
                      lines = statline, yloc = statloc, fontdict = fontdict, 
                      ax = ax)
    
    sns.despine(ax = ax, offset = 0, trim = False);
    
    
    ## format axis ticks & legends.
    ax.set_xlabel(xlabel, **fontdict["label"]);
    ax.set_ylabel(ylabel, **fontdict["label"]);    
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    
    lgndbbox = (1.02 if legend_out else 0.5, 0.3 if legend_out else 0, 
                0.4, 0.4)
    lgnd = ax.legend(loc = "lower left" if legend_out else "best", 
                     frameon = False, bbox_to_anchor = lgndbbox, 
                     ncols = 1 if legend_vert else data[hue].nunique(), 
                     markerscale = 0.9, alignment = "left", 
                     title = legend_title, labelcolor = colors[-1], 
                     prop = fontdict["label"], 
                     title_fontproperties = fontdict["title"])
    if not legend_vert:
        lgnd.set(bbox_to_anchor = (0.45, -0.65, 0.4, 0.4));
    
    if fill:
        [ptch.set(**lineprop) for ptch in lgnd.get_patches()];                 # boundary lines for legend icons
    
    ax.set_title(title, wrap = True, y = 1.02, **fontdict["title"]);

    return ax


## make a ROC curve for a given set of labels & preds.
def make_roc_plot(data, label, pred, group, ax, title = None, fill = False, 
                  alpha = 0.4, colors = None, legend_title = None, 
                  fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    if colors is None:
        colors = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                  "#FFC72C", "#708090", "#A9A9A9", "#000000"]
        colors = colors[3:6] + [colors[-3]]
    
    lineprop = {"linestyle": "-", "linewidth": 2}
    baseprop = {"linestyle": "--", "linewidth": 1.5, "color": colors[-1]}
    mrkrprop = {"marker": "o", "markersize": 6, "markeredgewidth": 1.5}
    lgndttl  = group if (legend_title is None) else legend_title
    
    
    ## main plot.
    for (grp, data_grp), clr in zip(data.groupby(by = group, sort = False), 
                                    colors):
        roc_grp = RocCurveDisplay.from_predictions(
            y_true = data_grp[label], y_pred = data_grp[pred], 
            drop_intermediate = False, pos_label = 1, 
            plot_chance_level = (grp == data[group].iloc[-1]), 
            name = grp, color = clr, **lineprop, **mrkrprop, 
            chance_level_kw = baseprop, ax = ax)
        
        if fill:
            ax.fill_between(x = roc_grp.fpr, y1 = roc_grp.fpr, 
                            y2 = roc_grp.tpr, color = clr, alpha = alpha)
        
        ax.set_aspect("auto")                                                  # stop forcing square-sized plots
    
    
    ## format axis ticks & legends.
    ax.axis([-0.05, 1.05, -0.05, 1.05]);
    ax.set_xticks(np.arange(0, 1.2, 0.2));
    ax.set_yticks(np.arange(0, 1.2, 0.2));
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.set_xlabel("1 $-$ Specificity", labelpad = 8, **fontdict["label"]);
    ax.set_ylabel("Sensitivity", labelpad = 8, **fontdict["label"]);

    lgnd = ax.legend(loc = (1.06, 0.25), title = lgndttl, 
                     prop = fontdict["label"], 
                     title_fontproperties = fontdict["title"])
    for lgndtxt in lgnd.get_texts():
        lgndtxt.set_text( lgndtxt.get_text().replace(") (", ", ") )
        lgndtxt.set_text( lgndtxt.get_text().replace("Chance level", "Random") )
    
    ax.set_title(title, wrap = True, y = 1.02, **fontdict["title"]);
    
    return ax


## make Kaplan-Meier plots for two groups.
def make_km_plot(ax, data1, data2, stat, colors = None, ci_alpha = 0.15, 
                 risk_counts = True, title = None, legend = True, 
                 legend_title = None, fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
        
    if colors is None:
        colors   = ["#E08DAC", "#7595D0"]
    
    lineprop = {"ls": "-", "lw": 2}
    
    lgndttl  = "Risk group" if (legend_title is None) else legend_title
    lbls     = [f"{data1.label} (n = {len(data1.durations)})", 
               f"{data2.label} (n = {len(data2.durations)})"]
    
    
    ## make plots.
    ax = data1.plot(show_censors = True, ci_show = True, color = colors[0], 
                    ci_alpha = ci_alpha, ax = ax, **lineprop)
    ax = data2.plot(show_censors = True, ci_show = True, color = colors[1], 
                    ci_alpha = ci_alpha, ax = ax, **lineprop)
    ax.text(x = 250, y = 0.20, s = f"Log-rank $P$ = {stat.p_value:0.3g}", 
            **fontdict["label"]);
    if risk_counts:                                                            # at-risk counts below the plots
        add_at_risk_counts(data1, data2, labels = lbls, rows_to_show = None, 
                           ax = ax, **fontdict["label"]);
    sns.despine(ax = ax, offset = 0, trim = False);
    
    ## format ticks & labels.
    ax.set_ylim([-0.1, 1.1]);
    ax.set_yticks(np.arange(0, 1.2, 0.2));
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    
    if legend:
        ax.legend(loc = (1.06, 0.25), title = lgndttl, prop = fontdict["label"], 
                  title_fontproperties = fontdict["title"]);
    else:
        ax.legend([ ], [ ]);
    
    ax.set_xlabel("Time in days", y = -0.02, **fontdict["label"]);
    ax.set_ylabel("Survival proabibility", x = 0.01, **fontdict["label"]);
    ax.set_title(title, wrap = True, y = 1.01, **fontdict["title"]);
    
    return ax


## make a radar chart.
## define angles & prepare plot.
def RadarChart(num_vars, frame = "circle"):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {"circle", "polygon"}
        Shape of frame surrounding Axes.

    """
    ## calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint = False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            ## Paths with non-unit interpolation steps correspond to gridlines,
            ## in which case we force interpolation (to defeat PolarTransform's
            ## autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        
        if frame == "polygon":
            PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            ## rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed = True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed = closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            ## FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels, **kwargs)

        def _gen_axes_patch(self):
            ## The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            ## in axes coordinates.
            if frame == "circle":
                return Circle(xy = (0.5, 0.5), radius = 0.5,
                              edgecolor = "#000000")
            elif frame == "polygon":
                return RegularPolygon(xy = (0.5, 0.5), numVertices = num_vars, 
                                      radius = 0.5, edgecolor = "#000000")
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                ## spine_type must be "left"/"right"/"top"/"bottom"/"circle".
                spine = Spine(axes = self, spine_type = "circle", 
                              path = Path.unit_regular_polygon(num_vars))
                ## unit_regular_polygon gives a polygon of radius 1 centered at
                ## (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                ## 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5)
                                    + self.transAxes)
                return {"polar": spine}
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

    register_projection(RadarAxes)
    
    return theta


## make radar lines at specified ticks.
def make_radar_lines(theta, data, ax, rstep = 0.1, labels = None, title = None, 
                     color = "magenta", alpha = 0.25, ls = "-", lw = 1.5, 
                     mrkr = "o", ms = 6, fontdict = None):
    
    rgrids = np.arange(0, 1 + rstep, rstep)
    lprop  = {"linestyle": ls, "linewidth": lw, "color": color}
    mprop  = {"marker": mrkr, "markersize": ms, "markeredgewidth": lw, 
              "markerfacecolor": color}
    fprop  = {"facecolor": color, "alpha": alpha}
    if fontdict is None:
        fontdict  = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}}

    ax.set_rgrids(rgrids)
    ax.plot(theta, data, **lprop, **mprop)
    ax.fill(theta, data, label = "_nolegend_", **fprop, **lprop)
    
    if labels is not None:
        ax.set_varlabels(labels, ma = "center", **fontdict["label"]); 
    
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    if title is not None:
        ax.set_title(f"{title}\n", position = (0.5, 1.2), ha = "center", 
                     va = "center", ma = "center", **fontdict["title"]);
    
    return ax


## adjust tick spacing in radar chart.
def pad_radar_ticks(ticks, pads):
    ## add padding to radar tick labels for better visualization.
    
    n_theta   = len(ticks)
    ticks_pad = [[ ]] * n_theta
    for k, tk in enumerate(ticks):
        if (k == 0) | (k == (n_theta - 1)):
            ticks_pad[k] = tk
        elif k >= 1 and k <= floor(n_theta / 2):
            ticks_pad[k] = "\n".join([
                x + " " * pads[0] for x in tk.split("\n")])
        elif k > floor(n_theta / 2) and k < (n_theta - 1):
            ticks_pad[k] = "\n".join([
                " " * pads[1] + x for x in tk.split("\n")])
    
    return ticks_pad


## make a pair of donut plots - one inside another.
def make_donutplots(data, x, outer, inner, ax, labels = False, title = None, 
                    outer_order = None, inner_order = None, donut_size = 0.35, 
                    colors = None, fontdict = None):
    ## plot parameters.
    if colors is None:
        colors = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                  "#FFC72C", "#A9A9A9", "#000000"]
        colors = [colors[0], colors[1], colors[2], colors[5]]
    
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    wdgprop = {"edgecolor": "#000000", "linestyle": "-", "linewidth": 2, 
               "antialiased": True, "width": donut_size}
    txtprop = {"size": 14, "weight": "demibold", "ha": "center"}
    
    
    ## prepare data.
    outer_data = data.groupby(by = outer).sum(numeric_only = True)    
    inner_data = data.groupby(by = [outer, inner]).sum(numeric_only = True)
    if outer_order is not None:
        outer_data = outer_data.loc[outer_order]
        if inner_order is not None:
            inner_data = inner_data.loc[list(product(outer_order, inner_order))]
        else:
            inner_data = inner_data.loc[outer_order]
    
    outer_labels, inner_labels = None, None
    if isinstance(labels, bool):    labels = [labels] * 2
    if any(labels):
        if labels[0]:    outer_labels = outer_data.index.tolist()
        if labels[1]:    inner_labels = inner_data.index.get_level_values(1)
    
    
    ## make main plot.
    ax.pie(data = outer_data, x = x, radius = 1 + donut_size, 
           labels = outer_labels, labeldistance = 1.1, autopct = "%0.1f%%", 
           pctdistance = 0.80, colors = colors[:2], counterclock = False, 
           shadow = False, wedgeprops = wdgprop, textprops = txtprop)
    ax.pie(data = inner_data, x = x, radius = 0.95, labels = inner_labels, 
           labeldistance = 0.40, autopct = "%0.1f%%", pctdistance = 0.70, 
           colors = colors[2:], counterclock = False, shadow = False, 
           wedgeprops = wdgprop, textprops = txtprop)

    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    
    ax.set_title(title, y = 1.16, wrap = True, **fontdict["title"]);
    
    return ax


## extract p-values with star (*) significance.
def get_pvals_from_score(score_data, resp_col = "Response", 
                         resp_vals = ["R", "NR"], model_col = "model", 
                         score_col = "score"):
    score_stat = score_data.groupby(
        by = model_col, sort = False).apply(
        lambda df: pd.Series(mannwhitneyu(
            df[score_col][df[resp_col].eq(resp_vals[0])], 
            df[score_col][df[resp_col].eq(resp_vals[1])], 
            alternative = "greater", nan_policy = "omit"), 
            index = ["U1", "pval"]), 
        include_groups = False)
    
    score_stat["annot"] = score_stat.pval.apply(
        lambda p: ("***" if (p <= 0.001) else "**" if (p <= 0.01) 
                   else "*" if (p <= 0.05) else "ns"))
    
    return score_stat


#%% read all data for figs 1 & 5.

data_path = "../../data/TransNEO/transneo_analysis/mdl_data/"
data_file = ["zhangTNBC2021_predictions_chemo_th0.99_top3500_ENS2_allfeatures_3foldCVtune_27Sep2024.pkl", 
             "zhangTNBC2021_predictions_chemo_immuno_th0.99_top3500_ENS2_allfeatures_3foldCVtune_27Sep2024.pkl", 
             "bassezBC2021_predictions_chemo_immuno_th0.99_top3500_ENS2_allfeatures_3foldCVtune_29Sep2024.pkl", 
             "tcga_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_07Dec2024.pkl"]


## get Zhang et al. SC prediction data.
y_test_cm1, y_pred_cm1, th_test_cm1, perf_test_cm1 = read_pkl_data(
    data_path + data_file[0], mode = "resp")

y_test_im1, y_pred_im1, th_test_im1, perf_test_im1 = read_pkl_data(
    data_path + data_file[1], mode = "resp")


## get Bassez et al. SC prediction data.
y_test_im2, y_pred_im2, th_test_im2, perf_test_im2 = read_pkl_data(
    data_path + data_file[2], mode = "resp")


## get TCGA-BRCA survival data.
subtypes_all = ["ER+/HER2-", "TNBC"]                                           # define BC subtype labels

y_test_surv, y_pred_surv, th_test_surv, clin_test_surv = read_pkl_data(
    data_path + data_file[3], mode = "surv")
  
clin_test_surv["Clinical_subtype"] = clin_test_surv.ER_status.map(
    lambda x: subtypes_all[0] if (x == "Positive") else subtypes_all[1])


#%% prepare data for fig 1-II.

response_all   = {0: ["R", "NR"], 1: ["E", "NE"]}                              # define patient response labels (0 = RECIST, 1 = clonotype expansion)
treatments_all = ["Chemotherapy", "Chemotherapy + ICB"]                        # define treatment labels

ds_info1       = pd.DataFrame(
    [[trt, len(y), sum(y == 1), sum(y == 0), f"{trt} (n = {len(y)})"] 
     for trt, y in zip(treatments_all, [y_test_cm1, y_test_im1])], 
    columns = ["Treatment", "n"] + response_all[0] + ["label"])

ds_info2      = pd.DataFrame(
    [[treatments_all[1], len(y_test_im2), sum(y_test_im2 == 1), 
     sum(y_test_im2 == 0), f"{treatments_all[1]} (n = {len(y_test_im2)})"]], 
    columns = ["Treatment", "n"] + response_all[1] + ["label"], index = [0])


## prepare data for fig 1C.
fig_data1_II = [
    pd.DataFrame({
        "Subtype"  : subtypes_all[1:] * 4, "Response": response_all[0] * 2, 
        "Treatment": np.repeat(treatments_all, repeats = 2), 
        "Count"    : [y_test_cm1.eq(1).sum(), y_test_cm1.eq(0).sum(), 
                      y_test_im1.eq(1).sum(), y_test_im1.eq(0).sum()] }), 
    pd.DataFrame({
        "Subtype"  : subtypes_all[1:] * 4, "Response": response_all[1] * 2, 
        "Treatment": np.repeat(treatments_all, repeats = 2), 
        "Count"    : [0, 0, y_test_im2.eq(1).sum(), y_test_im2.eq(0).sum()] })]


#%% make fig 1-II.

svdat = False                                                                  # set as True to save data

## plot parameters.
sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 8, "xtick.major.width": 2, 
    "ytick.major.size": 8, "ytick.major.width": 2, 
    "xtick.bottom": True, "ytick.left": True, 
    "axes.spines.top": False, "axes.spines.right": False, 
    "axes.linewidth": 2, "axes.edgecolor": "#000000", 
    "grid.linewidth": 1, "grid.color": "#000000", "grid.alpha": 0.8, 
    "legend.frameon": False, "legend.edgecolor": "#000000", 
    "legend.framealpha": 0.9, "legend.markerscale": 1.2, 
    "font.family": "sans"})

colors       = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                "#FFC72C", "#708090", "#A9A9A9", "#000000"]

panel_fonts  = {"weight": "bold", "size": 32, "color": "#000000"}
label_fonts  = {"weight": "regular", "size": 14, "color": "#000000"}
legend_fonts = {"item" : {"size": 12, "weight": "regular"}, 
                "title": {"size": 16, "weight": "bold"}}
plot_fonts   = {
    "label": {"family": "sans", "size": 14, "weight": "regular"}, 
    "title": {"family": "sans", "size": 16, "weight": "bold"}, 
    "super": {"family": "sans", "size": 20, "weight": "bold"}}

## nested donut plots.
fig_dntsize1 = 0.60
fig_colors1  = [colors[0], colors[1], colors[4], colors[3]]
fig_ttls1    = [f"{dn} (n = {ds.n.sum()})" for dn, ds in zip(
    ["Zhang et al.", "Bassez et al."], [ds_info1, ds_info2])]
fig_lgnd1    = [" / ".join(x) for x in zip(*response_all.values())] + \
    treatments_all
fig_ftnt1    = "\n".join(["*Chemotherapy = Taxane / Taxane-Anthracycline", 
                          " ICB = Atezolizumab / Pembrolizumab"])

fig1_II, ax1_II = plt.subplots(figsize = (14, 7), nrows = 1, ncols = 2)
ax1_II = dict(zip(["C1", "C2"], ax1_II))

for k, ax in enumerate(ax1_II.values()):
    ax = make_donutplots(data = fig_data1_II[k], x = "Count", 
                         outer = "Response", inner = "Treatment", 
                         outer_order = response_all[k], 
                         inner_order = treatments_all, colors = fig_colors1, 
                         donut_size = fig_dntsize1, title = fig_ttls1[k], 
                         ax = ax)
    
    if k == 1:                                                                 # add common legend
        ax.legend(labels = fig_lgnd1, loc = (1.20, 0.45), ncols = 2, 
                  title = f"Response{' ' * 5}Treatment", alignment = "left", 
                  prop = legend_fonts["item"], 
                  title_fontproperties = legend_fonts["title"]);
        fig1_II.text(x = 0.71, y = 0.38, s = fig_ftnt1, 
                     **legend_fonts["item"]);                                  # add drug info

fig1_II.text(x = 0.0, y = 0.80, s = "C", **panel_fonts);                       # add panel label

fig1_II.tight_layout(h_pad = 0, w_pad = 4)
plt.show()


## save figures.
## default DPI is 100. need to update to 600.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file1_II = "dataset_summary_sc.pdf"
    fig1_II.savefig(fig_path + fig_file1_II, dpi = 600)
    print(fig_file1_II)


#%% prepare data for fig 5. 

filterwarnings(action = "ignore")                                              # suppress future-warnings

def get_pred_data(y_true, y_pred, lbl = None, lblname = "label", 
                  resp = ["R", "NR"]):
    ## build score matrix.
    score_data = pd.DataFrame({
        "Response": y_true.replace(to_replace = dict(zip([1, 0], resp))), 
        "score"   : y_pred.astype(float)}).dropna(
        how = "any", axis = 0)
    
    if lbl is not None:
        score_data.insert(loc = 1, column = lblname, value = lbl)
    
    ## perform R vs. NR wilcoxon test.
    score_stat = score_data.pipe(
        lambda df: pd.Series(mannwhitneyu(
            df.score[df.Response.eq(resp[0])], df.score[df.Response.eq(resp[1])], 
            alternative = "greater", nan_policy = "omit"), 
            index = ["U1", "pval"]))

    score_stat["annot"] = ("***" if (score_stat.pval <= 0.001) else 
                           "**" if (score_stat.pval <= 0.01) else 
                           "*" if (score_stat.pval <= 0.05) else 
                           "ns")
    
    return score_data, score_stat


## init data list for fig 5: panels A-D.
fig_data5, fig_stat5 = [[ ] for _ in range(4)], [[ ] for _ in range(2)]

## Zhang et al.
## get data for fig 5A. 
model1       = "B-cells"
fig_data5[0] = [[ ] for _ in range(2)]                                         # R vs. NR scores
fig_stat5[0] = fig_data5[0].copy()                                             # R vs. NR p-values

fig_data5[0][0], fig_stat5[0][0] = get_pred_data(y_true  = y_test_cm1, 
                                                 y_pred  = y_pred_cm1[model1], 
                                                 resp    = response_all[0], 
                                                 lbl     = ds_info1.label[0], 
                                                 lblname = "Treatment")

fig_data5[0][1], fig_stat5[0][1] = get_pred_data(y_true  = y_test_im1, 
                                                 y_pred  = y_pred_im1[model1], 
                                                 resp    = response_all[0], 
                                                 lbl     = ds_info1.label[1], 
                                                 lblname = "Treatment")

fig_data5[0] = pd.concat(fig_data5[0], axis = 0)
fig_data5[0].Treatment = fig_data5[0].Treatment.map(
    lambda x: x.replace(" (", "\n("))
fig_data5[0].insert(loc = 2, column = "model", value = model1)
fig_stat5[0] = pd.concat(fig_stat5[0], axis = 1, 
                         keys = fig_data5[0].Treatment.unique()).T


## get data for fig 5B.
fig_data5[1] = fig_data5[0].replace(
    to_replace = {"Response" : dict(zip(response_all[0], [1, 0])), 
                  "Treatment": {ds.replace(" (", "\n("): ds 
                                for ds in ds_info1.label}}).infer_objects(
    copy = False)


## Bassez et al.
## get data for fig 5C.
model2       = ["B-cells", "Myeloid", "Endothelial"]
fig_data5[2] = [[ ] for _ in range(len(model2))]                               # R vs. NR scores
fig_stat5[1] = fig_data5[2].copy()                                             # R vs. NR p-values

for k, mdl in enumerate(model2):
    lbl = f"{mdl}\n(n = {y_pred_im2[mdl].dropna().shape[0]})"
    fig_data5[2][k], fig_stat5[1][k] = get_pred_data(y_true  = y_test_im2, 
                                                     y_pred  = y_pred_im2[mdl], 
                                                     resp    = response_all[1], 
                                                     lbl     = lbl, 
                                                     lblname = "model")
del k, mdl, lbl                                                                # reduce clutter

fig_data5[2] = pd.concat(fig_data5[2], axis = 0)
fig_data5[2].insert(loc = 1, column = "Treatment", value = treatments_all[1])
fig_stat5[1] = pd.concat(fig_stat5[1], axis = 1, 
                         keys = fig_data5[2].model.unique()).T


## get data for fig 5D.
fig_data5[3] = fig_data5[2].replace(
    to_replace = {"Response" : dict(zip(response_all[1], [1, 0])), 
                  "model"    : {mdl: mdl.replace("\n(", " (") 
                                for mdl in fig_stat5[1].index}}).infer_objects(
                                        copy = False)


#%% make fig 5.

svdat = False                                                                  # set as True to save data

## plot parameters.
sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 8, "xtick.major.width": 2, 
    "ytick.major.size": 8, "ytick.major.width": 2, 
    "xtick.bottom": True, "ytick.left": True, 
    "axes.spines.top": False, "axes.spines.right": False, 
    "axes.linewidth": 2, "axes.edgecolor": "#000000", 
    "grid.linewidth": 1, "grid.color": "#000000", "grid.alpha": 0.8, 
    "legend.frameon": False, "legend.edgecolor": "#000000", 
    "legend.framealpha": 0.9, "legend.markerscale": 1.2, 
    "font.family": "sans"})

colors       = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                "#FFC72C", "#708090", "#A9A9A9", "#000000"]

panel_fonts  = {"weight": "bold", "size": 32, "color": "#000000"}
label_fonts  = {"weight": "regular", "size": 14, "color": "#000000"}
legend_fonts = {"item" : {"size": 12, "weight": "regular"}, 
                "title": {"size": 16, "weight": "bold"}}

## violin plots + ROC plots.
fig_llocs5  = [[0.08, 0.43], [0.96, 0.52]]
fig_ploc5   = [0.4, 0.7]
fig_ylim5   = [[0.2, 0.4], [0.8, 0.7]]
fig_ttls5   = [f"Zhang et al. (n = {ds_info1.n.sum()})\n", 
               f"Bassez et al. (n = {ds_info2.n.sum()})\n"]
fig_colors5 = [colors[k] for k in [3, 4, 5, -1]]


fig5, ax5 = plt.subplots(figsize = (16, 7), nrows = 2, ncols = 2)
ax5 = dict(zip(list("ABCD"), ax5.ravel()))


## make violins.
for k, lbl in enumerate(list("AC")):
    match lbl: 
        case "A":
            xcol, lgndttl = "Treatment", "Response"
        case "C":
            xcol, lgndttl = "model", "Clonotype\nExpansion"
    
    ax = ax5[lbl]
    ax = make_violinplot(data = fig_data5[2 * k], x = xcol, y = "score", 
                         hue = "Response", stats = fig_stat5[k], 
                         hue_order = response_all[k], inner = "quart", 
                         split = True, dodge = True, statloc = fig_ploc5[k], 
                         title = fig_ttls5[k], ylabel = "Prediction score", 
                         legend_vert = True, legend_out = False, 
                         legend_title = lgndttl, ax = ax)
    ax.set_ylim([0 - fig_ylim5[k][0], 1 + fig_ylim5[k][1]]);
    ax.set_yticks(np.arange(0, 1.2, 0.2).round(1));
    ax.get_legend().set(bbox_to_anchor = (-0.20, 0.70));
    fig5.text(x = fig_llocs5[0][0], y = fig_llocs5[1][k], s = lbl, 
              **panel_fonts);                                                  # add panel labels


## make ROC curves.
for k, lbl in enumerate(list("BD")):
    match lbl:
        case "B":
            grp, lgndttl = "Treatment", "Treatment"
        case "D":
            grp, lgndttl = "model", "Cell type"
    
    ax = ax5[lbl]
    ax = make_roc_plot(data = fig_data5[2 * k + 1], label = "Response", 
                       pred = "score", group = grp, colors = fig_colors5, 
                       fill = True, alpha = 0.15, title = fig_ttls5[k], 
                       legend_title = lgndttl, ax = ax)
    fig5.text(x = fig_llocs5[0][1], y = fig_llocs5[1][k], s = lbl, 
              **panel_fonts);                                                  # add panel labels

fig5.tight_layout(h_pad = 2, w_pad = 4)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file5 = "all_predictions_sc_survival_chemo_immuno_th0.99_ENS2_5foldCV_v2.pdf"
    fig5.savefig(fig_path + fig_file5, dpi = 600)
    print(fig_file5)


#%% make supplementary figs.
#%% prepare data for supp fig 9.

## prepare data for supp fig 9A-C.
fig_dataS9, fig_statS9 = [[ ] for _ in range(3)], [[ ] for _ in range(3)]      # R vs. NR scores & p-values

for k, (y_t, y_p) in enumerate(zip([y_test_cm1, y_test_im1, y_test_im2], 
                                   [y_pred_cm1, y_pred_im1, y_pred_im2])):
    y_p = y_p.copy().rename(
        columns = lambda x: f"{'Pseudobulk' if x == 'bulk' else x}\n(n = {y_p[x].notna().sum()})")
    
    for mdl in y_p.columns:        
        dat, stat = get_pred_data(y_true  = y_t, 
                                  y_pred  = y_p[mdl], 
                                  resp    = response_all[int(k == 2)],         # k < 2: Zhang et al., k == 2: Bassez et al.
                                  lbl     = mdl, 
                                  lblname = "model")
        dat  = dat.replace(
            regex = {"model": {"_": "\n", "Bulk": "Pseudobulk"}})
        
        stat = stat.rename(index = mdl.replace(
            "_", "\n").replace(
            "Bulk", "Pseudobulk"))
        
        fig_dataS9[k].append( dat );    fig_statS9[k].append( stat )
    
    fig_dataS9[k] = pd.concat(fig_dataS9[k], axis = 0)
    fig_statS9[k] = pd.concat(fig_statS9[k], axis = 1).T
    
del mdl, k, y_t, y_p, dat, stat                                                # reduce clutter


## prepare data for supp fig 9D-F.
[fig_dataS9.append(
    perf[["AUC"]].reset_index(
        names = "model").replace(
        regex = {"_": "\n", "Bulk": "Pseudobulk"}))
    for perf in [perf_test_cm1, perf_test_im1, perf_test_im2]];


#%% make supp fig 9-I.

svdat = False                                                                  # set as True to save data

## plot parameters.
sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 8, "xtick.major.width": 2, 
    "ytick.major.size": 8, "ytick.major.width": 2, 
    "xtick.bottom": True, "ytick.left": True, 
    "axes.spines.top": False, "axes.spines.right": False, 
    "axes.linewidth": 2, "axes.edgecolor": "#000000", 
    "grid.linewidth": 1, "grid.color": "#000000", "grid.alpha": 0.8, 
    "legend.frameon": False, "legend.edgecolor": "#000000", 
    "legend.framealpha": 0.9, "legend.markerscale": 1.2, 
    "font.family": "sans"})

colors      = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
               "#FFC72C", "#708090", "#A9A9A9", "#000000"]

panel_fonts = {"weight": "bold", "size": 36, "color": "#000000"}
label_fonts = {"weight": "regular", "size": 14, "color": "#000000"}

## violin plots.
fig_llocsS9 = [[-0.01, 0.44], [0.96, 0.46]]
fig_plocS9  = [0.4, 0.75]
fig_ylimS9  = [0.5, 0.75]

fig_ttlsS9  = ["Zhang et al.: Chemotherapy\n", 
               "Zhang et al.: Chemotherapy + ICB\n", 
               "Bassez et al.: Chemotherapy + ICB\n"]


figS9_I, axS9_I = plt.subplot_mosaic(
    mosaic = [["A", "B"], ["C", "C"]], figsize = (18, 8), 
    height_ratios = [1, 1], width_ratios = [1, 1], sharey = False)

## make violins.
for k, (lbl, ax) in enumerate(axS9_I.items()):
    ax = make_violinplot(data = fig_dataS9[k], x = "model", y = "score", 
                         hue = "Response", stats = fig_statS9[k], 
                         hue_order = response_all[int(k == 2)], inner = "quart", 
                         split = True, dodge = True, 
                         statloc = fig_plocS9[k // 2], statline = False, 
                         title = fig_ttlsS9[k], legend_vert = True, 
                         legend_out = True, 
                         legend_title = "Response" if k < 2 else "Clonotype\nExpansion", 
                         ax = ax)
    ax.set_ylim([0 - fig_ylimS9[k // 2] - 0.1, 1 + fig_ylimS9[k // 2]]);
    ax.set_yticks(np.arange(0, 1.2, 0.2).round(1));
    if k == 0:    ax.legend([ ], [ ]);
    if not k % 2:
        ax.set_ylabel("Prediction score", y = 0.52, **legend_fonts["item"]);
    
    figS9_I.text(x = fig_llocsS9[0][k % 2], y = fig_llocsS9[1][k // 2], 
                 s = lbl, **panel_fonts);                                      # add panel labels


figS9_I.tight_layout(h_pad = 4, w_pad = 2)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS9_I = "all_predictions_sc_chemo_th0.99_ENS2_allfeatures_5foldCV.pdf"
    figS9_I.savefig(fig_path + fig_fileS9_I, dpi = 600)
    print(fig_fileS9_I)


#%% make supp fig 9-II.

svdat = False                                                                  # set as True to save data

## plot parameters.
sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 8, "xtick.major.width": 2, 
    "ytick.major.size": 8, "ytick.major.width": 2, 
    "xtick.bottom": True, "ytick.left": True, 
    "axes.spines.top": False, "axes.spines.right": False, 
    "axes.linewidth": 2, "axes.edgecolor": "#000000", 
    "grid.linewidth": 1, "grid.color": "#000000", "grid.alpha": 0.8, 
    "legend.frameon": False, "legend.edgecolor": "#000000", 
    "legend.framealpha": 0.9, "legend.markerscale": 1.2, 
    "font.family": "sans"})

colors       = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                "#FFC72C", "#708090", "#A9A9A9", "#000000"]

panel_fonts  = {"weight": "bold", "size": 36, "color": "#000000"}
label_fonts  = {"weight": "regular", "size": 14, "color": "#000000"}
legend_fonts = {"item" : {"size": 12, "weight": "regular"}, 
                "title": {"size": 16, "weight": "bold"}}

## radar charts.
fig_llocsS9 = [0.02, 0.52]
fig_thetaS9 = RadarChart(num_vars = len(fig_dataS9[3]), frame = "circle")
fig_baseS9  = [0.5] * len(fig_thetaS9)
fig_ticksS9 = ["B-cells", "Myeloid" + " " * 4, "T-cells",                      # only 4 ticks- so fix manually
               " " * 10 + "Pseudobulk"]
fig_ttlsS9  = ["Zhang et al.: Chemotherapy", 
               "Zhang et al.: Chemotherapy + ICB"]

figS9_II, axS9_II = plt.subplots(figsize = (12, 5), nrows = 1, ncols = 2, 
                                 subplot_kw = {"projection": "radar"})
axS9_II = dict(zip(list("DE"), axS9_II))

## make radars.
for k, (lbl, ax) in enumerate(axS9_II.items(), start = 3):
    ax = make_radar_lines(theta = fig_thetaS9, data = fig_dataS9[k]["AUC"], 
                          labels = fig_ticksS9, color = colors[3], alpha = 0.4, 
                          ls = "-", lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_thetaS9, data = fig_baseS9, 
                          title = fig_ttlsS9[k - 3], color = colors[-3], 
                          alpha = 0.15, ls = ":", ms = 8, ax = ax)
    ax.set_rlim([0.25, 1.05])
    figS9_II.text(x = fig_llocsS9[k - 3], y = 0.96, s = lbl, **panel_fonts);   # add panel labels

figS9_II.tight_layout(h_pad = 0, w_pad = 0)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS9_II = "all_aucs_sc_zhang_chemo_th0.99_ENS2_allfeatures_5foldCV.pdf"
    figS9_II.savefig(fig_path + fig_fileS9_II, dpi = 600)
    print(fig_fileS9_II)


#%% make supp fig 7-III.

svdat = False                                                                  # set as True to save data

## plot parameters.
sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 8, "xtick.major.width": 2, 
    "ytick.major.size": 8, "ytick.major.width": 2, 
    "xtick.bottom": True, "ytick.left": True, 
    "axes.spines.top": False, "axes.spines.right": False, 
    "axes.linewidth": 2, "axes.edgecolor": "#000000", 
    "grid.linewidth": 1, "grid.color": "#000000", "grid.alpha": 0.8, 
    "legend.frameon": False, "legend.edgecolor": "#000000", 
    "legend.framealpha": 0.9, "legend.markerscale": 1.2, 
    "font.family": "sans"})

colors       = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                "#FFC72C", "#708090", "#A9A9A9", "#000000"]

panel_fonts  = {"weight": "bold", "size": 36, "color": "#000000"}
label_fonts  = {"weight": "regular", "size": 14, "color": "#000000"}
legend_fonts = {"item" : {"size": 12, "weight": "regular"}, 
                "title": {"size": 16, "weight": "bold"}}

## radar charts.
fig_thetaS9 = RadarChart(num_vars = len(fig_dataS9[-1]), frame = "circle")
fig_baseS9  = [0.5] * len(fig_thetaS9)
fig_ticksS9 = pad_radar_ticks(ticks = fig_dataS9[-1].model, pads = [12, 8])
fig_ticksS9[-1] = " " * 6 + fig_ticksS9[-1]                                    # manually fix pseudobulk label

figS9_III, axS9_III = plt.subplots(figsize = (6, 5), nrows = 1, ncols = 1, 
                                   subplot_kw = {"projection": "radar"})

## make radars.
axS9_III = make_radar_lines(theta = fig_thetaS9, data = fig_dataS9[-1]["AUC"], 
                            labels = fig_ticksS9, color = colors[3], 
                            alpha = 0.4, ls = "-", lw = 2, ms = 8, 
                            ax = axS9_III)
axS9_III = make_radar_lines(theta = fig_thetaS9, data = fig_baseS9, 
                            title = "Bassez et al.: Chemotherapy + ICB", 
                            color = colors[-3], alpha = 0.15, ls = ":", ms = 8, 
                            ax = axS9_III)
axS9_III.set_rlim([0.25, 0.85])
axS9_III.legend(labels = ["Cell type", "Random"], loc = (1.16, 0.45), 
                title = "AUC", prop = legend_fonts["item"], 
                title_fontproperties = legend_fonts["title"])                  # format legends
figS9_III.text(x = -0.02, y = 0.94, s = "F", **panel_fonts);                   # add panel label

figS9_III.tight_layout(h_pad = 0, w_pad = 0)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS9_III = "all_aucs_sc_bassez_chemo_th0.99_ENS2_allfeatures_5foldCV.pdf"
    figS9_III.savefig(fig_path + fig_fileS9_III, dpi = 600)
    print(fig_fileS9_III)


#%% prepare data for supp figs 10-11.
## CE, ENDO, PB, NE, MYL, B, CAF

def get_surv_data(y_test, y_pred, clin, group, models, endpoint):
    surv_data = { }
    for mdl in models:
        ## get data for K-M plot.
        km_dat = pd.concat([y_pred[mdl][group].rename(index = "Group"), 
                            y_test[[endpoint, f"{endpoint}_time"]], 
                            clin["Clinical_subtype"]], axis = 1)
        
        km_mdl, km_stat = { }, { }
        for sb, dat in km_dat.groupby(by = "Clinical_subtype", sort = True):
            ## fit K-M models per subtype.
            km_mdl[sb] = { 
                grp: KaplanMeierFitter(
                    alpha = 0.05, label = grp).fit(
                    event_observed = df[endpoint], 
                    durations      = df[f"{endpoint}_time"], 
                    label          = ("High" if grp else "Low") + "-score")
                for grp, df in dat.groupby(by = "Group", sort = True)}
            
            
            ## do log-rank test per subtype.
            km_stat[sb] = logrank_test(
                event_observed_A = dat[dat.Group.eq(1)][endpoint], 
                durations_A      = dat[dat.Group.eq(1)][f"{endpoint}_time"], 
                event_observed_B = dat[dat.Group.eq(0)][endpoint], 
                durations_B      = dat[dat.Group.eq(0)][f"{endpoint}_time"])
        
        ## save data for plotting.
        surv_data[mdl.replace("_", " ")] = {
            sb: {"data1": km_mdl[sb][1], "data2": km_mdl[sb][0], 
                 "stat": km_stat[sb]} for sb in subtypes_all}
        
    return surv_data


## survival info.
surv_info = pd.concat([y_test_surv, clin_test_surv], axis = 1).groupby(
    by = "Clinical_subtype", sort = True).apply(
    lambda dat: pd.Series({
        "n"           : len(dat), 
        "OS_events"   : dat.OS.sum(), 
        "OS_time_med" : int(dat.OS_time.median()), 
        "PFI_events"  : dat.PFI.sum(), 
        "PFI_time_med": int(dat.PFI_time.median())}), 
    include_groups = False).reset_index()
surv_info["label"] = surv_info.apply(
    lambda x: f"TCGA-BRCA: {x.Clinical_subtype} (n = {x.n})", axis = 1)


## cell types to consider.
models = ["Cancer_Epithelial", "Endothelial", "Plasmablasts", 
          "Normal_Epithelial", "Myeloid", "B-cells", "CAFs"]


## prepare data for supp fig 10-11: OS, PFI.
fig_dataS10 = get_surv_data(y_test   = y_test_surv, 
                            y_pred   = y_pred_surv, 
                            clin     = clin_test_surv, 
                            group    = "groups_05", 
                            models   = models, 
                            endpoint = "OS")

fig_dataS11 = get_surv_data(y_test   = y_test_surv, 
                            y_pred   = y_pred_surv, 
                            clin     = clin_test_surv, 
                            group    = "groups_05", 
                            models   = models, 
                            endpoint = "PFI")


#%% make supp fig 10.

svdat = False                                                                  # set as True to save data

## plot parameters.
sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 8, "xtick.major.width": 2, 
    "ytick.major.size": 8, "ytick.major.width": 2, 
    "xtick.bottom": True, "ytick.left": True, 
    "axes.spines.top": False, "axes.spines.right": False, 
    "axes.linewidth": 2, "axes.edgecolor": "#000000", 
    "grid.linewidth": 1, "grid.color": "#000000", "grid.alpha": 0.8, 
    "legend.frameon": False, "legend.edgecolor": "#000000", 
    "legend.framealpha": 0.9, "legend.markerscale": 1.2, 
    "font.family": "sans"})

colors       = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                "#FFC72C", "#708090", "#A9A9A9", "#000000"]

panel_fonts  = {"weight": "bold", "size": 32, "color": "#000000"}
label_fonts  = {"weight": "regular", "size": 14, "color": "#000000"}
legend_fonts = {"item" : {"size": 12, "weight": "regular"}, 
                "title": {"size": 16, "weight": "bold"}}

## K-M plots.
fig_llocsS10 = [[0.00, 0.45], (np.arange(7, 0, -1) / 7 - 0.02).tolist()]
fig_lblsS10  = np.reshape(list(string.ascii_uppercase[:14]), [7, 2])

figS10, axS10 = plt.subplots(figsize = (16, 18), nrows = 7, ncols = 2)
axS10 = dict(zip(fig_lblsS10.ravel(), axS10.ravel()))

## make plots.
for i, (lbls, mdl) in enumerate(zip(fig_lblsS10, fig_dataS10.keys())):
    for j, (lbl, sb) in enumerate(zip(lbls, subtypes_all)):
        ax  = axS10[lbl]
        ttl = surv_info.label[j].replace(":", " OS:") + "\n" if i == 0 else ""
        ttl += f"Cell type = {mdl}"
        ax  = make_km_plot(**fig_dataS10[mdl][sb], title = ttl, 
                           risk_counts = False, legend = ((i, j) == (3, 1)), 
                           ax = ax)
        ax.set_xlabel(None);    ax.set_ylabel(None)
        figS10.text(x = fig_llocsS10[0][j], y = fig_llocsS10[1][i], s = lbl, 
                    **panel_fonts);                                            # add panel labels
        
## add common labels.
figS10.supxlabel("Time in days", y = 0.00, x = 0.47, **label_fonts);
figS10.supylabel("Survival probability", x = -0.02, y = 0.50, **label_fonts);

figS10.tight_layout(h_pad = 4, w_pad = 6)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS10 = "all_predictions_survival_os_tcga_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS10.savefig(fig_path + fig_fileS10, dpi = 600)
    print(fig_fileS10)


#%% make supp fig 11.

svdat = False                                                                  # set as True to save data

## plot parameters.
sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 8, "xtick.major.width": 2, 
    "ytick.major.size": 8, "ytick.major.width": 2, 
    "xtick.bottom": True, "ytick.left": True, 
    "axes.spines.top": False, "axes.spines.right": False, 
    "axes.linewidth": 2, "axes.edgecolor": "#000000", 
    "grid.linewidth": 1, "grid.color": "#000000", "grid.alpha": 0.8, 
    "legend.frameon": False, "legend.edgecolor": "#000000", 
    "legend.framealpha": 0.9, "legend.markerscale": 1.2, 
    "font.family": "sans"})

colors       = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                "#FFC72C", "#708090", "#A9A9A9", "#000000"]

panel_fonts  = {"weight": "bold", "size": 32, "color": "#000000"}
label_fonts  = {"weight": "regular", "size": 14, "color": "#000000"}
legend_fonts = {"item" : {"size": 12, "weight": "regular"}, 
                "title": {"size": 16, "weight": "bold"}}

## K-M plots.
fig_llocsS11 = [[0.00, 0.45], (np.arange(7, 0, -1) / 7 - 0.02).tolist()]
fig_lblsS11  = np.reshape(list(string.ascii_uppercase[:14]), [7, 2])

figS11, axS11 = plt.subplots(figsize = (16, 18), nrows = 7, ncols = 2)
axS11 = dict(zip(fig_lblsS11.ravel(), axS11.ravel()))

## make plots.
for i, (lbls, mdl) in enumerate(zip(fig_lblsS11, fig_dataS11.keys())):
    for j, (lbl, sb) in enumerate(zip(lbls, subtypes_all)):
        ax  = axS11[lbl]
        ttl = surv_info.label[j].replace(":", " PFI:") + "\n" if i == 0 else ""
        ttl += f"Cell type = {mdl}"
        ax  = make_km_plot(**fig_dataS11[mdl][sb], title = ttl, 
                           risk_counts = False, legend = ((i, j) == (3, 1)), 
                           ax = ax)
        ax.set_xlabel(None);    ax.set_ylabel(None)
        figS11.text(x = fig_llocsS11[0][j], y = fig_llocsS11[1][i], s = lbl, 
                    **panel_fonts);                                            # add panel labels
        
## add common labels.
figS11.supxlabel("Time in days", y = 0.00, x = 0.47, **label_fonts);
figS11.supylabel("Survival probability", x = -0.02, y = 0.50, **label_fonts);

figS11.tight_layout(h_pad = 4, w_pad = 6)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS11 = "all_predictions_survival_pfi_tcga_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS11.savefig(fig_path + fig_fileS11, dpi = 600)

