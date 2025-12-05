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
from math import nan, ceil, floor
from itertools import product
from functools import reduce
from operator import add
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


#%% functions.

def read_pkl_data(data_path):
    with open(data_path, mode = "rb") as file:
        data_obj  = pickle.load(file)
        y_test    = data_obj["label"]
        y_pred    = data_obj["pred"]
        th_test   = data_obj["th"]
        perf_test = data_obj["perf"]
        del data_obj
        
    return y_test, y_pred, th_test, perf_test


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


def make_km_plot(ax, data_grp1, data_grp2, stat, colors = None, 
                 ci_alpha = 0.15, risk_counts = True, title = None, 
                 legend = True, legend_title = None, fontdict = None):
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
    lbls     = [f"{data_grp1.label} (n = {len(data_grp1.durations)})", 
               f"{data_grp2.label} (n = {len(data_grp2.durations)})"]
    
    
    ## make plots.
    ax = data_grp1.plot(show_censors = True, ci_show = True, color = colors[0], 
                        ci_alpha = ci_alpha, ax = ax, **lineprop)
    ax = data_grp2.plot(show_censors = True, ci_show = True, color = colors[1], 
                        ci_alpha = ci_alpha, ax = ax, **lineprop)
    ax.text(x = 250, y = 0.20, s = f"Log-rank $P$ = {stat.p_value:0.3g}", 
            **fontdict["label"]);
    if risk_counts:                                                            # at-risk counts below the plots
        add_at_risk_counts(data_grp1, data_grp2, labels = lbls, 
                           rows_to_show = None, ax = ax, **fontdict["label"]);
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
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint = False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        
        if frame == "polygon":
            PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
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
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels, **kwargs)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
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
                # spine_type must be "left"/"right"/"top"/"bottom"/"circle".
                spine = Spine(axes = self, spine_type = "circle", 
                              path = Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5)
                                    + self.transAxes)
                return {"polar": spine}
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

    register_projection(RadarAxes)
    
    return theta


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


#%% read all data for fig. 5.

data_path = "../../data/TransNEO/transneo_analysis/mdl_data/"
data_file = ["zhangTNBC2021_predictions_chemo_th0.99_top3500_ENS2_allfeatures_3foldCVtune_27Sep2024.pkl", 
             "zhangTNBC2021_predictions_chemo_immuno_th0.99_top3500_ENS2_allfeatures_3foldCVtune_27Sep2024.pkl", 
             "bassezBC2021_predictions_chemo_immuno_th0.99_top3500_ENS2_allfeatures_3foldCVtune_29Sep2024.pkl", 
             "tcga_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_07Dec2024.pkl"]


## get zhang et al. sc prediction data.
with open(data_path + data_file[0], "rb") as file:
    data_obj      = pickle.load(file)
    y_test_cm1    = data_obj["label"]
    y_pred_cm1    = data_obj["pred"]
    th_test_cm1   = data_obj["th"]
    perf_test_cm1 = data_obj["perf"]
    del data_obj

with open(data_path + data_file[1], "rb") as file:
    data_obj      = pickle.load(file)
    y_test_im1    = data_obj["label"]
    y_pred_im1    = data_obj["pred"]
    th_test_im1   = data_obj["th"]
    perf_test_im1 = data_obj["perf"]
    del data_obj


## get bassez et al. sc prediction data.
with open(data_path + data_file[2], "rb") as file:
    data_obj      = pickle.load(file)
    y_test_im2    = data_obj["label"]
    y_pred_im2    = data_obj["pred"]
    th_test_im2   = data_obj["th"]
    perf_test_im2 = data_obj["perf"]
    del data_obj


## get tcga-brca survival data.
with open(data_path + data_file[3], "rb") as file:
    data_obj       = pickle.load(file)
    y_test_surv    = data_obj["label"]
    y_pred_surv    = data_obj["pred"]
    th_test_surv   = data_obj["th"]
    clin_test_surv = data_obj["clin"]
    del data_obj
    
clin_test_surv["Clinical_subtype"] = clin_test_surv.ER_status.map(
    lambda x: "ER+,HER2-" if (x == "Positive") else "TNBC")


#%% prepare data for fig. 1-II.

fig_data1C = [
    pd.DataFrame({
        "Subtype"  : ["TNBC"] * 4, "Response": ["R", "NR"] * 2, 
        "Treatment": ["Chemotherapy"] * 2 + ["Chemotherapy + ICB"] * 2, 
        "Count"    : [y_test_cm1.eq(1).sum(), y_test_cm1.eq(0).sum(), 
                      y_test_im1.eq(1).sum(), y_test_im1.eq(0).sum()] }), 
    pd.DataFrame({
        "Subtype"  : ["TNBC"] * 4, "Response": ["R", "NR"] * 2, 
        "Treatment": ["Chemotherapy"] * 2 + ["Chemotherapy + ICB"] * 2, 
        "Count"    : [0, 0, y_test_im2.eq(1).sum(), y_test_im2.eq(0).sum()] })
]

ds_names   = [f"Zhang et al. (n = {fig_data1C[0].Count.sum()})", 
              f"Bassez et al. (n = {fig_data1C[1].Count.sum()})"]


#%% make fig. 1-II.

svdat = False

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
fig_lblord1  = [["R", "NR"], ["Chemotherapy", "Chemotherapy + ICB"]]
fig_colors1  = [colors[0], colors[1], colors[4], colors[3]]
fig_legend1  = ["R / E", "NR / NE", "Chemotherapy", "Chemotherapy + ICB"]
add_legend   = "\n".join(["*Chemotherapy = Taxane / Taxane-Anthracycline", 
                          " ICB = Atezolizumab / Pembrolizumab"])

fig1_II, ax1_II = plt.subplots(figsize = (14, 7), nrows = 1, ncols = 2)
ax1_II = dict(zip(["C1", "C2"], ax1_II))

ax1_II["C1"] = make_donutplots(data = fig_data1C[0], x = "Count", 
                               outer = "Response", inner = "Treatment", 
                               outer_order = fig_lblord1[0], 
                               inner_order = fig_lblord1[1], 
                               colors = fig_colors1, donut_size = fig_dntsize1, 
                               title = ds_names[0], ax = ax1_II["C1"])

ax1_II["C2"] = make_donutplots(data = fig_data1C[1], x = "Count", 
                               outer = "Response", inner = "Treatment", 
                               outer_order = fig_lblord1[0], 
                               inner_order = fig_lblord1[1], 
                               colors = fig_colors1, donut_size = fig_dntsize1, 
                               title = ds_names[1], ax = ax1_II["C2"])

ax1_II["C2"].legend(labels = fig_legend1, loc = (1.20, 0.45), ncols = 2, 
                    title = f"Response{' ' * 5}Treatment", 
                    alignment = "left", prop = legend_fonts["item"], 
                    title_fontproperties = legend_fonts["title"]);
fig1_II.text(x = 0.71, y = 0.38, s = add_legend, **legend_fonts["item"]);

fig1_II.text(x = 0.0, y = 0.80, s = "C", **panel_fonts);

fig1_II.tight_layout(h_pad = 0, w_pad = 4)
plt.show()


## save figures.
## default DPI is 100. need to update to 600.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file1_II = "dataset_summary_sc.pdf"
    fig1_II.savefig(fig_path + fig_file1_II, dpi = 600)


#%% prepare data for fig. 5 - zhang et al. sc. 

cell_type1 = "B-cells"

## treatment info.
drug_info1 = pd.DataFrame(
    [{"n": len(y), "R": sum(y == 1), "NR": sum(y == 0)} 
     for y in [y_test_cm1, y_test_im1]], 
    index = ["Chemotherapy", "Chemotherapy + ICB"]).reset_index(
    names = "Treatment")
drug_info1["label"] = drug_info1.apply(
    lambda df: f"{df.Treatment} (n = {df.n})", axis = 1).tolist()

## get data for fig. 5A.
fig_data5A = pd.DataFrame({
    "Response" : pd.concat([y_test_cm1, y_test_im1]).replace(
        to_replace = {1: "R", 0: "NR"}), 
    "score"    : pd.concat([y_pred_cm1, y_pred_im1])[cell_type1], 
    "Treatment": reduce(add, drug_info1.apply(
        lambda x: [x.label] * x.n, axis = 1).tolist()) })

fig_stat5A = get_pvals_from_score(
    score_data = fig_data5A, resp_col = "Response", model_col = "Treatment", 
    score_col = "score", resp_vals = ["R", "NR"])

## get data for fig. 5B.
fig_data5B = fig_data5A.copy().replace(
    to_replace = {"Response": {"R": 1, "NR": 0}}).infer_objects(
    copy = False)
fig_data5B["Treatment"] = fig_data5B.Treatment.replace(
    regex = {"Immunotherapy ": "Immunotherapy\n"})


#%% prepare data for fig. 5 - bassez et al. sc. 

cell_type2 = ["B-cells", "Myeloid", "Endothelial"]

## cell type info.
cell_info2 = pd.DataFrame({
    ctp_: pd.concat([y_test_im2, pred_], axis = 1).dropna(
        axis = 0)[
        "Response"].pipe(
        lambda y: {"n": len(y), "E": sum(y == 1), "NE": sum(y == 0)}) 
        for ctp_, pred_ in y_pred_im2[cell_type2].items()}).T.reset_index(
    names = ["cell_type"])
cell_info2["label"] = cell_info2.apply(
    lambda x: f"{x.cell_type} (n = {x.n})", axis = 1)

## get data for fig. 5C.
fig_data5C = pd.concat([
    y_test_im2.replace(to_replace = {1: "E", 0: "NE"}), 
    y_pred_im2[cell_type2]], axis = 1).sort_values(
        by = "Response", ascending = False).melt(
        id_vars = "Response", var_name = "cell_type", 
        value_name = "score", ignore_index = False).dropna(
        axis = 0, subset = ["score"]).replace(
        to_replace = {"cell_type": cell_info2.set_index(
            keys = "cell_type")["label"].to_dict()})

fig_stat5C = get_pvals_from_score(
    score_data = fig_data5C, resp_col = "Response", model_col = "cell_type", 
    score_col = "score", resp_vals = ["E", "NE"])

## get data for fig. 5D.
fig_data5D = fig_data5C.copy().replace(
    to_replace = {"Response": {"E": 1, "NE": 0}}).infer_objects(
    copy = False)


#%% make fig. 5.

svdat = False

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

## all plots.
fig5, ax5 = plt.subplots(figsize = (16, 7), nrows = 2, ncols = 2, 
                         height_ratios = [1, 1], width_ratios = [1, 1])
ax5 = dict(zip(list("ABCD"), ax5.ravel()))

fig_llocs5 = [[0.08, 0.44], [0.96, 0.52]]
fig_ttls5  = {"A": f"Zhang et al. (n = {fig_data5A.index.nunique()})\n", 
              "B": f"Zhang et al. (n = {fig_data5B.index.nunique()})\n", 
              "C": f"Bassez et al. (n = {fig_data5C.index.nunique()})\n", 
              "D": f"Bassez et al. (n = {fig_data5D.index.nunique()})\n"}


## make violins - zhang et al. sc.
fig_ylim5A = [0.2, 0.4];   fig_ploc5A = 0.4

ax5["A"]   = make_violinplot(data = fig_data5A, x = "Treatment", y = "score", 
                             hue = "Response", stats = fig_stat5A, 
                             hue_order = ["R", "NR"], inner = "quart", 
                             split = True, dodge =  True, statloc = fig_ploc5A, 
                             statline = False, title = fig_ttls5["A"], 
                             legend_vert = True, legend_out = False, 
                             legend_title = "Response", ax = ax5["A"])
ax5["A"].set_ylim([0 - fig_ylim5A[0], 1 + fig_ylim5A[1]]);
ax5["A"].set_yticks(ticks  = np.arange(0, 1.2, 0.2), 
                    labels = np.arange(0, 1.2, 0.2));
ax5["A"].yaxis.set_major_formatter("{x:0.1f}");
ax5["A"].set_xticks(ticks  = range(len(drug_info1)), 
                    labels = drug_info1.label.map(
                        lambda x: x.replace(" (", "\n(")));
ax5["A"].tick_params(axis = "both", labelsize = label_fonts["size"]);
ax5["A"].set_ylabel("Prediction score", **legend_fonts["item"]);
ax5["A"].get_legend().set_bbox_to_anchor([-0.60, 0.25, 0.4, 0.4]);
fig5.text(x = fig_llocs5[0][0], y = fig_llocs5[1][0], s = "A", **panel_fonts);


## make roc curves - zhang et al. sc.
fig_colors5B = [colors[3], colors[4], colors[-1]]

ax5["B"]     = make_roc_plot(data = fig_data5B, label = "Response", 
                             pred = "score", group = "Treatment", 
                             colors = fig_colors5B, fill = True, 
                             alpha = 0.15, title = fig_ttls5["B"], 
                             ax = ax5["B"])
fig5.text(x = fig_llocs5[0][1], y = fig_llocs5[1][0], s = "B", **panel_fonts);

## make violins - bassez et al. sc.
fig_ylim5C  = [0.8, 0.7];   fig_ploc5C = 0.7

ax5["C"]    = make_violinplot(data = fig_data5C, x = "cell_type", y = "score", 
                              hue = "Response", stats = fig_stat5C, 
                              hue_order = ["E", "NE"], inner = "quart", 
                              split = True, dodge =  True, statloc = fig_ploc5C, 
                              statline = False, title = fig_ttls5["C"], 
                              legend_vert = True, legend_out = False, 
                              legend_title = "Clonotype\nExpansion", 
                              ax = ax5["C"])
ax5["C"].set_ylim([0 - fig_ylim5C[0], 1 + fig_ylim5C[1]]);
ax5["C"].set_yticks(ticks  = np.arange(0, 1.2, 0.2), 
                    labels = np.arange(0, 1.2, 0.2));
ax5["C"].yaxis.set_major_formatter("{x:0.1f}");
ax5["C"].set_xticks(ticks  = range(len(cell_info2)), 
                    labels = cell_info2.label.map(
                        lambda x: x.replace(" (", "\n(")));
ax5["C"].tick_params(axis = "both", labelsize = label_fonts["size"]);
ax5["C"].set_ylabel("Prediction score", **legend_fonts["item"]);
ax5["C"].get_legend().set_bbox_to_anchor([-0.60, 0.35, 0.4, 0.4]);
fig5.text(x = fig_llocs5[0][0], y = fig_llocs5[1][1], s = "C", **panel_fonts);


## make roc curves - zhang et al. sc.
fig_colors5D = [colors[3], colors[4], colors[5], colors[-1]]

ax5["D"]     = make_roc_plot(data = fig_data5D, label = "Response", 
                             pred = "score", group = "cell_type", 
                             colors = fig_colors5D, fill = True, 
                             alpha = 0.15, title = fig_ttls5["D"], 
                             legend_title = "Cell type", ax = ax5["D"])
fig5.text(x = fig_llocs5[0][1], y = fig_llocs5[1][1], s = "D", **panel_fonts);

fig5.tight_layout(h_pad = 1, w_pad = 4)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file5 = "all_predictions_sc_survival_chemo_immuno_th0.99_ENS2_5foldCV_v2.pdf"
    fig5.savefig(fig_path + fig_file5, dpi = 600)


#%% prepare data for supp. fig. 7.

def get_cell_info(y_pred):
    cell_info = y_pred.apply(
        lambda x: x.dropna().size).reset_index().set_axis(
        labels = ["cell_type", "n"], axis = 1)
    
    cell_info["label"] = cell_info.apply(
        lambda x: f"{x.cell_type} (n = {x.n})", axis = 1).replace(
        regex = {"Bulk": "Pseudobulk"})
    
    return cell_info
    

## get cell type info.
cell_info_cm1 = get_cell_info(y_pred_cm1)
cell_info_im1 = get_cell_info(y_pred_im1)
cell_info_im2 = get_cell_info(y_pred_im2)

## prepare data for supp. fig. 7-I.
fig_dataS7A   = pd.concat([y_test_cm1.replace(to_replace = {1: "R", 0: "NR"}), 
                           y_pred_cm1.rename(columns = {"Bulk": "Pseudobulk"})], 
                          axis = 1).melt(
    id_vars = "Response", var_name = "cell_type", value_name = "score")

fig_dataS7B   = pd.concat([y_test_im1.replace(to_replace = {1: "R", 0: "NR"}), 
                           y_pred_im1.rename(columns = {"Bulk": "Pseudobulk"})], 
                          axis = 1).melt(
    id_vars = "Response", var_name = "cell_type", value_name = "score")

fig_statS7A   = get_pvals_from_score(
    score_data = fig_dataS7A, resp_col = "Response", model_col = "cell_type", 
    score_col = "score", resp_vals = ["R", "NR"])

fig_statS7B   = get_pvals_from_score(
    score_data = fig_dataS7B, resp_col = "Response", model_col = "cell_type", 
    score_col = "score", resp_vals = ["R", "NR"])

fig_dataS7C   = pd.concat([y_test_im2.replace(to_replace = {1: "E", 0: "NE"}), 
                           y_pred_im2.rename(columns = {"Bulk": "Pseudobulk"})], 
                          axis = 1).melt(
    id_vars = "Response", var_name = "cell_type", value_name = "score")

fig_statS7C   = get_pvals_from_score(
    score_data = fig_dataS7C, resp_col = "Response", model_col = "cell_type", 
    score_col = "score", resp_vals = ["E", "NE"])

## prepare data for supp. fig. 7-II.
fig_dataS7DE  = pd.DataFrame({
    "chemo": perf_test_cm1["AUC"], "icb": perf_test_im1["AUC"]}).reset_index(
    names = ["cell_type"])
fig_dataS7DE["cell_type"] = fig_dataS7DE.cell_type.map(
    lambda x: x.replace("_", "\n").replace("Bulk", "Pseudobulk"))

fig_dataS7F   = perf_test_im2[["AUC"]].set_axis(
    labels = ["icb"], axis = 1).reset_index(
    names = ["cell_type"])
fig_dataS7F["cell_type"] = fig_dataS7F.cell_type.map(
    lambda x: x.replace("_", "\n").replace("Bulk", "Pseudobulk"))


#%% make supp. fig. 7-I.

svdat = False

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
fig_ttlsS7  = {"A": "Zhang et al.: Chemotherapy\n", 
               "B": "Zhang et al.: Chemotherapy + ICB\n", 
               "C": "Bassez et al.: Chemotherapy + ICB\n"}

fig_ylimS7  = 0.5;     fig_plocS7 = 0.4
fig_llocsS7 = [[-0.01, 0.44], [0.96, 0.46]]


figS7_I, axS7_I = plt.subplot_mosaic(
    mosaic = [["A", "B"], ["C", "C"]], figsize = (18, 8), 
    height_ratios = [1, 1], width_ratios = [1, 1], sharey = False)

## make violins.
axS7_I["A"] = make_violinplot(data = fig_dataS7A, x = "cell_type", y = "score", 
                              hue = "Response", stats = fig_statS7A, 
                              hue_order = ["R", "NR"], inner = "quart", 
                              split = True, dodge = True, statloc = fig_plocS7, 
                              statline = False, title = fig_ttlsS7["A"], 
                              legend_vert = True, legend_out = True, 
                              legend_title = "Response", ax = axS7_I["A"])
axS7_I["A"].set_ylim([0 - fig_ylimS7, 1 + fig_ylimS7]);
axS7_I["A"].set_yticks(ticks  = np.arange(0, 1.2, 0.2), 
                       labels = np.arange(0, 1.2, 0.2));
axS7_I["A"].set_xticks(ticks  = range(len(cell_info_cm1)), 
                       labels = cell_info_cm1.label.map(
                           lambda x: x.replace(" (", "\n(")));
axS7_I["A"].yaxis.set_major_formatter("{x:0.1f}");
axS7_I["A"].tick_params(axis = "both", labelsize = label_fonts["size"]);
axS7_I["A"].set_ylabel("Prediction score", **legend_fonts["item"]);
axS7_I["A"].legend([ ], [ ]);
figS7_I.text(x = fig_llocsS7[0][0], y = fig_llocsS7[1][0], s = "A", 
             **panel_fonts);

axS7_I["B"] = make_violinplot(data = fig_dataS7B, x = "cell_type", y = "score", 
                              hue = "Response", stats = fig_statS7B, 
                              hue_order = ["R", "NR"], inner = "quart", 
                              split = True, dodge = True, statloc = fig_plocS7, 
                              statline = False, title = fig_ttlsS7["B"], 
                              legend_vert = True, legend_out = True, 
                              legend_title = "Response", ax = axS7_I["B"])
axS7_I["B"].set_ylim([0 - fig_ylimS7, 1 + fig_ylimS7]);
axS7_I["B"].set_yticks(ticks  = np.arange(0, 1.2, 0.2), 
                       labels = np.arange(0, 1.2, 0.2));
axS7_I["B"].yaxis.set_major_formatter("{x:0.1f}");
axS7_I["B"].set_xticks(ticks  = range(len(cell_info_im1)), 
                       labels = cell_info_im1.label.map(
                           lambda x: x.replace(" (", "\n(")));
axS7_I["B"].tick_params(axis = "both", labelsize = label_fonts["size"]);
axS7_I["B"].get_legend().set_bbox_to_anchor([1.06, 0.35, 0.4, 0.4]);
figS7_I.text(x = fig_llocsS7[0][1], y = fig_llocsS7[1][0], s = "B", 
             **panel_fonts);

fig_ylimS7C = [0.8, 0.7];   fig_plocS7C = 0.72

axS7_I["C"] = make_violinplot(data = fig_dataS7C, x = "cell_type", y = "score", 
                              hue = "Response", stats = fig_statS7C, 
                              hue_order = ["E", "NE"], inner = "quart", 
                              split = True, dodge = True, 
                              statloc = fig_plocS7C, 
                              statline = False, title = fig_ttlsS7["C"], 
                              legend_vert = True, legend_out = True, 
                              legend_title = "Clonotype\nExpansion", 
                              ax = axS7_I["C"])
axS7_I["C"].set_ylim([0 - fig_ylimS7C[0], 1 + fig_ylimS7C[1]]);
axS7_I["C"].set_yticks(ticks  = np.arange(0, 1.2, 0.2), 
                       labels = np.arange(0, 1.2, 0.2));
axS7_I["C"].yaxis.set_major_formatter("{x:0.1f}");
axS7_I["C"].set_xticks(ticks  = range(len(cell_info_im2)), 
                       labels = cell_info_im2.label.map(
                           lambda x: x.replace(" (", "\n(").replace("_", "\n")));
axS7_I["C"].tick_params(axis = "both", labelsize = label_fonts["size"]);
axS7_I["C"].set_ylabel("Prediction score", **legend_fonts["item"]);
axS7_I["C"].get_legend().set_bbox_to_anchor([1.03, 0.35, 0.4, 0.4]);
figS7_I.text(x = fig_llocsS7[0][0], y = fig_llocsS7[1][1], s = "C", 
             **panel_fonts);

figS7_I.tight_layout(h_pad = 4, w_pad = 2)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS7_I = "all_predictions_sc_chemo_th0.99_ENS2_allfeatures_5foldCV.pdf"
    figS7_I.savefig(fig_path + fig_fileS7_I, dpi = 600)


#%% make supp. fig. 7-II.

svdat = False

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
fig_ttlsS7   = {"D": "Zhang et al.: Chemotherapy", 
                "E": "Zhang et al.: Chemotherapy + ICB"}
fig_llocsS7  = [0.02, 0.52]


## make radar axes/ticks.
fig_thetaS7 = RadarChart(num_vars = len(fig_dataS7DE), frame = "circle")
fig_baseS7  = [0.5] * len(fig_thetaS7)
fig_ticksS7 = ["B-cells", "Myeloid" + " " * 4, "T-cells",                      # only 4 ticks- so fix manually
               " " * 10 + "Pseudobulk"]

figS7_II, axS7_II = plt.subplots(figsize = (12, 5), nrows = 1, ncols = 2, 
                                 subplot_kw = {"projection": "radar"})
axS7_II = dict(zip(list("DE"), axS7_II))

## make radars.
axS7_II["D"] = make_radar_lines(theta = fig_thetaS7, 
                                data = fig_dataS7DE["chemo"], 
                                labels = fig_ticksS7, color = colors[3], 
                                alpha = 0.4, ls = "-", lw = 2, ms = 8, 
                                ax = axS7_II["D"])
axS7_II["D"] = make_radar_lines(theta = fig_thetaS7, data = fig_baseS7, 
                                title = fig_ttlsS7["D"], color = colors[-3], 
                                alpha = 0.15, ls = ":", ms = 8, 
                                ax = axS7_II["D"])
axS7_II["D"].set_rlim([0.25, 1.05]);
figS7_II.text(x = fig_llocsS7[0], y = 0.96, s = "D", **panel_fonts);

axS7_II["E"] = make_radar_lines(theta = fig_thetaS7, 
                                data = fig_dataS7DE["icb"], 
                                labels = fig_ticksS7, color = colors[3], 
                                alpha = 0.4, ls = "-", lw = 2, ms = 8, 
                                ax = axS7_II["E"])
axS7_II["E"] = make_radar_lines(theta = fig_thetaS7, data = fig_baseS7, 
                                title = fig_ttlsS7["E"], color = colors[-3], 
                                alpha = 0.15, ls = ":", ms = 8, 
                                ax = axS7_II["E"])
axS7_II["E"].set_rlim([0.25, 1.05]);
figS7_II.text(x = fig_llocsS7[1], y = 0.96, s = "E", **panel_fonts);

figS7_II.tight_layout(h_pad = 0, w_pad = 0)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS7_II = "all_aucs_sc_zhang_chemo_th0.99_ENS2_allfeatures_5foldCV.pdf"
    figS7_II.savefig(fig_path + fig_fileS7_II, dpi = 600)


#%% make supp. fig. 7-III.

svdat = False

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
fig_ttlsS7  = "Bassez et al.: Chemotherapy + ICB"

## make radar axes/ticks.
fig_thetaS7 = RadarChart(num_vars = len(fig_dataS7F), frame = "circle")
fig_baseS7  = [0.5] * len(fig_thetaS7)
fig_ticksS7 = pad_radar_ticks(
    ticks = fig_dataS7F.cell_type.replace(" (", "\n("), pads = [12, 8])
fig_ticksS7[-1] = " " * 6 + fig_ticksS7[-1]                                    # manually fix pseudobulk label

figS7_III, axS7_III = plt.subplots(figsize = (6, 5), nrows = 1, ncols = 1, 
                                   subplot_kw = {"projection": "radar"})

## make radars.
axS7_III = make_radar_lines(theta = fig_thetaS7, data = fig_dataS7F["icb"], 
                            labels = fig_ticksS7, color = colors[3], 
                            alpha = 0.4, ls = "-", lw = 2, ms = 8, 
                            ax = axS7_III)
axS7_III = make_radar_lines(theta = fig_thetaS7, data = fig_baseS7, 
                            title = fig_ttlsS7, color = colors[-3], 
                            alpha = 0.15, ls = ":", ms = 8, ax = axS7_III)
axS7_III.set_rlim([0.25, 0.85]);
figS7_III.text(x = -0.02, y = 0.94, s = "F", **panel_fonts);

## format legends.
axS7_III.legend(labels = ["Cell type", "Random"], loc = (1.16, 0.45), 
                title = "AUC", prop = legend_fonts["item"], 
                title_fontproperties = legend_fonts["title"])

figS7_III.tight_layout(h_pad = 0, w_pad = 0)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS7_III = "all_aucs_sc_bassez_chemo_th0.99_ENS2_allfeatures_5foldCV.pdf"
    figS7_III.savefig(fig_path + fig_fileS7_III, dpi = 600)


#%% prepare data for supp. fig. 8.
## CE, ENDO, PB, NE, MYL, B, CAF

var_surv   = "OS"
var_group  = "groups_05"

cell_types = ["Cancer_Epithelial", "Endothelial", "Plasmablasts", 
              "Normal_Epithelial", "Myeloid", "B-cells", "CAFs"]

## survival info.
surv_info  = pd.concat([clin_test_surv, y_test_surv], axis = 1).groupby(
    by = "Clinical_subtype", sort = True).apply(
    lambda df: pd.Series({
        "n": len(df), f"{var_surv}_events": df[var_surv].sum(), 
        f"{var_surv}_time_median": df[f"{var_surv}_time"].median()}, 
        dtype = int), 
    include_groups = False).reset_index()
surv_info["label"] = surv_info.apply(
    lambda x: f"TCGA-BRCA: {x.Clinical_subtype} (n = {x.n})", axis = 1)


fig_dataS8 = { }
for ctp in cell_types:
    ## get data for kaplan-meier plot.
    km_data_ctp = pd.concat([
        y_pred_surv[ctp], y_test_surv[[var_surv, f"{var_surv}_time"]], 
        clin_test_surv[["Clinical_subtype"]]], axis = 1)
    
    ## get subtype-specific data.
    km_erpos1  = km_data_ctp.pipe(
        lambda df: df[df.Clinical_subtype.eq("ER+,HER2-") & 
                      df[var_group].eq(1)])
    km_erpos2  = km_data_ctp.pipe(
        lambda df: df[df.Clinical_subtype.eq("ER+,HER2-") & 
                      df[var_group].eq(0)])
    
    km_tnbc1   = km_data_ctp.pipe(
        lambda df: df[df.Clinical_subtype.eq("TNBC") & 
                      df[var_group].eq(1)])
    km_tnbc2   = km_data_ctp.pipe(
        lambda df: df[df.Clinical_subtype.eq("TNBC") & 
                      df[var_group].eq(0)])
    
    ## model for each subtype.
    ## ER+, HER2-.
    fig_data_ctp11 = KaplanMeierFitter(alpha = 0.05).fit(
        event_observed = km_erpos1[var_surv], 
        durations = km_erpos1[f"{var_surv}_time"], 
        label = "High-score")
    fig_data_ctp12 = KaplanMeierFitter(alpha = 0.05).fit(
        event_observed = km_erpos2[var_surv], 
        durations = km_erpos2[f"{var_surv}_time"], 
        label = "Low-score")
    
    ## TNBC.
    fig_data_ctp21 = KaplanMeierFitter(alpha = 0.05).fit(
        event_observed = km_tnbc1[var_surv], 
        durations = km_tnbc1[f"{var_surv}_time"], 
        label = "High-score")
    fig_data_ctp22 = KaplanMeierFitter(alpha = 0.05).fit(
        event_observed = km_tnbc2[var_surv], 
        durations = km_tnbc2[f"{var_surv}_time"], 
        label = "Low-score")
    
    ## perform log-rank test.
    fig_stat_ctp   = {
        "ER+,HER2-": logrank_test(
            event_observed_A = km_erpos1[var_surv], 
            event_observed_B = km_erpos2[var_surv], 
            durations_A = km_erpos1[f"{var_surv}_time"], 
            durations_B = km_erpos2[f"{var_surv}_time"]), 
        "TNBC": logrank_test(
            event_observed_A = km_tnbc1[var_surv], 
            event_observed_B = km_tnbc2[var_surv], 
            durations_A = km_tnbc1[f"{var_surv}_time"], 
            durations_B = km_tnbc2[f"{var_surv}_time"])}

    fig_dataS8[ctp] = {
        "ER+,HER2-": {"data_grp1": fig_data_ctp11, "data_grp2": fig_data_ctp12, 
                      "stat": fig_stat_ctp["ER+,HER2-"]}, 
        "TNBC"     : {"data_grp1": fig_data_ctp21, "data_grp2": fig_data_ctp22, 
                      "stat": fig_stat_ctp["TNBC"]} }


#%% prepare data for supp. fig. 9.
## CE, ENDO, PB, NE, MYL, B, CAF

var_surv   = "PFI"
var_group  = "groups_05"

cell_types = ["Cancer_Epithelial", "Endothelial", "Plasmablasts", 
              "Normal_Epithelial", "Myeloid", "B-cells", "CAFs"]

## survival info.
surv_info  = pd.concat([clin_test_surv, y_test_surv], axis = 1).groupby(
    by = "Clinical_subtype", sort = True).apply(
    lambda df: pd.Series({
        "n": len(df), f"{var_surv}_events": df[var_surv].sum(), 
        f"{var_surv}_time_median": df[f"{var_surv}_time"].median()}, 
        dtype = int), 
    include_groups = False).reset_index()
surv_info["label"] = surv_info.apply(
    lambda x: f"TCGA-BRCA: {x.Clinical_subtype} (n = {x.n})", axis = 1)


fig_dataS9 = { }
for ctp in cell_types:
    ## get data for kaplan-meier plot.
    km_data_ctp = pd.concat([
        y_pred_surv[ctp], y_test_surv[[var_surv, f"{var_surv}_time"]], 
        clin_test_surv[["Clinical_subtype"]]], axis = 1)
    
    ## get subtype-specific data.
    km_erpos1  = km_data_ctp.pipe(
        lambda df: df[df.Clinical_subtype.eq("ER+,HER2-") & 
                      df[var_group].eq(1)])
    km_erpos2  = km_data_ctp.pipe(
        lambda df: df[df.Clinical_subtype.eq("ER+,HER2-") & 
                      df[var_group].eq(0)])
    
    km_tnbc1   = km_data_ctp.pipe(
        lambda df: df[df.Clinical_subtype.eq("TNBC") & 
                      df[var_group].eq(1)])
    km_tnbc2   = km_data_ctp.pipe(
        lambda df: df[df.Clinical_subtype.eq("TNBC") & 
                      df[var_group].eq(0)])
    
    ## model for each subtype.
    ## ER+, HER2-.
    fig_data_ctp11 = KaplanMeierFitter(alpha = 0.05).fit(
        event_observed = km_erpos1[var_surv], 
        durations = km_erpos1[f"{var_surv}_time"], 
        label = "High-score")
    fig_data_ctp12 = KaplanMeierFitter(alpha = 0.05).fit(
        event_observed = km_erpos2[var_surv], 
        durations = km_erpos2[f"{var_surv}_time"], 
        label = "Low-score")
    
    ## TNBC.
    fig_data_ctp21 = KaplanMeierFitter(alpha = 0.05).fit(
        event_observed = km_tnbc1[var_surv], 
        durations = km_tnbc1[f"{var_surv}_time"], 
        label = "High-score")
    fig_data_ctp22 = KaplanMeierFitter(alpha = 0.05).fit(
        event_observed = km_tnbc2[var_surv], 
        durations = km_tnbc2[f"{var_surv}_time"], 
        label = "Low-score")
    
    ## perform log-rank test.
    fig_stat_ctp   = {
        "ER+,HER2-": logrank_test(
            event_observed_A = km_erpos1[var_surv], 
            event_observed_B = km_erpos2[var_surv], 
            durations_A = km_erpos1[f"{var_surv}_time"], 
            durations_B = km_erpos2[f"{var_surv}_time"]), 
        "TNBC": logrank_test(
            event_observed_A = km_tnbc1[var_surv], 
            event_observed_B = km_tnbc2[var_surv], 
            durations_A = km_tnbc1[f"{var_surv}_time"], 
            durations_B = km_tnbc2[f"{var_surv}_time"])}

    fig_dataS9[ctp] = {
        "ER+,HER2-": {"data_grp1": fig_data_ctp11, "data_grp2": fig_data_ctp12, 
                      "stat": fig_stat_ctp["ER+,HER2-"]}, 
        "TNBC"     : {"data_grp1": fig_data_ctp21, "data_grp2": fig_data_ctp22, 
                      "stat": fig_stat_ctp["TNBC"]} }


#%% make supp. fig. 8.

svdat = False

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

## all plots.
fig_llocsS8  = [[0.00, 0.45], (np.arange(7, 0, -1) / 7 - 0.02).tolist()]
fig_labelsS8 = np.reshape(list(string.ascii_uppercase[:14]), [7, 2])

figS8, axS8 = plt.subplots(figsize = (16, 18), nrows = 7, ncols = 2)
for k, (lbls, (ctp, ctp_data)) in enumerate(
        zip(fig_labelsS8, fig_dataS8.items())):
    ttl = "Cell type = " + ctp.replace('_', ' ')
    axS8[k, 0] = make_km_plot(
        **ctp_data["ER+,HER2-"], title = ttl, risk_counts = False, 
        legend = False, ax = axS8[k, 0])
    axS8[k, 0].set_xlabel(None);    axS8[k, 0].set_ylabel(None)
    figS8.text(x = fig_llocsS8[0][0], y = fig_llocsS8[1][k], s = lbls[0], 
               **panel_fonts)
    
    axS8[k, 1] = make_km_plot(
        **ctp_data["TNBC"], title = ttl, risk_counts = False, 
        legend = bool(k == 3), ax = axS8[k, 1])
    axS8[k, 1].set_xlabel(None);    axS8[k, 1].set_ylabel(None)
    figS8.text(x = fig_llocsS8[0][1], y = fig_llocsS8[1][k], s = lbls[1], 
               **panel_fonts)
    
    if k == 0:
        axS8[k, 0].set_title(
            surv_info.label[0].replace(":", " OS:") + "\n" + ttl, 
            y = 1.02, **legend_fonts["title"]);
        axS8[k, 1].set_title(
            surv_info.label[1].replace(":", " OS:") + "\n" + ttl, 
            y = 1.02, **legend_fonts["title"]);

figS8.supxlabel("Time in days", y = 0.00, x = 0.47, **label_fonts);
figS8.supylabel("Survival probability", x = -0.02, y = 0.50, **label_fonts);

figS8.tight_layout(h_pad = 4, w_pad = 6)

plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS8 = "all_predictions_survival_os_tcga_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS8.savefig(fig_path + fig_fileS8, dpi = 600)


#%% make supp. fig. 9.

svdat = False

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

## all plots.
fig_llocsS9  = [[0.00, 0.45], (np.arange(7, 0, -1) / 7 - 0.02).tolist()]
fig_labelsS9 = np.reshape(list(string.ascii_uppercase[:14]), [7, 2])

figS9, axS9 = plt.subplots(figsize = (16, 18), nrows = 7, ncols = 2)
for k, (lbls, (ctp, ctp_data)) in enumerate(
        zip(fig_labelsS9, fig_dataS9.items())):
    ttl = "Cell type = " + ctp.replace('_', ' ')
    axS9[k, 0] = make_km_plot(
        **ctp_data["ER+,HER2-"], title = ttl, risk_counts = False, 
        legend = False, ax = axS9[k, 0])
    axS9[k, 0].set_xlabel(None);    axS9[k, 0].set_ylabel(None)
    figS9.text(x = fig_llocsS9[0][0], y = fig_llocsS9[1][k], s = lbls[0], 
               **panel_fonts)
    
    axS9[k, 1] = make_km_plot(
        **ctp_data["TNBC"], title = ttl, risk_counts = False, 
        legend = bool(k == 3), ax = axS9[k, 1])
    axS9[k, 1].set_xlabel(None);    axS9[k, 1].set_ylabel(None)
    figS9.text(x = fig_llocsS9[0][1], y = fig_llocsS9[1][k], s = lbls[1], 
               **panel_fonts)
    
    if k == 0:
        axS9[k, 0].set_title(
            surv_info.label[0].replace(":", " PFI:") + "\n" + ttl, 
            y = 1.02, **legend_fonts["title"]);
        axS9[k, 1].set_title(
            surv_info.label[1].replace(":", " PFI:") + "\n" + ttl, 
            y = 1.02, **legend_fonts["title"]);

figS9.supxlabel("Time in days", y = 0.00, x = 0.47, **label_fonts);
figS9.supylabel("Survival probability", x = -0.02, y = 0.50, **label_fonts);

figS9.tight_layout(h_pad = 4, w_pad = 6)

plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS9 = "all_predictions_survival_pfi_tcga_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS9.savefig(fig_path + fig_fileS9, dpi = 600)

