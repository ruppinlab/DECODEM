#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:58:37 2024

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

import numpy as np, pandas as pd, pickle
import matplotlib.pyplot as plt, seaborn as sns
# from argparse import ArgumentParser
from miscellaneous import date_time, tic, write_xlsx
# from itertools import combinations
from math import nan, ceil, floor
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from _functions import (MakeClassifier, EnsembleClassifier, train_pipeline, 
                        predict_proba_scaled, get_best_threshold, 
                        classifier_performance, binary_performance, 
                        make_barplot2)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from functools import reduce


#%% functions.

## read data into different objects.
def read_data(file):
    ## load data.
    with open(file, "rb") as f:
        obj  = pickle.load(f)
        exp, resp, frac, conf, clin = \
            obj["exp"], obj["resp"], obj["frac"], obj["conf"], obj["clin"]
    
    ## sanity check.
    if conf.columns.tolist() == frac.columns.tolist():
        cells = conf.columns.tolist()
    else:
        raise ValueError("cell types are not the same between cell fraction and confidence score matrices!")
    
    return exp, resp, frac, conf, clin, cells


def rescale(X, mode = "norm"):
    if mode.lower() == "std":
        X[:] = StandardScaler().fit_transform(X)
    else:
        X[:] = MinMaxScaler().fit_transform(X)
    return X


## make nested barplots for multiple variables.
def make_barplot3_base(data, x, y, hue, ax, fontdict, title, xlabels, 
                       xrot = 0, bar_labels = True, ylabel = None, 
                       width = 0.8, bline = False, yrefs = None, 
                       legend = True, legend_title = None, skipcolor = False):
    ## parameters.
    # col_palette = "tab10"
    if data[hue].nunique() > 3:
        # col_palette = "CMRmap"
        # col_palette = "crest"
        col_palette = "tab20_r"
    else:
        col_list = ["#75D0B0", "#FFC72C", "#88BECD"]
        if skipcolor:
            col_list = col_list[1:] + [col_list[0]]
        
        col_palette = dict(zip(data[hue].unique(), col_list))
    
    col_bline  = "#DCDDDE"
    bline_prop = {"linestyle": "--", "linewidth": 2}
    
    n_bars = data[hue].nunique()
    if bar_labels:    
        bar_lbl = [data[data[hue].eq(bb)][y].round(2) \
                   for bb in data[hue].unique()]                               # bar heights as labels
        bar_font = fontdict["label"].copy()
        bar_font["fontsize"] -= 8
    
    if data[y].max() <= 1:
        yticks = np.arange(0, 1.4, 0.2)
    else:
        yticks = np.linspace(
            0, np.round(data[y].max() * 1.2), num = 7).round(1)
        
    ylabels = yticks.round(2).tolist()[:-1] + [""]
    
    # lgnd_loc = [0.8, 0.84]
    # lgnd_loc = "best"
    lgnd_loc  = "lower left"
    lgnd_bbox = (1.0, 0.5, 0.6, 0.6)
    lgnd_font = {pn.replace("font", ""): pv \
                 for pn, pv in fontdict["label"].items()}
    
    ## build plot.
    sns.barplot(data = data, x = x, y = y, hue = hue, orient = "v", 
                width = width, palette = col_palette, edgecolor = [0.3]*3, 
                saturation = 0.9, dodge = True, ax = ax);                      # nested barplots
    if bar_labels:
        [ax.bar_label(ax.containers[nn], labels = bar_lbl[nn], padding = 0.2, 
                      rotation = 40, **bar_font) \
         for nn in range(n_bars)];
    
    ## add baseline.
    if bline & (yrefs is not None):
        [ax.axhline(y = yrefs[nn], xmin = 0, xmax = data.shape[0] / n_bars, 
                    color = col_bline, **bline_prop) \
         for nn in range(n_bars)];
        
    ## format plot ticks & labels.
    ax.set_title(title, y = 1.02, **fontdict["super"]);
    ax.set_xlabel(None);     ax.set_ylabel(ylabel, **fontdict["label"]);
    # ax.set_ylim([yticks.min() - 0.1, yticks.max()]);
    ax.set_yticks(ticks = yticks, labels = ylabels, **fontdict["label"]);
    if xrot != 0:
        ax.set_xticklabels(xlabels, rotation = xrot, rotation_mode = "anchor", 
                           ha = "right", va = "center", ma = "center", 
                           position = (0, -0.02), **fontdict["label"]);
    else:
        ax.set_xticklabels(xlabels, ma = "center", position = (0, -0.02), 
                           **fontdict["label"]);
    
    if legend:
        lgnd_ttl = {pn.replace("font", ""): pv \
                    for pn, pv in fontdict["title"].items()}
        # lgnd_ttl.update({"multialignment": "center"})
        lgnd = ax.legend(loc = lgnd_loc, bbox_to_anchor = lgnd_bbox, 
                  prop = lgnd_font, title = legend_title, 
                  title_fontproperties = lgnd_ttl, frameon = True);
        lgnd.get_title().set_multialignment("center");
    else:
        ax.legend([ ], [ ], frameon = False)
    
    return ax


## make nested barplots with breaks for multiple variables.
def make_barplot3(data, x, y, hue, ax, fontdict, title, xlabels, xrot = 0, 
                  bar_labels = False, ylabel = None, width = 0.8, 
                  bline = False, yrefs = None, legend = True, 
                  legend_title = None, skipcolor = False):
    
    if np.isinf(data[y].max()):                                                # make nested barplots with breaks
        max_val  = 200
        data_brk = data.replace({y: {np.inf: max_val}})                        # replace infinity with a dummy value
        
        ax_t, ax_b = ax
        ax_t_b, ax_b_t = 60, 45
        
        ax_t = make_barplot3_base(
            data = data_brk, x = x, y = y, hue = hue, width = width, 
            title = title, bar_labels = bar_labels, xlabels = xlabels, 
            xrot = xrot, ylabel = ylabel, bline = bline, yrefs = yrefs, 
            legend = False, legend_title = legend_title, 
            skipcolor = skipcolor, ax = ax_t, fontdict = fontdict)
        ax_t.set_xlim([-0.5, len(xlabels) - 0.5]);
        ax_t.set_ylim([ax_t_b, np.round(max_val * 1.05)]);
        ax_t.spines["bottom"].set_visible(False);
        ax_t.set_yticks(ticks = np.linspace(80, max_val, num = 3), 
                        labels = [100, 1000, "$\infty$"], **fontdict["label"]);
        ax_t.tick_params(axis = "x", which = "both", bottom = False, 
                         labelbottom = False);
        
        data_brk.loc[data_brk[y].gt(ax_b_t), y] = ax_b_t                       # hack for new py version (?)
        ax_b = make_barplot3_base(
            data = data_brk, x = x, y = y, hue = hue, width = width, 
            title = None, bar_labels = bar_labels, xlabels = xlabels, 
            xrot = xrot, ylabel = ylabel, bline = bline, yrefs = yrefs, 
            legend = legend, legend_title = legend_title, 
            skipcolor = skipcolor, ax = ax_b, fontdict = fontdict)
        ax_b.set_xlim([-0.5, len(xlabels) - 0.5]);
        ax_b.set_ylim([0, ax_b_t]);     ax_b.spines["top"].set_visible(False);
        ax_b.set_yticks(ticks = [0, 20, 40], labels = [0, 20, 40], 
                        **fontdict["label"]);
        
        ## add breakpoint indicators.
        d_brk = 0.005                                                          # axis breakpoint tick size
        brkargs = dict(transform = ax_t.transAxes, color = "black", 
                       clip_on = False)
        ax_t.plot((-d_brk, +d_brk), (-d_brk, +d_brk), **brkargs)               # add breakpoint marker tick for top plot
        ax_t.plot((1 - d_brk, 1 + d_brk), (-d_brk, +d_brk), **brkargs)
        
        brkargs.update(transform = ax_b.transAxes, color = "black", 
                       clip_on = False)
        ax_b.plot((-d_brk, +d_brk), (1 - d_brk, 1 + d_brk), **brkargs)         # add breakpoint marker tick for bottom plot
        ax_b.plot((1 - d_brk, 1 + d_brk), (1 - d_brk, 1 + d_brk), **brkargs)
        
        if legend:
            ax_b.get_legend().set(bbox_to_anchor = (1.0, 0.0, 0.6, 0.6))       # set legend location
        
        ax = ax_t, ax_b
        
    else:
        ax = make_barplot3_base(
            data = data, x = x, y = y, hue = hue, width = width, title = title, 
            bar_labels = bar_labels, xlabels = xlabels, xrot = xrot, 
            ylabel = ylabel, bline = bline, yrefs = yrefs, legend = legend, 
            legend_title = legend_title, skipcolor = skipcolor, ax = ax, 
            fontdict = fontdict)
        
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


#%% read data.

## specify sample subset.
use_samples = "chemo"
use_samples = use_samples.replace("+", "_")

## load data.
data_path = ["../../data/TransNEO/transneo_analysis/", 
             "../../data/TransNEO/TransNEO_SammutShare/validation/", 
             "../../data/BrighTNess/validation/"]

data_file = [f"transneo_data_{use_samples}_v2.pkl", 
             f"transneo_validation_{use_samples}_v2.pkl", 
             f"brightness_data_{use_samples}_v2.pkl"]

(exp_all_tn, resp_pCR_tn, cell_frac_tn, 
 conf_score_tn, clin_info_tn, _) = read_data(data_path[0] + data_file[0])

(exp_all_tn_val, resp_pCR_tn_val, cell_frac_tn_val, 
 conf_score_tn_val, clin_info_tn_val, _) = read_data(data_path[1] + data_file[1])

(exp_all_bn, resp_pCR_bn, cell_frac_bn, 
 conf_score_bn, clin_info_bn, _) = read_data(data_path[2] + data_file[2])
clin_info_bn = clin_info_bn[clin_info_bn["planned_arm_code"] == "B"]           # keep arm B only
exp_all_bn = {ctp_: exp_[clin_info_bn.index] \
              for ctp_, exp_ in exp_all_bn.items()}
resp_pCR_bn, cell_frac_bn = \
    resp_pCR_bn.loc[clin_info_bn.index], cell_frac_bn.loc[clin_info_bn.index]


cell_types = reduce(np.intersect1d, map(lambda df: df.columns, [
    cell_frac_tn, cell_frac_tn_val, cell_frac_bn])).tolist()

print("\ncell types =", *cell_types, sep = "\n\t")


#%% association of abundance & response per cell type.

ds_info = pd.DataFrame(
    [{"n": len(y), "R": sum(y == 1), "NR": sum(y == 0)} 
     for y in [resp_pCR_tn, resp_pCR_tn_val, resp_pCR_bn]], 
    index = ["TransNEO", "ARTemis + PBCP", "BrighTNess"]).reset_index(
    names = "Dataset")
ds_info["label"] = ds_info.apply(
    lambda x: f"{x.Dataset} (n = {x.n})", axis = 1)


y_pred_frac_tn    = rescale(cell_frac_tn, mode = "std")
perf_test_frac_tn = pd.DataFrame({
    ctp_: classifier_performance(resp_pCR_tn, pred_) \
        for ctp_, pred_ in y_pred_frac_tn.items()}).T

y_pred_frac_tn_val    = rescale(cell_frac_tn_val, mode = "std")
perf_test_frac_tn_val = pd.DataFrame({
    ctp_: classifier_performance(resp_pCR_tn_val, pred_) \
        for ctp_, pred_ in y_pred_frac_tn_val.items()}).T

y_pred_frac_bn    = rescale(cell_frac_bn, mode = "std")
perf_test_frac_bn = pd.DataFrame({
    ctp_: classifier_performance(resp_pCR_bn, pred_) \
        for ctp_, pred_ in y_pred_frac_bn.items()}).T


## generate data for visualization - supp. fig. 4-II.
mdl_ord     = perf_test_frac_tn.sort_values(
    by = ["AUC", "AP"], ascending = False).index.tolist()
mdl_names   = [mdl.replace("_", "\n") for mdl in mdl_ord]

fig_dataS4E = pd.DataFrame({ds: perf["AUC"] for ds, perf in zip(
    ds_info.label, [perf_test_frac_tn, perf_test_frac_tn_val, 
                    perf_test_frac_bn])}).loc[
    mdl_ord].reset_index(
    names = "model")

fig_dataS4F = pd.DataFrame({ds: perf["AP"] for ds, perf in zip(
    ds_info.label, [perf_test_frac_tn, perf_test_frac_tn_val, 
                    perf_test_frac_bn])}).loc[
    mdl_ord].reset_index(
    names = "model")

fig_dataS4_II = [fig_dataS4E, fig_dataS4F]


#%% generate supp. fig. 4-II.

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
fig_thetaS4  = RadarChart(num_vars = len(mdl_names), frame = "circle")
fig_baseS4   = [0.5] * len(fig_thetaS4)
fig_ttlsS4   = ["Cell fractions: AUC", "Cell fractions: AP"]
fig_llocsS4  = [0.01, 0.38]
fig_ticksS4  = pad_radar_ticks(ticks = mdl_names, pads = [12, 8])
fig_colorsS4 = [colors[4], colors[3], colors[5], colors[-1]]


figS4_II, axS4_II = plt.subplots(figsize = (14, 5), dpi = 600, 
                                 nrows = 1, ncols = 2, 
                                 subplot_kw = {"projection": "radar"})
axS4_II = dict(zip(list("EF"), axS4_II))

## make radars.
for ds, clr in zip(ds_info.label, fig_colorsS4):
    ## radar for aucs.
    axS4_II["E"] = make_radar_lines(theta = fig_thetaS4, data = fig_dataS4E[ds], 
                                    labels = fig_ticksS4, color = clr, 
                                    alpha = 0.4, ls = "-", lw = 2, ms = 8, 
                                    title = fig_ttlsS4[0], ax = axS4_II["E"])
    axS4_II["E"].set_rlim([0.10, 0.95]);
    
    ## radar for aps.
    axS4_II["F"] = make_radar_lines(theta = fig_thetaS4, data = fig_dataS4F[ds], 
                                    labels = fig_ticksS4, color = clr, 
                                    alpha = 0.4, ls = "-", lw = 2, ms = 8, 
                                    title = fig_ttlsS4[1], ax = axS4_II["F"])
    axS4_II["F"].set_rlim([0.10, 0.75]);

## format legends.
axS4_II["F"].legend(labels = ds_info.label, loc = (1.36, 0.4), 
                    title = "Dataset", prop = legend_fonts["item"], 
                    title_fontproperties = legend_fonts["title"])

[figS4_II.text(x = loc, y = 0.9, s = lbl, **panel_fonts) 
 for loc, lbl in zip(fig_llocsS4, axS4_II.keys())];

figS4_II.tight_layout(h_pad = 0, w_pad = 7)
plt.show()


## save figures.
if svdat:
    fig_path  = data_path[0] + "plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS4 = "all_chemo_abundance_response_association.pdf"
    figS4_II.savefig(fig_path + fig_fileS4, dpi = "figure")

