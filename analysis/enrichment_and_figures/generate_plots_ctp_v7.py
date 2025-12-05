#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:28:46 2025

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
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from math import nan, ceil, floor
from scipy.stats import mannwhitneyu
from itertools import product
from _functions import classifier_performance, make_barplot2
from warnings import filterwarnings


#%% functions.

## load model predictions from saved pickle.
def read_pkl_data(data_path):
    with open(data_path, mode = "rb") as file:
        data_obj  = pickle.load(file)
        y_test    = data_obj["label"]
        y_pred    = data_obj["pred"]
        th_test   = data_obj["th"]
        perf_test = data_obj["perf"]
        del data_obj
        
    return y_test, y_pred, th_test, perf_test


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
                  "#FFC72C", "#A9A9A9", "#000000"]
    
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
    # calculate evenly-spaced axis angles
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


## make horizontal barplots.
def make_hbars(data, x, y, ax, width = 0.8, color = None, xlabel = None, 
               ylabel = None, title = None, fontdict = None):
    ## plot parameters.
    if color is None:
        colors = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
                  "#FFC72C", "#A9A9A9", "#000000"]
        colors = [colors[4], colors[-1]]
    else:
        colors = [color, "#000000"]
    
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    barprop = {"linestyle": "-", "linewidth": 2, "edgecolor": colors[-1]}
    
    sns.barplot(data = data, x = x, y = y, orient = "h", width = width, 
                color = colors[0], saturation = 0.7, fill = True, 
                dodge = True, **barprop, ax = ax)
    # ax.bar_label(ax.containers[0], fmt = "%0.2f", padding = 0.4, 
    #              **fontdict["label"]);
    sns.despine(ax = ax, offset = 0, trim = False);
    
    
    ## format axis ticks & labels.
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.set_xlabel(xlabel, **fontdict["label"]);
    ax.set_ylabel(ylabel, **fontdict["label"]);
    ax.set_title(title, wrap = True, y = 1.02, **fontdict["title"]);
    
    return ax


## make barplots with continuation (for diagnostic odds ratio).
## cut plot at maxplt & add dots to indicate continuation.
def make_dot_barplot(data, x, y, hue, ax, maxplt = 5, dgts = 1, baseline = True, 
                     xrot = 0, xlabel = None, ylabel = "Odds ratio", 
                     title = None, legend_title = None, colors = None, 
                     fontdict = None):
    ## plot parameters. 
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    if colors is None:
        colors = "tab20b_r"
    else:
        colors = colors[:data[hue].nunique()]
    
    barprop  = {"ls": "-", "lw": 2, "ec": "#000000"}
    lineprop = {"ls": "--", "lw": 2.5, "color": "#000000"}
    mrkrprop = {"marker": "o", "ms": 2.5, "color": "#000000"}
    
    
    ## prepare data. 
    data_plt = data.copy()
    data_plt[y] = np.where(data[y] > maxplt, maxplt, data[y])                  # bound data to maxplt
    num_bars = data_plt[hue].nunique()                                         # #grouped bars
    
    bar_lbls = data.groupby(                                                   # formatted bar labels
        by = hue, sort = False).apply(
        lambda df: [("$\\bf\infty$" if np.isinf(val) else str(val)) + 
                    ("\n" * 2 if val > maxplt else "") 
                    for val in df[y].round(dgts)], 
        include_groups = False).tolist()
    
    bar_offx = np.linspace(-0.25, 0.25, num = num_bars)                        # offsets of each bar in grouped barplot
    dot_offy = 0.15                                                            # offset between bar & continuing elipsis
    
    
    ## make grouped barplot. 
    sns.barplot(data = data_plt, x = x, y = y, hue = hue, orient = "v", 
                width = 0.8, dodge = True, gap = 0, palette = colors, 
                saturation = 0.8, **barprop, ax = ax)
    
    ## add baseline.
    if baseline:
        ax.axhline(y = 1.0, xmin = 0, xmax = 0.99, **lineprop)
        
    
    ## add elipsis (...) for bounded bars.
    for bb, lbls in enumerate(data.groupby(
            by = hue, sort = False).apply(
            lambda df: df[y].tolist(), include_groups = False).tolist()):
        for xx, lbl in enumerate(lbls):
            if lbl > maxplt:
                data_dots = ([xx + bar_offx[bb]] * 3, 
                             maxplt + dot_offy + np.arange(3) / 6)
                ax.plot(*data_dots, linewidth = 0, **mrkrprop)
    
    ## add bar labels.
    [ax.bar_label(ax.containers[bb], labels = bar_lbls[bb], padding = 4, 
                  rotation = 0, **fontdict["label"]) for bb in range(num_bars)];
    
    
    ## format axes & legends.
    ax.set_xlim([-0.7, data[x].nunique() - 0.3]);
    ax.set_ylim([0, maxplt + 0.5]);
    ax.set_yticks(np.arange(0, int(maxplt + 1)).round(1));
    ax.set_xticklabels(ax.get_xticklabels(), rotation = xrot, 
                       rotation_mode = "anchor", 
                       ha = "center" if xrot == 0 else "right", 
                       va = "top" if xrot == 0 else "center", ma = "center", 
                       position = (0, 0) if xrot == 0 else (0, -0.02), 
                       **fontdict["label"])
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.set_xlabel(xlabel, labelpad = 6, **fontdict["label"]);
    ax.set_ylabel(ylabel, labelpad = 6, **fontdict["label"]);
    ax.set_title(title, wrap = True, pad = 8, y = 1.12, **fontdict["title"]);
    
    ax.legend(loc = (1.02, 0.35), frameon = False, 
              title = hue if legend_title is None else legend_title, 
              prop = fontdict["label"], 
              title_fontproperties = fontdict["title"]);
    
    return ax


#%% read data.

data_path = ["../../data/TransNEO/transneo_analysis/mdl_data/", 
             "../../data/TransNEO/use_data/", 
             "../../data/TransNEO/TransNEO_SammutShare/", 
             "../../data/BrighTNess/"]

data_file = ["transneo_predictions_chemo_th0.99_ENS2_25features_5foldCV_20Mar2023.pkl", 
             "tn_valid_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl", 
             "brightness_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl", 
             "transneo_predictions_chemo_th0.99_ENS2_25features_LeaveOneOutCV_22Apr2025.pkl", 
             "tn_valid_predictions_chemo_th0.99_ENS2_25features_LeaveOneOutTune_22Apr2025.pkl", 
             "brightness_predictions_chemo_th0.99_ENS2_25features_LeaveOneOutTune_22Apr2025.pkl", 
             "transneo_predictions_weighted_chemo_th0.99_ENS2_25features_5foldCV_12Sep2025.pkl", 
             "tn_valid_predictions_weighted_chemo_th0.99_ENS2_25features_3foldCVtune_12Sep2025.pkl", 
             "brightness_predictions_weighted_chemo_th0.99_ENS2_25features_3foldCVtune_12Sep2025.pkl", 
             "transneo-diagnosis-MLscores.tsv", 
             "TransNEO_SupplementaryTablesAll.xlsx", 
             "transneo-diagnosis-clinical-features.xlsx", 
             "GSE164458_BrighTNess_clinical_info_SRD_04Oct2022.xlsx", 
             "transneo_predictions_v2_chemo_th0.99_ENS2_25features_5foldCV_15Sep2025.pkl", 
             "tn_valid_predictions_v2_chemo_th0.99_ENS2_25features_3foldCVtune_15Sep2025.pkl", 
             "brightness_predictions_v2_chemo_th0.99_ENS2_25features_3foldCVtune_15Sep2025.pkl"]


## model prediction scores.
y_test_tn, y_pred_tn, th_test_tn, perf_test_tn = read_pkl_data(
    data_path[0] + data_file[0])

y_test_tn_val, y_pred_tn_val, th_test_tn_val, perf_test_tn_val = read_pkl_data(
    data_path[0] + data_file[1])
th_test_tn_val = th_test_tn_val["mean"]

y_test_bn, y_pred_bn, th_test_bn, perf_test_bn = read_pkl_data(
    data_path[0] + data_file[2])
th_test_bn     = th_test_bn["mean"]


## model prediction scores - LOO.
y_test_tn_loo, y_pred_tn_loo, th_test_tn_loo, perf_test_tn_loo = read_pkl_data(
    data_path[0] + data_file[3])

y_test_tn_val_loo, y_pred_tn_val_loo, th_test_tn_val_loo, \
    perf_test_tn_val_loo = read_pkl_data(
        data_path[0] + data_file[4])

y_test_bn_loo, y_pred_bn_loo, th_test_bn_loo, perf_test_bn_loo = read_pkl_data(
    data_path[0] + data_file[5])


## model prediction scores - weighted expression.
y_test_tn_wf, y_pred_tn_wf, th_test_tn_wf, perf_test_tn_wf = read_pkl_data(
    data_path[0] + data_file[6])

y_test_tn_val_wf, y_pred_tn_val_wf, th_test_tn_val_wf, \
    perf_test_tn_val_wf = read_pkl_data(
        data_path[0] + data_file[7])

y_test_bn_wf, y_pred_bn_wf, th_test_bn_wf, perf_test_bn_wf = read_pkl_data(
    data_path[0] + data_file[8])


assert (np.allclose(y_test_tn, y_test_tn_loo) and 
        np.allclose(y_test_tn_val, y_test_tn_val_loo) and 
        np.allclose(y_test_bn, y_test_bn_loo) and
        np.allclose(y_test_tn, y_test_tn_wf) and 
        np.allclose(y_test_tn_val, y_test_tn_val_wf) and 
        np.allclose(y_test_bn, y_test_bn_wf))

cell_types = sorted(np.setdiff1d(y_pred_tn.columns, "Bulk"))

## clinical info.
clin_info_tn     = pd.read_excel(
    data_path[1] + data_file[10], sheet_name = "Supplementary Table 1", 
    skiprows = 1, header = 0, index_col = 0)

clin_info_tn_val = pd.read_excel(
    data_path[1] + data_file[10], sheet_name = "Supplementary Table 5", 
    skiprows = 1, header = 0, index_col = 0)

clin_info_bn = pd.read_excel(
    data_path[3] + data_file[12], sheet_name = "samples", 
    header = 0, index_col = 0)

samples_sammut_tn     = clin_info_tn.index.tolist()
samples_sammut_tn_val = clin_info_tn_val.index.tolist()

## clinical data for available samples.
clin_data_tn     = clin_info_tn.loc[y_test_tn.index].copy()
clin_data_tn_val = clin_info_tn_val.loc[y_test_tn_val.index].copy()
clin_data_bn     = clin_info_bn.loc[y_test_bn.index].copy()


# ## clinical info from Sammut et al.
# clin_info_tn_sammut = pd.read_excel(
#     data_path[2] + data_file[11], sheet_name = "training", 
#     header = 0, index_col = 0)

# clin_info_tn_val_sammut = pd.read_excel(
#     data_path[2] + data_file[11], sheet_name = "validation", 
#     header = 0, index_col = 0)


## sammut et al. scores.
y_pred_sammut_all    = pd.read_table(data_path[1] + data_file[9], sep = "\t", 
                                     header = 0, index_col = 0)

y_pred_sammut_tn     = y_pred_sammut_all.pipe(
    lambda df: df[df.Class.eq("Training")]).drop(
    columns = ["Class"]).apply(
    lambda x: (x - x.min()) / (x.max() - x.min()), axis = 0)                   # rescale to spread in [0, 1] for fair comparison

y_pred_sammut_tn_val = y_pred_sammut_all.pipe(
    lambda df: df[df.Class.eq("Validation")]).drop(
    columns = ["Class"]).apply(
    lambda x: (x - x.min()) / (x.max() - x.min()), axis = 0)                   # rescale to spread in [0, 1] for fair comparison
y_pred_sammut_tn_val["Cohort"] = y_pred_sammut_tn_val.index.map(
    lambda idx: "PBCP" if ("PBCP" in idx) else "ARTEMIS")

## harmonize sample IDs for artemis + pbcp.
pbcp_id_conv = dict(zip(
    np.setdiff1d(y_pred_sammut_tn_val.index, samples_sammut_tn_val), 
    np.setdiff1d(samples_sammut_tn_val, y_pred_sammut_tn_val.index) ))

y_pred_sammut_tn_val.rename(index = pbcp_id_conv, inplace = True)


## model prediction scores - v2.
cell_types2 = ["CD4+_T-cells", "CD8+_T-cells"]

y_test_tn2, y_pred_tn2, _, _ = read_pkl_data(
    data_path[0] + data_file[13])

y_test_tn_val2, y_pred_tn_val2, _, _ = read_pkl_data(
    data_path[0] + data_file[14])

y_test_bn2, y_pred_bn2, _, _ = read_pkl_data(
    data_path[0] + data_file[15])

y_pred_tn2     = y_pred_tn2[cell_types2 + ["Bulk"]]
y_pred_tn_val2 = y_pred_tn_val2[cell_types2 + ["Bulk"]]
y_pred_bn2     = y_pred_bn2[cell_types2 + ["Bulk"]]


#%% prepare prediction scores & model performance scores.

## prediction scores - transneo.
samples_tn_sm = np.intersect1d(
    y_pred_tn.index, y_pred_sammut_tn.index).tolist()
y_test_tn_sm  = y_test_tn.loc[samples_tn_sm]
y_pred_tn_sm  = pd.concat([
    y_pred_tn, y_pred_sammut_tn["Clinical+RNA"].rename(
    index = "Sammut et al.")], axis = 1).loc[
    samples_tn_sm]

## prediction scores - artemis + pbcp.
samples_tn_val_sm = np.intersect1d(
    y_test_tn_val.index, y_pred_tn_val.index).tolist()
y_test_tn_val_sm  = y_test_tn_val.loc[samples_tn_val_sm]
y_pred_tn_val_sm  = pd.concat([
    y_pred_tn_val, y_pred_sammut_tn_val["Clinical+RNA"].rename(
    index = "Sammut et al.")], axis = 1).loc[
    samples_tn_val_sm]

## prediction scores - brightness.
y_test_bn_sm = y_test_bn
y_pred_bn_sm = y_pred_bn.copy()
y_pred_bn_sm["Sammut et al."] = nan


## dataset info.
ds_info = pd.DataFrame(
    [{"n": len(y), "R": sum(y == 1), "NR": sum(y == 0)} 
     for y in [y_test_tn_sm, y_test_tn_val_sm, y_test_bn_sm]], 
    index = ["TransNEO", "ARTemis + PBCP", "BrighTNess"]).reset_index(
    names = "Dataset")
ds_info["label"] = ds_info.apply(
    lambda x: f"{x.Dataset} (n = {x.n})", axis = 1)


## performance scores.
perf_test_tn_sm     = pd.DataFrame({
    mdl: classifier_performance(y_test_tn_sm, y_pred) 
    for mdl, y_pred in y_pred_tn_sm.items()}).T

perf_test_tn_val_sm = pd.DataFrame({
    mdl: classifier_performance(y_test_tn_val_sm, y_pred) 
    for mdl, y_pred in y_pred_tn_val_sm.items()}).T

perf_test_bn_sm     = pd.DataFrame({
    mdl: classifier_performance(y_test_bn_sm, y_pred) 
    for mdl, y_pred in y_pred_bn_sm.dropna(axis = 1).items()}).T
perf_test_bn_sm.loc["Sammut et al."] = nan


print(f"""
prepared prediction scores & performance scores!
dataset info:\n{ds_info.set_index(keys = 'Dataset')}\n
performance snapshot: 
{pd.concat([perf_test_tn_sm.AUC, perf_test_tn_val_sm.AUC, perf_test_bn_sm.AUC], 
           axis = 1, keys = ds_info.Dataset).loc[
           cell_types + ['Bulk','Sammut et al.']].round(4)}
""")


#%% prepare data for fig 1.

subtypes_all = ["ER+/HER2-", "TNBC"]                                           # define BC subtype labels
response_all = ["R", "NR"]                                                     # define patient response labels


## make focused clinical data matrix.
clin_data_tn_sm     = clin_data_tn.loc[y_test_tn_sm.index].pipe(
    lambda df: pd.DataFrame({
        "Subtype"  : df["ER.status"].map(
            lambda x: subtypes_all[0] if (x == "POS") else subtypes_all[1]), 
        "Response" : df["pCR.RD"].map(
            lambda x: response_all[0] if (x == "pCR") else response_all[1]), 
        "Treatment": df["NAT.regimen"].replace(
            regex = {"Carboplatin": "Cb"}) }))

clin_data_tn_val_sm = clin_data_tn_val.loc[y_test_tn_val_sm.index].pipe(
    lambda df: pd.DataFrame({
        "Subtype"  : df["ER.status"].map(
            lambda x: subtypes_all[0] if (x == "POS") else subtypes_all[1]), 
        "Response" : df["pCR.RD"].map(
            lambda x: response_all[0] if (x == "pCR") else response_all[1]), 
        "Treatment": df["Chemo.Regimen"].replace(
            regex = {"Carboplatin": "Cb"}) }))

clin_data_bn_sm     = clin_data_bn.loc[y_test_bn_sm.index].pipe(
    lambda df: pd.DataFrame({
        "Subtype"  : subtypes_all[1:] * len(df), 
        "Response" : df["pathologic_complete_response"].map(
            lambda x: response_all[0] if (x == "pCR") else response_all[1]), 
        "Treatment": df["treatment"].replace(
            to_replace = {"Carboplatin+Paclitaxel": "P-Cb"}) }))


## format data for fig 1.
def get_subtype_response_counts(clin):
    stat_data = pd.DataFrame({
        "Subtype" : np.repeat(subtypes_all, repeats = len(response_all)), 
        "Response": np.tile(response_all, reps = len(subtypes_all)) })
    stat_data["Count"] = stat_data.apply(
        lambda x: clin.eq(x).all(axis = 1).sum(), axis = 1)
    
    return stat_data


fig_data1_I = [
    pd.concat([
        clin.drop(columns = "Treatment").pipe(get_subtype_response_counts) 
        for clin in [clin_data_tn_sm, clin_data_tn_val_sm, clin_data_bn_sm]], 
        axis = 0, keys = ds_info.label.values), 
    pd.DataFrame([
        clin.Treatment.value_counts() / len(clin) 
        for clin in [clin_data_tn_sm, clin_data_tn_val_sm, clin_data_bn_sm]], 
        index = ds_info.label.values).T.sort_index(
        ascending = True).reset_index(
        names = "Regimen") ]


#%% make fig 1-I.

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

## nested donut plots + barplots.
fig_dntsize1 = 0.60
fig_legend1  = list(map(lambda x: x + " " * 6, response_all)) + subtypes_all
fig_colors1  = [colors[0], colors[1], colors[2], colors[5]]

fig1_I, ax1_I = plt.subplot_mosaic(
    mosaic = [["A1", "A2", "A3"], ["B1", "B2", "B3"]], 
    figsize = (18, 9), width_ratios = [1, 1, 1])

## make plots.
for k, ds in enumerate(ds_info.label, start = 1):
    ## data distribution.
    ax = ax1_I[f"A{k}"]
    ax = make_donutplots(data = fig_data1_I[0].loc[ds], x = "Count", 
                         outer = "Response", inner = "Subtype", 
                         outer_order = response_all, inner_order = subtypes_all, 
                         donut_size = fig_dntsize1, colors = fig_colors1, 
                         title = ds, ax = ax)
    
    if k == len(ds_info):                                                      # add common legend
        ax.legend(labels = fig_legend1, loc = (1.2, 0.45), ncols = 2, 
                  title = f"Response{' ' * 4}Subtype", alignment = "left", 
                  prop = legend_fonts["item"], 
                  title_fontproperties = legend_fonts["title"]);
        fig1_I.text(x = 0.0, y = 0.95, s = "A", **panel_fonts);                # add panel labels
    
    
    ## drug distribution.
    ax = ax1_I[f"B{k}"]
    ax = make_hbars(data = fig_data1_I[1], x = ds, y = "Regimen", width = 0.8, 
                    color = colors[4], ax = ax1_I[f"B{k}"])
    ax.set_xlim([0, max(0.8, fig_data1_I[1][ds].max())]);
    
    match k:
        case 1:                                                                # add common y-label
            ax.set_ylabel("Treatment regimen", labelpad = 12, **label_fonts);
        case 2:                                                                # add common x-label
            ax.set_xlabel("Proportion of patients", labelpad = 12, 
                          **label_fonts);        
        case _:
            fig1_I.text(x = 0.0, y = 0.48, s = "B", **panel_fonts);            # add panel labels

fig1_I.tight_layout(h_pad = 2, w_pad = 4)
plt.show()


## save figures.
if svdat:
    fig_path    = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file1_I = "dataset_summary_bulk.pdf"
    fig1_I.savefig(fig_path + fig_file1_I, dpi = 600)
    print(fig_file1_I)


#%% prepare data for fig 2.

def get_pred_data(y_true, y_pred, models):
    ## build score matrix.
    y_true     = y_true.rename(
        index = "Response").replace(
        to_replace = {1: "R", 0: "NR"})
    
    score_data = pd.concat(
        [y_true, y_pred[models]], axis = 1).melt(
        id_vars = "Response", var_name = "model", value_name = "score")
    
    ## perform R vs. NR wilcoxon test.
    score_stat = score_data.groupby(
        by = "model", sort = False).apply(
        lambda df: pd.Series(mannwhitneyu(
            df.score[df.Response.eq("R")], df.score[df.Response.eq("NR")], 
            alternative = "greater", nan_policy = "omit"), 
            index = ["U1", "pval"]), 
        include_groups = False)
    
    score_stat["annot"] = score_stat.pval.map(
        lambda p: ("***" if (p <= 0.001) else "**" if (p <= 0.01) else 
                   "*" if (p <= 0.05) else "ns"))
    
    return score_data, score_stat


## get model orders: order by AUC & AP.
mdl_all   = perf_test_tn_sm.index.tolist()
mdl_ord   = perf_test_tn_sm.loc[
    cell_types].sort_values(
    by = ["AUC", "AP"], ascending = False).pipe(
    lambda df: df.index.tolist() + mdl_all[-2:])

mdl_names = [mdl.replace("_", "\n") for mdl in mdl_ord]


## get data for fig 2A-C.
fig_data2_I = [[ ] for k in range(len(ds_info))]                               # R vs. NR scores
fig_stat2_I = fig_data2_I.copy()                                               # R vs. NR p-values

fig_data2_I[0], fig_stat2_I[0] = get_pred_data(y_true = y_test_tn_sm, 
                                               y_pred = y_pred_tn_sm, 
                                               models = mdl_ord)

fig_data2_I[1], fig_stat2_I[1] = get_pred_data(y_true = y_test_tn_val_sm, 
                                               y_pred = y_pred_tn_val_sm, 
                                               models = mdl_ord)

fig_data2_I[2], fig_stat2_I[2] = get_pred_data(y_true = y_test_bn_sm, 
                                               y_pred = y_pred_bn_sm, 
                                               models = mdl_ord)

fig_data2_I = pd.concat(fig_data2_I, axis = 0, keys = ds_info.label.values)
fig_stat2_I = pd.concat(fig_stat2_I, axis = 0, keys = ds_info.label.values)


## get data for fig 2D-I.
fig_data2_II = pd.concat({                                                     # AUCs + APs
    met: pd.concat(
        [perf_test_tn_sm[met], perf_test_tn_val_sm[met], perf_test_bn_sm[met]], 
        axis = 1, keys = ds_info.label.values).loc[
        mdl_ord].set_axis(
        labels = mdl_names, axis = 0).reset_index(
        names = "model")    
    for met in ["AUC", "AP"]}, axis = 1)


## get data for fig 2J.
fig_data2_III = pd.concat(                                                     # DORs
    [perf_test_tn.DOR, perf_test_tn_val.DOR, perf_test_bn.DOR], 
    axis = 1, keys = ds_info.label.values).loc[
    mdl_ord[:-1]].rename(
    index = dict(zip(mdl_ord, mdl_names))).reset_index(
    names = "model").melt(
    id_vars = "model", var_name = "Dataset", value_name = "score")


#%% make fig 2-I.

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
fig_llocs2 = [0.95, 0.66, 0.35]
fig_ploc2  = 0.4
fig_ylim2  = 0.5

fig2_I, ax2_I = plt.subplots(figsize = (18, 8), nrows = 3, ncols = 1, 
                             sharex = True, sharey = True)
ax2_I = dict(zip(list("ABC"), ax2_I))

## make violins.
for k, (ds, (lbl, ax)) in enumerate(zip(ds_info.label, ax2_I.items())):
    ax = make_violinplot(data = fig_data2_I.loc[ds], x = "model", y = "score", 
                         hue = "Response", stats = fig_stat2_I.loc[ds], 
                         order = mdl_ord, hue_order = response_all, 
                         inner = "quart", split = True, dodge = True, 
                         statloc = fig_ploc2, statline = False, 
                         title = f"{ds}\n", legend_vert = True, 
                         legend_out = True, legend_title = "Response", ax = ax)
    ax.set_ylim([0 - fig_ylim2, 1 + fig_ylim2]);
    fig2_I.text(x = 0.0, y = fig_llocs2[k], s = lbl, **panel_fonts);           # add panel labels
    if lbl != "B":    ax.legend([ ], [ ]);
    if lbl == "C":
        ax.set_yticks(np.arange(0, 1.5, 0.5).round(1));
        ax.set_xticks(ticks = range(len(mdl_names)), labels = mdl_names, 
                      rotation = 45, ha = "right", va = "top", ma = "center", 
                      position = (0, 0.02), **label_fonts);
        fig2_I.supylabel("Prediction score", x = 0.01, y = 0.53, **label_fonts);

fig2_I.tight_layout(h_pad = 4, w_pad = 0)
plt.show()


## save figures.
if svdat:
    fig_path    = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file2_I = "all_predictions_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    fig2_I.savefig(fig_path + fig_file2_I, dpi = 600)
    print(fig_file2_I)


#%% make fig 2-II.

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
fig_llocs2 = [[0.02, 0.32, 0.62], [0.90, 0.45]]
fig_rlims2 = [[0.30, 0.95], [0.30, 0.95], [0.30, 0.85]]
fig_theta2 = RadarChart(num_vars = len(mdl_names), frame = "circle")
fig_base21 = [0.5] * len(fig_theta2)
fig_base22 = {ds: [info.R / info.n] * len(fig_theta2) 
              for ds, info in ds_info.set_index(keys = "label").iterrows()}
fig_ticks2 = pad_radar_ticks(ticks = fig_data2_II["AUC"].model, pads = [12, 4])

fig2_II, ax2_II = plt.subplots(figsize = (18, 12), nrows = 2, ncols = 3, 
                               subplot_kw = {"projection": "radar"})
ax2_II = dict(zip(list("DEFGHI"), ax2_II.ravel()))

## make radars.
## AUC radars.
for k, (lbl, ds) in enumerate(zip(list("DEF"), ds_info.label)):
    ax = ax2_II[lbl]
    ax = make_radar_lines(theta = fig_theta2, data = fig_data2_II["AUC"][ds], 
                          labels = fig_ticks2, color = colors[3], alpha = 0.4, 
                          ls = "-", lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_theta2, data = fig_base21, title = ds, 
                          color = colors[-3], alpha = 0.15, ls = ":", ms = 8, 
                          ax = ax)
    ax.set_rlim(fig_rlims2[k]);
    fig2_II.text(x = fig_llocs2[0][k], y = fig_llocs2[1][0], s = lbl, 
                 **panel_fonts);                                               # add panel labels

    if lbl == "F":
        ax.legend(labels = ["Cell type", "Random"], loc = (1.22, 0.4), 
                  title = "AUC", prop = legend_fonts["item"], 
                  title_fontproperties = legend_fonts["title"])

## AP radars.
for k, (lbl, ds) in enumerate(zip(list("GHI"), ds_info.label)):
    ax = ax2_II[lbl]
    ax = make_radar_lines(theta = fig_theta2, data = fig_data2_II["AP"][ds], 
                          labels = fig_ticks2, color = colors[3], alpha = 0.4, 
                          ls = "-", lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_theta2, data = fig_base22[ds], 
                          color = colors[-3], alpha = 0.15, ls = ":", ms = 8, 
                          ax = ax)
    ax.set_rlim(fig_rlims2[k]);
    fig2_II.text(x = fig_llocs2[0][k], y = fig_llocs2[1][1], s = lbl, 
                 **panel_fonts);                                               # add panel labels

    if lbl == "I":
        ax.legend(labels = ["Cell type", "Random"], loc = (1.22, 0.4), 
                  title = "AP", prop = legend_fonts["item"], 
                  title_fontproperties = legend_fonts["title"])

fig2_II.tight_layout(h_pad = -2, w_pad = 2)
plt.show()


## save figures.
if svdat:
    fig_path     = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file2_II = "all_aucs_aps_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    fig2_II.savefig(fig_path + fig_file2_II, dpi = 600)
    print(fig_file2_II)


#%% make fig 2-III.

filterwarnings(action = "ignore")                                              # suppress user-warnings from plotting
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

## barplots.
fig_colors2 = [colors[4], colors[3], colors[5]]

fig2_III, ax2_III = plt.subplots(figsize = (18, 5), nrows = 1, ncols = 1)

## make bars with dots to indicate continuation.
ax2_III = make_dot_barplot(data = fig_data2_III, x = "model", y = "score", 
                           hue = "Dataset", baseline = True, maxplt = 5.2, 
                           xrot = 35, ylabel = None, colors = fig_colors2, 
                           title = "Diagnostic odds ratio", ax = ax2_III)
fig2_III.text(x = -0.025, y = 0.9, s = "J", **panel_fonts);                    # add panel labels

fig2_III.tight_layout(h_pad = 0, w_pad = 0)
plt.show()


## save figures.
if svdat:
    fig_path      = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file2_III = "all_dors_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    fig2_III.savefig(fig_path + fig_file2_III, dpi = 600)
    print(fig_file2_III)


#%% prepare data for fig 3.

## cell type shorthands.
ctp_abbv = {"B-cells"           : "B", 
            "CAFs"              : "CAF", 
            "Cancer_Epithelial" : "CE", 
            "Endothelial"       : "ENDO", 
            "Myeloid"           : "MYL", 
            "Normal_Epithelial" : "NE", 
            "PVL"               : "PVL", 
            "Plasmablasts"      : "PB", 
            "T-cells"           : "T", 
            "Bulk"              : "Bulk"}


def get_ens_data(perfs, n_ens, score = "AUC", ds = None):
    perfs_ens  = perfs[
        perfs.index.map(lambda x: x.count("+") == n_ens - 1)].sort_values(
        by = ["AUC", "AP"], ascending = [False, False])
    
    perfs_data = pd.concat(
        [perfs_ens, perfs.loc[["Bulk"]]], axis = 0).rename(
        index = lambda mdl: " + ".join([ctp_abbv[x] for x in mdl.split("+")]))[
        [score]]
    if ds is not None:
        perfs_data.columns = [ds]
    
    return perfs_data
    

## get data for fig 3A-F.
n_top     = 10                                                                 # keep only top ensembles
n_ctp_ens = [2, 3]                                                             # #cells in mult-cell-ensembles         

fig_data3_I, fig_data3_II = [[ ], [ ]], [ ]                                    # AUCs + APs, DORs : [two-cell-ensemble, three-cell-ensemble]
for n_ens in n_ctp_ens:
    for scr in ["AUC", "AP", "DOR"]:
        ## gather ensemble scores.
        dat = pd.concat([
            get_ens_data(perf, n_ens = n_ens, score = scr, 
                         ds = ds_info.label[k+1]) 
            for k, perf in enumerate([perf_test_tn_val, perf_test_bn])], 
            axis = 1)
        
        ## sort cell-type-ensemble scores by AUCs and save data.
        match scr:
            case "AUC":                                                        # panel A + D
                mdls = dat.drop(
                    index = "Bulk").sort_values(
                    by = dat.columns.tolist(), ascending = False)[
                    :n_top].index.tolist()
                dat  = dat.loc[mdls + ["Bulk"]].reset_index(names = "model")
                
                fig_data3_I[n_ens - 2].append( dat )
            case "AP":                                                         # panel B + E
                mdls = fig_data3_I[n_ens - 2][0].model
                dat  = dat.loc[mdls].reset_index(names = "model")
                fig_data3_I[n_ens - 2].append( dat )
            case "DOR":                                                        # panel C + F
                mdls = fig_data3_I[n_ens - 2][0].model
                dat  = dat.loc[
                    mdls].reset_index(
                    names = "model").melt(
                    id_vars = "model", var_name = "Dataset", 
                    value_name = "score")
                fig_data3_II.append( dat )

del n_ens, scr, mdls, dat                                                      # reduce clutter 

fig_data3_I  = pd.concat([pd.concat(dat, axis = 1, keys = ["AUC", "AP"]) 
                          for dat in fig_data3_I], axis = 0, keys = n_ctp_ens)
fig_data3_II = pd.concat(fig_data3_II, axis = 0, keys = n_ctp_ens)
        

#%% make fig 3-I.

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

## radar charts.
fig_llocs3   = [[0.05, 0.52], [0.98, 0.52]]
fig_theta3   = RadarChart(num_vars = n_top + 1, frame = "circle")
fig_base31   = [0.5] * len(fig_theta3)
fig_base32   = {ds: [info.R / info.n] * len(fig_theta3) 
                for ds, info in ds_info.set_index(keys = "label")[1:].iterrows()}
fig_ticks3   = [
    pad_radar_ticks(ticks = fig_data3_I.loc[2].AUC.model, pads = [4, 12]), 
    pad_radar_ticks(ticks = fig_data3_I.loc[3].AUC.model, pads = [18, 20])]
fig_lbls3    = ["Two-cell-type ensemble", "Three-cell-type ensemble"]
fig_colors3  = [colors[3], colors[5], colors[-3], colors[-2]]
fig_lgndprop = dict(loc = (0.20, -0.45), title = "Dataset", 
                    prop = legend_fonts["item"], 
                    title_fontproperties = legend_fonts["title"])

fig3_I, ax3_I = plt.subplots(figsize = (14, 12), nrows = 2, ncols = 2, 
                             subplot_kw = {"projection": "radar"})
ax3_I = dict(zip(list("ABDE"), ax3_I.ravel()))

## make radars.
## AUC radars.
for n_ens, lbl in zip(n_ctp_ens, list("AD")):
    ax, dat = ax3_I[lbl], fig_data3_I.loc[n_ens]["AUC"]
    for j, ds in enumerate(ds_info.label[1:]):
        ax = make_radar_lines(theta = fig_theta3, data = dat[ds], 
                              labels = fig_ticks3[n_ens - 2], 
                              color = fig_colors3[j], alpha = 0.4, ls = "-", 
                              lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_theta3, data = fig_base31, 
                          title = "AUC" if n_ens == 2 else None, 
                          color = fig_colors3[-2], alpha = 0.15, 
                          ls = ":", ms = 8, ax = ax)
    ax.set_rlim([0.30, 0.96])
    ax.set_ylabel(fig_lbls3[n_ens - 2] + "\n" * 5, y = 0.55, labelpad = 12, 
                  **legend_fonts["title"]);
    fig3_I.text(x = fig_llocs3[0][0], y = fig_llocs3[1][n_ens - 2], s = lbl,   # add panel labels
                **panel_fonts);
    
    if n_ens == 3:
        ax.legend(labels = ds_info.label.tolist()[1:] + ["Random"], 
                  loc = (0.20, -0.45), title = "Dataset", 
                  prop = legend_fonts["item"], 
                  title_fontproperties = legend_fonts["title"]);

## AP radars.
for n_ens, lbl in zip(n_ctp_ens, list("BE")):
    ax, dat = ax3_I[lbl], fig_data3_I.loc[n_ens]["AP"]
    for j, ds in enumerate(ds_info.label[1:]):
        ax = make_radar_lines(theta = fig_theta3, data = dat[ds], 
                              labels = fig_ticks3[n_ens - 2], 
                              color = fig_colors3[j], alpha = 0.4, ls = "-", 
                              lw = 2, ms = 8, ax = ax)
        ax = make_radar_lines(theta = fig_theta3, data = fig_base32[ds], 
                              title = "AP" if n_ens == 2 else None, 
                              color = fig_colors3[-(j+1)], alpha = 0.15, 
                              ls = ":", ms = 8, ax = ax)
    ax.set_rlim([0.10, 0.85])
    fig3_I.text(x = fig_llocs3[0][1], y = fig_llocs3[1][n_ens - 2], s = lbl,   # add panel labels
                **panel_fonts);
    
    if n_ens == 3:
        ax.legend(labels = ds_info.label.tolist()[1:] + \
                  ds_info.Dataset[1:].map(lambda x: f"Random ({x})").tolist(), 
                  loc = (0.20, -0.45), title = "Dataset", 
                  prop = legend_fonts["item"], 
                  title_fontproperties = legend_fonts["title"]);

fig3_I.tight_layout(h_pad = 6, w_pad = 6)
plt.show()
    

## save figures.
if svdat:
    fig_path    = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file3_I = "top_ensemble_aucs_aps_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    fig3_I.savefig(fig_path + fig_file3_I, dpi = 600)
    print(fig_file3_I)


#%% make fig 3-II.

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

## barplots.
fig_llocs3  = [0.94, 0.56]
fig_colors3 = [colors[3], colors[5]]

fig3_II, ax3_II = plt.subplots(figsize = (8, 12), nrows = 2, ncols = 1)
ax3_II = dict(zip(list("CF"), ax3_II))

## make bars with dots to indicate continuation.
for n_ens, (lbl, ax) in zip(n_ctp_ens, ax3_II.items()):
    ax = make_dot_barplot(data = fig_data3_II.loc[n_ens], x = "model", 
                          y = "score", hue = "Dataset", baseline = True, 
                          maxplt = 5.2, xrot = 35, colors = fig_colors3, 
                          ylabel = None, ax = ax)
    
    if lbl == "C":
        ax.set_title("Diagnostic odds ratio\n", pad = 12, y = 1.2, 
                     **legend_fonts["title"]);
        ax.legend([ ], [ ]);
    else:
        ax.get_legend().set(bbox_to_anchor = (0.30, -0.85));
    
    fig3_II.text(x = 0.01, y = fig_llocs3[n_ens - 2], s = lbl, **panel_fonts); # add panel labels

fig3_II.tight_layout(h_pad = 8, w_pad = 0)
plt.show()


## save figures.
if svdat:
    fig_path     = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file3_II = "top_ensemble_dors_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    fig3_II.savefig(fig_path + fig_file3_II, dpi = 600)
    print(fig_file3_II)


#%% make supplementary figs.
#%% prepare data for supp fig 2.

## prepare LOO results.
## prediction scores.
y_pred_tn_loo_sm     = y_pred_tn_loo.loc[samples_tn_sm].copy()
y_pred_tn_val_loo_sm = y_pred_tn_val_loo.loc[samples_tn_val_sm].copy()
y_pred_bn_loo_sm     = y_pred_bn_loo.copy()


## performance scores.
perf_test_tn_loo_sm     = pd.DataFrame({
    mdl: classifier_performance(y_test_tn_sm, y_pred) 
    for mdl, y_pred in y_pred_tn_loo_sm.items()}).T

perf_test_tn_val_loo_sm = pd.DataFrame({
    mdl: classifier_performance(y_test_tn_val_sm, y_pred) 
    for mdl, y_pred in y_pred_tn_val_loo_sm.items()}).T

perf_test_bn_loo_sm     = pd.DataFrame({
    mdl: classifier_performance(y_test_bn_sm, y_pred) 
    for mdl, y_pred in y_pred_bn_loo_sm.dropna(axis = 1).items()}).T
perf_test_bn_loo_sm.loc["Sammut et al."] = nan


print(f"""
prepared prediction scores & performance scores!
dataset info:\n{ds_info.set_index(keys = "Dataset")}\n
performance snapshot: 
{pd.concat([perf_test_tn_loo_sm.AUC, perf_test_tn_val_loo_sm.AUC, perf_test_bn_loo_sm.AUC], 
           axis = 1, keys = ds_info.Dataset).loc[
           cell_types + ["Bulk"]].round(4)}
""")


## prepare data for figures.
mdl_ord    = fig_data2_I.loc[ds_info.label[0]].model.unique().tolist()[:-1]
mdl_names  = [mdl.replace("_", "\n") for mdl in mdl_ord]

fig_dataS2 = pd.concat({
    met: pd.concat(
        [perf_test_tn_loo_sm[met], perf_test_tn_val_loo_sm[met], 
         perf_test_bn_loo_sm[met]], 
        axis = 1, keys = ds_info.label.values).loc[
        mdl_ord].set_axis(
        labels = mdl_names, axis = 0).reset_index(
        names = "model")
    for met in ["AUC", "AP"]}, axis = 1)


#%% make supp fig 2.

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
title_fonts  = {"weight": "bold", "size": 18, "color": "#000000"}
label_fonts  = {"weight": "regular", "size": 14, "color": "#000000"}
legend_fonts = {"item" : {"size": 12, "weight": "regular"}, 
                "title": {"size": 16, "weight": "bold"}}

## radar charts.
fig_llocsS2 = [0.95, 0.45]
fig_limsS2  = [[0.30, 0.95], [0.10, 0.85]]
fig_thetaS2 = RadarChart(num_vars = len(mdl_names), frame = "circle")
fig_baseS21 = [0.5] * len(fig_thetaS2)
fig_baseS22 = {ds: [info.R / info.n] * len(fig_thetaS2) 
               for ds, info in ds_info.set_index(keys = "label").iterrows()}
fig_ticksS2 = pad_radar_ticks(ticks = fig_dataS2.AUC.model, pads = [12, 4])
fig_ticksS2[5] = fig_ticksS2[5].strip()

figS2, axS2 = plt.subplot_mosaic(
    mosaic = [["A1", "A2", "A3"], ["B1", "B2", "B3"]], figsize = (18, 12), 
    subplot_kw = {"projection": "radar"})

## make radars.
for k, ds in enumerate(ds_info.label, start = 1):
    ## AUC radars.
    ax = axS2[f"A{k}"]
    ax = make_radar_lines(theta = fig_thetaS2, data = fig_dataS2["AUC"][ds], 
                          labels = fig_ticksS2, color = colors[3], alpha = 0.4, 
                          ls = "-", lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_thetaS2, data = fig_baseS21, title = ds, 
                          color = colors[-3], alpha = 0.15, ls = ":", ms = 8, 
                          ax = ax)
    ax.set_rlim(fig_limsS2[0])
    if k == 1:
        ax.set_ylabel("AUC" + "\n" * 4, labelpad = 12, y = 0.55, 
                      **legend_fonts["title"]);
        figS2.text(x = 0.0, y = fig_llocsS2[0], s = "A", **panel_fonts);       # add panel labels
    
    ## AP radars.
    ax = axS2[f"B{k}"]
    ax = make_radar_lines(theta = fig_thetaS2, data = fig_dataS2["AP"][ds], 
                          labels = fig_ticksS2, color = colors[3], alpha = 0.4, 
                          ls = "-", lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_thetaS2, data = fig_baseS22[ds], 
                          color = colors[-3], alpha = 0.15, ls = ":", ms = 8, 
                          ax = ax)
    ax.set_rlim(fig_limsS2[1])
    if k == 1:
        ax.set_ylabel("AP" + "\n" * 4, labelpad = 12, y = 0.55, 
                      **legend_fonts["title"]);
        figS2.text(x = 0.0, y = fig_llocsS2[1], s = "B", **panel_fonts);       # add panel labels
    elif k == len(ds_info):
        ax.legend(labels = ["Cell type", "Random"], loc = (1.06, 0.9), 
                  title = "Performance", prop = legend_fonts["item"], 
                  title_fontproperties = legend_fonts["title"]);


figS2.tight_layout(h_pad = 1, w_pad = 2)
plt.show()


## save figures.
if svdat:
    fig_path   = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS2 = "all_aucs_aps_chemo_th0.99_ENS2_25features_LeaveOneOutCV.pdf"
    figS2.savefig(fig_path + fig_fileS2, dpi = 600)


#%% prepare data for supp fig 3.

## prepare results for weighted gene expression.
## prediction scores.
y_pred_tn_wf_sm     = y_pred_tn_wf.loc[samples_tn_sm].copy()
y_pred_tn_val_wf_sm = y_pred_tn_val_wf.loc[samples_tn_val_sm].copy()
y_pred_bn_wf_sm     = y_pred_bn_wf.copy()


## performance scores.
perf_test_tn_wf_sm     = pd.DataFrame({
    mdl: classifier_performance(y_test_tn_sm, y_pred) 
    for mdl, y_pred in y_pred_tn_wf_sm.items()}).T

perf_test_tn_val_wf_sm = pd.DataFrame({
    mdl: classifier_performance(y_test_tn_val_sm, y_pred) 
    for mdl, y_pred in y_pred_tn_val_wf_sm.items()}).T

perf_test_bn_wf_sm     = pd.DataFrame({
    mdl: classifier_performance(y_test_bn_sm, y_pred) 
    for mdl, y_pred in y_pred_bn_wf_sm.dropna(axis = 1).items()}).T
perf_test_bn_wf_sm.loc["Sammut et al."] = nan


print(f"""
prepared prediction scores & performance scores!
dataset info:\n{ds_info.set_index(keys = "Dataset")}\n
performance snapshot: 
{pd.concat([perf_test_tn_wf_sm.AUC, perf_test_tn_val_wf_sm.AUC, perf_test_bn_wf_sm.AUC], 
           axis = 1, keys = ds_info.Dataset).loc[
           cell_types + ["Bulk"]].round(4)}
""")


## prepare data for figures.
mdl_ord   = fig_data2_I.loc[ds_info.label[0]].model.unique().tolist()[:-1]
mdl_names = [mdl.replace("_", "\n") for mdl in mdl_ord]

fig_dataS3 = pd.concat({
    met: pd.concat(
        [perf_test_tn_wf_sm[met], perf_test_tn_val_wf_sm[met], 
         perf_test_bn_wf_sm[met]], 
        axis = 1, keys = ds_info.label.values).loc[
        mdl_ord].set_axis(
        labels = mdl_names, axis = 0).reset_index(
        names = "model")
    for met in ["AUC", "AP"]}, axis = 1)


#%% make supp fig 3.

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
title_fonts  = {"weight": "bold", "size": 18, "color": "#000000"}
label_fonts  = {"weight": "regular", "size": 14, "color": "#000000"}
legend_fonts = {"item" : {"size": 12, "weight": "regular"}, 
                "title": {"size": 16, "weight": "bold"}}

## radar charts.
fig_llocsS3 = [0.95, 0.45]
fig_limsS3  = [[0.30, 0.95], [0.10, 0.85]]
fig_thetaS3 = RadarChart(num_vars = len(mdl_names), frame = "circle")
fig_baseS31 = [0.5] * len(fig_thetaS3)
fig_baseS32 = {ds: [info.R / info.n] * len(fig_thetaS3) 
               for ds, info in ds_info.set_index(keys = "label").iterrows()}
fig_ticksS3 = pad_radar_ticks(ticks = fig_dataS3.AUC.model, pads = [12, 4])
fig_ticksS3[5] = fig_ticksS3[5].strip()

figS3, axS3 = plt.subplot_mosaic(
    mosaic = [["A1", "A2", "A3"], ["B1", "B2", "B3"]], figsize = (18, 12), 
    subplot_kw = {"projection": "radar"})

## make radars.
for k, ds in enumerate(ds_info.label, start = 1):
    ## AUC radars.
    ax = axS3[f"A{k}"]
    ax = make_radar_lines(theta = fig_thetaS3, data = fig_dataS3["AUC"][ds], 
                          labels = fig_ticksS3, color = colors[3], alpha = 0.4, 
                          ls = "-", lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_thetaS3, data = fig_baseS31, title = ds, 
                          color = colors[-3], alpha = 0.15, ls = ":", ms = 8, 
                          ax = ax)
    ax.set_rlim(fig_limsS3[0])
    if k == 1:
        ax.set_ylabel("AUC" + "\n" * 4, labelpad = 12, y = 0.55, 
                      **legend_fonts["title"]);
        figS3.text(x = 0.0, y = fig_llocsS3[0], s = "A", **panel_fonts);       # add panel labels
    
    ## AP radars.
    ax = axS3[f"B{k}"]
    ax = make_radar_lines(theta = fig_thetaS3, data = fig_dataS3["AP"][ds], 
                          labels = fig_ticksS3, color = colors[3], alpha = 0.4, 
                          ls = "-", lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_thetaS3, data = fig_baseS32[ds], 
                          color = colors[-3], alpha = 0.15, ls = ":", ms = 8, 
                          ax = ax)
    ax.set_rlim(fig_limsS3[1])
    if k == 1:
        ax.set_ylabel("AP" + "\n" * 4, labelpad = 12, y = 0.55, 
                      **legend_fonts["title"]);
        figS3.text(x = 0.0, y = fig_llocsS3[1], s = "B", **panel_fonts);       # add panel labels
    elif k == len(ds_info):
        ax.legend(labels = ["Cell type", "Random"], loc = (1.06, 0.9), 
                  title = "Performance", prop = legend_fonts["item"], 
                  title_fontproperties = legend_fonts["title"]);


figS3.tight_layout(h_pad = 1, w_pad = 2)
plt.show()


## save figures.
if svdat:
    fig_path   = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS3 = "all_aucs_aps_weighted_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS3.savefig(fig_path + fig_fileS3, dpi = 600)


#%% prepare data for supp fig 4.

subtypes_all = ["ER+/HER2-", "TNBC"]

## prepare data for division by subtype.
def get_samples_by_subtype(clin_data, col = "ER.status"):
    return {sb: clin_data[clin_data[col].eq(val)].index.tolist() 
            for sb, val in zip(subtypes_all, ["POS", "NEG"])}

def get_perf_ctp(y_true, y_pred):
    return y_pred.apply(
        lambda y_pc: pd.Series(classifier_performance(y_true, y_pc))).T


## get data by subtype.
clin_data_tn_sm     = clin_data_tn.loc[samples_tn_sm].copy()

smpl_subtype_tn     = get_samples_by_subtype(clin_data_tn_sm)
smpl_subtype_tn_val = get_samples_by_subtype(clin_data_tn_val)


## comput performance by subtype.
mdl_base  = ["Bulk", "Sammut et al."]
mdl_list  = cell_types + mdl_base
mdl_list2 = np.append(np.setdiff1d(
    y_pred_tn_val_sm.columns, cell_types + mdl_base), mdl_base).tolist()

perf_test_subtype_tn      = pd.concat({
    sb: get_perf_ctp(y_test_tn_sm.loc[smpl], 
                     y_pred_tn_sm.loc[smpl, mdl_list]) 
    for sb, smpl in smpl_subtype_tn.items()}, axis = 1)

perf_test_subtype_tn_val  = pd.concat({
    sb: get_perf_ctp(y_test_tn_val.loc[smpl], 
                     y_pred_tn_val_sm.loc[smpl, mdl_list]) 
    for sb, smpl in smpl_subtype_tn_val.items()}, axis = 1)

perf_test_subtype_tn_val2 = pd.concat({
    sb: get_perf_ctp(y_test_tn_val.loc[smpl], 
                     y_pred_tn_val_sm.loc[smpl, mdl_list2]) 
    for sb, smpl in smpl_subtype_tn_val.items()}, axis = 1)


## prepare data for figure.
fig_dataS4, fig_ticksS4 = [ ], [ ]

## individual cell types.
for perf_ in [perf_test_subtype_tn, perf_test_subtype_tn_val]:
    for sb_ in subtypes_all:
        ord_ = perf_[
            sb_].drop(
            index = mdl_base).sort_values(
            by = ["AUC", "AP"], ascending = False).pipe(
            lambda df: df.index.tolist() + mdl_base)
        
        dat_ = perf_[
            sb_].loc[
            ord_].reset_index(
            names = "model").melt(
            id_vars = "model", var_name = "metric", value_name = "score")
        
        fig_ticksS4.append( [mdl.replace("_", "\n") for mdl in ord_] )
        fig_dataS4.append( dat_ )
    
del sb_, perf_, ord_, dat_                                                     # reduce clutter


## cell type ensembles.
## pick top ensembles first.
n_top      = 5
mdl_combos = [
    perf_test_tn_val.pipe(
        lambda df: df[
            df.index.map(lambda x: x.count("+") == n_ens - 1)]).sort_values(
        by = ["AUC", "AP"], ascending = False).index.tolist()[:n_top] 
    for n_ens in n_ctp_ens]
mdl_combos = {
    mdls: " + ".join([ctp_abbv[mdl] for mdl in mdls.split("+")]) 
    for mdls in mdl_combos[0] + mdl_combos[1]}

## order by subtype-specific performance.
for sb_ in subtypes_all:
    ord_ = perf_test_subtype_tn_val2[
        sb_].loc[
        list(mdl_combos)].sort_values(
        by = ["AUC", "AP"], ascending = False).pipe(
        lambda df: df.index.tolist() + mdl_base)
    
    dat_ = perf_test_subtype_tn_val2[
        sb_].loc[
        ord_].rename(
        index = mdl_combos).reset_index(
        names = "model").melt(
        id_vars = "model", var_name = "metric", value_name = "score")
    
    fig_ticksS4.append( [" + ".join([ctp_abbv[mdl] for mdl in mdls.split("+")]) 
                         for mdls in ord_[:-2]] + ord_[-2:] )
    fig_dataS4.append( dat_ )

del sb_, ord_, dat_                                                            # reduce clutter


#%% generate supp fig 4.

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

fontdict     = {"label": dict(fontsize = 56, fontweight = "regular"), 
                "title": dict(fontsize = 60, fontweight = "semibold"), 
                "super": dict(fontsize = 64, fontweight = "bold"),
                "plabel": dict(fontsize = 120, fontweight = "bold")}

## barplots.
fig_llocsS4  = [[0.02, 0.48], [1.00, 0.66, 0.33]]
fig_colorsS4 = [colors[3], colors[4]]
fig_ttlsS4   = [
    f"TransNEO: {sb} (n = {len(smpl)})\n" 
    for sb, smpl in smpl_subtype_tn.items()] + [
    f"ARTemis + PBCP: {sb} (n = {len(smpl)})\n" 
    for sb, smpl in smpl_subtype_tn_val.items()] + [None] * 2

figS4, axS4 = plt.subplots(figsize = (80, 48), nrows = 3, ncols = 2)
axS4 = dict(zip(list("ABCDEF"), axS4.ravel()))

## make bars.
for k, (lbl, ax) in enumerate(axS4.items()):
    ax = make_barplot2(data = fig_dataS4[k], x = "model", y = "score", 
                       hue = "metric", width = 0.5, colors = fig_colorsS4, 
                       title = fig_ttlsS4[k], legend = (lbl == "D"), 
                       xlabels = fig_ticksS4[k], xrot = 35, 
                       bar_label_align = True, ax = ax, fontdict = fontdict)
    ax.set_ylim([-0.04, 1.04])
    figS4.text(x = fig_llocsS4[0][k % 2], 
               y = fig_llocsS4[1][int(k > 1) + int(k > 3)], s = lbl, 
               **fontdict["plabel"]);                                          # add panel labels
    if lbl == "D":
        ax.get_legend().set(bbox_to_anchor = (1.0, 0.3), title = "Performance");

figS4.tight_layout(h_pad = 8, w_pad = 12)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS4 = "all_performance_subtype_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS4.savefig(fig_path + fig_fileS4, dpi = 600)


#%% prepare data for supp fig 6.
## prediction scores & performances for T-cell subtypes.

y_pred_tn_sm2     = y_pred_tn2.loc[samples_tn_sm].copy()
y_pred_tn_val_sm2 = y_pred_tn_val2.loc[samples_tn_val_sm].copy()
y_pred_bn_sm2     = y_pred_bn2.copy()

perf_test_tn_sm2     = pd.DataFrame({
    mdl: classifier_performance(y_test_tn_sm, y_pred) 
    for mdl, y_pred in y_pred_tn_sm2.items()}).T

perf_test_tn_val_sm2 = pd.DataFrame({
    mdl: classifier_performance(y_test_tn_val_sm, y_pred) 
    for mdl, y_pred in y_pred_tn_val_sm2.items()}).T

perf_test_bn_sm2     = pd.DataFrame({
    mdl: classifier_performance(y_test_bn_sm, y_pred) 
    for mdl, y_pred in y_pred_bn_sm2.items()}).T


print(f"""
prepared prediction scores & performance scores!
dataset info:\n{ds_info.set_index(keys = "Dataset")}\n
performance snapshot: 
{pd.concat([perf_test_tn_sm2.AUC, perf_test_tn_val_sm2.AUC, perf_test_bn_sm2.AUC], 
           axis = 1, keys = ds_info.Dataset).loc[
           cell_types2 + ["Bulk"]].round(4)}
""")


## get model orders.
mdl_ord   = cell_types2 + ["Bulk"]
mdl_names = [mdl.replace("_", " ").replace("+", "$^+$") 
             for mdl in mdl_ord]


## get data for supp fig 5A-C.
fig_dataS6_I = [[ ] for k in range(len(ds_info))]                               # R vs. NR scores
fig_statS6_I = fig_dataS6_I.copy()                                              # R vs. NR p-values
fig_dataS6_I[0], fig_statS6_I[0] = get_pred_data(y_true = y_test_tn_sm, 
                                                 y_pred = y_pred_tn_sm2, 
                                                 models = mdl_ord)

fig_dataS6_I[1], fig_statS6_I[1] = get_pred_data(y_true = y_test_tn_val_sm, 
                                                 y_pred = y_pred_tn_val_sm2, 
                                                 models = mdl_ord)

fig_dataS6_I[2], fig_statS6_I[2] = get_pred_data(y_true = y_test_bn_sm, 
                                                 y_pred = y_pred_bn_sm2, 
                                                 models = mdl_ord)


## get data for supp fig 5D-F.
fig_dataS6_II = pd.concat([
    perf.reset_index(
        names = "model").melt(
        id_vars = "model", var_name = "metric", value_name = "score")    
    for perf in [perf_test_tn_sm2, perf_test_tn_val_sm2, perf_test_bn_sm2]], 
    axis = 1, keys = ds_info.label.values)


#%% make supp fig 6-I.

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

colors   = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
            "#FFC72C", "#708090", "#A9A9A9", "#000000"]

fontdict = {"label": dict(fontsize = 12, fontweight = "regular"), 
            "title": dict(fontsize = 16, fontweight = "semibold"), 
            "super": dict(fontsize = 20, fontweight = "bold"),
            "plabel": dict(fontsize = 36, fontweight = "bold")}

## violin plots + barplots.
fig_llocsS6  = [[0.01, 0.315, 0.615], [0.97, 0.47]]
fig_plocS6   = 0.45
fig_ylimS6   = [0.5, 0.05]
fig_colorsS6 = [[colors[k] for k in [0, 1, -1]], 
                [colors[k] for k in [3, 4, -1]]]

figS6, axS6 = plt.subplots(figsize = (18, 7), nrows = 2, ncols = 3, 
                             sharex = False, sharey = False)
axS6 = dict(zip(list("ABCDEF"), axS6.ravel()))

## make violins.
for k, (ds, lbl) in enumerate(zip(ds_info.label, list("ABC"))):
    ax = axS6[lbl]
    ax = make_violinplot(data = fig_dataS6_I[k], x = "model", y = "score", 
                         hue = "Response", stats = fig_statS6_I[k], 
                         order = mdl_ord, hue_order = ["R", "NR"], 
                         inner = "quart", split = True, dodge = True, 
                         colors = fig_colorsS6[0], statloc = fig_plocS6, 
                         statline = False, title = ds, legend_vert = True, 
                         legend_out = True, legend_title = "Response", ax = ax)
    
    ## format ticks & labels.
    ax.set_ylim([0 - fig_ylimS6[0], 1 + fig_ylimS6[0]]);
    ax.set_xticklabels([""] * len(mdl_names));
    match k:
        case 0:
            ax.legend([ ], [ ]);
            ax.set_ylabel("Prediction score", x = -0.02, **fontdict["label"]);
            ax.set_yticks(np.arange(0, 1.25, 0.25));
        case 1:
            ax.legend([ ], [ ]);
            ax.set_yticks(ticks = np.arange(0, 1.25, 0.25), labels = [""] * 5);
        case _:
            ax.get_legend().set(bbox_to_anchor = (1.06, 0.4));
            ax.set_yticks(ticks = np.arange(0, 1.25, 0.25), labels = [""] * 5);
    figS6.text(x = fig_llocsS6[0][k], y = fig_llocsS6[1][0], s = lbl, 
               **fontdict["plabel"]);                                          # add panel labels

## make bars.
for k, (ds, lbl) in enumerate(zip(ds_info.label, list("DEF"))):
    ax = axS6[lbl]
    ax = make_barplot2(data = fig_dataS6_II[ds], x = "model", y = "score", 
                       hue = "metric", width = 0.6, colors = fig_colorsS6[1], 
                       lw = 2, xlabels = mdl_names, bar_label_align = False, 
                       title = None, legend_title = "Performance", 
                       fontdict = fontdict, ax = ax)
    
    ## format ticks & labels.
    ax.set_ylim([0 - fig_ylimS6[1], 1 + fig_ylimS6[1]]);
    ax.set_xticks(ticks = range(len(mdl_names)), labels = mdl_names, 
                  **fontdict["label"]);
    match k:
        case 0:
            ax.legend([ ], [ ]);
            ax.set_ylabel("Model performance", x = -0.02, **fontdict["label"]);
            ax.set_yticks(np.arange(0, 1.25, 0.25));
            ax.yaxis.set_major_formatter("{x:0.2f}");
        case 1:
            ax.legend([ ], [ ]);
            ax.set_yticks(ticks = np.arange(0, 1.25, 0.25), labels = [""] * 5);
        case _:
            ax.get_legend().set(bbox_to_anchor = (1.02, 0.4), 
                                frame_on = False)
            ax.set_yticks(ticks = np.arange(0, 1.25, 0.25), labels = [""] * 5);
        
    figS6.text(x = fig_llocsS6[0][k], y = fig_llocsS6[1][1], s = lbl,          # add panel labels
               **fontdict["plabel"]);


figS6.tight_layout(h_pad = 4, w_pad = 4)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS6 = "all_predictions_aucs_aps_v2_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS6.savefig(fig_path + fig_fileS6, dpi = 600)

