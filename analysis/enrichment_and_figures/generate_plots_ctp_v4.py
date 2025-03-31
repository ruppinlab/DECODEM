#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 18:13:23 2024

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
# from miscellaneous import date_time, tic, write_xlsx
from math import nan, ceil, floor
from scipy.stats import mannwhitneyu
from itertools import product
from collections import Counter
# from sklearn.preprocessing import MinMaxScaler
from _functions import (classifier_performance, binary_performance, 
                        make_barplot1, make_barplot2, make_barplot3, 
                        make_boxplot1, make_boxplot2, make_performance_plot)


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
               "antialiased": False, "width": donut_size}
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


#%% read data.

data_path = ["../../data/TransNEO/transneo_analysis/mdl_data/", 
             "../../data/TransNEO/use_data/", 
             "../../data/TransNEO/TransNEO_SammutShare/", 
             "../../data/BrighTNess/"]

data_file = ["transneo_predictions_chemo_th0.99_ENS2_25features_5foldCV_20Mar2023.pkl", 
             "tn_valid_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl", 
             "brightness_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl", 
             "transneo-diagnosis-MLscores.tsv", 
             "TransNEO_SupplementaryTablesAll.xlsx", 
             "transneo-diagnosis-clinical-features.xlsx", 
             "GSE164458_BrighTNess_clinical_info_SRD_04Oct2022.xlsx"]


## model prediction scores.
y_test_tn, y_pred_tn, th_test_tn, perf_test_tn = read_pkl_data(
    data_path[0] + data_file[0])

y_test_tn_val, y_pred_tn_val, th_test_tn_val, perf_test_tn_val = read_pkl_data(
    data_path[0] + data_file[1])

y_test_bn, y_pred_bn, th_test_bn, perf_test_bn = read_pkl_data(
    data_path[0] + data_file[2])


## clinical info.
clin_info_tn     = pd.read_excel(
    data_path[1] + data_file[4], sheet_name = "Supplementary Table 1", 
    skiprows = 1, header = 0, index_col = 0)

clin_info_tn_val = pd.read_excel(
    data_path[1] + data_file[4], sheet_name = "Supplementary Table 5", 
    skiprows = 1, header = 0, index_col = 0)

clin_info_bn = pd.read_excel(
    data_path[3] + data_file[6], sheet_name = "samples", 
    header = 0, index_col = 0)

samples_sammut_tn     = clin_info_tn.index.tolist()
samples_sammut_tn_val = clin_info_tn_val.index.tolist()

## clinical data for available samples.
clin_data_tn     = clin_info_tn.loc[y_test_tn.index].copy()
clin_data_tn_val = clin_info_tn_val.loc[y_test_tn_val.index].copy()
clin_data_bn     = clin_info_bn.loc[y_test_bn.index].copy()


# ## clinical info from Sammut et al.
# clin_info_tn_sammut = pd.read_excel(
#     data_path[2] + data_file[5], sheet_name = "training", 
#     header = 0, index_col = 0)

# clin_info_tn_val_sammut = pd.read_excel(
#     data_path[2] + data_file[5], sheet_name = "validation", 
#     header = 0, index_col = 0)


## sammut et al. scores.
y_pred_sammut_all    = pd.read_table(data_path[1] + data_file[3], sep = "\t", 
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


#%% prepare prediction scores & model performance scores.

cell_types = sorted(np.setdiff1d(y_pred_tn.columns, "Bulk"))

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
perf_test_bn.loc["Sammut et al."] = nan


print(f"""
prepared prediction scores & performance scores!
dataset info:\n{ds_info.set_index(keys = "Dataset")}\n
performance snapshot: 
{pd.concat([perf_test_tn_sm.AUC, perf_test_tn_val_sm.AUC, perf_test_bn_sm.AUC], 
           axis = 1, keys = ds_info.Dataset).loc[
           cell_types + ["Bulk","Sammut et al."]].round(4)}
""")


#%% prepare data for fig 2.

def get_pred_data(y_true, y_pred, models):
    ## build score dataframe.
    y_true     = y_true.rename(
        index = "Response").replace(
        to_replace = {1: "R", 0: "NR"})
    
    score_data = pd.concat(
        [y_true, y_pred[models]], axis = 1).melt(
        id_vars = "Response", var_name = "model", value_name = "score")
    
    ## perform R vs. NR wilcoxon test.
    # score_stat = pd.DataFrame({
    #     mdl: mannwhitneyu(*score_data.groupby(
    #         by = "Response").apply(
    #         lambda df: df[df.model.eq(mdl)]["score"].tolist()).sort_index(
    #         ascending = False), alternative = "greater") 
    #     for mdl in models}, index = ["U1", "pval"]).T
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


## get model orders.
mdl_all   = perf_test_tn_sm.index.tolist()
mdl_ord   = perf_test_tn_sm.loc[
    cell_types].sort_values(
    by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + mdl_all[-2:]

mdl_names = [mdl.replace("_", "\n") for mdl in mdl_ord]


## get data for fig. 2A-C.
fig_data2A, fig_stat2A = get_pred_data(y_true = y_test_tn_sm, 
                                       y_pred = y_pred_tn_sm, 
                                       models = mdl_ord)

fig_data2B, fig_stat2B = get_pred_data(y_true = y_test_tn_val_sm, 
                                       y_pred = y_pred_tn_val_sm, 
                                       models = mdl_ord)

fig_data2C, fig_stat2C = get_pred_data(y_true = y_test_bn_sm, 
                                       y_pred = y_pred_bn_sm, 
                                       models = mdl_ord)

fig_data2_I = [fig_data2A, fig_data2B, fig_data2C]
fig_stat2_I = [fig_stat2A, fig_stat2B, fig_stat2C]


## get data for fig. 2D-F.
fig_data2_II = pd.concat(
    [perf_test_tn_sm.AUC, perf_test_tn_val_sm.AUC, perf_test_bn_sm.AUC], 
    axis = 1, keys = ds_info.Dataset).loc[
    mdl_ord].set_axis(
    labels = mdl_names, axis = 0).reset_index(
    names = "model")


#%% make fig. 2-I.

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
fig_ttls2  = ds_info.apply(lambda x: f"{x.Dataset} (n = {x.n})\n", axis = 1)
fig_ylim2  = 0.5
fig_ploc2  = 0.4
fig_llocs2 = [0.95, 0.66, 0.35]

fig2_I, ax2_I = plt.subplots(figsize = (18, 8), dpi = 600, nrows = 3, 
                             ncols = 1, sharex = True, sharey = True)
ax2_I = dict(zip(list("ABC"), ax2_I))

## make violins.
for k, (lbl, ax) in enumerate(ax2_I.items()):
    ax = make_violinplot(data = fig_data2_I[k], x = "model", y = "score", 
                         hue = "Response", stats = fig_stat2_I[k], 
                         order = mdl_ord, hue_order = ["R", "NR"], 
                         inner = "quart", split = True, dodge = True, 
                         statloc = fig_ploc2, statline = False, 
                         title = fig_ttls2[k], legend_vert = True, 
                         legend_out = True, legend_title = "Response", 
                         ax = ax)
    ax.set_ylim([0 - fig_ylim2, 1 + fig_ylim2]);
    ax.set_xticks(ticks = range(len(mdl_names)), 
                  labels = [""] * len(mdl_names));
    if k != 1:    ax.legend([ ], [ ]);
    fig2_I.text(x = 0.0, y = fig_llocs2[k], s = lbl, **panel_fonts);

## format ticks & labels.
ax2_I["C"].set_yticks(ticks = np.arange(0, 1.5, 0.5), 
                      labels = np.arange(0, 1.5, 0.5));
ax2_I["C"].yaxis.set_major_formatter("{x:0.1f}");
ax2_I["C"].set_xticks(ticks = range(len(mdl_names)), labels = mdl_names, 
                      rotation = 45, ha = "right", va = "top", ma = "center", 
                      position = (0, 0.02), **label_fonts);
fig2_I.supylabel("Prediction score", x = 0.01, y = 0.53, **label_fonts);

fig2_I.tight_layout(h_pad = 2, w_pad = 0)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file2_I = "all_predictions_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    fig2_I.savefig(fig_path + fig_file2_I, dpi = "figure")


#%% make fig. 2-II.

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
fig_theta2 = RadarChart(num_vars = len(mdl_names), frame = "circle")
fig_base2  = [0.5] * len(fig_theta2)
fig_ttls2  = ds_info.apply(lambda x: f"{x.Dataset} (n = {x.n})", axis = 1)
fig_llocs2 = [0.02, 0.32, 0.62]
fig_ticks2 = pad_radar_ticks(ticks = fig_data2_II.model, pads = [12, 4])
fig_rlims2 = [[0.30, 0.95], [0.30, 0.95], [0.20, 0.85]]

fig2_II, ax2_II = plt.subplots(figsize = (18, 6), dpi = 600, nrows = 1, 
                               ncols = 3, subplot_kw = {"projection": "radar"})
ax2_II = dict(zip(list("DEF"), ax2_II))

## make radars.
for k, (ds, (lbl, ax)) in enumerate(zip(ds_info.Dataset, ax2_II.items())):
    ax = make_radar_lines(theta = fig_theta2, data = fig_data2_II[ds], 
                          labels = fig_ticks2, color = colors[3], 
                          alpha = 0.4, ls = "-", lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_theta2, data = fig_base2, 
                          title = fig_ttls2[k], color = colors[-3], 
                          alpha = 0.15, ls = ":", ms = 8, ax = ax)
    ax.set_rlim(fig_rlims2[k]);
    fig2_II.text(x = fig_llocs2[k], y = 0.9, s = lbl, **panel_fonts);
# ax2_II["F"].set_rlim([0.30, 0.85]);

## format legends.
ax2_II["F"].legend(labels = ["Cell type", "Random"], loc = (1.12, 0.4), 
                   title = "AUC", prop = legend_fonts["item"], 
                   title_fontproperties = legend_fonts["title"])

fig2_II.tight_layout(h_pad = 0, w_pad = 2)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file2_II = "all_aucs_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    fig2_II.savefig(fig_path + fig_file2_II, dpi = "figure")


#%% prepare data for fig. 3-I.

## cell type shorthands.
ctp_abbv = {"B-cells": "B", "CAFs": "CAF", "Cancer_Epithelial": "CE", 
            "Endothelial": "ENDO", "Myeloid": "MYL", "Normal_Epithelial": "NE", 
            "PVL": "PVL", "Plasmablasts": "PB", "T-cells": "T", "Bulk": "Bulk"}

def get_ens_data(perfs, n_ens, score = "AUC", ds = None):
    n_ens      = n_ens - 1
    perfs_ens  = perfs[
        perfs.index.map(lambda x: x.count("+") == n_ens)].sort_values(
        by = ["AUC", "AP"], ascending = [False, False])
    
    perfs_data = pd.concat(
        [perfs_ens, perfs.loc[["Bulk"]]], axis = 0).rename(
        index = lambda mdl: " + ".join([ctp_abbv[x] for x in mdl.split("+")]))[
        [score]]
    if ds is not None:
        perfs_data.columns = [ds]
    
    return perfs_data
    

## get model ensemble data.
ds_names             = ds_info.apply(
    lambda df: f"{df.Dataset} (n = {df.n})", axis = 1).tolist()[1:]

perf_test_tn_val_sm2 = get_ens_data(
    perfs = perf_test_tn_val_sm, n_ens = 2, score = "AUC", ds = ds_names[0])

perf_test_tn_val_sm3 = get_ens_data(
    perfs = perf_test_tn_val_sm, n_ens = 3, score = "AUC", ds = ds_names[0])

perf_test_bn_sm2     = get_ens_data(
    perfs = perf_test_bn_sm, n_ens = 2, score = "AUC", ds = ds_names[1])

perf_test_bn_sm3     = get_ens_data(
    perfs = perf_test_bn_sm, n_ens = 3, score = "AUC", ds = ds_names[1])


## get data for fig. 3A-B.
n_top       = 10                                                               # keep only top ensembles

fig_data3A  = pd.concat(
    [perf_test_tn_val_sm2, perf_test_bn_sm2], axis = 1).pipe(
    lambda df: pd.concat(
        [df[:-1].sort_values(
            by = df.columns.tolist(), 
            ascending = [False] * df.shape[1])[:n_top], 
        df.loc[["Bulk"]]], axis = 0)).reset_index(
    names = "model")


fig_data3B  = pd.concat(
    [perf_test_tn_val_sm3, perf_test_bn_sm3], axis = 1).pipe(
    lambda df: pd.concat(
        [df[:-1].sort_values(
            by = df.columns.tolist(), 
            ascending = [False] * df.shape[1])[:n_top], 
        df.loc[["Bulk"]]], axis = 0)).reset_index(
    names = "model")

fig_data3_I = [fig_data3A, fig_data3B]


#%% make fig. 3-I.

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

## radar charts.
fig_theta3  = RadarChart(num_vars = n_top + 1, frame = "circle")
fig_base3   = [0.5] * len(fig_theta3)
fig_ttls3   = ["Two-cell-type ensemble", "Three-cell-type ensemble"]
fig_llocs3  = [0.01, 0.38]
fig_colors3 = [colors[3], colors[5], colors[-3]]
fig_ticks3  = [
    pad_radar_ticks(ticks = fig_data3_I[0].model, pads = [4, 12]), 
    pad_radar_ticks(ticks = fig_data3_I[1].model, pads = [18, 20]) ]

fig3_I, ax3_I = plt.subplots(figsize = (14, 5), dpi = 600, nrows = 1, 
                             ncols = 2, subplot_kw = {"projection": "radar"})
ax3_I = dict(zip(list("AB"), ax3_I))

## make radars.
for i, (lbl, ax) in enumerate(ax3_I.items()):
    for j, ds in enumerate(ds_names):
        ax = make_radar_lines(theta = fig_theta3, data = fig_data3_I[i][ds], 
                              labels = fig_ticks3[i], color = fig_colors3[j], 
                              alpha = 0.4, ls = "-", lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_theta3, data = fig_base3, 
                          title = fig_ttls3[i], color = fig_colors3[-1], 
                          alpha = 0.15, ls = ":", ms = 8, ax = ax)
    ax.set_rlim([0.30, 0.96])
    fig3_I.text(x = fig_llocs3[i], y = 0.9, s = lbl, **panel_fonts);
    
ax3_I["B"].legend(labels = ds_names + ["Random"], 
                  loc = (1.36, 0.45), title = "AUC", 
                  prop = legend_fonts["item"], 
                  title_fontproperties = legend_fonts["title"])

fig3_I.tight_layout(h_pad = 0, w_pad = 7)
plt.show()
    

## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file3_I = "top_ensemble_aucs_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    fig3_I.savefig(fig_path + fig_file3_I, dpi = "figure")


#%% prepare data for fig. 1.

clin_data_tn_sm     = clin_data_tn.loc[y_test_tn_sm.index].pipe(
    lambda df: pd.DataFrame({
        "Subtype"  : df["ER.status"].map(
            lambda x: "ER+,HER2-" if x == "POS" else "TNBC"), 
        "Response" : df["pCR.RD"].map(
            lambda x: "R" if x == "pCR" else "NR"), 
        "Treatment": df["NAT.regimen"] }))

clin_data_tn_val_sm = clin_data_tn_val.loc[y_test_tn_val_sm.index].pipe(
    lambda df: pd.DataFrame({
        "Subtype"  : df["ER.status"].map(
            lambda x: "ER+,HER2-" if x == "POS" else "TNBC"), 
        "Response" : df["pCR.RD"].map(
            lambda x: "R" if x == "pCR" else "NR"), 
        "Treatment": df["Chemo.Regimen"] }))

clin_data_bn_sm     = clin_data_bn.loc[y_test_bn_sm.index].pipe(
    lambda df: pd.DataFrame({
        "Subtype"  : ["TNBC"] * len(df), 
        "Response" : df["pathologic_complete_response"].map(
            lambda x: "R" if x == "pCR" else "NR"), 
        "Treatment": df["treatment"].replace(
            to_replace = {"Carboplatin+Paclitaxel": "P-Carboplatin"}) }))


## prepare data for fig. 1A.
def get_subtype_response_counts(clin):
    sbtyp = ["ER+,HER2-", "TNBC"]
    resp  = ["R", "NR"]
    
    stat_data = pd.DataFrame({
        "Subtype" : np.repeat(sbtyp, repeats = len(resp)), 
        "Response": np.tile(resp, reps = len(sbtyp)) })
    
    stat_data["Count"] = stat_data.apply(
        lambda x: clin.eq(x).all(axis = 1).sum(), axis = 1)
    
    return stat_data


def get_drug_fractions(clin):
    return pd.Series(Counter(clin), name = "Fraction") / len(clin)
    


ds_info["label"] = ds_info.apply(
    lambda x: f"{x.Dataset} (n = {x.n})", axis = 1)

fig_data1A = [
    clin_data_tn_sm.drop(
        columns = "Treatment").pipe(
        get_subtype_response_counts), 
    clin_data_tn_val_sm.drop(
        columns = "Treatment").pipe(
        get_subtype_response_counts), 
    clin_data_bn_sm.drop(
        columns = "Treatment").pipe(
        get_subtype_response_counts) ]


fig_data1B = pd.DataFrame({
    ds_info.label[k]: clin["Treatment"].pipe(
        get_drug_fractions) for k, clin in enumerate([
            clin_data_tn_sm, clin_data_tn_val_sm, clin_data_bn_sm])})
fig_data1B = fig_data1B.reset_index(
    names = "Regimen").replace(
    regex = {"Regimen": {"Carboplatin": "Cb"}})
    #     .melt(
    # id_vars = "Regimen", var_name = "Dataset", value_name = "Proportion")


#%% make fig. 1-I.

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
fig_lblord1  = [["R", "NR"], ["ER+,HER2-", "TNBC"]]
fig_colors1  = [colors[0], colors[1], colors[2], colors[5]]
fig_legend1A = np.append(
    fig_data1A[2].Response.map(lambda x: f"{x}{' ' * 6}").unique(), 
    fig_data1A[2].Subtype.unique())

fig1_I, ax1_I = plt.subplot_mosaic(
    mosaic = [["A1", "A2", "A3"], ["B1", "B2", "B3"]], 
    # mosaic = [["A1", "A2", "A3"], ["B", "B", "B"]], 
    figsize = (18, 9), dpi = 600, width_ratios = [1, 1, 1])

ax1_I["A1"] = make_donutplots(data = fig_data1A[0], x = "Count", 
                              outer = "Response", inner = "Subtype", 
                              outer_order = fig_lblord1[0], 
                              inner_order = fig_lblord1[1], 
                              colors = fig_colors1, donut_size = fig_dntsize1, 
                              title = ds_info.label[0], ax = ax1_I["A1"])

ax1_I["A2"] = make_donutplots(data = fig_data1A[1], x = "Count", 
                              outer = "Response", inner = "Subtype", 
                              outer_order = fig_lblord1[0], 
                              inner_order = fig_lblord1[1], 
                              colors = fig_colors1, donut_size = fig_dntsize1, 
                              title = ds_info.label[1], ax = ax1_I["A2"])

ax1_I["A3"] = make_donutplots(data = fig_data1A[2], x = "Count", 
                              outer = "Response", inner = "Subtype", 
                              outer_order = fig_lblord1[0], 
                              inner_order = fig_lblord1[1], 
                              colors = fig_colors1, donut_size = fig_dntsize1, 
                              title = ds_info.label[2], ax = ax1_I["A3"])

ax1_I["A3"].legend(labels = fig_legend1A, loc = (1.20, 0.45), ncols = 2, 
                   title = f"Response{' ' * 4}Subtype", 
                   alignment = "left", prop = legend_fonts["item"], 
                   title_fontproperties = legend_fonts["title"]);

fig1_I.text(x = 0.0, y = 0.95, s = "A", **panel_fonts);


ax1_I["B1"] = make_hbars(data = fig_data1B, x = ds_info.label[0], 
                          y = "Regimen", width = 0.8, color = colors[4], 
                          # title = ds_info.label[0], 
                          ax = ax1_I["B1"])
ax1_I["B1"].set_xticks(np.arange(0, 0.9, 0.2));

ax1_I["B2"] = make_hbars(data = fig_data1B, x = ds_info.label[1], 
                          y = "Regimen", width = 0.8, color = colors[4], 
                          # title = ds_info.label[1], 
                          ax = ax1_I["B2"])
# ax1_I["B2"].set_yticks(ticks = fig_data1B.Regimen, 
#                        labels = [" "] * len(fig_data1B));
ax1_I["B2"].set_xticks(np.arange(0, 1.0, 0.2));

ax1_I["B3"] = make_hbars(data = fig_data1B, x = ds_info.label[2], 
                          y = "Regimen", width = 0.8, color = colors[4], 
                          # title = ds_info.label[2], 
                          ax = ax1_I["B3"])
# ax1_I["B3"].set_yticks(ticks = fig_data1B.Regimen, 
#                        labels = [" "] * len(fig_data1B));
ax1_I["B3"].set_xticks(np.arange(0, 1.2, 0.2));
ax1_I["B3"].set_xlim([0, 1]);
fig1_I.supxlabel("Proportion of patients", y = 0.015, **label_fonts);
fig1_I.supylabel("Treatment regimen", x = 0.010, y = 0.25, **label_fonts);

# ax1_I["B"] = make_barplot3(data = fig_data1B, x = "Dataset", y = "Proportion", 
#                             hue = "Regimen", bar_labels = False, 
#                             xlabels = fig_data1B["Dataset"].unique(), 
#                             title = "Neoadjuvant treatment", 
#                             ylabel = "Proportion of samples", ax = ax1_I["B"], 
#                             fontdict = plot_fonts)
# ax1_I["B"].get_legend().set(bbox_to_anchor = (1.0, -0.15, 0.6, 0.4), 
#                             frame_on = False, title = "Regimen");
# ax1_I["B"].set_ylim([0.0, 1.05]);
fig1_I.text(x = 0.0, y = 0.48, s = "B", **panel_fonts);

fig1_I.tight_layout(h_pad = 2, w_pad = 4)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file1_I = "dataset_summary_bulk.pdf"
    fig1_I.savefig(fig_path + fig_file1_I, dpi = "figure")


#%% prepare data for supp. fig. 2.

## get data for supp. fig. 2A-C.
fig_dataS2_I  = pd.concat(
    [perf_test_tn_sm.AP, perf_test_tn_val_sm.AP, perf_test_bn_sm.AP], 
    axis = 1, keys = ds_info.Dataset).loc[
    mdl_ord].set_axis(
    labels = mdl_names, axis = 0).reset_index(
    names = "model")

## get data for supp. fig. 2D.
fig_dataS2D    = pd.concat([
    perf_test_tn.DOR, perf_test_tn_val.DOR, perf_test_bn.DOR], 
    axis = 1, keys = ds_info.label).loc[
    mdl_ord].set_axis(
    labels = mdl_names, axis = 0).dropna(
    how = "all", axis = 0).reset_index(
    names = "model").melt(
    id_vars = "model", var_name = "Dataset", value_name = "score")


## get data for supp. fig. 2E-F.
ds_names      = ds_info.apply(
    lambda df: f"{df.Dataset} (n = {df.n})", axis = 1).tolist()[1:]

fig_dataS2E   = pd.concat([
    get_ens_data(perfs = perf_test_tn_val_sm, score = "AP", n_ens = 2, 
                 ds = ds_names[0]), 
    get_ens_data(perfs = perf_test_bn_sm, score = "AP", n_ens = 2, 
                 ds = ds_names[1])], axis = 1).loc[
    fig_data3A.model].reset_index(
    names = "model")

fig_dataS2F   = pd.concat([
    get_ens_data(perfs = perf_test_tn_val_sm, score = "AP", n_ens = 3, 
                 ds = ds_names[0]), 
    get_ens_data(perfs = perf_test_bn_sm, score = "AP", n_ens = 3, 
                 ds = ds_names[1])], axis = 1).loc[
    fig_data3B.model].reset_index(
    names = "model")

fig_dataS2_III = [fig_dataS2E, fig_dataS2F]


## get data for supp. fig. 2G-H.
fig_dataS2G    = pd.concat([
    get_ens_data(perfs = perf_test_tn_val, score = "DOR", n_ens = 2, 
                 ds = ds_names[0]), 
    get_ens_data(perfs = perf_test_bn, score = "DOR", n_ens = 2, 
                 ds = ds_names[1])], axis = 1).loc[
    fig_data3A.model].reset_index(
    names = "model").melt(
    id_vars = "model", var_name = "Dataset", value_name = "score")

fig_dataS2H    = pd.concat([
    get_ens_data(perfs = perf_test_tn_val, score = "DOR", n_ens = 3, 
                 ds = ds_names[0]), 
    get_ens_data(perfs = perf_test_bn, score = "DOR", n_ens = 3, 
                 ds = ds_names[1])], axis = 1).loc[
    fig_data3B.model].reset_index(
    names = "model").melt(
    id_vars = "model", var_name = "Dataset", value_name = "score")


#%% make supp. fig. 2-I.

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
fig_thetaS2_I = RadarChart(num_vars = len(mdl_names), frame = "circle")
fig_baseS2_I  = {ds: [info.R / info.n] * len(fig_thetaS2_I) 
                 for ds, info in ds_info.set_index(keys = "Dataset").iterrows()}
fig_ttlsS2_I  = ds_info.apply(lambda x: f"{x.Dataset} (n = {x.n})", axis = 1)
fig_llocsS2_I = [0.02, 0.32, 0.62]
fig_ticksS2_I = pad_radar_ticks(ticks = fig_dataS2_I.model, pads = [12, 4])

figS2_I, axS2_I = plt.subplots(figsize = (18, 6), dpi = 600, nrows = 1, 
                               ncols = 3, subplot_kw = {"projection": "radar"})
axS2_I = dict(zip(list("ABC"), axS2_I))

## make radars.
for k, (ds, (lbl, ax)) in enumerate(zip(ds_info.Dataset, axS2_I.items())):
    ax = make_radar_lines(theta = fig_thetaS2_I, data = fig_dataS2_I[ds], 
                          labels = fig_ticksS2_I, color = colors[3], 
                          alpha = 0.4, ls = "-", lw = 2, ms = 8, ax = ax)
    ax = make_radar_lines(theta = fig_thetaS2_I, data = fig_baseS2_I[ds], 
                          title = fig_ttlsS2_I[k], color = colors[-3], 
                          alpha = 0.15, ls = ":", ms = 8, ax = ax)
    ax.set_rlim([0.10, 0.85]);
    figS2_I.text(x = fig_llocsS2_I[k], y = 0.9, s = lbl, **panel_fonts);

## format legends.
axS2_I["C"].legend(labels = ["Cell type", "Random"], loc = (1.12, 0.4), 
                   title = "AP", prop = legend_fonts["item"], 
                   title_fontproperties = legend_fonts["title"])

figS2_I.tight_layout(h_pad = 0, w_pad = 2)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS2_I = "all_aps_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS2_I.savefig(fig_path + fig_fileS2_I, dpi = "figure")


#%% make supp. fig. 2-II.

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
    "label": {"family": "sans", "size": 12, "weight": "regular"}, 
    "title": {"family": "sans", "size": 16, "weight": "bold"}, 
    "super": {"family": "sans", "size": 20, "weight": "bold"}}

## radar charts.
fig_ttlS2_II    = "Model performance"
fig_llocsS2_II  = [0.01, 0.38]
fig_colorsS2_II = [colors[4], colors[3], colors[5], colors[-3], colors[-2]]

figS2_II, axS2_II = plt.subplots(figsize = (14, 5), dpi = 600, nrows = 2, 
                                 ncols = 1)
axS2_II = dict(zip(["D1", "D2"], axS2_II))

axS2_II["D1"], axS2_II["D2"] = make_barplot3(
    data = fig_dataS2D, x = "model", y = "score", hue = "Dataset", 
    bar_labels = False, xlabels = fig_dataS2D["model"].unique(), xrot = 35, 
    title = fig_ttlS2_II, colors = fig_colorsS2_II, 
    ax = (axS2_II["D1"], axS2_II["D2"]), fontdict = plot_fonts)
axS2_II["D2"].axhline(y = 1, xmin = 0, xmax = fig_dataS2F["model"].nunique(), 
                      color = colors[-1], ls = "--", lw = 1.5);
axS2_II["D2"].get_legend().set(bbox_to_anchor = (1.0, 0.45, 0.6, 0.4), 
                               frame_on = False, title = "Dataset");

figS2_II.supylabel("Odds ratio", x = 0.02, **label_fonts);
figS2_II.text(x = 0.01, y = 0.98, s = "D", **panel_fonts);

figS2_II.tight_layout(h_pad = 0, w_pad = 0)
plt.show()
    

## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS2_II = "all_dors_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS2_II.savefig(fig_path + fig_fileS2_II, dpi = "figure")


#%% make supp. fig. 2-III.

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

## radar charts.
fig_thetaS2_III  = RadarChart(num_vars = n_top + 1, frame = "circle")
fig_baseS2_III   = {ds: [info.R / info.n] * len(fig_thetaS2_III) 
                   for ds, info in ds_info.set_index(keys = "Dataset").iterrows()}
fig_ttlsS2_III   = ["Two-cell-type ensemble", "Three-cell-type ensemble"]
fig_llocsS2_III  = [0.01, 0.38]
fig_colorsS2_III = [colors[3], colors[5], colors[-3], colors[-2]]
fig_ticksS2_III  = [
    pad_radar_ticks(ticks = fig_dataS2_III[0].model, pads = [4, 12]), 
    pad_radar_ticks(ticks = fig_dataS2_III[1].model, pads = [18, 20]) ]

figS2_III, axS2_III = plt.subplots(figsize = (14, 5), dpi = 600, 
                                   nrows = 1, ncols = 2, 
                                   subplot_kw = {"projection": "radar"})
axS2_III = dict(zip(list("EF"), axS2_III))

## make radars.
for i, (lbl, ax) in enumerate(axS2_III.items()):
    for j, ds in enumerate(ds_names):
        ax = make_radar_lines(
            theta = fig_thetaS2_III, data = fig_dataS2_III[i][ds], 
            labels = fig_ticksS2_III[i], color = fig_colorsS2_III[j], 
            alpha = 0.4, ls = "-", lw = 2, ms = 8, ax = ax)
    for j, ds in enumerate(ds_names, start = 1):
        ds = ds.split(" (")[0]
        ax = make_radar_lines(
            theta = fig_thetaS2_III, data = fig_baseS2_III[ds], 
            title = fig_ttlsS2_III[i], color = fig_colorsS2_III[-j], 
            alpha = 0.15, ls = ":", ms = 8, ax = ax)
    ax.set_rlim([0.10, 0.85])
    figS2_III.text(x = fig_llocsS2_III[i], y = 0.9, s = lbl, **panel_fonts);
    
axS2_III["F"].legend(labels = ds_names + [f"Random ({ds.split(' (')[0]})" 
                                          for ds in ds_names], 
                     loc = (1.36, 0.45), title = "AP", 
                     prop = legend_fonts["item"], 
                     title_fontproperties = legend_fonts["title"])

figS2_III.tight_layout(h_pad = 0, w_pad = 7)
plt.show()
    

## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS2_III = "top_ensemble_aps_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS2_III.savefig(fig_path + fig_fileS2_III, dpi = "figure")


#%% make supp. fig. 2-IV.

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
    "label": {"family": "sans", "size": 12, "weight": "regular"}, 
    "title": {"family": "sans", "size": 16, "weight": "bold"}, 
    "super": {"family": "sans", "size": 20, "weight": "bold"}}

## radar charts.
fig_ttlS2_IV    = "Diagnostic odds ratio"
fig_ttlsS2_IV   = ["Two-cell-type ensemble", "Three-cell-type ensemble"]
fig_llocsS2_IV  = [0.01, 0.38]
fig_colorsS2_IV = [colors[3], colors[5], colors[-3], colors[-2]]

figS2_IV, axS2_IV = plt.subplot_mosaic(
    mosaic = [["G1", "H1"], ["G2", "H2"]], figsize = (14, 5), dpi = 600, 
    layout = "constrained", height_ratios = [1, 1], width_ratios = [1, 1], 
    sharex = False, sharey = False)

axS2_IV["G1"], axS2_IV["G2"] = make_barplot3(
    data = fig_dataS2G, x = "model", y = "score", hue = "Dataset", 
    bar_labels = False, xlabels = fig_dataS2G["model"].unique(), xrot = 35, 
    legend = False, title = fig_ttlsS2_IV[0], colors = fig_colorsS2_IV, 
    ax = (axS2_IV["G1"], axS2_IV["G2"]), fontdict = plot_fonts)
axS2_IV["G2"].axhline(y = 1, xmin = 0, xmax = fig_dataS2G["model"].nunique(), 
                      color = colors[-1], ls = "--", lw = 1.5);
figS2_IV.text(x = fig_llocsS2_IV[0], y = 0.98, s = "G", **panel_fonts);

axS2_IV["H1"], axS2_IV["H2"] = make_barplot3(
    data = fig_dataS2H, x = "model", y = "score", hue = "Dataset", 
    bar_labels = False, xlabels = fig_dataS2H["model"].unique(), xrot = 35, 
    title = fig_ttlsS2_IV[1], colors = fig_colorsS2_IV, 
    ax = (axS2_IV["H1"], axS2_IV["H2"]), fontdict = plot_fonts)
axS2_IV["H2"].axhline(y = 1, xmin = 0, xmax = fig_dataS2G["model"].nunique(), 
                      color = colors[-1], ls = "--", lw = 1.5);
axS2_IV["H2"].get_legend().set(bbox_to_anchor = (1.0, 0.45, 0.6, 0.4), 
                               frame_on = False, title = "Dataset");
figS2_IV.text(x = fig_llocsS2_IV[1], y = 0.98, s = "H", **panel_fonts);

figS2_IV.supylabel("Odds ratio", x = -0.01, **label_fonts);

# figS2_IV.tight_layout(h_pad = 0, w_pad = 7)
plt.show()
    

## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS2_IV = "top_ensemble_dors_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS2_IV.savefig(fig_path + fig_fileS2_IV, dpi = "figure")


#%% prepare data for supp. fig. 3.

## prepare data for division by subtype.
def get_samples_by_subtype(clin_data, col = "ER.status"):
    return { "ER+,HER2-": clin_data[clin_data[col].eq("POS")].index.tolist(), 
             "TNBC": clin_data[clin_data[col].eq("NEG")].index.tolist() }

def get_perf_ctp(y_true, y_pred):
    return y_pred.apply(
        lambda y_pc: pd.Series(classifier_performance(y_true, y_pc))).T


## get data by subtype.
clin_data_tn_sm   = clin_data_tn.loc[samples_tn_sm].copy()
# y_pred_tn_all     = pd.concat(
#     [y_pred_tn_sm, y_pred_sammut_tn["Clinical+RNA"].rename("Sammut et al.")], 
#     axis = 1)
# y_pred_tn_val_all = pd.concat(
#     [y_pred_tn_val, y_pred_sammut_tn_val["Clinical+RNA"].rename("Sammut et al.")], 
#     axis = 1)

smpl_subtype_tn     = get_samples_by_subtype(clin_data_tn_sm)
smpl_subtype_tn_val = get_samples_by_subtype(clin_data_tn_val)


## comput performance by subtype.
model_list  = cell_types + ["Bulk", "Sammut et al."]
model_list2 = np.append(np.setdiff1d(
    y_pred_tn_val_sm.columns, cell_types + ["Bulk", "Sammut et al."]), 
    ["Bulk", "Sammut et al."]).tolist()

perf_test_subtype_tn      = {
    styp_: get_perf_ctp(y_test_tn_sm.loc[smpl_], 
                        y_pred_tn_sm.loc[smpl_, model_list]) \
        for styp_, smpl_ in smpl_subtype_tn.items()}

perf_test_subtype_tn_val  = {
    styp_: get_perf_ctp(y_test_tn_val.loc[smpl_], 
                        y_pred_tn_val_sm.loc[smpl_, model_list]) \
        for styp_, smpl_ in smpl_subtype_tn_val.items()}

perf_test_subtype_tn_val2 = {
    styp_: get_perf_ctp(y_test_tn_val.loc[smpl_], 
                        y_pred_tn_val_sm.loc[smpl_, model_list2]) \
        for styp_, smpl_ in smpl_subtype_tn_val.items()}


## prepare data for figure.
## row 1 - TransNEO.
mdl_ord       = perf_test_subtype_tn["ER+,HER2-"].drop(
    index = ["Bulk", "Sammut et al."]).sort_values(
        by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + \
            ["Bulk", "Sammut et al."]
fig_dataS3A   = perf_test_subtype_tn["ER+,HER2-"].loc[
    mdl_ord].reset_index().rename(columns = {"index": "cell_type"}).melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")
fig_xticksS3A = [mdl.replace("_", "\n") for mdl in mdl_ord]

mdl_ord       = perf_test_subtype_tn["TNBC"].drop(
    index = ["Bulk", "Sammut et al."]).sort_values(
        by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + \
            ["Bulk", "Sammut et al."]
fig_dataS3B   = perf_test_subtype_tn["TNBC"].loc[
    mdl_ord].reset_index().rename(columns = {"index": "cell_type"}).melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")
fig_xticksS3B = [mdl.replace("_", "\n") for mdl in mdl_ord]


## row 2 - ARTemis + PBCP.
mdl_ord       = perf_test_subtype_tn_val["ER+,HER2-"].drop(
    index = ["Bulk", "Sammut et al."]).sort_values(
        by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + \
            ["Bulk", "Sammut et al."]
fig_dataS3C   = perf_test_subtype_tn_val["ER+,HER2-"].loc[
    mdl_ord].reset_index().rename(columns = {"index": "cell_type"}).melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")
fig_xticksS3C = [mdl.replace("_", "\n") for mdl in mdl_ord]

mdl_ord       = perf_test_subtype_tn_val["TNBC"].drop(
    index = ["Bulk", "Sammut et al."]).sort_values(
        by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + \
            ["Bulk", "Sammut et al."]
fig_dataS3D   = perf_test_subtype_tn_val["TNBC"].loc[
    mdl_ord].reset_index().rename(columns = {"index": "cell_type"}).melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")
fig_xticksS3D = [mdl.replace("_", "\n") for mdl in mdl_ord]


## row 3 - ARTemsis + PBCP (combos).
# mdl_ords      = dict(zip(mdl_ord3_tn_val + mdl_ord2_tn_val, 
#                          fig_xticks3B + fig_xticks3A))
# mdl_ords.pop("Bulk");

mdl_ords = (
    perf_test_tn_val_sm.pipe(
        lambda df: df.loc[df.index.map(
            lambda x: x.count("+") == 1)]).sort_values(
                by = ["AUC", "AP"], ascending = False).index.tolist()[:5] + 
    perf_test_tn_val_sm.pipe(
        lambda df: df.loc[df.index.map(
            lambda x: x.count("+") == 2)]).sort_values(
                by = ["AUC", "AP"], ascending = False).index.tolist()[:5])
mdl_ords = dict(zip(mdl_ords, (fig_data3A.model.tolist()[:5] + 
                               fig_data3B.model.tolist()[:5])))

mdl_ord       = perf_test_subtype_tn_val2["ER+,HER2-"].loc[
    list(mdl_ords.keys())].sort_values(
        by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + \
            ["Bulk", "Sammut et al."]
fig_dataS3E   = perf_test_subtype_tn_val2["ER+,HER2-"].loc[
    mdl_ord].reset_index().rename(columns = {"index": "cell_type"}).melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")
fig_xticksS3E = [mdl_ords[mdl] for mdl in mdl_ord[:-2]] + \
    ["Bulk", "Sammut et al."]

mdl_ord       = perf_test_subtype_tn_val2["TNBC"].loc[
    list(mdl_ords.keys())].sort_values(
        by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + \
            ["Bulk", "Sammut et al."]
fig_dataS3F   = perf_test_subtype_tn_val2["TNBC"].loc[
    mdl_ord].reset_index().rename(columns = {"index": "cell_type"}).melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")
fig_xticksS3F = [mdl_ords[mdl] for mdl in mdl_ord[:-2]] + \
    ["Bulk", "Sammut et al."]


#%% generate supp. fig. 3.

svdat = False

## plot parameters.
fontname = "sans"
fontdict = {"label": dict(fontfamily = fontname, fontsize = 56, 
                          fontweight = "regular"), 
            "title": dict(fontfamily = fontname, fontsize = 60, 
                          fontweight = "semibold"), 
            "super": dict(fontfamily = fontname, fontsize = 64, 
                          fontweight = "bold"),
            "plabel": dict(fontfamily = "sans-serif", fontsize = 120, 
                           fontweight = "bold")}

colors   = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
            "#FFC72C", "#708090", "#A9A9A9", "#000000"]

sns.set_style("ticks")
plt.rcParams.update({"xtick.major.size": 12, "xtick.major.width": 4, 
                     "ytick.major.size": 12, "ytick.major.width": 4, 
                     "xtick.bottom": True, "ytick.left": True, 
                     "axes.edgecolor": "#000000", "axes.linewidth": 4})

fig_colorsS3  = [colors[3], colors[4]]
fig_llocsS3   = [[0.02, 0.48], [1.00, 0.66, 0.33]]
fig_xrotS3    = 35

figS3, axS3 = plt.subplot_mosaic(
    mosaic = [["A", "B"], ["C", "D"], ["E", "F"]], figsize = (80, 48), 
    dpi = 600, layout = "constrained", height_ratios = [1] * 3, 
    width_ratios = [1] * 2)

axS3["A"] = make_barplot2(data = fig_dataS3A, x = "cell_type", y = "score", 
                      hue = "metric", width = 0.5, colors = fig_colorsS3, 
                      title = f"TransNEO: ER+,HER2- (n = {len(smpl_subtype_tn['ER+,HER2-'])})", 
                      legend = False, xlabels = fig_xticksS3A, 
                      xrot = fig_xrotS3, bar_label_align = True, ax = axS3["A"], 
                      fontdict = fontdict)
axS3["A"].set_ylim([-0.04, 1.04]);
axS3["A"].set_title(axS3["A"].get_title(), y = 1.10, **fontdict["super"]);
# axS3["A"].set_title("A", x = -0.025, y = 1.02, **fontdict["plabel"]);
figS3.text(x = fig_llocsS3[0][0], y = fig_llocsS3[1][0], s = "A", 
           color = "#000000", **fontdict["plabel"]);

axS3["B"] = make_barplot2(data = fig_dataS3B, x = "cell_type", y = "score", 
                     hue = "metric", width = 0.5, colors = fig_colorsS3, 
                     title = f"TransNEO: TNBC (n = {len(smpl_subtype_tn['TNBC'])})", 
                     legend = False, xlabels = fig_xticksS3B, 
                     xrot = fig_xrotS3, bar_label_align = True, ax = axS3["B"], 
                     fontdict = fontdict)
axS3["B"].set_ylim([-0.04, 1.04]);
axS3["B"].set_title(axS3["B"].get_title(), y = 1.10, **fontdict["super"]);
# axS3["B"].set_title("B", x = -0.025, y = 1.02, **fontdict["plabel"]);
figS3.text(x = fig_llocsS3[0][1], y = fig_llocsS3[1][0], s = "B", 
           color = "#000000", **fontdict["plabel"]);

axS3["C"] = make_barplot2(data = fig_dataS3C, x = "cell_type", y = "score", 
                      hue = "metric", width = 0.5, colors = fig_colorsS3, 
                      title = f"ARTemis + PBCP: ER+,HER2- (n = {len(smpl_subtype_tn_val['ER+,HER2-'])})", 
                      legend = False, xlabels = fig_xticksS3C, xrot = fig_xrotS3, 
                      bar_label_align = True, ax = axS3["C"], 
                      fontdict = fontdict)
axS3["C"].set_ylim([-0.04, 1.04]);
# axS3["C"].set_title("C", x = -0.025, y = 1.02, **fontdict["plabel"]);
figS3.text(x = fig_llocsS3[0][0], y = fig_llocsS3[1][1], s = "C", 
           color = "#000000", **fontdict["plabel"]);

axS3["D"] = make_barplot2(data = fig_dataS3D, x = "cell_type", y = "score", 
                     hue = "metric", width = 0.5, colors = fig_colorsS3, 
                     title = f"ARTemis + PBCP: TNBC (n = {len(smpl_subtype_tn_val['TNBC'])})", 
                     legend = True, legend_title = "Performance", 
                     xlabels = fig_xticksS3B, xrot = fig_xrotS3, 
                     bar_label_align = True, ax = axS3["D"], 
                     fontdict = fontdict)
axS3["D"].set_ylim([-0.04, 1.04]);
axS3["D"].get_legend().set(bbox_to_anchor = (1.0, 0.3, 0.6, 0.6));
# axS3["D"].set_title("D", x = -0.025, y = 1.02, **fontdict["plabel"]);
figS3.text(x = fig_llocsS3[0][1], y = fig_llocsS3[1][1], s = "D", 
           color = "#000000", **fontdict["plabel"]);

axS3["E"] = make_barplot2(data = fig_dataS3E, x = "cell_type", y = "score", 
                      hue = "metric", width = 0.5, colors = fig_colorsS3, 
                      title = "", legend = False, xlabels = fig_xticksS3E, 
                      xrot = fig_xrotS3, bar_label_align = True, ax = axS3["E"], 
                      fontdict = fontdict)
axS3["E"].set_ylim([-0.04, 1.04]);
# axS3["E"].set_title("E", x = -0.025, y = 1.02, **fontdict["plabel"]);
figS3.text(x = fig_llocsS3[0][0], y = fig_llocsS3[1][2], s = "E", 
           color = "#000000", **fontdict["plabel"]);

axS3["F"] = make_barplot2(data = fig_dataS3F, x = "cell_type", y = "score", 
                     hue = "metric", width = 0.5, colors = fig_colorsS3, 
                     title = "", legend = False, xlabels = fig_xticksS3F, 
                     xrot = fig_xrotS3, bar_label_align = True, ax = axS3["F"], 
                     fontdict = fontdict)
axS3["F"].set_ylim([-0.04, 1.04]);
# axS3["F"].set_title("F", x = -0.025, y = 1.02, **fontdict["plabel"]);
figS3.text(x = fig_llocsS3[0][1], y = fig_llocsS3[1][2], s = "F", 
           color = "#000000", **fontdict["plabel"]);

figS3.tight_layout(h_pad = 8, w_pad = 12)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS3 = "all_performance_subtype_chemo_th0.99_ENS2_25features_5foldCV.pdf"
    figS3.savefig(fig_path + fig_fileS3, dpi = "figure")

