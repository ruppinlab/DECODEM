#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 13:58:37 2025

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
from miscellaneous import date_time, tic, write_xlsx
from math import floor
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from _functions import classifier_performance, make_barplot3
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import mannwhitneyu
from functools import reduce


#%% functions.

## load input data from saved pickle.
def load_data(path):
    with open(path, mode = "rb") as file:
        obj = pickle.load(file)
    
    exps, frac, conf = [obj[kk] for kk in ["exp", "frac", "conf"]]
    resp, clin       = [obj[kk] for kk in ["resp", "clin"]]
    del obj                                                                    # release memory
        
    ## sanity check.
    if conf.columns.tolist() == frac.columns.tolist():
        cells = conf.columns.tolist()
    else:
        raise ValueError("cell types are not the same between cell fraction and confidence score matrices!")
    
    return exps, resp, frac, conf, clin, cells


## rescale data by specified method.
def rescale(X, mode = "norm"):
    Xn = X.copy()
    if mode.lower() == "std":
        Xn[:] = StandardScaler().fit_transform(X)
    else:
        Xn[:] = MinMaxScaler().fit_transform(X)
    return Xn


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
def make_violinplot(data, x, y, hue, ax, orient = "v", stats = None, gap = 0.08, 
                    width = 0.8, dodge = True, split = False, fill = True, 
                    order = None, hue_order = None, statloc = 0.35, 
                    statline = False, inner = "quart", vnorm = "area", 
                    colors = None, xlabel = None, ylabel = None, title = None, 
                    legend_out = True, legend_title = None, legend_vert = True, 
                    fontdict = None):
    
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
    
    if inner.lower() == "box":
        innerprop = boxprop
    elif inner.lower() == "quart":
        innerprop = {"linestyle": "-", "linewidth": 1.5, "color": colors[-1]}
    elif inner.lower() == "point":
        innerprop = {"marker": "o", "facecolor": colors[4], 
                     "edgecolor": colors[-1], "alpha": 0.7, "linewidths": 0.5}
    
    
    ## main plot.
    sns.violinplot(
        data = data, x = x, y = y, hue = hue, width = width, orient = orient, 
        dodge = dodge, gap = gap, order = order, hue_order = hue_order, 
        inner = inner, inner_kws = innerprop, split = split, fill = fill, 
        palette = colors[:data[hue].nunique()], saturation = 0.8, 
        density_norm = vnorm, **lineprop, ax = ax)
    
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


#%% read data.

use_samples = "chemo"

data_path   = ["../../data/TransNEO/transneo_analysis/", 
               "../../data/TransNEO/TransNEO_SammutShare/validation/", 
               "../../data/BrighTNess/validation/"]

data_file   = [f"transneo_data_{use_samples}_v2.pkl",                            # original 9 cell types
               f"transneo_validation_{use_samples}_v2.pkl", 
               f"brightness_data_{use_samples}_v2.pkl", 
               f"transneo_data_{use_samples}_v3.pkl",                            # separate T-cell subtypes
               f"transneo_validation_{use_samples}_v3.pkl", 
               f"brightness_data_{use_samples}_v3.pkl"]


## load data.
print("loading & preparing data...");    _tic = tic()

## original 9 cell types.
_, resp_pCR_tn, cell_frac_tn, _, _, _ = \
    load_data(data_path[0] + data_file[0])

_, resp_pCR_tn_val, cell_frac_tn_val, _, _, _ = \
    load_data(data_path[1] + data_file[1])

_, resp_pCR_bn, cell_frac_bn, _, clin_info_bn, _ = \
    load_data(data_path[2] + data_file[2])

clin_info_bn = clin_info_bn[clin_info_bn["planned_arm_code"] == "B"]           # keep arm B only
resp_pCR_bn, cell_frac_bn = (resp_pCR_bn.loc[clin_info_bn.index], 
                             cell_frac_bn.loc[clin_info_bn.index])

cell_types = reduce(np.intersect1d, map(lambda df: df.columns, [
    cell_frac_tn, cell_frac_tn_val, cell_frac_bn])).tolist()


## separate T-cell subtypes.
_, resp_pCR_tn11, cell_frac_tn11, _, _, _ = \
    load_data(data_path[0] + data_file[3])

_, resp_pCR_tn_val11, cell_frac_tn_val11, _, _, _ = \
    load_data(data_path[1] + data_file[4])

_, resp_pCR_bn11, cell_frac_bn11, _, _, _ = \
    load_data(data_path[2] + data_file[5])
resp_pCR_bn11, cell_frac_bn11 = (resp_pCR_bn11.loc[clin_info_bn.index], 
                                 cell_frac_bn11.loc[clin_info_bn.index])

cell_types11       = ["CD4+_T-cells", "CD8+_T-cells"]                          # keep CD4/CD8 T-cells only
cell_frac_tn11     = cell_frac_tn11[cell_types11]
cell_frac_tn_val11 = cell_frac_tn_val11[cell_types11]
cell_frac_bn11     = cell_frac_bn11[cell_types11]


print("done!", end = "");    _tic.toc()


print(f"""
dataset summary: 
cohorts   = TransNEO (n = {len(resp_pCR_tn):,}), ARTesmis + PBCP (n = {len(resp_pCR_tn_val):,}), BrighTNess (n = {len(resp_pCR_bn):,})
treatment = {use_samples} + therapy, response = RCB ('pCR' vs. 'RD')
cell type = {', '.join(cell_types + cell_types11)}
""")


#%% association of abundance & response per cell type.

ds_info = pd.DataFrame(
    [{"n": len(y), "R": sum(y == 1), "NR": sum(y == 0)} 
     for y in [resp_pCR_tn, resp_pCR_tn_val, resp_pCR_bn]], 
    index = ["TransNEO", "ARTemis + PBCP", "BrighTNess"]).reset_index(
    names = "Dataset")
ds_info["label"] = ds_info.apply(
    lambda x: f"{x.Dataset} (n = {x.n})", axis = 1)


## rescale prediction in [0, 1].
y_pred_frac_tn        = rescale(cell_frac_tn, mode = "norm")
y_pred_frac_tn_val    = rescale(cell_frac_tn_val, mode = "norm")
y_pred_frac_bn        = rescale(cell_frac_bn, mode = "norm")

perf_test_frac_tn     = pd.DataFrame({
    ctp_: classifier_performance(resp_pCR_tn, pred_) 
    for ctp_, pred_ in y_pred_frac_tn.items()}).T

perf_test_frac_tn_val = pd.DataFrame({
    ctp_: classifier_performance(resp_pCR_tn_val, pred_) 
    for ctp_, pred_ in y_pred_frac_tn_val.items()}).T

perf_test_frac_bn     = pd.DataFrame({
    ctp_: classifier_performance(resp_pCR_bn, pred_) 
    for ctp_, pred_ in y_pred_frac_bn.items()}).T


## get model orders.
mdl_ord = pd.concat(                                                           # order by mean abundance
    [y_pred_frac_tn, y_pred_frac_tn_val, y_pred_frac_bn], axis = 0).mean(
    axis = 0).sort_values(
    ascending = False).index.tolist()
mdl_names = [mdl.replace("_", "\n") for mdl in mdl_ord]


## for different T-cell subtypes.
y_pred_frac_tn11        = rescale(cell_frac_tn11, mode = "norm")
y_pred_frac_tn_val11    = rescale(cell_frac_tn_val11, mode = "norm")
y_pred_frac_bn11        = rescale(cell_frac_bn11, mode = "norm")

perf_test_frac_tn11     = pd.DataFrame({
    ctp_: classifier_performance(resp_pCR_tn, pred_) 
    for ctp_, pred_ in y_pred_frac_tn11.items()}).T

perf_test_frac_tn_val11 = pd.DataFrame({
    ctp_: classifier_performance(resp_pCR_tn_val, pred_) 
    for ctp_, pred_ in y_pred_frac_tn_val11.items()}).T

perf_test_frac_bn11     = pd.DataFrame({
    ctp_: classifier_performance(resp_pCR_bn, pred_) 
    for ctp_, pred_ in y_pred_frac_bn11.items()}).T


## get model orders.
mdl_names11 = [mdl.replace("_", " ").replace("+", "$^+$") 
               for mdl in cell_types11]


#%% generate data for visualization - supp. fig. 5.

def get_pred_data(y_true, y_pred, models, althyp = "greater"):
    ## build score dataframe.
    y_true     = y_true.rename(
        index = "Response").replace(
        to_replace = {1: "R", 0: "NR"}).infer_objects(
        copy = False)
    
    score_data = pd.concat(
        [y_true, y_pred[models]], axis = 1).melt(
        id_vars = "Response", var_name = "model", value_name = "score")
    
    ## perform R vs. NR wilcoxon test.
    score_stat = score_data.groupby(
        by = "model", sort = False).apply(
        lambda df: pd.Series(mannwhitneyu(
            df.score[df.Response.eq("R")], df.score[df.Response.eq("NR")], 
            alternative = althyp, nan_policy = "omit"), 
            index = ["U1", "pval"]), 
        include_groups = False)
    
    score_stat["annot"] = score_stat.pval.map(
        lambda p: ("***" if (p <= 0.001) else "**" if (p <= 0.01) else 
                   "*" if (p <= 0.05) else "ns"))
    
    return score_data, score_stat


## prepare data for panels A-B.
alt_hyp = "two-sided"

fig_dataS5A1, fig_statS5A1 = get_pred_data(y_true = resp_pCR_tn, 
                                           y_pred = y_pred_frac_tn, 
                                           models = mdl_ord, 
                                           althyp = alt_hyp)

fig_dataS5A2, fig_statS5A2 = get_pred_data(y_true = resp_pCR_tn_val, 
                                           y_pred = y_pred_frac_tn_val, 
                                           models = mdl_ord, 
                                           althyp = alt_hyp)

fig_dataS5A3, fig_statS5A3 = get_pred_data(y_true = resp_pCR_bn, 
                                           y_pred = y_pred_frac_bn, 
                                           models = mdl_ord, 
                                           althyp = alt_hyp)


fig_dataS5B1, fig_statS5B1 = get_pred_data(y_true = resp_pCR_tn, 
                                           y_pred = y_pred_frac_tn11, 
                                           models = cell_types11, 
                                           althyp = alt_hyp)

fig_dataS5B2, fig_statS5B2 = get_pred_data(y_true = resp_pCR_tn_val, 
                                           y_pred = y_pred_frac_tn_val11, 
                                           models = cell_types11, 
                                           althyp = alt_hyp)

fig_dataS5B3, fig_statS5B3 = get_pred_data(y_true = resp_pCR_bn, 
                                           y_pred = y_pred_frac_bn11, 
                                           models = cell_types11, 
                                           althyp = alt_hyp)

fig_dataS5A = pd.concat([fig_dataS5A1, fig_dataS5A2, fig_dataS5A3], axis = 1, 
                        keys = ds_info.label.values)
fig_dataS5B = pd.concat([fig_dataS5B1, fig_dataS5B2, fig_dataS5B3], axis = 1, 
                        keys = ds_info.label.values)
fig_statS5A = pd.concat([fig_statS5A1, fig_statS5A2, fig_statS5A3], axis = 1, 
                        keys = ds_info.label.values)
fig_statS5B = pd.concat([fig_statS5B1, fig_statS5B2, fig_statS5B3], axis = 1, 
                        keys = ds_info.label.values)


## prepare data for panels C-D.
fig_dataS5C = pd.concat({
    met: pd.concat(
        [perf_test_frac_tn[met], perf_test_frac_tn_val[met], 
         perf_test_frac_bn[met]], axis = 1, 
        keys = ds_info.label.values).loc[
        mdl_ord].reset_index(
        names = "model")
    for met in ["AUC", "AP"]}, axis = 1)

fig_dataS5D = pd.concat(
    [perf_test_frac_tn11, perf_test_frac_tn_val11, perf_test_frac_bn11], 
    axis = 0).reset_index(
    names = "model")
fig_dataS5D.insert(
    loc = 1, column = "cohort", value = np.repeat(
        ds_info.label, repeats = len(cell_types11)).tolist())
fig_dataS5D.sort_values(by = "model", ascending = True, inplace = True)


#%% generate supp. fig. 5-I.

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

## violin plots.
fig_colorsS5 = [colors[k] for k in [0, 1, -1]]
fig_ylimS5   = 0.45
fig_plocS5   = 0.45

figS5_I, axS5_I = plt.subplots(figsize = (18, 8), nrows = 3, ncols = 2, 
                               sharex = False, sharey = True, 
                               width_ratios = [0.8, 0.2])
axS5_I = dict(zip(["A1", "B1", "A2", "B2", "A3", "B3"], axS5_I.ravel()))

## make violins.
for k, ds in enumerate(ds_info.label.values, start = 1):
    ax = [axS5_I[f"A{k}"], axS5_I[f"B{k}"]]
    
    ax[0] = make_violinplot(data = fig_dataS5A[ds].dropna(), x = "model", 
                            y = "score", hue = "Response", 
                            stats = fig_statS5A[ds], order = mdl_ord, 
                            hue_order = ["R", "NR"], width = 0.4, 
                            inner = "quart", vnorm = "count", split = True, 
                            gap = 0.15, dodge = True, statloc = fig_plocS5, 
                            statline = False, colors = fig_colorsS5, 
                            legend_vert = True, legend_out = True, 
                            legend_title = "Response", ax = ax[0])
    
    ax[1] = make_violinplot(data = fig_dataS5B[ds].dropna(), x = "model", 
                            y = "score", hue = "Response", 
                            stats = fig_statS5B[ds], order = cell_types11, 
                            hue_order = ["R", "NR"], width = 0.4, 
                            inner = "quart", vnorm = "count", split = True, 
                            gap = 0.15, dodge = True, statloc = fig_plocS5, 
                            statline = False, colors = fig_colorsS5, 
                            legend_vert = True, legend_out = True, 
                            legend_title = "Response", ax = ax[1])
    
    ## format ticks & legends.
    if k < 3:
        [ax_.set_xticks(ticks = range(len(ax_.get_xticks())), 
                        labels = [""] * len(ax_.get_xticks()))
         for ax_ in ax];
    else:
        [ax_.set_xticks(ticks = range(len(xt_)), labels = xt_, rotation = 45, 
                        ha = "right", va = "top", ma = "center", 
                        position = (0, 0.02), **label_fonts) 
         for ax_, xt_ in zip(ax, [mdl_names, mdl_names11])];
    
    [ax_.set_yticks(np.arange(0, 1.5, 0.5)) for ax_ in ax];
    
    ax[0].legend([ ], [ ], frameon = False);
    ax[1].legend([ ], [ ], frameon = False) if k != 2 else None
    
    ax[0].set_title(ds, x = 0.7, y = 1.18, **legend_fonts["title"]);
    

## format labels.
figS5_I.text(x = 0.010, y = 0.95, s = "A", **panel_fonts);
figS5_I.text(x = 0.725, y = 0.95, s = "B", **panel_fonts);
figS5_I.supylabel("Cell abundance (rescaled)", x = 0.015, y = 0.53, 
                  ha = "center", va = "center", ma = "center", **label_fonts);

figS5_I.tight_layout(h_pad = 2, w_pad = 2)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS5_I = "all_predictions_chemo_cell_abundance_v2.pdf"
    figS5_I.savefig(fig_path + fig_fileS5_I, dpi = 600)
    print(fig_fileS5_I)


#%% generate supp. fig. 5-II.

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
fig_colorsS5 = [colors[k] for k in [4, 3, 5, -1]]
fig_thetaS5  = RadarChart(num_vars = len(mdl_names), frame = "circle")
fig_baseS5   = [0.5] * len(fig_thetaS5)
fig_ticksS5  = pad_radar_ticks(ticks = mdl_names, pads = [12, 8])


figS5_II, axS5_II = plt.subplots(figsize = (14, 5), nrows = 1, ncols = 2, 
                                 subplot_kw = {"projection": "radar"})
axS5_II = dict(zip(["C1", "C2"], axS5_II))

## make radars.
for ds, clr in zip(ds_info.label, fig_colorsS5):
    axS5_II["C1"] = make_radar_lines(data = fig_dataS5C["AUC"][ds], 
                                     theta = fig_thetaS5, labels = fig_ticksS5, 
                                     color = clr, alpha = 0.4, ls = "-", lw = 2, 
                                     ms = 8, ax = axS5_II["C1"])
    
    axS5_II["C2"] = make_radar_lines(data = fig_dataS5C["AP"][ds], 
                                     theta = fig_thetaS5, labels = fig_ticksS5, 
                                     color = clr, alpha = 0.4, ls = "-", lw = 2, 
                                     ms = 8, ax = axS5_II["C2"])
    
    ## format ticks & titles.
    axS5_II["C1"].set_rlim([0.10, 0.95]);   axS5_II["C2"].set_rlim([0.0, 0.75]);
    [ax_.set_title(ttl_, y = 1.16, **title_fonts) 
     for ax_, ttl_ in zip(axS5_II.values(), ["AUC", "AP"])];
    
## format legends & labels.
axS5_II["C2"].legend(labels = ds_info.label, loc = (1.24, 0.4), 
                     title = "Dataset", prop = legend_fonts["item"], 
                     title_fontproperties = legend_fonts["title"])
figS5_II.text(x = 0.01, y = 0.9, s = "C", **panel_fonts);

figS5_II.tight_layout(h_pad = 0, w_pad = 4)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS5_II = "all_chemo_abundance_response_association.pdf"
    figS5_II.savefig(fig_path + fig_fileS5_II, dpi = 600)
    print(fig_fileS5_II)


#%% generate supp. fig. 5-III.

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

fontdict = {"label": dict(fontsize = 12, fontweight = "regular"), 
            "title": dict(fontsize = 16, fontweight = "semibold"), 
            "super": dict(fontsize = 20, fontweight = "bold"),
            "plabel": dict(fontsize = 36, fontweight = "bold")}

## grouped barplots.
fig_colorsS5 = [colors[k] for k in [4, 3, 5]]

figS5_III, axS5_III = plt.subplots(figsize = (8, 5), nrows = 2, ncols = 1, 
                                   sharex = True, sharey = False)
axS5_III = dict(zip(["D1", "D2"], axS5_III))

for k, (ax, met) in enumerate(zip(axS5_III.values(), ["AUC", "AP"])):
    ax = make_barplot3(data = fig_dataS5D, x = "model", y = met, hue = "cohort", 
                       width = 0.6, colors = fig_colorsS5, lw = 2, 
                       bar_labels = True, xlabels = mdl_names11, title = met, 
                       legend_title = "Dataset", fontdict = fontdict, ax = ax)
    
    ## format ticks & labels.
    ax.set_ylim([0, 1]);
    ax.set_yticks(np.arange(0, 1.25, 0.25));
    ax.yaxis.set_major_formatter("{x:0.2f}");
    if k < 1:
        ax.legend([ ], [ ], frameon = False);
    else:
        ax.get_legend().set(bbox_to_anchor = (1.1, 0.6, 0.4, 0.4), 
                            frame_on = False)

figS5_III.text(x = -0.025, y = 0.95, s = "D", **fontdict["plabel"]);

figS5_III.tight_layout(h_pad = 2, w_pad = 0)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS5_III = "all_chemo_Tcells_abundance_response_association.pdf"
    figS5_III.savefig(fig_path + fig_fileS5_III, dpi = 600)
    print(fig_fileS5_III)

