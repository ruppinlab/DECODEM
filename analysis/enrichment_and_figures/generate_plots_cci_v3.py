#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 00:02:10 2024

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
import matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
# from miscellaneous import date_time, tic, write_xlsx
from functools import reduce
from operator import add
from math import ceil
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from _functions import (classifier_performance, make_boxplot2, make_barplot2, 
                        make_barplot1)


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


def make_roc_plot(data, label, pred, group, ax, title = None, 
                  fill = False, alpha = 0.4, colors = None, fontdict = None):
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
    
    
    ## main plot.
    for (grp, data_grp), clr in zip(data.groupby(by = group, sort = False), 
                                    colors):
        roc_grp = RocCurveDisplay.from_predictions(
            y_true = data_grp[label], y_pred = data_grp[pred], 
            drop_intermediate = True, pos_label = 1, 
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

    lgnd = ax.legend(loc = (1.06, 0.25), title = group, 
                     prop = fontdict["label"], 
                     title_fontproperties = fontdict["title"])
    for lgndtxt in lgnd.get_texts():
        lgndtxt.set_text( lgndtxt.get_text().replace(") (", ", ") )
        lgndtxt.set_text( lgndtxt.get_text().replace("Chance level", "Random") )
    
    ax.set_title(title, wrap = True, y = 1.02, **fontdict["title"]);
    
    return ax


def make_lollipop_plot(data, x, y, ax, size = 150, yticks = "left", 
                       colors = None, title = None, xlabel = None, 
                       offset = 0.035, fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {
            "label": {"family": "sans", "size": 12, "weight": "regular"}, 
            "title": {"family": "sans", "size": 16, "weight": "bold"}, 
            "super": {"family": "sans", "size": 20, "weight": "bold"}}
    
    if colors is None:
        colors   = ["#B075D0", "#708090", "#000000"]
    
    lineprop = {"linestyle": "-", "linewidth": 4, "color": colors[-2], 
                "alpha": 0.9}
    baseprop = {"linestyle": "--", "linewidth": 1.5, "color": colors[-1], 
                "alpha": 0.8}
    mrkrprop = {"marker": "o", "linewidths": 1.5, "color": colors[0], 
                "edgecolor": colors[-1], "alpha": 0.8}
    
    ## add offset to horizontal lines [to skip overlap with the dots].
    data[f"{x}_offset"] = data[x].map(
        lambda val: val - (offset if (val > 0) else -offset))
    
    
    ## main plot.
    ax.hlines(data = data, y = y, xmin = 0, xmax = f"{x}_offset", **lineprop)
    ax.scatter(data = data, x = x, y = y, s = size, **mrkrprop)
    ax.axvline(x = 0, ymin = 0, ymax = 1, **baseprop)
    ax.invert_yaxis();                                                         # top-to-bottom by importance
    sns.despine(ax = ax, offset = 0, trim = False, top = False, right = False);
    # sns.despine(ax = ax, offset = 0, trim = False, 
    #             left = (yticks.lower() == "right"), 
    #             right = (yticks.lower() == "left"));
    
    
    ## format axis ticks & legends.
    if yticks.lower() == "right":                                              # put yticks to the right
        ax.invert_xaxis()
        plt.tick_params(axis = "y", left = False, right = True, 
                        labelleft = False, labelright = True);
    
    ax.set_xticks(np.arange(-0.9, 1.0, 0.3));
    ax.set_xlim([-0.95, 0.95])
    ax.set_xlabel(xlabel, labelpad = 8, **fontdict["label"]);
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    
    ax.set_title(title, wrap = True, y = 1.02, **fontdict["title"]);
    
    return ax


#%% read data.

data_path = "../../data/TransNEO/transneo_analysis/mdl_data/"
data_file = ["transneo_lirics_predictions_chemo_filteredCCI_th0.99_RF_allfeatures_5foldCV_25Mar2023.pkl", 
             "tn_valid_lirics_predictions_chemo_filteredCCI_th0.99_RF_allfeatures_3foldCV_25Mar2023.pkl", 
             "brightness_lirics_predictions_chemo_filteredCCI_th0.99_RF_allfeatures_3foldCV_25Mar2023.pkl", 
             "transneo_predictions_chemo_th0.99_ENS2_25features_5foldCV_20Mar2023.pkl", 
             "tn_valid_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl", 
             "brightness_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl", 
             "transneo_lirics_feature_importance_chemo_filteredCCI_th0.99_RF_allfeatures_5foldCV_25Mar2023.xlsx", 
             "tn_valid_lirics_feature_importance_chemo_filteredCCI_th0.99_RF_allfeatures_3foldCV_25Mar2023.xlsx", 
             "brightness_lirics_feature_importance_chemo_filteredCCI_th0.99_RF_allfeatures_3foldCV_25Mar2023.xlsx", 
             "transneo_lirics_predictions_chemo_allCCI_RF_allfeatures_5foldCV_25Mar2023.pkl", 
             "tn_valid_lirics_predictions_chemo_allCCI_RF_allfeatures_3foldCV_25Mar2023.pkl", 
             "brightness_lirics_predictions_chemo_allCCI_RF_allfeatures_3foldCV_25Mar2023.pkl"]


## cci-specific predictions.
cclr = "ramilowski"

(y_test_cci_tn, y_pred_cci_tn, th_test_cci_tn, 
 perf_test_cci_tn) = read_pkl_data(data_path + data_file[0])

(y_test_cci_tn_val, y_pred_cci_tn_val, th_test_cci_tn_val, 
 perf_test_cci_tn_val) = read_pkl_data(data_path + data_file[1])

(y_test_cci_bn, y_pred_cci_bn, th_test_cci_bn, 
 perf_test_cci_bn) = read_pkl_data(data_path + data_file[2])


## cell-type-specific predictions.
(y_test_exp_tn, y_pred_exp_tn, th_test_exp_tn, 
 perf_test_exp_tn) = read_pkl_data(data_path + data_file[3])

(y_test_exp_tn_val, y_pred_exp_tn_val, th_test_exp_tn_val, 
 perf_test_exp_tn_val) = read_pkl_data(data_path + data_file[4])

(y_test_exp_bn, y_pred_exp_bn, th_test_exp_bn, 
 perf_test_exp_bn) = read_pkl_data(data_path + data_file[5])


## cci importances.
feat_imp_tn     = pd.read_excel(
    data_path + data_file[6], header = 0, index_col = 0)

feat_imp_tn_val = pd.read_excel(
    data_path + data_file[7], header = 0, index_col = 0)

feat_imp_bn     = pd.read_excel(
    data_path + data_file[8], header = 0, index_col = 0)


## cci-specific predictions for all ccis.
(y_test_cci_all_tn, y_pred_cci_all_tn, th_test_cci_all_tn, 
 perf_test_cci_all_tn) = read_pkl_data(data_path + data_file[9])

(y_test_cci_all_tn_val, y_pred_cci_all_tn_val, th_test_cci_all_tn_val, 
 perf_test_cci_all_tn_val) = read_pkl_data(data_path + data_file[10])

(y_test_cci_all_bn, y_pred_cci_all_bn, th_test_cci_all_bn, 
 perf_test_cci_all_bn) = read_pkl_data(data_path + data_file[11])


#%% prepare data for fig. 4.

## dataset info.
ds_info = pd.DataFrame(
    [{"n": len(y), "R": sum(y == 1), "NR": sum(y == 0)} 
     for y in [y_test_cci_tn, y_test_cci_tn_val, y_test_cci_bn]], 
    index = ["TransNEO", "ARTemis + PBCP", "BrighTNess"]).reset_index(
    names = "Dataset")
ds_info["label"] = ds_info.apply(
    lambda df: f"{df.Dataset} (n = {df.n})", axis = 1).tolist()


## get data for fig. 4A.
y_test_cci_all = pd.DataFrame({
    "Response": pd.concat(
        [y_test_cci_tn, y_test_cci_tn_val, y_test_cci_bn], axis = 0).replace(
        to_replace = {1: "R", 0: "NR"}),     
    "Dataset" : reduce(add, ds_info.apply(
        lambda df: [df.label] * df.n, axis = 1)) })

y_pred_cci_all = pd.concat(
    [y_pred_cci_tn[cclr], y_pred_cci_tn_val[cclr], y_pred_cci_bn[cclr]], 
    axis = 0).rename(
    index = "score")

fig_data4A    = pd.concat([y_test_cci_all, y_pred_cci_all], axis = 1)

# fig_stat4A    = pd.DataFrame({
#     ds: mannwhitneyu(*fig_data4A.groupby(
#         by = "Response").apply(
#         lambda df: df[df.Dataset.eq(ds)]["score"].tolist()).sort_index(
#         ascending = False), alternative = "greater") 
#     for ds in ds_info.label}, index = ["U1", "pval"]).T
fig_stat4A = fig_data4A.groupby(
    by = "Dataset", sort = False).apply(
    lambda df: pd.Series(mannwhitneyu(
        df.score[df.Response.eq("R")], df.score[df.Response.eq("NR")], 
        alternative = "greater", nan_policy = "omit"), 
        index = ["U1", "pval"]), 
    include_groups = False)
fig_stat4A["annot"] = fig_stat4A.pval.map(
    lambda p: ("***" if (p <= 0.001) else "**" if (p <= 0.01) else 
               "*" if (p <= 0.05) else "ns"))


## get data for fig. 4B.
# fig_ctp_4B = perf_test_exp_tn.AUC.idxmax()
# fig_data4B = pd.DataFrame({
#     "Response"  : pd.concat([y_test_cci_tn, y_test_exp_tn], 
#                             axis = 0), 
#     "Prediction": pd.concat([y_pred_cci_tn[cclr], 
#                               y_pred_exp_tn["Bulk"]], axis = 0), 
#     "model"     : np.repeat(["CCIs", "Bulk"], 
#                             repeats = len(y_test_cci_tn)) })
fig_data4B = fig_data4A.replace(
    to_replace = {"Response": {"R": 1, "NR": 0}}).infer_objects(
    copy = False)


## get data for fig. 4C-D.
def get_cci_data(feat_imp, n_top = 10):
    cci_imp = pd.DataFrame({
        "CCI": feat_imp.index.map(
            lambda x: x.replace("-", "$-$").replace(
                "::", "$::$").replace("_", " ")), 
        "MDI": feat_imp.apply(
            lambda df: df.MDI * (1 if df.Direction > 0 else -1), axis = 1) })
    
    cci_imp = cci_imp.reset_index(drop = True)[:n_top]
    
    return cci_imp


## feature importances.
n_top      = 10                                                                # keep only top CCIs
fig_data4C = get_cci_data(feat_imp = feat_imp_tn_val, n_top = n_top)
fig_data4D = get_cci_data(feat_imp = feat_imp_bn, n_top = n_top)


#%% make fig. 4-I.

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
fig4_I, ax4_I = plt.subplots(figsize = (18, 8), dpi = 600, nrows = 2, 
                             ncols = 2, height_ratios = [0.8, 1], 
                             width_ratios = [1, 1])
ax4_I = dict(zip(list("ABCD"), ax4_I.ravel()))

fig_llocs4 = [[0.18, 0.48], [0.96, 0.51]]

## make violins.
fig_ylim4A = 0.3;   fig_ploc4A = 0.2

ax4_I["A"] = make_violinplot(data = fig_data4A, x = "Dataset", y = "score", 
                             hue = "Response", stats = fig_stat4A, 
                             hue_order = ["R", "NR"], inner = "quart", 
                             split = True, dodge =  True, statloc = fig_ploc4A, 
                             statline = False, title = "Prediction score", 
                             legend_vert = True, legend_out = False, 
                             legend_title = "Response", ax = ax4_I["A"])
ax4_I["A"].set_ylim([0 - fig_ylim4A, 1 + fig_ylim4A]);
ax4_I["A"].set_yticks(ticks  = np.arange(0, 1.2, 0.2), 
                      labels = np.arange(0, 1.2, 0.2));
ax4_I["A"].yaxis.set_major_formatter("{x:0.1f}");
ax4_I["A"].set_xticks(ticks  = range(len(ds_info)), 
                      labels = ds_info.label.map(
                          lambda x: x.replace(" (", "\n(")));
ax4_I["A"].tick_params(axis = "both", labelsize = 12);
ax4_I["A"].get_legend().set_bbox_to_anchor([-0.50, 0.25, 0.4, 0.4]);
fig4_I.text(x = fig_llocs4[0][0], y = fig_llocs4[1][0], s = "A", 
            **panel_fonts);


## make roc curves.
fig_colors4B = [colors[4], colors[3], colors[5], colors[-1]]
# fig_colors4B = [colors[3], colors[4], colors[-3]]

ax4_I["B"]   = make_roc_plot(data = fig_data4B, label = "Response", 
                             pred = "score", group = "Dataset", 
                             colors = fig_colors4B, fill = True, 
                             alpha = 0.15, title = "Model performance", 
                             ax = ax4_I["B"])
fig4_I.text(x = fig_llocs4[0][1], y = fig_llocs4[1][0], s = "B", 
            **panel_fonts);


## make lollipop plots.
fig_size4CD   = 100
fig_off4CD    = 0.035
# fig_xlbl4CD   = "Mean decrease in Gini impurity (signed)"
fig_xlbl4CD   = "Feature importance (signed)"
fig_ttls4CD   = ds_info.label.iloc[1:].tolist()
fig_colors4CD = [colors[3], colors[-3], colors[-1]]
# fig_colors4C  = [colors[3], colors[-3], colors[-1]]
# fig_colors4D  = [colors[5], colors[-3], colors[-1]]

ax4_I["C"] = make_lollipop_plot(data = fig_data4C, x = "MDI", y = "CCI", 
                                size = fig_size4CD, offset = fig_off4CD, 
                                colors = fig_colors4CD, title = fig_ttls4CD[0], 
                                xlabel = fig_xlbl4CD, yticks = "left", 
                                ax = ax4_I["C"])
# ax4_I["C"].set_xlim([-0.85, 0.85]);
fig4_I.text(x = fig_llocs4[0][0], y = fig_llocs4[1][1], s = "C", 
            **panel_fonts);

ax4_I["D"] = make_lollipop_plot(data = fig_data4D, x = "MDI", y = "CCI", 
                                size = fig_size4CD, offset = fig_off4CD, 
                                colors = fig_colors4CD, title = fig_ttls4CD[1], 
                                xlabel = fig_xlbl4CD, yticks = "right", 
                                ax = ax4_I["D"])
# ax4_I["D"].set_xlim([-0.80, 0.80]);
# fig4_I.text(x = 0.49, y = 0.18, s = "Cell-cell interactions", 
#             rotation = 90, **label_fonts)
fig4_I.text(x = fig_llocs4[0][1], y = fig_llocs4[1][1], s = "D", 
            **panel_fonts);


fig4_I.tight_layout(h_pad = 2, w_pad = 4)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file4_I = "all_predictions_cci_validation_chemo_th0.99_RF_allfeatures_5foldCV.pdf"
    fig4_I.savefig(fig_path + fig_file4_I, dpi = "figure")


#%% prepare data for supp. fig. 6.

## get data for supp. fig. 6A.
y_test_cci_all_all = pd.DataFrame({
    "Response": pd.concat(
        [y_test_cci_all_tn, y_test_cci_all_tn_val, y_test_cci_all_bn], 
        axis = 0).replace(
        to_replace = {1: "R", 0: "NR"}),     
    "Dataset" : reduce(add, ds_info.apply(
        lambda df: [df.label] * df.n, axis = 1)) })

y_pred_cci_all_all = pd.concat(
    [y_pred_cci_all_tn[cclr], y_pred_cci_all_tn_val[cclr], 
     y_pred_cci_all_bn[cclr]], axis = 0).rename(
    index = "score")

fig_dataS6A = pd.concat([y_test_cci_all_all, y_pred_cci_all_all], axis = 1)

fig_statS6A = fig_dataS6A.groupby(
    by = "Dataset", sort = False).apply(
    lambda df: pd.Series(mannwhitneyu(
        df.score[df.Response.eq("R")], df.score[df.Response.eq("NR")], 
        alternative = "greater", nan_policy = "omit"), 
        index = ["U1", "pval"]), 
    include_groups = False)
fig_statS6A["annot"] = fig_statS6A.pval.map(
    lambda p: ("***" if (p <= 0.001) else "**" if (p <= 0.01) else 
               "*" if (p <= 0.05) else "ns"))


## get data for supp. fig. 6B.
fig_dataS6B = fig_dataS6A.replace(
    to_replace = {"Response": {"R": 1, "NR": 0}}).infer_objects(
    copy = False)


## get data for supp. fig. 6C.
fig_dataS6C = pd.DataFrame([
    {"All cell types": perf_test_cci_all_tn.loc[cclr, "AP"], 
     "Top cell types": perf_test_cci_tn.loc[cclr, "AP"]}, 
    {"All cell types": perf_test_cci_all_tn_val.loc[cclr, "AP"], 
     "Top cell types": perf_test_cci_tn_val.loc[cclr, "AP"]}, 
    {"All cell types": perf_test_cci_all_bn.loc[cclr, "AP"], 
     "Top cell types": perf_test_cci_bn.loc[cclr, "AP"]}], 
    index = ds_info.label).reset_index(
    names = "Dataset").melt(
    id_vars = "Dataset", var_name = "CCI_list", value_name = "score")


## get data for supp. fig. 6D.
fig_dataS6D = pd.DataFrame([
    {"All cell types": perf_test_cci_all_tn.loc[cclr, "DOR"], 
     "Top cell types": perf_test_cci_tn.loc[cclr, "DOR"]}, 
    {"All cell types": perf_test_cci_all_tn_val.loc[cclr, "DOR"], 
     "Top cell types": perf_test_cci_tn_val.loc[cclr, "DOR"]}, 
    {"All cell types": perf_test_cci_all_bn.loc[cclr, "DOR"], 
     "Top cell types": perf_test_cci_bn.loc[cclr, "DOR"]}], 
    index = ds_info.label).reset_index(
    names = "Dataset").melt(
    id_vars = "Dataset", var_name = "CCI_list", value_name = "score")


#%% make supp. fig. 6-I.

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

fontdict     = {
    "label": {"family": "sans", "size": 12, "weight": "regular"}, 
    "title": {"family": "sans", "size": 16, "weight": "bold"}, 
    "super": {"family": "sans", "size": 20, "weight": "bold"}}


## all plots.
figS6_I, axS6_I = plt.subplots(figsize = (16, 7), dpi = 600, nrows = 2, 
                               ncols = 2, height_ratios = [1, 1], 
                               width_ratios = [1, 1])
axS6_I = dict(zip(list("ABCD"), axS6_I.ravel()))

fig_llocsS6 = [[0.09, 0.41], [0.96, 0.51]]

## make violins.
fig_ylimS6A = 0.3;   fig_plocS6A = 0.2

axS6_I["A"] = make_violinplot(data = fig_dataS6A, x = "Dataset", y = "score", 
                              hue = "Response", stats = fig_statS6A, 
                              hue_order = ["R", "NR"], inner = "quart", 
                              split = True, dodge =  True, 
                              statloc = fig_plocS6A, statline = False, 
                              title = "Prediction score", legend_vert = True, 
                              legend_out = False, legend_title = "Response", 
                              ax = axS6_I["A"])
axS6_I["A"].set_ylim([0 - fig_ylimS6A, 1 + fig_ylimS6A]);
axS6_I["A"].set_yticks(ticks  = np.arange(0, 1.2, 0.2), 
                       labels = np.arange(0, 1.2, 0.2));
axS6_I["A"].yaxis.set_major_formatter("{x:0.1f}");
axS6_I["A"].set_xticks(ticks  = range(len(ds_info)), 
                       labels = ds_info.label.map(
                          lambda x: x.replace(" (", "\n(")));
axS6_I["A"].tick_params(axis = "both", labelsize = 12);
axS6_I["A"].get_legend().set_bbox_to_anchor([-0.50, 0.25, 0.4, 0.4]);
figS6_I.text(x = fig_llocsS6[0][0], y = fig_llocsS6[1][0], s = "A", 
            **panel_fonts);


## make roc curves.
fig_colorsS6B = [colors[4], colors[3], colors[5], colors[-1]]
# fig_colorsS6B = [colors[3], colors[4], colors[-3]]

axS6_I["B"]   = make_roc_plot(data = fig_dataS6B, label = "Response", 
                              pred = "score", group = "Dataset", 
                              colors = fig_colorsS6B, fill = True, 
                              alpha = 0.15, title = "Model performance", 
                              ax = axS6_I["B"])
figS6_I.text(x = fig_llocsS6[0][1], y = fig_llocsS6[1][0], s = "B", 
             **panel_fonts);


## make barplots.
fig_colorsS6C = [colors[4], colors[3]]

axS6_I["C"]   = make_barplot2(data = fig_dataS6C, x = "Dataset", y = "score", 
                              hue = "CCI_list", width = 0.5, lw = 2, 
                              colors = fig_colorsS6C, bar_label_align = False, 
                              title = "", xlabels = ds_info.label.map(
                                  lambda x: x.replace(" (", "\n(")), 
                              legend = False, offset = 0, ax = axS6_I["C"], 
                              fontdict = fontdict)
axS6_I["C"].set_ylim([-0.04, 1.04]);
axS6_I["C"].set_title("CCIs: AP", **fontdict["title"]);
figS6_I.text(x = fig_llocsS6[0][0], y = fig_llocsS6[1][1], s = "C", 
            **panel_fonts) 


axS6_I["D"]   = make_barplot2(data = fig_dataS6D, x = "Dataset", y = "score", 
                              hue = "CCI_list", width = 0.5, lw = 2, 
                              colors = fig_colorsS6C, bar_label_align = True, 
                              title = "", xlabels = ds_info.label.map(
                                  lambda x: x.replace(" (", "\n(")), 
                              legend = True, legend_title = "CCIs", offset = 0, 
                              ax = axS6_I["D"], fontdict = fontdict)
axS6_I["D"].set_ylim([-1, 27]);
axS6_I["D"].set_title("CCIs: Odds ratio", **fontdict["title"]);
axS6_I["D"].get_legend().set(bbox_to_anchor = (1.06, 0.45, 0.4, 0.4));
figS6_I.text(x = fig_llocsS6[0][1], y = fig_llocsS6[1][1], s = "D", 
            **panel_fonts) 

figS6_I.tight_layout(h_pad = 2, w_pad = 4)
plt.show()


## save figures.
if svdat:
    fig_path = data_path + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS6_I = "all_predictions_cci_all_cell_types_chemo_th0.99_RF_allfeatures_5foldCV.pdf"
    figS6_I.savefig(fig_path + fig_fileS6_I, dpi = "figure")

