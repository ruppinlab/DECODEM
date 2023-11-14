#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:29:35 2023

@author: dhrubas2
"""

## set up necessary directories/paths.
_wpath_ = "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
_mpath_ = "miscellaneous/py/"

## load necessary packages.
import os, sys
sys.path.append(_wpath_);       os.chdir(_wpath_)                              # current path
if _mpath_ not in sys.path:
    sys.path.append(_mpath_)                                                   # to load miscellaneous

import numpy as np, pandas as pd, pickle
import matplotlib.pyplot as plt, seaborn as sns
from miscellaneous import date_time, tic, write_xlsx
from scipy.stats import mannwhitneyu
from machine_learning._functions import (
    make_barplot2, make_boxplot2, add_stat, make_importance_plot)


#%% read all data for fig 4.

data_path = "../data/TransNEO/transneo_analysis/mdl_data/"
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


## CCI-specific predictions.
with open(data_path + data_file[0], "rb") as file:
    data_obj     = pickle.load(file)
    y_test_tn    = data_obj["label"]
    y_pred_tn    = data_obj["pred"]
    th_test_tn   = data_obj["th"]
    perf_test_tn = data_obj["perf"]
    del data_obj


with open(data_path + data_file[1], "rb") as file:
    data_obj         = pickle.load(file)
    y_test_tn_val    = data_obj["label"]
    y_pred_tn_val    = data_obj["pred"]
    th_test_tn_val   = data_obj["th"]
    perf_test_tn_val = data_obj["perf"]
    del data_obj


with open(data_path + data_file[2], "rb") as file:
    data_obj     = pickle.load(file)
    y_test_bn    = data_obj["label"]
    y_pred_bn    = data_obj["pred"]
    th_test_bn   = data_obj["th"]
    perf_test_bn = data_obj["perf"]
    del data_obj


## cell-type-specific predictions.
with open(data_path + data_file[3], "rb") as file:
    data_obj         = pickle.load(file)
    y_test_exp_tn    = data_obj["label"]
    y_pred_exp_tn    = data_obj["pred"]
    th_test_exp_tn   = data_obj["th"]
    perf_test_exp_tn = data_obj["perf"]
    del data_obj


with open(data_path + data_file[4], "rb") as file:
    data_obj             = pickle.load(file)
    y_test_exp_tn_val    = data_obj["label"]
    y_pred_exp_tn_val    = data_obj["pred"]
    th_test_exp_tn_val   = data_obj["th"]
    perf_test_exp_tn_val = data_obj["perf"]
    del data_obj


with open(data_path + data_file[5], "rb") as file:
    data_obj         = pickle.load(file)
    y_test_exp_bn    = data_obj["label"]
    y_pred_exp_bn    = data_obj["pred"]
    th_test_exp_bn   = data_obj["th"]
    perf_test_exp_bn = data_obj["perf"]
    del data_obj


## CCI importance lists.
read_data = lambda file: pd.read_excel(file, header = 0, index_col = 0)

cclr           = "ramilowski"
featimp_tn     = read_data(data_path + data_file[6])
featimp_tn_val = read_data(data_path + data_file[7])
featimp_bn     = read_data(data_path + data_file[8])


## CCI-specific predictions for all CCIs.
with open(data_path + data_file[9], "rb") as file:
    data_obj      = pickle.load(file)
    y_test_tn0    = data_obj["label"]
    y_pred_tn0    = data_obj["pred"]
    th_test_tn0   = data_obj["th"]
    perf_test_tn0 = data_obj["perf"]
    del data_obj


with open(data_path + data_file[10], "rb") as file:
    data_obj          = pickle.load(file)
    y_test_tn_val0    = data_obj["label"]
    y_pred_tn_val0    = data_obj["pred"]
    th_test_tn_val0   = data_obj["th"]
    perf_test_tn_val0 = data_obj["perf"]
    del data_obj


with open(data_path + data_file[11], "rb") as file:
    data_obj      = pickle.load(file)
    y_test_bn0    = data_obj["label"]
    y_pred_bn0    = data_obj["pred"]
    th_test_bn0   = data_obj["th"]
    perf_test_bn0 = data_obj["perf"]
    del data_obj


#%% generate data for fig. 4 - transneo.

## panel A.
fig_data4A = pd.DataFrame({
    "score": pd.concat([y_pred_tn[cclr], y_pred_tn_val[cclr], y_pred_bn[cclr]]), 
    "response": pd.concat([y_test_tn, y_test_tn_val, y_test_bn]).replace({
        1: "R", 0: "NR"}), 
    "cohort": (["TransNEO"] * len(y_test_tn) + 
               ["ARTemis + PBCP"] * len(y_test_tn_val) + 
               ["BrighTNess"] * len(y_test_bn)) })

fig_stat4A = pd.DataFrame({
    set_: mannwhitneyu(*fig_data4A.groupby("response").apply(
        lambda df: df[df["cohort"] == set_]["score"].tolist()).sort_index(
            ascending = False), alternative = "greater") \
    for set_ in fig_data4A["cohort"].unique()}, index = ["U1", "pval"]).T
fig_stat4A["annot"] = fig_stat4A.pval.apply(
    lambda p: "***" if (p <= 0.001) else "**" if (p <= 0.01) \
        else "*" if (p <= 0.05) else "ns")

fig_xticks4A = fig_data4A["cohort"].unique().tolist()


## panel B.
ctp_list = ["Cancer_Epithelial", "Bulk"]

fig_data4B = pd.concat([
    perf_test_tn.loc[cclr].rename("CCI").to_frame().T, 
    perf_test_exp_tn.loc[ctp_list]], axis = 0)[["AUC", "AP"]].rename_axis(
        "model").reset_index().melt(
            id_vars = ["model"], var_name = "metric", value_name = "score")

fig_xticks4B = ["CCIs"] + [mdl.replace("_", "\n") for mdl in ctp_list]


#%% generate data for fig. 4 - artemis + pbcp.

## panel C.
ctp_list = ["CAFs", "Bulk"]

fig_data4C = pd.concat([
    perf_test_tn_val.loc[cclr].rename("CCI").to_frame().T, 
    perf_test_exp_tn_val.loc[ctp_list]], axis = 0)[["AUC", "AP"]].rename_axis(
        "model").reset_index().melt(
            id_vars = ["model"], var_name = "metric", value_name = "score")

fig_xticks4C = ["CCIs"] + [mdl.replace("_", "\n") for mdl in ctp_list]


## panel E.
num_disp   = 10                                                                # plot top features
fig_data4E = featimp_tn_val.reset_index()[:num_disp][[
    "CCIannot", "MDI", "Direction"]]
fig_data4E["Direction"] = fig_data4E["Direction"].apply(
    lambda x: "$(+)$" if x > 0 else "$(-)$")
fig_data4E.replace(
    regex = {"CCIannot": {"-": " $-$ ", "::": "$::$", "_": " "}}, 
    inplace = True)


#%% generate data for fig. 4 - brightness.

## panel D.
ctp_list = ["Myeloid", "Bulk"]

fig_data4D = pd.concat([
    perf_test_bn.loc[cclr].rename("CCI").to_frame().T, 
    perf_test_exp_bn.loc[ctp_list]], axis = 0)[["AUC", "AP"]].rename_axis(
        "model").reset_index().melt(
            id_vars = ["model"], var_name = "metric", value_name = "score")

fig_xticks4D = ["CCIs"] + [mdl.replace("_", "\n") for mdl in ctp_list]


## panel F.
num_disp   = 10                                                                # plot top features
fig_data4F = featimp_bn.reset_index()[:num_disp][[
    "CCIannot", "MDI", "Direction"]]
fig_data4F["Direction"] = fig_data4F["Direction"].apply(
    lambda x: "$(+)$" if x > 0 else "$(-)$")
fig_data4F.replace(
    regex = {"CCIannot": {"-": " $-$ ", "::": "$::$", "_": " "}}, 
    inplace = True)


#%% generate fig. 4.

svdat = False                                                                  # set True to save figure 

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

sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 12, "xtick.major.width": 4, "ytick.major.size": 12, 
    "ytick.major.width": 4, "xtick.bottom": True, "ytick.left": True, 
    "axes.edgecolor": "#000000", "axes.linewidth": 4})

## make plot.
ax4_ylim = [-0.01, 1.08]
fig4, ax4 = plt.subplot_mosaic(
    figsize = (48, 56), 
    mosaic = [["A", "A", "A", ".", "B", "B", "B"], ["."] * 7, 
              ["C", "C", "C", ".", "D", "D", "D"], ["."] * 7, 
              [".", ".", "E", ".", "F", ".", "."], 
              [".", ".", "E", ".", "F", ".", "."]], 
    layout = None, height_ratios = [1, 0.04, 1, 0.06, 0.8, 0.8], 
    width_ratios = [1, 1, 0.5, 0.15, 0.5, 1, 1])

ax4["A"] = make_boxplot2(
    data = fig_data4A, x = "cohort", y = "score", hue = "response", 
    hue_order = ["R", "NR"], width = 0.5, title = "", legend_title = "Response", 
    xlabels = fig_xticks4A, ax = ax4["A"], fontdict = fontdict)
ax4["A"] = add_stat(
    stats = fig_stat4A, data = fig_data4A, x = "cohort", y = "score", 
    align = True, ax = ax4["A"], fontdict = fontdict)
ax4["A"].set_ylim(ax4_ylim);
ax4["A"].get_legend().set(bbox_to_anchor = (-0.4, 0.25, 0.6, 0.6));
ax4["A"].set_title("A", x = -0.05, y = 1.04, **fontdict["plabel"]);

ax4["B"] = make_barplot2(
    data = fig_data4B, x = "model", y = "score", hue = "metric", width = 0.5, 
    title = "", legend = False, xlabels = fig_xticks4B, 
    bar_label_align = False, ax = ax4["B"], fontdict = fontdict)
ax4["B"].set_ylim(ax4_ylim);
ax4["B"].set_yticklabels([""] * len(ax4["B"].get_yticks()));
ax4["B"].set_title("B", x = -0.05, y = 1.04, **fontdict["plabel"]);

ax4["C"] = make_barplot2(
    data = fig_data4C, x = "model", y = "score", hue = "metric", width = 0.5, 
    title = "", legend = False, xlabels = fig_xticks4C, 
    bar_label_align = False, ax = ax4["C"], fontdict = fontdict)
ax4["C"].set_ylim(ax4_ylim);
ax4["C"].set_title("C", x = -0.05, y = 1.04, **fontdict["plabel"]);

ax4["D"] = make_barplot2(
    data = fig_data4D, x = "model", y = "score", hue = "metric", width = 0.5, 
    title = "", legend_title = "Performance", xlabels = fig_xticks4D, 
    bar_label_align = False, ax = ax4["D"], fontdict = fontdict)
ax4["D"].set_ylim(ax4_ylim);
ax4["D"].set_yticklabels([""] * len(ax4["D"].get_yticks()));
ax4["D"].get_legend().set(bbox_to_anchor = (1.0, 0.925, 0.6, 0.6));
ax4["D"].set_title("D", x = -0.05, y = 1.04, **fontdict["plabel"]);

ax4["E"] = make_importance_plot(
    data = fig_data4E, x = "MDI", y = "CCIannot", hue = "Direction", 
    hue_order = ["$(+)$", "$(-)$"], title = "", yticks = "left", 
    xlabel = False, legend_title = "Directionality", ax = ax4["E"], 
    fontdict = fontdict)
ax4["E"].set_xticks(ticks = np.arange(0, 1.5, 0.5), 
                    labels = np.arange(0, 1.5, 0.5), **fontdict["label"]);
ax4["E"].get_legend().set(bbox_to_anchor = (-8.2, 0.35, 0.6, 0.6));
fig4.text(x = 0.098, y = 0.43, s = "E", color = "black", **fontdict["plabel"]);

ax4["F"] = make_importance_plot(
    data = fig_data4F, x = "MDI", y = "CCIannot", hue = "Direction", 
    hue_order = ["$(+)$", "$(-)$"], title = "", yticks = "right", 
    xlabel = False, legend = False, ax = ax4["F"], fontdict = fontdict)
ax4["F"].set_xticks(ticks = np.arange(0, 1.5, 0.5), 
                    labels = np.arange(0, 1.5, 0.5), **fontdict["label"]);
fig4.text(x = 0.518, y = 0.43, s = "F", color = "black", **fontdict["plabel"]);
fig4.supxlabel("Mean decrease in Gini impurity", y = 0.09, x = 0.51, 
               ha = "center", va = "baseline", **fontdict["label"]);

plt.show()


## save figures.
if svdat:
    fig_path = "../data/plots/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file4 = data_file[1].replace("tn_valid", "all").replace(
        "predictions", "scores_AUC_AP_MDI").replace(".pkl", "_v2.pdf")
    fig4.savefig(fig_path + fig_file4, dpi = "figure")


#%% prepare data for fig. s4 - all ccis.

fig_dataS4A = pd.DataFrame({
    "score": pd.concat([y_pred_tn0[cclr], y_pred_tn_val0[cclr], 
                        y_pred_bn0[cclr]]), 
    "response": pd.concat([y_test_tn0, y_test_tn_val0, y_test_bn0]).replace({
        1: "R", 0: "NR"}), 
    "cohort": (["TransNEO"] * len(y_test_tn0) + 
               ["ARTemis + PBCP"] * len(y_test_tn_val0) + 
               ["BrighTNess"] * len(y_test_bn0)) })

fig_statS4A = pd.DataFrame({
    set_: mannwhitneyu(*fig_dataS4A.groupby("response").apply(
        lambda df: df[df["cohort"] == set_]["score"].tolist()).sort_index(
            ascending = False), alternative = "greater") \
    for set_ in fig_dataS4A["cohort"].unique()}, index = ["U1", "pval"]).T
fig_statS4A["annot"] = fig_statS4A.pval.apply(
    lambda p: "***" if (p <= 0.001) else "**" if (p <= 0.01) \
        else "*" if (p <= 0.05) else "ns")


fig_dataS4B = pd.DataFrame({
    "TransNEO": perf_test_tn0.loc[cclr, ["AUC", "AP"]], 
    "ARTemis + PBCP": perf_test_tn_val0.loc[cclr, ["AUC", "AP"]], 
    "BrighTNess": perf_test_bn0.loc[cclr, ["AUC", "AP"]] }).rename_axis(
        "metric").reset_index().melt(
            id_vars = ["metric"], var_name = "cohort", value_name = "score")


fig_dataS4C = pd.DataFrame({
    "cohort": ["TransNEO", "ARTemis + PBCP", "BrighTNess"], 
    "All cell types": [perf_test_tn0.DOR[cclr], perf_test_tn_val0.DOR[cclr], 
                 perf_test_bn0.DOR[cclr]], 
    "Prominent cell types": [perf_test_tn.DOR[cclr], perf_test_tn_val.DOR[cclr], 
                      perf_test_bn.DOR[cclr]] }).melt(
        id_vars = ["cohort"], var_name = "feature_set", value_name = "score"
)


fig_dataS4D = pd.DataFrame({
    "cohort": ["TransNEO", "ARTemis + PBCP", "BrighTNess"], 
    "All cell types": [perf_test_tn0.SEN[cclr], perf_test_tn_val0.SEN[cclr], 
                 perf_test_bn0.SEN[cclr]], 
    "Prominent cell types": [perf_test_tn.SEN[cclr], perf_test_tn_val.SEN[cclr], 
                      perf_test_bn.SEN[cclr]] }).melt(
        id_vars = ["cohort"], var_name = "feature_set", value_name = "score"
)

fig_xticksS4 = [ds.replace(" + ", " + \n") \
                for ds in fig_dataS4A["cohort"].unique()]


#%% generate fig. s4.

svdat = False                                                                  # set True to save figure 

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

sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 12, "xtick.major.width": 4, "ytick.major.size": 12, 
    "ytick.major.width": 4, "xtick.bottom": True, "ytick.left": True, 
    "axes.edgecolor": "#000000", "axes.linewidth": 4})

## make plot.
axS4_ylim = [-0.02, 1.16]
figS4, axS4 = plt.subplot_mosaic(
    figsize = (52, 24), mosaic = [["A", "B"], ["C", "D"]], 
    layout = "constrained", height_ratios = [1.0, 1.0], 
    width_ratios = [1.0, 1.0], sharex = True, sharey = False)

axS4["A"] = make_boxplot2(
    data = fig_dataS4A, x = "cohort", y = "score", hue = "response", 
    hue_order = ["R", "NR"], width = 0.5, title = "", 
    legend_title = "Response", xlabels = fig_xticksS4, xrot = 0, 
    ax = axS4["A"], fontdict = fontdict)
axS4["A"] = add_stat(
    stats = fig_statS4A, data = fig_dataS4A, x = "cohort", 
    y = "score", align = True, ax = axS4["A"], fontdict = fontdict)
axS4["A"].set_ylim(axS4_ylim);
axS4["A"].get_legend().set(bbox_to_anchor = (-0.45, 0.25, 0.6, 0.6));
axS4["A"].set_title("A", x = -0.05, y = 1.04, **fontdict["plabel"]);

axS4["B"] = make_barplot2(
    data = fig_dataS4B, x = "cohort", y = "score", hue = "metric", width = 0.5, 
    title = "", legend_title = "Performance", xlabels = fig_xticksS4, 
    bar_label_align = True, xrot = 0, ax = axS4["B"], fontdict = fontdict)
axS4["B"].set_ylim(axS4_ylim);
axS4["B"].get_legend().set(bbox_to_anchor = (1.0, 0.25, 0.6, 0.6));
axS4["B"].set_title("B", x = -0.05, y = 1.04, **fontdict["plabel"]);

axS4["C"] = make_barplot2(
    data = fig_dataS4C, x = "cohort", y = "score", hue = "feature_set", 
    width = 0.5, title = "", legend = False, xlabels = fig_xticksS4, 
    bar_label_align = True, xrot = 0, ax = axS4["C"], fontdict = fontdict)
axS4["C"].set_ylim([-0.5, 29.0]);
axS4["C"].set_title("C", x = -0.05, y = 1.04, **fontdict["plabel"]);

axS4["D"] = make_barplot2(
    data = fig_dataS4D, x = "cohort", y = "score", hue = "feature_set", 
    width = 0.5, title = "", legend_title = "CCIs", xlabels = fig_xticksS4, 
    bar_label_align = True, xrot = 0, ax = axS4["D"], fontdict = fontdict)
axS4["D"].set_ylim(axS4_ylim);
axS4["D"].get_legend().set(bbox_to_anchor = (1.0, 0.25, 0.6, 0.6));
axS4["D"].set_title("D", x = -0.05, y = 1.04, **fontdict["plabel"]);


figS4.tight_layout(w_pad = 8, h_pad = 8)
plt.show()


## save figures.
if svdat:
    fig_path = "../data/plots/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS4 = data_file[-1].replace(
        "brightness_lirics_predictions_", "all_lirics_all_").replace(
            ".pkl", ".pdf")
    figS4.savefig(fig_path + fig_fileS4, dpi = "figure")
    

