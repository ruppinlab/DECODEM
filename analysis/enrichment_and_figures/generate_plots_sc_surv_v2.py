#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:47:02 2023

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
from sklearn.metrics import RocCurveDisplay
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from machine_learning._functions import make_boxplot2, add_stat


#%% read all data for fig. 5.

data_path = "../data/TransNEO/transneo_analysis/mdl_data/"
data_file = ["zhangTNBC2021_predictions_chemo_th0.99_ENS2_allfeatures_3foldCVtune_28Apr2023.pkl", 
             "zhangTNBC2021_predictions_chemo_immuno_th0.99_ENS2_allfeatures_3foldCVtune_28Apr2023.pkl",
             "tcga_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_10Nov2023.pkl"]


## get single-cell data.
with open(data_path + data_file[0], "rb") as file:
    data_obj     = pickle.load(file)
    y_test_cm    = data_obj["label"]
    y_pred_cm    = data_obj["pred"]
    th_test_cm   = data_obj["th"]
    perf_test_cm = data_obj["perf"]
    del data_obj

with open(data_path + data_file[1], "rb") as file:
    data_obj     = pickle.load(file)
    y_test_im    = data_obj["label"]
    y_pred_im    = data_obj["pred"]
    th_test_im   = data_obj["th"]
    perf_test_im = data_obj["perf"]
    del data_obj


## get tcga survival data.
with open(data_path + data_file[2], "rb") as file:
    data_obj       = pickle.load(file)
    y_test_surv    = data_obj["label"]
    y_pred_surv    = data_obj["pred"]
    th_test_surv   = data_obj["th"]
    clin_test_surv = data_obj["clin"]
    del data_obj


#%% prepare data for fig. 5 - sc.

cell_type = "B-cells"

fig_data5A = pd.DataFrame({
    "response": pd.concat([y_test_cm, y_test_im]).replace({1: "R", 0: "NR"}), 
    "score": pd.concat([y_pred_cm, y_pred_im])[cell_type], 
    "treatment": (["Chemotherapy"] * len(y_test_cm) + 
                  ["Chemotherapy + immunotherapy"] * len(y_test_im)) })

fig_stat5A = pd.DataFrame({
    trt_: mannwhitneyu(*fig_data5A.groupby("response").apply(
        lambda df: df[df["treatment"] == trt_]["score"].tolist()).sort_index(
            ascending = False), alternative = "greater") \
    for trt_ in fig_data5A["treatment"].unique()}, index = ["U1", "pval"]).T
fig_stat5A["annot"] = fig_stat5A.pval.apply(
    lambda p: "***" if (p <= 0.001) else "**" if (p <= 0.01) \
        else "*" if (p <= 0.05) else "ns")

fig_xticks5A = [trt.replace(" + ", " + \n") \
                for trt in fig_data5A["treatment"].unique()]


fig_data5B = {"Chemotherapy": pd.DataFrame({
    "response": y_test_cm, "score": y_pred_cm[cell_type]}), 
              "Chemotherapy + immunotherapy": pd.DataFrame({
    "response": y_test_im, "score": y_pred_im[cell_type]})}


#%% survival related functions.

def make_stratification_groups(data, group_by, group_id = [1, 0]):
    if group_id[0] == -1:
        group_id[0] = int(group_by.split("_q")[1]) - 1                         # quantile groups
    
    group_low_risk, group_high_risk = [
        data[data[group_by] == grp].copy() for grp in group_id]
    
    return group_low_risk, group_high_risk


def do_logrank_test(data, group_ids = ["low_risk", "high_risk"]):
    event, event_time = data[group_ids[0]].columns
    lr_test = logrank_test(durations_A = data[group_ids[0]][event_time], 
                           event_observed_A = data[group_ids[0]][event], 
                           durations_B = data[group_ids[1]][event_time], 
                           event_observed_B = data[group_ids[1]][event])
    
    return lr_test


#%% prepare data for fig. 5 - tcga survival.

cell_type = "Cancer_Epithelial"

group_by  = "groups_med"
group_id  = [-1, 1] if ("_q" in group_by) else [1, 0]
metrics   = {"OS": "Overall survival", "PFI": "Progression-free interval"}

low_risk, high_risk = make_stratification_groups(
    data = y_pred_surv[cell_type], group_by = group_by, group_id = group_id)

fig_data5C = {"low_risk": low_risk[["OS", "OS_time"]].dropna(how = "any"), 
              "high_risk": high_risk[["OS", "OS_time"]].dropna(how = "any")}

fig_stat5C = do_logrank_test(fig_data5C)

fig_data5D = {"low_risk": low_risk[["PFI", "PFI_time"]].dropna(how = "any"), 
              "high_risk": high_risk[["PFI", "PFI_time"]].dropna(how = "any")}
fig_stat5D = do_logrank_test(fig_data5D)


#%% plot functions.

def make_performance_plot(data, x, y, ax, fontdict, title = None, 
                          legend_title = None):
    ## plot parameters.
    colors    = ["#E08DAC", "#7595D0", "#B075D0", "#C3D075", "#7D7575", 
                 "#000000"]
    
    markers   = ["o", "s", "D", "p"];    mrkr_size = 25
    lnprop    = {"main": {"linestyle": "-", "linewidth": 8}, 
                 "base": {"linestyle": ":", "linewidth": 6}}
    lgnd_loc  = "lower left"
    lgnd_bbox = (1.0, 0.4, 0.6, 0.6)
    lgnd_ttl  = {pn.replace("font", ""): pv \
                 for pn, pv in fontdict["title"].items()}
    lgnd_font = {pn.replace("font", ""): pv \
                 for pn, pv in fontdict["label"].items()}
    points = np.arange(0, 1.2, 0.2).round(1)
    
    ## make plot.
    for nn_, (grp_, data_) in enumerate(data.items()):
        y_true_, y_pred_ = data_[x], data_[y]
        _ = RocCurveDisplay.from_predictions(
            y_true_, y_pred_, drop_intermediate = False, name = grp_, 
            color = colors[nn_], marker = markers[nn_], 
            markersize = mrkr_size, ax = ax, **lnprop["main"])
    
    ax.axline([0, 0], [1, 1], color = colors[-1], **lnprop["base"]);
    sns.despine(ax = ax, offset = 2, trim = False);                            # keeping axes lines only
    
    ## format plot ticks & labels.
    ax.set_xticks(ticks = points, labels = points, ma = "center", 
                  **fontdict["label"]);
    ax.set_yticks(ticks = points, labels = points, ma = "center", 
                  **fontdict["label"]);
    ax.set_xlim([-0.02, 1.02]);     ax.set_ylim([-0.02, 1.02]);
    ax.tick_params(axis = "both", which = "major", 
                   labelsize = fontdict["label"]["fontsize"]);
    ax.set_title(title, **fontdict["title"]);
    ax.set_xlabel("1 $-$ Specificity", labelpad = 12, **fontdict["label"]);
    ax.set_ylabel("Sensitivity", labelpad = 12, **fontdict["label"]);
    
    lgnd = ax.legend(loc = lgnd_loc, bbox_to_anchor = lgnd_bbox, 
                     prop = lgnd_font, title = legend_title, 
                     title_fontproperties = lgnd_ttl, frameon = False)
    lgnd.get_title().set_multialignment("center");
    for lgnd_txt_ in lgnd.get_texts():
        txt_ = lgnd_txt_.get_text().replace(" + ", " + \n").replace(
            "(AUC ", "").replace(")", "")
        lgnd_txt_.set_text(txt_)
        
    return ax


def make_km_plots(data, event, stat, title, ax, fontdict, 
                  event_time = None, alpha = 0.05, group_labels = None, 
                  xlabel = "Time in days", ylabel = "Survival probability", 
                  legend = True, legend_title = "Risk group"):
    ## plot parameters.
    t_range = np.arange(
        0, max([dat.max().max() for dat in data.values()]) + 1e3, 
        2e3).astype(int)
    points  = np.arange(0, 1.2, 0.2).round(1)
    
    if event_time is None:
        event_time = f"{event}_time"
    
    if group_labels is None:
        group_labels = [grp.capitalize().replace("_", "-") \
                        for grp in data.keys()]
    
    colors  = ["#E08DAC", "#F6CF6D"]
    lnprop = {"linestyle": "-", "linewidth": 8, "marker": "P", 
              "markersize": 25}
    
    lgnd_bbox = (1.0, 0.5, 0.6, 0.6)
    lgnd_font = {pn.replace("font", ""): pv \
                 for pn, pv in fontdict["label"].items()}
    lgnd_ttl  = {pn.replace("font", ""): pv \
                 for pn, pv in fontdict["title"].items()}

    
    ## make plots.
    kmf = KaplanMeierFitter()
    for nn, (grp, grp_data) in enumerate(data.items()):
        kmf.fit(durations = grp_data[event_time], 
                event_observed = grp_data[event], alpha = alpha, 
                label = group_labels[nn])
    
        kmf.plot_survival_function(
            show_censors = True, ci_force_lines = False, ci_alpha = alpha, 
            at_risk_counts = False, ax = ax, color = colors[nn], **lnprop);
    
    sns.despine(ax = ax, offset = 2, trim = False);                            # keeping axes lines only
    
    ax.set_xticks(ticks = t_range, labels = t_range, **fontdict["label"]);
    ax.set_yticks(ticks = points, labels = points, **fontdict["label"]);
    ax.tick_params(axis = "both", which = "major", 
                   labelsize = fontdict["label"]["fontsize"]);
    ax.set_ylim([-0.02, 1.04]);
    ax.set_title(title, **fontdict["title"]);
    ax.set_xlabel(xlabel, labelpad = 12, **fontdict["label"]);
    ax.set_ylabel(ylabel, labelpad = 12, **fontdict["label"]);
    
    if legend:
        ax.legend(loc = "lower left", bbox_to_anchor = lgnd_bbox, 
                  prop = lgnd_font, title = legend_title, 
                  title_fontproperties = lgnd_ttl, frameon = False);
    else:
        ax.legend([ ], [ ], frameon = False)
    
    ## add log-rank test p-value.
    annot = f"Log-rank $P$ = {stat.p_value.round(4)}"
    ax.text(x = 250, y = 0.15, s = annot, **fontdict["label"]);
    
    return ax


#%% generate fig. 5.

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
ax5_ylim = [-0.08, 1.08]
fig5, ax5 = plt.subplot_mosaic(
    figsize = (48, 28), mosaic = [["A", "B"], ["C", "D"]], 
    layout = "constrained", height_ratios = [1.0, 1.0], 
    width_ratios = [1.0, 1.0], sharex = False, sharey = False)

ax5["A"] = make_boxplot2(
    data = fig_data5A, x = "treatment", y = "score", hue = "response", 
    hue_order = ["R", "NR"], width = 0.5, title = "SC-TNBC response", 
    legend_title = "Response", xlabels = fig_xticks5A, ax = ax5["A"], 
    fontdict = fontdict)
ax5["A"] = add_stat(
    stats = fig_stat5A, data = fig_data5A, x = "treatment", y = "score", 
    align = True, ax = ax5["A"], fontdict = fontdict)
ax5["A"].set_ylim(ax5_ylim);
ax5["A"].get_legend().set(bbox_to_anchor = (-0.5, 0.4, 0.6, 0.6));
fig5.text(x = 0.092, y = 0.985, s = "A", color = "black", **fontdict["plabel"]);

ax5["B"] = make_performance_plot(
    data = fig_data5B, x = "response", y = "score", 
    title = "SC-TNBC performance", legend_title = "AUC", ax = ax5["B"], 
    fontdict = fontdict)
ax5["B"].set_xlim(ax5_ylim);        ax5["B"].set_ylim(ax5_ylim);
fig5.text(x = 0.426, y = 0.985, s = "B", color = "black", **fontdict["plabel"]);

ax5["C"] = make_km_plots(
    data = fig_data5C, stat = fig_stat5C, event = "OS", 
    title = "\nTCGA-BRCA OS", xlabel = None, legend = False, 
    ax = ax5["C"], fontdict = fontdict)
ax5["C"].set_ylim(ax5_ylim);
fig5.text(x = 0.092, y = 0.45, s = "C", color = "black", **fontdict["plabel"]);

ax5["D"] = make_km_plots(
    data = fig_data5D, stat = fig_stat5D, event = "PFI", 
    title = "\nTCGA-BRCA PFI", xlabel = None, ylabel = None, 
    legend_title = "Risk group", group_labels = [
        "Low-risk, likely responder", "High-risk, unlikely responder"], 
    ax = ax5["D"], fontdict = fontdict)
ax5["D"].set_ylim(ax5_ylim);
fig5.text(x = 0.426, y = 0.45, s = "D", color = "black", **fontdict["plabel"]);
fig5.supxlabel("Time in days", y = -0.04, x = 0.42, **fontdict["label"]);

plt.show()


## save figures.
if svdat:
    fig_path = "../data/plots/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file5 = data_file[1].replace(
        "zhangTNBC2021_predictions_", "sc_survival_").replace(
        "allfeatures", "all25features").replace("pkl", "pdf")
    fig5.savefig(fig_path + fig_file5, dpi = "figure")


#%% prepare data for fig. s5 - sc.

fig_dataS5A = pd.concat([
    y_test_cm.rename("response").replace({1: "R", 0: "NR"}), y_pred_cm], 
    axis = 1).sort_values(by = "response", ascending = False).melt(
        id_vars = "response", var_name = "cell_type", value_name = "score")

fig_statS5A = pd.DataFrame({
    ctp_: mannwhitneyu(*fig_dataS5A.groupby("response").apply(
        lambda df: df[df["cell_type"] == ctp_]["score"].tolist()).sort_index(
            ascending = False), alternative = "greater") \
    for ctp_ in fig_dataS5A["cell_type"].unique()}, index = ["U1", "pval"]).T
fig_statS5A["annot"] = fig_statS5A.pval.apply(
    lambda p: "***" if (p <= 0.001) else "**" if (p <= 0.01) \
        else "*" if (p <= 0.05) else "ns")


fig_dataS5B = {ctp: pd.concat(
    [y_test_cm, pred], axis = 1, keys = ["response", "score"]).dropna() \
        for ctp, pred in y_pred_cm.items()}
fig_dataS5B["Pseudobulk"] = fig_dataS5B["Bulk"];    fig_dataS5B.pop("Bulk");


fig_dataS5C = pd.concat([
    y_test_im.rename("response").replace({1: "R", 0: "NR"}), y_pred_im], 
    axis = 1).dropna().sort_values(by = "response", ascending = False).melt(
        id_vars = "response", var_name = "cell_type", value_name = "score")

fig_statS5C = pd.DataFrame({
    ctp_: mannwhitneyu(*fig_dataS5C.groupby("response").apply(
        lambda df: df[df["cell_type"] == ctp_]["score"].tolist()).sort_index(
            ascending = False), alternative = "greater") \
    for ctp_ in fig_dataS5C["cell_type"].unique()}, index = ["U1", "pval"]).T
fig_statS5C["annot"] = fig_statS5C.pval.apply(
    lambda p: "***" if (p <= 0.001) else "**" if (p <= 0.01) \
        else "*" if (p <= 0.05) else "ns")


fig_dataS5D = {ctp: pd.concat(
    [y_test_im, pred], axis = 1, keys = ["response", "score"]).dropna() \
        for ctp, pred in y_pred_im.items()}
fig_dataS5D["Pseudobulk"] = fig_dataS5D["Bulk"];    fig_dataS5D.pop("Bulk");

fig_xticksS5 = np.append(fig_dataS5A["cell_type"].unique()[:-1], 
                           "Pseudobulk").tolist()


#%% prepare data for fig. s5 - tcga survival.

cell_type = "Endothelial"

group_by  = "groups_med"
group_id  = [-1, 1] if ("_q" in group_by) else [1, 0]
metrics   = {"OS": "Overall survival", "PFI": "Progression-free interval"}

low_risk, high_risk = make_stratification_groups(
    data = y_pred_surv[cell_type], group_by = group_by, group_id = group_id)

fig_dataS5E = {"low_risk": low_risk[["OS", "OS_time"]].dropna(how = "any"), 
               "high_risk": high_risk[["OS", "OS_time"]].dropna(how = "any")}

fig_statS5E = do_logrank_test(fig_dataS5E)

fig_dataS5F = {"low_risk": low_risk[["PFI", "PFI_time"]].dropna(how = "any"), 
               "high_risk": high_risk[["PFI", "PFI_time"]].dropna(how = "any")}
fig_statS5F = do_logrank_test(fig_dataS5F)


#%% generate fig. s5.

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
axS5_ylim   = [-0.08, 1.12]
axS5_ticks  = {"ticks": np.arange(0, 1.25, 0.25), 
               "labels": [f"{x:0.2f}" for x in np.arange(0, 1.25, 0.25)]}
figS5, axS5 = plt.subplot_mosaic(
    figsize = (48, 36), mosaic = [["A", "B"], ["C", "D"], ["E", "F"]], 
    layout = "constrained", height_ratios = [0.8, 0.8, 1.0], 
    width_ratios = [1.0, 0.8], sharex = False, sharey = False)

axS5["A"] = make_boxplot2(
    data = fig_dataS5A, x = "cell_type", y = "score", hue = "response", 
    hue_order = ["R", "NR"], width = 0.5, 
    title = "SC-TNBC response\nChemotherapy", legend = False, 
    xlabels = fig_xticksS5, xrot = 0, ax = axS5["A"], fontdict = fontdict)
axS5["A"] = add_stat(
    stats = fig_statS5A, data = fig_dataS5A, x = "cell_type", y = "score", 
    align = True, ax = axS5["A"], fontdict = fontdict)
axS5["A"].set_ylim(axS5_ylim);
axS5["A"].set_xticklabels([""] * len(axS5["A"].get_xticks()));
axS5["A"].set_yticks(**axS5_ticks, **fontdict["label"]);
axS5["A"].yaxis.set_major_formatter("{x:0.2f}");
figS5.text(x = 0.108, y = 0.97, s = "A", color = "black", **fontdict["plabel"]);


axS5["B"] = make_performance_plot(
    data = fig_dataS5B, x = "response", y = "score", 
    title = "SC-TNBC performance\nChemotherapy", legend_title = "AUC", 
    ax = axS5["B"], fontdict = fontdict)
axS5["B"].set_xlim(axS5_ylim);      axS5["B"].set_ylim(axS5_ylim);
axS5["B"].set_xticks(ticks = axS5_ticks["ticks"], labels = [""] * 5, 
                     **fontdict["label"]);
axS5["B"].set_yticks(**axS5_ticks, **fontdict["label"]);
axS5["B"].set_xlabel(None);
axS5["B"].get_legend().set(bbox_to_anchor = (1.0, 0.1, 0.6, 0.6));
figS5.text(x = 0.472, y = 0.97, s = "B", color = "black", **fontdict["plabel"]);


axS5["C"] = make_boxplot2(
    data = fig_dataS5C, x = "cell_type", y = "score", hue = "response", 
    hue_order = ["R", "NR"], width = 0.5, 
    title = "\nChemotherapy + \nimmunotherapy", legend_title = "Response", 
    xlabels = fig_xticksS5, xrot = 0, ax = axS5["C"], fontdict = fontdict)
axS5["C"] = add_stat(
    stats = fig_statS5C, data = fig_dataS5C, x = "cell_type", y = "score", 
    align = True, ax = axS5["C"], fontdict = fontdict)
axS5["C"].set_ylim(axS5_ylim);
axS5["C"].get_legend().set(bbox_to_anchor = (-0.55, 0.85, 0.6, 0.6));
axS5["C"].set_yticks(**axS5_ticks, **fontdict["label"]);
axS5["C"].yaxis.set_major_formatter("{x:0.2f}");
figS5.text(x = 0.108, y = 0.66, s = "C", color = "black", **fontdict["plabel"]);


axS5["D"] = make_performance_plot(
    data = fig_dataS5D, x = "response", y = "score", 
    title = "\nChemotherapy +\nimmunotherapy", legend_title = "AUC", 
    ax = axS5["D"], fontdict = fontdict)
axS5["D"].set_xticks(**axS5_ticks, **fontdict["label"]);
axS5["D"].set_yticks(**axS5_ticks, **fontdict["label"]);
axS5["D"].set_xlim(axS5_ylim);      axS5["D"].set_ylim(axS5_ylim);
axS5["D"].get_legend().set(bbox_to_anchor = (1.0, 0.1, 0.6, 0.6));
figS5.text(x = 0.472, y = 0.66, s = "D", color = "black", **fontdict["plabel"]);


axS5["E"] = make_km_plots(
    data = fig_dataS5E, stat = fig_statS5E, event = "OS", 
    title = f"\nTCGA-BRCA OS\n({cell_type})", xlabel = None, legend = False, 
    ax = axS5["E"], fontdict = fontdict)
axS5["E"].set_ylim(axS5_ylim);
axS5["E"].set_yticks(**axS5_ticks, **fontdict["label"]);
figS5.text(x = 0.108, y = 0.315, s = "E", color = "black", **fontdict["plabel"]);

axS5["F"] = make_km_plots(
    data = fig_dataS5F, stat = fig_statS5F, event = "PFI", 
    title = f"\nTCGA-BRCA PFI\n({cell_type})", xlabel = None, ylabel = None, 
    legend_title = "Risk group", group_labels = [
        "Low-risk, likely responder", "High-risk, unlikely responder"], 
    ax = axS5["F"], fontdict = fontdict)
axS5["F"].set_ylim(axS5_ylim);
axS5["F"].set_yticks(**axS5_ticks, **fontdict["label"]);
axS5["F"].get_legend().set(bbox_to_anchor = (1.0, 0.4, 0.6, 0.6));
figS5.text(x = 0.472, y = 0.315, s = "F", color = "black", **fontdict["plabel"]);
figS5.supxlabel("Time in days", y = -0.04, x = 0.46, **fontdict["label"]);

plt.show()


## save figures.
if svdat:
    fig_path = "../data/plots/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS5 = "all_sc_scores_AUC_survival_KMplots.pdf"
    figS5.savefig(fig_path + fig_fileS5, dpi = "figure")

