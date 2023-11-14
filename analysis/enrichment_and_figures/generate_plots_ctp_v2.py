#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:14:05 2023

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
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from machine_learning._functions import (
    classifier_performance, binary_performance, make_barplot2, make_barplot3, 
    make_boxplot2, add_stat, make_piechart)


#%% read all data.

data_path = ["../data/TransNEO/transneo_analysis/mdl_data/", 
             "../data/TransNEO/use_data/", 
             "../data/TransNEO_SammutShare/", 
             "../data/BrighTNess/"]

data_file = ["transneo_predictions_chemo_th0.99_ENS2_25features_5foldCV_20Mar2023.pkl", 
             "tn_valid_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl", 
             "brightness_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl", 
             "transneo-diagnosis-MLscores.tsv", 
             "TransNEO_SupplementaryTablesAll.xlsx", 
             "transneo-diagnosis-clinical-features.xlsx", 
             "GSE164458_BrighTNess_clinical_info_SRD_04Oct2022.xlsx"]

## transneo results.
with open(data_path[0] + data_file[0], "rb") as file:
    data_obj = pickle.load(file)
    y_test_tn    = data_obj["label"]
    y_pred_tn    = data_obj["pred"]
    th_test_tn   = data_obj["th"]
    perf_test_tn = data_obj["perf"]
    del data_obj

## artemis + pbcp results.
with open(data_path[0] + data_file[1], "rb") as file:
    data_obj         = pickle.load(file)
    y_test_tn_val    = data_obj["label"]
    y_pred_tn_val    = data_obj["pred"]
    th_test_tn_val   = data_obj["th"]
    perf_test_tn_val = data_obj["perf"]
    del data_obj

## brightness results.
with open(data_path[0] + data_file[2], "rb") as file:
    data_obj     = pickle.load(file)
    y_test_bn    = data_obj["label"]
    y_pred_bn    = data_obj["pred"]
    th_test_bn   = data_obj["th"]
    perf_test_bn = data_obj["perf"]
    del data_obj


## clinical info.
clin_info_tn_sammut = pd.read_excel(
    data_path[2] + data_file[5], sheet_name = "training", 
    header = 0, index_col = 0)
clin_info_tn = pd.read_excel(
    data_path[1] + data_file[4], sheet_name = "Supplementary Table 1", 
    skiprows = 1, header = 0, index_col = 0)

clin_info_tn_val_sammut = pd.read_excel(
    data_path[2] + data_file[5], sheet_name = "validation", 
    header = 0, index_col = 0)
clin_info_tn_val = pd.read_excel(
    data_path[1] + data_file[4], sheet_name = "Supplementary Table 5", 
    skiprows = 1, header = 0, index_col = 0)
samples_sammut_tn_val = clin_info_tn_val.index.tolist()

clin_info_bn = pd.read_excel(
    data_path[3] + data_file[6], sheet_name = "samples", 
    header = 0, index_col = 0)


## clinical data for available samples.
clin_data_tn     = clin_info_tn.loc[y_test_tn.index].copy()
clin_data_tn_val = clin_info_tn_val.loc[y_test_tn_val.index].copy()
clin_data_bn     = clin_info_bn.loc[y_test_bn.index].copy()


## sammut et al. scores.
y_pred_sammut_all = pd.read_table(data_path[1] + data_file[3], sep = "\t", 
                                  header = 0, index_col = 0)

y_pred_sammut_tn    = y_pred_sammut_all[
    y_pred_sammut_all.Class == "Training"].drop(columns = ["Class"])
y_pred_sammut_tn[:] = MinMaxScaler().fit_transform(y_pred_sammut_tn)           # rescale to spread in [0, 1] for fair comparison


y_pred_sammut_tn_val    = y_pred_sammut_all[
    y_pred_sammut_all.Class == "Validation"].drop(columns = ["Class"])
y_pred_sammut_tn_val[:] = MinMaxScaler().fit_transform(y_pred_sammut_tn_val)   # rescale to spread in [0, 1] for fair comparison
y_pred_sammut_tn_val["Cohort"] = y_pred_sammut_tn_val.index.map(
    lambda idx: "PBCP" if ("PBCP" in idx) else "ARTEMIS")

pbcp_id_conv = dict(zip(
    np.setdiff1d(y_pred_sammut_tn_val.index, samples_sammut_tn_val), 
    np.setdiff1d(samples_sammut_tn_val, y_pred_sammut_tn_val.index) ))

y_pred_sammut_tn_val.rename(index = pbcp_id_conv, inplace = True)


#%% prepare data for fig. 2 - transneo.

cell_types = sorted(np.setdiff1d(y_pred_tn.columns, "Bulk"), 
                    key = lambda x: x.lower())

get_clf_perf = lambda y_pred: pd.Series(classifier_performance(
    y_test_tn.loc[y_pred.index], y_pred))
get_bin_perf = lambda y_pred: pd.Series(binary_performance(
    y_test_tn.loc[y_pred.index], y_pred))


## get performances for sammut samples.
samples_sammut_tn   = np.intersect1d(
    y_pred_tn.index, y_pred_sammut_tn.index).tolist()
y_test_tn_sm        = y_test_tn.loc[samples_sammut_tn]
y_pred_sammut_tn_sm = y_pred_sammut_tn.loc[samples_sammut_tn]
y_pred_tn_sm        = y_pred_tn.loc[samples_sammut_tn]

perf_test_sammut_tn  = y_pred_sammut_tn_sm.apply(get_clf_perf).T
perf_test_tn_sm      = y_pred_tn_sm.apply(get_clf_perf).T
perf_test_comp_tn    = perf_test_tn_sm.copy()
perf_test_comp_tn.loc["Sammut et al."] = perf_test_sammut_tn.loc["Clinical+RNA"]


## get data for panels A + B.
## model ordering- rank by AUC first, then AP.
mdl_ord = perf_test_comp_tn.loc[cell_types].sort_values(
    by = ["AUC", "AP"], ascending = [False, False]).index.tolist() \
    + ["Bulk", "Sammut et al."]

fig_xticks2AB = [mdl.replace("_", "\n") for mdl in mdl_ord]

## panel A.
fig_data2A = pd.concat([
    y_test_tn[samples_sammut_tn].rename("response").replace({1: "R", 0: "NR"}), 
    y_pred_tn_sm[mdl_ord[:-1]], 
    y_pred_sammut_tn_sm["Clinical+RNA"].rename(mdl_ord[-1]) ], axis = 1).melt(
        id_vars = ["response"], var_name = "cell_type", value_name = "score")

fig_stat2A = pd.DataFrame({
    ctp_: mannwhitneyu(*fig_data2A.groupby("response").apply(
        lambda df: df[df["cell_type"] == ctp_]["score"].tolist()).sort_index(
            ascending = False), alternative = "greater") \
    for ctp_ in mdl_ord}, index = ["U1", "pval"]).T
fig_stat2A["annot"] = fig_stat2A.pval.apply(
    lambda p: "***" if (p <= 0.001) else "**" if (p <= 0.01) \
        else "*" if (p <= 0.05) else "ns")

## panel B.
fig_data2B = perf_test_comp_tn.loc[mdl_ord, ["AUC", "AP"]].rename_axis(
    "cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")


#%% prepare data for fig. 2 - artermis + pbcp.

get_clf_perf = lambda y_pred: pd.Series(classifier_performance(
    y_test_tn_val.loc[y_pred.index], y_pred))
get_bin_perf = lambda y_pred: pd.Series(binary_performance(
    y_test_tn_val.loc[y_pred.index], y_pred))

## get performances for sammut samples.
samples_sammut_tn_val   = np.intersect1d(
    y_pred_tn_val.index, y_pred_sammut_tn_val.index).tolist()
y_pred_sammut_tn_val_sm = y_pred_sammut_tn_val.loc[samples_sammut_tn_val]
y_pred_tn_val_sm        = y_pred_tn_val.loc[samples_sammut_tn_val]

perf_test_sammut_tn_val_sm = y_pred_sammut_tn_val_sm.drop(
    columns = ["Cohort"]).apply(get_clf_perf).T
perf_test_tn_val_sm        = y_pred_tn_val_sm.apply(get_clf_perf).T


## get data for panel C.
mdl_ord1   = ["CAFs", "Normal_Epithelial", "Cancer_Epithelial", "Endothelial", 
              "Myeloid", "B-cells", "Plasmablasts", "Bulk"] + ["Sammut et al."]

perf_test_tn_val_sm1 = perf_test_tn_val_sm.loc[mdl_ord1[:-1]]
perf_test_tn_val_sm1.loc["Sammut et al."] = \
    perf_test_sammut_tn_val_sm.loc["Clinical+RNA"]

fig_xticks2C = [mdl.replace("_", "\n") for mdl in mdl_ord1]
fig_data2C   = perf_test_tn_val_sm1.rename_axis("cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")


#%% prepare data for fig. 2 - brightness.

## get data for panel D.
mdl_ord1   = ["Myeloid", "Plasmablasts", "Cancer_Epithelial", "CAFs", 
              "Normal_Epithelial", "Endothelial", "B-cells", "Bulk"]
perf_test_bn1 = perf_test_bn.loc[mdl_ord1, ["AUC", "AP"]]

fig_xticks2D = [mdl.replace("_", "\n") for mdl in mdl_ord1]
fig_data2D   = perf_test_bn1.rename_axis("cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")


#%% generate fig. 2.

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

fig2, ax2 = plt.subplot_mosaic(
    figsize = (56, 44), mosaic = [["A", "A"], ["B", "B"], ["C", "D"]], 
    layout = "constrained", height_ratios = [1]*3, width_ratios = [1]*2)

ax2["A"] = make_boxplot2(
    data = fig_data2A, x = "cell_type", y = "score", hue = "response", 
    hue_order = ["R", "NR"], width = 0.5, title = "", 
    legend_title = "Response", xlabels = [""] * len(fig_xticks2AB), 
    ax = ax2["A"], fontdict = fontdict)
ax2["A"] = add_stat(
    stats = fig_stat2A, data = fig_data2A, x = "cell_type", y = "score", 
    align = True, ax = ax2["A"], fontdict = fontdict)
ax2["A"].set_ylim([-0.04, 1.14]);
ax2["A"].set_title("A", x = -0.025, y = 1.02, **fontdict["plabel"]);

ax2["B"] = make_barplot2(
    data = fig_data2B, x = "cell_type", y = "score", hue = "metric", 
    width = 0.5, title = "", legend = False, xlabels = fig_xticks2AB, 
    xrot = 40, bar_label_align = True, ax = ax2["B"], fontdict = fontdict)
ax2["B"].set_ylim([0.0, 1.04]);
ax2["B"].set_title("B", x = -0.025, y = 1.02, **fontdict["plabel"]);

ax2["C"] = make_barplot2(
    data = fig_data2C, x = "cell_type", y = "score", hue = "metric", 
    title = "", width = 0.5, legend = False, xlabels = fig_xticks2C, 
    xrot = 40, bar_label_align = True, ax = ax2["C"], fontdict = fontdict)
ax2["C"].set_ylim([0.0, 1.04]);
ax2["C"].set_title("C", x = -0.05, y = 1.02, **fontdict["plabel"]);

ax2["D"] = make_barplot2(
    data = fig_data2D, x = "cell_type", y = "score", hue = "metric", 
    title = "", width = 0.5, legend_title = "Performance", 
    xlabels = fig_xticks2D, xrot = 40, bar_label_align = True, ax = ax2["D"], 
    fontdict = fontdict)
ax2["D"].get_legend().set(bbox_to_anchor = (1.0, 0.8, 0.6, 0.6));
ax2["D"].set_ylim([0.0, 1.04]);
ax2["D"].set_title("D", x = -0.05, y = 1.02, **fontdict["plabel"]);

# fig2.tight_layout(h_pad = 0, w_pad = 0)
plt.show()


## save figures.
if svdat:
    fig_path = "../data/plots/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file2 = data_file[1].replace(
        "tn_valid_predictions", "all_AUC_AP").replace(".pkl", ".pdf")
    fig2.savefig(fig_path + fig_file2, dpi = "figure")
    

#%% prepare data for fig. 3 - artemis + pbcp.

## model ordering from cv analysis.
ctp_abbv = dict(zip(cell_types, ["B", "CAF", "CE", "ENDO", "MYL", "NE", 
                                 "PB", "PVL", "T"]))
mdl_top = perf_test_tn_val_sm[perf_test_tn_val_sm.AUC.gt(
    perf_test_tn_val_sm.loc["Bulk", "AUC"])].index.tolist()                    # cell-type-models that outperform Bulk
mdl_all = perf_test_tn_val_sm.index.tolist()

keep_top = 5                                                                   # keep #top ensembles for two- and three-cell-type ensembles
mdl_ord_tn_val, mdl_names_tn_val = ["Bulk"], ["Bulk"]
for nn in range(3):
    ## pick only 'nn+1'-cell-type ensembles by counting '+' and order by AUC.
    mdl_nn_ = sorted([mdl_ for mdl_ in mdl_all if mdl_.count("+") == nn], 
                      key = lambda mdl_: perf_test_tn_val_sm.loc[mdl_, "AUC"], 
                      reverse = True)[:keep_top]
    mdl_ord_tn_val = mdl_nn_ + mdl_ord_tn_val
    
    ## format cell type names (shorthand for ensembles).
    if nn > 0:                                                                 # ensemble models
        for ctp_, c_ in ctp_abbv.items():
            mdl_nn_ = [mdl_.replace(ctp_, c_) for mdl_ in mdl_nn_]
        mdl_nn_ = [mdl_.replace("+", " + ") for mdl_ in mdl_nn_]
    else:                                                                      # individual models
        mdl_nn_ = [mdl_.replace("_", "\n") for mdl_ in mdl_nn_]
    
    mdl_names_tn_val = mdl_nn_ + mdl_names_tn_val
del mdl_nn_


## get data for panels A + B.
mdl_ord2_tn_val = [
    mdl for mdl in mdl_ord_tn_val if mdl.count("+") == 1] + ["Bulk"]

fig_xticks3A = [mdl for mdl in mdl_names_tn_val \
                if mdl.count("+") == 1] + ["Bulk"]
fig_data3A   = perf_test_tn_val_sm.loc[
    mdl_ord2_tn_val, ["AUC", "AP"]].rename_axis("cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")

mdl_ord3_tn_val = [
    mdl for mdl in mdl_ord_tn_val if mdl.count("+") == 2] + ["Bulk"]

fig_xticks3B = [mdl for mdl in mdl_names_tn_val \
                if mdl.count("+") == 2] + ["Bulk"]
fig_data3B   = perf_test_tn_val_sm.loc[
    mdl_ord3_tn_val, ["AUC", "AP"]].rename_axis("cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")


#%% prepare data for fig. 3 - brightness.

## model ordering from cv analysis.
ctp_abbv = dict(zip(cell_types, ["B", "CAF", "CE", "ENDO", "MYL", "NE", 
                                 "PB", "PVL", "T"]))
mdl_top = perf_test_bn[perf_test_bn.AUC.gt(
    perf_test_bn.loc["Bulk", "AUC"])].index.tolist()                           # cell-type-models that outperform Bulk
mdl_all = perf_test_bn.index.tolist()

keep_top = 5                                                                   # keep #top ensembles for two- and three-cell-type ensembles
mdl_ord_bn, mdl_names_bn = ["Bulk"], ["Bulk"]
for nn in range(3):
    ## pick only 'nn+1'-cell-type ensembles by counting '+' and order by AUC.
    mdl_nn_ = sorted([mdl_ for mdl_ in mdl_all if mdl_.count("+") == nn], 
                      key = lambda mdl_: perf_test_bn.loc[mdl_, "AUC"], 
                      reverse = True)[:keep_top]
    mdl_ord_bn = mdl_nn_ + mdl_ord_bn
    
    ## format cell type names (shorthand for ensembles).
    if nn > 0:                                                                 # ensemble models
        for ctp_, c_ in ctp_abbv.items():
            mdl_nn_ = [mdl_.replace(ctp_, c_) for mdl_ in mdl_nn_]
        mdl_nn_ = [mdl_.replace("+", " + ") for mdl_ in mdl_nn_]
    else:                                                                      # individual models
        mdl_nn_ = [mdl_.replace("_", "\n") for mdl_ in mdl_nn_]
    
    mdl_names_bn = mdl_nn_ + mdl_names_bn
del mdl_nn_


## get data for panels C + D.
mdl_ord2_bn = [mdl for mdl in mdl_ord_bn if mdl.count("+") == 1] + ["Bulk"]

fig_xticks3C = [mdl for mdl in mdl_names_bn if mdl.count("+") == 1] + ["Bulk"]
fig_data3C   = perf_test_bn.loc[mdl_ord2_bn, ["AUC", "AP"]].rename_axis(
    "cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")

mdl_ord3_bn = [mdl for mdl in mdl_ord_bn if mdl.count("+") == 2] + ["Bulk"]

fig_xticks3D = [mdl for mdl in mdl_names_bn if mdl.count("+") == 2] + ["Bulk"]
fig_data3D   = perf_test_bn.loc[mdl_ord3_bn, ["AUC", "AP"]].rename_axis(
    "cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")


#%% generate fig. 3-I. 

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

fig3, ax3 = plt.subplot_mosaic(
    figsize = (52, 32), mosaic = [["A", "B"], ["C", "D"]], 
    layout = "constrained", sharex = False, sharey = True)

ax3["A"] = make_barplot2(
    data = fig_data3A, x = "cell_type", y = "score", hue = "metric", 
    width = 0.5, title = "", legend = False, xlabels = fig_xticks3A, 
    xrot = 40, ax = ax3["A"], fontdict = fontdict)
ax3["A"].set_ylim([0.0, 1.04]);
ax3["A"].set_title("A", x = -0.05, y = 1.02, **fontdict["plabel"]);

ax3["B"] = make_barplot2(
    data = fig_data3B, x = "cell_type", y = "score", hue = "metric", 
    width = 0.5, title = "", legend_title = "Performance", 
    xlabels = fig_xticks3B, xrot = 40, ax = ax3["B"], fontdict = fontdict)
ax3["B"].set_ylim([0.0, 1.04]);
ax3["B"].set_title("B", x = -0.05, y = 1.02, **fontdict["plabel"]);

ax3["C"] = make_barplot2(
    data = fig_data3C, x = "cell_type", y = "score", hue = "metric", 
    width = 0.5, title = "", legend = False, xlabels = fig_xticks3C, 
    xrot = 40, ax = ax3["C"], fontdict = fontdict)
ax3["C"].set_ylim([0.0, 1.04]);
ax3["C"].set_title("C", x = -0.05, y = 1.02, **fontdict["plabel"]);

ax3["D"] = make_barplot2(
    data = fig_data3D, x = "cell_type", y = "score", hue = "metric", 
    width = 0.5, title = "", legend = False, xlabels = fig_xticks3D, 
    xrot = 40, ax = ax3["D"], fontdict = fontdict)
ax3["D"].set_ylim([0.0, 1.04]);
ax3["D"].set_title("D", x = -0.05, y = 1.02, **fontdict["plabel"]);

## add cell type abbreviations as annotation.
edge_prop = {"linestyle": "-", "linewidth": 3, "edgecolor": "#000000"}
ctp_abbv_tbl = pd.DataFrame(ctp_abbv.items(), columns = [
    "cell_type", "shorthand"]).drop(index = [1, 7, 8])                         # remove unused shorthands
ctp_abbv_tbl["cell_type"] = ctp_abbv_tbl["cell_type"].apply(
    lambda x: x.replace("_", "\n"))
ctp_tbl = ax3["D"].table(
    cellText = ctp_abbv_tbl.values, colWidths = [0.15, 0.08], 
    cellLoc = "center", loc = "lower right", bbox = (1.02, 0.0, 0.4, 1.0))
ctp_tbl.auto_set_font_size(False)
ctp_tbl.set(fontsize = fontdict["label"]["fontsize"]);
for tk, tc in ctp_tbl.get_celld().items():                                     # make table with outside borders only
    if tk[0] == 0:
        if tk[1] == 0:    tc.visible_edges = "TL"
        else:             tc.visible_edges = "TR"
    elif tk[0] == (ctp_abbv_tbl.shape[0] - 1):
        if tk[1] == 0:    tc.visible_edges = "BL"
        else:             tc.visible_edges = "BR"
    else:
        if tk[1] == 0:    tc.visible_edges = "L"
        else:             tc.visible_edges = "R"
    tc.set(**edge_prop)

ax3["D"].text(
    x = len(fig_xticks3D) + 0.8, y = 1.08, s = "Cell type shorthand", 
    ha = "center", **fontdict["title"]);

fig3.tight_layout(w_pad = -12, h_pad = -4)
plt.show()


## save figures.
if svdat:
    fig_path = "../data/plots/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file2 = data_file[1].replace(
        "tn_valid", "tn_valid_brightness").replace(
            "predictions", "combo_AUC_AP").replace(".pkl", ".pdf")
    fig3.savefig(fig_path + fig_file2, dpi = "figure")


#%% preapre data for fig. s1 - data summary.

def get_resp_frac(data):
    return {"R": data.eq(1).mean(), "NR": data.eq(0).mean()}
    

def get_subtype_frac(data):
    return {"ER+ / HER2-": data["ER.status"].eq("POS").mean(), 
            "TN": data["ER.status"].eq("NEG").mean()}

def get_drug_frac(data):
    return pd.Series(Counter(data)) / len(data)


fig_dataS1A = pd.DataFrame({
    "TransNEO": get_resp_frac(y_test_tn), 
    "ARTemis + PBCP": get_resp_frac(y_test_tn_val), 
    "BrighTNess": get_resp_frac(y_test_bn)
}).rename_axis("Response").reset_index()

fig_dataS1B = pd.DataFrame({
    "TransNEO": get_subtype_frac(clin_data_tn), 
    "ARTemis + PBCP": get_subtype_frac(clin_data_tn_val), 
    "BrighTNess": {"ER+ / HER2-": 0, "TN": clin_data_bn.shape[0]}
}).rename_axis("Subtypes").reset_index()

fig_dataS1C = pd.DataFrame({
    "TransNEO": get_drug_frac(clin_data_tn["NAT.regimen"]), 
    "ARTemis + PBCP": get_drug_frac(clin_data_tn_val["Chemo.Regimen"]), 
    "BrighTNess": get_drug_frac(clin_data_bn["treatment"].replace(
        {"Carboplatin+Paclitaxel": "P-Carboplatin"}) )
}).rename_axis("treatment").reset_index().replace(
    regex = {"treatment": {"Carboplatin": "Cb"}}).melt(
    id_vars = ["treatment"], var_name = "cohort", value_name = "fraction")

fig_xticksS1C = fig_dataS1C.cohort.unique().tolist()


#%% generate fig. s1. 

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

pie_explode = None

sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 12, "xtick.major.width": 4, "ytick.major.size": 12, 
    "ytick.major.width": 4, "xtick.bottom": True, "ytick.left": True, 
    "axes.edgecolor": "#000000", "axes.linewidth": 4})

figS1, axS1 = plt.subplot_mosaic(
    figsize = (52, 32), mosaic = [["A1", "A1", "A2", "A2", "A3", "A3"], 
                                  ["B1", "B1", "B2", "B2", "B3", "B3"], 
                                  [".", "C", "C", "C", "C", "."]], 
    layout = "constrained", height_ratios = [1.0, 1.0, 1.2], 
    width_ratios = [0.005, 0.995, 0.5, 0.5, 0.995, 0.005])

axS1["A1"] = make_piechart(
    data = fig_dataS1A, x = "TransNEO", y = "Response", explode = pie_explode, 
    title = f"TransNEO (n = {y_test_tn.size})", ax = axS1["A1"], 
    fontdict = fontdict)

axS1["A2"] = make_piechart(
    data = fig_dataS1A, x = "ARTemis + PBCP", y = "Response", 
    explode = pie_explode, title = f"ARTemis + PBCP (n = {y_test_tn_val.size})", 
    ax = axS1["A2"], fontdict = fontdict)

axS1["A3"] = make_piechart(
    data = fig_dataS1A, x = "BrighTNess", y = "Response", explode = pie_explode, 
    title = f"BrighTNess (n = {y_test_bn.size})", ax = axS1["A3"], 
    fontdict = fontdict)

figS1.text(x = 0.02, y = 1.02, s = "A", color = "black", **fontdict["plabel"]);

axS1["B1"] = make_piechart(
    data = fig_dataS1B, x = "TransNEO", y = "Subtypes", explode = pie_explode, 
    title = None, ax = axS1["B1"], fontdict = fontdict)

axS1["B2"] = make_piechart(
    data = fig_dataS1B, x = "ARTemis + PBCP", y = "Subtypes", 
    explode = pie_explode, title = None, ax = axS1["B2"], fontdict = fontdict)

axS1["B3"] = make_piechart(
    data = fig_dataS1B, x = "BrighTNess", y = "Subtypes", 
    explode = pie_explode, title = None, ax = axS1["B3"], fontdict = fontdict)

figS1.text(x = 0.02, y = 0.68, s = "B", color = "black", **fontdict["plabel"]);

axS1["C"] = make_barplot3(
    data = fig_dataS1C, x = "cohort", y = "fraction", hue = "treatment", 
    bar_labels = False, xlabels = fig_xticksS1C, title = "Treatment regimen", 
    ylabel = "Fraction of samples", ax = axS1["C"], fontdict = fontdict)
axS1["C"].get_legend().set(bbox_to_anchor = (1.0, -0.25, 0.6, 0.4), ncols = 2);
axS1["C"].set_ylim([0.0, 1.04]);

figS1.text(x = 0.02, y = 0.38, s = "C", color = "black", **fontdict["plabel"]);

plt.show()


## save figures.
if svdat:
    fig_path = "../data/plots/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS1 = "all_responders_subtypes_treatment_stats.pdf"
    figS1.savefig(fig_path + fig_fileS1, dpi = "figure")


#%% prepare data for fig. s2 - additional performance.

## DOR + SEN for cell-type-specific models.
mdl_keep = ["Cancer_Epithelial", "Endothelial", "Plasmablasts", 
            "Normal_Epithelial", "Myeloid", "B-cells", "CAFs"]

fig_dataS2A = pd.DataFrame({
    "cell_type": np.tile(mdl_keep + ["Bulk"], reps = 3), 
    "odds_ratio": (perf_test_tn.DOR[mdl_keep + ["Bulk"]].tolist() + 
                   perf_test_tn_val.DOR[mdl_keep + ["Bulk"]].tolist() + 
                   perf_test_bn.DOR[mdl_keep + ["Bulk"]].tolist()), 
    "cohort": np.repeat(["TransNEO", "ARTemis + PBCP", "BrighTNess"], 
                        repeats = len(mdl_keep) + 1) })


fig_dataS2C = pd.DataFrame({
    "cell_type": np.tile(mdl_keep + ["Bulk"], reps = 3), 
    "sensitivity": (perf_test_tn.SEN[mdl_keep + ["Bulk"]].tolist() + 
                    perf_test_tn_val.SEN[mdl_keep + ["Bulk"]].tolist() + 
                    perf_test_bn.SEN[mdl_keep + ["Bulk"]].tolist()), 
    "cohort": np.repeat(["TransNEO", "ARTemis + PBCP", "BrighTNess"], 
                        repeats = len(mdl_keep) + 1) })

fig_xticksS2AC = [mdl.replace("_", "\n") for mdl in mdl_keep] + ["Bulk"]


## DOR + SEN for multi-cell-ensemble models.
mdl_ens = np.setdiff1d(
    np.intersect1d(mdl_ord3_tn_val, mdl_ord3_bn).tolist() + 
    np.intersect1d(mdl_ord2_tn_val, mdl_ord2_bn).tolist(), "Bulk").tolist()

fig_dataS2B = pd.DataFrame({
    "cell_type": np.tile(mdl_ens + ["Bulk"], reps = 2), 
    "odds_ratio": (perf_test_tn_val.DOR[mdl_ens + ["Bulk"]].tolist() + 
                   perf_test_bn.DOR[mdl_ens + ["Bulk"]].tolist()), 
    "cohort": np.repeat(["ARTemis + PBCP", "BrighTNess"], 
                        repeats = len(mdl_ens) + 1) })


fig_dataS2D = pd.DataFrame({
    "cell_type": np.tile(mdl_ens + ["Bulk"], reps = 2), 
    "sensitivity": (perf_test_tn_val.SEN[mdl_ens + ["Bulk"]].tolist() + 
                   perf_test_bn.SEN[mdl_ens + ["Bulk"]].tolist()), 
    "cohort": np.repeat(["ARTemis + PBCP", "BrighTNess"], 
                        repeats = len(mdl_ens) + 1) })

fig_xticksS2BD = [" + ".join([ctp_abbv[ctp] for ctp in mdl.split("+")]) \
                  for mdl in mdl_ens] + ["Bulk"]


#%% generate fig. s2.

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

figS2, axS2 = plt.subplot_mosaic(
    figsize = (60, 28), mosaic = [["A1", "B1"], ["A2", "B2"], ["C", "D"]], 
    layout = "constrained", height_ratios = [0.5, 0.5, 1], 
    width_ratios = [1.0, 0.8], sharex = False, sharey = False)

axS2["A1"], axS2["A2"] = make_barplot3(
    data = fig_dataS2A, x = "cell_type", y = "odds_ratio", hue = "cohort",
    width = 0.5, bline = True, yrefs = [1] * 3, title = "", 
    xlabels = [""] * len(fig_xticksS2AC), xrot = 40, legend_title = "Dataset", 
    ax = (axS2["A1"], axS2["A2"]), fontdict = fontdict)
axS2["A1"].set_ylim([50.0, 208.0]);    axS2["A2"].set_ylim([0.0, 50.0]);
axS2["A2"].get_legend().set(bbox_to_anchor = (-0.5, 0.0, 0.6, 0.6));
axS2["A1"].set_title("A", x = -0.05, y = 1.02, **fontdict["plabel"]);

axS2["B1"], axS2["B2"] = make_barplot3(
    data = fig_dataS2B, x = "cell_type", y = "odds_ratio", hue = "cohort", 
    width = 0.33, bline = True, yrefs = [1] * 2, title = "", 
    xlabels = [""] * len(fig_xticksS2BD), xrot = 40, legend = False, 
    skipcolor = True, ax = (axS2["B1"], axS2["B2"]), fontdict = fontdict)
axS2["B1"].set_ylim([50.0, 208.0]);    axS2["B2"].set_ylim([0.0, 50.0]);
axS2["B1"].set_yticklabels([""]* len(axS2["B1"].get_yticks()));
axS2["B2"].set_yticklabels([""]* len(axS2["B2"].get_yticks()));
axS2["B1"].set_title("B", x = -0.05, y = 1.02, **fontdict["plabel"]);

axS2["C"] = make_barplot3(
    data = fig_dataS2C, x = "cell_type", y = "sensitivity", hue = "cohort", 
    width = 0.5, bar_labels = False, xlabels = fig_xticksS2AC, xrot = 40, 
    title = "", legend = False, ax = axS2["C"], fontdict = fontdict)
axS2["C"].set_ylim([0.0, 1.04]);
axS2["C"].set_title("C", x = -0.05, y = 1.02, **fontdict["plabel"]);

axS2["D"] = make_barplot3(
    data = fig_dataS2D, x = "cell_type", y = "sensitivity", hue = "cohort", 
    width = 0.33, bar_labels = False, xlabels = fig_xticksS2BD, xrot = 40, 
    title = "", legend = False, skipcolor = True, ax = axS2["D"], 
    fontdict = fontdict)
axS2["D"].set_ylim([0.0, 1.04]);
axS2["D"].set_yticklabels([""]* len(axS2["D"].get_yticks()));
axS2["D"].set_title("D", x = -0.05, y = 1.02, **fontdict["plabel"]);


## add cell type abbreviations as annotation.
edge_prop = {"linestyle": "-", "linewidth": 3, "edgecolor": "#000000"}
ctp_abbv_tbl = pd.DataFrame(ctp_abbv.items(), columns = [
    "cell_type", "shorthand"]).drop(index = [0, 1, 7, 8])                      # remove unused shorthands
ctp_abbv_tbl["cell_type"] = ctp_abbv_tbl["cell_type"].apply(
    lambda x: x.replace("_", "\n"))
ctp_tbl = axS2["C"].table(
    cellText = ctp_abbv_tbl.values, colWidths = [0.15, 0.08], 
    cellLoc = "center", edges = "BRTL", 
    loc = "lower right", bbox = (-0.48, 0.0, 0.35, 0.85))
ctp_tbl.auto_set_font_size(False)
ctp_tbl.set(fontsize = fontdict["label"]["fontsize"]);
for tk, tc in ctp_tbl.get_celld().items():                                     # make table with outside borders only
    if tk[0] == 0:
        if tk[1] == 0:    tc.visible_edges = "TL"
        else:             tc.visible_edges = "TR"
    elif tk[0] == (ctp_abbv_tbl.shape[0] - 1):
        if tk[1] == 0:    tc.visible_edges = "BL"
        else:             tc.visible_edges = "BR"
    else:
        if tk[1] == 0:    tc.visible_edges = "L"
        else:             tc.visible_edges = "R"
    tc.set(**edge_prop)

axS2["C"].text(
    x = -2.9, y = 0.92, s = "Cell type shorthand", ha = "center", 
    **fontdict["title"]);

plt.show()


## save figures.
if svdat:
    fig_path = "../data/plots/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS2 = data_file[1].replace(
        "tn_valid_predictions_", "all_DOR_SEN_").replace(".pkl", ".pdf")
    figS2.savefig(fig_path + fig_fileS2, dpi = "figure")

