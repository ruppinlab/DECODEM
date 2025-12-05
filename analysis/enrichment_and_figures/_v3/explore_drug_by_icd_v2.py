#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:25:22 2024

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
# from miscellaneous import date_time, tic, write_xlsx
from scipy.stats import mannwhitneyu
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from _functions import (classifier_performance, binary_performance, 
                        make_barplot1, make_barplot2, make_boxplot1, 
                        make_boxplot2, make_performance_plot)


#%% read all data.
## clinical info + predictions.

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



#%% get treatment-wise scores.

drug_cnt_all = pd.DataFrame(map(Counter, [
    clin_data_tn["NAT.regimen"], clin_data_tn_val["Chemo.Regimen"], 
    clin_data_bn["treatment"].replace({"Carboplatin+Paclitaxel": "P-Carboplatin"})]), 
    index = ["TransNEO", "ARTemis + PBCP", "BrighTNess"]).T
drug_cnt_all.loc["Total"] = drug_cnt_all.sum()

print(f"""
sample distribution across treatment regimens:
{drug_cnt_all}
""")


## get treatment-wise scores.
def get_drug_samples(clin, drug, col):
    return clin[clin[col].eq(drug)].index.tolist()

def get_scores(y_test, y_pred, smpl):
    try:
        scores = {ctp_: classifier_performance(y_test.loc[smpl], pred_.loc[smpl]) \
                  for ctp_, pred_ in y_pred.items()}
    except:
        scores = {ctp_: np.nan for ctp_ in y_pred}
    
    return scores

smpls_drug  = {ds: { } for ds in drug_cnt_all.columns}
scores_drug = {ds: { } for ds in drug_cnt_all.columns}
for drg in drug_cnt_all.index[:-1]:
    smpls_drug["TransNEO"][drg] = get_drug_samples(
            clin = clin_data_tn, col = "NAT.regimen", drug = drg)
    smpls_drug["ARTemis + PBCP"][drg] = get_drug_samples(
        clin = clin_data_tn_val, col = "Chemo.Regimen", drug = drg)
    if drg != "P-Carboplatin":
        smpls_drug["BrighTNess"][drg] = get_drug_samples(
            clin = clin_data_bn, col = "treatment", drug = drg)
    else:
        smpls_drug["BrighTNess"][drg] = get_drug_samples(
            clin = clin_data_bn, col = "treatment", 
            drug = "Carboplatin+Paclitaxel")
    
    ## performance.
    scores_drug["TransNEO"][drg] = get_scores(
        y_test = y_test_tn, y_pred = y_pred_tn, 
        smpl = smpls_drug["TransNEO"][drg])
    scores_drug["ARTemis + PBCP"][drg] = get_scores(
        y_test = y_test_tn_val, y_pred = y_pred_tn_val, 
        smpl = smpls_drug["ARTemis + PBCP"][drg])
    scores_drug["BrighTNess"][drg] = get_scores(
        y_test = y_test_bn, y_pred = y_pred_bn, 
        smpl = smpls_drug["BrighTNess"][drg])


#%% only consider the most frequent drug combo per cohort.

cell_types_all      = list(y_pred_tn.keys())

n_datasets          = pd.DataFrame([
    ["TransNEO", "T-FEC", len(smpls_drug["TransNEO"]["T-FEC"])], 
    ["ARTemis + PBCP", "T-FEC", len(smpls_drug["ARTemis + PBCP"]["T-FEC"])], 
    ["BrighTNess", "P-Cb", len(smpls_drug["BrighTNess"]["P-Carboplatin"])]], 
    columns = ["Dataset", "Drug", "n"])
n_datasets["label"] = n_datasets.apply(
    lambda x: f"{x.Dataset}: {x.Drug} (n = {x.n})", axis = 1)

scores_tn_t_fec     = pd.DataFrame(scores_drug["TransNEO"]["T-FEC"]).T
scores_tn_val_t_fec = pd.DataFrame(scores_drug["ARTemis + PBCP"]["T-FEC"]).T
scores_bn_p_cb      = pd.DataFrame(scores_drug["BrighTNess"]["P-Carboplatin"]).T


fig_ctps   = ["Cancer_Epithelial", "Myeloid", "Plasmablasts", "B-cells", 
              "Endothelial", "Bulk"]

fig_ordS5A = scores_tn_t_fec.loc[
    np.setdiff1d(cell_types_all, "Bulk")].sort_values(
        by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + ["Bulk"]

fig_ordS5B = scores_tn_val_t_fec.loc[
    np.setdiff1d(fig_ctps, "Bulk")].sort_values(
        by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + ["Bulk"]

fig_ordS5C = scores_bn_p_cb.loc[
    np.setdiff1d(fig_ctps, "Bulk")].sort_values(
        by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + ["Bulk"]


fig_dataS5A = scores_tn_t_fec.loc[fig_ordS5A].rename_axis(
    index = "cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")

fig_dataS5B = scores_tn_val_t_fec.loc[fig_ordS5B].rename_axis(
    index = "cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")

fig_dataS5C = scores_bn_p_cb.loc[fig_ordS5C].rename_axis(
    index = "cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")


fig_ctp_combo = ["Endothelial+Myeloid+Plasmablasts", 
                 "Myeloid+Plasmablasts+B-cells", 
                 "Myeloid+Plasmablasts", 
                 "Cancer_Epithelial+Myeloid", 
                 "Cancer_Epithelial+Plasmablasts"]

fig_ordS5D    = scores_tn_val_t_fec.loc[fig_ctp_combo].sort_values(
    by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + ["Bulk"]

fig_ordS5E    = scores_bn_p_cb.loc[fig_ctp_combo].sort_values(
    by = ["AUC", "AP"], ascending = [False] * 2).index.tolist() + ["Bulk"]

fig_dataS5D   = scores_tn_val_t_fec.loc[fig_ordS5D].rename_axis(
    index = "cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")

fig_dataS5E   = scores_bn_p_cb.loc[fig_ordS5E].rename_axis(
    index = "cell_type").reset_index().melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")


ctp_map        = {"Cancer_Epithelial": "CE", 
                  "Endothelial"      : "ENDO", 
                  "Myeloid"          : "MYL", 
                  "Plasmablasts"     : "PB", 
                  "B-cells"          : "B"}

fig_xticksS5A  = [ctp.replace("_", "\n") for ctp in fig_ordS5A]
fig_xticksS5B  = [ctp.replace("_", "\n") for ctp in fig_ordS5B]
fig_xticksS5C  = [ctp.replace("_", "\n") for ctp in fig_ordS5C]
# fig_xticksS5DE = ["ENDO + MYL + PB", "MYL + PB + B", "MYL + PB", 
#                  "CE + MYL", "CE + PB", "Bulk"]
fig_xticksS5D  = [" + ".join([ctp_map[ctp] for ctp in combo.split("+")]) \
                 for combo in fig_ordS5D[:-1]] + ["Bulk"]
fig_xticksS5E  = [" + ".join([ctp_map[ctp] for ctp in combo.split("+")]) \
                 for combo in fig_ordS5E[:-1]] + ["Bulk"]


#%% generate plot.

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

fig_colorsS5  = [colors[3], colors[4]]
fig_llocsS5   = [[0.02, 0.48], [1.00, 0.66, 0.33]]
fig_xrotS5    = 35

figS5, axS5  = plt.subplot_mosaic(
    mosaic = [["A", "A"], ["B", "C"], ["D", "E"]], figsize = (56, 44), 
    dpi = 600, layout = "constrained", height_ratios = [1] * 3, 
    width_ratios = [1] * 2)

axS5["A"]    = make_barplot2(data = fig_dataS5A, x = "cell_type", y = "score", 
                             hue = "metric", width = 0.5, 
                             title = n_datasets.label[0], legend = False, 
                             xlabels = fig_xticksS5A, xrot = fig_xrotS5, 
                             colors = fig_colorsS5, bar_label_align = True, 
                             ax = axS5["A"], fontdict = fontdict)
axS5["A"].set_ylim([-0.04, 1.04]);
figS5.text(x = fig_llocsS5[0][0], y = fig_llocsS5[1][0], s = "A", 
           **fontdict["plabel"]) 

axS5["B"] = make_barplot2(data = fig_dataS5B, x = "cell_type", y = "score", 
                          hue = "metric", width = 0.5, 
                          title = n_datasets.label[1], legend = False, 
                          xlabels = fig_xticksS5B, xrot = fig_xrotS5, 
                          colors = fig_colorsS5, bar_label_align = True, 
                          ax = axS5["B"], fontdict = fontdict)
axS5["B"].set_ylim([-0.04, 1.04]);
figS5.text(x = fig_llocsS5[0][0], y = fig_llocsS5[1][1], s = "B", 
           **fontdict["plabel"]) 

axS5["C"] = make_barplot2(data = fig_dataS5C, x = "cell_type", y = "score", 
                          hue = "metric", width = 0.5, 
                          title = n_datasets.label[2], legend = True, 
                          legend_title = "Performance", 
                          xlabels = fig_xticksS5C, xrot = fig_xrotS5, 
                          colors = fig_colorsS5, bar_label_align = True, 
                          ax = axS5["C"], fontdict = fontdict)
axS5["C"].get_legend().set(bbox_to_anchor = (1.0, 0.1, 0.6, 0.6));
axS5["C"].set_ylim([-0.04, 1.04]);
figS5.text(x = fig_llocsS5[0][1], y = fig_llocsS5[1][1], s = "C", 
           **fontdict["plabel"]) 

axS5["D"] = make_barplot2(data = fig_dataS5D, x = "cell_type", y = "score", 
                          hue = "metric", width = 0.5, title = "", 
                          legend = False, legend_title = "Performance", 
                          xlabels = fig_xticksS5D, xrot = fig_xrotS5, 
                          colors = fig_colorsS5, bar_label_align = True, 
                          ax = axS5["D"], fontdict = fontdict)
# axS5["D"].get_legend().set(bbox_to_anchor = (1.0, 0.1, 0.6, 0.6));
axS5["D"].set_ylim([-0.04, 1.04]);
figS5.text(x = fig_llocsS5[0][0], y = fig_llocsS5[1][2], s = "D", 
           **fontdict["plabel"]) 

axS5["E"] = make_barplot2(data = fig_dataS5E, x = "cell_type", y = "score", 
                          hue = "metric", width = 0.5, title = "", 
                          legend = False, xlabels = fig_xticksS5E, 
                          xrot = fig_xrotS5, colors = fig_colorsS5, 
                          bar_label_align = True, ax = axS5["E"], 
                          fontdict = fontdict)
axS5["E"].set_ylim([-0.04, 1.04]);
figS5.text(x = fig_llocsS5[0][1], y = fig_llocsS5[1][2], s = "E", 
           **fontdict["plabel"]) 

figS5.tight_layout(w_pad = 4, h_pad = 6)
plt.show()


#%% save figure.

svdat = False

## save figures.
if svdat:
    fig_path   = data_path[0] + "../plots/final_plots6/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS5 = "all_performance_top_icd_drugs_chemo_th0.99_25features_5foldCV.pdf"
    figS5.savefig(fig_path + fig_fileS5, dpi = "figure")

