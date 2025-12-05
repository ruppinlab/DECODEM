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
from miscellaneous import date_time, tic, write_xlsx
from math import nan
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from _functions import classifier_performance, make_barplot2


#%% functions.

## load result data from saved pickle.
def load_data(path):
    with open(path, mode = "rb") as file:
        obj = pickle.load(file)
    
    y_test, y_pred, ths, perf = \
        [obj[kk] for kk in ["label", "pred", "th", "perf"]]
    del obj                                                                    # release memory
    
    return y_test, y_pred, ths, perf


#%% read all data.

use_samples = "chemo"

data_path   = ["../../data/TransNEO/transneo_analysis/mdl_data/", 
               "../../data/TransNEO/use_data/", 
               "../../data/TransNEO/TransNEO_SammutShare/", 
               "../../data/BrighTNess/"]

data_file   = ["transneo_predictions_chemo_th0.99_ENS2_25features_5foldCV_20Mar2023.pkl", 
               "tn_valid_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl", 
               "brightness_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl", 
               "transneo-diagnosis-MLscores.tsv", 
               "TransNEO_SupplementaryTablesAll.xlsx", 
               "transneo-diagnosis-clinical-features.xlsx", 
               "GSE164458_BrighTNess_clinical_info_SRD_04Oct2022.xlsx"]

## load data.
print("loading & preparing data...");    _tic = tic()

## model predictions.
y_test_tn, y_pred_tn, th_test_tn, perf_test_tn = \
    load_data(data_path[0] + data_file[0])                                     # transneo

y_test_tn_val, y_pred_tn_val, th_test_tn_val, perf_test_tn_val = \
    load_data(data_path[0] + data_file[1])                                     # artemis + pbcp

y_test_bn, y_pred_bn, th_test_bn, perf_test_bn = \
    load_data(data_path[0] + data_file[2])                                     # brightness


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
    y_pred_sammut_all.Class == "Training"].drop(
    columns = ["Class"])
y_pred_sammut_tn[:] = MinMaxScaler().fit_transform(y_pred_sammut_tn)           # rescale to spread in [0, 1] for fair comparison


y_pred_sammut_tn_val    = y_pred_sammut_all[
    y_pred_sammut_all.Class == "Validation"].drop(
    columns = ["Class"])
y_pred_sammut_tn_val[:] = MinMaxScaler().fit_transform(y_pred_sammut_tn_val)   # rescale to spread in [0, 1] for fair comparison
y_pred_sammut_tn_val["Cohort"] = y_pred_sammut_tn_val.index.map(
    lambda idx: "PBCP" if ("PBCP" in idx) else "ARTEMIS")

pbcp_id_conv = dict(zip(
    np.setdiff1d(y_pred_sammut_tn_val.index, samples_sammut_tn_val), 
    np.setdiff1d(samples_sammut_tn_val, y_pred_sammut_tn_val.index) ))

y_pred_sammut_tn_val.rename(index = pbcp_id_conv, inplace = True)


print("done!", end = "");    _tic.toc()


print(f"""
dataset summary: 
cohorts   = TransNEO (n = {len(y_test_tn):,}), ARTesmis + PBCP (n = {len(y_test_tn_val):,}), BrighTNess (n = {len(y_test_bn):,})
treatment = chemotherapy, response = RCB ('pCR' vs. 'RD')
cell type = {', '.join(y_pred_tn.columns)}
""")


#%% get treatment-wise scores.

drug_cnt_all = pd.DataFrame(map(Counter, [
    clin_data_tn["NAT.regimen"].replace(regex = {"Carboplatin": "Cb"}), 
    clin_data_tn_val["Chemo.Regimen"].replace(regex = {"Carboplatin": "Cb"}), 
    clin_data_bn["treatment"].replace(
        to_replace = {"Carboplatin+Paclitaxel": "P-Cb"})]), 
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
        scores = {
            ctp_: classifier_performance(y_test.loc[smpl], pred_.loc[smpl]) 
            for ctp_, pred_ in y_pred.items()}
    except:
        scores = {ctp_: nan for ctp_ in y_pred}
    
    return scores


smpls_drug  = {ds: { } for ds in drug_cnt_all.columns}
scores_drug = {ds: { } for ds in drug_cnt_all.columns}
for drg in drug_cnt_all.index[:-1]:
    smpls_drug["TransNEO"][drg] = get_drug_samples(
            clin = clin_data_tn, col = "NAT.regimen", drug = drg)
    smpls_drug["ARTemis + PBCP"][drg] = get_drug_samples(
        clin = clin_data_tn_val, col = "Chemo.Regimen", drug = drg)
    if drg == "P-Cb":
        smpls_drug["BrighTNess"][drg] = get_drug_samples(
            clin = clin_data_bn, col = "treatment", 
            drug = "Carboplatin+Paclitaxel")
    else:
        smpls_drug["BrighTNess"][drg] = get_drug_samples(
            clin = clin_data_bn, col = "treatment", drug = drg)
    
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


#%% prepare data for supp fig 7.
## only consider the most frequent drug combo per cohort.
## transneo / artemis + pbcp: T-FEC, brightness: P-Cb

cell_types_all      = y_pred_tn.columns.tolist()

n_datasets          = pd.DataFrame([
    ["TransNEO", "T-FEC", len(smpls_drug["TransNEO"]["T-FEC"])], 
    ["ARTemis + PBCP", "T-FEC", len(smpls_drug["ARTemis + PBCP"]["T-FEC"])], 
    ["BrighTNess", "P-Cb", len(smpls_drug["BrighTNess"]["P-Cb"])]], 
    columns = ["Dataset", "Drug", "n"])

n_datasets["label"] = n_datasets.apply(
    lambda x: f"{x.Dataset}: {x.Drug} (n = {x.n})", axis = 1)

scores_tn_t_fec     = pd.DataFrame(scores_drug["TransNEO"]["T-FEC"]).T
scores_tn_val_t_fec = pd.DataFrame(scores_drug["ARTemis + PBCP"]["T-FEC"]).T
scores_bn_p_cb      = pd.DataFrame(scores_drug["BrighTNess"]["P-Cb"]).T


## individual cell types.
fig_ctpsS7 = ["Cancer_Epithelial", "Myeloid", "Plasmablasts", "B-cells", 
              "Endothelial", "Bulk"]

fig_ordS7, fig_dataS7 = [ ], [ ]
for scr_ in [scores_tn_t_fec, scores_tn_val_t_fec, scores_bn_p_cb]:
    ord_ = scr_.loc[
        np.setdiff1d(fig_ctpsS7 if len(fig_ordS7) > 0 else cell_types_all,     # include all cell types for TransNEO only
                     "Bulk")].sort_values(
        by = ["AUC", "AP"], ascending = False).pipe(
        lambda df: df.index.tolist() + ["Bulk"])
    
    dat_ = scr_.loc[
        ord_].reset_index(
        names = "model").melt(
        id_vars = "model", var_name = "metric", value_name = "score")
    
    fig_ordS7.append(ord_);    fig_dataS7.append(dat_)

del scr_, ord_, dat_                                                           # reduce clutter


## cell type ensembles.
fig_combosS7 = ["Endothelial+Myeloid+Plasmablasts", 
                "Myeloid+Plasmablasts+B-cells", 
                "Myeloid+Plasmablasts", 
                "Cancer_Epithelial+Myeloid", 
                "Cancer_Epithelial+Plasmablasts"]

for scr_ in [scores_tn_val_t_fec, scores_bn_p_cb]:
    ord_ = scr_.loc[
        fig_combosS7].sort_values(
        by = ["AUC", "AP"], ascending = False).pipe(
        lambda df: df.index.tolist() + ["Bulk"])
    
    dat_ = scr_.loc[
        ord_].reset_index(
        names = "model").melt(
        id_vars = "model", var_name = "metric", value_name = "score")
    
    fig_ordS7.append(ord_);    fig_dataS7.append(dat_)

del scr_, ord_, dat_                                                           # reduce clutter


## cell type shorthands for ticks.
ctp_map       = {"Cancer_Epithelial": "CE", 
                 "Endothelial"      : "ENDO", 
                 "Myeloid"          : "MYL", 
                 "Plasmablasts"     : "PB", 
                 "B-cells"          : "B"}

fig_ticksS7 = [ ]
for ord_ in fig_ordS7:
    if ord_[0].count("+") > 0:                                                 # combo
        ticks_ = [" + ".join([ctp_map[mdl] for mdl in mdls.split("+")]) 
                  for mdls in ord_[:-1]] + ord_[-1:]
    else:                                                                      # individual
        ticks_ = [mdl.replace("_", "\n") for mdl in ord_]
        
    fig_ticksS7.append(ticks_)


#%% make supp fig 7.

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


fontdict = {"label": dict(fontsize = 56, fontweight = "regular"), 
            "title": dict(fontsize = 60, fontweight = "semibold"), 
            "super": dict(fontsize = 64, fontweight = "bold"), 
            "plabel": dict(fontsize = 120, fontweight = "bold")}

colors   = ["#E08DAC", "#7595D0", "#75D0B0", "#B075D0", "#C3D075", 
            "#FFC72C", "#708090", "#A9A9A9", "#000000"]

## barplots.
fig_llocsS7   = [[0.02, 0.48], [1.00, 0.66, 0.33]]
fig_colorsS7  = [colors[3], colors[4]]
fig_ttlsS7    = n_datasets.label.tolist() + [None] * 2

figS7, axS7 = plt.subplot_mosaic(
    mosaic = [["A", "A"], ["B", "C"], ["D", "E"]], figsize = (56, 44), 
    height_ratios = [1] * 3, width_ratios = [1] * 2)

for k, (lbl, ax) in enumerate(axS7.items()):
    ax = make_barplot2(data = fig_dataS7[k], x = "model", y = "score", 
                       hue = "metric", width = 0.5, title = fig_ttlsS7[k], 
                       xlabels = fig_ticksS7[k], xrot = 35, 
                       colors = fig_colorsS7, legend = (lbl == "C"), 
                       bar_label_align = True, ax = ax, fontdict = fontdict)
    ax.set_ylim([-0.04, 1.04])
    if lbl == "C":
        ax.get_legend().set(bbox_to_anchor = (1.0, 0.1), title = "Performance");
    
    figS7.text(x = fig_llocsS7[0][int(lbl in "CE")], 
               y = fig_llocsS7[1][0 if lbl == "A" else 1 if lbl in "BC" else 2], 
               s = lbl, **fontdict["plabel"]);                                 # add panel labels

figS7.tight_layout(w_pad = 4, h_pad = 6)
plt.show()


## save figures.
if svdat:
    fig_path   = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS7 = "all_performance_top_icd_drugs_chemo_th0.99_25features_5foldCV.pdf"
    figS7.savefig(fig_path + fig_fileS7, dpi = 600)
    print(fig_fileS7)

