#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:06:01 2023

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
# from itertools import combinations
from machine_learning._functions import (
    classifier_performance, make_barplot2, make_barplot3)
# from sklearn.model_selection import StratifiedKFold
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


#%% read data.

## specify sample subset.
use_samples = "chemo"
use_samples = use_samples.replace("+", "_")

## load data.
data_path = ["../data/TransNEO/transneo_analysis/", 
             "../data/TransNEO_SammutShare/validation/", 
             "../data/BrighTNess/validation/"]

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


## common cell types.
cell_types = reduce(np.intersect1d, map(lambda df: df.columns, [
    cell_frac_tn, cell_frac_tn_val, cell_frac_bn])).tolist()

print("\ncell types =", *cell_types, sep = "\n\t")


#%% association of abundance & response per cell type.

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


## generate data for visualization - fig s3-II.
mdl_ord = perf_test_frac_tn.AUC.sort_values(ascending = False).index.tolist()

fig_dataS3E = perf_test_frac_tn.loc[mdl_ord].reset_index().rename(
    columns = {"index": "cell_type"}).melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")

fig_dataS3F = perf_test_frac_tn_val.loc[mdl_ord].reset_index().rename(
    columns = {"index": "cell_type"}).melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")

fig_dataS3G = perf_test_frac_bn.loc[mdl_ord].reset_index().rename(
    columns = {"index": "cell_type"}).melt(
        id_vars = ["cell_type"], var_name = "metric", value_name = "score")

fig_xticksS3 = list(map(lambda ctp: ctp.replace("_", "\n"), mdl_ord))


#%% generate fig s3-II.

svdat = False                                                                  # set True to save figure 

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

axS3_yticks = np.arange(0, 1.25, 0.25)
figS3, axS3 = plt.subplot_mosaic(
    figsize = (48, 28), mosaic = [["E"], ["F"], ["G"]], layout = "constrained", 
    height_ratios = [1, 1, 1], sharex = True, sharey = True)

axS3["E"] = make_barplot2(
    data = fig_dataS3E, x = "cell_type", y = "score", hue = "metric", 
    width = 0.5, title = "TransNEO", legend = False, xlabels = fig_xticksS3, 
    xrot = 40, bar_label_align = True, ax = axS3["E"], fontdict = fontdict)
axS3["E"].set_ylim([0.0, 1.04]);
figS3.text(x = 0.02, y = 0.980, s = "E", color = "black", **fontdict["plabel"]);

axS3["F"] = make_barplot2(
    data = fig_dataS3F, x = "cell_type", y = "score", hue = "metric", 
    width = 0.5, title = "ARTemis + PBCP", legend_title = "Performance", 
    xlabels = fig_xticksS3, xrot = 40, bar_label_align = True, ax = axS3["F"], 
    fontdict = fontdict)
axS3["F"].get_legend().set(bbox_to_anchor = (1.0, 0.2, 0.6, 0.6));
axS3["F"].set_ylim([0.0, 1.04]);
figS3.text(x = 0.02, y = 0.670, s = "F", color = "black", **fontdict["plabel"]);

axS3["G"] = make_barplot2(
    data = fig_dataS3G, x = "cell_type", y = "score", hue = "metric", 
    width = 0.5, title = "BrighTNess", legend = False, xlabels = fig_xticksS3, 
    xrot = 40, bar_label_align = True, ax = axS3["G"], fontdict = fontdict)
axS3["G"].set_ylim([0.0, 1.08]);
axS3["G"].set_yticks(ticks = axS3_yticks, labels = axS3_yticks, **fontdict["label"]);
axS3["G"].yaxis.set_major_formatter("{x:0.2f}");
figS3.text(x = 0.02, y = 0.365, s = "G", color = "black", **fontdict["plabel"]);

figS3.tight_layout(h_pad = 8)
plt.show()


## save figures.
if svdat:
    datestamp = date_time()
    fig_path  = data_path[0] + "plots/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_fileS3 = f"all_chemo_abundance_response_association_{datestamp}.pdf"
    figS3.savefig(fig_path + fig_fileS3, dpi = "figure")


