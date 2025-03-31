#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:19:27 2023

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
# from argparse import ArgumentParser
from miscellaneous import date_time, tic, write_xlsx
from itertools import combinations
from functools import reduce
from operator import add
from _functions import (MakeClassifier, EnsembleClassifier, train_pipeline, 
                        predict_proba_scaled, get_best_threshold, 
                        classifier_performance, binary_performance, 
                        make_barplot1, make_barplot2)
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold


#%% functions.

## get confident genes for a cell type.
def get_conf_genes(conf, th = 0.99):
    genes = conf[conf.ge(th)].index.tolist()
    return genes


#%% read data.

# use_samples = "all"
# use_samples = "chemo_targeted"
use_samples = "chemo"

data_path = ["../../data/TransNEO/transneo_analysis/", 
             "../../data/TransNEO/TransNEO_SammutShare/validation/"]

data_file = [f"transneo_data_{use_samples}_v2.pkl", 
             f"transneo_validation_{use_samples}_v2.pkl"]


## load data.
with open(data_path[0] + data_file[0], "rb") as file:
    data_train_obj   = pickle.load(file)
    exp_all_train    = data_train_obj["exp"]
    resp_pCR_train   = data_train_obj["resp"]
    cell_frac_train  = data_train_obj["frac"]
    conf_score_train = data_train_obj["conf"]
    clin_info_train  = data_train_obj["clin"]
    del data_train_obj

with open(data_path[1] + data_file[1], "rb") as file:
    data_test_obj   = pickle.load(file)
    exp_all_test    = data_test_obj["exp"]
    resp_pCR_test   = data_test_obj["resp"]
    cell_frac_test  = data_test_obj["frac"]
    conf_score_test = data_test_obj["conf"]
    clin_info_test  = data_test_obj["clin"]
    del data_test_obj

if conf_score_train.columns.tolist() == conf_score_test.columns.tolist():
    cell_types = conf_score_train.columns.tolist()
else:
    raise ValueError("the cell types are not the same between training and test datasets!")


#%% prepare data.

conf_th = 0.99                                                                 # confident gene cut-off
genes, X_all_train, X_all_test = { }, { }, { }
for ctp_ in cell_types + ["Bulk"]:
    ## get confident genes.
    try:                                                                       # individual cell type
        gn_ctp_ = np.intersect1d(
            get_conf_genes(conf_score_train[ctp_], th = conf_th), 
            get_conf_genes(conf_score_test[ctp_], th = conf_th) )
    except:                                                                    # Bulk
        gn_ctp_ = np.intersect1d(conf_score_train.index, conf_score_test.index)
    
    gn_ctp_ = gn_ctp_.tolist()
    
    ## get expression data (append cell type to gene symbols).
    X_ctp_train_ = exp_all_train[ctp_].loc[gn_ctp_].T.rename(
        columns  = lambda gn: f"{gn}__{ctp_}")
    X_ctp_test_  = exp_all_test[ctp_].loc[gn_ctp_].T.rename(
        columns  = lambda gn: f"{gn}__{ctp_}")
    
    ## save data.
    genes[ctp_], X_all_train[ctp_], X_all_test[ctp_] = \
        gn_ctp_, X_ctp_train_, X_ctp_test_

del gn_ctp_, X_ctp_train_, X_ctp_test_

## get response labels.
y_train = resp_pCR_train.loc[X_all_train["Bulk"].index].copy()
y_test  = resp_pCR_test.loc[X_all_test["Bulk"].index].copy()


#%% modeling parameters.

## input: cell types - individual / combo.
use_ctp = np.append(cell_types, "Bulk").tolist()                               # all individual cell types + Bulk 
# use_ctp = list(combinations(cell_types, r = 2))                                # all two-cell-type ensembles 
# use_ctp = list(combinations(cell_types, r = 3))                                # all three-cell-type ensembles 
# use_ctp = [("B-cells", "Myeloid", "PVL"), ("all",  ), ("Bulk", )]
# use_ctp = [("Cancer_Epithelial", "Endothelial", "Myeloid", "Plasmablasts"), 
#             ("all", ), ("Bulk", )]
# use_ctp = ["all", "Bulk"]
# use_ctp = [("Endothelial", "Myeloid", "Plasmablasts"), ("CAFs", ), ("Bulk", )]
# 
top_cell_types = ["Cancer_Epithelial", "Endothelial", "Myeloid", 
                  "Plasmablasts", "B-cells", "Normal_Epithelial"]
# use_ctp = list(combinations(cell_types, r = 1)) + \
#             reduce(add, [list(combinations(top_cell_types, r)) \
#                           for r in range(2, 4)]) + [("Bulk", )]

# use_ctp = list(combinations(top_cell_types, r = 2)) + [("Bulk", )]
# use_ctp = list(combinations(top_cell_types, r = 3)) + [("Bulk", )]
# use_ctp = list(combinations(top_cell_types, r = 5)) + [("Bulk", )]

## format cell types list.
if isinstance(use_ctp, list):
    if not isinstance(use_ctp[0], tuple):
        use_ctp = [tuple([ctp_]) for ctp_ in use_ctp]
elif isinstance(use_ctp, tuple):
    use_ctp = [use_ctp]
elif isinstance(use_ctp, str):
    use_ctp = [tuple([use_ctp])]


## model parameters.
num_feat_max = 25                                                              # maximum #features to use
var_th       = 0.1
mdl_seed     = 86420


## choose classifier: LR, RF, SVM, XGB, ENS1 (L+R+S), ENS2 (L+R+S+X).
use_mdl = "ENS2"
use_mdl = use_mdl.upper()
mdl_list_ind = ["LR", "RF", "SVM", "XGB"]                                      # individual classifier list
if use_mdl == "ENS1":
    mdl_list = np.setdiff1d(mdl_list_ind, "XGB").tolist()
elif use_mdl == "ENS2":
    mdl_list = mdl_list_ind.copy()


## cv parameters.
tune_seed = 84
cv_seed   = 4


#%% model per cell type/combo.

## get parameters.
num_split_rep   = 5
num_splits      = 3
stratify_splits = False
use_mets        = ["AUC", "AP", "ACC", "DOR", "SEN", "PPV", "SPC"]             # list of performance metrics to use


_tic = tic()

## start modeling per cell type.
y_pred_val    = { };    th_test_val     = { };    perf_test_val = { }
pipe_test_val = { };    params_test_val = { }                                  # save all pipelines for SHAP
X_train_all   = { };    X_test_all      = { }                                  # save all train/test splits for SHAP
for use_ctp_ in use_ctp:
    ## get training & test sets.
    ctp_list = tuple(cell_types) if (use_ctp_[0] == "all") else use_ctp_
    X_train = pd.concat([X_all_train[ctp_] for ctp_ in ctp_list], axis = 1)
    X_test  = pd.concat([X_all_test[ctp_] for ctp_ in ctp_list], axis = 1)
    if use_samples == "all":                                                   # flag for anti-HER2 therapy
        X_train["aHER2"] = clin_info_train["aHER2.cycles"].notna().astype(int)
        X_test["aHER2"]  = clin_info_test["anti.her2.cycles"].notna(
            ).astype(int)
    
    ctp_mdl = "+".join(use_ctp_)                                               # cell-type model name
    
    print(f"""\n
    samples = {use_samples}, cell type = {ctp_mdl}
    available #genes = {X_train.shape[1]}, max #features = {num_feat_max}
    model = {use_mdl}, #repetitions = {num_split_rep}
    sample size: training = {X_train.shape[0]}, test = {X_test.shape[0]}""")
    
    
    ## start modeling per repition.
    y_pred_rep    = { };    th_test_rep     = { };    perf_test_rep = { }
    pipe_test_rep = { };    params_test_rep = { }                              # save all pipelines for SHAP
    for use_seed in range(num_split_rep):
        print(f"\nsplit seed = {use_seed}")
        rep_mdl = f"seed{use_seed}"                                            # repetition model name
                
        ## make cv splits for tuning.
        if stratify_splits:
            cv_tune = StratifiedKFold(n_splits = num_splits, shuffle = True, 
                                      random_state = use_seed)
        else:
            cv_tune = KFold(n_splits = num_splits, shuffle = True, 
                            random_state = use_seed)
        
        ## train model.
        try:                                                                   # individual classifier
            pipe_tuned, params_tuned = train_pipeline(
                model = use_mdl, train_data = (X_train, y_train), 
                max_features = num_feat_max, var_th = var_th, 
                cv_tune = cv_tune, mdl_seed = mdl_seed, 
                tune_seed = tune_seed)
            
        except:                                                                # ensemble classifier
            # step I: fit individual models.
            pipes_mdl = { };    params_mdl = { }
            for mdl in mdl_list:
                pipes_mdl[mdl], params_mdl[mdl] = train_pipeline(
                    model = mdl, train_data = (X_train, y_train), 
                    max_features = num_feat_max, var_th = var_th, 
                    cv_tune = cv_tune, mdl_seed = mdl_seed, 
                    tune_seed = tune_seed)
            
            # step II: get ensemble model.
            pipe_tuned = EnsembleClassifier(models = list(pipes_mdl.values()))
            pipe_tuned.fit(X_train, y_train)
            params_tuned = params_mdl.copy()
        
        
        ## get prediction performances.
        y_fit  = predict_proba_scaled(pipe_tuned, X_train, scale = True)
        th_fit = get_best_threshold(y_train, y_fit[:, 1], curve = "PR")
        y_pred = predict_proba_scaled(pipe_tuned, X_test, scale = True)
        y_pred_th = (y_pred >= th_fit).astype(int)
        perf_test = pd.concat([
                pd.Series(classifier_performance(y_test, y_pred[:, 1])), 
                pd.Series(binary_performance(y_test, y_pred_th[:, 1])) ])
        
        print(f"performance for seed {use_seed} = {perf_test[use_mets].round(4).to_dict()}")
        
        
        ## save results for this repetition.
        y_pred_rep[rep_mdl]      = y_pred[:, 1]
        th_test_rep[rep_mdl]     = th_fit
        perf_test_rep[rep_mdl]   = perf_test[use_mets]
        pipe_test_rep[rep_mdl]   = pipe_tuned
        params_test_rep[rep_mdl] = params_tuned
            
    
    ## overall performance across all repetitions.
    y_pred_rep            = pd.DataFrame(y_pred_rep)
    y_pred_rep["mean"]    = y_pred_rep.mean(axis = 1)
    th_test_rep           = pd.Series(th_test_rep)
    th_test_rep["mean"]   = th_test_rep.mean()
    perf_test_rep         = pd.DataFrame(perf_test_rep)
    perf_test_rep["mean"] = perf_test_rep.mean(axis = 1)
    # print(f"\noverall performance: \n{perf_test_rep.round(4)}")
    
    
    ## combine prediction across all repetitions & get performance.
    y_pred_full    = y_pred_rep["mean"]
    # y_pred_full    = MinMaxScaler().fit_transform(pd.DataFrame(
    #     y_pred_rep["mean"])).squeeze()
    y_pred_th_full = (y_pred_full >= th_test_rep["mean"]).astype(int)
    perf_test_full = pd.concat([
        pd.Series(classifier_performance(y_test, y_pred_full)), 
        pd.Series(binary_performance(y_test, y_pred_th_full)) ])
        
    perf_test_all = pd.concat([perf_test_rep["mean"], perf_test_full[use_mets]], 
                              axis = 1, keys = ["mean_perf", "mean_pred"])
    
    print(os.system("clear"))                                                  # clears console
    print(f"\noverall performance for cell type = {'+'.join(use_ctp_)}: \n{perf_test_all.round(4)}")
    
    
    ## save results for this cell type.
    y_pred_val[ctp_mdl]      = y_pred_full
    th_test_val[ctp_mdl]     = th_test_rep
    perf_test_val[ctp_mdl]   = perf_test_all["mean_pred"]
    # perf_test_val[ctp_mdl]   = perf_test_all["mean_perf"]
    pipe_test_val[ctp_mdl]   = pipe_test_rep
    params_test_val[ctp_mdl] = params_test_rep
    X_train_all[ctp_mdl]     = X_train
    X_test_all[ctp_mdl]      = X_test


## fianl performance for all cell types.
y_pred_val    = pd.DataFrame(y_pred_val).set_index(X_test.index)               # mean prediction matrix
th_test_val   = pd.DataFrame(th_test_val).T
perf_test_val = pd.DataFrame(perf_test_val).T

# print(os.system("clear"))                                                      # clears console
print(f"""\n{'-' * 64}
validation performance for treatment = {use_samples}:
cohort = TransNEO validation (Artemis + PBCP; n = {y_test.size})
{perf_test_val.round(4)}""")

_tic.toc()


#%% save full prediction & performance tables.

svdat = False

if svdat:
    # datestamp = date_time()
    datestamp = "23Mar2023"
    
    ## save full predictions & performance.
    out_path = data_path[0] + "mdl_data/"
    out_file = f"tn_valid_predictions_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.pkl"
    out_dict = {"label": y_test,   "pred": y_pred_val, 
                "th": th_test_val, "perf": perf_test_val}
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    with open(out_path + out_file, "wb") as file:
        pickle.dump(out_dict, file)
    
    
    ## save models for feature importance.
    out_file = f"tn_valid_models_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.pkl"
    out_dict = {"pipeline": pipe_test_val, "params": params_test_val, 
                "train": X_train_all,      "test": X_test_all}
    
    with open(out_path + out_file, "wb") as file:
        pickle.dump(out_dict, file)
    

    ## save complete performance into xlsx file.
    out_path = _wpath_ + "results/"
    out_file = f"tn_valid_results_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.xlsx"
    out_dict = perf_test_val.copy()
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    write_xlsx(out_path + out_file, out_dict)


#%% load saved data.

lddat = False

if lddat:
    res_path = data_path[0] + "mdl_data/"
    res_file = "tn_valid_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl"
    with open(res_path + res_file, "rb") as file:
        data_obj      = pickle.load(file)
        y_test_val    = data_obj["label"]
        y_pred_val    = data_obj["pred"]
        th_test_val   = data_obj["th"]
        perf_test_val = data_obj["perf"]
        del data_obj
    
    assert np.all(y_test == y_test_val)                                        # sanity check


#%% get Sammut et al. validation scores and calculate performance.

## read & prepare scores.
sammut_path = ["../../data/TransNEO/use_data/", 
               "../../data/TransNEO/TransNEO_SammutShare/"]
sammut_file = ["transneo-diagnosis-MLscores.tsv", 
               "TransNEO_SupplementaryTablesAll.xlsx", 
               "transneo-diagnosis-clinical-features.xlsx"]

y_score_sammut = pd.read_table(sammut_path[0] + sammut_file[0], sep = "\t", 
                               header = 0, index_col = 0)
clin_info_test_sammut = pd.read_excel(
    sammut_path[1] + sammut_file[2], sheet_name = "validation", 
    header = 0, index_col = 0)
clin_info_test_sammut_supp = pd.read_excel(
    sammut_path[0] + sammut_file[1], sheet_name = "Supplementary Table 5", 
    skiprows = 1, header = 0, index_col = 0)
samples_sammut = clin_info_test_sammut_supp.index.tolist()

## get ML scores.
y_score_val_sammut = y_score_sammut[
    y_score_sammut.Class == "Validation"].drop(columns = "Class")
y_score_val_sammut["Cohort"] = ["PBCP" if ("PBCP" in idx) else "ARTEMIS" \
                                for idx in y_score_val_sammut.index]

pbcp_id_conv = dict(zip(np.setdiff1d(y_score_val_sammut.index, samples_sammut), 
                        np.setdiff1d(samples_sammut, y_score_val_sammut.index)))

y_score_val_sammut.rename(index = pbcp_id_conv, inplace = True)

## get clinical response & sample set.
y_test_sammut = clin_info_test_sammut.loc[y_score_val_sammut.index, 
                                          ["pCR.RD", "anti.her2.cycles"]]
y_test_sammut["pCR.RD"] = (y_test_sammut["pCR.RD"] == "pCR").astype(int)
y_test_sammut["Cohort"] = ["ARTEMIS" if ("A" in idx) else "PBCP" \
                           for idx in y_test_sammut.index]

aHER2 = y_test_sammut["anti.her2.cycles"].notna()
samples_ct_sammut = y_test_sammut[aHER2].index.tolist()
samples_cm_sammut = y_test_sammut[~aHER2].index.tolist()

## calculate performance.
perf_val_sammut_all = y_score_val_sammut.drop(columns = "Cohort").apply(
    lambda pred_: pd.Series(classifier_performance(
        y_test_sammut["pCR.RD"], pred_))).T

perf_val_sammut_ct = y_score_val_sammut.loc[samples_ct_sammut].drop(
    columns = "Cohort").apply(lambda pred_: pd.Series(classifier_performance(
        y_test_sammut.loc[samples_ct_sammut, "pCR.RD"], pred_))).T

perf_val_sammut_cm = y_score_val_sammut.loc[samples_cm_sammut].drop(
    columns = "Cohort").apply(lambda pred_: pd.Series(classifier_performance(
        y_test_sammut.loc[samples_cm_sammut, "pCR.RD"], pred_))).T

if use_samples == "all":
    perf_val_sammut = perf_val_sammut_all.copy()
elif use_samples == "chemo_targeted":
    perf_val_sammut = perf_val_sammut_ct.copy()
elif use_samples == "chemo":
    perf_val_sammut = perf_val_sammut_cm.copy()

print(f"""\nperformance of Sammut et al. model for treatment = {use_samples}:
cohort = TransNEO validation (Artemis + PBCP; n = {len(samples_cm_sammut)})
{perf_val_sammut.round(4)}""")


#%% plot parameters.

fontname = "sans"
fontdict = {"label": dict(fontfamily = fontname, fontsize = 48, 
                          fontweight = "regular"), 
            "title": dict(fontfamily = fontname, fontsize = 52, 
                          fontweight = "semibold"), 
            "super": dict(fontfamily = fontname, fontsize = 56, 
                          fontweight = "bold")}

## model ordering from cv analysis.
ctp_abbv = dict(zip(cell_types, ["B", "CAF", "CE", "ENDO", "MYL", "NE", 
                                 "PB", "PVL", "T"]))
mdl_top = perf_test_val[perf_test_val.AUC.gt(
                perf_test_val.loc["Bulk", "AUC"])].index.tolist()              # cell-type-models that outperform Bulk

keep_top = 3
mdl_ord, mdl_names = ["Bulk"], ["Bulk"]
for nn in range(3):
    ## pick only 'nn+1' cell-type-ensembles by counting '+' and order by AUC.
    mdl_nn_ = sorted([mdl_ for mdl_ in mdl_top if mdl_.count("+") == nn], 
                     key = lambda mdl_: perf_test_val.loc[mdl_, "AUC"], 
                     reverse = True)[:keep_top]
    mdl_ord = mdl_nn_ + mdl_ord
    
    ## format cell type names (shorthand for ensembles).
    if nn > 0:                                                                 # ensemble models
        for ctp_, c_ in ctp_abbv.items():
            mdl_nn_ = [mdl_.replace(ctp_, c_) for mdl_ in mdl_nn_]
        mdl_nn_ = [mdl_.replace("+", "$-$") for mdl_ in mdl_nn_]
    else:                                                                      # individual models
        mdl_nn_ = [mdl_.replace("_", "\n") for mdl_ in mdl_nn_]
    
    mdl_names = mdl_nn_ + mdl_names
del mdl_nn_


## make comparison matrix.
assert(y_test.index.tolist() == samples_cm_sammut)                             # check if performances are calculated for the same samples
perf_test_comp = perf_test_val.loc[mdl_ord, ["AUC", "AP"]].copy()
perf_test_comp.loc["Sammut et al."] = perf_val_sammut.loc["Clinical+RNA"]
mdl_names.append("Sammut et al.")

## make base title.
fig_title = "Performance of cell-type-specific models"
if use_samples == "all":
    fig_title += " for the whole Artemis + PBCP cohorts "
elif use_samples == "chemo_targeted":
    fig_title += " for HER2+ patients treated with chemotherapy and Trastuzumab in PBCP cohorts "
elif use_samples == "chemo":
    fig_title += " for HER2- patients treated with chemotherapy alone in Artemis + PBCP cohorts "
fig_title += f"(n = {y_test.size})"


#%% make combined AUC/AP barplots.

svdat = False

## make compact plot.
fig_title1 = fig_title.replace("Performance", "Summary statistics").replace(
    "in", "\nin")

fig_data1 = perf_test_comp[["AUC", "AP"]].reset_index().rename(
    columns = {"index": "model"}).melt(
        id_vars = ["model"], var_name = "metric", value_name = "score")

fig1, ax1 = plt.subplots(figsize = (40, 14), nrows = 1, ncols = 1)
ax1 = make_barplot2(data = fig_data1, x = "model", y = "score", hue = "metric", 
                    width = 0.5, title = "ARTemis + PBCP", xlabels = mdl_names, 
                    xrot = 40, ax = ax1, fontdict = fontdict)
ax1.set_ylim([0, 1.04]);
fig1.tight_layout()
plt.show()


## save figure.
if svdat:
    # datestamp = date_time()
    datestamp = "23Mar2023"
    fig_path = data_path[0] + "plots/final_plots2/"
    if not os.path.exists(fig_path):                                           # create figure dir
        os.mkdir(fig_path)
    
    fig_file = f"sammut_validation_{use_samples}_AUC_AP_top_models_th{conf_th}_{use_mdl}_{num_feat_max}features_{datestamp}_v2.pdf"
    fig1.savefig(fig_path + fig_file, dpi = "figure")


#%% make combined OR/SEN barplots.

svdat = False

## make plots for all cell types. 
fig_data2 = perf_test_val.loc[mdl_ord, ["DOR", "SEN"]].reset_index(
    ).rename(columns = {"index": "model"})

## odds ratio.
fig_title21 = fig_title.replace("Performance", "Odds ratio").replace(
    "in", "\nin")
fig21, ax21 = plt.subplots(figsize = (36, 11), nrows = 1, ncols = 1)
ax21 = make_barplot1(data = fig_data2, x = "model", y = "DOR", ax = ax21, 
                     title = fig_title21, xlabels = mdl_names[:-1], 
                     bline = True, fontdict = fontdict)
fig21.tight_layout()
plt.show()

# sensitivity.
fig_title22 = fig_title.replace("Performance", "Sensitivity").replace(
    "in", "\nin")
fig22, ax22 = plt.subplots(figsize = (36, 11), nrows = 1, ncols = 1)
ax22 = make_barplot1(data = fig_data2, x = "model", y = "SEN", ax = ax22, 
                      title = fig_title22, xlabels = mdl_names[:-1], 
                      fontdict = fontdict)
fig22.tight_layout()
plt.show()


## save figures.
if svdat:
    # datestamp = date_time()
    datestamp = "23Mar2023"
    fig_path = data_path[0] + "plots/final_plots2/"
    if not os.path.exists(fig_path):                                           # create figure dir
        os.mkdir(fig_path)
    
    fig_file = f"sammut_validation_{use_samples}_OR_top_models_th{conf_th}_{use_mdl}_{num_feat_max}features_{datestamp}_v2.pdf"
    fig21.savefig(fig_path + fig_file, dpi = "figure")
    
    fig_file = f"sammut_validation_{use_samples}_SEN_top_models_th{conf_th}_{use_mdl}_{num_feat_max}features_{datestamp}_v2.pdf"
    fig22.savefig(fig_path + fig_file, dpi = "figure")


#%% make cell-type-specific AUC/AP barplots.

svdat = False

fig_title11 = "Performance summary for ARTemis + PBCP"

# cell_type_ord1 = [mdl for mdl in mdl_ord if mdl.count("+") == 0] + \
#     ["Sammut et al."]
cell_type_ord1 = ["CAFs", "Normal_Epithelial", "Cancer_Epithelial", 
                  "Endothelial", "Myeloid", "B-cells", "Plasmablasts", "Bulk"] + \
    ["Sammut et al."]
mdl_names_ord1 = [mdl.replace("_", "\n") for mdl in cell_type_ord1]

fig_data11 = perf_test_val[["AUC", "AP"]].copy()
fig_data11.loc["Sammut et al."] = perf_val_sammut.loc["Clinical+RNA"]
fig_data11 = fig_data11.loc[cell_type_ord1].rename_axis("model").reset_index(
    ).melt(id_vars = ["model"], var_name = "metric", value_name = "score")

sns.set_style("ticks")
plt.rcParams.update({"xtick.major.size": 12, "xtick.major.width": 4, 
                     "ytick.major.size": 12, "ytick.major.width": 4, 
                     # "xtick.color": [0.8]*3, "ytick.color": [0.8]*3, 
                     "xtick.bottom": True, "ytick.left": True})
fig11, ax11 = plt.subplots(figsize = (44, 16), nrows = 1, ncols = 1)
ax11 = make_barplot2(data = fig_data11, x = "model", y = "score", 
                     hue = "metric", ax = ax11, title = "ARTemis + PBCP", 
                     xlabels = mdl_names_ord1, xrot = 40, 
                     legend_title = "Metric", fontdict = fontdict)
ax11.grid(visible = True, which = "major", axis = "y", 
          color = [0.75]*3, linewidth = 1);
fig11.tight_layout()
plt.show()


## save figure.
if svdat:
    # datestamp = date_time()
    datestamp = "23Mar2023"
    fig_path = data_path[0] + "plots/final_plots4/"
    fig_file = f"sammut_validation_{use_samples}_AUC_AP_top_models_th{conf_th}_{use_mdl}_{num_feat_max}features_{datestamp}.pdf"
    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    fig11.savefig(fig_path + fig_file, dpi = "figure")


#%% make cell-type-combo AUC/AP barplots.

svdat = False

fig_title12 = "Performance summary for ARTemis + PBCP, two-cell-type ensembles"

cell_type_ord2 = [mdl for mdl in mdl_ord if mdl.count("+") == 1] + ["Bulk"]
mdl_names_ord2 = [mdl for mdl in mdl_names if mdl.count("$-$") == 1] + ["Bulk"]

fig_data12 = perf_test_val.loc[cell_type_ord2, ["AUC", "AP"]].rename_axis(
    "model").reset_index().melt(
        id_vars = ["model"], var_name = "metric", value_name = "score")

fig12, ax12 = plt.subplots(figsize = (24, 11), nrows = 1, ncols = 1)
ax12 = make_barplot2(data = fig_data12, x = "model", y = "score", 
                     hue = "metric", ax = ax12, title = fig_title12, 
                     xlabels = mdl_names_ord2, fontdict = fontdict)
fig12.tight_layout()
plt.show()


fig_title13 = "Performance summary for ARTemis + PBCP, three-cell-type ensembles"

cell_type_ord3 = [mdl for mdl in mdl_ord if mdl.count("+") == 2] + ["Bulk"]
mdl_names_ord3 = [mdl for mdl in mdl_names if mdl.count("$-$") == 2] + ["Bulk"]

fig_data13 = perf_test_val.loc[cell_type_ord3, ["AUC", "AP"]].rename_axis(
    "model").reset_index().melt(
        id_vars = ["model"], var_name = "metric", value_name = "score")

fig13, ax13 = plt.subplots(figsize = (24, 11), nrows = 1, ncols = 1)
ax13 = make_barplot2(data = fig_data13, x = "model", y = "score", 
                     hue = "metric", ax = ax13, title = fig_title13, 
                     xlabels = mdl_names_ord3, fontdict = fontdict)
fig13.tight_layout()
plt.show()


## save figure.
if svdat:
    # datestamp = date_time()
    datestamp = "23Mar2023"
    fig_path = data_path[0] + "plots/final_plots3/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file = f"sammut_validation_{use_samples}_AUC_AP_top_models_combo2_th{conf_th}_{use_mdl}_{num_feat_max}features_{datestamp}.pdf"
    fig12.savefig(fig_path + fig_file, dpi = "figure")
            
    fig_file = f"sammut_validation_{use_samples}_AUC_AP_top_models_combo3_th{conf_th}_{use_mdl}_{num_feat_max}features_{datestamp}.pdf"
    fig13.savefig(fig_path + fig_file, dpi = "figure")


#%% make cell-type-combo AUC/AP barplots - v2.

svdat = False

fig_title23 = ["Top two-cell-type ensembles for ARTemis + PBCP", 
               "Top three-cell-type ensembles for ARTemis + PBCP"]

cell_type_ord2 = [mdl for mdl in mdl_ord if mdl.count("+") == 1] + ["Bulk"]
# mdl_names_ord2 = [mdl.replace("$-$", " + ") for mdl in mdl_names if mdl.count("$-$") == 1] + ["Bulk"]
mdl_names_ord2 = ["\n" + mdl.replace("+", "\n+ ").replace("_", " ") for mdl in cell_type_ord2]
fig_data232 = perf_test_val.loc[cell_type_ord2, ["AUC", "AP"]].rename_axis(
    "model").reset_index().melt(
        id_vars = ["model"], var_name = "metric", value_name = "score")

cell_type_ord3 = [mdl for mdl in mdl_ord if mdl.count("+") == 2] + ["Bulk"]
# mdl_names_ord3 = [mdl.replace("$-$", " + ") for mdl in mdl_names if mdl.count("$-$") == 2] + ["Bulk"]
mdl_names_ord3 = ["\n" + mdl.replace("+", " \n+ ").replace("_", " ") for mdl in cell_type_ord3]
fig_data233 = perf_test_val.loc[cell_type_ord3, ["AUC", "AP"]].rename_axis(
    "model").reset_index().melt(
        id_vars = ["model"], var_name = "metric", value_name = "score")

sns.set_style("ticks")
plt.rcParams.update({"xtick.major.size": 12, "xtick.major.width": 4, 
                      "ytick.major.size": 12, "ytick.major.width": 4, 
                      # "xtick.color": [0.8]*3, "ytick.color": [0.8]*3, 
                      "xtick.bottom": True, "ytick.left": True})
fig23, (ax232, ax233) = plt.subplots(figsize = (56, 20), nrows = 1, 
                                     ncols = 2, sharey = True, )
ax232 = make_barplot2(data = fig_data232, x = "model", y = "score", 
                      hue = "metric", ax = ax232, title = " ", 
                      xlabels = mdl_names_ord2, xrot = 40, legend = False, 
                      fontdict = fontdict)
ax232.grid(visible = True, which = "major", axis = "y", 
            color = [0.75]*3, linewidth = 1);
ax233 = make_barplot2(data = fig_data233, x = "model", y = "score", 
                      hue = "metric", ax = ax233, title = " ", 
                      xlabels = mdl_names_ord3, xrot = 40, 
                      legend_title = "Metric", fontdict = fontdict)
ax233.grid(visible = True, which = "major", axis = "y", 
            color = [0.75]*3, linewidth = 1);

fig23.tight_layout(w_pad = -20)
plt.show()


## save figure.
if svdat:
    # datestamp = date_time()
    datestamp = "23Mar2023"
    fig_path = data_path[0] + "plots/final_plots3/"
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file = f"sammut_validation_{use_samples}_AUC_AP_top_combo_models_th{conf_th}_{use_mdl}_{num_feat_max}features_{datestamp}_v2.pdf"
    fig23.savefig(fig_path + fig_file, dpi = "figure")     