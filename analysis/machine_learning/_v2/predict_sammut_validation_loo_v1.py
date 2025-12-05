#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:51:59 2025

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
from itertools import combinations
from functools import reduce
from operator import add
from _functions import (MakeClassifier, EnsembleClassifier, train_pipeline, 
                        predict_proba_scaled, get_best_threshold, 
                        classifier_performance, binary_performance)
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut


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
use_mets = ["AUC", "AP", "ACC", "DOR", "SEN", "PPV", "SPC"]                    # list of performance metrics to use

## generate LOO splits.
cv_tune  = LeaveOneOut()


_tic = tic()

## start modeling per cell type.
y_pred_val    = { };    th_test_val     = { };    perf_test_val = { }
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
    model = {use_mdl}, training-cv = Leave-one-out
    sample size: training = {X_train.shape[0]}, test = {X_test.shape[0]}""")
    
    
    ## train model.
    try:                                                                   # individual classifier
        pipe_tuned, params_tuned = train_pipeline(
            model = use_mdl, train_data = (X_train, y_train), 
            max_features = num_feat_max, var_th = var_th, 
            cv_tune = cv_tune, mdl_seed = mdl_seed, 
            tune_seed = tune_seed, scoring = "accuracy")
        
    except:                                                                # ensemble classifier
        # step I: fit individual models.
        pipes_mdl = { };    params_mdl = { }
        for mdl in mdl_list:
            pipes_mdl[mdl], params_mdl[mdl] = train_pipeline(
                model = mdl, train_data = (X_train, y_train), 
                max_features = num_feat_max, var_th = var_th, 
                cv_tune = cv_tune, mdl_seed = mdl_seed, 
                tune_seed = tune_seed, scoring = "accuracy")
        
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
    
    print(f"performance for LOO = {perf_test[use_mets].round(4).to_dict()}")
    
    
    ## save results for this cell type.
    y_pred_val[ctp_mdl]    = y_pred[:, 1]
    th_test_val[ctp_mdl]   = th_fit
    perf_test_val[ctp_mdl] = perf_test[use_mets]


## fianl performance for all cell types.
y_pred_val    = pd.DataFrame(y_pred_val).set_index(X_test.index)               # mean prediction matrix
th_test_val   = pd.Series(th_test_val)
perf_test_val = pd.DataFrame(perf_test_val).T

# print(os.system("clear"))                                                    # clears console
print(f"""\n{'-' * 64}
validation performance for treatment = {use_samples}:
tuning CV = Leave-one-out
cohort = TransNEO validation (Artemis + PBCP; n = {y_test.size})
{perf_test_val.round(4)}""")

_tic.toc()


#%% save full prediction & performance tables.

svdat = False

if svdat:
    datestamp = date_time()
    
    ## save full predictions & performance.
    out_path = data_path[0] + "mdl_data/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    
    out_file = f"tn_valid_predictions_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_LeaveOneOutTune_{datestamp}.pkl"
    out_dict = {"label": y_test,      "pred": y_pred_val, 
                "th"   : th_test_val, "perf": perf_test_val}
    with open(out_path + out_file, "wb") as file:
        pickle.dump(out_dict, file)
    
    
    ## save complete performance into xlsx file.
    out_path = _wpath_ + "results/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    
    out_file = f"tn_valid_results_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_LeaveOneOutTune_{datestamp}.xlsx"
    out_dict = perf_test_val.copy()
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

