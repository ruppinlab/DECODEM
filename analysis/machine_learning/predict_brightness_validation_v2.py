#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:55:04 2023

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
from miscellaneous import date_time, tic, write_xlsx
from itertools import combinations
from functools import reduce
from operator import add
from machine_learning._functions import (
    EnsembleClassifier, train_pipeline, predict_proba_scaled, 
    get_best_threshold, classifier_performance, binary_performance)
from sklearn.model_selection import StratifiedKFold, KFold


#%% functions.

## get confident genes for a cell type.
def get_conf_genes(conf, th = 0.99):
    genes = conf[conf.ge(th)].index.tolist()
    return genes


#%% read data.

use_samples = "chemo"

data_path = ["../data/TransNEO/transneo_analysis/", 
             "../data/BrighTNess/validation/"]

data_file = [f"transneo_data_{use_samples}_v2.pkl", 
             f"brightness_data_{use_samples}_v2.pkl"]


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
top_cell_types = ["Cancer_Epithelial", "Endothelial", "Myeloid", 
                  "Plasmablasts", "B-cells", "Normal_Epithelial"]
use_ctp = np.append(cell_types, "Bulk").tolist()                               # all individual cell types + Bulk 
# use_ctp = list(combinations(cell_types, r = 2))                                # all two-cell-type ensembles 
# use_ctp = list(combinations(cell_types, r = 3))                                # all three-cell-type ensembles 

# ## top cell types + two-cell ensembles + three-cell ensembles.
# use_ctp = list(combinations(cell_types, r = 1)) + \
#             reduce(add, [list(combinations(top_cell_types, r)) \
#                           for r in range(2, 4)]) + [("Bulk", )]

# use_ctp = list(combinations(top_cell_types, r = 2)) + [("Bulk", )]
# use_ctp = list(combinations(top_cell_types, r = 3)) + [("Bulk", )]

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

## dataset options.
## arm B: paclitaxel + carboplatin (TC), arm C: paclitaxel (T).
use_arm = "B"                                                                  # use None for both arms
if use_arm is not None:
    arm_samples = clin_info_test[
        clin_info_test["planned_arm_code"] == use_arm.upper()].index.tolist()
    X_all_test = {ctp_: X_.loc[arm_samples] for ctp_, X_ in X_all_test.items()}
    y_test     = y_test.loc[arm_samples]

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
cohort = BrighTNess, Arm B (n = {y_test.size})
{perf_test_val.round(4)}""")

_tic.toc()


#%% save full prediction & performance tables.

svdat = False                                                                  # set True to save results 

if svdat:
    datestamp = date_time()
    
    ## save full predictions & performance.
    out_path = data_path[0] + "mdl_data/"
    out_file = f"brightness_predictions_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.pkl"
    out_dict = {"label": y_test,   "pred": y_pred_val, 
                "th": th_test_val, "perf": perf_test_val}
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist 
    with open(out_path + out_file, "wb") as file:
        pickle.dump(out_dict, file)
    
    
    ## save models for feature importance.
    out_file = f"brightness_models_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.pkl"
    out_dict = {"pipeline": pipe_test_val, "params": params_test_val, 
                "train": X_train_all,      "test": X_test_all}
    
    with open(out_path + out_file, "wb") as file:
        pickle.dump(out_dict, file)
        
    
    ## save complete performance into xlsx file.
    out_path = _wpath_ + "results/"
    out_file = f"brightness_results_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.xlsx"
    out_dict = perf_test_val.copy()
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist 
    write_xlsx(out_path + out_file, out_dict)

