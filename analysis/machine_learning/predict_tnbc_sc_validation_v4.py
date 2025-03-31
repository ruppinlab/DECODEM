#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:27:45 2023

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
# import matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
# from argparse import ArgumentParser
from scipy.stats import mannwhitneyu
from miscellaneous import date_time, tic, write_xlsx
from _functions import (MakeClassifier, EnsembleClassifier, train_pipeline, 
                        predict_proba_scaled, get_best_threshold, 
                        classifier_performance, binary_performance)
from sklearn.model_selection import StratifiedKFold, KFold
from warnings import filterwarnings
from copy import copy
from tqdm import tqdm


#%% functions.

## get confident genes for a cell type in deconvolved data.
def get_conf_genes(conf, th = 0.99):
    genes = conf[conf.ge(th)].index.tolist()
    return genes


## get nonzero genes for a cell type in sc data.
## nonzero genes: genes with a certain percentage of non-zero cells.
def get_nz_genes(exp, th = 0.05):
    exp   = exp[exp.var(axis = 1).ne(0)]                                       # drop non-variable genes
    genes = exp[exp.ne(0).mean(axis = 1).gt(th)].index.tolist()
    return genes    


## get top variable genes for a cell type in deconvolved / sc data.
def get_var_genes(exp, th = 0.2):
    genes = exp[exp.var(axis = 1).ge(th)].index.tolist()
    return genes


#%% read data.

# use_samples = "chemo"                                                          # chemo / chemo_immuno
use_samples = "chemo_immuno"                                                   # chemo / chemo_immuno

data_path = ["../../data/TransNEO/transneo_analysis/", 
             "../../data/SC_data/ZhangTNBC2021/validation/"]

data_file = ["transneo_data_chemo_v2.pkl", 
             f"tnbc_sc_data_{use_samples}_v2.pkl"]

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
    exp_all_sc_test = data_test_obj["exp"]
    resp_PR_test    = data_test_obj["resp"]
    cell_ids_test   = data_test_obj["cells"]
    clin_info_test  = data_test_obj["clin"]
    del data_test_obj


## prepare data.
if "PseudoBulk" in exp_all_sc_test.keys():
    exp_all_sc_test["Bulk"] = exp_all_sc_test["PseudoBulk"].copy()


clin_info_test["Response"] = clin_info_test.Efficacy.eq("PR").astype(int)      # add response column just in case (CR/PR = 1; no CR in data)
if "Cell.id" in clin_info_test.columns:
    clin_info_test.set_index("Cell.id", inplace = True)


resp_PR_test.sort_values(by = ["Patient.id", "Sample.id"], ascending = True, 
                         inplace = True)


## get overlaps.
cell_types = np.intersect1d(
    list(exp_all_train.keys()), list(exp_all_sc_test.keys())).tolist()         # common cell types
if "Bulk" in cell_types:
    cell_types = np.append(np.setdiff1d(cell_types, "Bulk"), "Bulk").tolist()

samples_train = sorted(clin_info_train.index)
samples_test  = np.unique(clin_info_test["Sample.id"]).tolist()



print(f"""dataset summary:
training cohort = TransNEO (n = {resp_pCR_train.size:,})
validation cohort (SC) = Zhang TNBC 2021 (n = {resp_PR_test['Patient.id'].nunique():,}, nCells = {resp_PR_test.size:,})
treatment = {use_samples + 'therapy'}, response = RECIST ('CR'/'PR' vs. 'SD'/'PD')
available cell types = {cell_types}
""")


#%% prepare data.

conf_th = 0.99                                                                 # confident gene cut-off for decon
var_top = 3500                                                                 # #highly-variable-genes for sc
genes, X_all_train, X_all_test = { }, { }, { }
for ctp_ in tqdm(cell_types):
    ## get genes to use.
    try:                                                                       # cell types
        gn_ctp_ = exp_all_sc_test[ctp_].filter(
            items = get_conf_genes(conf_score_train[ctp_], th = conf_th), 
            axis = 0).var(
            axis = 1).sort_values(
            ascending = False).iloc[
            :var_top].index.tolist()
    except:                                                                    # bulk
        gn_ctp_ = np.intersect1d(
            conf_score_train.index, 
            exp_all_sc_test[ctp_].index).tolist()
    
    ## get expression data.
    if ctp_ != "Bulk":
        cells_ctp_test_ = cell_ids_test[ctp_].groupby("Sample.id")
        X_ctp_test_     = { }
        for smpl_ in samples_test:
            try:
                cells_smpl_ = cells_ctp_test_.get_group(smpl_)["Cell.id"]
                if len(cells_smpl_) >= 3:                                      # need to have 3+ cells per sample
                    exp_smpl_ = exp_all_sc_test[ctp_][cells_smpl_].mean(
                        axis = 1)
                    X_ctp_test_[smpl_] = exp_smpl_[gn_ctp_]
            except:                                                            # cell type not present in this sample
                continue
        X_ctp_test_ = pd.DataFrame(X_ctp_test_).T
    else:
        X_ctp_test_ = exp_all_sc_test[ctp_].loc[gn_ctp_, samples_test].T
    X_ctp_test_.rename(columns = lambda gn: f"{gn}__{ctp_}", inplace = True)
    
    X_ctp_train_ = exp_all_train[ctp_].loc[gn_ctp_, samples_train].T.rename(
        columns  = lambda gn: f"{gn}__{ctp_}")
    
    ## save data.
    genes[ctp_], X_all_train[ctp_], X_all_test[ctp_] = \
        gn_ctp_, X_ctp_train_, X_ctp_test_

del (ctp_, gn_ctp_, X_ctp_train_, X_ctp_test_, cells_ctp_test_, 
     smpl_, cells_smpl_, exp_smpl_)

## get response labels.
y_all_train = resp_pCR_train.loc[samples_train].copy()
y_all_test  = resp_PR_test.groupby("Sample.id").first().loc[
    samples_test, "Response"]

print(f"""\ndataset sizes: 
      train = { {ctp_: X_.shape for ctp_, X_ in X_all_train.items()} }
      test  = { {ctp_: X_.shape for ctp_, X_ in X_all_test.items()} }""")


#%% modeling parameters.

use_ctp = copy(cell_types)
# use_ctp = "B-cells"

## format cell types list.
if isinstance(use_ctp, list):
    if not isinstance(use_ctp[0], tuple):
        use_ctp = [tuple([ctp_]) for ctp_ in use_ctp]
elif isinstance(use_ctp, tuple):
    use_ctp = [use_ctp]
elif isinstance(use_ctp, str):
    use_ctp = [tuple([use_ctp])]


## model parameters.
num_feat_max = "all"                                                           # maximum #features to use
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

filterwarnings(action = "ignore")                                              # suppress fit failing/convergence warnings


## get parameters.
num_split_rep   = 5
num_splits      = 3
stratify_splits = False
use_mets        = ["AUC", "AP", "ACC", "DOR", "SEN", "PPV", "SPC"]               # list of performance metrics to use


_tic = tic()

## start modeling per cell type.
y_pred_val = { };    th_test_val = { };     perf_test_val = { }
for use_ctp_ in use_ctp:
    ## get training & test sets.
    ctp_list = tuple(cell_types) if (use_ctp_[0] == "all") else use_ctp_
    X_train = pd.concat([X_all_train[ctp_] for ctp_ in ctp_list], axis = 1)
    X_test  = pd.concat([X_all_test[ctp_] for ctp_ in ctp_list], axis = 1)
    y_train = y_all_train.loc[X_train.index]
    y_test  = y_all_test.loc[X_test.index]
    
    print(f"""\n
    samples = {use_samples}, cell type = {"+".join(use_ctp_)}
    available #genes = {X_train.shape[1]}, max #features = {num_feat_max}
    model = {use_mdl}, #repetitions = {num_split_rep}
    sample size: training = {X_train.shape[0]}, test = {X_test.shape[0]}""")
    
    
    ## start modeling per repition.
    y_pred_rep = { };    th_test_rep =  { };     perf_test_rep = { }
    for use_seed in range(num_split_rep):
        print(f"\nsplit seed = {use_seed}")
        
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
        y_pred_rep[f"seed{use_seed}"]    = y_pred[:, 1]
        th_test_rep[f"seed{use_seed}"]   = th_fit
        perf_test_rep[f"seed{use_seed}"] = perf_test[use_mets]
    
    
    ## overall performance across all repetitions.
    y_pred_rep            = pd.DataFrame(y_pred_rep, index = y_test.index)
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
    y_pred_val["+".join(use_ctp_)]    = y_pred_full
    th_test_val["+".join(use_ctp_)]   = th_test_rep
    perf_test_val["+".join(use_ctp_)] = perf_test_all["mean_pred"]
    # perf_test_val["+".join(use_ctp_)] = perf_test_all["mean_perf"]
    

## fianl performance for all cell types.
y_pred_val    = pd.DataFrame(y_pred_val)                                       # mean prediction matrix
th_test_val   = pd.DataFrame(th_test_val).T
perf_test_val = pd.DataFrame(perf_test_val).T


## check R vs. NR score difference.
y_pred_diff = pd.DataFrame({
    ctp_: mannwhitneyu(y_pred_[y_test.eq(1)], y_pred_[y_test.eq(0)], 
                       alternative = "greater", nan_policy = "omit") 
    for ctp_, y_pred_ in y_pred_val.items()}, index = ["U1", "pval"]).T
y_pred_diff["pval_sig"] = y_pred_diff.pval.map(
    lambda p: ("***" if (p <= 0.001) else "**" if (p <= 0.01) else 
               "*" if (p <= 0.05) else "ns"))


# print(os.system("clear"))                                                    # clears console
print(f"""\n{'-' * 64}
validation performance for treatment = {use_samples}:
cohort = Zhang TNBC SC Cohort (n = {y_test.size})
{perf_test_val.round(4)}""")

_tic.toc()


#%% save full prediction & performance tables.

svdat = False

if svdat:
    datestamp = date_time()
    
    ## save full predictions & performance.
    out_path = data_path[0] + "mdl_data/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    
    out_file = f"zhangTNBC2021_predictions_{use_samples}_th{conf_th}_top{var_top}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.pkl"
    out_dict = {"label": y_all_test,  "pred": y_pred_val, 
                "th"   : th_test_val, "perf": perf_test_val}
    
    with open(out_path + out_file, "wb") as file:
        pickle.dump(out_dict, file)
    print(out_file)
    
    
    ## save complete performance into xlsx file.
    out_path = _wpath_ + "results/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    
    out_file = f"zhangTNBC2021_results_{use_samples}_th{conf_th}_top{var_top}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.xlsx"
    out_dict = perf_test_val.copy()
    
    write_xlsx(out_path + out_file, out_dict)
    print(out_file)
    
