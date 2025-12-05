#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:53:23 2025

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
from miscellaneous import date_time, tic, write_xlsx
from itertools import combinations
from _functions import (MakeClassifier, EnsembleClassifier, train_pipeline, 
                        predict_proba_scaled, get_best_threshold, 
                        classifier_performance, binary_performance)
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from warnings import filterwarnings


#%% functions.

## load input data from saved pickle.
def load_data(path):
    with open(path, mode = "rb") as file:
        obj = pickle.load(file)
    
    exps, frac, conf = [obj[kk] for kk in ["exp", "frac", "conf"]]
    resp, clin       = [obj[kk] for kk in ["resp", "clin"]]
    del obj                                                                    # release memory
    
    return exps, resp, frac, conf, clin


## get confident genes for a cell type.
def get_conf_genes(conf, th = 0.99):
    genes = conf[conf.ge(th)].index.tolist()
    return genes


#%% read data.

use_samples = "chemo"

data_path = "../../data/TransNEO/transneo_analysis/"
data_file = f"transneo_data_{use_samples}_v2.pkl"

## load data.
print("loading & preparing data...");    _tic = tic()

exp_all, resp_pCR, cell_frac, conf_score, clin_info = \
    load_data(data_path + data_file)


if conf_score.columns.tolist() == cell_frac.columns.tolist():
    cell_types = conf_score.columns.tolist()
else:
    raise ValueError("cell types are not the same between cell fraction and confidence score!")


print("done!", end = "");    _tic.toc()


print(f"""
dataset summary:
cohort     = TransNEO (n = {resp_pCR.size:,}) [CV]
treatment  = {use_samples + 'therapy'}, response = RCB ('pCR' vs. 'RD')
cell types = {', '.join(cell_types)}
""")


#%%  prepare data for modeling.

conf_th = 0.99                                                                 # confident gene cut-off

y_all = resp_pCR.copy()
genes, X_all = { }, { }
for ctp_ in cell_types + ["Bulk"]:
    try:                                                                       # individual cell type
        gn_ctp_ = get_conf_genes(conf_score[ctp_], th = conf_th)
    except:                                                                    # Bulk
        gn_ctp_ = conf_score.index.tolist()
    
    X_ctp_ = exp_all[ctp_].loc[gn_ctp_, y_all.index].T.copy()
    genes[ctp_], X_all[ctp_] = gn_ctp_, X_ctp_

del ctp_, gn_ctp_, X_ctp_


print(f"""
dataset size: 
CV = { {ctp_: X_.shape for ctp_, X_ in X_all.items()} }
""", end = "")


#%% modeling parameters.

## input: cell types - individual / combo.
# use_ctp = "Bulk"                                                               # one individual cell type
# use_ctp = ["Cancer_Epithelial", "Endothelial"]                                 # two individual cell types
# use_ctp = ["Cancer_Epithelial", "Endothelial", "Myeloid", 
#            "Plasmablasts", "Bulk"]                                             # top cell types: chemo
# use_ctp = ["PVL", "B-cells", "Myeloid", "Bulk"]                                # top cell types: chemo + targeted

# use_ctp = ("B-cells", "Myeloid")                                               # B-M ensemble (input as tuple)
# use_ctp = ("B-cells", "Myeloid", "PVL")
# use_ctp = ("B-cells", "Myeloid", "Endothelial")
# use_ctp = ("B-cells", "Myeloid", "Endothelial", "PVL")
# use_ctp = ("B-cells", "Myeloid", "T-cells")
# use_ctp = "all"                                                                # all cell type ensemble

## input: list of individual cell types / combos.
use_ctp = np.append(cell_types, "Bulk").tolist()                               # all individual cell types + Bulk 
# use_ctp = list(combinations(cell_types, r = 2))                                # all two-cell-type ensembles 
# top_ctp = ["Cancer_Epithelial", "Endothelial", "Myeloid", "Plasmablasts"]
# use_ctp = list(combinations(top_ctp, r = 2)) + [("Bulk", )]                    # two-cell-type ensembles for top cell types

## format ctp parameter as [(ctp1), (ctp2), ...].
if isinstance(use_ctp, list):
    if not isinstance(use_ctp[0], tuple):
        use_ctp = [(ctp_, ) for ctp_ in use_ctp]
elif isinstance(use_ctp, tuple):
    use_ctp = [use_ctp]
elif isinstance(use_ctp, str):
    use_ctp = [(use_ctp, )]


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


## CV parameters.
tune_seed = 84
cv_tune   = LeaveOneOut()


#%% model response per classifier.

filterwarnings(action = "ignore")                                              # suppress fit failing/convergence warnings

## get parameters.
num_split_rep = 5
num_splits    = 5
use_mets      = ["AUC", "AP", "ACC", "DOR"]                                    # list of performance metrics to use


_tic = tic()

## start modeling per cell type.
y_pred_ctp, th_test_ctp, perf_test_ctp = { }, { }, { }
for use_ctp_ in use_ctp:
    ## combine cell type datasets (append cell type to gene names).
    ctp_list = cell_types if (use_ctp_[0] == "all") else use_ctp_
    y_data   = y_all.copy()
    X_data   = pd.concat([X_all[ctp_].rename(
        columns = lambda gn: f"{gn}__{ctp_}") for ctp_ in ctp_list], 
        axis = 1)
        
    print(f"""\n
    samples = {use_samples}, cell type = {'+'.join(use_ctp_)}
    available #genes = {X_data.shape[1]:,}, max #features = {num_feat_max}
    model = {use_mdl}, #repetitions = {num_split_rep}
    sample size: {np.ceil(X_data.shape[0] / num_splits):,}
    """, end = "")
    
    
    ## start modeling per repetition.
    y_pred_split_rep, th_test_split_rep, perf_test_split_rep = { }, { }, { }
    for use_seed in range(num_split_rep):
        __tic = tic()
        print(f"\nsplit seed = {use_seed}")        
        
        ## make CV splits.
        split_idx = StratifiedKFold(n_splits = num_splits, shuffle = True, 
                                    random_state = use_seed)
        split_idx = list(split_idx.split(X_all["Bulk"], y_all))
        
        ## start modeling per CV fold.
        y_pred_split = pd.Series(dtype = float, index = y_data.index)
        th_test_split, perf_test_split = { }, { }
        for use_split, (train_idx, test_idx) in enumerate(split_idx):
            ## get training & test data.
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
            
            ## train model.
            ## optimize for accuracy with LOO- as AUC can't be computed.
            try:                                                               # individual classifier
                pipe_tuned, params_tuned = train_pipeline(
                    model = use_mdl, train_data = (X_train, y_train), 
                    max_features = num_feat_max, var_th = var_th, 
                    cv_tune = cv_tune, mdl_seed = mdl_seed, 
                    tune_seed = tune_seed, scoring = "accuracy")
                
            except:                                                            # ensemble classifier
                ## step I: fit individual models.
                pipes_mdl, params_mdl = { }, { }
                for mdl in mdl_list:
                    pipes_mdl[mdl], params_mdl[mdl] = train_pipeline(
                        model = mdl, train_data = (X_train, y_train), 
                        max_features = num_feat_max, var_th = var_th, 
                        cv_tune = cv_tune, mdl_seed = mdl_seed, 
                        tune_seed = tune_seed, scoring = "accuracy")
                
                ## step II: get ensemble model.
                pipe_tuned = EnsembleClassifier(models = list(pipes_mdl.values()))
                pipe_tuned.fit(X_train, y_train)
                params_tuned = params_mdl.copy()
            
            ## get prediction performances.
            y_fit  = predict_proba_scaled(pipe_tuned, X_train, scale = True)
            th_fit = get_best_threshold(y_train, y_fit[:, 1], curve = "PR")    # classification threshold
            y_pred = predict_proba_scaled(pipe_tuned, X_test, scale = True)
            y_pred_th = (y_pred >= th_fit).astype(int)
            perf_test = pd.concat([
                pd.Series(classifier_performance(y_test, y_pred[:, 1])), 
                pd.Series(binary_performance(y_test, y_pred_th[:, 1])) ])
            
            
            ## save results for this split.
            y_pred_split.iloc[test_idx]          = y_pred[:, 1]
            th_test_split[f"split{use_split}"]   = th_fit
            perf_test_split[f"split{use_split}"] = perf_test[use_mets]
        
        
        ## save results for this repetition.
        perf_test_split = pd.DataFrame(perf_test_split)
        perf_test_split["mean"] = perf_test_split.mean(axis = 1)
        print(f"performance: {perf_test_split['mean'].round(4).to_dict()}")
        
        y_pred_split_rep[f"seed{use_seed}"]    = y_pred_split
        th_test_split_rep[f"seed{use_seed}"]   = th_test_split
        perf_test_split_rep[f"seed{use_seed}"] = perf_test_split["mean"]
        
        __tic.toc()
    
    
    ## overall results for this cell type.
    y_pred_split_rep            = pd.DataFrame(y_pred_split_rep)
    y_pred_split_rep["mean"]    = y_pred_split_rep.mean(axis = 1)
    th_test_split_rep           = pd.DataFrame(th_test_split_rep)
    perf_test_split_rep         = pd.DataFrame(perf_test_split_rep)
    perf_test_split_rep["mean"] = perf_test_split_rep.mean(axis = 1)
    
    print(os.system("clear"))                                                  # clears console
    print(f"\noverall performance for cell type = {'+'.join(use_ctp_)}: \n{perf_test_split_rep['mean'].round(4)}")
    
    
    ## save results for this cell type.
    y_pred_ctp["+".join(use_ctp_)]    = y_pred_split_rep
    th_test_ctp["+".join(use_ctp_)]   = th_test_split_rep
    perf_test_ctp["+".join(use_ctp_)] = perf_test_split_rep


## mean performance table.
perf_test_ctp_mean = pd.DataFrame({
    ctp_: perf_["mean"] for ctp_, perf_ in perf_test_ctp.items()}).T


## performance for full predictions.
y_pred_ctp_full  = pd.DataFrame({
    ctp_: pred_["mean"] for ctp_, pred_ in y_pred_ctp.items()})                # build full prediction
th_test_ctp_full = pd.Series({
    ctp_: ths_.mean().mean() for ctp_, ths_ in th_test_ctp.items()})

for ctp_, y_pred_ in y_pred_ctp_full.items():
    y_pred_th_ = (y_pred_ >= th_test_ctp_full[ctp_]).astype(int)
    perf_ctp_ = pd.concat([
        pd.Series(classifier_performance(y_all, y_pred_)), 
        pd.Series(binary_performance(y_all, y_pred_th_)) ])
    perf_test_ctp[ctp_]["full"] = perf_ctp_

del ctp_, y_pred_, perf_ctp_

perf_test_ctp_full = pd.DataFrame({
    ctp_: perf_["full"] for ctp_, perf_ in perf_test_ctp.items()}).T


print(f"""
{'-' * 64}
overall CV performance for treatment = {use_samples}:
cohort = TransNEO (n = {y_all.size:,})
model  = {use_mdl}, CV = Leave-one-out, repititions = None
\n{perf_test_ctp_full.round(4)}
""")

_tic.toc()


#%% save full prediction & performance tables.

svdat = False                                                                  # set as True to save data

if svdat:
    datestamp = date_time()
    
    ## save full predictions & performance.
    out_path = data_path + "mdl_data/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if doesn't exist
    
    out_file = f"transneo_predictions_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_LeaveOneOutCV_{datestamp}.pkl"
    out_dict = {"label": y_all,            "pred": y_pred_ctp_full, 
                "th"   : th_test_ctp_full, "perf": perf_test_ctp_full}
    with open(out_path + out_file, mode = "wb") as file:
        pickle.dump(out_dict, file)
    print(out_file)
    
    
    ## save complete performance into xlsx file.
    out_path = _wpath_ + "results/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if doesn't exist
    
    out_file = f"transneo_results_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_LeaveOneOutCV_{datestamp}.xlsx"
    out_dict = perf_test_ctp.copy()
    write_xlsx(out_path + out_file, data = out_dict)
    print(out_file)


#%% performance for Sammut samples.

sammut_path      = "../../data/TransNEO/use_data/"
sammut_file      = "transneo-diagnosis-MLscores.tsv"

y_pred_sammut    = pd.read_table(sammut_path + sammut_file, sep = "\t", 
                                 header = 0, index_col = None)
y_pred_sammut    = y_pred_sammut.pipe(
    lambda df: df[df["Class"].eq("Training")]).drop(
    columns = ["Class"]).set_index(
    keys = "Trial.ID")

y_data_sm        = y_data.filter(items = y_pred_sammut.index, axis = 0)
y_pred_sm        = y_pred_sammut.loc[y_data_sm.index]

perf_test_sammut = y_pred_sm.apply(
    lambda y_pred_: pd.Series(classifier_performance(y_data_sm, y_pred_))).T


## subset predictions.
y_pred_ctp_sm    = y_pred_ctp_full.filter(items = y_pred_sammut.index, axis = 0)

perf_test_ctp_sm = y_pred_ctp_sm.apply(
    lambda y_pred_: pd.Series(classifier_performance(y_data_sm, y_pred_))).T

print(f"""
{'-' * 64}
overall CV performance for TransNEO (n = {y_data_sm.size:,})
treatment = {use_samples}
{perf_test_ctp_sm.round(4)}
""")

