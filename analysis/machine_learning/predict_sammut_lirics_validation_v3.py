#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:03:52 2023

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
from _functions import (MakeClassifier, EnsembleClassifier, train_pipeline, 
                        predict_proba_scaled, get_best_threshold, 
                        classifier_performance, binary_performance)
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import fisher_exact
from tqdm import tqdm
from warnings import filterwarnings


#%% functions.

## load input CCI data from saved pickle.
def load_cci_data(path):
    with open(path, mode = "rb") as file:
        obj = pickle.load(file)
    
    ccis, cclr = [obj[kk] for kk in ["cci", "cclr"]]
    frac, conf = [obj[kk] for kk in ["frac", "conf"]]
    resp, clin = [obj[kk] for kk in ["resp", "clin"]]
    del obj                                                                    # release memory
    
    return ccis, cclr, resp, clin, conf, frac


## get confident genes for selected cell types.
## join: select genes based on its confidence level across cell types (any / all).
def get_conf_genes(conf, th = 0.99, ctps = None, join = "any"):
    use_conf = conf if (ctps is None) else conf[ctps]
    genes    = conf[use_conf.ge(th).apply(join, axis = 1)].index.tolist()
    return genes


## filter ligand - receptor list by cell type and/or genes.
def filter_cclr(data, by = None, ctps = None, genes = None):
    if by is None:    return data                                              # no filtering
    
    keep_ctps  = data[
        ["LigandCell", "ReceptorCell"]].isin(
        ctps).apply("all", axis = 1)
    keep_genes = data[
        ["LigandGene", "ReceptorGene"]].isin(
        genes).apply("all", axis = 1)
    
    match by.lower():
        case "cell":
            out = data[keep_ctps]
        case "gene":
            out = data[keep_genes]
        case "both":
            out = data[keep_ctps & keep_genes]
    
    return out


## get MDI-based feature importance from RF. 
def get_feature_importance(pipe, scale = True, name = "MDI"):
    featnames = pipe.feature_names_in_[
        pipe["var_filter"].get_support()][
        pipe["selector"].get_support()]
    
    featimps  = pd.Series(pipe["classifier"].feature_importances_, 
                          index = featnames, name = name)
    if scale:    featimps = featimps / featimps.max()
    
    return featimps    


## get feature association directionalities. 
def get_feature_association(data, featlist = None, alternative = "two-sided"):
    X, y = data[0].astype(bool), data[1].astype(bool)
    if featlist is None:    featlist = X.columns.tolist()
    
    featassoc = { }
    for feat in tqdm(featlist):
        ctab = pd.DataFrame([[sum(X[feat] & y), sum(X[feat] & ~y)], 
                             [sum(~X[feat] & y), sum(~X[feat] & ~y)]], 
                            index = ["on", "off"], columns = ["R", "NR"])
        featassoc[feat] = fisher_exact(ctab, alternative = alternative)
    
    featassoc = pd.DataFrame(featassoc, index = ["OR", "pval"]).T
    
    return featassoc


#%% read data.

use_samples = "chemo"

data_path = ["../../data/TransNEO/transneo_analysis/", 
             "../../data/TransNEO/TransNEO_SammutShare/validation/"]

data_file = [f"transneo_lirics_data_{use_samples}_v3.pkl", 
             f"transneo_validation_lirics_data_{use_samples}.pkl"]

## load data.
print("loading & preparing data...");    _tic = tic()

(cci_all_train, cclr_all_train, resp_pCR_train, clin_info_train, 
 conf_score_train, cell_frac_train) = load_cci_data(data_path[0] + data_file[0])

(cci_all_test, cclr_all_test, resp_pCR_test, clin_info_test, 
 conf_score_test, cell_frac_test) = load_cci_data(data_path[1] + data_file[1])


## get cell types & sample lists.
if conf_score_train.columns.tolist() == conf_score_test.columns.tolist():
    cell_types = conf_score_train.columns.tolist()
else:
    raise ValueError("the cell types between training & test are not the same!")


## combine two CCI lists into one.
cclr_all_train["all"] = pd.concat(
    cclr_all_train.values(), axis = 0).drop_duplicates(
    keep = "first")
cci_all_train["all"]  = pd.concat(
    cci_all_train.values(), axis = 0).reset_index(
    names = "CCLR").drop_duplicates(
    keep = "first").set_index(
    keys = "CCLR")

cclr_all_test["all"] = pd.concat(
    cclr_all_test.values(), axis = 0).drop_duplicates(
    keep = "first")
cci_all_test["all"]  = pd.concat(
    cci_all_test.values(), axis = 0).reset_index(
    names = "CCLR").drop_duplicates(
    keep = "first").set_index(
    keys = "CCLR")


## sanity checks.        
if cclr_all_train["all"].index.tolist() != cci_all_train["all"].index.tolist():
    raise ValueError("the CCLR lists are not the same between CCI data and annotations in the combined training data!")

if cclr_all_test["all"].index.tolist() != cci_all_test["all"].index.tolist():
    raise ValueError("the CCLR lists are not the same between CCI data and annotations in the combined test data!")


print("done!", end = "");    _tic.toc()


print(f"""
dataset summary:
cohorts    = TransNEO (n = {resp_pCR_train.size:,}) [training], ARTemis + PBCP (n = {resp_pCR_test.size:,}) [validation]
treatment  = {use_samples + 'therapy'}, response = RCB ('pCR' vs. 'RD')
cell types = {', '.join(cell_types)}
available #CCIs: 
    training   = { {lst: len(dat) for lst, dat in cclr_all_train.items()} }
    validation = { {lst: len(dat) for lst, dat in cclr_all_test.items()}}
""")


#%% prepare data.

## filter CCIs for cell types & genes to use.
filter_cci = True
if filter_cci:
    conf_th    = 0.99                                                          # confident gene cut-off
    filter_by  = "both"                                                        # 'both' / 'cell' / 'gene': what to filter by
    join_genes = "any"                                                         # 'any' / 'all': union / intersection of confident genesets
    use_ctps   = ["Cancer_Epithelial", "Endothelial", "Myeloid", "Plasmablasts"]
    use_genes = np.intersect1d(
        get_conf_genes(conf_score_train, th = conf_th, ctps = use_ctps, 
                       join = join_genes), 
        get_conf_genes(conf_score_test, th = conf_th, ctps = use_ctps, 
                       join = join_genes) ).tolist() 


## finalize data.
y_train, y_test = resp_pCR_train.copy(), resp_pCR_test.copy()

cclr_list, X_all_train, X_all_test = { }, { }, { }
for cclr_ in cclr_all_train:
    ## get L-R list.
    cclr_lr_ = cclr_all_train[cclr_].loc[
        np.intersect1d(cclr_all_train[cclr_].index, cclr_all_test[cclr_].index)]
    if filter_cci:                                                             # filter LR list by cell types & genes
        cclr_lr_ = filter_cclr(cclr_lr_, by = filter_by, ctps = use_ctps, 
                               genes = use_genes)
    
    ## get CCI data.
    X_train_cclr_ = cci_all_train[cclr_].loc[cclr_lr_.index, y_train.index].T
    X_test_cclr_  = cci_all_test[cclr_].loc[cclr_lr_.index, y_test.index].T
    cclr_list[cclr_], X_all_train[cclr_], X_all_test[cclr_] = \
        cclr_lr_, X_train_cclr_, X_test_cclr_

del cclr_lr_, X_train_cclr_, X_test_cclr_


print(f"""
dataset sizes: 
train = { {ctp_: X_.shape for ctp_, X_ in X_all_train.items()} }
test  = { {ctp_: X_.shape for ctp_, X_ in X_all_test.items()} }
""", end = "")


#%% modeling parameters.

## model parameters.
num_feat_max = "all"                                                           # maximum #features to use
var_th       = 0.08
mdl_seed     = 86420


## choose classifier: LR, RF, SVM, XGB, ENS1 (L+R+S), ENS2 (L+R+S+X).
use_mdl = "RF"
use_mdl = use_mdl.upper()
mdl_list_ind = ["LR", "RF", "SVM", "XGB"]                                      # individual classifier list
if use_mdl == "ENS1":
    mdl_list = np.setdiff1d(mdl_list_ind, "XGB").tolist()
elif use_mdl == "ENS2":
    mdl_list = mdl_list_ind.copy()
elif use_mdl =="ENS3":
    mdl_list = np.setdiff1d(mdl_list_ind, ["LR", "XGB"]).tolist()


## CV parameters.
tune_seed = 84
cv_seed   = 4


#%% model response per classifer.

filterwarnings(action = "ignore")                                              # suppress fit failing/convergence warnings

## get parameters.
num_split_rep   = 5
num_splits      = 3
stratify_splits = False
use_cclr        = ["ramilowski", "wang", "all"]                                # all L-R lists
use_mets        = ["AUC", "AP", "ACC", "DOR"]                                  # list of performance metrics to use


_tic = tic()

## start modeling per LR list.
y_pred_val, th_test_val, perf_test_val = { }, { }, { }
pipes_test_val = { }
for use_cclr_ in use_cclr:
    ## get dataset.
    cclr_lr         = cclr_list[use_cclr_]
    X_train, X_test = X_all_train[use_cclr_], X_all_test[use_cclr_]
    
    print(f"""\n
    samples = {use_samples}, L-R list = {use_cclr_} 
    available #CCIs = {X_train.shape[1]:,}, max #features = {num_feat_max}
    model = {use_mdl}, #repetitions = {num_split_rep}
    sample size: training = {X_train.shape[0]:,}, test = {X_test.shape[0]:,}
    """, end = "")
        
    
    ## start modeling per repetition.
    y_pred_rep = { };    th_test_rep = { };    perf_test_rep = { } 
    pipes_test_rep = { }
    for use_seed in range(num_split_rep):
        print(f"\nsplit seed = {use_seed}")
        
        ## make CV splits for tuning.
        cv_tune = StratifiedKFold if stratify_splits else KFold
        cv_tune = cv_tune(n_splits = num_splits, shuffle = True, 
                          random_state = use_seed)
        
        ## train model.
        try:                                                                   # individual classifier
            pipe_tuned, params_tuned = train_pipeline(
                model = use_mdl, train_data = (X_train, y_train), 
                max_features = num_feat_max, var_th = var_th, 
                cv_tune = cv_tune, mdl_seed = mdl_seed, 
                tune_seed = tune_seed, scoring = "roc_auc")
            
        except:                                                                # ensemble classifier
            # step I: fit individual models.
            pipes_mdl = { };    params_mdl = { }
            for mdl in mdl_list:
                pipes_mdl[mdl], params_mdl[mdl] = train_pipeline(
                    model = mdl, train_data = (X_train, y_train), 
                    max_features = num_feat_max, var_th = var_th, 
                    cv_tune = cv_tune, mdl_seed = mdl_seed, 
                    tune_seed = tune_seed, scoring = "roc_auc")
            
            # step II: get ensemble model.
            pipe_tuned = EnsembleClassifier(models = list(pipes_mdl.values()))
            pipe_tuned.fit(X_train, y_train)
            params_tuned = params_mdl.copy()
        
        
        ## get prediction performances.
        y_fit  = predict_proba_scaled(pipe_tuned, X_train, scale = True)
        th_fit = get_best_threshold(y_train, y_fit[:, 1], curve = "PR")        # classification threshold
        y_pred = predict_proba_scaled(pipe_tuned, X_test, scale = True)
        y_pred_th = (y_pred >= th_fit).astype(int)
        perf_test = pd.concat([
                pd.Series(classifier_performance(y_test, y_pred[:, 1])), 
                pd.Series(binary_performance(y_test, y_pred_th[:, 1])) ])
        
        print(f"performance for seed {use_seed} = {perf_test[use_mets].round(4).to_dict()}")
        
        
        ## save results for this repetition.
        y_pred_rep[f"seed{use_seed}"]     = y_pred[:, 1]
        th_test_rep[f"seed{use_seed}"]    = th_fit
        perf_test_rep[f"seed{use_seed}"]  = perf_test[use_mets]
        pipes_test_rep[f"seed{use_seed}"] = pipe_tuned
        
            
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
    print(f"\noverall performance for L-R list = {use_cclr_} (m = {cclr_lr.shape[0]}): \n{perf_test_all.round(4)}")
    
    
    ## save results for this L-R list.
    y_pred_val[use_cclr_]     = y_pred_full
    th_test_val[use_cclr_]    = th_test_rep
    perf_test_val[use_cclr_]  = perf_test_all["mean_pred"]
    # perf_test_val[use_cclr_]  = perf_test_all["mean_perf"]
    pipes_test_val[use_cclr_] = pipes_test_rep
    

## fianl performance for all L-R lists.
y_pred_val    = pd.DataFrame(y_pred_val).set_index(keys = X_test.index)        # mean prediction matrix
th_test_val   = pd.DataFrame(th_test_val).T
perf_test_val = pd.DataFrame(perf_test_val).T


## performance for ensemble model.
y_pred_val["ensemble"] = y_pred_val.drop(columns = "all").mean(axis = 1)
th_mean_val            = th_test_val.drop(index = "all")["mean"].mean(axis = 0)
y_pred_val_ens_th      = y_pred_val["ensemble"].ge(th_mean_val).astype(int)
perf_test_val_ens      = pd.concat([
    pd.Series(classifier_performance(y_test, y_pred_val["ensemble"])), 
    pd.Series(binary_performance(y_test, y_pred_val_ens_th)) ])
perf_test_val.loc["ensemble"] = perf_test_val_ens[use_mets]

del th_mean_val, y_pred_val_ens_th, perf_test_val_ens


# print(os.system("clear"))                                                      # clears console
print(f"""
{'-' * 64}
validation performance for treatment = {use_samples}:
using cell types: {', '.join(use_ctps) if filter_cci else 'all {len(cell_types)} cell types'}
cohort = Artemis + PBCP (n = {y_test.size:,})
#CCIs  = { {lst: len(dat) for lst, dat in cclr_list.items()} }
model  = {use_mdl}, CV = {'Stratified ' if stratify_splits else ''}{num_splits}-fold, #repititions = {num_split_rep}
\n{perf_test_val.round(4)}
""")

_tic.toc()


#%% save full prediction & performance tables.

svdat = False                                                                  # set as True to save data

if svdat:
    datestamp = date_time()
        
    ## save full predictions & performance.
    out_path = data_path[0] + "mdl_data/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    
    out_file = f"tn_valid_lirics_predictions_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.pkl"
    out_file = out_file.replace(f"_{use_mdl}", 
                                f"_filteredCCI_th{conf_th}_{use_mdl}" \
                                    if filter_cci else f"_allCCI_{use_mdl}")
    
    out_dict = {"label": y_test,   "pred": y_pred_val, 
                "th": th_test_val, "perf": perf_test_val, 
                "pipe": pipes_test_val}
    with open(out_path + out_file, mode = "wb") as file:
        pickle.dump(out_dict, file)
    print(out_file)
    

    ## save complete performance into xlsx file.
    out_path = _wpath_ + "results/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    
    out_file = f"tn_valid_lirics_results_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.xlsx"
    out_file = out_file.replace(f"_{use_mdl}", 
                                f"_filteredCCI_th{conf_th}_{use_mdl}" \
                                    if filter_cci else f"_allCCI_{use_mdl}")
    
    out_dict = perf_test_val.copy()
    write_xlsx(out_path + out_file, data = out_dict)
    print(out_file)
    
    
    ## save list of CCIs used.
    out_path = data_path[0] + "mdl_data/"
    out_file = f"tn_valid_lirics_feature_list_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.xlsx"
    out_dict = cclr_list.copy()
    write_xlsx(out_path + out_file, data = out_dict)
    print(out_file)


#%% compute feature importance.
## RF: mean decrease in impurity (MDI), permutation-based importance
## default: MDI; accumulation of the Gini impurity decrease within each tree

svdat = False                                                                  # set as True to save data

## get parameters.
na_th     = 0.6                                                                # missing value cut-off
imp_th    = 1e-3                                                               # importance cut-off
cclr      = "ramilowski"
pipes_all = pipes_test_val[cclr]


## compute importance.
featimp_all = pd.DataFrame({sd_: get_feature_importance(pipe_, scale = True) 
                            for sd_, pipe_ in pipes_all.items()})
featimp_all.dropna(thresh = na_th * len(pipes_all), inplace = True)            # drop feature if not important for na_th fraction of repetitions
featimp_all["mean"]         = featimp_all.mean(axis = 1)
featimp_all[["OR", "pval"]] = get_feature_association(
    data = (X_train, y_train), featlist = featimp_all.index)
featimp_all["pval_signed"] = featimp_all[["OR", "pval"]].apply(
    lambda res: res.iloc[1] * (1 if (res.iloc[0] > 1) else -1),                # directionality: + / - if OR > 1 / < 1
    axis = 1)
featimp_all.sort_values(by = "mean", ascending = False, inplace = True)


## finalize feature importance.
featimp_final = featimp_all[
    featimp_all["mean"].gt(imp_th)][
    ["mean", "pval_signed"]].set_axis(
    labels = ["MDI", "Direction"], axis = 1).reset_index(
    names = "CCI").reset_index().replace(
    regex = {"_Epithelial": "-Epithelial"})
featimp_final[["LigandCell", "ReceptorCell", "LigandGene", "ReceptorGene"]] = \
    featimp_final["CCI"].apply(lambda x: x.split("_")).tolist()
featimp_final[["LigandCell", "ReceptorCell"]] = featimp_final[
    ["LigandCell", "ReceptorCell"]].replace(
    regex = {"-Epithelial": "_Epithelial"})
featimp_final["CCIannot"] = featimp_final.apply(
    lambda df: f"{df.LigandCell}-{df.ReceptorCell}::{df.LigandGene}-{df.ReceptorGene}", 
    axis = 1)
featimp_final = featimp_final.set_index(
    keys = "CCIannot")[
    ["LigandCell", "ReceptorCell", "LigandGene", "ReceptorGene", 
     "MDI", "Direction"]]

n_disp = 20
print(f"""
top {n_disp} most predictive CCIs for ARTemis + PBCP (n = {y_test.size:,}): 
{featimp_final.iloc[:n_disp, -2:].map(lambda x: f"{x:0.4}")}
""")


## save data.
if svdat:
    datestamp = date_time()
    out_path  = data_path[0] + "mdl_data/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    
    out_file  = f"tn_valid_lirics_feature_importance_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.xlsx"
    out_file  = out_file.replace(f"_{use_mdl}", 
                                 f"_filteredCCI_th{conf_th}_{use_mdl}" \
                                     if filter_cci else f"_allCCI_{use_mdl}")
    
    out_dict  = {"MDI": featimp_final}
    write_xlsx(out_path + out_file, data = out_dict)
    print(out_file)

