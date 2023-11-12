#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:27:49 2023

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
from machine_learning._functions import (
    EnsembleClassifier, train_pipeline, predict_proba_scaled, 
    get_best_threshold, classifier_performance, binary_performance)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import fisher_exact
from tqdm import tqdm


#%% functions.

## get confident genes for selected cell types.
## join: select genes based on its confidence level across cell types (any/all).
def get_conf_genes(conf_all, th = 0.99, ctps = None, join = "any"):
    if ctps is None:                                                           # for all available cell types
        genes = conf_all[conf_all.ge(th).apply(join, axis = 1)].index.tolist()
    else:
        genes = conf_all[conf_all[ctps].ge(th).apply(
            join, axis = 1)].index.tolist()
    
    return genes


## filter ligand-receptor list by cell type and/or genes.
def filter_cclr(lr_data, filter_by = None, ctps = None, genes = None):
    if filter_by is None:
        return lr_data
    else:
        keep_ctps  = lr_data[["LigandCell", "ReceptorCell"]].isin(
            ctps).apply("all", axis = 1)
        keep_genes = lr_data[["LigandGene", "ReceptorGene"]].isin(
            genes).apply("all", axis = 1)
        
        if filter_by.lower() == "cell":
            lr_data = lr_data[keep_ctps]
        elif filter_by.lower() == "gene":
            lr_data = lr_data[keep_genes]
        elif filter_by.lower() == "both":
            lr_data = lr_data[keep_ctps & keep_genes]
            
        return lr_data


## get MDI-based feature importance from RF. 
def get_feature_importance(pipe, scale = True):
    featimps  = pipe["classifier"].feature_importances_
    featnames = pipe.feature_names_in_[
        pipe["var_filter"].get_support()][pipe["selector"].get_support()]
    featimps  = pd.Series(featimps, index = featnames, name = "MDI")
    if scale:
        featimps = featimps / featimps.max()                                   # rescale between [0, 1] 
    
    return featimps    


## get feature association directionalities. 
def get_feature_association(data, featlist = None, alternative = "two-sided"):
    x, y = data[0].astype(bool), data[1].astype(bool)
    if featlist is None:
        featlist = x.columns.tolist()
    
    featassoc = { }
    for feat in tqdm(featlist):
        ctab = pd.DataFrame([[sum(x[feat] & y), sum(x[feat] & ~y)], 
                             [sum(~x[feat] & y), sum(~x[feat] & ~y)]], 
                            index = ["on", "off"], columns = ["R", "NR"])
        featassoc[feat] = fisher_exact(ctab, alternative = alternative)
    
    featassoc = pd.DataFrame(featassoc, index = ["OR", "pval"]).T
    
    return featassoc


#%% read data.

## specify sample subset.
use_samples = "chemo"
use_samples = use_samples.replace("+", "_")
              
## load data.
data_path = "../data/TransNEO/transneo_analysis/"
data_file = f"transneo_lirics_data_{use_samples}_v3.pkl"

with open(data_path + data_file, "rb") as file:
    data_obj   = pickle.load(file)
    cci_all    = data_obj["cci"]
    cclr_all   = data_obj["cclr"]
    resp_pCR   = data_obj["resp"]
    clin_info  = data_obj["clin"]
    cell_frac  = data_obj["frac"]
    conf_score = data_obj["conf"]

cell_types = conf_score.columns.tolist()


## combine two CCI lists into one.
cclr_all["all"] = pd.concat(cclr_all.values()).drop_duplicates()
cci_all["all"]  = pd.concat(cci_all.values()).reset_index().rename(
    columns = {"index": "CCLR"}).drop_duplicates().set_index("CCLR")

if cclr_all["all"].index.tolist() != cci_all["all"].index.tolist():
    raise ValueError("the CCLR lists are not the same between CCI data and annotations in the combined data!")


#%% prepare data.

## filter CCIs for cell types & genes to use.
top_ctps = ["Cancer_Epithelial", "Endothelial", "Myeloid", "Plasmablasts"]

filter_cci = True
if filter_cci:
    conf_th = 0.99                                                             # confident gene cut-off
    filter_by  = "both"                                                        # "both" / "cell" / "gene": what to filter by
    join_genes = "any"                                                         # "any" / "all": union / intersection of confident genesets
    use_ctps  = top_ctps
    use_genes = get_conf_genes(conf_score, th = conf_th, ctps = use_ctps, 
                               join = join_genes.lower())
    

## finalize data.
y_all = resp_pCR.copy()
cclr_list, X_all = { }, { }
for cclr_, cci_ in cci_all.items():
    ## get LR list.
    cclr_lr_ = cclr_all[cclr_].copy()
    if filter_cci:                                                             # filter LR list by cell types & genes
        cclr_lr_ = filter_cclr(cclr_lr_, filter_by = filter_by, 
                               ctps = use_ctps, genes = use_genes)
    
    ## get CCI data.
    X_cclr_ = cci_.loc[cclr_lr_.index, y_all.index].T
    if use_samples == "all":
        X_cclr_["aHER2"] = clin_info["aHER2.cycles"].notna().astype(int)       # flag for anti-HER2 therapy
    
    cclr_list[cclr_], X_all[cclr_] = cclr_lr_, X_cclr_

del cclr_lr_, X_cclr_


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


## cv parameters.
tune_seed = 84
cv_seed   = 4
cv_tune   = StratifiedKFold(n_splits = 3, shuffle = True, 
                            random_state = cv_seed)


#%% model per CCI list.

## get parameters.
# use_cclr      = ["ramilowski", "wang", "all"]                                  # all ligand-receptor lists
use_cclr      = ["ramilowski"]
num_split_rep = 5
num_splits    = 5
use_mets      = ["AUC", "AP", "ACC", "DOR", "SEN", "PPV", "SPC"]               # list of performance metrics to use


_tic = tic()

## start modeling per LR list.
y_pred_cclr, th_test_cclr, perf_test_cclr = { }, { }, { }
y_pred_cclr_full, th_test_cclr_full, perf_test_cclr_full = { }, { }, { }
pipes_all_cclr = { }
for use_cclr_ in use_cclr:
    ## get dataset.
    cclr_lr = cclr_list[use_cclr_]
    X_data, y_data = X_all[use_cclr_].copy(), y_all.copy()
    
    print(f"\nmodel = {use_mdl}, L-R list = {use_cclr_} (m = {cclr_lr.shape[0]}), max #features = {num_feat_max}")
    
    ## start modeling per repetition.
    y_pred_split_rep, th_test_split_rep, perf_test_split_rep = { }, { }, { }
    pipes_all_split_rep = { }
    for use_seed in range(num_split_rep):
        __tic = tic()
        print(f"\nsplit seed = {use_seed}")
        
        ## make cv splits.
        split_idx = StratifiedKFold(n_splits = num_splits, shuffle = True, 
                                    random_state = use_seed)
        split_idx = list(split_idx.split(X_all["all"], y_all))
        
        ## start modeling per cv fold.
        y_pred_split = pd.Series(dtype = float, index = y_data.index)
        th_test_split, perf_test_split = { }, { }
        pipes_all_split = { }
        for use_split, (train_idx, test_idx) in enumerate(split_idx):
            ## get training & test data.
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
            
            ## model response.
            try:                                                               # individual classifier
                pipe_tuned, params_tuned = train_pipeline(
                    model = use_mdl, train_data = (X_train, y_train), 
                    max_features = num_feat_max, var_th = var_th, 
                    cv_tune = cv_tune, mdl_seed = mdl_seed, 
                    tune_seed = tune_seed)
                
            except:                                                            # ensemble classifier
                ## step I: fit individual models.
                pipes_mdl = { };    params_mdl = { }
                for mdl in mdl_list:
                    pipes_mdl[mdl], params_mdl[mdl] = train_pipeline(
                        model = mdl, train_data = (X_train, y_train), 
                        max_features = num_feat_max, var_th = var_th, 
                        cv_tune = cv_tune, mdl_seed = mdl_seed, 
                        tune_seed = tune_seed)
                
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
            pipes_all_split[f"split{use_split}"] = pipe_tuned
        
        
        ## save results for this repetition.
        perf_test_split = pd.DataFrame(perf_test_split)
        perf_test_split["mean"] = perf_test_split.mean(axis = 1)
        print(f"performance: {perf_test_split['mean'].round(4).to_dict()}")
        
        y_pred_split_rep[f"seed{use_seed}"]    = y_pred_split
        th_test_split_rep[f"seed{use_seed}"]   = th_test_split
        perf_test_split_rep[f"seed{use_seed}"] = perf_test_split["mean"]
        pipes_all_split_rep[f"seed{use_seed}"] = pipes_all_split
        
        __tic.toc()
    
    
    ## overall results for this LR list.
    y_pred_split_rep              = pd.DataFrame(y_pred_split_rep)
    y_pred_split_rep["mean"]      = y_pred_split_rep.mean(axis = 1)
    th_test_split_rep             = pd.DataFrame(th_test_split_rep)
    th_test_split_rep.loc["mean"] = th_test_split_rep.mean(axis = 0)
    perf_test_split_rep           = pd.DataFrame(perf_test_split_rep)
    perf_test_split_rep["mean"]   = perf_test_split_rep.mean(axis = 1)
    
    ## calculate full prediction vector performance.
    y_pred_cclr_    = y_pred_split_rep["mean"]
    th_test_cclr_   = th_test_split_rep.loc["mean"].mean()
    y_pred_th_cclr_ = (y_pred_cclr_ >= th_test_cclr_).astype(int)
    perf_test_split_rep["full"] = pd.concat([
        pd.Series(classifier_performance(y_all, y_pred_cclr_)), 
        pd.Series(binary_performance(y_all, y_pred_th_cclr_)) ])[use_mets]
    
    print(os.system("clear"))                                                  # clears console
    print(f"\noverall performance for cell type = {use_cclr_} (m = {cclr_lr.shape[0]}): \n{perf_test_split_rep[['mean', 'full']].round(4)}")
    
    
    ## save results for this LR list.
    y_pred_cclr[use_cclr_]    = y_pred_split_rep
    th_test_cclr[use_cclr_]   = th_test_split_rep
    perf_test_cclr[use_cclr_] = perf_test_split_rep
    pipes_all_cclr[use_cclr_] = pipes_all_split_rep
    
    ## save full results for this LR list.
    y_pred_cclr_full[use_cclr_]    = y_pred_cclr_
    th_test_cclr_full[use_cclr_]   = th_test_cclr_
    perf_test_cclr_full[use_cclr_] = perf_test_split_rep["full"]
    
del y_pred_cclr_, y_pred_th_cclr_, th_test_cclr_


## performance tables.
y_pred_cclr_full    = pd.DataFrame(y_pred_cclr_full)
th_test_cclr_full   = pd.Series(th_test_cclr_full)
perf_test_cclr_full = pd.DataFrame(perf_test_cclr_full).T
perf_test_cclr_mean = pd.DataFrame({
    cclr_: perf_["mean"] for cclr_, perf_ in perf_test_cclr.items()}).T


print(os.system("clear"))                                                      # clears console
print(f"""\n{'-' * 64}
overall CV performance for TransNEO cohort (n = {y_all.size}): 
treatment = {use_samples}
cell_types = {use_ctps if filter_cci else 'all'}
available #CCIs = { {cclr_: len(cclr_lr_) for cclr_, cclr_lr_ in cclr_list.items()} }
model = {use_mdl}, max #features = {num_feat_max}
\n{perf_test_cclr_full.round(4)}""")


_tic.toc()


#%% save full prediction & performance tables.

svdat = False                                                                  # set True to save results 

if svdat:
    datestamp = date_time()
    
    ## save full predictions & performance.
    out_path = data_path + "mdl_data/"
    out_file = f"transneo_lirics_predictions_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.pkl"
    out_file = out_file.replace(f"_{use_mdl}", 
                                f"_filteredCCI_th{conf_th}_{use_mdl}" \
                                    if filter_cci else f"_allCCI_{use_mdl}")
    out_dict = {"label": y_all,          "pred": y_pred_cclr_full, 
                "th": th_test_cclr_full, "perf": perf_test_cclr_full, 
                "pipe": pipes_all_cclr}
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    with open(out_path + out_file, "wb") as file:
        pickle.dump(out_dict, file)

    ## save complete performance into xlsx file.
    out_path = _wpath_ + "results/"
    out_file = f"transneo_lirics_results_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.xlsx"
    out_file = out_file.replace(f"_{use_mdl}", 
                                f"_filteredCCI_th{conf_th}_{use_mdl}" \
                                    if filter_cci else f"_allCCI_{use_mdl}")
    out_dict = perf_test_cclr.copy()
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    write_xlsx(out_path + out_file, out_dict)
    

#%% get feature importance.
## RF: mean decrease in impurit (MDI), permutation-based importance
## default: MDI; accumulation of the gini impurity decrease within each tree

svdat = False                                                                  # set True to save results 

## parameters.
cclr      = "ramilowski"
pipes_all = pipes_all_cclr[cclr]

na_th  = 0.6                                                                   # NA fractions to allow for a feature
imp_th = 1e-3                                                                  # importance threshold for keeping a feature

## compute feature importance.
featimp_all = { }
for seed_ in range(num_split_rep):
    featimp_split_ = { }
    for split_ in range(num_splits):
        ## get feature importance per cv fold.
        pipe_ = pipes_all[f"seed{seed_}"][f"split{split_}"]
        featimp_ = get_feature_importance(pipe_, scale = True)
        featimp_split_[f"split{split_}"] = featimp_
    
    ## summarize importance for this repetition.
    featimp_split_ = pd.DataFrame(featimp_split_)
    featimp_split_.dropna(thresh = int(na_th * num_splits), inplace = True)    # keep feature if important for at least na_th fraction of cv folds
    featimp_split_["mean"] = featimp_split_.mean(axis = 1)
    featimp_all[f"seed{seed_}"] = featimp_split_["mean"]

## summarize importance across all repetitions.
featimp_all = pd.DataFrame(featimp_all).dropna(
    thresh = na_th * len(pipes_all))                                           # drop feature if not important for na_th fraction of repetitions
featimp_all["mean"] = featimp_all.mean(axis = 1)
featimp_all.sort_values(by = "mean", ascending = False, inplace = True)
featimp_all[["OR", "pval"]] = get_feature_association(
    data = (X_data, y_data), featlist = featimp_all.index)
featimp_all["pval_signed"] = featimp_all[["OR", "pval"]].apply(
    lambda res: res.iloc[1] * (1 if (res.iloc[0] > 1.0) else -1), axis = 1)


## finalize feature importance.
featimp_final = featimp_all[featimp_all["mean"].gt(imp_th)][
    ["mean", "pval_signed"]].rename(
        columns = {"mean": "MDI", "pval_signed": "Direction"}).rename_axis(
        "Lcell_Rcell_Lgene_Rgene").reset_index().replace(
        regex = {"_Epithelial": "-Epithelial"})
featimp_final[["LigandCell", "ReceptorCell", "LigandGene", "ReceptorGene"]] = \
    featimp_final["Lcell_Rcell_Lgene_Rgene"].apply(
        lambda x: x.split("_")).tolist()
featimp_final[["LigandCell", "ReceptorCell"]] = featimp_final[[
    "LigandCell", "ReceptorCell"]].replace(
        regex = {"-Epithelial": "_Epithelial"})
featimp_final["CCIannot"] = featimp_final.apply(
    lambda df: f"{df.LigandCell}-{df.ReceptorCell}::{df.LigandGene}-{df.ReceptorGene}", 
    axis = 1)
featimp_final = featimp_final.set_index("CCIannot")[[
    "LigandCell", "ReceptorCell", "LigandGene", "ReceptorGene", 
    "MDI", "Direction"]]


n_top = 20
print(f"\ntop {n_top} most predictive CCIs for TransNEO cohort = \n{featimp_final.iloc[:n_top, -2:]}")


## save data.
if svdat:
    datestamp = date_time()
    out_path  = data_path + "mdl_data/"
    out_file  = f"transneo_lirics_feature_importance_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.xlsx"
    out_file  = out_file.replace(f"_{use_mdl}", 
                                 f"_filteredCCI_th{conf_th}_{use_mdl}" \
                                     if filter_cci else f"_allCCI_{use_mdl}")
    out_dict  = {"MDI": featimp_final}
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    write_xlsx(out_path + out_file, out_dict)

