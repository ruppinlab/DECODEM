#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:27:49 2023

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
import matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
# from argparse import ArgumentParser
# from itertools import product
from miscellaneous import date_time, tic, write_xlsx
from _functions import (MakeClassifier, EnsembleClassifier, train_pipeline, 
                        predict_proba_scaled, get_best_threshold, 
                        classifier_performance, binary_performance, 
                        make_performance_plot)
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
        featimps = featimps / featimps.max()
    
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
# use_samples = "all"
# use_samples = "chemo+targeted"
use_samples = "chemo"
use_samples = use_samples.replace("+", "_")
              
## load data.
data_path = "../../data/TransNEO/transneo_analysis/"
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
top_ctps = {
    "all": ["B-cells", "Myeloid", "Plasmablasts"], 
    "chemo_targeted": ["PVL", "B-cells", "Myeloid"], 
    "chemo": ["Cancer_Epithelial", "Endothelial", "Myeloid", "Plasmablasts"] }

filter_cci = True
if filter_cci:
    conf_th = 0.99                                                             # confident gene cut-off
    filter_by  = "both"                                                        # "both" / "cell" / "gene": what to filter by
    join_genes = "any"                                                         # "any" / "all": union / intersection of confident genesets
    use_ctps  = top_ctps[use_samples]
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


#%% model response per classifier.

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


## get other parameters.
use_cclr      = ["ramilowski", "wang", "all"]                                  # all ligand-receptor lists
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
                pipe_tuned = EnsembleClassifier(
                    models = list(pipes_mdl.values()))
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
    print(f"\noverall performance for L-R list = {use_cclr_} (m = {cclr_lr.shape[0]}): \n{perf_test_split_rep[['mean', 'full']].round(4)}")
    
    
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


## performance for ensemble prediction.
y_pred_cclr_ens    = y_pred_cclr_full.drop(columns = "all").mean(axis = 1)
th_test_cclr_ens   = th_test_cclr_full.drop(index = "all").mean()
y_pred_cclr_th_ens = (y_pred_cclr_ens >= th_test_cclr_ens).astype(int)
perf_test_cclr_ens = pd.concat([
    pd.Series(classifier_performance(y_all, y_pred_cclr_ens)), 
    pd.Series(binary_performance(y_all, y_pred_cclr_th_ens)) ])

y_pred_cclr_full["enseble"]         = y_pred_cclr_ens
th_test_cclr_full["ensemble"]       = th_test_cclr_ens
perf_test_cclr_full.loc["ensemble"] = perf_test_cclr_ens[use_mets]

del y_pred_cclr_ens, y_pred_cclr_th_ens, th_test_cclr_ens, perf_test_cclr_ens


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

svdat = False

if svdat:
    # datestamp = date_time()
    datestamp = "25Mar2023"
    
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
    

#%% compute feature importance.
## RF: mean decrease in impurit (MDI), permutation-based importance
## default: MDI; accumulation of the gini impurity decrease within each tree

svdat = False

cclr = "ramilowski"
pipes_all = pipes_all_cclr[cclr]

na_th  = 0.6                                                                   # NA fractions to allow for a feature
imp_th = 1e-3                                                                  # importance threshold for keeping a feature

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
    lambda res: res[1] * (1 if (res[0] > 1.0) else -1), axis = 1)


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

print(f"\ntop 20 most predictive CCIs for TransNEO cohort = \n{featimp_final.iloc[:20, -2:]}")


## save data.
if svdat:
    # datestamp = date_time()
    datestamp = "25Mar2023"
    out_path  = data_path + "mdl_data/"
    out_file  = f"transneo_lirics_feature_importance_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.xlsx"
    out_file  = out_file.replace(f"_{use_mdl}", 
                                 f"_filteredCCI_th{conf_th}_{use_mdl}" \
                                     if filter_cci else f"_allCCI_{use_mdl}")
    out_dict  = {"MDI": featimp_final}
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    write_xlsx(out_path + out_file, out_dict)


#%% make feature importance plot.

svdat = False

fontdict = {"label": dict(fontfamily = "monospace", fontsize = 30, 
                          fontweight = "regular"), 
            "title": dict(fontfamily = "monospace", fontsize = 34, 
                          fontweight = "semibold"), 
            "super": dict(fontfamily = "monospace", fontsize = 38, 
                          fontweight = "bold")}

num_disp = 15                                                                  # plot top features
fig_data = featimp_final.reset_index()[:num_disp]

plt_type = "line"                                                              # bar / line : barplot / lollipop plot
plt_clr  = ["#75D0A6", "#EED5B8"]                                              # plot colors 
offset   = 0.015                                                               # offset between line and marker for lollipop plot

fig2, ax2 = plt.subplots(figsize = (24, 18), nrows = 1, ncols = 1)
if plt_type == "bar":
    sns.barplot(data = fig_data, x = "MDI", y = "CCIannot", orient = "h", 
                color = plt_clr[0], edgecolor = [0.3]*3, saturation = 0.9, 
                dodge = True, ax = ax2)
elif plt_type == "line":
    ax2.hlines(y = fig_data["CCIannot"], xmin = 0, 
               xmax = fig_data["MDI"] - offset, color = plt_clr[1], 
               linestyle = "-", linewidth = 14);
    
    sns.scatterplot(data = fig_data, x = "MDI", y = "CCIannot", 
                    markers = True, color = plt_clr[0], s = 1000, ax = ax2);

ax2.set_xlabel("Mean decrease in impurity (MDI)", **fontdict["label"]);
ax2.set_ylabel(None);    ax2.set_xlim([0, 1]);
ax2.set_yticks(ticks = range(num_disp), labels = fig_data["CCIannot"], 
              **fontdict["label"]);
ax2.tick_params(axis = "both", which = "major", 
                labelsize = fontdict["label"]["fontsize"]);
# ax2.set_title(f"Top predictive cell-cell interactions (n = {num_disp})", 
#               **fontdict["title"]);
ax2.set_title("Top CCIs for TransNEO", **fontdict["title"]);

fig2.tight_layout()
plt.show()


## save figure.
if svdat:
    # datestamp = date_time()
    datestamp = "25Mar2023"
    
    ## figures for all cell types.
    fig_path = data_path + "plots/final_plots3/"
    fig_file = f"transneo_lirics_{use_samples}_feature_importance_{plt_type}plot_CCI_{use_mdl}_{num_feat_max}features_{datestamp}.pdf"
    fig_file = fig_file.replace("CCI_", 
                     f"filteredCCI_th{conf_th}_" if filter_cci else "allCCI_")
    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    fig2.savefig(fig_path + fig_file, dpi = "figure")


#%% make ROC/PR curve plots.

svdat = False

fontdict = {"label": dict(fontfamily = "monospace", fontsize = 30, 
                          fontweight = "regular"), 
            "title": dict(fontfamily = "monospace", fontsize = 34, 
                          fontweight = "semibold"), 
            "super": dict(fontfamily = "monospace", fontsize = 38, 
                          fontweight = "bold")}


## get bulk prediction data.
bulk_path = data_path + "mdl_data/"
bulk_file = "transneo_predictions_chemo_th0.99_ENS2_25features_5foldCV_20Mar2023.pkl"

with open(bulk_path + bulk_file, "rb") as file:
    data_obj           = pickle.load(file)
    y_test_all         = data_obj["label"]
    y_pred_ctp_full    = data_obj["pred"]
    th_test_ctp_full   = data_obj["th"]
    perf_test_ctp_full = data_obj["perf"]
    del data_obj

y_pred_plt_bulk = y_pred_ctp_full["Bulk"]


## get CCI prediction data.
cclr = "ramilowski"
y_pred_plt_cclr = y_pred_cclr_full[cclr]
y_preds = {"LRI": y_pred_plt_cclr, "Bulk": y_pred_plt_bulk}


## make performance plots.
fig_title1 = "Performance of cell-cell interaction based model"
if use_samples == "all":
    fig_title1 += " for the whole TransNEO cohort "
elif use_samples == "chemo_targeted":
    fig_title1 += " for HER2+ patients\ntreated with chemotherapy and Trastuzumab in TransNEO cohort "
elif use_samples == "chemo":
    fig_title1 += " for HER2- patients\ntreated with chemotherapy alone in TransNEO cohort "
fig_title1 += f"(n = {y_test_all.size})"


# mpl.style.use("ggplot")
fig1, (ax11, ax12) = plt.subplots(figsize = (28, 16), nrows = 1, ncols = 2)
ax11 = make_performance_plot(
    y_all, y_preds, curve = "ROC", ax = ax11, fontdict = fontdict)
ax12 = make_performance_plot(
    y_all, y_preds, curve = "PR", ax = ax12, fontdict = fontdict)
fig1.suptitle(fig_title1, y = 0.98, **fontdict["super"]);

fig1.tight_layout()
plt.show()


## save figures.
if svdat:
    # datestamp = date_time()
    datestamp = "25Mar2023"
    
    ## figures for all cell types.
    fig_path = data_path + "plots/final_plots2/"
    if not os.path.exists(fig_path):                                           # create figure dir
        os.mkdir(fig_path)
    
    fig_file = f"transneo_lirics_{use_samples}_AUC_AP_bulk_CCI_{use_mdl}_{num_feat_max}features_{datestamp}.pdf"
    fig_file = fig_file.replace("CCI_", 
                     f"filteredCCI_th{conf_th}_" if filter_cci else "allCCI_")
    fig1.savefig(fig_path + fig_file, dpi = "figure")
    

#%% make score boxplot.

svdat = False

cclr = "ramilowski"
y_score_plt = pd.concat([y_pred_cclr_full[cclr], 
                         y_test_all.replace({1: "R", 0: "NR"})], axis = 1)

fig3, ax3 = plt.subplots(figsize = (20, 12), nrows = 1, ncols = 1)
sns.boxplot(data = y_score_plt, x = "pCR.RD", y = cclr, ax = ax3, 
            orient = "v", saturation = 0.8, palette = "tab10", whis = 1.5)
ax3.set_xlabel(None);       ax3.set_ylabel(None);
ax3.tick_params(axis = "both", which = "major", 
                labelsize = fontdict["label"]["fontsize"]);
ax3.set_title(fig_title1.replace("Performance", "Scores"), 
              **fontdict["title"]);

fig3.tight_layout()
plt.show()


## save figures.
if svdat:
    # datestamp = date_time()
    datestamp = "25Mar2023"
    
    ## figures for all cell types.
    fig_path = data_path + "plots/final_plots2/"
    if not os.path.exists(fig_path):                                           # create figure dir
        os.mkdir(fig_path)
    
    fig_file = f"transneo_lirics_{use_samples}_scores_CCI_feature_importance_{use_mdl}_{num_feat_max}features_{datestamp}.pdf"
    fig_file = fig_file.replace("CCI_", 
                     f"filteredCCI_th{conf_th}_" if filter_cci else "allCCI_")
    fig3.savefig(fig_path + fig_file, dpi = "figure")

