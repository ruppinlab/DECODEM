#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:11:57 2023

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
from miscellaneous import date_time, tic, write_xlsx
from _functions import (MakeClassifier, EnsembleClassifier, train_pipeline, 
                        predict_proba_scaled, get_best_threshold, 
                        classifier_performance, binary_performance, 
                        make_performance_plot)
from sklearn.model_selection import StratifiedKFold, KFold
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

use_samples = "chemo"

data_path = ["../../data/TransNEO/transneo_analysis/", 
             "../../data/BrighTNess/validation/"]

data_file = [f"transneo_lirics_data_{use_samples}_v3.pkl", 
             f"brightness_lirics_data_{use_samples}.pkl"]

## load data.
with open(data_path[0] + data_file[0], "rb") as file:
    data_obj_train   = pickle.load(file)
    cci_all_train    = data_obj_train["cci"]
    cclr_all_train   = data_obj_train["cclr"]
    resp_pCR_train   = data_obj_train["resp"]
    clin_info_train  = data_obj_train["clin"]
    cell_frac_train  = data_obj_train["frac"]
    conf_score_train = data_obj_train["conf"]
    del data_obj_train

with open(data_path[1] + data_file[1], "rb") as file:
    data_obj_test   = pickle.load(file)
    cci_all_test    = data_obj_test["cci"]
    cclr_all_test   = data_obj_test["cclr"]
    resp_pCR_test   = data_obj_test["resp"]
    clin_info_test  = data_obj_test["clin"]
    cell_frac_test  = data_obj_test["frac"]
    conf_score_test = data_obj_test["conf"]
    del data_obj_test

## get cell types & sample lists.
if conf_score_train.columns.tolist() == conf_score_test.columns.tolist():
    cell_types = conf_score_train.columns.tolist()
else:
    raise ValueError("the cell types between training & test are not the same!")


## combine two CCI lists into one.
cclr_all_train["all"] = pd.concat(cclr_all_train.values()).drop_duplicates()
cclr_all_test["all"]  = pd.concat(cclr_all_test.values()).drop_duplicates()
cci_all_train["all"]  = pd.concat(cci_all_train.values()).reset_index(
    ).rename(columns = {"index": "CCLR"}).drop_duplicates().set_index("CCLR")
cci_all_test["all"]   = pd.concat(cci_all_test.values()).reset_index(
    ).rename(columns = {"index": "CCLR"}).drop_duplicates().set_index("CCLR")

if cclr_all_train["all"].index.tolist() != cci_all_train["all"].index.tolist():
    raise ValueError("the CCLR lists are not the same between CCI data and annotations in the combined training data!")

if cclr_all_test["all"].index.tolist() != cci_all_test["all"].index.tolist():
    raise ValueError("the CCLR lists are not the same between CCI data and annotations in the combined test data!")


#%% prepare data.

## filter CCIs for cell types & genes to use.
top_ctps = {
    "all": ["B-cells", "Myeloid", "Plasmablasts"], 
    "chemo_targeted": ["PVL", "B-cells", "Myeloid"], 
    "chemo": ["Cancer_Epithelial", "Endothelial", "Myeloid", "Plasmablasts"] }
    # "chemo": ["B-cells", "Myeloid", "T-cells"] }                               # for sc validation - cell types present in Zhang et al.

filter_cci = True
if filter_cci:
    conf_th = 0.99                                                             # confident gene cut-off
    filter_by  = "both"                                                        # "both" / "cell" / "gene": what to filter by
    join_genes = "any"                                                         # "any" / "all": union / intersection of confident genesets
        
    # use_ctps = ["Endothelial", "Myeloid", "Plasmablasts"]
    # use_ctps = ["Cancer_Epithelial", "B-cells", "Myeloid", "T-cells"]
    # use_ctps = ["Cancer_Epithelial", "B-cells", "Myeloid", "T-cells", 
    #             "Plasmablasts"]
    # use_ctps = ["Cancer_Epithelial", "Myeloid", "PVL"]
    # use_ctps = ["Cancer_Epithelial", "Endothelial", "Myeloid", "Plasmablasts", 
    #             "B-cells", "Normal_Epithelial"]
    use_ctps = top_ctps[use_samples]
    use_genes = np.intersect1d(
        get_conf_genes(conf_score_train, th = conf_th, ctps = use_ctps, 
                       join = join_genes.lower()), 
        get_conf_genes(conf_score_test, th = conf_th, ctps = use_ctps, 
                       join = join_genes.lower()) ).tolist() 


## finalize data.
y_train, y_test = resp_pCR_train.copy(), resp_pCR_test.copy()
cclr_list, X_all_train, X_all_test = { }, { }, { }
for cclr_ in cclr_all_train:
    ## get LR list.
    cclr_lr_ = cclr_all_train[cclr_].loc[np.intersect1d(
        cclr_all_train[cclr_].index, cclr_all_test[cclr_].index)]
    if filter_cci:                                                             # filter LR list by cell types & genes
        cclr_lr_ = filter_cclr(cclr_lr_, filter_by = filter_by, 
                               ctps = use_ctps, genes = use_genes)
    
    ## get CCI data.
    X_train_cclr_ = cci_all_train[cclr_].loc[cclr_lr_.index, y_train.index].T
    X_test_cclr_  = cci_all_test[cclr_].loc[cclr_lr_.index, y_test.index].T
    
    cclr_list[cclr_], X_all_train[cclr_], X_all_test[cclr_] = \
        cclr_lr_, X_train_cclr_, X_test_cclr_

del cclr_lr_, X_train_cclr_, X_test_cclr_


#%% model response per classifer.

## dataset options.
## arm B: paclitaxel + carboplatin (TC), arm C: paclitaxel (T).
use_arm = "B"                                                                  # use None for both arms
if use_arm is not None:
    arm_samples = clin_info_test[
        clin_info_test["planned_arm_code"] == use_arm.upper()].index.tolist()
    X_all_test = {
        cclr_: X_.loc[arm_samples] for cclr_, X_ in X_all_test.items()}
    y_test     = y_test.loc[arm_samples]


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
num_splits    = 3
use_mets      = ["AUC", "AP", "ACC", "DOR", "SEN", "PPV", "SPC"]               # list of performance metrics to use


_tic = tic()

## start modeling per LR list.
y_pred_val = { };    th_test_val = { };    perf_test_val = { }
pipes_test_val = { }
for use_cclr_ in use_cclr:
    ## get dataset.
    cclr_lr = cclr_list[use_cclr_]
    X_train, X_test = X_all_train[use_cclr_], X_all_test[use_cclr_]
    
    print(f"""\n
    samples = {use_samples}, L-R list = {use_cclr_} (m = {cclr_lr.shape[0]})
    model = {use_mdl}, max #features = {num_feat_max}, #repetitions = {num_split_rep}
    sample size: training = {X_train.shape[0]}, test = {X_test.shape[0]}""")
        
    
    ## start modeling per repetition.
    y_pred_rep = { };    th_test_rep = { };    perf_test_rep = { } 
    pipes_test_rep = { }
    for use_seed in range(num_split_rep):
        print(f"\nsplit seed = {use_seed}")
        
        ## make cv splits for tuning.
        # cv_tune = StratifiedKFold(n_splits = num_splits, shuffle = True, 
        #                           random_state = use_seed)
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
y_pred_val    = pd.DataFrame(y_pred_val).set_index(X_test.index)               # mean prediction matrix
th_test_val   = pd.DataFrame(th_test_val).T
perf_test_val = pd.DataFrame(perf_test_val).T


## performance for ensemble model.
y_pred_val["ensemble"] = y_pred_val[use_cclr[:-1]].mean(axis = 1)
th_mean_val            = th_test_val.drop(index = "all")["mean"].mean()
y_pred_val_ens_th      = y_pred_val["ensemble"].ge(th_mean_val).astype(int)
perf_test_val_ens      = pd.concat([
    pd.Series(classifier_performance(y_test, y_pred_val["ensemble"])), 
    pd.Series(binary_performance(y_test, y_pred_val_ens_th)) ])
perf_test_val.loc["ensemble"] = perf_test_val_ens[use_mets]

del th_mean_val, y_pred_val_ens_th, perf_test_val_ens

# print(os.system("clear"))                                                      # clears console
print(f"""\n{'-' * 64}
validation performance for treatment = {use_samples}:
cohort = BrighTNess, Arm B (n = {y_test.size})
cell_types = {use_ctps if filter_cci else 'all'}
available #CCIs = { {cclr_: len(cclr_lr_) for cclr_, cclr_lr_ in cclr_list.items()} }
model = {use_mdl}, max #features = {num_feat_max}
{perf_test_val.round(4)}""")

_tic.toc()


#%% save full prediction & performance tables.

svdat = False

if svdat:
    # datestamp = date_time()
    datestamp = "25Mar2023"
    # datestamp = "14Aug2023"
    
    ## save full predictions & performance.
    out_path = data_path[0] + "mdl_data/"
    out_file = f"brightness_lirics_predictions_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.pkl"
    out_file = out_file.replace(f"_{use_mdl}", 
                                f"_filteredCCI_th{conf_th}_{use_mdl}" \
                                    if filter_cci else f"_allCCI_{use_mdl}")
    out_dict = {"label": y_test,   "pred": y_pred_val, 
                "th": th_test_val, "perf": perf_test_val, 
                "pipe": pipes_test_val}
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    with open(out_path + out_file, "wb") as file:
        pickle.dump(out_dict, file)

    ## save complete performance into xlsx file.
    out_path = _wpath_ + "results/"
    out_file = f"brightness_lirics_results_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.xlsx"
    out_file = out_file.replace(f"_{use_mdl}", 
                                f"_filteredCCI_th{conf_th}_{use_mdl}" \
                                    if filter_cci else f"_allCCI_{use_mdl}")
    out_dict = perf_test_val.copy()
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    write_xlsx(out_path + out_file, out_dict)
    
    
    ## save list of CCIs used.
    out_path = data_path[0] + "mdl_data/"
    out_file = f"brightness_lirics_feature_list_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.xlsx"
    out_dict = cclr_list.copy()
    
    write_xlsx(out_path + out_file, out_dict)    
    

#%% compute feature importance.

svdat = False

cclr      = "ramilowski"
pipes_all = pipes_test_val[cclr]

na_th  = 0.6                                                                   # NA fractions to allow for a feature
imp_th = 1e-3                                                                  # importance threshold for keeping a feature

featimp_all = pd.DataFrame({sd_: get_feature_importance(pipe_, scale = True) \
                            for sd_, pipe_ in pipes_all.items()})
featimp_all.dropna(thresh = na_th * len(pipes_all), inplace = True)
featimp_all["mean"] = featimp_all.mean(axis = 1)
featimp_all.sort_values(by = "mean", ascending = False, inplace = True)
featimp_all[["OR", "pval"]] = get_feature_association(
    data = (X_train, y_train), featlist = featimp_all.index)
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
featimp_final[["LigandCell", "ReceptorCell"]] = \
    featimp_final[["LigandCell", "ReceptorCell"]].replace(
        regex = {"-Epithelial": "_Epithelial"})
featimp_final["CCIannot"] = featimp_final.apply(
    lambda df: f"{df.LigandCell}-{df.ReceptorCell}::{df.LigandGene}-{df.ReceptorGene}", 
    axis = 1)
featimp_final = featimp_final.set_index("CCIannot")[[
    "LigandCell", "ReceptorCell", "LigandGene", "ReceptorGene", 
    "MDI", "Direction"]]

print(f"\ntop 20 most predictive CCIs for BrighTNess cohort = \n{featimp_final.iloc[:20, -2:]}")


## save data.
if svdat:
    # datestamp = date_time()
    datestamp = "25Mar2023"
    out_path  = data_path[0] + "mdl_data/"
    out_file  = f"brightness_lirics_feature_importance_{use_samples}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV_{datestamp}.xlsx"
    out_file  = out_file.replace(f"_{use_mdl}", 
                                 f"_filteredCCI_th{conf_th}_{use_mdl}" \
                                     if filter_cci else f"_allCCI_{use_mdl}")
    out_dict  = {"MDI": featimp_final}
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    write_xlsx(out_path + out_file, out_dict)


#%% make feature importance plot.

svdat = False

fontname = "sans"
fontdict = {"label": dict(fontfamily = fontname, fontsize = 48, 
                          fontweight = "regular"), 
            "title": dict(fontfamily = fontname, fontsize = 52, 
                          fontweight = "semibold"), 
            "super": dict(fontfamily = fontname, fontsize = 56, 
                          fontweight = "bold")}

num_disp = 10                                                                  # plot top features
fig_data = featimp_final.reset_index()[:num_disp]
fig_data.replace(regex = {"CCIannot": {"-": " $-$ ", "_": " ", "::": " :: "}}, 
                 inplace = True)


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

ax2.set_xlabel("Mean decrease in impurity", **fontdict["label"]);
ax2.set_ylabel(None);    ax2.set_xlim([0, 1]);
ax2.set_yticks(ticks = range(num_disp), labels = fig_data["CCIannot"], 
              **fontdict["label"]);
ax2.tick_params(axis = "both", which = "major", 
                labelsize = fontdict["label"]["fontsize"]);
# ax2.set_title(f"Top predictive cell-cell interactions (n = {num_disp})", 
#               **fontdict["title"]);
ax2.set_title("BrighTNess", **fontdict["title"]);

fig2.tight_layout()
plt.show()


## save figure.
if svdat:
    # datestamp = date_time()
    datestamp = "27Mar2023"
    
    ## figures for all cell types.
    fig_path = data_path[0] + "plots/final_plots3/"
    fig_file = f"brightness_lirics_{use_samples}_feature_importance_{plt_type}plot_CCI_{use_mdl}_{num_feat_max}features_{datestamp}.pdf"
    fig_file = fig_file.replace("CCI_", 
                     f"filteredCCI_th{conf_th}_" if filter_cci else "allCCI_")
    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    fig2.savefig(fig_path + fig_file, dpi = "figure")


#%% make ROC/PR curve plots.

svdat = False

fontname = "sans"
fontdict = {"label": dict(fontfamily = fontname, fontsize = 30, 
                          fontweight = "regular"), 
            "title": dict(fontfamily = fontname, fontsize = 34, 
                          fontweight = "semibold"), 
            "super": dict(fontfamily = fontname, fontsize = 38, 
                          fontweight = "bold")}


## get bulk prediction data.
bulk_path = data_path[0] + "mdl_data/"
bulk_file = "brightness_predictions_chemo_th0.99_ENS2_25features_3foldCVtune_23Mar2023.pkl"

with open(bulk_path + bulk_file, "rb") as file:
    data_obj          = pickle.load(file)
    y_test_val        = data_obj["label"]
    y_pred_ctp_val    = data_obj["pred"]
    th_test_ctp_val   = data_obj["th"]
    perf_test_ctp_val = data_obj["perf"]
    del data_obj

y_pred_plt_bulk = y_pred_ctp_val["Bulk"]


## get CCI prediction data.
cclr = "ramilowski"
y_pred_plt_val = y_pred_val[cclr]
y_preds = {"LRI": y_pred_plt_val, "Bulk": y_pred_plt_bulk}


## make performance plots.
fig_title1 = "Performance of cell-cell interaction based model"
if use_samples == "all":
    fig_title1 += " for the whole BrighTNess cohort, Arm B "
elif use_samples == "chemo_targeted":
    fig_title1 += " for HER2+ patients\ntreated with chemotherapy and Trastuzumab in BrighTNess cohort, Arm B "
elif use_samples == "chemo":
    fig_title1 += " for HER2- patients\ntreated with chemotherapy alone in BrighTNess cohort, Arm B "
fig_title1 += f"(n = {y_test.size})"

# mpl.style.use("ggplot")
fig1, (ax11, ax12) = plt.subplots(figsize = (28, 16), nrows = 1, ncols = 2)
ax11 = make_performance_plot(
    y_test, y_preds, curve = "ROC", ax = ax11, fontdict = fontdict)
ax12 = make_performance_plot(
    y_test, y_preds, curve = "PR", ax = ax12, fontdict = fontdict)
fig1.suptitle(fig_title1, y = 0.98, **fontdict["super"]);

fig1.tight_layout()
plt.show()


## save figures.
if svdat:
    # datestamp = date_time()
    datestamp = "27Mar2023"
    
    ## figures for all cell types.
    fig_path = data_path[0] + "plots/final_plots2/"
    if not os.path.exists(fig_path):                                           # create figure dir
        os.mkdir(fig_path)
    
    fig_file = f"brightness_lirics_{use_samples}_AUC_AP_bulk_CCI_{use_mdl}_{num_feat_max}features_{datestamp}.pdf"
    fig_file = fig_file.replace("CCI_", 
                     f"filteredCCI_th{conf_th}_" if filter_cci else "allCCI_")
    fig1.savefig(fig_path + fig_file, dpi = "figure")
    
