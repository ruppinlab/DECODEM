#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:04:38 2024

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
from math import nan, ceil
# from functools import reduce
from miscellaneous import date_time, tic, write_xlsx
from _functions import (EnsembleClassifier, train_pipeline, 
                        predict_proba_scaled, classifier_performance, 
                        binary_performance, get_best_threshold)
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import concordance_index


#%% functions.

## get confident genes for a cell type.
def get_conf_genes(conf, th = 0.99):
    genes = conf[conf.ge(th)].index.tolist()
    return genes


#%% read data.

use_samples = "chemo"

data_path = ["../../data/TransNEO/transneo_analysis/", 
             "../../data/TCGA/validation/"]

data_file = [f"transneo_data_{use_samples}_v2.pkl", 
             "tcga_brca_data_surv_her2neg.pkl"]


## load data.
with open(data_path[0] + data_file[0], mode = "rb") as file:
    data_train_obj   = pickle.load(file)
    exp_all_train    = data_train_obj["exp"]
    resp_pCR_train   = data_train_obj["resp"]
    cell_frac_train  = data_train_obj["frac"]
    conf_score_train = data_train_obj["conf"]
    clin_info_train  = data_train_obj["clin"]
    del data_train_obj

with open(data_path[1] + data_file[1], mode = "rb") as file:
    data_test_obj   = pickle.load(file)
    exp_all_test    = data_test_obj["exp"]
    resp_surv_test  = data_test_obj["resp"]
    cell_frac_test  = data_test_obj["frac"].rename(                            # match cell type names betn. exp & cell fraction
        columns = lambda x: x.replace(" ", "_"))
    conf_score_test = data_test_obj["conf"].rename(
        columns = lambda x: x.replace(" ", "_"))
    clin_info_test  = data_test_obj["clin"]
    resp_surv_test  = resp_surv_test.set_index(
        keys = "Sample_ID").replace(
        to_replace = {-2147483648: nan})                                       # saved from R- turned NA into large negative integer
    del data_test_obj

if conf_score_train.columns.tolist() == conf_score_test.columns.tolist():
    cell_types = conf_score_train.columns.tolist()
else:
    raise ValueError("the cell types are not the same between training and test datasets!")


## keep only early-stage patients.
## source: https://www.cancer.gov/publications/dictionaries/cancer-terms/def/early-stage-breast-cancer
## definition: breast cancer that has not spread beyond the breast or the 
## axillary lymph nodes. This includes ductal carcinoma in situ and stage I, 
## stage IIA, stage IIB, and stage IIIA breast cancers. 
clin_info_test = clin_info_test.pipe(
    lambda df: df[df.Stage.replace(    
        regex = {"Stage ": ""}).map(
        lambda x: ((x == "I") or (x == "IA") or (x == "IB") or 
                   (x == "II") or (x == "IIA") or (x == "IIB") or 
                   (x == "IIIA")))])
clin_info_test["Clinical_subtype"] = clin_info_test.ER_status.map(
    lambda x: "ER+,HER2-" if (x == "Positive") else "TNBC")

exp_all_test   = {ctp_: exp_[clin_info_test.index] 
                  for ctp_, exp_ in exp_all_test.items()}
resp_surv_test = resp_surv_test.loc[clin_info_test.index]
cell_frac_test = cell_frac_test.loc[clin_info_test.index]


#%% prepare data.

conf_th = 0.99                                                                 # confident gene cut-off
genes, X_all_train, X_all_test = { }, { }, { }
for ctp_ in tqdm(cell_types + ["Bulk"]):
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
y_train     = resp_pCR_train.loc[X_all_train["Bulk"].index].copy()
y_test_surv = resp_surv_test.loc[X_all_test["Bulk"].index].copy()


#%% modeling parameters.

use_ctp = np.append(cell_types, "Bulk").tolist()
# use_ctp = "Cancer_Epithelial"


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
y_pred_val = { };    th_test_val = { };
for use_ctp_ in use_ctp:
    ## get training & test sets.
    ctp_list = tuple(cell_types) if (use_ctp_[0] == "all") else use_ctp_
    X_train  = pd.concat([X_all_train[ctp_] for ctp_ in ctp_list], axis = 1)
    X_test   = pd.concat([X_all_test[ctp_] for ctp_ in ctp_list], axis = 1)
    ctp_mdl = "+".join(use_ctp_)                                               # cell-type model name
    
                
    print(f"""\n
    samples = {use_samples}, cell type = {"+".join(use_ctp_)}
    available #genes = {X_train.shape[1]}, max #features = {num_feat_max}
    model = {use_mdl}, #repetitions = {num_split_rep}
    sample size: training = {X_train.shape[0]}, test = {X_test.shape[0]}""")
    
    
    ## start modeling per repition.
    y_pred_rep = { };    th_test_rep =  { };    perf_test_rep = { }
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
                
        
        ## save results for this repetition.
        y_pred_rep[rep_mdl]      = y_pred[:, 1]
        th_test_rep[rep_mdl]     = th_fit
                
        
    ## overall performance across all repetitions.
    y_pred_rep            = pd.DataFrame(y_pred_rep)
    y_pred_rep["mean"]    = y_pred_rep.mean(axis = 1)
    th_test_rep           = pd.Series(th_test_rep)
    th_test_rep["mean"]   = th_test_rep.mean()
    
    
    ## combine prediction across all repetitions & get performance.
    y_pred_full    = y_pred_rep["mean"]
    
    # print(os.system("clear"))                                                  # clears console
    
    
    ## save results for this cell type.
    y_pred_val[ctp_mdl]      = y_pred_full
    th_test_val[ctp_mdl]     = th_test_rep
    

## fianl performance for all cell types.
y_pred_val  = pd.DataFrame(y_pred_val).set_index(X_test.index)                 # mean prediction matrix
th_test_val = pd.DataFrame(th_test_val).T

print(os.system("clear"))                                                      # clears console
print(f"""\n{'-' * 64}
prediction complete for survival analysis! 
cohort = TCGA-BRCA (HER2-; n = {len(y_test_surv):,})""")

_tic.toc()


#%% prepare data for survival analysis.

print("\npreparing data for survival analysis...")

y_pred_surv_val = {ctp_: pd.DataFrame({
        "score"     : y_pred_, 
        "groups_th" : y_pred_.gt(th_test_val.loc[ctp_, "mean"]).astype(int), 
        "groups_med": y_pred_.gt(y_pred_.median()).astype(int), 
        "groups_avg": y_pred_.gt(y_pred_.mean()).astype(int), 
        "groups_05" : y_pred_.gt(0.5).astype(int), 
        "groups_q4" : pd.qcut(y_pred_, q = 4, labels = False), 
        "groups_q3" : pd.qcut(y_pred_, q = 3, labels = False) })
    for ctp_, y_pred_ in tqdm(y_pred_val.items(), total = y_pred_val.shape[1])}

print("\ndone!")


#%% save full prediction.

svdat = False

if svdat:
    datestamp = date_time()
        
    ## save full predictions & performance.
    out_path = data_path[0] + "mdl_data/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    
    out_file = f"tcga_predictions_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.pkl"
    out_dict = {"label": y_test_surv, "pred": y_pred_surv_val, 
                "th": th_test_val, "clin": clin_info_test}
    
    with open(out_path + out_file, "wb") as file:
        pickle.dump(out_dict, file)
    

#%% do cox regression. 
## CE, ENDO, PB, NE, MYL, B, CAF

ctp_surv     = "Cancer_Epithelial"
var_surv     = "OS"

## get data for cox regression.
cph_data_ctp = pd.DataFrame({
    "Score"  : y_pred_surv_val[ctp_surv].score, 
    "Age"    : pd.cut(clin_info_test.Age, bins = 3, labels = False),           # bin Age into 3 categories
    "Stage"  : clin_info_test.Stage.replace(                                   # bin Stage into 3 categories
        regex = {"Stage ": "", "A": "", "B": ""}).replace(
        to_replace = {"I": 0, "II": 1, "III": 2}).infer_objects(
        copy = False), 
    "Subtype": clin_info_test.Clinical_subtype })

cph_data_ctp = pd.concat([
    cph_data_ctp, y_test_surv[[var_surv, f"{var_surv}_time"]]], 
    axis = 1).groupby(
    by = "Subtype").apply(
    lambda df: df, include_groups = False)


## model for each subtype.
cph_erpos = CoxPHFitter(
    baseline_estimation_method = "breslow", alpha = 0.05, 
    penalizer = 0.001, l1_ratio = 0.1).fit(
    cph_data_ctp.loc["ER+,HER2-"], event_col = var_surv, 
    duration_col = f"{var_surv}_time")

cph_tnbc  = CoxPHFitter(
    baseline_estimation_method = "breslow", alpha = 0.05, 
    penalizer = 0.001, l1_ratio = 0.1).fit(
    cph_data_ctp.loc["TNBC"], event_col = var_surv, 
    duration_col = f"{var_surv}_time")


## display results.
cph_cols  = {"exp(coef)"          : "HR", 
             "exp(coef) lower 95%": "HR_low_95", 
             "exp(coef) upper 95%": "HR_high_95", 
             "p"                  : "pval"}
cph_res_erpos = cph_erpos.summary[list(cph_cols)].rename(columns = cph_cols)
cph_res_tnbc  = cph_tnbc.summary[list(cph_cols)].rename(columns = cph_cols)
cph_res_cidx  = {"ER+,HER2-": cph_erpos.concordance_index_, 
                 "TNBC"     : cph_tnbc.concordance_index_}

print(f"""
performed cox regression! 
confounding variables = {cph_data_ctp.loc['TNBC'].columns[1:-2].tolist()}
results:

ER+,HER2-: C-index = {cph_res_cidx['ER+,HER2-']:0.4f}
{cph_res_erpos.round(4)}\n

TNBC: C-index = {cph_res_cidx['TNBC']:0.4f}
{cph_res_tnbc.round(4)}
""")


#%% make forest plots.

def make_forest_plot(ax, data, x, xlow, xhigh, p = None, capsize = 0.05, 
                     title = None, fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {"label": {"size": 14, "weight": "regular"}, 
                    "title": {"size": 18, "weight": "bold"}}
    
    mrkrprop = {"s": 150, "marker": "s", "lw": 2, "c": "#B075D0", 
                "alpha": 0.8, "ec": "#000000"}
    lineprop = {"color": "#000000", "lw": 2, "ls": "-"}
    baseprop = {"color": "#000000", "lw": 1.5, "ls": "--"}
    
    data["var_no"]   = range(len(data))
    data["cap_low"]  = data.var_no - capsize / 2
    data["cap_high"] = data.var_no + capsize / 2
    data["xlow1"]    = data[x] - (0.15 - 0.02)
    data["xhigh1"]   = data[x] + (0.15 - 0.02)
    
    ## make plot.
    ax.scatter(data = data, x = x, y = "var_no", **mrkrprop)                   # main box
    [ax.hlines(data = data, y = "var_no", xmin = xs, xmax = xe, **lineprop)    # add lines surrounding box
     for xs, xe in [(xlow, "xlow1"), ("xhigh1", xhigh)]]
    [ax.vlines(data = data, x = xpos, ymin = "cap_low", ymax = "cap_high",     # add caps at end
               **lineprop) for xpos in [xlow, xhigh]]
    ax.axvline(x = 1, ymin = 0, ymax = 1.95, **baseprop)                       # make baseline
    sns.despine(ax = ax, offset = 0, trim = True, left = True);
    
    if p is not None:                                                          # add p-value
        p_ax = ax.twinx()
        p_ax.set_yticks(ticks  = data.var_no, 
                        labels = data[p].map(lambda p: f"$P = {p:0.3n}$"))
        p_ax.tick_params(axis = "y", labelright = True, labelleft = False, 
                         labelsize = fontdict["label"]["size"], length = 0)
        p_ax.set_ylim([-0.5, len(data) - 0.5]);
        sns.despine(ax = p_ax, offset = 0, trim = True, left = True);
    
    ## format ticks & labels.
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.tick_params(axis = "y", length = 0);
    ax.set_ylim([-0.5, len(data) - 0.5]);
    ax.set_yticks(ticks = data.var_no, labels = data.index);
    ax.set_xlabel("Hazard Ratio (95% CI)", y = -0.01, **fontdict["label"]);
    ax.set_title(title, wrap = True, y = 1.01, **fontdict["title"]);
    
    return ax


## make subtype-specific plots.
sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 8, "xtick.major.width": 2, 
    "ytick.major.size": 8, "ytick.major.width": 2, 
    "xtick.bottom": True, "ytick.left": True, 
    "axes.spines.top": False, "axes.spines.right": False, 
    "axes.linewidth": 2, "axes.edgecolor": "#000000", 
    "grid.linewidth": 1, "grid.color": "#000000", "grid.alpha": 0.8, 
    "legend.frameon": False, "legend.edgecolor": "#000000", 
    "legend.framealpha": 0.9, "legend.markerscale": 1.2, 
    "font.family": "sans"})


fig_ttls1 = [f"{sb} (n = {len(cph_data_ctp.loc[sb])})" 
             for sb in ["ER+,HER2-", "TNBC"]]

fig1, ax1 = plt.subplots(figsize = (14, 8), nrows = 1, ncols = 2, 
                         sharey = True)

# cph_erpos.plot(hazard_ratios = True, c = "#000000", lw = 4, ax = ax1[0])
# cph_tnbc.plot(hazard_ratios = True, c = "#000000", lw = 4, ax = ax1[1])

ax1[0] = make_forest_plot(data = cph_res_erpos, x = "HR", xlow = "HR_low_95", 
                          xhigh = "HR_high_95", p = "pval", 
                          title = fig_ttls1[0], ax = ax1[0])
ax1[1] = make_forest_plot(data = cph_res_tnbc, x = "HR", xlow = "HR_low_95", 
                          xhigh = "HR_high_95", p = "pval", 
                          title = fig_ttls1[1], ax = ax1[1])

fig1.suptitle(f"Cox proportional hazard fit for {var_surv} in TCGA-BRCA\nCell type = {ctp_surv.replace('_', ' ')}", 
              wrap = True, y = 0.99, fontsize = 20, fontweight = "bold");

fig1.tight_layout(w_pad = 4)

plt.show()


#%% do kaplan-meier fits.
## CE, ENDO, PB, NE, MYL, B, CAF

ctp_surv      = "Cancer_Epithelial"
var_surv      = "OS"
var_group     = "groups_05"

## get data for kaplan-meier plot.
km_data_ctp   = pd.concat([
    y_pred_surv_val[ctp_surv], 
    y_test_surv[[var_surv, f"{var_surv}_time"]], 
    clin_info_test[["Clinical_subtype"]]], axis = 1)

km_ctp_erpos1 = km_data_ctp.pipe(
    lambda df: df[df.Clinical_subtype.eq("ER+,HER2-") & df[var_group].eq(1)])
km_ctp_erpos2 = km_data_ctp.pipe(
    lambda df: df[df.Clinical_subtype.eq("ER+,HER2-") & df[var_group].eq(0)])

km_ctp_tnbc1  = km_data_ctp.pipe(
    lambda df: df[df.Clinical_subtype.eq("TNBC") & df[var_group].eq(1)])
km_ctp_tnbc2  = km_data_ctp.pipe(
    lambda df: df[df.Clinical_subtype.eq("TNBC") & df[var_group].eq(0)])


## model for each subtype.
## ER+, HER2-.
km_erpos1     = KaplanMeierFitter(alpha = 0.05).fit(
    event_observed = km_ctp_erpos1[var_surv], 
    durations = km_ctp_erpos1[f"{var_surv}_time"], 
    label = "High-score")
km_erpos2     = KaplanMeierFitter(alpha = 0.05).fit(
    event_observed = km_ctp_erpos2[var_surv], 
    durations = km_ctp_erpos2[f"{var_surv}_time"], 
    label = "Low-score")

## TNBC.
km_tnbc1      = KaplanMeierFitter(alpha = 0.05).fit(
    event_observed = km_ctp_tnbc1[var_surv], 
    durations = km_ctp_tnbc1[f"{var_surv}_time"], 
    label = "High-score")
km_tnbc2      = KaplanMeierFitter(alpha = 0.05).fit(
    event_observed = km_ctp_tnbc2[var_surv], 
    durations = km_ctp_tnbc2[f"{var_surv}_time"], 
    label = "Low-score")


km_res_lr     = {
    "ER+,HER2-": logrank_test(
        event_observed_A = km_ctp_erpos1[var_surv], 
        event_observed_B = km_ctp_erpos2[var_surv], 
        durations_A = km_ctp_erpos1[f"{var_surv}_time"], 
        durations_B = km_ctp_erpos2[f"{var_surv}_time"]), 
    "TNBC": logrank_test(
        event_observed_A = km_ctp_tnbc1[var_surv], 
        event_observed_B = km_ctp_tnbc2[var_surv], 
        durations_A = km_ctp_tnbc1[f"{var_surv}_time"], 
        durations_B = km_ctp_tnbc2[f"{var_surv}_time"])}


#%% make kaplan-meier plots.

def make_km_plot(ax, data_grp1, data_grp2, stat, colors = None, 
                 ci_alpha = 0.15, title = None, legend = True, 
                 legend_title = None, fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {"label": {"size": 14, "weight": "regular"}, 
                    "title": {"size": 18, "weight": "bold"}}
        
    if colors is None:
        colors   = ["#E08DAC", "#7595D0"]
    
    lineprop = {"ls": "-", "lw": 2}
    
    lgndttl  = "Risk group" if (legend_title is None) else legend_title
    lbls     = [f"{data_grp1.label} (n = {len(data_grp1.durations)})", 
               f"{data_grp2.label} (n = {len(data_grp2.durations)})"]
    
    
    ## make plots.
    ax = data_grp1.plot(show_censors = True, ci_show = True, color = colors[0], 
                        ci_alpha = ci_alpha, ax = ax, **lineprop)
    ax = data_grp2.plot(show_censors = True, ci_show = True, color = colors[1], 
                        ci_alpha = ci_alpha, ax = ax, **lineprop)
    ax.text(x = 250, y = 0.20, s = f"Log-rank $P$ = {stat.p_value:0.3g}", 
            **fontdict["label"]);
    add_at_risk_counts(data_grp1, data_grp2, labels = lbls, 
                       rows_to_show = None, ax = ax, **fontdict["label"]);     # at-risk counts below the plots
    sns.despine(ax = ax, offset = 0, trim = False);
    
    ## format ticks & labels.
    ax.set_ylim([-0.1, 1.1]);
    ax.set_yticks(np.arange(0, 1.2, 0.2));
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    
    if legend:
        ax.legend(loc = (1.06, 0.25), title = lgndttl, prop = fontdict["label"], 
                  title_fontproperties = fontdict["title"]);
    else:
        ax.legend([ ], [ ]);
    
    ax.set_xlabel("Time in days", y = -0.02, **fontdict["label"]);
    ax.set_ylabel("Survival proabibility", x = 0.01, **fontdict["label"]);
    ax.set_title(title, wrap = True, y = 1.01, **fontdict["title"]);
    
    return ax


## make subtype-specific plots.
sns.set_style("ticks")
plt.rcParams.update({
    "xtick.major.size": 8, "xtick.major.width": 2, 
    "ytick.major.size": 8, "ytick.major.width": 2, 
    "xtick.bottom": True, "ytick.left": True, 
    "axes.spines.top": False, "axes.spines.right": False, 
    "axes.linewidth": 2, "axes.edgecolor": "#000000", 
    "grid.linewidth": 1, "grid.color": "#000000", "grid.alpha": 0.8, 
    "legend.frameon": False, "legend.edgecolor": "#000000", 
    "legend.framealpha": 0.9, "legend.markerscale": 1.2, 
    "font.family": "sans"})

fig_ttls2 = [f"{sb} (n = {len(cph_data_ctp.loc[sb])})" 
             for sb in ["ER+,HER2-", "TNBC"]]

fig2, ax2 = plt.subplots(figsize = (16, 8), nrows = 1, ncols = 2, 
                         sharey = True)

ax2[0] = make_km_plot(data_grp1 = km_erpos1, data_grp2 = km_erpos2, 
                      stat = km_res_lr["ER+,HER2-"], title = fig_ttls2[0], 
                      legend = False, ax = ax2[0])

ax2[1] = make_km_plot(data_grp1 = km_tnbc1, data_grp2 = km_tnbc2, 
                      stat = km_res_lr["TNBC"], title = fig_ttls2[1], 
                      legend = True, ax = ax2[1])

fig2.suptitle(f"Kaplan-Meier fit for {var_surv} in TCGA-BRCA\nCell type = {ctp_surv.replace('_', ' ')}", 
              wrap = True, y = 0.99, fontsize = 20, fontweight = "bold");

fig2.tight_layout(w_pad = 2)

plt.show()

