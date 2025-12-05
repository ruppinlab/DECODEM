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
from math import nan
from miscellaneous import date_time, tic, write_xlsx
from _functions import (EnsembleClassifier, train_pipeline, 
                        predict_proba_scaled, classifier_performance, 
                        binary_performance, get_best_threshold)
from sklearn.model_selection import StratifiedKFold, KFold
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
# from lifelines.utils import concordance_index
from tqdm import tqdm
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

data_path = ["../../data/TransNEO/transneo_analysis/", 
             "../../data/TCGA/validation/"]

data_file = [f"transneo_data_{use_samples}_v2.pkl", 
             "tcga_brca_data_surv_her2neg.pkl"]


## load data.
print("loading & preparing data...");    _tic = tic()

(exp_all_train, resp_pCR_train, cell_frac_train, 
 conf_score_train, clin_info_train) = load_data(data_path[0] + data_file[0])

(exp_all_test, resp_surv_test, cell_frac_test, 
 conf_score_test, clin_info_test)   = load_data(data_path[1] + data_file[1]) 
cell_frac_test, conf_score_test     = [
    dat.rename(columns = lambda x: x.replace(" ", "_"))                        # match cell type names betn. exp & cell fraction
    for dat in [cell_frac_test, conf_score_test]]
resp_surv_test = resp_surv_test.set_index(
    keys = "Sample_ID").replace(
    to_replace = {-2147483648: nan})                                           # saved from R- turned NA into large negative integer


if conf_score_train.columns.tolist() == conf_score_test.columns.tolist():
    cell_types = conf_score_train.columns.tolist()
else:
    raise ValueError("cell types are not the same between training and test data!")


## keep only early-stage patients.
## source: https://www.cancer.gov/publications/dictionaries/cancer-terms/def/early-stage-breast-cancer
## definition: breast cancer that has not spread beyond the breast or the 
## axillary lymph nodes. This includes ductal carcinoma in situ and stage I, 
## stage IIA, stage IIB, and stage IIIA breast cancers. 
stage_early    = np.array(["I", "IA", "IB", "II", "IIA", "IIB", "IIIA"])
clin_info_test = clin_info_test.pipe(
    lambda df: df[df.Stage.replace(
        regex = {"Stage ": ""}).map(
        lambda x: any(x == stage_early)) ])
clin_info_test["Clinical_subtype"] = clin_info_test.ER_status.map(
    lambda x: "ER+/HER2-" if (x == "Positive") else "TNBC")

exp_all_test   = {ctp_: exp_[clin_info_test.index] 
                  for ctp_, exp_ in exp_all_test.items()}
resp_surv_test = resp_surv_test.loc[clin_info_test.index]
cell_frac_test = cell_frac_test.loc[clin_info_test.index]

print("done!", end = "");    _tic.toc()


print(f"""
dataset summary:
cohort     = TransNEO (n = {resp_pCR_train.size:,}) [training], TCGA-BRCA (HER2-; n = {resp_surv_test.size:,}) [validation-survival]
treatment  = {use_samples + 'therapy'}, response = Survival (OS / PFI; 'Low-risk' vs. 'High-risk')
cell types = {', '.join(cell_types)}
""")


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


print(f"""
dataset sizes: 
train = { {ctp_: X_.shape for ctp_, X_ in X_all_train.items()} }
test  = { {ctp_: X_.shape for ctp_, X_ in X_all_test.items()} }
""", end = "")


#%% modeling parameters.

use_ctp = np.append(cell_types, "Bulk").tolist()                               # all individual cell types + Bulk 
# use_ctp = "Cancer_Epithelial"                                                  # a single cell type


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


## CV parameters.
tune_seed = 84
cv_seed   = 4


#%% model per cell type/combo.

filterwarnings(action = "ignore")                                              # suppress fit failing/convergence warnings

## get parameters.
num_split_rep   = 5
num_splits      = 3
stratify_splits = False
use_mets        = ["AUC", "AP", "ACC", "DOR"]                                  # list of performance metrics to use


_tic = tic()

## start modeling per cell type.
y_pred_val, th_test_val = { }, { }
for use_ctp_ in use_ctp:
    ## get training & test sets.
    ctp_list = tuple(cell_types) if (use_ctp_[0] == "all") else use_ctp_
    X_train  = pd.concat([X_all_train[ctp_] for ctp_ in ctp_list], axis = 1)
    X_test   = pd.concat([X_all_test[ctp_] for ctp_ in ctp_list], axis = 1)
    ctp_mdl = "+".join(use_ctp_)                                               # cell-type model name
    
                
    print(f"""\n
    samples = {use_samples}, cell type = {"+".join(use_ctp_)}
    available #genes = {X_train.shape[1]:,}, max #features = {num_feat_max}
    model = {use_mdl}, #repetitions = {num_split_rep}
    sample size: training = {X_train.shape[0]:,}, test = {X_test.shape[0]:,}
    """)
    
    
    ## start modeling per repition.
    y_pred_rep, th_test_rep = { }, { }
    for use_seed in range(num_split_rep):
        print(f"\nsplit seed = {use_seed}")
        rep_mdl = f"seed{use_seed}"                                            # repetition model name
        
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
            pipes_mdl, params_mdl = { }, { }
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
                
        
        ## save results for this repetition.
        y_pred_rep[rep_mdl]  = y_pred[:, 1]
        th_test_rep[rep_mdl] = th_fit
                
        
    ## overall performance across all repetitions.
    y_pred_rep          = pd.DataFrame(y_pred_rep)
    y_pred_rep["mean"]  = y_pred_rep.mean(axis = 1)
    th_test_rep         = pd.Series(th_test_rep)
    th_test_rep["mean"] = th_test_rep.mean()
    
    
    ## combine prediction across all repetitions & get performance.
    y_pred_full = y_pred_rep["mean"]
    
    # print(os.system("clear"))                                                  # clears console
    
    
    ## save results for this cell type.
    y_pred_val[ctp_mdl]  = y_pred_full
    th_test_val[ctp_mdl] = th_test_rep
    

## fianl performance for all cell types.
y_pred_val  = pd.DataFrame(y_pred_val).set_index(keys = X_test.index)          # mean prediction matrix
th_test_val = pd.DataFrame(th_test_val).T

print(os.system("clear"))                                                      # clears console
print(f"""
{'-' * 64}
prediction complete for survival analysis! 
cohort = TCGA-BRCA (HER2-; n = {len(y_test_surv):,})
""")


## prepare data for survival analysis.
print("\npreparing data for survival analysis...")

y_pred_surv_val = {ctp_: pd.DataFrame({
        "score"     : y_pred_, 
        "groups_th" : y_pred_.gt(th_test_val.loc[ctp_, "mean"]).astype(int),   # split by learned threshold
        "groups_med": y_pred_.gt(y_pred_.median()).astype(int),                # split by median score
        "groups_avg": y_pred_.gt(y_pred_.mean()).astype(int),                  # split by mean score
        "groups_05" : y_pred_.gt(0.5).astype(int),                             # split by 0.5
        "groups_q4" : pd.qcut(y_pred_, q = 4, labels = False),                 # split into 4 quartiles
        "groups_q3" : pd.qcut(y_pred_, q = 3, labels = False) })               # split into 3 quartiles
    for ctp_, y_pred_ in tqdm(y_pred_val.items(), total = y_pred_val.shape[1])}

print("\ndone!")

_tic.toc()


#%% save full prediction.

svdat = False                                                                  # set as True to save data

if svdat:
    datestamp = date_time()
        
    ## save full predictions & performance.
    out_path = data_path[0] + "mdl_data/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    
    out_file = f"tcga_predictions_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.pkl"
    out_dict = {"label": y_test_surv, "pred": y_pred_surv_val, 
                "th": th_test_val,    "clin": clin_info_test}
    with open(out_path + out_file, mode = "wb") as file:
        pickle.dump(out_dict, file)
    print(out_file)
    

#%% plot functions.

## make forest plots with hazard ratios & p-values.
def make_forest_plot(ax, data, x, xlow, xhigh, stat = None, capsize = 0.05, 
                     colors = None, title = None, xlabel = None, 
                     fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {"label": {"size": 14, "weight": "regular"}, 
                    "title": {"size": 18, "weight": "bold"}}
    if colors is None:
        colors   = ["#B075D0", "#000000"]
    
    mrkrprop = {"s": 150, "marker": "s", "lw": 2, "c": colors[0], 
                "alpha": 0.8, "ec": colors[-1]}
    lineprop = {"color": colors[-1], "lw": 2, "ls": "-"}
    baseprop = {"color": colors[-1], "lw": 1.5, "ls": "--"}
    
    
    ## prepare data.
    data["var_no"]   = range(len(data))
    data["cap_low"]  = data.var_no - capsize / 2
    data["cap_high"] = data.var_no + capsize / 2
    data["xlow1"]    = data[x] - (0.15 - 0.02)
    data["xhigh1"]   = data[x] + (0.15 - 0.02)
    if stat is not None:
        data["annot"] = data[stat].map(lambda p: f"$P$ = {p:0.3}")
    
    
    ## make plot.
    ax.scatter(data = data, x = x, y = "var_no", **mrkrprop)                   # main box
    [ax.hlines(data = data, y = "var_no", xmin = xs, xmax = xe, **lineprop)    # add lines surrounding box
     for xs, xe in [(xlow, "xlow1"), ("xhigh1", xhigh)]]
    [ax.vlines(data = data, x = xpos, ymin = "cap_low", ymax = "cap_high",     # add caps at end
               **lineprop) for xpos in [xlow, xhigh]]
    ax.axvline(x = 1, ymin = 0, ymax = 1.95, **baseprop)                       # make baseline
    sns.despine(ax = ax, offset = 0, trim = True, left = True);
    
    if stat is not None:                                                       # add p-value
        p_ax = ax.twinx()
        p_ax.set_yticks(ticks = data.var_no, labels = data.annot)
        p_ax.tick_params(axis = "y", labelright = True, labelleft = False, 
                         labelsize = fontdict["label"]["size"], length = 0)    # add values to the right
        p_ax.set_ylim([-0.5, len(data) - 0.5]);
        sns.despine(ax = p_ax, offset = 0, trim = True, left = True);
    
    ## format ticks & labels.
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    ax.tick_params(axis = "y", length = 0);
    ax.set_ylim([-0.5, len(data) - 0.5]);
    ax.set_yticks(ticks = data.var_no, labels = data.index);
    ax.set_xlabel("Hazard Ratio (95% CI)" if (xlabel is None) else xlabel, 
                  y = -0.01, **fontdict["label"]);
    ax.set_title(title, wrap = True, y = 1.02, **fontdict["title"]);
    
    return ax


## make Kaplan-Meier plots for two groups.
def make_km_plot(ax, data1, data2, stat = None, ci_alpha = 0.15, 
                 colors = None, title = None, ylabel = None, legend = True, 
                 legend_title = None, fontdict = None):
    ## plot parameters.
    if fontdict is None:
        fontdict = {"label": {"size": 14, "weight": "regular"}, 
                    "title": {"size": 18, "weight": "bold"}}
        
    if colors is None:
        colors   = ["#E08DAC", "#7595D0", "#000000"]
    
    lineprop = {"ls": "-", "lw": 2}
    
    lgndttl  = "Risk group" if (legend_title is None) else legend_title
    lbls     = [f"{data1.label} (n = {len(data1.durations):,})", 
               f"{data2.label} (n = {len(data2.durations):,})"]
    
    
    ## make plots.
    ax = data1.plot(show_censors = True, ci_show = True, color = colors[0], 
                    ci_alpha = ci_alpha, ax = ax, **lineprop)
    ax = data2.plot(show_censors = True, ci_show = True, color = colors[1], 
                    ci_alpha = ci_alpha, ax = ax, **lineprop)
    add_at_risk_counts(data1, data2, labels = lbls, 
                       rows_to_show = None, ax = ax, **fontdict["label"]);     # at-risk counts below the plots
    sns.despine(ax = ax, offset = 0, trim = False);
    
    if stat is not None:                                                       # add p-value
        ax.text(x = 250, y = 0.20, s = f"Log-rank $P$ = {stat.p_value:0.3g}", 
                color = colors[-1], **fontdict["label"]);
    
    ## format ticks & labels.
    ax.set_ylim([-0.1, 1.1]);
    ax.set_yticks(np.arange(0, 1.2, 0.2));
    ax.tick_params(axis = "both", labelsize = fontdict["label"]["size"]);
    if legend:
        ax.legend(loc = (1.06, 0.25), title = lgndttl, prop = fontdict["label"], 
                  title_fontproperties = fontdict["title"]);
    else:
        ax.legend([ ], [ ], frameon = False);
    
    ax.set_xlabel("Time in days", y = -0.02, **fontdict["label"]);
    ax.set_ylabel("Survival proabibility" if ylabel is None else ylabel, 
                  x = 0.01, **fontdict["label"]);
    ax.set_title(title, wrap = True, y = 1.02, **fontdict["title"]);
    
    return ax


#%% do Cox regression. 
## CE, ENDO, PB, NE, MYL, B, CAF

ctp_surv = "Cancer_Epithelial"
var_surv = "OS"

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
    by = "Subtype", sort = True).apply(
    lambda df: df, include_groups = False)


## model for each subtype.
mdl_params  = dict(baseline_estimation_method = "breslow", alpha = 0.05, 
                   penalizer = 0.001, l1_ratio = 0.1)
fit_params  = dict(event_col = var_surv, duration_col = f"{var_surv}_time")
cph_fits_sb = {
    sb: CoxPHFitter(**mdl_params).fit(cph_data_ctp.loc[sb], **fit_params) 
    for sb in cph_data_ctp.index.levels[0]}


## display results.
keep_cols   = {"exp(coef)"          : "HR", 
               "exp(coef) lower 95%": "HR_low_95", 
               "exp(coef) upper 95%": "HR_high_95", 
               "p"                  : "pval"}
cph_res_sb  = {
    sb: mdl.summary[list(keep_cols)].rename(columns = keep_cols)
    for sb, mdl in cph_fits_sb.items()}
cph_cidx_sb = {sb: mdl.concordance_index_ for sb, mdl in cph_fits_sb.items()}


print(f"""
performed cox regression for cell type = {ctp_surv}
confounding variables = {cph_data_ctp.columns[1:-2].tolist()}
results:
""", end = "")
[print(f"""
{sb}: C-index = {cph_cidx_sb[sb]:0.4}
{cph_res_sb[sb].map(lambda x: f'{x:0.4}')}
""", end = "") for sb in cph_res_sb];


#%% do Kaplan-Meier fits.
## CE, ENDO, PB, NE, MYL, B, CAF

# ctp_surv  = "Cancer_Epithelial"
# var_surv  = "OS"
endpoints = {"OS": "Overall Survival", "PFI": "Progression-free interval"}
var_group = "groups_05"

## get data for KM plot.
km_data_ctp = pd.DataFrame({
    "Score"  : y_pred_surv_val[ctp_surv].score, 
    "Group"  : y_pred_surv_val[ctp_surv][var_group].replace(
        to_replace = {1: "High-score", 0: "Low-score"}).infer_objects(
        copy = False), 
    "Subtype": clin_info_test.Clinical_subtype })
            
km_data_ctp = pd.concat([
    km_data_ctp, y_test_surv[[var_surv, f"{var_surv}_time"]]], 
    axis = 1).groupby(
    by = ["Subtype", "Group"], sort = True).apply(
    lambda df: df, include_groups = False)


## model for each subtype.       
km_res_sb = {
    sb: {grp: KaplanMeierFitter(
            alpha = 0.05, label = grp).fit(
            event_observed = km_data_ctp.loc[(sb, grp), var_surv], 
            durations = km_data_ctp.loc[(sb, grp), f"{var_surv}_time"]) 
         for grp in km_data_ctp.index.levels[1] }
    for sb in km_data_ctp.index.levels[0] }


## log-rank tests.
km_lr_sb  = {
    sb: km_data_ctp.loc[sb].pipe(
        lambda df: logrank_test(
            event_observed_A = df.loc["High-score", var_surv], 
            durations_A      = df.loc["High-score", f"{var_surv}_time"], 
            event_observed_B = df.loc["Low-score", var_surv], 
            durations_B      = df.loc["Low-score", f"{var_surv}_time"]))
    for sb in km_data_ctp.index.levels[0] }


print(f"""
performed Kaplan-Meier fitting for cell type = {ctp_surv}
group = {var_group}
Log-rank P-values: { {sb: float(stat.p_value.round(4)) for sb, stat in km_lr_sb.items()} }
""")


#%% make survival plots.

svdat = False                                                                  # set as True to save data

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

fontdict = {"label": {"size": 14, "weight": "regular", "linespacing": 1.2}, 
            "title": {"size": 16, "weight": "bold", "linespacing": 1.5}, 
            "super": {"size": 20, "weight": "bold", "linespacing": 1.5}}
txtprop  = {"wrap": "True", "ha": "center", "va": "top", "ma": "center"}

fig, axs = plt.subplots(figsize = (16, 10), nrows = 2, ncols = 2, 
                        height_ratios = [1.0, 0.8])

## make plots.
for (sb, dat), ax in zip(km_res_sb.items(), axs[0]):                           # KM plots
    ax = make_km_plot(data1 = dat["High-score"], data2 = dat["Low-score"], 
                      stat = km_lr_sb[sb], legend = (sb == "TNBC"), 
                      title = f"{sb} (n = {len(km_data_ctp.loc[sb]):,})", 
                      ax = ax)

for (sb, dat), ax in zip(cph_res_sb.items(), axs[1]):                          # forest plots
    ax = make_forest_plot(data = dat, x = "HR", xlow = "HR_low_95", 
                          xhigh = "HR_high_95", stat = "pval", capsize = 0.15, 
                          ax = ax)
    ax.text(x = 1, y = 3, s = f"C-index = {cph_cidx_sb[sb]:0.3}", 
            color = "#000000", **txtprop, **fontdict["label"])

## format ticks & labels.
for ax in axs[:, 1]:
    ax.set_ylabel(None);    ax.set_yticklabels([""] * len(ax.get_yticks()))

fig.suptitle(f"{endpoints[var_surv]} stratification for cell type = {ctp_surv.replace('_', ' ')}", 
             y = 0.99, **txtprop, **fontdict["super"]);

fig.tight_layout(w_pad = 3, h_pad = 2)
plt.show()


## save figures.
if svdat:
    fig_path = data_path[0] + "../plots/final_plots7/"    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    
    fig_file = f"prediction_survival_{var_surv}_{ctp_surv}_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_{num_splits}foldCV.pdf"
    fig.savefig(fig_path + fig_file, dpi = 600)
    print(fig_file)

