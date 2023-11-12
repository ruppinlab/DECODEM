#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:01:30 2023

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
import matplotlib.pyplot as plt, seaborn as sns
from functools import reduce
from miscellaneous import date_time, tic, write_xlsx
from machine_learning._functions import (
    EnsembleClassifier, train_pipeline, predict_proba_scaled, 
    get_best_threshold)
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


#%% functions.

## get confident genes for a cell type.
def get_conf_genes(conf, th = 0.99):
    genes = conf[conf.ge(th)].index.tolist()
    return genes


#%% read data.

use_samples = "chemo"

data_path = ["../data/TransNEO/transneo_analysis/", 
             "../data/TCGA/"]

data_file = [f"transneo_data_{use_samples}_v2.pkl"]


## load data.
with open(data_path[0] + data_file[0], "rb") as file:
    data_train_obj   = pickle.load(file)
    exp_all_train    = data_train_obj["exp"]
    resp_pCR_train   = data_train_obj["resp"]
    cell_frac_train  = data_train_obj["frac"]
    conf_score_train = data_train_obj["conf"]
    clin_info_train  = data_train_obj["clin"]
    del data_train_obj


## get TCGA data.
cell_types = {"Cancer_Epithelial": "Cancer", 
              "Endothelial": "Endothelial", 
              "CAFs": "Fibroblast"}

exp_all_test = { }
for ctp_, ctp_tcga_ in cell_types.items():
    data_file.append( f"TCGA_breast_deconvolved_{ctp_tcga_}.txt" )
    exp_all_test[ctp_] = pd.read_table(data_path[1] + data_file[-1], 
                                       header = 0, index_col = 0)

data_file += ["TCGA_breast_deconvolved_cell_fraction.txt", 
              "TCGA_breast_deconvolved_confidence_score.txt", 
              "tcga_survival.tsv", "tcga_drug_response.tsv"]
cell_frac_test  = pd.read_table(data_path[1] + data_file[4], header = 0, 
                                index_col = 0)
conf_score_all  = pd.read_table(data_path[1] + data_file[5], header = 0, 
                                index_col = 0)
conf_score_test = conf_score_all[cell_types.values()].rename(
    columns = {ctp_tcga_: ctp_ for ctp_, ctp_tcga_ in cell_types.items()})


## prepare clinical / survival data.
surv_info_test = pd.read_table(data_path[1] + data_file[6], header = 0, 
                               index_col = 1)
surv_info_test = surv_info_test[surv_info_test.cancer == "BRCA"]

clin_info_test = pd.read_table(data_path[1] + data_file[7], header = 0, 
                               index_col = None)
clin_info_test = clin_info_test[clin_info_test.cancers == "BRCA"]


## get data for common patients.
## trim IDs for deconvolved data.
samples_tcga_decon = reduce(
    np.intersect1d, [exp_.columns for exp_ in exp_all_test.values()] + \
        [cell_frac_test.index.tolist()])
samples_tcga_decon = {
    smpl_: "-".join(smpl_.split("-")[:-1]) for smpl_ in samples_tcga_decon}

exp_all_test = {ctp_: exp_.rename(columns = samples_tcga_decon) \
                for ctp_, exp_ in exp_all_test.items()}
cell_frac_test.rename(index = samples_tcga_decon, inplace = True)

## deconvolved expression & survival data for common IDs.
samples_tcga = np.intersect1d(list(samples_tcga_decon.values()), 
                              surv_info_test.index).tolist()

exp_all_test   = {ctp_: exp_[samples_tcga] \
                  for ctp_, exp_ in exp_all_test.items()}
cell_frac_test = cell_frac_test.loc[samples_tcga]
surv_info_test = surv_info_test.loc[samples_tcga]
resp_surv_test = surv_info_test.drop(columns = ["Redaction", "cancer"]).copy()


#%% prepare data.

conf_th = 0.99                                                                 # confident gene cut-off
genes, X_all_train, X_all_test = { }, { }, { }
for ctp_ in cell_types.keys():
    ## get confident genes.
    gn_ctp_ = np.intersect1d(
        get_conf_genes(conf_score_train[ctp_], th = conf_th), 
        get_conf_genes(conf_score_test[ctp_], th = conf_th) ).tolist()
    
    ## get expression data (append cell type to gene symbols).
    X_ctp_train_ = exp_all_train[ctp_].loc[gn_ctp_].T.rename(
        columns  = lambda gn: f"{gn}__{ctp_}")
    X_ctp_test_  = exp_all_test[ctp_].loc[gn_ctp_].T.rename(
        columns  = lambda gn: f"{gn}__{ctp_}")
    X_ctp_test_  = X_ctp_test_.reset_index().drop_duplicates(
        subset = "index", keep = False).set_index("index").rename_axis(None)   # drop duplicate IDs
    
    ## save data.
    genes[ctp_], X_all_train[ctp_], X_all_test[ctp_] = \
        gn_ctp_, X_ctp_train_, X_ctp_test_

del gn_ctp_, X_ctp_train_, X_ctp_test_

## get response labels.
y_train = resp_pCR_train.loc[X_all_train["CAFs"].index].copy()
y_test  = resp_surv_test.loc[X_all_test["CAFs"].index].copy()


#%% modeling parameters.

# use_ctp = "Cancer_Epithelial"
use_ctp = list(cell_types.keys())

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
cv_seed = 4


#%% model per cell type/combo.

## get parameters.
num_split_rep   = 5
num_splits      = 3
stratify_splits = False
use_mets        = ["AUC", "AP", "ACC", "DOR", "SEN", "PPV", "SPC"]             # list of performance metrics to use


_tic = tic()

## start modeling per cell type.
y_pred_val = { };    th_test_val = { }
for use_ctp_ in use_ctp:
    ## get training & test sets.
    ctp_list = tuple(cell_types) if (use_ctp_[0] == "all") else use_ctp_
    X_train  = pd.concat([X_all_train[ctp_] for ctp_ in ctp_list], axis = 1)
    X_test   = pd.concat([X_all_test[ctp_] for ctp_ in ctp_list], axis = 1)
                
    print(f"""\n
    samples = {use_samples}, cell type = {"+".join(use_ctp_)}
    available #genes = {X_train.shape[1]}, max #features = {num_feat_max}
    model = {use_mdl}, #repetitions = {num_split_rep}
    sample size: training = {X_train.shape[0]}, test = {X_test.shape[0]}""")
    
    
    ## start modeling per repition.
    y_pred_rep = { };    th_test_rep =  { };    # perf_test_rep = { }
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
        
        
        ## save results for this repetition.
        y_pred_rep[f"seed{use_seed}"]  = y_pred[:, 1]
        th_test_rep[f"seed{use_seed}"] = th_fit
        
    
    ## overall performance across all repetitions.
    y_pred_rep          = pd.DataFrame(y_pred_rep)
    y_pred_rep["mean"]  = y_pred_rep.mean(axis = 1)
    th_test_rep         = pd.Series(th_test_rep)
    th_test_rep["mean"] = th_test_rep.mean()
    
    
    ## combine prediction across all repetitions & get performance.
    y_pred_full    = y_pred_rep["mean"]
    print("\nprediction complete!")
    
    
    ## save results for this cell type.
    y_pred_val["+".join(use_ctp_)]  = y_pred_full
    th_test_val["+".join(use_ctp_)] = th_test_rep
    

## fianl performance for all cell types.
y_pred_val  = pd.DataFrame(y_pred_val).set_index(X_test.index)                 # mean prediction matrix
th_test_val = pd.DataFrame(th_test_val).T


_tic.toc()


## prepare data for survival analysis.
print("\npreparing data for survival analysis...")

y_pred_surv_val = { }
for ctp_, y_pred_ in tqdm(y_pred_val.items()):
    y_pred_surv_ = pd.DataFrame(y_pred_.rename("score"))
    
    ## group by various cut-off points.
    y_pred_surv_["groups_med"] = y_pred_.ge(y_pred_.median()).astype(int)
    y_pred_surv_["groups_avg"] = y_pred_.ge(y_pred_.mean()).astype(int)
    y_pred_surv_["groups_05"]  = y_pred_.ge(0.05).astype(int)
    y_pred_surv_["groups_th"]  = y_pred_.ge(
        th_test_val.loc[ctp_, "mean"]).astype(int)
    
    ## group by quantiles.
    qq3 = np.quantile(y_pred_, q = [0.33, 0.67])
    y_pred_surv_["groups_q3"] = np.where(
        y_pred_.lt(qq3[0]), 1, np.where(
            y_pred_.ge(qq3[1]), 3, 2) )
    
    qq4 = np.quantile(y_pred_, q = [0.25, 0.5, 0.75])
    y_pred_surv_["groups_q4"] = np.where(
        y_pred_.lt(qq4[0]), 1, np.where(
            y_pred_.lt(qq4[1]), 2, np.where(
                y_pred_.ge(qq4[2]), 4, 3) ) )
    
    y_pred_surv_ = pd.concat([y_pred_surv_, y_test], axis = 1)
    y_pred_surv_val[ctp_] = y_pred_surv_

del ctp_, y_pred_, y_pred_surv_, qq3, qq4

print("\ndone!")


#%% save full prediction.

svdat = False                                                                  # set True to save results 

if svdat:
    datestamp = date_time()
        
    ## save full predictions & performance.
    out_path = data_path[0] + "mdl_data/"
    out_file = f"tcga_predictions_{use_samples}_th{conf_th}_{use_mdl}_{num_feat_max}features_{num_splits}foldCVtune_{datestamp}.pkl"
    out_dict = {"label": y_test, "pred": y_pred_surv_val, "th": th_test_val, 
                "clin": clin_info_test}
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist
    with open(out_path + out_file, "wb") as file:
        pickle.dump(out_dict, file)
    

#%% survival plot functions.

## functions.
def select_group(df, col, groups = [1, 0]):
    if groups[0] == -1:
        groups[0] = int(col.split("_q")[1]) - 1
    
    df_grp_r, df_grp_nr = [df[df[col] == nn].copy() for nn in groups]
    return df_grp_r, df_grp_nr


def make_kmplot_group(ax, data, event, event_time = None, label = None, 
                      alpha = 0.05, linestyle = "-", linewidth = 12, 
                      color = "#75D0A6", **kwarg):
    if event_time is None:
        event_time = f"{event}_time"
    
    kmf.fit(durations = data[event_time], event_observed = data[event], 
            label = label, alpha = alpha);
    
    mrkprop = {"marker": "P", "markersize": 25}
    kmf.plot(ax = ax, ci_alpha = alpha, linestyle = linestyle, 
             linewidth = linewidth, color = color, **mrkprop);
    
    return ax


#%% make cell-type-specific survival plots. 

svdat = False                                                                  # set True to save figure 

## select cell type for KM plot.
ctp_plt = "Cancer_Epithelial"
print(f"\ngenerating survival plot for cell type = {ctp_plt}")


## plot parameters.
fontname = "sans"
fontdict = {"label": dict(fontfamily = fontname, fontsize = 48, 
                          fontweight = "regular"), 
            "title": dict(fontfamily = fontname, fontsize = 52, 
                          fontweight = "semibold"), 
            "super": dict(fontfamily = fontname, fontsize = 56, 
                          fontweight = "bold"),
            "plabel": dict(fontfamily = "sans-serif", fontsize = 108, 
                           fontweight = "bold")}

sns.set_style("white")
plt.rcParams.update({"xtick.major.size": 12, "xtick.major.width": 4, 
                     "ytick.major.size": 12, "ytick.major.width": 4, 
                     "xtick.bottom": True, "ytick.left": True, 
                     "axes.edgecolor": "#000000", "axes.linewidth": 4})

lgnd_bbox = (1.0, 0.5, 0.6, 0.6)
lgnd_font = {pn.replace("font", ""): pv for pn, pv in fontdict["label"].items()}
lgnd_ttl  = {pn.replace("font", ""): pv for pn, pv in fontdict["title"].items()}

colors    = ["#E08DAC", "#F6CF6D"]


## group parameters.
grp_      = "med"
grp_lbls_ = ["Low risk", "High risk"]
mets_     = ["OS", "PFI"]
met_names = {"OS" : "Overall survival", 
             "PFI": "Progression-free interval", 
             "DSS": "Disease-specific survival", 
             "DFI": "Disease-free interval"}

## make plots.
lr_tests  = { }
fig_title = f"Kaplan-Meier curves for TCGA-BRCA, cell type = {ctp_plt.replace('_', ' ')}"
fig, axs  = plt.subplots(figsize = (44, 16), nrows = 1, ncols = 2, 
                         sharex = True, sharey = True)
kmf       = KaplanMeierFitter()
for nn_, met_ in enumerate(mets_):
    grp1, grp2 = select_group(y_pred_surv_val[ctp_plt], col = f"groups_{grp_}", 
                              groups = [1, 0] if "q" not in grp_ else [-1, 1])
    grp1.dropna(subset = [met_, f"{met_}_time"], how = "any", inplace = True)
    grp2.dropna(subset = [met_, f"{met_}_time"], how = "any", inplace = True)
    td_range = np.arange(0, y_pred_surv_val[ctp_plt].max().max() + 1e3, 2e3, 
                         dtype = int)
    
    ## perform log-rank test.
    lr_test_ = logrank_test(
        durations_A = grp1[f"{met_}_time"], event_observed_A = grp1[met_], 
        durations_B = grp2[f"{met_}_time"], event_observed_B = grp2[met_])
    lr_tests[met_] = lr_test_.summary.squeeze().rename(met_)
        
    ## make KM plots w/ p-values.
    fig_annot = f"Log-rank $p$ = {lr_test_.p_value.round(4)}"
    plt_title = met_names[met_]
    
    ax = axs.ravel()[nn_]    
    ax = make_kmplot_group(data = grp1, event = met_, label = grp_lbls_[0], 
                           linewidth = 6, color = colors[0], ax = ax);
    ax = make_kmplot_group(data = grp2, event = met_, label = grp_lbls_[1], 
                           linewidth = 6, color = colors[1], ax = ax);
    sns.despine(ax = ax, offset = 2, trim = False);                            # keeping axes lines only
    ax.text(x = 250, y = 0.15, s = fig_annot, **fontdict["label"]);
    
    ax.set_xticks(ticks = td_range, labels = td_range, **fontdict["label"]);
    ax.set_yticks(ticks = np.arange(0, 1.4, 0.2), 
                  labels = np.arange(0, 1.2, 0.2).round(1).tolist() + [""], 
                  **fontdict["label"]);
    ax.tick_params(axis = "both", which = "major", 
                   labelsize = fontdict["label"]["fontsize"]);
    ax.set_ylim([-0.02, 1.12]);
    ax.set_title(plt_title, **fontdict["title"]);
    ax.set_xlabel("");      ax.set_ylabel("");
    if nn_ == len(mets_) - 1:
        ax.legend(loc = "lower left", bbox_to_anchor = lgnd_bbox, 
                  prop = lgnd_font, title = "Risk group", 
                  title_fontproperties = lgnd_ttl, frameon = False);
    else:
        ax.legend([ ], [ ], frameon = False)

## finalize plot.
fig.supxlabel("Time in days", x = 0.45, y = 0.005, **fontdict["label"]);
fig.supylabel("Survival probability", x = 0.005, **fontdict["label"]);
fig.suptitle(fig_title, y = 0.995, **fontdict["super"]);

fig.tight_layout(w_pad = 10)
plt.show()

lr_tests = pd.concat(lr_tests, axis = 1)
print(f"""
group sizes: low-risk: {grp1.shape[0]}, high-risk: {grp2.shape[0]}
log-rank test results: \n{lr_tests.round(4)}
""")


## save plot.
if svdat:
    datestamp = date_time()
    fig_path  = data_path[0] + "plots/"
    fig_file  = f"tcga_brca_survival_stratification_{ctp_plt}_th{conf_th}_{use_mdl}_{num_feat_max}features_{datestamp}_v2.pdf"
    
    os.makedirs(fig_path, exist_ok = True)                                     # creates figure dir if it doesn't exist
    fig.savefig(fig_path + fig_file, dpi = "figure")

