#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:11:28 2022

@author: dhrubas2
"""

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from math import nan, inf, ceil
from scipy.stats import hmean, gmean, mannwhitneyu, ttest_ind_from_stats
from itertools import combinations
# from statannotations.Annotator import Annotator
from sklearn.base import (BaseEstimator, RegressorMixin, 
                          TransformerMixin, clone)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, 
                                       chi2, f_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             confusion_matrix, RocCurveDisplay, 
                             PrecisionRecallDisplay)
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
# from sklearnex import patch_sklearn, unpatch_sklearn
# patch_sklearn()


#%% classifiers.

class EnsembleClassifier(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        self.clf_ = [ ]
        for _clf in self.models:
            self.clf_.append( clone(_clf) )
            self.clf_[-1].fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        y_proba = [_clf.predict_proba(X) for _clf in self.clf_]
        y_proba = np.stack(y_proba, axis = 2).mean(axis = 2)
        
        return y_proba


class MakeClassifier:
    def __init__(self, model, class_weight = "balanced", random_state = None):
        self.model = model
        self.random_state = random_state
        self.class_weight = class_weight
    
    def get_model(self):
        self.clf_list_ = { }
        
        # init list of models.
        self.clf_list_["LR"] = LogisticRegression(
            C = 0.9, solver = "saga", penalty = "elasticnet", max_iter = 1000, 
            l1_ratio = 0.2, class_weight = self.class_weight, 
            n_jobs = -1, random_state = self.random_state
        )
        self.clf_list_["RF"] = RandomForestClassifier(
            n_estimators = 150, criterion = "gini", 
            class_weight = self.class_weight, 
            n_jobs = -1, random_state = self.random_state
        )
        self.clf_list_["SVM"] = SupportVectorClassifier(
            kernel = "rbf", C = 0.9, probability = True, degree = 3, 
            tol = 1e-4, class_weight = self.class_weight, 
            random_state = self.random_state
        )
        self.clf_list_["XGB"] = XGBClassifier(
            n_estimators = 150, booster = "gbtree", eval_metric = "auc", 
            learning_rate = 0.1, min_child_weight = 1.5, max_depth = 5, 
            gamma = 0.05, reg_lambda = 0.9, reg_alpha = 0.3, subsample = 0.8, 
            # scale_pos_weight = 1, early_stopping_rounds = 10, 
            verbosity = 1, n_jobs = -1, random_state = self.random_state
        )
        
        self.clf_ = self.clf_list_[self.model.upper()]
        
        return self.clf_
    
    def get_param(self):
        self.param_list_ = { }
        
        # init list of tunable parameters.
        self.param_list_["LR"] = dict(
            l1_ratio = np.arange(0.1, 1, 0.2), 
            C = np.arange(0.1, 1, 0.1)
        )
        self.param_list_["RF"] = dict(
            n_estimators = [50, 100, 150, 200], 
            min_samples_split = np.arange(3, 11, 2), 
            min_samples_leaf = np.arange(1, 9, 2), 
        )
        self.param_list_["SVM"] = dict(
            kernel = ["linear", "rbf", "poly", "sigmoid"], 
            C = np.arange(0.1, 1, 0.1) 
        )
        self.param_list_["XGB"] = dict(
            # n_estimators = [50, 100, 150, 200], 
            learning_rate = np.arange(0.05, 0.35, 0.05), 
            # min_child_weight = np.arange(0.5, 3, 0.1), 
            # max_depth = np.arange(3, 10, 1), 
            reg_lambda = np.arange(0.1, 1, 0.2), 
            reg_alpha = np.arange(0.1, 1, 0.2)
        )
        
        self.params_ = self.param_list_[self.model.upper()]
        
        return self.params_


#%% pipeline functions.

def train_pipeline(model, train_data, max_features = 20, var_th = 0.1, 
                   cv_tune = None, mdl_seed = 86420, tune_seed = 84):
    ## make pipeline.
    clf_obj = MakeClassifier(model = model, random_state = mdl_seed)
    pipe = Pipeline([
        ("var_filter", VarianceThreshold(threshold = var_th)), 
        ("selector",   SelectKBest(score_func = f_classif)), 
        ("normalizer", StandardScaler()), 
        ("classifier", clf_obj.get_model())
    ])
    
    ## get parameter grid.
    num_feat_max = max_features
    num_feat_var_ = pipe[0].fit(*train_data).get_support().sum()               # #features after variance filtering
    if (num_feat_max == "all") or (num_feat_max > num_feat_var_):
        num_feat_max = num_feat_var_
    
    params = { }
    params = {"selector__k": np.arange(2, num_feat_max + 1, 1)}
    params.update({
        f"classifier__{pn}": pv for pn, pv in clf_obj.get_param().items()
    })
    
    ## tune pipeline.
    if cv_tune is None:
        cv_tune = StratifiedKFold(n_splits = 3, shuffle = True, 
                                  random_state = 4)
        
    grid = RandomizedSearchCV(
        pipe, params, scoring = "roc_auc", n_iter = 20, cv = cv_tune, 
        refit = True, n_jobs = -1, random_state = tune_seed, verbose = 1, 
        return_train_score = True
    )
    
    grid.fit(*train_data)
    
    return grid.best_estimator_, grid.best_params_


## get classification scores from pipeline & rescale in [0, 1].
def predict_proba_scaled(pipe, X, scale = True):
    y_score = pipe.predict_proba(X)
    if scale:
        y_score = MinMaxScaler().fit_transform(y_score)
    
    return y_score


#%% pipelines.

class MakePipeline:
    def __init__(self, model, random_state = 86420, max_features = 20, var_th = 0.1):
        self.model = model
        self.random_state = random_state
        self.max_features = max_features
        self.var_th = var_th
    
    def build_pipe(self):
        # make pipeline.
        self.clf_obj_ = MakeClassifier(
            model = self.model, random_state = self.random_state
        )
    
        self.pipe = Pipeline([
            ("var_filter", VarianceThreshold(threshold = self.var_th)), 
            ("selector",   SelectKBest(score_func = f_classif)), 
            ("normalizer", StandardScaler()), 
            ("classifier", self.clf_obj_.get_model())
        ])
        
        # get parameters to tune.
        self.params = { }
        self.params["selector__k"] = np.arange(2, self.max_features + 1, 1)
        self.params.update({
            f"classifier__{pn}": pv for pn, pv in self.clf_obj_.get_param().items()
        })
        
        return self
    
    def get_pipe(self):
        self.build_pipe()
        return self.pipe, self.params
    
    def fit(self, X, y, score = "roc_auc", n_iter = 20, cv = None, random_state = 84):
        self.score_ = score
        self.n_iter_ = n_iter
        self.tune_seed_ = random_state
        if cv is None:
            self.cv_seed_ = 4
            self.cv_tune_ = StratifiedKFold(
                n_splits = 3, shuffle = True, random_state = self.cv_seed_
            )
        else:
            self.cv_tune_ = cv
        
        # make pipeline.
        self.build_pipe()
        
        # tune pipeline.
        self.grid_ = RandomizedSearchCV(
            self.pipe, self.params, scoring = self.score_, 
            n_iter = self.n_iter_, cv = self.cv_tune_, refit = True, 
            random_state = self.tune_seed_, verbose = 1, 
            return_train_score = False
        )
        
        self.grid_.fit(X, y)
        
        return self.grid_.best_estimator_, self.grid_.best_params_


#%% performance metrics.

## classifier performance over all thresholds.
def classifier_performance(y_true, y_pred):
    """ y_pred: probability score from the classifier (class = 1) """
    
    scores = {"AUC": roc_auc_score(y_true, y_pred), 
              "AP": average_precision_score(y_true, y_pred)}
    
    return scores


def binary_performance(y_true, y_pred):
    """ y_pred: binary score after thresholding """
    
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels = [1, 0]).ravel()
    scores = {
        "ACC"        : (tp + tn) / (tp + fp + fn + tn), 
        "Precision"  : tp / (tp + fp) if (tp + fp > 0) else nan,  
        "Recall"     : tp / (tp + fn) if (tp + fn > 0) else nan, 
        "DOR"        : nan, 
        "FracResp"   : nan, 
        "Sensitivity": nan, 
        "Specificity": tn / (tn + fp) if (tn + fp > 0) else nan
    }
    
    ## aliases.
    scores["Sensitivity"] = scores["Recall"]
    scores["FracResp"]    = scores["Recall"]
    # scores["F1-score"]    = hmean([scores["Precision"], scores["Recall"]])
    scores["F1-score"]    = 2 * tp / (2 * tp + fp + fn)
    scores["BACC"]        = np.mean([scores["Sensitivity"], scores["Specificity"]])
    scores["MCC"]         = nan    
    scores["PPV"], scores["NPV"] = scores["Precision"], tn / (tn + fn) \
                                        if (tn + fn > 0) else nan
    scores["TPR"], scores["TNR"] = scores["Sensitivity"], scores["Specificity"]
    scores["SEN"], scores["SPC"] = scores["Sensitivity"], scores["Specificity"]
    
    ## calculate odds ratio.
    dor_num, dor_den = tp * tn, fp * fn
    if dor_den == 0:
        if dor_num != 0:
            scores["DOR"] = inf
    else:
        scores["DOR"] = dor_num / dor_den
    
    ## calculate MCC.
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if mcc_den == 0:
        if mcc_num != 0:
            scores["MCC"] = inf
    else:
        scores["MCC"] = mcc_num / mcc_den
        
    return scores

def get_best_threshold(y_true, y_pred, curve = "ROC", th_range = None):
    """ finds the best classification threshold based on ROC/PR curve """
    
    # get threshold range.
    if th_range is None:
        th_range = np.arange(0, 1.01, 0.01)
    
    # compute performance metrics for all thresholds.
    # tpr = recall = sensitivity, fpr = 1 - specificity, ppv = precision
    metrics_all = { }
    for th in th_range:
        y_pred_th = (y_pred >= th).astype(int)
        tp, fn, fp, tn = confusion_matrix(y_true, y_pred_th, 
                                          labels = [1, 0]).ravel()
        metrics_all[th] = {"tpr": tp / (tp + fn) if (tp + fn > 0) else nan, 
                           "fpr": fp / (fp + tn) if (fp + fn > 0) else nan, 
                           "ppv": tp / (tp + fp) if (tp + fp > 0) else nan, 
                           "f1_score": 2*tp / (2*tp + fp + fn) \
                               if (2*tp + fp + fn > 0) else nan}
                
    # get best threshold.
    metrics_all = pd.DataFrame(metrics_all).T
    if curve.upper() == "ROC":
        metrics_all["j_stat"] = metrics_all["tpr"] - metrics_all["fpr"]
        th_best = metrics_all["j_stat"].idxmax()
    elif curve.upper() == "PR":
        th_best = metrics_all["f1_score"].idxmax()
        # th_best = metrics_all["ppv"].idxmax()
    else:
        raise ValueError("only defined for curve = 'ROC' or curve = 'PR'!")
    
    return th_best


#%% plot functions.

## list of color palettes:
## 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

## make barplots for one variable only- with or without break in y-axis.
def make_barplot1(data, x, y, ax, fontdict, title, xlabels, xrot = 0, 
                  ylabel = None, bline = False, width = 0.8):
    ## color palettes.
    # col_palette = "CMRmap"
    # col_palette = "crest"
    # col_bar     = "#75D0A6"
    col_bar     = "#75D0B0"
    col_bline   = "#DCDDDE"
    bline_prop  = {"linestyle": "--", "linewidth": 2}
    
    ## if data contains infinity- plot a barplot with breakpoints.
    if data[y].eq(inf).any():
        ## plot parameters.
        bar_lbl  = data[y].round(2).replace(inf, "$\infty$").tolist()          # bar heights as labels
        max_plt  = 100
        max_val  = ceil(data[y].replace(inf, nan).max()) + 2
        data_plt = data.replace(inf, max_plt)                                  # replace 'inf' with max_plt for plotting
                
        ## build plot.
        ax_t, ax_b = ax
        ax_t.spines["bottom"].set_visible(False)
        ax_t.tick_params(axis = "x", which = "both", bottom = False);
        ax_b.spines["top"].set_visible(False)
        
        ## limit both panels to desired ranges.
        ylims_b = [0, max_val];         ylims_t = [max_plt - 4, max_plt + 2]
        ax_b.set_ylim(ylims_b);         ax_b.set_xlim(0, data.shape[0] + 1);
        ax_t.set_ylim(ylims_t);         ax_t.set_xlim(0, data.shape[0] + 1);
        
        ax_b = sns.barplot(data = data_plt, x = x, y = y, orient = "v", 
                           color = col_bar, edgecolor = [0.3]*3, 
                           saturation = 0.9, width = width, ax = ax_b)
        ax_t = sns.barplot(data = data_plt, x = x, y = y, orient = "v", 
                           color = col_bar, edgecolor = [0.3]*3, 
                           saturation = 0.9, width = width, ax = ax_t)
        
        ## format plot ticks & labels.
        ylabels_t = [100, 1000, "$\infty$"]                                    # custom yticks for top plot
        ax_t.set_xlabel(None);      ax_t.set_ylabel(None);
        ax_t.set_xticklabels([ ]);
        ax_t.set_yticks(ticks = np.arange(*ylims_t, 2), labels = ylabels_t, 
                        **fontdict["label"]);
        ax_t.bar_label(ax_t.containers[0], labels = bar_lbl, 
                       **fontdict["label"]);                                   # add bar heights as labels for top plot (infinity)
        
        ax_b.set_xlabel(None);      ax_b.set_ylabel(None);
        # ax_b.set_xticklabels(xlabels, rotation = 0, **fontdict["label"]);      # add model names as xticks for bottom plot
        if xrot != 0:
            ax_b.set_xticklabels(xlabels, rotation = xrot, 
                                 rotation_mode = "anchor", ha = "right", 
                                 va = "center", ma = "center", 
                                 position = (0, -0.02), **fontdict["label"]);
        else:
            ax.set_xticklabels(xlabels, ma = "center", position = (0, -0.02), 
                               **fontdict["label"]);
        
        ax_b.tick_params(axis = "both", which = "major", 
                         labelsize = fontdict["label"]["fontsize"]);
        ax_b.bar_label(ax_b.containers[0], labels = bar_lbl, 
                       **fontdict["label"]);                                   # add bar heights as labels for bottom plot (non-infinity)
                
        ## add breakpoint indicators.
        d_brk = 0.005                                                          # axis breakpoint tick size
        brkargs = dict(transform = ax_t.transAxes, color = "black", 
                       clip_on = False)
        ax_t.plot((-d_brk, +d_brk), (-d_brk, +d_brk), **brkargs)               # add breakpoint marker tick for top plot
        ax_t.plot((1 - d_brk, 1 + d_brk), (-d_brk, +d_brk), **brkargs)
        
        brkargs.update(transform = ax_b.transAxes)
        ax_b.plot((-d_brk, +d_brk), (1 - d_brk, 1 + d_brk), **brkargs)         # add breakpoint marker tick for bottom plot
        ax_b.plot((1 - d_brk, 1 + d_brk), (1 - d_brk, 1 + d_brk), **brkargs)
        
        ## add baseline.
        if bline:
            ax_b.axhline(y = 1, xmin = 0.5, xmax = data.shape[0] - 0.5, 
                         color = col_bline, **bline_prop);
            
        plt.suptitle(title, **fontdict["super"]);
        ax = ax_t, ax_b
    
    ## do regular barplot.
    else:
        ## parameters.
        bar_lbl = data[y].round(2)                                             # bar heights as labels
        max_val = np.round(data[y].max(), 1)
        if max_val <= 1:
            max_val += 0.4;     step = 0.2
            max_val += (max_val * 10) % 2 / 10
            # max_val = min(1.2, max_val)
        else:
            max_val += 8;       step = 4
            max_val += (max_val % 2)
        
        yticks = np.arange(0, max_val, step)
        ylabels = yticks.round(2).tolist()[:-1] + [""]
        
        ## build plot.
        sns.barplot(data = data, x = x, y = y, orient = "v", width = width, 
                    color = col_bar, edgecolor = [0.3]*3, saturation = 0.9, 
                    ax = ax);
        ax.bar_label(ax.containers[0], labels = bar_lbl, **fontdict["label"]);
        
        ## add baseline.
        if bline:
            ax.axhline(y = 1, xmin = 0, xmax = data.shape[0], 
                       color = col_bline, **bline_prop);
        
        ## format plot ticks & labels.
        ax.set_title(title, **fontdict["super"]);
        ax.set_xlabel(None);    ax.set_ylabel(ylabel, **fontdict["label"]);
        # ax.set_xticklabels(xlabels, rotation = 0, **fontdict["label"]);
        if xrot != 0:
            ax.set_xticklabels(xlabels, rotation = xrot, 
                               rotation_mode = "anchor", ha = "right", 
                               va = "center", ma = "center", 
                               position = (0, -0.02), **fontdict["label"]);
        else:
            ax.set_xticklabels(xlabels, ma = "center", position = (0, -0.02), 
                               **fontdict["label"]);
        ax.set_yticks(ticks = yticks, labels = ylabels, **fontdict["label"]);
        ax.tick_params(axis = "y", labelsize = fontdict["label"]["fontsize"]);
    
    return ax


## make nested barplots for multiple variables.
def make_barplot2(data, x, y, hue, ax, fontdict, title, xlabels, 
                  xrot = 0, colors = None, ylabel = None, width = 0.8, 
                  bline = False, yrefs = None, legend = True, 
                  legend_title = None, bar_label_align = True, offset = 2, 
                  trim = False, lw = 4):
    
    ## parameters.
    ## set colors.
    col_palette = dict(zip(data[hue].unique(), 
                           # ["#D0759F", "#88BECD"]))
                           # ["#75D0A6", "#CD9788"]))
                           # ["#75D0A6", "#FFD700"]))
                           # ["#D0759F", "#88BECD"]))
                           # ["#88BECD", "#D0759F"]))
                           # ["#75D0C1", "#FFD744"]))
                           # ["#75D0B0", "#FFD788"]))
                           # ["#8ADBFF", "#FFB9DB"]))
                           ## ["#75D0B0", "#FFC72C"]))
                           # ["#D07595", "#7595D0"]))
                           ["#E08DAC", "#F6CF6D"]))
    
    if colors is not None:
        col_palette = dict(zip(col_palette.keys(), colors))
    
    
    ## set lines/edges.
    bar_prop  = {"linestyle": "-", "linewidth": lw, "edgecolor": "#000000"}
    line_prop = {"linestyle": "--", "linewidth": lw, "color": "#000000"}
    
    ## set bar labels.
    n_bars  = data[hue].nunique()
    bar_lbl = [data[data[hue].eq(bb)][y] for bb in data[hue].unique()]         # bar heights as labels
    if bar_label_align:
        spc = " " * 3                                                          # insert space in bar labels for better alignment
        bar_lbl = [[f"{x:0.2f}{spc}" for x in bar_lbl[0]], 
                   [f"{spc}{x:0.2f}" for x in bar_lbl[1]]]
    else:
        bar_lbl = [[f"{x:0.2f}" for x in bb] for bb in bar_lbl]
    
    ## set axis ticks.
    if data[y].max() <= 1:
        yticks = np.arange(0, 1.2, 0.2)
    else:
        yticks = np.linspace(0, np.round(data[y].max() * 1.2), num = 7)
    
    # ylabels = yticks.round(2).tolist()[:-1] + [""]
    ylabels = yticks.round(2).tolist()
        
    ## set legend.
    # lgnd_loc  = [0.8, 0.84]
    lgnd_loc  = "lower left"
    lgnd_bbox = (1.0, 0.5, 0.6, 0.6)
    lgnd_font = {pn.replace("font", ""): pv \
                 for pn, pv in fontdict["label"].items()}
    
    
    ## build plot.
    sns.barplot(data = data, x = x, y = y, hue = hue, orient = "v", 
                width = width, palette = col_palette, saturation = 0.8, 
                dodge = True, ax = ax, **bar_prop);                            # nested barplots
    [ax.bar_label(ax.containers[nn], labels = bar_lbl[nn], padding = 0.2, 
                  rotation = 0, **fontdict["label"]) for nn in range(n_bars)];
    sns.despine(ax = ax, offset = offset, trim = trim);                        # keeping axes lines only
    
    ## add baseline.
    if bline & (yrefs is not None):
        [ax.axhline(y = yrefs[nn], xmin = 0, xmax = data.shape[0] / n_bars, 
                    label = "baseline", **line_prop) for nn in range(n_bars)];
        
    ## format plot ticks & labels.
    ax.set_title(title, y = 1.02, **fontdict["super"]);
    ax.set_xlabel(None);     ax.set_ylabel(ylabel, **fontdict["label"]);
    ax.set_yticks(ticks = yticks, labels = ylabels, **fontdict["label"]);
    ax.set_xticks(range(len(xlabels)));
    if xrot != 0:
        ax.set_xticklabels(xlabels, rotation = xrot, rotation_mode = "anchor", 
                           ha = "right", va = "center", ma = "center", 
                           position = (0, -0.02), **fontdict["label"]);
    else:
        ax.set_xticklabels(xlabels, ma = "center", position = (0, -0.02), 
                           **fontdict["label"]);
    
    if legend:
        lgnd_ttl = {pn.replace("font", ""): pv \
                    for pn, pv in fontdict["title"].items()}
        lgnd = ax.legend(loc = lgnd_loc, bbox_to_anchor = lgnd_bbox, 
                         prop = lgnd_font, title = legend_title, 
                         title_fontproperties = lgnd_ttl, frameon = False);
        lgnd.get_title().set_multialignment("center");
        [ptch.set(**bar_prop) for ptch in lgnd.get_patches()];
    else:
        ax.legend([ ], [ ], frameon = False)
    
    return ax


## make ROC / PR curves for multiple models in the same plot.
def make_performance_plot(y_true, y_preds, curve, ax, fontdict):
    ## plot parameters.
    colors    = ["tab:purple", "tab:olive", "tab:pink"]                        # currently set for three plots
    lstyles   = ["--", "-.", ":"]
    lgnd_loc  = {"ROC": "lower right", "PR": "lower left"}
    lgnd_font = {
        pn.replace("font", ""): pv for pn, pv in fontdict["label"].items()}
    
    lw = [6, 4]
    
    ## make plots.
    if curve.upper() == "ROC":
        for nn_, (mdl_, y_pred_) in enumerate(y_preds.items()):
            RocCurveDisplay.from_predictions(
                y_true, y_pred_, ax = ax, name = mdl_, color = colors[nn_], 
                linestyle = lstyles[nn_], linewidth = lw[0]);
        
        ax.axline([0, 0], [1, 1], color = "lightgray", linestyle = ":",        # baseline AUC at 0.5
                  linewidth = lw[1]);
        # , label = "Baseline (AUC = 0.50)");
        
        ax.set_xlabel("1 $-$ Specificity", **fontdict["label"]);
        ax.set_ylabel("Sensitivity", **fontdict["label"]);
        ax.set_title("ROC curve", **fontdict["title"]);
        
    elif curve.upper() == "PR":
        for nn_, (mdl_, y_pred_) in enumerate(y_preds.items()):
            PrecisionRecallDisplay.from_predictions(
                y_true, y_pred_, ax = ax, name = mdl_, color = colors[nn_], 
                linestyle = lstyles[nn_], linewidth = lw[0]);
        
        ap_base = y_true.mean()                                                # baseline AP (fraction of class 1)
        ax.axhline(y = ap_base, xmin = 0, xmax = 1, color = "lightgray", 
                   linestyle = ":", linewidth = lw[1]); 
                   # label = f"Baseline (AP = {ap_base:0.2f})");
        
        ax.set_xlabel("Recall", **fontdict["label"]);
        ax.set_ylabel("Precision", **fontdict["label"]);
        ax.set_title("Precision-recall curve", **fontdict["title"]);
    
    ## finalize parameters.
    ax.set_xticks(np.arange(0, 1.1, 0.1), **fontdict["label"]);
    ax.set_yticks(np.arange(0, 1.1, 0.1), **fontdict["label"]);
    ax.tick_params(axis = "both", which = "major", 
                   labelsize = fontdict["label"]["fontsize"]);
    ax.legend(loc = lgnd_loc[curve.upper()], prop = lgnd_font, title = None);
    
    return ax


## make nested barplots for multiple variables.
def make_boxplot2(data, x, y, hue, hue_order, ax, fontdict, title, xlabels, 
                  xrot = 0, colors = None, ylabel = None, width = 0.8, 
                  legend = True, legend_title = None, offset = 8, trim = False):
    
    ## parameters.
    ## set colors.
    col_palette = dict(zip(data[hue].unique(), 
                           # ["#D0759F", "#88BECD"]))
                           # ["#75D0A6", "#CD9788"]))
                           # ["#75D0A6", "#88BECD"]))
                           # ["#7D26CD", "#76CD26"]))
                           # ["#88BECD", "#CD9788"]))
                           # ["#75D0A6", "#D0759F"]))
                           # ["#75D0A6", "#FFD700"]))
                           # ["#D0759F", "#88BECD"]))
                           # ["#88BECD", "#D0759F"]))
                           # ["#75D0C1", "#FFB400"]))
                           # ["#75B2D0", "#D075B2"]))
                           # ["#75D0B0", "#FFD788"]))
                           # ["#78D0CD", "#EEB9DB"]))
                           # ["#8ADBFF", "#FFB9DB"]))
                           ## ["#75D0B0", "#FFC72C"]))
                           ["#E08DAC", "#F6CF6D"]))
    
    if colors is not None:
        col_palette = dict(zip(col_palette.keys(), colors))
    
        
    ## set lines/edges.
    box_prop   = {"linestyle": "-", "linewidth": 4, "edgecolor": "#000000"}
    whis_prop  = {"linestyle": "-", "linewidth": 4, "color": "#000000"}
    med_prop   = {"linestyle": "-", "linewidth": 4, "color": "#000000"}
    cap_prop   = {"linestyle": "-", "linewidth": 4, "color": "#000000"}
    flier_prop = {"marker": "o", "markersize": 12, 
                  "markeredgecolor": "#000000", "markerfacecolor": "#000000"}
    
    
    ## set axis ticks.
    yticks  = np.arange(0, 1.2, 0.2)
    # ylabels = yticks.round(2).tolist()[:-1] + [""]
    ylabels = yticks.round(2).tolist()
    
    ## set legend.
    # lgnd_loc  = [0.8, 0.84]
    lgnd_loc  = "lower left"
    lgnd_bbox = (1.0, 0.5, 0.6, 0.6)
    lgnd_font = {pn.replace("font", ""): pv \
                 for pn, pv in fontdict["label"].items()}
    
    
    ## build plot.
    sns.boxplot(data = data, x = x, y = y, hue = hue, hue_order = hue_order, 
                orient = "v", width = width, palette = col_palette, 
                saturation = 0.9, dodge = True, whis = 1.5, notch = False, 
                boxprops = box_prop, whiskerprops = whis_prop, 
                medianprops = med_prop, capprops = cap_prop, 
                flierprops = flier_prop, ax = ax);                             # nested boxplots
    sns.despine(ax = ax, offset = 2, trim = trim);                             # keeping axes lines only
    
    
    ## format plot ticks & labels.
    ax.set_title(title, y = 1.02, **fontdict["super"]);
    ax.set_xlabel(None);     ax.set_ylabel(ylabel, **fontdict["label"]);
    ax.set_xlim([0, data[x].nunique()]);
    ax.set_ylim([yticks.min() - 0.1, yticks.max()]);
    ax.set_yticks(ticks = yticks, labels = ylabels, **fontdict["label"]);
    if xrot != 0:
        ax.set_xticklabels(xlabels, rotation = xrot, rotation_mode = "anchor",
                           ha = "right", va = "center", ma = "center", 
                           position = (0, 0), **fontdict["label"]);
    else:
        ax.set_xticklabels(xlabels, ma = "center", position = (0, -0.02), 
                           **fontdict["label"]);
    
    if legend:
        lgnd_ttl = {pn.replace("font", ""): pv \
                    for pn, pv in fontdict["title"].items()}
        lgnd = ax.legend(loc = lgnd_loc, bbox_to_anchor = lgnd_bbox, 
                         prop = lgnd_font, title = legend_title, 
                         title_fontproperties = lgnd_ttl, frameon = False);
        lgnd.get_title().set_multialignment("center");
        [ptch.set(**box_prop) for ptch in lgnd.get_patches()];
    else:
        ax.legend([ ], [ ], frameon = False)
    
    return ax


## make barplots for a single variable.
def make_boxplot1(data, x, y, ax, fontdict, title, xlabels, width = 0.8, 
                  ylabel = None, x_order = None):
    ## parameters.
    # col_palette = "tab10"
    col_palette = dict(zip(data[x].unique(), 
                           # ["#D0759F", "#88BECD"]))
                           # ["#75D0A6", "#CD9788"]))
                           # ["#75D0A6", "#88BECD"]))
                           # ["#7D26CD", "#76CD26"]))
                           # ["#88BECD", "#CD9788"]))
                           # ["#75D0A6", "#D0759F"]))
                           # ["#75D0A6", "#FFD700"]))
                           # ["#78D0CD", "#EEB9DB"]))
                           ["#75D0B0", "#FFC72C"]))
    
    yticks  = np.arange(0, 1.4, 0.2)
    ylabels = yticks.round(2).tolist()[:-1] + [""]
    
    # lgnd_loc  = [0.78, 0.88]
    # lgnd_loc  = "best"
    # lgnd_font = {pn.replace("font", ""): pv \
    #              for pn, pv in fontdict["label"].items()}
    
    ## build plot.
    mprop = {"marker": "o", "markersize": 10, "markeredgecolor": [0.9]*3}
    sns.boxplot(data = data, x = x, y = y, order = x_order, orient = "v", 
                palette = col_palette, saturation = 0.9, width = width, 
                boxprops = {"edgecolor": [0.3]*3}, flierprops = mprop, 
                dodge = True, whis = 1.5, notch = False, ax = ax);
    
    # ## add p-values.
    # pairs = list(combinations(data[x].unique(), r = 2))
    # ann = Annotator(ax = ax, pairs = pairs, data = data, x = x, y = y, 
    #                 order = x_order)
    # ann.configure(test = "Mann-Whitney", text_format = "simple", 
    #               show_test_name = False, loc = "outside")
    # ann.apply_and_annotate()
    
    ## format plot ticks & labels.
    ax.set_title(title, **fontdict["title"]);
    ax.set_xlabel(None);     ax.set_ylabel(ylabel, **fontdict["label"]);
    ax.set_yticks(ticks = yticks, labels = ylabels, **fontdict["label"]);
    ax.set_xticklabels(xlabels, rotation = 0, **fontdict["label"]);
    # ax.legend(loc = lgnd_loc, prop = lgnd_font, title = None);
    
    return ax


def add_stat(ax, data, x, y, fontdict, test = "wilcox", pval = True, 
             xpos = [0, 1]):
    ## get significance level by asterisks.
    def sig_ast(pval):
        if pval <= 0.001:       return "***"
        elif pval <= 0.01:      return "**"
        elif pval <= 0.05:      return "*"
        else:                   return "ns"
    
    ## get statistical significance for R vs. NR (1 vs. 0).
    if test.lower() == "wilcox":                                               # wilcoxon rank-sum test
        sdat = data.groupby(x).apply(
            lambda df: df[y].tolist()).sort_index(ascending = False).values
        stat = mannwhitneyu(*sdat, alternative = "greater", method = "auto")
        if pval:
            stxt = f"Wilcoxon $p$ = {stat.pvalue:0.02g}"
        else:
            stxt = sig_ast(stat.pvalue)
    elif test.lower() == "t":                                                  # t-test
        sdat = data.groupby(x).apply(
            lambda df: df[y].apply(["mean", "std", "size"])).sort_index(
                ascedending = False).values.ravel()
        stat = ttest_ind_from_stats(*sdat, alternative = "greater", 
                                    equal_var = False)
        if pval:
            stxt = f"t-test $p$ = {stat.pvalue:0.02g}"
        else:
            stxt = sig_ast(stat.pvalue)
    
    ## get plot parameters.        
    yticks  = ax.get_yticks()
    ylabels = ax.get_yticklabels()
    
    ## add p-value to plot.
    px = np.array(xpos)
    py = np.round(ax.dataLim.ymax + np.array([0.05, 0.1]), 2)
    ax.plot([px[0], px[0], px[1], px[1]], [py[0], py[1], py[1], py[0]],        # plot bounding lines
            linewidth = 2, color = "black");
    
    ax.text(x = px.mean(), y = py.max(), s = stxt, ha = "center",              # print p-value
            va = "bottom", color = "black", **fontdict["label"]);
    
    ## restore yticks & fix ylim.
    ax.set_yticks(ticks = yticks, labels = ylabels, **fontdict["label"]);
    ax.set_ylim([-0.05, ax.get_ylim()[1]]);
    
    return ax
    

####
## make nested barplots for multiple variables.
def make_barplot3_base(data, x, y, hue, ax, fontdict, title, xlabels, 
                       xrot = 0, bar_labels = True, ylabel = None, 
                       width = 0.8, bline = False, yrefs = None, 
                       legend = True, legend_title = None, colors = None,                        
                       skipcolor = False):
    ## parameters.
    # col_palette = "tab10"
    if data[hue].nunique() > 3:
        # col_palette = "CMRmap"
        # col_palette = "crest"
        col_palette = "tab20_r"
    else:
        if colors is None:
            col_list = ["#75D0B0", "#FFC72C", "#88BECD"]                           # ["green", "yellow", "blue"]-ish
        else:
            col_list = colors
        
        if skipcolor:
            col_list = col_list[1:] + [col_list[0]]
        
        col_palette = dict(zip(data[hue].unique(), col_list))
    
    col_bline  = "#000000"
    bline_prop = {"linestyle": "--", "linewidth": 1}
    
    n_bars = data[hue].nunique()
    if bar_labels:    
        bar_lbl = [data[data[hue].eq(bb)][y].round(2) \
                   for bb in data[hue].unique()]                               # bar heights as labels    
    
    if data[y].max() <= 1:
        yticks = np.arange(0, 1.4, 0.2)
    else:
        yticks = np.linspace(
            0, np.round(data[y].max() * 1.2), num = 7).round(1)
        
    ylabels = yticks.round(2).tolist()[:-1] + [""]
    
    # lgnd_loc = [0.8, 0.84]
    # lgnd_loc = "best"
    lgnd_loc  = "lower left"
    lgnd_bbox = (1.0, 0.5, 0.6, 0.6)
    lgnd_font = {pn.replace("font", ""): pv \
                 for pn, pv in fontdict["label"].items()}
    
    ## build plot.
    sns.barplot(data = data, x = x, y = y, hue = hue, orient = "v", 
                width = width, palette = col_palette, edgecolor = "#000000", 
                saturation = 0.8, dodge = True, ax = ax);                      # nested barplots
    if bar_labels:
        [ax.bar_label(ax.containers[nn], labels = bar_lbl[nn], padding = 0.2, 
                      rotation = 0, **fontdict["label"]) \
         for nn in range(n_bars)];
    
    ## add baseline.
    if bline & (yrefs is not None):
        [ax.axhline(y = yrefs[nn], xmin = 0, xmax = data.shape[0] / n_bars, 
                    color = col_bline, **bline_prop) \
         for nn in range(n_bars)];
        
    ## format plot ticks & labels.
    ax.set_title(title, y = 1.02, **fontdict["super"]);
    ax.set_xlabel(None);     ax.set_ylabel(ylabel, **fontdict["label"]);
    # ax.set_ylim([yticks.min() - 0.1, yticks.max()]);
    ax.set_yticks(ticks = yticks, labels = ylabels, **fontdict["label"]);
    ax.set_xticks(range(len(xlabels)));
    if xrot != 0:
        ax.set_xticklabels(xlabels, rotation = xrot, rotation_mode = "anchor", 
                           ha = "right", va = "center", ma = "center", 
                           position = (0, -0.02), **fontdict["label"]);
    else:
        ax.set_xticklabels(xlabels, ma = "center", position = (0, -0.02), 
                           **fontdict["label"]);
    
    if legend:
        lgnd_ttl = {pn.replace("font", ""): pv \
                    for pn, pv in fontdict["title"].items()}
        # lgnd_ttl.update({"multialignment": "center"})
        lgnd = ax.legend(loc = lgnd_loc, bbox_to_anchor = lgnd_bbox, 
                  prop = lgnd_font, title = legend_title, 
                  title_fontproperties = lgnd_ttl, frameon = True);
        lgnd.get_title().set_multialignment("center");
    else:
        ax.legend([ ], [ ], frameon = False)
    
    return ax


## make nested barplots with breaks for multiple variables.
def make_barplot3(data, x, y, hue, ax, fontdict, title, xlabels, xrot = 0, 
                  bar_labels = False, ylabel = None, width = 0.8, 
                  bline = False, yrefs = None, legend = True, 
                  legend_title = None, colors = None, skipcolor = False):
    
    if np.isinf(data[y].max()):                                                # make nested barplots with breaks
        max_val  = 200
        data_brk = data.replace({y: {np.inf: max_val}})                        # replace infinity with a dummy value
        
        ax_t, ax_b = ax
        ax_t_b, ax_b_t = 60, 45
        
        ax_t = make_barplot3_base(
            data = data_brk, x = x, y = y, hue = hue, width = width, 
            title = title, bar_labels = bar_labels, xlabels = xlabels, 
            xrot = xrot, ylabel = ylabel, bline = bline, yrefs = yrefs, 
            legend = False, legend_title = legend_title, colors = colors, 
            skipcolor = skipcolor, ax = ax_t, fontdict = fontdict)
        ax_t.set_xlim([-0.5, len(xlabels) - 0.5]);
        ax_t.set_ylim([ax_t_b, np.round(max_val * 1.05)]);
        ax_t.spines["bottom"].set_visible(False);
        ax_t.set_yticks(ticks = np.linspace(80, max_val, num = 3), 
                        labels = [100, 1000, "$\infty$"], **fontdict["label"]);
        ax_t.tick_params(axis = "x", which = "both", bottom = False, 
                         labelbottom = False);
        
        # data_brk.loc[data_brk[y].gt(ax_b_t), y] = ax_b_t                       # hack for new py version (?)
        ax_b = make_barplot3_base(
            data = data_brk, x = x, y = y, hue = hue, width = width, 
            title = None, bar_labels = bar_labels, xlabels = xlabels, 
            xrot = xrot, ylabel = ylabel, bline = bline, yrefs = yrefs, 
            legend = legend, legend_title = legend_title, colors = colors, 
            skipcolor = skipcolor, ax = ax_b, fontdict = fontdict)
        ax_b.set_xlim([-0.5, len(xlabels) - 0.5]);
        ax_b.set_ylim([0, ax_b_t]);     ax_b.spines["top"].set_visible(False);
        ax_b.set_yticks(ticks = [0, 20, 40], labels = [0, 20, 40], 
                        **fontdict["label"]);
        
        ## add breakpoint indicators.
        d_brk = 0.005                                                          # axis breakpoint tick size
        brkargs = dict(transform = ax_t.transAxes, color = "#000000", 
                       clip_on = False)
        ax_t.plot((-d_brk, +d_brk), (-d_brk, +d_brk), **brkargs)               # add breakpoint marker tick for top plot
        # ax_t.plot((1 - d_brk, 1 + d_brk), (-d_brk, +d_brk), **brkargs)
        
        brkargs.update(transform = ax_b.transAxes, color = "#000000", 
                       clip_on = False)
        ax_b.plot((-d_brk, +d_brk), (1 - d_brk, 1 + d_brk), **brkargs)         # add breakpoint marker tick for bottom plot
        # ax_b.plot((1 - d_brk, 1 + d_brk), (1 - d_brk, 1 + d_brk), **brkargs)
        
        if legend:
            ax_b.get_legend().set(bbox_to_anchor = (1.0, 0.0, 0.6, 0.6))       # set legend location
        
        ax = ax_t, ax_b
        
    else:
        ax = make_barplot3_base(
            data = data, x = x, y = y, hue = hue, width = width, title = title, 
            bar_labels = bar_labels, xlabels = xlabels, xrot = xrot, 
            ylabel = ylabel, bline = bline, yrefs = yrefs, legend = legend, 
            legend_title = legend_title, skipcolor = skipcolor, ax = ax, 
            fontdict = fontdict)
        
    return ax

