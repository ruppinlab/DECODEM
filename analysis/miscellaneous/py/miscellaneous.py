#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 13:13:07 2021

@author: dhrubas2
"""

import os
import numpy as np, pandas as pd
from math import sqrt, ceil, floor, nan, inf
from scipy.stats.stats import pearsonr, spearmanr
# from collections import deque, defaultdict, Counter
# from itertools import product
from datetime import datetime
from time import time
from tqdm import tqdm


#%% write definitions.

""" list files in a directory that contains given patterns 
    path        : path to check files in
    pattern     : a str or a list of str
    ignore_case : whether to maintain letter case when matching
"""
def list_files(path = ".", pattern = None, ignore_case = False):
    # return all files if no pattern is selected.
    if pattern is None:
        filelist = os.listdir(path)
    
    # check files that contains the pattern.
    else:
        _in_ = lambda sbtxt, txt: (sbtxt.lower() in txt.lower()) if ignore_case else (sbtxt in txt)
        if isinstance(pattern, str):
            filelist = [file for file in os.listdir(path) if _in_(pattern, file)]
        elif isinstance(pattern, list):
            filelist = set([file for pat in pattern for file in os.listdir(path) if _in_(pat, file)])
        else:
            raise ValueError("List of patterns must be a str or list!")
        
    return sorted(filelist, key = lambda x: x.lower())


""" prints description for a new cell/section """
def print_desc(txt, sep = 64):
    print("\n\n{0}\n{1}".format(txt, "-"*sep))


""" converts a list of lists to one single list """
def flat_list(lst, sort = False):
    flst = [x for sblst in lst for x in sblst]
    return sorted(flst) if sort else flst
    

#%% write definitions.

""" calculate time durations - similar to matlab tic/toc functions 
    can return the duration if out = True
"""
class tic_old:
    def __init__(self):
        global _dt_
        _dt_ = time()
    
    def toc(out = False):
        if "_dt_" not in globals():
            tic()
        
        global _dt_
        _dt_ = time() - _dt_
        
        # format into hh-mm-ss.
        _dt_hms_ = _format_(_dt_)
        
        print("Elapsed time = %s" % _print_(_dt_hms_))
        if out:     return _dt_hms_
    
def _format_(_dt_):
    _hms_ = {"h": nan, "m": nan, "s": ceil(_dt_)}
    _hms_["h"], _hms_["m"] = divmod(_hms_["s"], 3600)
    _hms_["m"], _hms_["s"] = divmod(_hms_["m"], 60)
    return _hms_

def _print_(_hms_):
    txt = "{0} hr {1} min {2} sec.".format(*_hms_.values())
    if _hms_["h"] == 0:
        txt = txt.replace("0 hr ", "")
        if _hms_["m"] == 0:
            txt = txt.replace("0 min ", "")
    return txt


""" calculate time durations - similar to matlab tic/toc functions 
    can return the duration if out = True
"""
class tic:
    def __init__(self, disp = False):
        self._dt_ = time()
        if disp:
            print(f"\nstart time = {datetime.now().strftime('%d%b%Y-%H:%M:%S')}")
    
    def toc(self, disp = False, disp_dt = True, return_dt = False):
        self._dt_ = time() - self._dt_
        if disp:
            print(f"\nend time = {datetime.now().strftime('%d%b%Y-%H:%M:%S')}")
        
        if disp_dt:
            self._dt_fmt_ = {"d": 0, "h": 0, "m": 0, "s": round(self._dt_)}
            if self._dt_fmt_["s"] > 60:
                self._dt_fmt_["m"], self._dt_fmt_["s"] = divmod(
                    self._dt_fmt_["s"], 60
            )
            if self._dt_fmt_["m"] > 60:
                self._dt_fmt_["h"], self._dt_fmt_["m"] = divmod(
                    self._dt_fmt_["m"], 60
            )
            if self._dt_fmt_["h"] > 24:
                self._dt_fmt_["d"], self._dt_fmt_["h"] = divmod(
                    self._dt_fmt_["h"], 24
            )
            
            disp_diag = " ".join(
                [f"{t_val}{t_ind}" for t_ind, t_val in self._dt_fmt_.items() if t_val > 0]
            )
                
            print(f"\nelapsed time = {disp_diag}\n")
            
        if return_dt:
            return self._dt_



""" returns current date (and time) in dd-mmm-yyyy (and hh-mm am/pm) format """
def date_time(date = True, time = False):
    fmt = ["%d%b%Y", "%I%M%p"]
    if date and time:           fmt = "_".join(fmt)
    elif time and not(date):    fmt = fmt[1]
    else:                       fmt = fmt[0]
    return datetime.now().strftime(fmt)


#%% write definitions.

""" calculates square root of a given number/array and 
    generates the closest set of integers
    method : rounding method - 'ceil', 'floor', 'round', or None (truncate)
"""
def sqrt_val(data, method = "round"):
    data = np.sqrt(data)
    if method is None:
        return data.astype(int)
    
    method = method.lower()
    if method  == "ceil":
        return np.ceil(data).astype(int)
    elif method == "floor":
        return np.floor(data).astype(int)
    else:
        return np.round(data).astype(int)


""" calculates statistical range """
def the_range(data, remove_nan = False):
    data = np.asarray(data[~np.isnan(data)] if remove_nan else data)
    return data.max(axis = 0) - data.min(axis = 0)
    

""" normalized standard deviation 
    given a data vector 'x'- generates std(x) / range(x)
"""
def std_norm(data, remove_nan = False):
    data = np.asarray(data[~np.isnan(data)] if remove_nan else data)
    if np.allclose(the_range(data), 0):
        return 0
    else:
        return data.std() / the_range(data)


""" given a dataset in dataframe (or array)-alike format, normalizes each 
    column using the specificed method
    data   : given dataset
    method : if "normalize", scales in [0, 1] per feature,
             if "standardize", calculates z-score per feature
"""
def normalize_data(data, method = "normalize"):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns = map(lambda x: "var%d" % x, range(data.shape[1])))
    
    # perform operation.
    method = method.lower()
    if method == "normalize":
        param = data.apply([np.min, the_range]).values
    elif method == "standardize":
        param = data.apply([np.mean, np.std]).values
    else:
        raise ValueError("Invalid option! Please choose 'normalize' or 'standardize'.")
    
    return (data - param[0]) / param[1]


#%% write definitions.

""" evaluate prediction performance using popular metrics
    y_true : ground truth labels
    y_pred : predicted labels
    metric : a single metric or list of metrics. available metrics are 
             "MSE"/"RMSE"/"NRMSE", "MAE"/"NMAE", "PCC"/"SCC", and "R2"
"""
def performance(y_true, y_pred, metrics = ["NRMSE", "PCC"]):
    y_true, y_pred = y_true.squeeze(), y_pred.squeeze()
    if isinstance(metrics, str):    metrics = [metrics]
    
    # base functions.
    MAE = lambda y_t, y_p = None: np.abs(y_t - (y_t.mean() if (y_p is None) else y_p)).mean()
    MSE = lambda y_t, y_p = None: np.square(y_t - (y_t.mean() if (y_p is None) else y_p)).mean()    
    
    # calculate performance.
    perf = { }
    for met in metrics:
        met = met.upper()
        if "MSE" in met:                                                       # squared errors
            res = MSE(y_true, y_pred)
            if (met == "NRMSE"):    res = np.sqrt(res / MSE(y_true))
            elif (met == "RMSE"):   res = np.sqrt(res)
        elif "MAE" in met:                                                     # absolute errors
            res = MAE(y_true, y_pred)
            if met == "NMAE":       res = res / MAE(y_true)
        elif "CC" in met:                                                      # correlations
            if met == "SCC":        res = spearmanr(y_true, y_pred)[0] 
            else:                   res = pearsonr(y_true, y_pred)[0]
        elif met == "R2":                                                      # r-squared
            # res = 1 - MSE(y_true, y_pred) / MSE(y_true)
            # if res < 0:     res = 0
            res = pearsonr(y_true, y_pred)[0]**2
        else:
            raise ValueError("Unknown metric!")
        
        perf[met] = res                                                        # save value
    
    return perf


#%% write definitions.

""" write multiple datasets in different sheets of an xlsx file 
    path   : path + filename to save data in
    data   : a single dataframe or dict of dataframes
    sheets : a list of sheet names for saving the dataframes. if not given- 
             uses the dict keys 
"""
def write_xlsx(path, data, sheets = None, verbose = True, 
               header = True, index = True):
    if os.path.exists(path):    os.remove(path)                                # remove file if exists
    
    with pd.ExcelWriter(path, mode = "w", engine = "openpyxl") as xw:
        if isinstance(data, pd.DataFrame):                                     # write a single sheet
            if sheets is None:
                data.to_excel(xw, header = header, index = index)
            else:
                if isinstance(sheets, list):    sheets = sheets[0]
                data.to_excel(xw, sheet_name = sheets, header = header, index = index)
        elif isinstance(data, dict):                                           
            if sheets is None:      sheets = list(data.keys())                 # use keys as sheet names
            
            if verbose:                                                        # print progress
                for sht, df in tqdm(zip(sheets, data.values())):
                    df.to_excel(xw, sheet_name = sht, header = header, index = index)
            else:
                for sht, df in zip(sheets, data.values()):
                    df.to_excel(xw, sheet_name = sht, header = header, index = index)
    
    