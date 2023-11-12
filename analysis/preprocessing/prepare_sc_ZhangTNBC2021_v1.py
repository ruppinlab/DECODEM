#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:08:35 2023

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
from miscellaneous import list_files, tic
from tqdm import tqdm


#%% functions.

def read_data(file, sheet = None, sep = "\t", **kwarg):
    try:
        data = pd.read_excel(file, sheet_name = sheet, header = 0, 
                             index_col = None, **kwarg)
    except:
        data = pd.read_table(file, sep = sep, header = 0, index_col = 0, 
                             **kwarg)
    
    return data


## isolate cells per cell type per patient/sample.
def get_cell_ids(clin, keep_ = None):
    if keep_ is None:
        keep_ = ["Sample.id", "Patient.id"]
    
    cell_ids = {ctp_: clin[clin["Cell.type"] == ctp_][keep_].reset_index() \
                for ctp_ in clin["Cell.type"].unique()}
    
    return cell_ids


## compute pseudo-bulk expression across all available cell types.
def get_pseudo_bulk(exp_sc, sample_list):
    exp_pblk = { }
    for smpl_ in tqdm(sample_list):
        exp_ctps_ = pd.concat([
            exp_.iloc[:, exp_.columns.map(lambda x: smpl_ in x)] \
                for exp_ in exp_sc.values()], axis = 1)
        exp_pblk[smpl_] = exp_ctps_.mean(axis = 1)
    
    exp_pblk = pd.DataFrame(exp_pblk)
    
    return exp_pblk
    

#%% read data.

data_path  = "../data/SC_data/ZhangTNBC2021/"
data_files = ["TNBC_scAnnot_tissue_pre_ZhangEtAl2021.xlsx"] + \
    list_files(data_path, pattern = "Chemo") + \
        list_files(data_path, pattern = "Anti-PD-L1_Chemo")


## available cell types + treatments.
cell_types = ["B-cells", "ILC", "Myeloid", "T-cells"]
treatments = ["Chemo", "Anti-PD-L1_Chemo"]

## get data.
clin_info = read_data(data_path + data_files[0], sheet = None)

_tic = tic()
exp_sc_chemo, exp_sc_chemo_immuno = { }, { }
for ctp_ in tqdm(cell_types):
    data_files += [
        f"TNBC_scExp_tissue_pre_Chemo_ZhangEtAl2021_{ctp_}.tsv", 
        f"TNBC_scExp_tissue_pre_Anti-PD-L1_Chemo_ZhangEtAl2021_{ctp_}.tsv"]
        
    exp_sc_chemo[ctp_]        = read_data(data_path + data_files[-2])
    exp_sc_chemo_immuno[ctp_] = read_data(data_path + data_files[-1])

del ctp_
_tic.toc()


#%% prepare response & pseudobulk expression.

## chemo.
clin_info_cm = clin_info["Chemo"].dropna(
    subset = "Efficacy", how = "any").set_index("Cell.id")
clin_info_cm["Response"] = clin_info_cm["Efficacy"].eq("PR").astype(int)       # efficacy is in RECIST scale (no CR) => PR == R
cell_ids_cm  = get_cell_ids(clin_info_cm)
resp_PR_cm   = clin_info_cm[["Patient.id", "Sample.id", "Response"]].copy()
exp_sc_cm    = {ctp_: exp_[cell_ids_cm[ctp_]["Cell.id"]] \
                for ctp_, exp_ in exp_sc_chemo.items()}
exp_sc_cm["PseudoBulk"] = get_pseudo_bulk(
    exp_sc_cm, sample_list = clin_info_cm["Sample.id"].unique())

data_cm     = {"exp": exp_sc_cm, "resp": resp_PR_cm, "cells": cell_ids_cm, 
               "clin": clin_info_cm}


## chemo + immuno.
clin_info_im = clin_info["Anti-PD-L1_Chemo"].dropna(
    subset = "Efficacy", how = "any").set_index("Cell.id")
clin_info_im["Response"] = clin_info_im["Efficacy"].eq("PR").astype(int)       # efficacy is in RECIST scale (no CR) => PR == R
cell_ids_im  = get_cell_ids(clin_info_im)
resp_PR_im   = clin_info_im[["Patient.id", "Sample.id", "Response"]].copy()
exp_sc_im    = {ctp_: exp_[cell_ids_im[ctp_]["Cell.id"]] \
                for ctp_, exp_ in exp_sc_chemo_immuno.items()}
exp_sc_im["PseudoBulk"] = get_pseudo_bulk(
    exp_sc_im, sample_list = clin_info_im["Sample.id"].unique())

data_im      = {"exp": exp_sc_im, "resp": resp_PR_im, "cells": cell_ids_im, 
                "clin": clin_info_im}


## all.
clin_info_all = pd.concat([clin_info_cm, clin_info_im], axis = 0)
cell_ids_all  = {ctp_: pd.concat([cell_ids_cm[ctp_], cell_ids_im[ctp_]], 
                                 axis = 0) for ctp_ in tqdm(cell_types)}
resp_PR_all   = clin_info_all[["Patient.id", "Sample.id", "Response"]].copy()
exp_sc_all    = {ctp_: pd.concat([exp_sc_cm[ctp_], exp_sc_im[ctp_]], axis = 1) \
                 for ctp_ in tqdm(np.append(cell_types, "PseudoBulk"))}

data_all      = {"exp": exp_sc_all, "resp": resp_PR_all, "cells": cell_ids_all, 
                 "clin": clin_info_all}


#%% save data.

svdat = False                                                                  # set True to save data 

if svdat:
    out_path = "../data/SC_data/ZhangTNBC2021/validation/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if doesn't exist
    
    ## save chemo data.
    print("\nsaving data for Chemo...", end = " ")
    out_file = "tnbc_sc_data_chemo_v2.pkl"
    with open(out_path + out_file, "wb") as file:
        pickle.dump(data_cm, file)
    print(f"done!\ndata is in = '{out_path + out_file}'")
    
    ## save chemo + immuno data.
    print("\nsaving data for Chemo + Immuno...", end = " ")
    out_file = "tnbc_sc_data_chemo_immuno_v2.pkl"
    with open(out_path + out_file, "wb") as file:
        pickle.dump(data_im, file)
    print(f"done!\ndata is in = '{out_path + out_file}'")
    
    ## save both chemo & chemo + immuno data.
    print("\nsaving data for both treatments...", end = " ")
    out_file = "tnbc_sc_data_all_v2.pkl"
    with open(out_path + out_file, "wb") as file:
        pickle.dump(data_all, file)
    print(f"done!\ndata is in = '{out_path + out_file}'")

