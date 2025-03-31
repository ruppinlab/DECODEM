#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:18:43 2024

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
from miscellaneous import list_files, date_time, tic, write_xlsx
# from itertools import product
from tqdm import tqdm


#%% functions.

def read_pkl_data(data_file):
    with open(data_file, "rb") as file:
        data_obj   = pickle.load(file)
        exp_all_sc = {ctp_: pd.DataFrame(exp_).set_index("Cell.id") 
                      for ctp_, exp_ in data_obj["exp"].items()}
        resp_E     = pd.DataFrame(data_obj["resp"]).set_index("Cell.id")
        cell_ids   = {ctp_: pd.DataFrame(cells_) 
                      for ctp_, cells_ in data_obj["cells"].items()}
        clin_info  = pd.DataFrame(data_obj["clin"]).set_index("Cell.id")
        del data_obj
        
    return exp_all_sc, resp_E, cell_ids, clin_info


def subset_data(exp_all, resp, cells, clin, samples):
    clin_sub = clin.pipe(
        lambda df: df[df["Sample.id"].map(lambda x: x in samples)])
    
    resp_sub = resp.pipe(
        lambda df: df[df["Sample.id"].map(lambda x: x in samples)])
    
    cells_sub, exp_all_sub = { }, { }
    for ctp_, cells_ in cells.items():
        cells_sub[ctp_]   = cells_.pipe(
            lambda df: df[df["Sample.id"].map(lambda x: x in samples)])
        exp_all_sub[ctp_] = exp_all[ctp_][cells_sub[ctp_]["Cell.id"]]
    del ctp_, cells_
    
    if "PseudoBulk" in exp_all.keys():
        exp_all_sub["PseudoBulk"] = exp_all["PseudoBulk"][samples]
    
    return exp_all_sub, resp_sub, cells_sub, clin_sub


def get_resp_ratio(resp):
    resp_smpl = resp.drop_duplicates(subset = "Sample.id")
    return {"R" : resp_smpl.Response.eq(1).sum(), 
            "NR": resp_smpl.Response.eq(0).sum()}
    

#%% read data.

data_path = "../../data/SC_data/BassezEtAl2021/validation/"
data_file = "bc_sc_data_bassez2021_all.pkl"

exp_all_sc, resp_E, cell_ids, clin_info = read_pkl_data(
    data_path + data_file)

resp_E.sort_values(by = ["Patient.id", "Sample.id"], ascending = False, 
                   inplace = True)

samples    = sorted(clin_info["Sample.id"].unique())                           # all samples regardless of subtype
cell_types = sorted(cell_ids.keys())


## generate pseudobulk data per sample.
exp_all_sc["PseudoBulk"] = pd.DataFrame({
    smpl: pd.concat([exp_all_sc[ctp_].pipe(
        lambda df: df.loc[:, df.columns.map(lambda x: smpl in x)]) 
    for ctp_ in cell_types], axis = 1).mean(axis = 1) 
    for smpl in tqdm(samples)})


#%% subset data based on subtype.

samples_sbtyp = clin_info.groupby(
    by = "subtype").apply(
    lambda df: df["Sample.id"].unique().tolist())

print(f"available samples by subtype = { {ctp_: len(smpl_) for ctp_, smpl_ in samples_sbtyp.items()} }")

exp_all_sc_her2, resp_E_her2, cell_ids_her2, clin_info_her2 = subset_data(
    exp_all_sc, resp_E, cell_ids, clin_info, samples = samples_sbtyp["HER2+"])

exp_all_sc_er, resp_E_er, cell_ids_er, clin_info_er = subset_data(
    exp_all_sc, resp_E, cell_ids, clin_info, samples = samples_sbtyp["ER+"])

exp_all_sc_tnbc, resp_E_tnbc, cell_ids_tnbc, clin_info_tnbc = subset_data(
    exp_all_sc, resp_E, cell_ids, clin_info, samples = samples_sbtyp["TNBC"])

print(f"""
subsetted data based on subtype! 
R:NR ratios: 
{ pd.DataFrame(map(get_resp_ratio, [resp_E_her2, resp_E_er, resp_E_tnbc]), 
             index = samples_sbtyp.keys()) }
""")


#%% save data.

svdat = True

if svdat:
    out_path = "../../data/SC_data/BassezEtAl2021/validation/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if doesn't exist
    
    print("saving data by subtype...")
    
    ## HER2+.
    out_file = "bc_sc_data_bassez2021_her2.pkl"
    out_data = {"exp"  : exp_all_sc_her2, "resp": resp_E_her2, 
                "cells": cell_ids_her2,   "clin": clin_info_her2}
    with open(out_path + out_file, mode = "wb") as file:
        pickle.dump(out_data, file)
    print(out_file)
    
    ## ER+.
    out_file = "bc_sc_data_bassez2021_er.pkl"
    out_data = {"exp"  : exp_all_sc_er, "resp": resp_E_er, 
                "cells": cell_ids_er,   "clin": clin_info_er}
    with open(out_path + out_file, mode = "wb") as file:
        pickle.dump(out_data, file)
    print(out_file)
    
    ## TNBC.
    out_file = "bc_sc_data_bassez2021_tnbc.pkl"
    out_data = {"exp"  : exp_all_sc_tnbc, "resp": resp_E_tnbc, 
                "cells": cell_ids_tnbc,   "clin": clin_info_tnbc}
    with open(out_path + out_file, mode = "wb") as file:
        pickle.dump(out_data, file)
    print(out_file)
    
    print("done!")
