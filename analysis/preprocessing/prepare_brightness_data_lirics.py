#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:37:56 2023

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
from functools import reduce
from operator import add


#%% functions.

def read_data(file, sheet = None, sep = "\t", **kwarg):
    try:
        data = pd.read_excel(
            file, sheet_name = sheet, header = 0, index_col = 0, **kwarg)
    except:
        data = pd.read_table(
            file, sep = sep, header = 0, index_col = 0, **kwarg)
        data.columns = list(map(lambda s: s.replace(" ", "_"), data.columns))
    
    return data

def make_cclr_list(cci_data, axis = 0):                                        # extracts ligand/receptor data as dataframe
    cclr_lst = cci_data.index if (axis == 0) else cci_data.columns
    cclr_lst = [cclr.replace("_Epithelial", "-Epithelial").split("_") \
                for cclr in cci_data.index]
    cclr_lst = pd.DataFrame(cclr_lst, index = cci_data.index, 
                            columns = ["LigandCell", "ReceptorCell", 
                                       "LigandGene", "ReceptorGene"])
    cclr_lst.replace(regex = {"-Epithelial": "_Epithelial"}, inplace = True)
    
    return cclr_lst


#%% read data.

data_path  = ["../data/BrighTNess/", 
              "../data/BrighTNess/out_lirics_brightness/", 
              "../data/BrighTNess/out_codefacs_brightness_v2/"]

data_files = ["GSE164458_BrighTNess_clinical_info_SRD_04Oct2022.xlsx", 
              "lirics_BRCA_brightness_SS_28Feb2023.xlsx", 
              "confidence_score.txt", 
              "estimated_cell_fractions.txt"] 

clin_info = read_data(data_path[0] + data_files[0], sheet = "samples")

cci_info  = read_data(data_path[1] + data_files[1], verbose = 1)

conf_score = read_data(data_path[2] + data_files[2])
cell_frac  = read_data(data_path[2] + data_files[3])


#%% prepare data.

## trial arm description from Loibi et al. 2018 [PMID: 29501363]:
## PBO = placebo, PARPi = PARP inhibitor := veliparib, taxane := paclitaxel
## arm A: TCV => patients receiving taxane + carboplatin + PARPi.
## arm B: TC  => patients receiving taxane + carboplatin + PARPi PBO.
## arm C: T   => patients receiving taxane + carboplatin PBO + PARPi PBO.

## filter patients by surgery (available RCB index).
samples_all = clin_info[
    clin_info["surgery"].eq(1) & \
    clin_info["residual_cancer_burden_class"].notna()].index.tolist()

clin_data = clin_info.loc[samples_all].copy()
resp_pCR  = (clin_data["pathologic_complete_response"] == "pCR").astype(int)


## prepare cci lists.
cci_data = {ds_: cci_[cci_.sum(axis = 1) > 0][samples_all] \
            for ds_, cci_ in cci_info.items()}
cclr_list = {ds_: make_cclr_list(cci_) for ds_, cci_ in cci_data.items()}


## prepare other data.
genes_all = np.union1d(*[np.union1d(
    reduce(add, cclr_.LigandGene.apply(lambda s: s.split(";"))), 
    reduce(add, cclr_.ReceptorGene.apply(lambda s: s.split(";")))) \
        for cclr_ in cclr_list.values()])                                      # all genes with significant CCIs

conf_cci, frac_cci = conf_score.loc[genes_all], cell_frac.loc[samples_all]


#%% separate data by treatment.

samples_ct = clin_data[clin_data["planned_arm_code"] == "A"].index.tolist()    # chemo + targeted (PARPi)
samples_cm = np.setdiff1d(samples_all, samples_ct).tolist()                    # chemo

data_all = {"cci": cci_data, "cclr": cclr_list, "resp": resp_pCR, 
            "conf": conf_cci, "frac": frac_cci, "clin": clin_data}

data_ct = {"cci": {ds_: cci_[samples_ct] for ds_, cci_ in cci_data.items()}, 
           "cclr": cclr_list, "resp": resp_pCR.loc[samples_ct], 
           "conf": conf_cci, "frac": frac_cci.loc[samples_ct], 
           "clin": clin_data.loc[samples_ct]}

data_cm = {"cci": {ds_: cci_[samples_cm] for ds_, cci_ in cci_data.items()}, 
           "cclr": cclr_list, "resp": resp_pCR.loc[samples_cm], 
           "conf": conf_cci, "frac": frac_cci.loc[samples_cm], 
           "clin": clin_data.loc[samples_cm]}


#%% save data.

svdat = False                                                                  # set True to save data 

if svdat:
    out_path = "../data/BrighTNess/validation/"
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist 
    
    out_file = "brightness_lirics_data_all_v2.pkl"
    print("\nsaving data for all samples...")
    with open(out_path + out_file, "wb") as file:
        pickle.dump(data_all, file)
    print(f"done. The file is in: {out_path + out_file}")
    
    out_file = "brightness_lirics_data_chemo_targeted_v2.pkl"
    print("\nsaving data for chemo+targeted samples...")
    with open(out_path + out_file, "wb") as file:
        pickle.dump(data_ct, file)
    print(f"done. The file is in: {out_path + out_file}")
    
    out_file = "brightness_lirics_data_chemo_v2.pkl"
    print("\nsaving data for chemo samples...")
    with open(out_path + out_file, "wb") as file:
        pickle.dump(data_cm, file)
    print(f"done. The file is in: {out_path + out_file}")

