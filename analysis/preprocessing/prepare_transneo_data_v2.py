#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:42:58 2022

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
from miscellaneous import list_files
from functools import reduce
from tqdm import tqdm


#%% data path & filenames. 

data_path = dict(
    bulk  = "../data/TransNEO/use_data/", 
    decon = "../data/TransNEO/CODEFACS_results/"
)
data_files = dict(
    clin  = "TransNEO_SupplementaryTablesAll.xlsx", 
    exp   = "transneo-diagnosis-RNAseq-TPM_SRD_26May2022.tsv"
)
data_files["decon"] = list_files(data_path["decon"], pattern = "expression")
data_files["conf"]  = "confidence_score.txt"
data_files["frac"]  = "estimated_cell_fractions.txt"


#%% get clinical data & available sample lists. 

clin_info = pd.read_excel(
    data_path["bulk"] + data_files["clin"], sheet_name = 0, skiprows = 1, 
    header = 0, index_col = 0
)

## filter by if BOTH RNA & clinical response is available.  
filter_by = (clin_info["RNA.sequenced"] == "YES") & \
    (~clin_info["RCB.category"].isna())


## clinical info for various patient subsets.
clin_all = clin_info[filter_by].copy()                                         # all available patients
clin_cm  = clin_all[clin_all["aHER2.cycles"].isna()]                           # chemotherapy only
clin_ct  = clin_all[~clin_all["aHER2.cycles"].isna()]                          # chemo + targeted therapy

samples_all, samples_cm, samples_ct = \
    clin_all.index.values, clin_cm.index.values, clin_ct.index.values

print(f"""sample counts (n):
      all = {samples_all.size}
      chemo only = {samples_cm.size}
      chemo + targeted = {samples_ct.size}\n""")


#%% get expression & abundance data. 

read_delim = lambda file, sep = "\t": pd.read_table(
    file, sep = sep, header = 0, index_col = 0
)
replace_space = lambda str_w_space, sep = "_": str_w_space.replace(" ", sep)

## read expression files.
exp_bulk = read_delim(data_path["bulk"] + data_files["exp"])

exp_decon = { }
for decon_file in tqdm(data_files["decon"]):
    ctp_ = decon_file.split("_")[1].split(".")[0].replace(" ", "_")            # cell type (replace space w/ underscore)
    exp_ = read_delim(data_path["decon"] + decon_file, sep = " ")
    exp_decon[ctp_] = exp_

## read other files.
conf_score = read_delim(data_path["decon"] + data_files["conf"])
conf_score.columns = map(replace_space, conf_score.columns)                    # cell type (replace space w/ underscore)
cell_frac  = read_delim(data_path["decon"] + data_files["frac"])
cell_frac.columns = map(replace_space, cell_frac.columns)                      # cell type (replace space w/ underscore)

## check cell type ordering.
if (list(exp_decon.keys()) == conf_score.columns.tolist()) and \
    (conf_score.columns.tolist() == cell_frac.columns.tolist()):
    cell_types = conf_score.columns.values
else:
    raise ValueError("cell types are not the same or not in the same order between expression, confidence score and cell fraction files!")

## put all expression data together.
exp_data = exp_decon.copy();    exp_data["Bulk"] = exp_bulk.copy()

print("\nexpression data sizes (p x n)): ")
print(pd.Series(dict(map(lambda itm: (itm[0], itm[1].shape), exp_data.items()))))
     
print(f"""\nother data sizes (n x p):
      confidence score = {conf_score.shape}
      cell fraction = {cell_frac.shape}""")


#%% separate data by sample subsets & save.

svdat = False                                                                  # set True to save data 

get_exp_smpl = lambda smpl: {ctp_: exp_[smpl] for ctp_, exp_ in exp_data.items()}
get_num_resp = lambda clin: (clin["pCR.RD"] == "pCR").astype(int)

## samples = all.
data_all = {
    "exp": get_exp_smpl(samples_all),   "resp": get_num_resp(clin_all), 
    "frac": cell_frac.loc[samples_all], "conf": conf_score,   "clin": clin_all
}

## samples = chemo.
data_cm = {
    "exp": get_exp_smpl(samples_cm),   "resp": get_num_resp(clin_cm), 
    "frac": cell_frac.loc[samples_cm], "conf": conf_score,   "clin": clin_cm
}

## samples = chemo + targeted.
data_ct = {
    "exp": get_exp_smpl(samples_ct),   "resp": get_num_resp(clin_ct), 
    "frac": cell_frac.loc[samples_ct], "conf": conf_score,   "clin": clin_ct
}


## final check for samples.
def check_samples(ds):
    exp_smpl = reduce(
        np.intersect1d, [exp_.columns for exp_ in ds["exp"].values()]
    )
    if ((exp_smpl.tolist() == ds["resp"].index.tolist()) and 
        (ds["resp"].index.tolist() == ds["frac"].index.tolist())):
        print("ok!")
    else:
        raise ValueError("samples are not the same or not in the same order between expression, response and cell fraction files!")
    
    
check_samples(data_all);    check_samples(data_cm);     check_samples(data_ct)


## save prepared data.
if svdat:
    print("\nsaving data...")
    out_path = "../data/TransNEO/transneo_analysis/"
    out_file = {"all": "transneo_data_all_v2.pkl", 
                "cm" : "transneo_data_chemo_v2.pkl", 
                "ct" : "transneo_data_chemo_targeted_v2.pkl"}
    
    os.makedirs(out_path, exist_ok = True)                                     # creates output dir if it doesn't exist 
    
    print("\tdata for all samples - ", end = "")
    with open(out_path + out_file["all"], "wb") as file:
        pickle.dump(data_all, file)
    print(f"saved in: {out_file['all']}")
    
    print("\tdata for chemo samples - ", end = "")
    with open(out_path + out_file["cm"], "wb") as file:
        pickle.dump(data_cm, file)
    print(f"saved in: {out_file['cm']}")
    
    print("\tdata for chemo + targeted samples - ", end = "")
    with open(out_path + out_file["ct"], "wb") as file:
        pickle.dump(data_ct, file)
    print(f"saved in: {out_file['ct']}")

