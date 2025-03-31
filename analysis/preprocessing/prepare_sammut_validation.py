#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:59:02 2023

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
    bulk  = "../data/TransNEO_SammutShare/", 
    decon = "../data/TransNEO_SammutShare/out_codefacs_tn_val_v2/"
)
data_files = dict(clin = "transneo-diagnosis-clinical-features.xlsx", 
                  bulk = "transneo-validation-TPM-coding-genes_v2.txt")
data_files["decon"] = list_files(data_path["decon"], pattern = "expression")
data_files["conf"]  = "confidence_score.txt"
data_files["frac"]  = "estimated_cell_fractions.txt"


#%% read all data first.

def read_delim(file, sep = "\t", skiprows = 0):
    data = pd.read_table(file, sep = sep, skiprows = skiprows, 
                         header = 0, index_col = 0)
    
    ## format column names.
    try:
        data.columns = data.columns.astype(int)
    except:
        data.columns = data.columns.astype(str)   
    
    return data

replace_space = lambda str_w_space, delim = "_": str_w_space.replace(" ", delim)


## read clinical data.
clin_info = pd.read_excel(data_path["bulk"] + data_files["clin"], 
                          sheet_name = "validation", header = 0, index_col = 0)
clin_info = clin_info[clin_info["pCR.RD"].notna()]


## read expression files.
exp_bulk = read_delim(data_path["bulk"] + data_files["bulk"])
exp_bulk.index = map(lambda ch: ch.upper(), exp_bulk.index)                    # gene symbols in upper case

exp_decon = { }
for decon_file in tqdm(data_files["decon"]):
    ctp_ = decon_file.split("_")[1].split(".")[0].replace(" ", "_")            # cell type (replace space w/ underscore)
    exp_ = read_delim(data_path["decon"] + decon_file, sep = " ")
    exp_.index = map(lambda ch: ch.upper(), exp_.index)                        # gene symbols in upper case
    exp_decon[ctp_] = exp_

## read other files.
conf_score = read_delim(data_path["decon"] + data_files["conf"])
conf_score.columns = map(replace_space, conf_score.columns)                    # replace spaces by underscores
conf_score.index = map(lambda ch: ch.upper(), conf_score.index)                # gene symbols in upper case

cell_frac = read_delim(data_path["decon"] + data_files["frac"])
cell_frac.columns = map(replace_space, cell_frac.columns)                      # replace spaces by underscores


## check cell type ordering.
if (list(exp_decon.keys()) == conf_score.columns.tolist()) and \
    (conf_score.columns.tolist() == cell_frac.columns.tolist()):
    cell_types = conf_score.columns.values
else:
    raise ValueError(
        "cell types are not the same or not in the same order between expression, confidence score and cell fraction files!"
    )

## get data for common sample set.
samples_all = reduce(
    np.intersect1d, [exp_bulk.columns, cell_frac.index, clin_info.index]
)
print(f"\nsample size to be used, n = {samples_all.size}")

clin_info = clin_info.loc[samples_all]
exp_data  = {ctp_: exp_[samples_all] for ctp_, exp_ in exp_decon.items()}
exp_data["Bulk"] = exp_bulk[samples_all]
cell_frac = cell_frac.loc[samples_all]

print("\nexpression data sizes (p x n)): ")
print(pd.Series(dict(map(lambda itm: (itm[0], itm[1].shape), exp_data.items()))))
     
print(f"""\nother data sizes (n x p):
      confidence score = {conf_score.shape}
      cell fraction = {cell_frac.shape}""")


## divide samples into treatment subsets & get clinical info.
aHER2 = clin_info["anti.her2.cycles"].notna()
samples_all = samples_all.tolist()
samples_ct  = clin_info[aHER2].index.tolist()
samples_cm  = clin_info[~aHER2].index.tolist()

clin_all = clin_info.filter(samples_all, axis = 0)
clin_ct  = clin_info.filter(samples_ct,  axis = 0)
clin_cm  = clin_info.filter(samples_cm,  axis = 0)

print(f"""\ndividing samples by treatment:
      all = {len(samples_all)}
      chemo+targeted = {len(samples_ct)}
      chemo = {len(samples_cm)}""")


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
    out_path = "../data/TransNEO_SammutShare/validation/"
    out_file = {"all": "transneo_validation_all_v2.pkl", 
                "cm" : "transneo_validation_chemo_v2.pkl", 
                "ct" : "transneo_validation_chemo_targeted_v2.pkl"}
    
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


    
