#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:26:25 2022

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
    bulk  = "../data/BrighTNess/", 
    decon = "../data/BrighTNess/out_codefacs_brightness_v2/"
)
data_files = dict(
    clin  = "GSE164458_BrighTNess_clinical_info_SRD_04Oct2022.xlsx", 
    bulk  = "GSE164458_BrighTNess_RNAseq_TPM_v2_SRD_09Oct2022.csv"
)
data_files["decon"] = list_files(data_path["decon"], pattern = "expression")
data_files["conf"]  = "confidence_score.txt"
data_files["frac"]  = "estimated_cell_fractions.txt"


#%% get clinical data & available sample size.

## trial arm description from Loibi et al. 2018 [PMID: 29501363]:
## PBO = placebo, PARPi = PARP inhibitor := veliparib, taxane := paclitaxel
## arm A: TCV => patients receiving taxane + carboplatin + PARPi.
## arm B: TC  => patients receiving taxane + carboplatin + PARPi PBO.
## arm C: T   => patients receiving taxane + carboplatin PBO + PARPi PBO.

clin_info = pd.read_excel(data_path["bulk"] + data_files["clin"], 
                          sheet_name = "samples", header = 0, index_col = 0)

## filter patients by surgery (available RCB index).
filter_by = clin_info["surgery"].astype(bool) & \
    ~clin_info["residual_cancer_burden_class"].isna()

clin_info_all = clin_info[filter_by]
clin_info_cm  = clin_info_all[clin_info_all["planned_arm_code"] != "A"]
clin_info_ct  = clin_info_all[clin_info_all["planned_arm_code"] == "A"]

samples_all, samples_cm, samples_ct = (
    clin_info_all.index.values, clin_info_cm.index.values, 
    clin_info_ct.index.values)

print(f"""sample counts (n):
      all = {samples_all.size}
      chemo only = {samples_cm.size}
      chemo + targeted (PARPi) = {samples_ct.size}\n""")


#%% get expression & abundance data.

def read_delim(file, sep = ",", skiprows = 0):
    data = pd.read_table(
        file, sep = sep, skiprows = skiprows, header = 0, index_col = 0
    )
    
    ## format column names.
    try:
        data.columns = data.columns.astype(int)
    except:
        data.columns = data.columns.astype(str)   
    
    return data


## read expression files.
exp_bulk = read_delim(data_path["bulk"] + data_files["bulk"])

exp_decon = { }
for decon_file in tqdm(data_files["decon"]):
    ctp_ = decon_file.split("_")[1].split(".")[0].replace(" ", "_")            # cell type (replace space w/ underscore)
    exp_ = read_delim(data_path["decon"] + decon_file, sep = " ", skiprows = 0)
    exp_decon[ctp_] = exp_

## read other files.
conf_score = read_delim(data_path["decon"] + data_files["conf"], sep = "\t")
conf_score.columns = map(lambda ctp_: ctp_.replace(" ", "_"), conf_score.columns)

cell_frac = read_delim(data_path["decon"] + data_files["frac"], sep = "\t")
cell_frac.columns = map(lambda ctp_: ctp_.replace(" ", "_"), cell_frac.columns)


## check cell type ordering.
if (list(exp_decon.keys()) == conf_score.columns.tolist()) and \
    (conf_score.columns.tolist() == cell_frac.columns.tolist()):
    cell_types = conf_score.columns.values
else:
    raise ValueError("cell types are not the same or not in the same order between expression, confidence score and cell fraction files!")

## put all expression data together.
exp_data = exp_decon.copy();    exp_data["Bulk"] = exp_bulk.copy()

print("\nexpression data sizes (p x n)): ")
print(pd.Series(
    dict(map(lambda itm: (itm[0], itm[1].shape), exp_data.items()))))
     
print(f"""\nother data sizes (n x p):
      confidence score = {conf_score.shape}
      cell fraction = {cell_frac.shape}""")


#%% separate data by sample subsets & save.

svdat = False                                                                  # set True to save data 

get_exp_smpl = lambda smpl: {ctp_: exp_[smpl] for ctp_, exp_ in exp_data.items()}
get_num_resp = lambda clin: (clin["pathologic_complete_response"] == "pCR").astype(int)


## samples = all (all three arms).
data_all = {
    "exp": get_exp_smpl(samples_all),   "resp": get_num_resp(clin_info_all), 
    "frac": cell_frac.loc[samples_all], "conf": conf_score,
    "clin": clin_info_all}

## samples = chemo (arm B + C).
data_cm = {
    "exp": get_exp_smpl(samples_cm),   "resp": get_num_resp(clin_info_cm), 
    "frac": cell_frac.loc[samples_cm], "conf": conf_score, 
    "clin": clin_info_cm}

## samples = chemo + targeted (PARPi - arm A).
data_ct = {
    "exp": get_exp_smpl(samples_ct),   "resp": get_num_resp(clin_info_ct), 
    "frac": cell_frac.loc[samples_ct], "conf": conf_score,
    "clin": clin_info_ct}


## final check for samples.
def check_samples(ds):
    exp_smpl = reduce(
        np.intersect1d, [exp_.columns for exp_ in ds["exp"].values()])
    
    if ((exp_smpl.tolist() == ds["resp"].index.tolist()) and 
        (ds["resp"].index.tolist() == ds["frac"].index.tolist())):
        print("ok!")
    else:
        raise ValueError(
            "samples are not the same or not in the same order between \
                expression, response and cell fraction files!")
    

check_samples(data_all);    check_samples(data_cm);     check_samples(data_ct)


## save prepared data.
if svdat:
    print("\nsaving data...")
    out_path = "../data/BrighTNess/validation/"
    out_file = {"all": "brightness_data_all_v2.pkl", 
                "cm" : "brightness_data_chemo_v2.pkl", 
                "ct" : "brightness_data_chemo_targeted_v2.pkl"}
    
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

