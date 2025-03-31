#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:30:59 2022

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

import numpy as np, pandas as pd
from miscellaneous import date_time, write_xlsx


#%% read clinical data.

data_path = "../data/BrighTNess/"
data_file = "GSE164458_series_matrix.txt"

## read data with variable #fields as dataframe.
## tip: https://stackoverflow.com/questions/55589500/reading-text-file-with-variable-columns-in-pandas-dataframe
## initialize 'names' with large #columns to read (greater than max. available 
## #fields- or error will occur).
clin_data_txt = pd.read_table(
    data_path + data_file, sep = "\t", engine = "python", dtype = str, 
    header = None, index_col = 0, 
    names = ["field"] + [f"ln{nn}" for nn in range(500)]
)
clin_data_txt = clin_data_txt.dropna(axis = 1, how = "all")                    # remove empty lines
clin_data_txt.index = [fld_.replace("!", "") for fld_ in clin_data_txt.index]  # remove '!' from field names


#%% clean data.

## get simplified treatment list per patient ['PBO' stands for placebo].
def get_treatments(arm_description):
    treatments = "+".join([
        trt.strip().split(" ")[0].capitalize() \
            for trt in arm_description.split("+") if "PBO" not in trt
    ])
    return treatments
    

## get sample charactistics.
keep = ["Sample_title", "Sample_characteristics_ch1"]

clin_data_samples = clin_data_txt.loc[keep].T.copy()                           # selects multiple rows named 'Sample_characteristics_ch1'
clin_data_samples.index = clin_data_samples[keep[0]].apply(
    lambda id_: f"Sample_{id_.split('_')[0]}"                                  # make sample names as 'Sample_{sample_id}'
).astype(str)
clin_data_samples.drop(columns = keep[0], inplace = True)
clin_data_samples.columns = clin_data_samples.iloc[0].apply(
    lambda char_: char_.split(": ")[0]                                         # keep characteristics name
).astype(str).values
clin_data_samples = clin_data_samples.applymap(                                # keep characteristics value
    lambda info_: info_.split(": ")[1]
)
clin_data_samples[clin_data_samples == "NA"] = np.nan                          # replace 'NA' with nan

## simplified treatment list for ease of filtering.
clin_data_samples["treatment"] = \
    clin_data_samples["description_of_planned_arm"].apply(get_treatments)

## from Methods/Procedures in Loibi et al. 2018 [PMID: 29501363]:
## "Patients who did not have surgery were counted as not achieving 
## pathological complete response."
clin_data_samples["surgery"] = \
    ~clin_data_samples["residual_cancer_burden_class"].isna()

## convert to appropriate data types.
for fld_, info_ in clin_data_samples.items():
    try:
        clin_data_samples[fld_] = info_.astype(float)
    except:
        clin_data_samples[fld_] = info_.astype(str)

## reorder columns for convenience.
ordered_cols = [
    "description_of_planned_arm", "pretreatment_lymphnode_stage", 
    "ac_planned_schedule", "smoking_history", "ecog_ps_baseline", 
    "planned_arm_code", "treatment", "surgery", 
    "residual_cancer_burden_class", "pathologic_complete_response"
]
clin_data_samples = clin_data_samples[ordered_cols]

print(f"\nvariable types = \n{clin_data_samples.dtypes}")


## additional info [all the fields with a single available line].
clin_data_supp = clin_data_txt[
    clin_data_txt.isna().sum(axis = 1) == (clin_data_txt.shape[1] - 1)
]
clin_data_supp = clin_data_supp.dropna(axis = 1, how = "all").squeeze()        # remove empty lines
clin_data_supp[clin_data_supp == "NA"] = np.nan
for fld_, info_ in clin_data_supp.items():                                     # convert to appropriate data types
    try:
        clin_data_supp[fld_] = float(info_)
    except:
        clin_data_supp[fld_] = str(info_)


#%% save data.

svdat = False                                                                  # set True to save data 

if svdat:
    datestamp = date_time()
    clin_file = f"GSE164458_BrighTNess_clinical_info_SRD_{datestamp}.xlsx"
    clin_info = {"samples": clin_data_samples, 
                 "additional_info": clin_data_supp}
    write_xlsx(data_path + clin_file, data = clin_info)
    
