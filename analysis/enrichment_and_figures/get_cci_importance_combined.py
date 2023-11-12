#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:02:43 2023

@author: dhrubas2
"""

## set up necessary directories/paths.
_wpath_ = "/Users/dhrubas2/OneDrive - National Institutes of Health/Projects/TMEcontribution/analysis/submission/Code/analysis/"
_mpath_ = "miscellaneous/py/"

## load necessary packages.
import os, sys
os.chdir(_wpath_)                                                              # current path
if _mpath_ not in sys.path:
    sys.path.append(_mpath_)                                                   # to load miscellaneous

import numpy as np, pandas as pd
from miscellaneous import date_time, tic, write_xlsx


#%% functions.

read_data = lambda file, **kwarg: pd.read_excel(
    file, header = 0, index_col = 0, **kwarg)


#%% get data.

data_path = "../data/TransNEO/transneo_analysis/mdl_data/"
data_file = ["tn_valid_lirics_feature_importance_chemo_filteredCCI_th0.99_RF_allfeatures_3foldCV_25Mar2023.xlsx", 
             "brightness_lirics_feature_importance_chemo_filteredCCI_th0.99_RF_allfeatures_3foldCV_25Mar2023.xlsx", 
             "tn_valid_lirics_feature_list_chemo_RF_allfeatures_3foldCV_25Mar2023.xlsx", 
             "brightness_lirics_feature_list_chemo_RF_allfeatures_3foldCV_25Mar2023.xlsx"]

cci_tn_val     = read_data(data_path + data_file[0])
cci_bn         = read_data(data_path + data_file[1])
cci_all_tn_val = read_data(data_path + data_file[2])
cci_all_bn     = read_data(data_path + data_file[3])


#%% combine CCI lists & rank.

cci_list_cmn   = np.intersect1d(cci_tn_val.index, cci_bn.index).tolist()
cci_cmn_tn_val = cci_tn_val.loc[cci_list_cmn]
cci_cmn_bn     = cci_bn.loc[cci_list_cmn]

cci_cmn            = cci_cmn_tn_val.rename(columns = {"MDI": "MDI_tn_val"})
cci_cmn["MDI_bn"]  = cci_cmn_bn.loc[cci_cmn.index, "MDI"]
cci_cmn["MDI_all"] = cci_cmn[["MDI_tn_val", "MDI_bn"]].mean(axis = 1)
# cci_cmn["MDI_all"] = cci_cmn[["MDI_tn_val", "MDI_bn"]].apply(
#     lambda x: np.sqrt(np.prod(x)), axis = 1)                                   # geometric mean
cci_cmn.sort_values(by = "MDI_all", ascending = True)

print(f"total #shared CCIs = {cci_cmn.shape[0]}")
print(f"displaying top 10 CCIs:\n{cci_cmn.head(10)}")


## get background CCI list.
cci_list_all = {"all": np.intersect1d(cci_all_tn_val.index, cci_all_bn.index), 
                "tn_val": np.setdiff1d(cci_all_tn_val.index, cci_all_bn.index), 
                "bn": np.setdiff1d(cci_all_bn.index, cci_all_tn_val.index)}

cci_all = pd.concat([cci_all_tn_val.loc[cci_list_all["all"]], 
                     cci_all_tn_val.loc[cci_list_all["tn_val"]], 
                     cci_all_bn.loc[cci_list_all["bn"]]], axis = 0)
cci_all[["tn_val", "bn"]] = np.concatenate([
    np.tile([1, 1], reps = (cci_list_all["all"].size, 1)), 
    np.tile([1, 0], reps = (cci_list_all["tn_val"].size, 1)), 
    np.tile([0, 1], reps = (cci_list_all["bn"].size, 1))])
cci_all.rename_axis("CCIannot", inplace = True)
cci_all.index = (cci_all.LigandCell + "-" + cci_all.ReceptorCell + "::" + 
                 cci_all.LigandGene + "-" + cci_all.ReceptorGene)

print(f"\ntotal #background CCIs = {cci_all.shape[0]}")


#%% save data.

svdat = False

if svdat:
    out_file = data_file[0].replace("tn_valid_", "all_")
    out_dict = {"topSharedCCIs": cci_cmn, "allCCIs": cci_all}
    
    write_xlsx(data_path + out_file, out_dict)

