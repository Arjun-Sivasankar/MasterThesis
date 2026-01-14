import gc
import os
import copy
import json
import random
import pickle
import pandas as pd
import numpy as np

# file path : mimic 
data_path = "mimic-iv-3.1/hosp"

d_icd_procedure_file = os.path.join(data_path, "d_icd_procedures.csv.gz")
d_icd_diagnosis_file = os.path.join(data_path, "d_icd_diagnoses.csv.gz")

# read files
d_icd_diag = pd.read_csv(d_icd_diagnosis_file, low_memory=False)
d_icd_proc = pd.read_csv(d_icd_procedure_file, low_memory=False)

# for icd9 codes:
d_icd_diag_icd9 = d_icd_diag[d_icd_diag['icd_version'] == 9]
d_icd_proc_icd9 = d_icd_proc[d_icd_proc['icd_version'] == 9]

# for icd10 codes:
d_icd_diag_icd10 = d_icd_diag[d_icd_diag['icd_version'] == 10]
d_icd_proc_icd10 = d_icd_proc[d_icd_proc['icd_version'] == 10]  

# save files
os.makedirs("./dataset/codes/", exist_ok=True)

d_icd_diag_icd9.to_pickle("./dataset/codes/icd9.pkl")
d_icd_diag_icd9.to_csv("./dataset/codes/icd9.csv", index=False)
d_icd_proc_icd9.to_pickle("./dataset/codes/icd9proc.pkl")

# for icd10 codes:
d_icd_diag_icd10.to_pickle("./dataset/codes/icd10.pkl")
d_icd_proc_icd10.to_pickle("./dataset/codes/icd10proc.pkl")