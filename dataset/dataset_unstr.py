import sys
sys.executable
import re

import gc
import os
import copy
import json
import random
import pickle
import numpy as np
import pandas as pd
# import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing

import torch
import torch.nn.functional as F

data_path = "/data/horse/ws/arsi805e-finetune/Thesis/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note"

discharge_detail_file = os.path.join(data_path, "discharge_detail.csv.gz")
discharge_file = os.path.join(data_path, "discharge.csv.gz")
radiology_detail_file = os.path.join(data_path, "radiology_detail.csv.gz")
radiology_file = os.path.join(data_path, "radiology.csv.gz")

discharge_pd = pd.read_csv(discharge_file, low_memory=False)

print(discharge_pd.columns)
print(discharge_pd.head())
print(discharge_pd.shape)

# 1) List out every section header exactly as it appears
fields = [
    "Name", "Unit No", "Admission Date", "Discharge Date",
    "Date of Birth", "Sex", "Service", "Allergies", "Attending",
    "Chief Complaint", "Major Surgical or Invasive Procedure",
    "History of Present Illness", "Past Medical History",
    "Social History", "Family History", "Physical Exam",
    "Pertinent Results", "Brief Hospital Course",
    "Medications on Admission", "Discharge Medications",
    "Discharge Disposition", "Discharge Diagnosis",
    "Discharge Condition", "Discharge Instructions", "Followup Instructions"
]

# 2) Build a single regex to capture each header and its body
pattern = re.compile(
    r'(?P<field>' + '|'.join(map(re.escape, fields)) + r'):'   # one of the headers
    r'\s*(?P<value>.*?)(?=(?:' + '|'.join(map(re.escape, fields)) + r'):|$)',  # up to next header or end
    re.DOTALL
)

# 3) A helper that takes one raw note, returns a dict { header: text }
def parse_sections(text):
    matches = pattern.finditer(text)
    return {m.group("field"): m.group("value").strip() for m in matches}

# 4) Apply across the DataFrame
parsed_df = discharge_pd["text"].apply(parse_sections).apply(pd.Series)

# 5) Join back (optional)
discharge_pd = discharge_pd.join(parsed_df)

print(discharge_pd.columns)
print(discharge_pd.head())
print(discharge_pd.shape)

# ## Radiology reports
# radiology_pd = pd.read_csv(radiology_file, low_memory=False)
# print(radiology_pd.columns)
# print(radiology_pd.head())
# print(radiology_pd.shape)

## save the discharge_pd
discharge_pd.to_pickle("/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/unstructured/discharge_pd.pkl")