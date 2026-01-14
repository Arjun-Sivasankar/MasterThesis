import pickle
import pandas as pd
import numpy as np
import os

icd10 = False
if icd10:
    mimic = pd.read_pickle("./dataset/icd10/mimic.pkl")
else:
    mimic = pd.read_pickle("./dataset/icd9/mimic.pkl")

print(mimic.columns)
print(mimic.head())
print(mimic.shape)

ehr_df = mimic

data = pd.read_pickle("./dataset/unstructured/discharge_pd.pkl")

print(data.columns)
print(data.head())
print(data.shape)

notes_df = data

ehr_df['hadm_id'] = ehr_df['hadm_id'].astype(str)
notes_df['hadm_id'] = notes_df['hadm_id'].astype(str)

merged_df = ehr_df.merge(notes_df, on="hadm_id", how="left")
print(merged_df.head())
print(merged_df.shape)
print(merged_df.columns)

if icd10:
    merged_df.to_pickle("./dataset/merged_icd10.pkl")
else:
    merged_df.to_pickle("./dataset/merged_icd9.pkl")

print("Merged dataframe saved.")

print("\nEXAMPLE:")
print(merged_df.iloc[0])