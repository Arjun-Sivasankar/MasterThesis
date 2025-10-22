import pandas as pd
import networkx as nx
from tqdm import tqdm
import os
import re
import pickle

# Paths to RRF files (customize)
# Update paths to match your actual directory structure
MRCONSO = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRCONSO.RRF'
MRREL = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRREL.RRF'
MRSTY = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRSTY.RRF'

# 1. Load UMLS Concept Names
def load_mrconso(path, langs=['ENG']):
    # See UMLS documentation for column order!
    columns = [
        'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI',
        'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
    ]
    df = pd.read_csv(path, sep='|', names=columns, dtype=str, index_col=False)
    df = df[df['LAT'].isin(langs)]
    return df

# First, format codes properly for consistent analysis
def format_icd9(code: str) -> str:
    code = re.sub(r"\s+","", str(code)).upper().rstrip(".")
    if not code: return ""
    if code[0].isdigit():
        if len(code)>3 and "." not in code: return code[:3]+"."+code[3:]
        return code
    if code[0] == "V":
        if len(code)>3 and "." not in code: return code[:3]+"."+code[3:]
        return code
    if code[0] == "E":
        if len(code)>4 and "." not in code: return code[:4]+"."+code[4:]
        return code
    return code

# Define functions to identify code types
def is_diagnosis_code(code):
    # Regular numeric codes (001-999)
    if code[0].isdigit():
        # Procedure codes start with 00-99 followed by decimal
        if re.match(r"^[0-9]{2}\.[0-9]{1,2}$", code):
            return False
        return True
    # V codes (supplementary classification)
    if code.startswith('V'):
        return True
    # E codes (external causes)
    if code.startswith('E'):
        return True
    return False

def is_procedure_code(code):
    # Procedure codes are numeric only and have specific format
    if code[0].isdigit():
        if re.match(r"^[0-9]{2}\.[0-9]{1,2}$", code):
            return True
    return False

print("Loading CONSO..." )
conso = load_mrconso(MRCONSO)
print("Loaded CONSO file.....")

icd9_concept_df = conso[
    (conso["TS"] == "P") &
    (conso["SAB"] == "ICD9CM") &
    (conso["SUPPRESS"] != "O")
]

# Apply formatting
icd9_concept_df['formatted_code'] = icd9_concept_df['CODE'].apply(format_icd9)

# Apply the classification
icd9_concept_df['is_diagnosis'] = icd9_concept_df['formatted_code'].apply(is_diagnosis_code)
icd9_concept_df['is_procedure'] = icd9_concept_df['formatted_code'].apply(is_procedure_code)

# Count by type
diagnosis_df = icd9_concept_df[icd9_concept_df['is_diagnosis']]
procedure_df = icd9_concept_df[icd9_concept_df['is_procedure']]
unclassified_df = icd9_concept_df[~(icd9_concept_df['is_diagnosis'] | icd9_concept_df['is_procedure'])]

print(f"Total ICD-9-CM codes: {len(icd9_concept_df)}")
print(f"Diagnosis codes: {len(diagnosis_df)} ({len(diagnosis_df)/len(icd9_concept_df)*100:.2f}%)")
print(f"Procedure codes: {len(procedure_df)} ({len(procedure_df)/len(icd9_concept_df)*100:.2f}%)")
print(f"Unclassified codes: {len(unclassified_df)} ({len(unclassified_df)/len(icd9_concept_df)*100:.2f}%)")

# Count unique codes and CUIs
print("\nUnique diagnosis codes:", len(diagnosis_df['CODE'].unique()))
print("Unique diagnosis CUIs:", len(diagnosis_df['CUI'].unique()))
print("\nUnique procedure codes:", len(procedure_df['CODE'].unique()))
print("Unique procedure CUIs:", len(procedure_df['CUI'].unique()))

# Show examples
print("\nExample diagnosis codes:")
print(diagnosis_df[['CODE', 'formatted_code', 'STR']].head())

print("\nExample procedure codes:")
print(procedure_df[['CODE', 'formatted_code', 'STR']].head() if len(procedure_df) > 0 else "No procedure codes found")

print("\nExample unclassified codes:")
print(unclassified_df[['CODE', 'formatted_code', 'STR']].head() if len(unclassified_df) > 0 else "No unclassified codes found")