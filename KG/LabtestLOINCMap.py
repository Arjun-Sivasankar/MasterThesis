import pandas as pd
import numpy as np
import ast
import pickle
import os
import multiprocessing as mp
from tqdm import tqdm

# Set these for optimal performance
NUM_PROCESSES = 7  # Leave one core free for system processes

print("Starting optimized LOINC mapping process...")

# Load your files
print("Loading MIMIC data...")
with open('/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl', 'rb') as f:
    mimic_df = pickle.load(f)

print("Loading lab mapping data...")
lab_map = pd.read_csv('/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/d_labitems_to_loinc.csv')
lab_map.columns = lab_map.columns.str.strip()

# Build mappings as dictionaries for faster lookups
itemid_to_loinc = dict(zip(lab_map['itemid (omop_source_code)'].astype(str), lab_map['omop_concept_code'].astype(str)))
vocab_map = dict(zip(lab_map['itemid (omop_source_code)'].astype(str), lab_map['omop_vocabulary_id'].astype(str)))

# Optimized MRCONSO loading - only load what we need
print("Loading MRCONSO (filtered to LOINC codes only)...")
mrconso_path = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRCONSO.RRF'
columns = [
    'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI',
    'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
]

# Create a cache file path for faster reloading
cache_dir = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/cache'
os.makedirs(cache_dir, exist_ok=True)
loinc_conso_cache = os.path.join(cache_dir, 'loinc_conso_cache.pkl')

# Try to load from cache first
if os.path.exists(loinc_conso_cache):
    print("Loading LOINC mapping from cache...")
    with open(loinc_conso_cache, 'rb') as f:
        conso = pickle.load(f)
else:
    print("Building LOINC mapping cache (this may take a while)...")
    # Read MRCONSO but filter immediately to only LOINC entries
    # Use chunking to reduce memory usage
    chunk_size = 1000000  # Adjust based on available memory
    chunks = []
    
    for chunk in tqdm(pd.read_csv(mrconso_path, sep='|', names=columns, 
                                 usecols=['CUI', 'LAT', 'SAB', 'CODE'], 
                                 dtype=str, chunksize=chunk_size, index_col=False)):
        # Filter to only English LOINC entries
        filtered = chunk[(chunk['LAT'] == 'ENG') & (chunk['SAB'] == 'LNC')]
        if not filtered.empty:
            chunks.append(filtered)
    
    conso = pd.concat(chunks) if chunks else pd.DataFrame(columns=['CUI', 'LAT', 'SAB', 'CODE'])
    
    # Save to cache for future runs
    with open(loinc_conso_cache, 'wb') as f:
        pickle.dump(conso, f)

print(f"Loaded {len(conso)} LOINC entries from MRCONSO")

# Create a fast lookup dictionary for LOINC → CUI mapping
print("Building lookup dictionaries...")
loinc_to_cui_map = {}
for _, row in conso.iterrows():
    loinc_to_cui_map[row['CODE']] = row['CUI']

# 1. Helper: parse lab_test cells robustly
def parse_codes(cell):
    if isinstance(cell, (np.ndarray, pd.Series)):
        out = []
        for x in cell:
            out += parse_codes(x)
        return out
    if cell is None:
        return []
    if isinstance(cell, float) and np.isnan(cell):
        return []
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        cell_str = cell.strip()
        if cell_str == '' or cell_str.lower() == 'nan':
            return []
        if cell_str.startswith('[') and cell_str.endswith(']'):
            try:
                value = ast.literal_eval(cell_str)
                if isinstance(value, list):
                    return value
                return [value]
            except Exception:
                return [cell_str]
        return [cell_str]
    return [cell]

# Optimized mapping functions
def map_codes_all(lab_codes):
    """Map lab codes to LOINC and CUI in one pass to avoid redundant work"""
    loinc_codes = []
    cuis = []
    
    for lab_code in lab_codes:
        base_itemid = lab_code.split('-')[0] if '-' in lab_code else lab_code
        loinc_code = itemid_to_loinc.get(base_itemid)
        vocab = vocab_map.get(base_itemid)
        
        if loinc_code:
            loinc_codes.append(loinc_code)
            
            if vocab == 'LOINC':
                cui = loinc_to_cui_map.get(loinc_code)
                if cui:
                    cuis.append(cui)
    
    return {
        'loinc': loinc_codes,
        'cuis': cuis
    }

# Worker function for parallel processing
def process_chunk(chunk_df):
    results = []
    for _, row in chunk_df.iterrows():
        lab_codes = parse_codes(row['lab_test'])
        mapping = map_codes_all(lab_codes)
        results.append(mapping)
    return results

# Split the dataframe into chunks for parallel processing
def split_dataframe(df, num_chunks):
    chunk_size = len(df) // num_chunks
    chunks = []
    for i in range(0, len(df), chunk_size):
        if i + chunk_size > len(df):
            chunks.append(df.iloc[i:])
        else:
            chunks.append(df.iloc[i:i+chunk_size])
    return chunks

# Main processing using multiprocessing
print(f"Processing {len(mimic_df)} records with {NUM_PROCESSES} parallel processes...")
chunks = split_dataframe(mimic_df, NUM_PROCESSES)

# Start multiprocessing pool
with mp.Pool(processes=NUM_PROCESSES) as pool:
    chunk_results = list(tqdm(
        pool.imap(process_chunk, chunks),
        total=len(chunks),
        desc="Processing chunks"
    ))

# Combine results from all chunks
all_results = []
for chunk in chunk_results:
    all_results.extend(chunk)

# Update the dataframe with the results
print("Updating dataframe with mapped values...")
mimic_df['lab_test_loinc'] = [result['loinc'] for result in all_results]
mimic_df['lab_test_cuis'] = [result['cuis'] for result in all_results]

print("Saving results...")
mimic_df.to_pickle('/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/mimic_with_lab_mappings.pkl')

print("Done! Process completed successfully.")