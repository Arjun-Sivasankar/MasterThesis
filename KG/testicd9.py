# import pandas as pd
# import numpy as np
# import json
# import re
# import pickle
# import os
# import sys
# from tqdm import tqdm
# from collections import defaultdict

# # Paths to RRF files and output directories
# MRCONSO = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRCONSO.RRF'
# MRREL = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRREL.RRF'
# MRSTY = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRSTY.RRF'

# MIMIC_ICD9_PATH = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl'
# DATASET_PATH = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl'
# MAPPING_PATH = '/data/horse/ws/arsi805e-finetune/Thesis/mappings/cui_to_icd9_EXACT.json'
# OUTPUT_DIR = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/mapping_analysis'

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Helper functions for ICD-9 code formatting and classification
# def format_icd9(code: str) -> str:
#     """Format ICD-9 diagnosis codes to standard format with decimal points."""
#     code = re.sub(r"\s+","", str(code)).upper().rstrip(".")
#     if not code: return ""
#     if code[0].isdigit():
#         if len(code)>3 and "." not in code: return code[:3]+"."+code[3:]
#         return code
#     if code[0] == "V":
#         if len(code)>3 and "." not in code: return code[:3]+"."+code[3:]
#         return code
#     if code[0] == "E":
#         if len(code)>4 and "." not in code: return code[:4]+"."+code[4:]
#         return code
#     return code

# def format_icd9_procedure(code: str) -> str:
#     """Format ICD-9 procedure codes to standard format with decimal points."""
#     code = re.sub(r"\s+","", str(code)).upper().rstrip(".")
#     if code.startswith("PRO_"):
#         code = code[4:]  # Remove 'PRO_' prefix
#     if not code: return ""
#     if code[0].isdigit():
#         if len(code)>2 and "." not in code: return code[:2]+"."+code[2:]
#         return code
#     return code 

# def is_diagnosis_code(code):
#     """Identify if a code is an ICD-9 diagnosis code."""
#     if code[0].isdigit():
#         if re.match(r"^[0-9]{2}\.[0-9]{1,2}$", code):
#             return False
#         return True
#     if code.startswith('V'):
#         return True
#     if code.startswith('E'):
#         return True
#     return False

# def is_procedure_code(code):
#     """Identify if a code is an ICD-9 procedure code."""
#     if code[0].isdigit():
#         if re.match(r"^[0-9]{2}\.[0-9]{1,2}$", code):
#             return True
#     return False

# def load_mrconso(path, langs=['ENG']):
#     """Load and filter the MRCONSO file from UMLS."""
#     columns = [
#         'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI',
#         'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
#     ]
#     df = pd.read_csv(path, sep='|', names=columns, dtype=str, index_col=False)
#     df = df[df['LAT'].isin(langs)]
#     return df

# # Phase 1: ICD-9 code analysis from UMLS
# print("\n===== Phase 1: ICD-9 Analysis from UMLS =====")

# print("Loading CONSO..." )
# try:
#     conso = load_mrconso(MRCONSO)
#     print(f"Loaded CONSO file with {len(conso):,} rows")

#     # Filter for ICD9CM concepts
#     icd9_concept_df = conso[
#         (conso["TS"] == "P") &
#         (conso["SAB"] == "ICD9CM") &
#         (conso["SUPPRESS"] != "O")
#     ]

#     # Apply formatting and classification
#     icd9_concept_df['formatted_code'] = icd9_concept_df['CODE'].apply(format_icd9)
#     icd9_concept_df['is_diagnosis'] = icd9_concept_df['formatted_code'].apply(is_diagnosis_code)
#     icd9_concept_df['is_procedure'] = icd9_concept_df['formatted_code'].apply(is_procedure_code)

#     # Separate by type
#     diagnosis_df = icd9_concept_df[icd9_concept_df['is_diagnosis']]
#     procedure_df = icd9_concept_df[icd9_concept_df['is_procedure']]
#     unclassified_df = icd9_concept_df[~(icd9_concept_df['is_diagnosis'] | icd9_concept_df['is_procedure'])]

#     # Report statistics
#     print(f"Total ICD-9-CM codes: {len(icd9_concept_df):,}")
#     print(f"Diagnosis codes: {len(diagnosis_df):,} ({len(diagnosis_df)/len(icd9_concept_df)*100:.2f}%)")
#     print(f"Procedure codes: {len(procedure_df):,} ({len(procedure_df)/len(icd9_concept_df)*100:.2f}%)")
#     print(f"Unclassified codes: {len(unclassified_df):,} ({len(unclassified_df)/len(icd9_concept_df)*100:.2f}%)")

#     # Count unique codes and CUIs
#     print(f"\nUnique diagnosis codes: {len(diagnosis_df['CODE'].unique()):,}")
#     print(f"Unique diagnosis CUIs: {len(diagnosis_df['CUI'].unique()):,}")
#     print(f"Unique procedure codes: {len(procedure_df['CODE'].unique()):,}")
#     print(f"Unique procedure CUIs: {len(procedure_df['CUI'].unique()):,}")
    
# except Exception as e:
#     print(f"Error loading MRCONSO directly with pandas: {str(e)}")
#     print("Will proceed with alternative methods")
#     diagnosis_df = pd.DataFrame()
#     procedure_df = pd.DataFrame()

# # Phase 2: Dataset Analysis
# print("\n===== Phase 2: Dataset Analysis =====")
# print("Loading dataset...")
# with open(DATASET_PATH, 'rb') as f:
#     data = pickle.load(f)
# print(f"Dataset loaded with {len(data):,} records")

# # Extract and format all codes from dataset
# all_icd_codes = set()
# all_pro_codes = set()
# for idx, row in data.iterrows():
#     icd_codes = row['icd_code']
#     pro_codes = row['pro_code']
#     if isinstance(icd_codes, np.ndarray):
#         all_icd_codes.update(icd_codes)
#     if isinstance(pro_codes, list):
#         all_pro_codes.update(pro_codes)

# formatted_icd_codes = {format_icd9(code) for code in all_icd_codes}
# formatted_pro_codes = {format_icd9_procedure(code) for code in all_pro_codes}

# print(f"Total unique diagnosis codes in dataset: {len(all_icd_codes):,}")
# print(f"Total unique procedure codes in dataset: {len(all_pro_codes):,}")
# print(f"Total unique formatted diagnosis codes in dataset: {len(formatted_icd_codes):,}")
# print(f"Total unique formatted procedure codes in dataset: {len(formatted_pro_codes):,}")

# # Check UMLS coverage if we loaded the CONSO file successfully
# if not diagnosis_df.empty:
#     diagnosis_coverage = formatted_icd_codes.intersection(set(diagnosis_df['formatted_code']))
#     procedure_coverage = formatted_pro_codes.intersection(set(procedure_df['formatted_code']))
    
#     print(f"\nDiagnosis codes in dataset covered by UMLS: {len(diagnosis_coverage):,}/{len(formatted_icd_codes):,}")
#     print(f"Procedure codes in dataset covered by UMLS: {len(procedure_coverage):,}/{len(formatted_pro_codes):,}")
#     print(f"Diagnosis coverage: {len(diagnosis_coverage)/len(formatted_icd_codes)*100:.2f}%")
#     print(f"Procedure coverage: {len(procedure_coverage)/len(formatted_pro_codes)*100:.2f}%")

# # Phase 3: CUI Mapping Analysis
# print("\n===== Phase 3: CUI Mapping Analysis =====")

# # Load the CUI to ICD-9 mappings
# print("Loading CUI to ICD-9 mappings...")
# with open(MAPPING_PATH, 'r') as f:
#     cui_to_icd9_map = json.load(f)
# print(f"Loaded {len(cui_to_icd9_map):,} CUIs with ICD-9 mappings")

# # Create reverse mapping
# icd9_to_cui = {}
# for cui, codes in cui_to_icd9_map.items():
#     for code in codes:
#         if code not in icd9_to_cui:
#             icd9_to_cui[code] = []
#         icd9_to_cui[code].append(cui)

# print(f"Created reverse mapping with {len(icd9_to_cui):,} unique ICD-9 codes")

# # Check coverage of dataset codes using the mapping
# codes_with_cui = set()
# codes_without_cui = set()
# dataset_code_to_cui = {}
# dataset_cui_set = set()

# print("\n=== Analyzing dataset ICD-9 to CUI mappings ===")
# for code in formatted_icd_codes:
#     if code in icd9_to_cui:
#         codes_with_cui.add(code)
#         dataset_code_to_cui[code] = icd9_to_cui[code]
#         dataset_cui_set.update(icd9_to_cui[code])
#     else:
#         codes_without_cui.add(code)

# # Calculate coverage metrics
# code_coverage_percent = len(codes_with_cui) / len(formatted_icd_codes) * 100

# print(f"Dataset ICD-9 codes with CUI mappings: {len(codes_with_cui):,}/{len(formatted_icd_codes):,} ({code_coverage_percent:.2f}%)")
# print(f"Dataset ICD-9 codes without CUI mappings: {len(codes_without_cui):,}")
# print(f"Total distinct CUIs referenced by dataset codes: {len(dataset_cui_set):,}")

# if len(codes_without_cui) > 0:
#     print("\nSample codes without mappings:")
#     for code in list(codes_without_cui)[:5]:
#         print(f"  {code}")

# # Check if the CUIs exist in CONSO
# print("\n=== Loading CONSO to verify CUIs ===")

# # First try the most efficient approach - loading just the CUI column
# try:
#     conso_cui = pd.read_csv(MRCONSO, sep='|', usecols=[0], names=['CUI'], dtype=str)
#     conso_cui_set = set(conso_cui['CUI'].unique())
#     print(f"CONSO loaded with {len(conso_cui_set):,} unique CUIs")
    
#     # Check CUI existence in CONSO
#     dataset_cuis_in_conso = dataset_cui_set.intersection(conso_cui_set)
#     dataset_cuis_not_in_conso = dataset_cui_set - conso_cui_set
    
#     cui_coverage_percent = len(dataset_cuis_in_conso) / len(dataset_cui_set) * 100
    
#     print("\n=== CUI Existence in CONSO Analysis ===")
#     print(f"Dataset CUIs found in CONSO: {len(dataset_cuis_in_conso):,}/{len(dataset_cui_set):,} ({cui_coverage_percent:.2f}%)")
#     print(f"Dataset CUIs not found in CONSO: {len(dataset_cuis_not_in_conso):,}")
    
#     if len(dataset_cuis_not_in_conso) > 0:
#         print("\nSample CUIs not found in CONSO:")
#         for cui in list(dataset_cuis_not_in_conso)[:5]:
#             print(f"  {cui}")
    
#     # Create comprehensive mapping
#     print("\n=== Generating Comprehensive ICD-9 to CUI Mapping for Dataset ===")
#     dataset_code_mapping = {}
#     for code in formatted_icd_codes:
#         cuis = dataset_code_to_cui.get(code, [])
#         cuis_in_conso = [cui for cui in cuis if cui in conso_cui_set]
        
#         dataset_code_mapping[code] = {
#             "has_mapping": code in codes_with_cui,
#             "cuis": cuis,
#             "cuis_in_conso": cuis_in_conso,
#             "all_cuis_in_conso": len(cuis) > 0 and len(cuis) == len(cuis_in_conso)
#         }
    
#     # Calculate final metrics
#     codes_with_valid_cuis = sum(1 for info in dataset_code_mapping.values() if info["all_cuis_in_conso"])
#     valid_mapping_percent = codes_with_valid_cuis / len(formatted_icd_codes) * 100
    
#     print(f"Dataset ICD-9 codes with valid CUI mappings in CONSO: {codes_with_valid_cuis:,}/{len(formatted_icd_codes):,} ({valid_mapping_percent:.2f}%)")
    
#     # Export mappings
#     export_data = []
#     for code, info in dataset_code_mapping.items():
#         export_data.append({
#             "icd_code": code,
#             "has_mapping": info["has_mapping"],
#             "cuis": ",".join(info["cuis"]) if info["cuis"] else "",
#             "cui_count": len(info["cuis"]),
#             "cuis_in_conso": ",".join(info["cuis_in_conso"]) if info["cuis_in_conso"] else "",
#             "all_cuis_in_conso": info["all_cuis_in_conso"]
#         })
    
#     mapping_df = pd.DataFrame(export_data)
#     mapping_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_icd9_cui_mapping.csv'), index=False)
    
#     # Create a clean version for knowledge graph
#     clean_mapping_df = mapping_df[mapping_df['all_cuis_in_conso']].copy()
#     clean_mapping_df = clean_mapping_df[['icd_code', 'cuis']].copy()
#     clean_mapping_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_icd9_cui_clean_mapping.csv'), index=False)
    
#     # Export problematic codes
#     problematic_df = mapping_df[~mapping_df['all_cuis_in_conso']].copy()
#     problematic_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_icd9_problematic_codes.csv'), index=False)
    
#     print(f"\nAnalysis complete. Full mapping saved to {os.path.join(OUTPUT_DIR, 'dataset_icd9_cui_mapping.csv')}")
#     print(f"Clean mapping for KG use saved to {os.path.join(OUTPUT_DIR, 'dataset_icd9_cui_clean_mapping.csv')}")
#     print(f"Problematic codes saved to {os.path.join(OUTPUT_DIR, 'dataset_icd9_problematic_codes.csv')}")
    
# except Exception as e:
#     print(f"Error loading CONSO for CUI verification: {str(e)}")
#     print("Using clean mapping from previous run...")
#     clean_mapping_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'dataset_icd9_cui_clean_mapping.csv'))
#     print(f"Loaded {len(clean_mapping_df)} clean mappings")

# # Phase 4: SAB Analysis for Clean Mappings
# print("\n===== Phase 4: SAB Analysis for Clean Mappings =====")

# # Load or use the clean mapping
# try:
#     if 'clean_mapping_df' not in locals():
#         clean_mapping_path = os.path.join(OUTPUT_DIR, 'dataset_icd9_cui_clean_mapping.csv')
#         print(f"Loading clean mapping from {clean_mapping_path}")
#         clean_mapping_df = pd.read_csv(clean_mapping_path)
    
#     print(f"Analyzing {len(clean_mapping_df)} ICD-9 code mappings")
    
#     # Extract all unique CUIs
#     all_cuis = set()
#     for cui_str in clean_mapping_df['cuis']:
#         cuis = cui_str.split(',')
#         all_cuis.update(cuis)
#     print(f"Found {len(all_cuis):,} unique CUIs in the mapping")
    
#     # Process MRCONSO directly line by line
#     print("Processing MRCONSO file directly...")
#     cui_data = defaultdict(list)
    
#     # Read file line by line
#     with open(MRCONSO, 'r') as f:
#         for line in tqdm(f, desc="Reading MRCONSO"):
#             parts = line.strip().split('|')
#             if len(parts) >= 15:  # Ensure we have enough columns
#                 cui = parts[0]
#                 lat = parts[1]
#                 sab = parts[11]
#                 str_val = parts[14]
                
#                 # Filter for our CUIs and English terms
#                 if cui in all_cuis and lat == 'ENG':
#                     cui_data[cui].append({
#                         'sab': sab,
#                         'str': str_val
#                     })
    
#     print(f"Found data for {len(cui_data):,} CUIs out of {len(all_cuis):,} in mapping")
    
#     # Process the data to create SAB information
#     cui_to_sabs = {}
#     for cui, entries in tqdm(cui_data.items(), desc="Processing CUI data"):
#         sabs = defaultdict(list)
#         for entry in entries:
#             sabs[entry['sab']].append(entry['str'])
        
#         unique_sabs = list(sabs.keys())
#         sab_descriptions = []
        
#         for sab in unique_sabs:
#             # Use the first string for each SAB
#             sab_descriptions.append(f"{sab}:{sabs[sab][0]}")
        
#         cui_to_sabs[cui] = {
#             'sabs': unique_sabs,
#             'sab_count': len(unique_sabs),
#             'sab_descriptions': sab_descriptions
#         }
    
#     # Create the enriched mapping
#     results = []
#     for _, row in tqdm(clean_mapping_df.iterrows(), desc="Creating output"):
#         icd_code = row['icd_code']
#         cuis = row['cuis'].split(',')
        
#         for cui in cuis:
#             if cui in cui_to_sabs:
#                 info = cui_to_sabs[cui]
#                 results.append({
#                     'icd_code': icd_code,
#                     'cui': cui,
#                     'sab_count': info['sab_count'],
#                     'sabs': ','.join(info['sabs']),
#                     'sab_descriptions': '|'.join(info['sab_descriptions'])
#                 })
    
#     # Save results
#     results_df = pd.DataFrame(results)
#     if len(results_df) > 0:
#         results_df.to_csv(os.path.join(OUTPUT_DIR, 'simple_icd9_cui_sab_mapping.csv'), index=False)
#         print("\nSample output:")
#         print(results_df.head())
#         print(f"\nMapping saved to: {os.path.join(OUTPUT_DIR, 'simple_icd9_cui_sab_mapping.csv')}")
#         print(f"Total rows: {len(results_df):,}")
    
#         # Count unique ICD codes and CUIs
#         unique_icd_codes = results_df['icd_code'].nunique()
#         unique_cuis = results_df['cui'].nunique()
#         print(f"Unique ICD-9 codes: {unique_icd_codes:,}")
#         print(f"Unique CUIs: {unique_cuis:,}")
        
#         # Check for ICD9CM in SABs
#         has_icd9cm = 0
#         missing_icd9cm = 0
#         icd_code_with_icd9cm = set()
#         icd_code_without_icd9cm = set()
        
#         print("Analyzing SAB coverage...")
#         for idx, row in tqdm(results_df.iterrows(), total=len(results_df)):
#             sabs = row['sabs'].split(',')
#             icd_code = row['icd_code']
#             cui = row['cui']
            
#             if 'ICD9CM' in sabs:
#                 has_icd9cm += 1
#                 icd_code_with_icd9cm.add(icd_code)
#             else:
#                 missing_icd9cm += 1
#                 icd_code_without_icd9cm.add(icd_code)
        
#         # Calculate percentages
#         has_percent = (has_icd9cm / len(results_df)) * 100
#         missing_percent = (missing_icd9cm / len(results_df)) * 100
        
#         # Count unique codes with and without ICD9CM
#         unique_with_icd9cm = len(icd_code_with_icd9cm)
#         unique_without_icd9cm = len(icd_code_without_icd9cm)
#         unique_with_percent = (unique_with_icd9cm / unique_icd_codes) * 100
#         unique_without_percent = (unique_without_icd9cm / unique_icd_codes) * 100
        
#         # Print results
#         print("\n===== RESULTS =====")
#         print(f"Total entries analyzed: {len(results_df):,}")
#         print(f"Entries with ICD9CM: {has_icd9cm:,} ({has_percent:.2f}%)")
#         print(f"Entries without ICD9CM: {missing_icd9cm:,} ({missing_percent:.2f}%)")
#         print(f"\nUnique ICD-9 codes: {unique_icd_codes:,}")
#         print(f"Unique ICD-9 codes with ICD9CM: {unique_with_icd9cm:,} ({unique_with_percent:.2f}%)")
#         print(f"Unique ICD-9 codes without ICD9CM: {unique_without_icd9cm:,} ({unique_without_percent:.2f}%)")
        
#         # Create SAB distribution summary
#         all_sab_counts = defaultdict(int)
#         for _, row in results_df.iterrows():
#             sabs = row['sabs'].split(',')
#             for sab in sabs:
#                 all_sab_counts[sab] += 1
        
#         sab_summary = pd.DataFrame({
#             'source_vocabulary': list(all_sab_counts.keys()),
#             'occurrence_count': list(all_sab_counts.values())
#         }).sort_values('occurrence_count', ascending=False)
        
#         sab_summary.to_csv(os.path.join(OUTPUT_DIR, 'source_vocabulary_distribution.csv'), index=False)
        
#         print("\nTop 10 source vocabularies:")
#         print(sab_summary.head(10))
#         print(f"\nFull SAB distribution saved to: {os.path.join(OUTPUT_DIR, 'source_vocabulary_distribution.csv')}")
    
#     else:
#         print("\nERROR: No results generated after processing MRCONSO.")
        
# except Exception as e:
#     print(f"Error in SAB analysis: {str(e)}")
#     print("Some parts of the analysis may not be complete.")

# print("\n===== Analysis Complete =====")

####### ----------------------------------------------------------------------- ##############

# import pandas as pd
# import numpy as np
# import json
# import re
# import pickle
# import os
# import sys
# from tqdm import tqdm
# from collections import defaultdict

# # Paths to RRF files and output directories
# MRCONSO = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRCONSO.RRF'
# DATASET_PATH = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl'
# MAPPING_PATH = '/data/horse/ws/arsi805e-finetune/Thesis/mappings/cui_to_icd9_EXACT.json'
# OUTPUT_DIR = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/procedure_analysis'

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Helper functions for ICD-9 code formatting and classification
# def format_icd9_procedure(code: str) -> str:
#     """Format ICD-9 procedure codes to standard format with decimal points."""
#     code = re.sub(r"\s+","", str(code)).upper().rstrip(".")
#     if code.startswith("PRO_"):
#         code = code[4:]  # Remove 'PRO_' prefix
#     if not code: return ""
#     if code[0].isdigit():
#         if len(code)>2 and "." not in code: return code[:2]+"."+code[2:]
#         return code
#     return code 

# def is_procedure_code(code):
#     """Identify if a code is an ICD-9 procedure code."""
#     if code[0].isdigit():
#         if re.match(r"^[0-9]{2}\.[0-9]{1,2}$", code):
#             return True
#     return False

# def load_mrconso(path, langs=['ENG']):
#     """Load and filter the MRCONSO file from UMLS."""
#     columns = [
#         'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI',
#         'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
#     ]
#     df = pd.read_csv(path, sep='|', names=columns, dtype=str, index_col=False)
#     df = df[df['LAT'].isin(langs)]
#     return df

# # Phase 1: Procedure Code Analysis from UMLS
# print("\n===== Phase 1: Procedure Code Analysis from UMLS =====")

# print("Loading CONSO to extract procedure codes..." )
# try:
#     # Loading only the columns we need for efficiency
#     columns = ['CUI', 'LAT', 'TS', 'SAB', 'CODE', 'STR', 'SUPPRESS']
#     conso = pd.read_csv(MRCONSO, sep='|', usecols=[0, 1, 2, 11, 13, 14, 16], names=columns, dtype=str)
#     conso = conso[conso['LAT'] == 'ENG']
#     print(f"Loaded CONSO file with {len(conso):,} rows")

#     # Filter for ICD9CM procedure concepts
#     icd9_procedure_df = conso[
#         (conso["SAB"] == "ICD9CM") &
#         (conso["SUPPRESS"] != "O")
#     ]
    
#     # Apply formatting
#     icd9_procedure_df['formatted_code'] = icd9_procedure_df['CODE'].apply(format_icd9_procedure)
    
#     # Verify these are actually procedure codes
#     icd9_procedure_df['is_procedure'] = icd9_procedure_df['formatted_code'].apply(is_procedure_code)
#     icd9_procedure_df = icd9_procedure_df[icd9_procedure_df['is_procedure']]
    
#     # Report statistics
#     print(f"Total ICD-9-CM procedure codes from UMLS: {len(icd9_procedure_df):,}")
#     print(f"Unique procedure codes: {len(icd9_procedure_df['CODE'].unique()):,}")
#     print(f"Unique procedure CUIs: {len(icd9_procedure_df['CUI'].unique()):,}")
    
#     # Create a procedure code to CUI mapping
#     proc_code_to_cui = defaultdict(list)
#     for _, row in icd9_procedure_df.iterrows():
#         proc_code_to_cui[row['formatted_code']].append(row['CUI'])
    
#     print(f"Created mapping with {len(proc_code_to_cui):,} unique procedure codes")
    
# except Exception as e:
#     print(f"Error loading MRCONSO directly with pandas: {str(e)}")
#     print("Will proceed with alternative methods")
#     icd9_procedure_df = pd.DataFrame()

# # Phase 2: Dataset Procedure Code Analysis
# print("\n===== Phase 2: Dataset Procedure Code Analysis =====")
# print("Loading dataset...")
# with open(DATASET_PATH, 'rb') as f:
#     data = pickle.load(f)
# print(f"Dataset loaded with {len(data):,} records")

# # Extract and format all procedure codes from dataset
# all_pro_codes = set()
# for idx, row in data.iterrows():
#     pro_codes = row.get('pro_code', [])
#     if isinstance(pro_codes, list):
#         all_pro_codes.update(pro_codes)
#     elif isinstance(pro_codes, np.ndarray):
#         all_pro_codes.update(pro_codes)

# # Format procedure codes
# formatted_pro_codes = {format_icd9_procedure(code) for code in all_pro_codes}
# formatted_pro_codes_list = [code for code in formatted_pro_codes if code]  # Remove empty codes

# print(f"Total unique raw procedure codes in dataset: {len(all_pro_codes):,}")
# print(f"Total unique formatted procedure codes in dataset: {len(formatted_pro_codes_list):,}")

# # Check procedure codes in dataset
# print("\nSample procedure codes from dataset:")
# sample_codes = list(formatted_pro_codes)[:5]
# for code in sample_codes:
#     print(f"  {code}")

# # Check UMLS coverage if we loaded the CONSO file successfully
# if not icd9_procedure_df.empty:
#     umls_proc_codes = set(icd9_procedure_df['formatted_code'])
#     proc_coverage = set(formatted_pro_codes_list).intersection(umls_proc_codes)
    
#     print(f"\nProcedure codes in dataset covered by UMLS: {len(proc_coverage):,}/{len(formatted_pro_codes_list):,}")
#     print(f"Procedure coverage: {len(proc_coverage)/len(formatted_pro_codes_list)*100:.2f}%")

# # Phase 3: CUI Mapping for Procedure Codes
# print("\n===== Phase 3: CUI Mapping for Procedure Codes =====")

# # Try to create a mapping using our original CUI mappings first
# with open(MAPPING_PATH, 'r') as f:
#     cui_to_icd9_map = json.load(f)
# print(f"Loaded {len(cui_to_icd9_map):,} CUIs with ICD-9 mappings")

# # We'll need to reverse-map and find procedure codes
# proc_codes_with_cui = set()
# proc_codes_without_cui = set()
# proc_code_to_cui_map = {}
# proc_cui_set = set()

# # Process MRCONSO directly for procedure codes
# print("\n=== Processing MRCONSO directly for procedure codes ===")
# proc_cui_data = defaultdict(list)

# try:
#     # Read file line by line
#     with open(MRCONSO, 'r') as f:
#         for line in tqdm(f, desc="Reading MRCONSO for procedures"):
#             parts = line.strip().split('|')
#             if len(parts) >= 15:  # Ensure we have enough columns
#                 cui = parts[0]
#                 lat = parts[1]
#                 sab = parts[11]
#                 code = parts[13]
#                 str_val = parts[14]
                
#                 # Only look for ICD9CM procedure codes
#                 if sab == "ICD9CM" and lat == "ENG":
#                     formatted_code = format_icd9_procedure(code)
#                     if is_procedure_code(formatted_code) and formatted_code in formatted_pro_codes:
#                         proc_cui_data[formatted_code].append({
#                             'cui': cui,
#                             'str': str_val
#                         })
#                         # Update our tracking sets
#                         proc_codes_with_cui.add(formatted_code)
#                         proc_code_to_cui_map[formatted_code] = proc_code_to_cui_map.get(formatted_code, []) + [cui]
#                         proc_cui_set.add(cui)
    
#     # Calculate codes without mappings
#     proc_codes_without_cui = set(formatted_pro_codes_list) - proc_codes_with_cui
    
#     # Calculate coverage metrics
#     code_coverage_percent = len(proc_codes_with_cui) / len(formatted_pro_codes_list) * 100 if formatted_pro_codes_list else 0
    
#     print(f"Dataset procedure codes with CUI mappings: {len(proc_codes_with_cui):,}/{len(formatted_pro_codes_list):,} ({code_coverage_percent:.2f}%)")
#     print(f"Dataset procedure codes without CUI mappings: {len(proc_codes_without_cui):,}")
#     print(f"Total distinct CUIs referenced by procedure codes: {len(proc_cui_set):,}")
    
#     if len(proc_codes_without_cui) > 0:
#         print("\nSample procedure codes without mappings:")
#         for code in list(proc_codes_without_cui)[:5]:
#             print(f"  {code}")
    
#     # Load CUI column from CONSO to verify our CUIs
#     print("\n=== Verifying procedure CUIs in CONSO ===")
#     conso_cui = pd.read_csv(MRCONSO, sep='|', usecols=[0], names=['CUI'], dtype=str)
#     conso_cui_set = set(conso_cui['CUI'].unique())
    
#     # Check CUI existence in CONSO
#     proc_cuis_in_conso = proc_cui_set.intersection(conso_cui_set)
#     proc_cuis_not_in_conso = proc_cui_set - conso_cui_set
    
#     cui_coverage_percent = len(proc_cuis_in_conso) / len(proc_cui_set) * 100 if proc_cui_set else 0
    
#     print(f"Procedure CUIs found in CONSO: {len(proc_cuis_in_conso):,}/{len(proc_cui_set):,} ({cui_coverage_percent:.2f}%)")
#     print(f"Procedure CUIs not found in CONSO: {len(proc_cuis_not_in_conso):,}")
    
#     # Generate comprehensive mapping for procedures
#     print("\n=== Generating Comprehensive Procedure Code to CUI Mapping ===")
#     proc_code_mapping = {}
#     for code in formatted_pro_codes_list:
#         cuis = proc_code_to_cui_map.get(code, [])
#         cuis_in_conso = [cui for cui in cuis if cui in conso_cui_set]
        
#         proc_code_mapping[code] = {
#             "has_mapping": code in proc_codes_with_cui,
#             "cuis": cuis,
#             "cuis_in_conso": cuis_in_conso,
#             "all_cuis_in_conso": len(cuis) > 0 and len(cuis) == len(cuis_in_conso)
#         }
    
#     # Calculate final metrics
#     proc_codes_with_valid_cuis = sum(1 for info in proc_code_mapping.values() if info["all_cuis_in_conso"])
#     valid_mapping_percent = proc_codes_with_valid_cuis / len(formatted_pro_codes_list) * 100 if formatted_pro_codes_list else 0
    
#     print(f"Procedure codes with valid CUI mappings: {proc_codes_with_valid_cuis:,}/{len(formatted_pro_codes_list):,} ({valid_mapping_percent:.2f}%)")
    
#     # Export mappings
#     export_data = []
#     for code, info in proc_code_mapping.items():
#         export_data.append({
#             "proc_code": code,
#             "has_mapping": info["has_mapping"],
#             "cuis": ",".join(info["cuis"]) if info["cuis"] else "",
#             "cui_count": len(info["cuis"]),
#             "cuis_in_conso": ",".join(info["cuis_in_conso"]) if info["cuis_in_conso"] else "",
#             "all_cuis_in_conso": info["all_cuis_in_conso"]
#         })
    
#     proc_mapping_df = pd.DataFrame(export_data)
#     proc_mapping_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_proc_cui_mapping.csv'), index=False)
    
#     # Create a clean version for knowledge graph
#     clean_proc_mapping_df = proc_mapping_df[proc_mapping_df['all_cuis_in_conso']].copy()
#     clean_proc_mapping_df = clean_proc_mapping_df[['proc_code', 'cuis']].copy()
#     clean_proc_mapping_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_proc_cui_clean_mapping.csv'), index=False)
    
#     # Export problematic codes
#     problematic_proc_df = proc_mapping_df[~proc_mapping_df['all_cuis_in_conso']].copy()
#     problematic_proc_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_proc_problematic_codes.csv'), index=False)
    
#     print(f"\nAnalysis complete. Full procedure mapping saved to {os.path.join(OUTPUT_DIR, 'dataset_proc_cui_mapping.csv')}")
#     print(f"Clean procedure mapping for KG saved to {os.path.join(OUTPUT_DIR, 'dataset_proc_cui_clean_mapping.csv')}")
#     print(f"Problematic procedure codes saved to {os.path.join(OUTPUT_DIR, 'dataset_proc_problematic_codes.csv')}")

#     # Phase 4: SAB Analysis for Procedure Mappings
#     print("\n===== Phase 4: SAB Analysis for Procedure Mappings =====")
    
#     if len(clean_proc_mapping_df) > 0:
#         print(f"Analyzing {len(clean_proc_mapping_df)} procedure code mappings")
        
#         # Extract all unique CUIs from procedures
#         proc_all_cuis = set()
#         for cui_str in clean_proc_mapping_df['cuis']:
#             cuis = cui_str.split(',')
#             proc_all_cuis.update(cuis)
#         print(f"Found {len(proc_all_cuis):,} unique CUIs in procedure mapping")
        
#         # Process MRCONSO for procedure CUIs
#         print("Processing MRCONSO for procedure CUIs...")
#         proc_cui_sab_data = defaultdict(list)
        
#         with open(MRCONSO, 'r') as f:
#             for line in tqdm(f, desc="Reading MRCONSO"):
#                 parts = line.strip().split('|')
#                 if len(parts) >= 15:  # Ensure we have enough columns
#                     cui = parts[0]
#                     lat = parts[1]
#                     sab = parts[11]
#                     str_val = parts[14]
                    
#                     # Filter for our procedure CUIs and English terms
#                     if cui in proc_all_cuis and lat == 'ENG':
#                         proc_cui_sab_data[cui].append({
#                             'sab': sab,
#                             'str': str_val
#                         })
        
#         print(f"Found data for {len(proc_cui_sab_data):,} procedure CUIs out of {len(proc_all_cuis):,}")
        
#         # Process the data for SAB information
#         proc_cui_to_sabs = {}
#         for cui, entries in tqdm(proc_cui_sab_data.items(), desc="Processing procedure CUI data"):
#             sabs = defaultdict(list)
#             for entry in entries:
#                 sabs[entry['sab']].append(entry['str'])
            
#             unique_sabs = list(sabs.keys())
#             sab_descriptions = []
            
#             for sab in unique_sabs:
#                 # Use the first string for each SAB
#                 sab_descriptions.append(f"{sab}:{sabs[sab][0]}")
            
#             proc_cui_to_sabs[cui] = {
#                 'sabs': unique_sabs,
#                 'sab_count': len(unique_sabs),
#                 'sab_descriptions': sab_descriptions
#             }
        
#         # Create the enriched mapping for procedures
#         proc_results = []
#         for _, row in tqdm(clean_proc_mapping_df.iterrows(), desc="Creating procedure output"):
#             proc_code = row['proc_code']
#             cuis = row['cuis'].split(',')
            
#             for cui in cuis:
#                 if cui in proc_cui_to_sabs:
#                     info = proc_cui_to_sabs[cui]
#                     proc_results.append({
#                         'proc_code': proc_code,
#                         'cui': cui,
#                         'sab_count': info['sab_count'],
#                         'sabs': ','.join(info['sabs']),
#                         'sab_descriptions': '|'.join(info['sab_descriptions'])
#                     })
        
#         # Save procedure SAB results
#         proc_sab_df = pd.DataFrame(proc_results)
#         if len(proc_sab_df) > 0:
#             proc_sab_df.to_csv(os.path.join(OUTPUT_DIR, 'simple_proc_cui_sab_mapping.csv'), index=False)
#             print("\nSample procedure SAB output:")
#             print(proc_sab_df.head())
#             print(f"\nProcedure SAB mapping saved to: {os.path.join(OUTPUT_DIR, 'simple_proc_cui_sab_mapping.csv')}")
#             print(f"Total rows: {len(proc_sab_df):,}")
        
#             # Check for ICD9CM in procedure SABs
#             proc_has_icd9cm = 0
#             proc_missing_icd9cm = 0
#             proc_code_with_icd9cm = set()
#             proc_code_without_icd9cm = set()
            
#             print("Analyzing procedure SAB coverage...")
#             for idx, row in tqdm(proc_sab_df.iterrows(), total=len(proc_sab_df)):
#                 sabs = row['sabs'].split(',')
#                 proc_code = row['proc_code']
#                 cui = row['cui']
                
#                 if 'ICD9CM' in sabs:
#                     proc_has_icd9cm += 1
#                     proc_code_with_icd9cm.add(proc_code)
#                 else:
#                     proc_missing_icd9cm += 1
#                     proc_code_without_icd9cm.add(proc_code)
            
#             # Calculate percentages for procedures
#             proc_has_percent = (proc_has_icd9cm / len(proc_sab_df)) * 100 if len(proc_sab_df) > 0 else 0
#             proc_missing_percent = (proc_missing_icd9cm / len(proc_sab_df)) * 100 if len(proc_sab_df) > 0 else 0
            
#             # Count unique procedure codes with and without ICD9CM
#             unique_proc_codes = proc_sab_df['proc_code'].nunique()
#             unique_proc_with_icd9cm = len(proc_code_with_icd9cm)
#             unique_proc_without_icd9cm = len(proc_code_without_icd9cm)
#             unique_proc_with_percent = (unique_proc_with_icd9cm / unique_proc_codes) * 100 if unique_proc_codes > 0 else 0
#             unique_proc_without_percent = (unique_proc_without_icd9cm / unique_proc_codes) * 100 if unique_proc_codes > 0 else 0
            
#             # Print procedure SAB results
#             print("\n===== PROCEDURE SAB RESULTS =====")
#             print(f"Total entries analyzed: {len(proc_sab_df):,}")
#             print(f"Entries with ICD9CM: {proc_has_icd9cm:,} ({proc_has_percent:.2f}%)")
#             print(f"Entries without ICD9CM: {proc_missing_icd9cm:,} ({proc_missing_percent:.2f}%)")
#             print(f"\nUnique procedure codes: {unique_proc_codes:,}")
#             print(f"Unique procedure codes with ICD9CM: {unique_proc_with_icd9cm:,} ({unique_proc_with_percent:.2f}%)")
#             print(f"Unique procedure codes without ICD9CM: {unique_proc_without_icd9cm:,} ({unique_proc_without_percent:.2f}%)")
        
#             # Create SAB distribution summary for procedures
#             proc_sab_counts = defaultdict(int)
#             for _, row in proc_sab_df.iterrows():
#                 sabs = row['sabs'].split(',')
#                 for sab in sabs:
#                     proc_sab_counts[sab] += 1
            
#             proc_sab_summary = pd.DataFrame({
#                 'source_vocabulary': list(proc_sab_counts.keys()),
#                 'occurrence_count': list(proc_sab_counts.values())
#             }).sort_values('occurrence_count', ascending=False)
            
#             proc_sab_summary.to_csv(os.path.join(OUTPUT_DIR, 'procedure_vocabulary_distribution.csv'), index=False)
            
#             print("\nTop procedure source vocabularies:")
#             print(proc_sab_summary.head(10))
#             print(f"\nFull procedure SAB distribution saved to: {os.path.join(OUTPUT_DIR, 'procedure_vocabulary_distribution.csv')}")
#         else:
#             print("\nNo procedure SAB data found.")
#     else:
#         print("\nNo clean procedure mappings available for SAB analysis.")
        
# except Exception as e:
#     print(f"Error in procedure code analysis: {str(e)}")
#     print("Some parts of the procedure analysis may not be complete.")

# print("\n===== Procedure Code Analysis Complete =====")

######## ----------------------------------------------------------------------- ##############

# import pandas as pd
# import numpy as np
# import ast
# import pickle
# import os
# import sys
# from tqdm import tqdm
# from collections import defaultdict

# # Paths to files and output directories
# MRCONSO = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRCONSO.RRF'
# DATASET_PATH = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/mimic_with_lab_mappings.pkl'
# OUTPUT_DIR = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/lab_test_analysis'

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Helper functions for parsing and flattening lists
# def parse_lab_codes(cell):
#     """Safely parse lab test codes from various formats"""
#     if isinstance(cell, (np.ndarray, pd.Series)):
#         out = []
#         for x in cell:
#             out += parse_lab_codes(x)
#         return out
#     if cell is None:
#         return []
#     if isinstance(cell, float) and np.isnan(cell):
#         return []
#     if isinstance(cell, list):
#         return cell
#     if isinstance(cell, str):
#         cell_str = cell.strip()
#         if cell_str == '' or cell_str.lower() == 'nan':
#             return []
#         if cell_str.startswith('[') and cell_str.endswith(']'):
#             try:
#                 value = ast.literal_eval(cell_str)
#                 if isinstance(value, list):
#                     return value
#                 return [value]
#             except Exception:
#                 return [cell_str]
#         return [cell_str]
#     return [cell]

# def load_mrconso_loinc(path):
#     """Load and filter the MRCONSO file for LOINC entries"""
#     columns = [
#         'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI',
#         'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
#     ]
    
#     # Create a cache file path for faster reloading
#     cache_dir = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/cache'
#     os.makedirs(cache_dir, exist_ok=True)
#     loinc_conso_cache = os.path.join(cache_dir, 'loinc_conso_analysis_cache.pkl')
    
#     # Try to load from cache first
#     if os.path.exists(loinc_conso_cache):
#         print("Loading LOINC mapping from cache...")
#         with open(loinc_conso_cache, 'rb') as f:
#             df = pickle.load(f)
#     else:
#         print("Building LOINC mapping cache (this may take a while)...")
#         # Read MRCONSO but filter immediately to only LOINC entries
#         # Use chunking to reduce memory usage
#         chunk_size = 1000000
#         chunks = []
        
#         for chunk in tqdm(pd.read_csv(path, sep='|', names=columns, 
#                                      usecols=['CUI', 'LAT', 'SAB', 'CODE', 'STR'], 
#                                      dtype=str, chunksize=chunk_size, index_col=False)):
#             # Filter to only English LOINC entries
#             filtered = chunk[(chunk['LAT'] == 'ENG') & (chunk['SAB'] == 'LNC')]
#             if not filtered.empty:
#                 chunks.append(filtered)
        
#         df = pd.concat(chunks) if chunks else pd.DataFrame(columns=['CUI', 'LAT', 'SAB', 'CODE', 'STR'])
        
#         # Save to cache for future runs
#         with open(loinc_conso_cache, 'wb') as f:
#             pickle.dump(df, f)
    
#     return df

# # Phase 1: LOINC Analysis from UMLS
# print("\n===== Phase 1: LOINC Analysis from UMLS =====")

# print("Loading LOINC codes from MRCONSO...")
# try:
#     loinc_df = load_mrconso_loinc(MRCONSO)
#     print(f"Loaded {len(loinc_df):,} LOINC entries from MRCONSO")
    
#     # Count unique LOINC codes and CUIs
#     unique_loinc_codes = len(loinc_df['CODE'].unique())
#     unique_loinc_cuis = len(loinc_df['CUI'].unique())
#     print(f"Unique LOINC codes: {unique_loinc_codes:,}")
#     print(f"Unique CUIs for LOINC codes: {unique_loinc_cuis:,}")
    
#     # Create mapping dictionaries for faster lookups
#     loinc_to_cui_map = {}
#     loinc_to_name_map = {}
    
#     for _, row in tqdm(loinc_df.iterrows(), desc="Building LOINC mappings", total=len(loinc_df)):
#         loinc_code = row['CODE']
#         cui = row['CUI']
#         name = row['STR']
        
#         loinc_to_cui_map[loinc_code] = cui
#         if loinc_code not in loinc_to_name_map:
#             loinc_to_name_map[loinc_code] = name
    
# except Exception as e:
#     print(f"Error loading MRCONSO for LOINC: {str(e)}")
#     print("Will proceed with alternative methods")
#     loinc_df = pd.DataFrame()
#     loinc_to_cui_map = {}
#     loinc_to_name_map = {}

# # Phase 2: Dataset Lab Test Analysis
# print("\n===== Phase 2: Dataset Lab Test Analysis =====")

# print("Loading dataset...")
# try:
#     data = pd.read_pickle(DATASET_PATH)
#     print(f"Dataset loaded with {len(data):,} records")
# except Exception as e:
#     print(f"Error loading dataset: {str(e)}")
#     print("Please ensure the dataset path is correct")
#     sys.exit(1)

# # Extract all lab test LOINCs from dataset
# all_loinc_codes = set()
# all_cui_codes = set()

# print("Extracting lab test codes...")
# for idx, row in tqdm(data.iterrows(), total=len(data), desc="Extracting lab tests"):
#     loinc_codes = parse_lab_codes(row.get('lab_test_loinc', []))
#     cui_codes = parse_lab_codes(row.get('lab_test_cuis', []))
    
#     all_loinc_codes.update(loinc_codes)
#     all_cui_codes.update(cui_codes)

# print(f"Total unique LOINC codes in dataset: {len(all_loinc_codes):,}")
# print(f"Total unique CUI codes in dataset: {len(all_cui_codes):,}")

# # Sample some codes
# if all_loinc_codes:
#     print("\nSample LOINC codes from dataset:")
#     for code in list(all_loinc_codes)[:5]:
#         print(f"  {code}")

# if all_cui_codes:
#     print("\nSample CUI codes from dataset:")
#     for code in list(all_cui_codes)[:5]:
#         print(f"  {code}")

# # Check UMLS coverage for the dataset's LOINC codes
# if not loinc_df.empty:
#     umls_loinc_codes = set(loinc_df['CODE'].unique())
#     loinc_coverage = all_loinc_codes.intersection(umls_loinc_codes)
    
#     print(f"\nLOINC codes in dataset covered by UMLS: {len(loinc_coverage):,}/{len(all_loinc_codes):,}")
#     coverage_percent = (len(loinc_coverage) / len(all_loinc_codes)) * 100 if all_loinc_codes else 0
#     print(f"LOINC coverage: {coverage_percent:.2f}%")
    
#     # Find missing LOINC codes
#     missing_loinc = all_loinc_codes - umls_loinc_codes
#     if missing_loinc:
#         print(f"LOINC codes not found in UMLS: {len(missing_loinc):,}")
#         print("Sample missing LOINC codes:")
#         for code in list(missing_loinc)[:5]:
#             print(f"  {code}")

# # Phase 3: CUI Analysis for Lab Tests
# print("\n===== Phase 3: CUI Analysis for Lab Tests =====")

# # Try to load the CUI column from MRCONSO to verify our CUIs
# print("Verifying lab test CUIs in CONSO...")
# try:
#     conso_cui = pd.read_csv(MRCONSO, sep='|', usecols=[0], names=['CUI'], dtype=str)
#     conso_cui_set = set(conso_cui['CUI'].unique())
    
#     # Check CUI existence in CONSO
#     lab_cuis_in_conso = all_cui_codes.intersection(conso_cui_set)
#     lab_cuis_not_in_conso = all_cui_codes - conso_cui_set
    
#     cui_coverage_percent = (len(lab_cuis_in_conso) / len(all_cui_codes)) * 100 if all_cui_codes else 0
    
#     print(f"Lab test CUIs found in CONSO: {len(lab_cuis_in_conso):,}/{len(all_cui_codes):,} ({cui_coverage_percent:.2f}%)")
#     print(f"Lab test CUIs not found in CONSO: {len(lab_cuis_not_in_conso):,}")
    
#     if lab_cuis_not_in_conso:
#         print("\nSample CUIs not found in CONSO:")
#         for cui in list(lab_cuis_not_in_conso)[:5]:
#             print(f"  {cui}")
    
# except Exception as e:
#     print(f"Error verifying CUIs: {str(e)}")
#     print("Skipping CUI verification")
#     lab_cuis_in_conso = all_cui_codes
#     lab_cuis_not_in_conso = set()
#     conso_cui_set = all_cui_codes

# # Create mapping from LOINC to CUI
# print("\n=== Generating LOINC to CUI Mapping ===")

# # Build a mapping from dataset
# loinc_to_cui_dataset = defaultdict(set)
# records_with_both = 0

# # Count how many records have both LOINC and CUI
# for idx, row in tqdm(data.iterrows(), total=len(data), desc="Analyzing LOINC-CUI pairs"):
#     loinc_codes = parse_lab_codes(row.get('lab_test_loinc', []))
#     cui_codes = parse_lab_codes(row.get('lab_test_cuis', []))
    
#     if loinc_codes and cui_codes:
#         records_with_both += 1
#         for loinc in loinc_codes:
#             for cui in cui_codes:
#                 loinc_to_cui_dataset[loinc].add(cui)

# print(f"Records with both LOINC and CUI mappings: {records_with_both:,}")
# print(f"LOINC codes with CUI mappings in dataset: {len(loinc_to_cui_dataset):,}")

# # Generate comprehensive mapping
# loinc_code_mapping = {}
# for loinc_code in all_loinc_codes:
#     # Get CUIs from our dataset mapping
#     cuis_from_dataset = list(loinc_to_cui_dataset.get(loinc_code, set()))
    
#     # Get CUI from UMLS
#     cui_from_umls = loinc_to_cui_map.get(loinc_code)
    
#     # Combine sources, prioritizing UMLS
#     all_cuis = []
#     if cui_from_umls:
#         all_cuis.append(cui_from_umls)
#     all_cuis.extend([cui for cui in cuis_from_dataset if cui not in all_cuis])
    
#     # Filter to only those CUIs that exist in CONSO
#     cuis_in_conso = [cui for cui in all_cuis if cui in conso_cui_set]
    
#     loinc_code_mapping[loinc_code] = {
#         "has_cui_mapping": len(all_cuis) > 0,
#         "cuis": all_cuis,
#         "cuis_in_conso": cuis_in_conso,
#         "all_cuis_in_conso": len(all_cuis) > 0 and len(cuis_in_conso) == len(all_cuis)
#     }

# # Calculate final metrics
# loinc_with_valid_cuis = sum(1 for info in loinc_code_mapping.values() if info["all_cuis_in_conso"])
# valid_mapping_percent = (loinc_with_valid_cuis / len(all_loinc_codes)) * 100 if all_loinc_codes else 0

# print(f"LOINC codes with valid CUI mappings: {loinc_with_valid_cuis:,}/{len(all_loinc_codes):,} ({valid_mapping_percent:.2f}%)")

# # Export mappings
# export_data = []
# for code, info in loinc_code_mapping.items():
#     loinc_name = loinc_to_name_map.get(code, "Unknown")
    
#     export_data.append({
#         "loinc_code": code,
#         "loinc_name": loinc_name,
#         "has_cui_mapping": info["has_cui_mapping"],
#         "cuis": ",".join(info["cuis"]) if info["cuis"] else "",
#         "cui_count": len(info["cuis"]),
#         "cuis_in_conso": ",".join(info["cuis_in_conso"]) if info["cuis_in_conso"] else "",
#         "all_cuis_in_conso": info["all_cuis_in_conso"]
#     })

# loinc_mapping_df = pd.DataFrame(export_data)
# loinc_mapping_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_loinc_cui_mapping.csv'), index=False)

# # Create a clean version for knowledge graph
# clean_loinc_mapping_df = loinc_mapping_df[loinc_mapping_df['all_cuis_in_conso']].copy()
# clean_loinc_mapping_df = clean_loinc_mapping_df[['loinc_code', 'loinc_name', 'cuis']].copy()
# clean_loinc_mapping_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_loinc_cui_clean_mapping.csv'), index=False)

# # Export problematic codes
# problematic_loinc_df = loinc_mapping_df[~loinc_mapping_df['all_cuis_in_conso']].copy()
# problematic_loinc_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_loinc_problematic_codes.csv'), index=False)

# print(f"\nAnalysis complete. Full LOINC mapping saved to {os.path.join(OUTPUT_DIR, 'dataset_loinc_cui_mapping.csv')}")
# print(f"Clean LOINC mapping for KG saved to {os.path.join(OUTPUT_DIR, 'dataset_loinc_cui_clean_mapping.csv')}")
# print(f"Problematic LOINC codes saved to {os.path.join(OUTPUT_DIR, 'dataset_loinc_problematic_codes.csv')}")

# # Phase 4: SAB Analysis for Lab Test CUIs
# print("\n===== Phase 4: SAB Analysis for Lab Test CUIs =====")

# # Load the clean mapping
# try:
#     if 'clean_loinc_mapping_df' not in locals():
#         clean_loinc_path = os.path.join(OUTPUT_DIR, 'dataset_loinc_cui_clean_mapping.csv')
#         print(f"Loading clean LOINC mapping from {clean_loinc_path}")
#         clean_loinc_mapping_df = pd.read_csv(clean_loinc_path)
    
#     print(f"Analyzing {len(clean_loinc_mapping_df)} LOINC code mappings")
    
#     # Extract all unique CUIs from the mapping
#     lab_all_cuis = set()
#     for cui_str in clean_loinc_mapping_df['cuis']:
#         if isinstance(cui_str, str):
#             cuis = cui_str.split(',')
#             lab_all_cuis.update(cuis)
    
#     print(f"Found {len(lab_all_cuis):,} unique CUIs in the LOINC mapping")
    
#     # Process MRCONSO for lab CUIs
#     print("Processing MRCONSO for lab test CUIs...")
#     lab_cui_sab_data = defaultdict(list)
    
#     with open(MRCONSO, 'r') as f:
#         for line in tqdm(f, desc="Reading MRCONSO for lab tests"):
#             parts = line.strip().split('|')
#             if len(parts) >= 15:  # Ensure we have enough columns
#                 cui = parts[0]
#                 lat = parts[1]
#                 sab = parts[11]
#                 str_val = parts[14]
                
#                 # Filter for our lab CUIs and English terms
#                 if cui in lab_all_cuis and lat == 'ENG':
#                     lab_cui_sab_data[cui].append({
#                         'sab': sab,
#                         'str': str_val
#                     })
    
#     print(f"Found data for {len(lab_cui_sab_data):,} lab CUIs out of {len(lab_all_cuis):,} in mapping")
    
#     # Process the data for SAB information
#     lab_cui_to_sabs = {}
#     for cui, entries in tqdm(lab_cui_sab_data.items(), desc="Processing lab CUI data"):
#         sabs = defaultdict(list)
#         for entry in entries:
#             sabs[entry['sab']].append(entry['str'])
        
#         unique_sabs = list(sabs.keys())
#         sab_descriptions = []
        
#         for sab in unique_sabs:
#             # Use the first string for each SAB
#             sab_descriptions.append(f"{sab}:{sabs[sab][0]}")
        
#         lab_cui_to_sabs[cui] = {
#             'sabs': unique_sabs,
#             'sab_count': len(unique_sabs),
#             'sab_descriptions': sab_descriptions
#         }
    
#     # Create the enriched mapping for lab tests
#     lab_results = []
#     for _, row in tqdm(clean_loinc_mapping_df.iterrows(), desc="Creating lab test output"):
#         loinc_code = row['loinc_code']
#         loinc_name = row['loinc_name']
        
#         if isinstance(row['cuis'], str):
#             cuis = row['cuis'].split(',')
            
#             for cui in cuis:
#                 if cui in lab_cui_to_sabs:
#                     info = lab_cui_to_sabs[cui]
#                     lab_results.append({
#                         'loinc_code': loinc_code,
#                         'loinc_name': loinc_name,
#                         'cui': cui,
#                         'sab_count': info['sab_count'],
#                         'sabs': ','.join(info['sabs']),
#                         'sab_descriptions': '|'.join(info['sab_descriptions'])
#                     })
    
#     # Save lab SAB results
#     lab_sab_df = pd.DataFrame(lab_results)
#     if len(lab_sab_df) > 0:
#         lab_sab_df.to_csv(os.path.join(OUTPUT_DIR, 'simple_loinc_cui_sab_mapping.csv'), index=False)
#         print("\nSample lab SAB output:")
#         print(lab_sab_df.head())
#         print(f"\nLab SAB mapping saved to: {os.path.join(OUTPUT_DIR, 'simple_loinc_cui_sab_mapping.csv')}")
#         print(f"Total rows: {len(lab_sab_df):,}")
    
#         # Check for LNC in lab SABs
#         has_lnc = 0
#         missing_lnc = 0
#         loinc_code_with_lnc = set()
#         loinc_code_without_lnc = set()
        
#         print("Analyzing lab SAB coverage...")
#         for idx, row in tqdm(lab_sab_df.iterrows(), total=len(lab_sab_df)):
#             sabs = row['sabs'].split(',')
#             loinc_code = row['loinc_code']
#             cui = row['cui']
            
#             if 'LNC' in sabs:
#                 has_lnc += 1
#                 loinc_code_with_lnc.add(loinc_code)
#             else:
#                 missing_lnc += 1
#                 loinc_code_without_lnc.add(loinc_code)
        
#         # Calculate percentages for lab tests
#         has_percent = (has_lnc / len(lab_sab_df)) * 100
#         missing_percent = (missing_lnc / len(lab_sab_df)) * 100
        
#         # Count unique LOINC codes with and without LNC
#         unique_loinc_codes = lab_sab_df['loinc_code'].nunique()
#         unique_with_lnc = len(loinc_code_with_lnc)
#         unique_without_lnc = len(loinc_code_without_lnc)
#         unique_with_percent = (unique_with_lnc / unique_loinc_codes) * 100
#         unique_without_percent = (unique_without_lnc / unique_loinc_codes) * 100
        
#         # Print lab SAB results
#         print("\n===== LAB TEST SAB RESULTS =====")
#         print(f"Total entries analyzed: {len(lab_sab_df):,}")
#         print(f"Entries with LNC: {has_lnc:,} ({has_percent:.2f}%)")
#         print(f"Entries without LNC: {missing_lnc:,} ({missing_percent:.2f}%)")
#         print(f"\nUnique LOINC codes: {unique_loinc_codes:,}")
#         print(f"Unique LOINC codes with LNC: {unique_with_lnc:,} ({unique_with_percent:.2f}%)")
#         print(f"Unique LOINC codes without LNC: {unique_without_lnc:,} ({unique_without_percent:.2f}%)")
    
#         # Create SAB distribution summary for lab tests
#         lab_sab_counts = defaultdict(int)
#         for _, row in lab_sab_df.iterrows():
#             sabs = row['sabs'].split(',')
#             for sab in sabs:
#                 lab_sab_counts[sab] += 1
        
#         lab_sab_summary = pd.DataFrame({
#             'source_vocabulary': list(lab_sab_counts.keys()),
#             'occurrence_count': list(lab_sab_counts.values())
#         }).sort_values('occurrence_count', ascending=False)
        
#         lab_sab_summary.to_csv(os.path.join(OUTPUT_DIR, 'lab_vocabulary_distribution.csv'), index=False)
        
#         print("\nTop lab source vocabularies:")
#         print(lab_sab_summary.head(10))
#         print(f"\nFull lab SAB distribution saved to: {os.path.join(OUTPUT_DIR, 'lab_vocabulary_distribution.csv')}")
#     else:
#         print("\nNo lab SAB data found.")

# except Exception as e:
#     print(f"Error in lab SAB analysis: {str(e)}")
#     print("Some parts of the lab analysis may not be complete.")

# print("\n===== Lab Test Analysis Complete =====")

######## ----------------------------------------------------------------------- ##############

import pandas as pd
import numpy as np
import ast
import pickle
import os
import sys
from tqdm import tqdm
from collections import defaultdict

# Paths to files and output directories
MRCONSO = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRCONSO.RRF'
DATASET_PATH = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/merged_icd9.pkl'
OUTPUT_DIR = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/med_analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper functions for parsing and flattening lists
def parse_atc_codes(cell):
    """Safely parse ATC codes from various formats"""
    if isinstance(cell, (np.ndarray, pd.Series)):
        out = []
        for x in cell:
            out += parse_atc_codes(x)
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

def load_mrconso_atc(path):
    """Load and filter the MRCONSO file for ATC entries"""
    columns = [
        'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI',
        'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
    ]
    
    # Create a cache file path for faster reloading
    cache_dir = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/cache'
    os.makedirs(cache_dir, exist_ok=True)
    atc_conso_cache = os.path.join(cache_dir, 'atc_conso_analysis_cache.pkl')
    
    # Try to load from cache first
    if os.path.exists(atc_conso_cache):
        print("Loading ATC mapping from cache...")
        with open(atc_conso_cache, 'rb') as f:
            df = pickle.load(f)
    else:
        print("Building ATC mapping cache (this may take a while)...")
        # Read MRCONSO but filter immediately to only ATC entries
        # Use chunking to reduce memory usage
        chunk_size = 1000000
        chunks = []
        
        for chunk in tqdm(pd.read_csv(path, sep='|', names=columns, 
                                     usecols=['CUI', 'LAT', 'SAB', 'CODE', 'STR'], 
                                     dtype=str, chunksize=chunk_size, index_col=False)):
            # Filter to only English ATC entries
            filtered = chunk[(chunk['LAT'] == 'ENG') & (chunk['SAB'] == 'ATC')]
            if not filtered.empty:
                chunks.append(filtered)
        
        df = pd.concat(chunks) if chunks else pd.DataFrame(columns=['CUI', 'LAT', 'SAB', 'CODE', 'STR'])
        
        # Save to cache for future runs
        with open(atc_conso_cache, 'wb') as f:
            pickle.dump(df, f)
    
    return df

# Phase 1: ATC Analysis from UMLS
print("\n===== Phase 1: ATC Analysis from UMLS =====")

print("Loading ATC codes from MRCONSO...")
try:
    atc_df = load_mrconso_atc(MRCONSO)
    print(f"Loaded {len(atc_df):,} ATC entries from MRCONSO")
    
    # Count unique ATC codes and CUIs
    unique_atc_codes = len(atc_df['CODE'].unique())
    unique_atc_cuis = len(atc_df['CUI'].unique())
    print(f"Unique ATC codes: {unique_atc_codes:,}")
    print(f"Unique CUIs for ATC codes: {unique_atc_cuis:,}")
    
    # Create mapping dictionaries for faster lookups
    atc_to_cui_map = {}
    atc_to_name_map = {}
    
    for _, row in tqdm(atc_df.iterrows(), desc="Building ATC mappings", total=len(atc_df)):
        atc_code = row['CODE']
        cui = row['CUI']
        name = row['STR']
        
        atc_to_cui_map[atc_code] = cui
        if atc_code not in atc_to_name_map:
            atc_to_name_map[atc_code] = name
    
except Exception as e:
    print(f"Error loading MRCONSO for ATC: {str(e)}")
    print("Will proceed with alternative methods")
    atc_df = pd.DataFrame()
    atc_to_cui_map = {}
    atc_to_name_map = {}

# Phase 2: Dataset Medication Analysis
print("\n===== Phase 2: Dataset Medication Analysis =====")

print("Loading dataset...")
try:
    data = pd.read_pickle(DATASET_PATH)
    print(f"Dataset loaded with {len(data):,} records")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    print("Please ensure the dataset path is correct")
    sys.exit(1)

# Extract all ATC codes directly from dataset (ndc column contains ATC codes)
all_atc_codes = set()

print("Extracting ATC codes from ndc column...")
for idx, row in tqdm(data.iterrows(), total=len(data), desc="Extracting medications"):
    atc_codes = parse_atc_codes(row.get('ndc', []))
    all_atc_codes.update(atc_codes)

print(f"Total unique ATC codes in dataset: {len(all_atc_codes):,}")

# Sample some codes
if all_atc_codes:
    print("\nSample ATC codes from dataset:")
    for code in list(all_atc_codes)[:5]:
        print(f"  {code}")

# Check UMLS coverage for the dataset's ATC codes
if not atc_df.empty:
    umls_atc_codes = set(atc_df['CODE'].unique())
    atc_coverage = all_atc_codes.intersection(umls_atc_codes)
    
    coverage_percent = (len(atc_coverage) / len(all_atc_codes)) * 100 if all_atc_codes else 0
    print(f"\nATC codes in dataset covered by UMLS: {len(atc_coverage):,}/{len(all_atc_codes):,} ({coverage_percent:.2f}%)")
    
    # Find missing ATC codes
    missing_atc = all_atc_codes - umls_atc_codes
    if missing_atc:
        print(f"ATC codes not found in UMLS: {len(missing_atc):,}")
        print("Sample missing ATC codes:")
        for code in list(missing_atc)[:5]:
            print(f"  {code}")

# Phase 3: CUI Analysis for Medications
print("\n===== Phase 3: CUI Analysis for Medications =====")

# Load all CUIs from UMLS for verification
print("Verifying medication CUIs in CONSO...")
try:
    conso_cui = pd.read_csv(MRCONSO, sep='|', usecols=[0], names=['CUI'], dtype=str)
    conso_cui_set = set(conso_cui['CUI'].unique())
    print(f"Loaded {len(conso_cui_set):,} unique CUIs from UMLS")
except Exception as e:
    print(f"Error loading CUIs: {str(e)}")
    print("Creating a placeholder CUI set")
    conso_cui_set = set()

# Get CUIs for ATC codes
med_cuis = set()
atc_code_to_cui = {}

# Use the ATC to CUI mapping from UMLS
if atc_to_cui_map:
    print("\n=== Using ATC to CUI mapping from UMLS ===")
    
    for atc in all_atc_codes:
        cui = atc_to_cui_map.get(atc)
        if cui:
            med_cuis.add(cui)
            atc_code_to_cui[atc] = [cui]
    
    print(f"Found CUIs for {len(atc_code_to_cui):,} out of {len(all_atc_codes):,} ATC codes")

# Check CUI existence in CONSO
med_cuis_in_conso = med_cuis.intersection(conso_cui_set)
med_cuis_not_in_conso = med_cuis - conso_cui_set

cui_coverage_percent = (len(med_cuis_in_conso) / len(med_cuis)) * 100 if med_cuis else 0

print(f"Medication CUIs found in CONSO: {len(med_cuis_in_conso):,}/{len(med_cuis):,} ({cui_coverage_percent:.2f}%)")
print(f"Medication CUIs not found in CONSO: {len(med_cuis_not_in_conso):,}")

if med_cuis_not_in_conso:
    print("\nSample CUIs not found in CONSO:")
    for cui in list(med_cuis_not_in_conso)[:5]:
        print(f"  {cui}")

# Generate comprehensive mapping
print("\n=== Generating Comprehensive ATC Code to CUI Mapping ===")
med_code_mapping = {}

# Process ATC codes
for atc in all_atc_codes:
    cuis = atc_code_to_cui.get(atc, [])
    cuis_in_conso = [cui for cui in cuis if cui in conso_cui_set]
    
    med_code_mapping[atc] = {
        "has_cui_mapping": len(cuis) > 0,
        "cuis": cuis,
        "cuis_in_conso": cuis_in_conso,
        "all_cuis_in_conso": len(cuis) > 0 and len(cuis_in_conso) == len(cuis)
    }

# Calculate final metrics
meds_with_valid_cuis = sum(1 for info in med_code_mapping.values() if info["all_cuis_in_conso"])
valid_mapping_percent = (meds_with_valid_cuis / len(all_atc_codes)) * 100 if all_atc_codes else 0

print(f"ATC codes with valid CUI mappings: {meds_with_valid_cuis:,}/{len(all_atc_codes):,} ({valid_mapping_percent:.2f}%)")

# Export mappings
export_data = []
for code, info in med_code_mapping.items():
    atc_name = atc_to_name_map.get(code, "Unknown")
    
    export_data.append({
        "atc_code": code,
        "atc_name": atc_name,
        "has_cui_mapping": info["has_cui_mapping"],
        "cuis": ",".join(info["cuis"]) if info["cuis"] else "",
        "cui_count": len(info["cuis"]),
        "cuis_in_conso": ",".join(info["cuis_in_conso"]) if info["cuis_in_conso"] else "",
        "all_cuis_in_conso": info["all_cuis_in_conso"]
    })

med_mapping_df = pd.DataFrame(export_data)
med_mapping_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_med_cui_mapping.csv'), index=False)

# Create a clean version for knowledge graph
clean_med_mapping_df = med_mapping_df[med_mapping_df['all_cuis_in_conso']].copy()
clean_med_mapping_df = clean_med_mapping_df[['atc_code', 'atc_name', 'cuis']].copy()
clean_med_mapping_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_med_cui_clean_mapping.csv'), index=False)

# Export problematic codes
problematic_med_df = med_mapping_df[~med_mapping_df['all_cuis_in_conso']].copy()
problematic_med_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_med_problematic_codes.csv'), index=False)

print(f"\nAnalysis complete. Full medication mapping saved to {os.path.join(OUTPUT_DIR, 'dataset_med_cui_mapping.csv')}")
print(f"Clean medication mapping for KG saved to {os.path.join(OUTPUT_DIR, 'dataset_med_cui_clean_mapping.csv')}")
print(f"Problematic medication codes saved to {os.path.join(OUTPUT_DIR, 'dataset_med_problematic_codes.csv')}")

# Phase 4: SAB Analysis for Medication CUIs
print("\n===== Phase 4: SAB Analysis for Medication CUIs =====")

# Load the clean mapping
try:
    if 'clean_med_mapping_df' not in locals():
        clean_med_path = os.path.join(OUTPUT_DIR, 'dataset_med_cui_clean_mapping.csv')
        print(f"Loading clean medication mapping from {clean_med_path}")
        clean_med_mapping_df = pd.read_csv(clean_med_path)
    
    print(f"Analyzing {len(clean_med_mapping_df)} medication code mappings")
    
    # Extract all unique CUIs from the mapping
    med_all_cuis = set()
    for cui_str in clean_med_mapping_df['cuis']:
        if isinstance(cui_str, str):
            cuis = cui_str.split(',')
            med_all_cuis.update(cuis)
    
    print(f"Found {len(med_all_cuis):,} unique CUIs in the medication mapping")
    
    # Process MRCONSO for medication CUIs
    print("Processing MRCONSO for medication CUIs...")
    med_cui_sab_data = defaultdict(list)
    
    with open(MRCONSO, 'r') as f:
        for line in tqdm(f, desc="Reading MRCONSO for medications"):
            parts = line.strip().split('|')
            if len(parts) >= 15:  # Ensure we have enough columns
                cui = parts[0]
                lat = parts[1]
                sab = parts[11]
                str_val = parts[14]
                
                # Filter for our medication CUIs and English terms
                if cui in med_all_cuis and lat == 'ENG':
                    med_cui_sab_data[cui].append({
                        'sab': sab,
                        'str': str_val
                    })
    
    print(f"Found data for {len(med_cui_sab_data):,} medication CUIs out of {len(med_all_cuis):,} in mapping")
    
    # Process the data for SAB information
    med_cui_to_sabs = {}
    for cui, entries in tqdm(med_cui_sab_data.items(), desc="Processing medication CUI data"):
        sabs = defaultdict(list)
        for entry in entries:
            sabs[entry['sab']].append(entry['str'])
        
        unique_sabs = list(sabs.keys())
        sab_descriptions = []
        
        for sab in unique_sabs:
            # Use the first string for each SAB
            sab_descriptions.append(f"{sab}:{sabs[sab][0]}")
        
        med_cui_to_sabs[cui] = {
            'sabs': unique_sabs,
            'sab_count': len(unique_sabs),
            'sab_descriptions': sab_descriptions
        }
    
    # Create the enriched mapping for medications
    med_results = []
    for _, row in tqdm(clean_med_mapping_df.iterrows(), desc="Creating medication output"):
        atc_code = row['atc_code']
        atc_name = row['atc_name']
        
        if isinstance(row['cuis'], str):
            cuis = row['cuis'].split(',')
            
            for cui in cuis:
                if cui in med_cui_to_sabs:
                    info = med_cui_to_sabs[cui]
                    med_results.append({
                        'atc_code': atc_code,
                        'atc_name': atc_name,
                        'cui': cui,
                        'sab_count': info['sab_count'],
                        'sabs': ','.join(info['sabs']),
                        'sab_descriptions': '|'.join(info['sab_descriptions'])
                    })
    
    # Save medication SAB results
    med_sab_df = pd.DataFrame(med_results)
    if len(med_sab_df) > 0:
        med_sab_df.to_csv(os.path.join(OUTPUT_DIR, 'simple_med_cui_sab_mapping.csv'), index=False)
        print("\nSample medication SAB output:")
        print(med_sab_df.head())
        print(f"\nMedication SAB mapping saved to: {os.path.join(OUTPUT_DIR, 'simple_med_cui_sab_mapping.csv')}")
        print(f"Total rows: {len(med_sab_df):,}")
    
        # Check for ATC in medication SABs
        has_atc = 0
        missing_atc = 0
        med_code_with_atc = set()
        med_code_without_atc = set()
        
        print("Analyzing medication SAB coverage...")
        for idx, row in tqdm(med_sab_df.iterrows(), total=len(med_sab_df)):
            sabs = row['sabs'].split(',')
            atc_code = row['atc_code']
            cui = row['cui']
            
            if 'ATC' in sabs:
                has_atc += 1
                med_code_with_atc.add(atc_code)
            else:
                missing_atc += 1
                med_code_without_atc.add(atc_code)
        
        # Calculate percentages for medications
        has_percent = (has_atc / len(med_sab_df)) * 100
        missing_percent = (missing_atc / len(med_sab_df)) * 100
        
        # Count unique ATC codes with and without ATC
        unique_atc_codes = med_sab_df['atc_code'].nunique()
        unique_with_atc = len(med_code_with_atc)
        unique_without_atc = len(med_code_without_atc)
        unique_with_percent = (unique_with_atc / unique_atc_codes) * 100
        unique_without_percent = (unique_without_atc / unique_atc_codes) * 100
        
        # Print medication SAB results
        print("\n===== MEDICATION SAB RESULTS =====")
        print(f"Total entries analyzed: {len(med_sab_df):,}")
        print(f"Entries with ATC: {has_atc:,} ({has_percent:.2f}%)")
        print(f"Entries without ATC: {missing_atc:,} ({missing_percent:.2f}%)")
        print(f"\nUnique ATC codes: {unique_atc_codes:,}")
        print(f"Unique ATC codes with ATC SAB: {unique_with_atc:,} ({unique_with_percent:.2f}%)")
        print(f"Unique ATC codes without ATC SAB: {unique_without_atc:,} ({unique_without_percent:.2f}%)")
    
        # Create SAB distribution summary for medications
        med_sab_counts = defaultdict(int)
        for _, row in med_sab_df.iterrows():
            sabs = row['sabs'].split(',')
            for sab in sabs:
                med_sab_counts[sab] += 1
        
        med_sab_summary = pd.DataFrame({
            'source_vocabulary': list(med_sab_counts.keys()),
            'occurrence_count': list(med_sab_counts.values())
        }).sort_values('occurrence_count', ascending=False)
        
        med_sab_summary.to_csv(os.path.join(OUTPUT_DIR, 'medication_vocabulary_distribution.csv'), index=False)
        
        print("\nTop medication source vocabularies:")
        print(med_sab_summary.head(10))
        print(f"\nFull medication SAB distribution saved to: {os.path.join(OUTPUT_DIR, 'medication_vocabulary_distribution.csv')}")
    else:
        print("\nNo medication SAB data found.")

except Exception as e:
    print(f"Error in medication SAB analysis: {str(e)}")
    print("Some parts of the medication analysis may not be complete.")

print("\n===== Medication Analysis Complete =====")