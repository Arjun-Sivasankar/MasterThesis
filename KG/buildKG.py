import pandas as pd
import networkx as nx
from tqdm import tqdm
import os

# Paths to RRF files (customize)
# Update paths to match your actual directory structure
MRCONSO = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRCONSO.RRF'
MRREL = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRREL.RRF'
MRSTY = '/data/horse/ws/arsi805e-finetune/Thesis/UMLS/2025AA/META/MRSTY.RRF'

DATA_DIR = 'KG/data_files'
os.makedirs(DATA_DIR, exist_ok=True)

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

# 2. Load UMLS Relationships
def load_mrrel(path):
    columns = [
        'CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2', 'STYPE2', 'RELA', 'RUI', 'SRUI',
        'SAB', 'SL', 'RG', 'DIR', 'SUPPRESS', 'CVF'
    ]
    return pd.read_csv(path, sep='|', names=columns, dtype=str, index_col=False)

print("Loading CONSO..." )
conso = load_mrconso(MRCONSO)
print("Loaded CONSO file.....")

print("Loading REL..." )
rel = load_mrrel(MRREL)
print("Loaded REL file.....")

print(rel.head())
print(rel.shape)
print(rel.info())

## Vocabs in UMLS relationships:
print("Vocabularies in UMLS relationships:")
print(rel.SAB.unique())

## conso info:
print("CONSO info:")
print(conso.info())
print(conso.head())
print(conso.shape)

## SNOMED in conso:
print("SNOMED in CONSO:")
print(conso[conso.SAB=='SNOMEDCT_US'])

## -----------------------------------------
## ICD-10-CM:
icd10_concept_df = conso[
    (conso["TS"] == "P") &
    (conso["SAB"] == "ICD10CM") &
    (conso["SUPPRESS"] != "O")
]

print(f"ICD-10-CM concepts: {icd10_concept_df.shape[0]}")
print(icd10_concept_df.head())

# Concepts that map to ranges of ICD10CM codes are often too broad and are omitted
icd10_concept_df = icd10_concept_df.loc[~icd10_concept_df['CODE'].str.contains('-')]

# only keep codes with 6 or less characters (two places after decimal point)
icd10_concept_df = icd10_concept_df.assign(len_icd=icd10_concept_df['CODE'].apply(lambda x: len(x)))
icd10_concept_df = icd10_concept_df.loc[icd10_concept_df['len_icd'] <= 6+1] # 7 bc of decimal points

n = len(icd10_concept_df)
print("Number of rows in ICD10CM concept df: %d" % n)

n = len(icd10_concept_df['CUI'].unique())
print("Number of CUI in ICD10CM concept df: %d" % n)

n = len(icd10_concept_df['CODE'].unique())
print("Number of codes in ICD10CM concept df: %d" % n)

print(icd10_concept_df.head())
print("-"*50)

## ICD-9-CM:
icd9_concept_df = conso[
    (conso["TS"] == "P") &
    (conso["SAB"] == "ICD9CM") &
    (conso["SUPPRESS"] != "O")
]

print(f"ICD-9-CM concepts: {icd9_concept_df.shape[0]}")
print(icd9_concept_df.head())

# Concepts that map to ranges of ICD10CM codes are often too broad and are omitted
icd9_concept_df = icd9_concept_df.loc[~icd9_concept_df['CODE'].str.contains('-')]

# remove ICDs < 1 (procedures)
icd9_concept_df = icd9_concept_df.loc[~icd9_concept_df['CODE'].str.startswith('00.')]

n = len(icd9_concept_df)
print("Number of rows in ICD10CM concept df: %d" % n)

n = len(icd9_concept_df['CUI'].unique())
print("Number of CUI in ICD10CM concept df: %d" % n)

n = len(icd9_concept_df['CODE'].unique())
print("Number of codes in ICD10CM concept df: %d" % n)

print(icd9_concept_df.head())
print("-"*50)

## SNOMED:
snomed_concept_df = conso[
    (conso["TS"] == "P") &
    (conso["SAB"] == "SNOMEDCT_US") &
    (conso["SUPPRESS"] != "O")
]

print(f"SNOMED concepts before filtering: {snomed_concept_df.shape[0]}")

# Import SNOMED Core Subset and KP List for filtering SNOMED concepts
try:
    core_df = pd.read_csv("/data/horse/ws/arsi805e-finetune/Thesis/SNOMEDCT_CORE_SUBSET_202506.txt", sep='|')
    core_df = core_df.loc[core_df['SNOMED_CONCEPT_STATUS'] == 'Current']
    core_df = core_df.assign(SNOMED_CID=core_df['SNOMED_CID'].astype(str))
    
    kp_df = pd.read_csv("/data/horse/ws/arsi805e-finetune/Thesis/KPList.txt", sep='\t', encoding='latin-1')
    kp_df['SCTID'] = kp_df['SCTID'].astype(str)
    
    # Filter SNOMED concepts by core subset and KP list
    snomed_concept_df = snomed_concept_df.loc[snomed_concept_df['CODE'].isin(
        core_df['SNOMED_CID'].tolist() + kp_df['SCTID'].tolist())]
    
    print(f"SNOMED concepts after filtering: {snomed_concept_df.shape[0]}")
except Exception as e:
    print(f"Warning: Could not filter SNOMED concepts using core subset and KP list: {e}")
    print("Using all SNOMED concepts instead.")

n = len(snomed_concept_df)
print("Number of rows in SNOMED concept df: %d" % n)

n = len(snomed_concept_df['CUI'].unique())
print("Number of CUI in SNOMED concept df: %d" % n)

n = len(snomed_concept_df['CODE'].unique())
print("Number of codes in SNOMED concept df: %d" % n)

print(snomed_concept_df.head())
print("-"*50)

## ATC (Anatomical Therapeutic Chemical Classification):
atc_concept_df = conso[
    (conso["TS"] == "P") &
    (conso["SAB"] == "ATC") &
    (conso["SUPPRESS"] != "O")
]

print(f"ATC concepts before filtering: {atc_concept_df.shape[0]}")

# Only keep ATC4 and ATC5 levels (medication classes and specific drugs)
atc_concept_df = atc_concept_df.assign(len_atc=atc_concept_df['CODE'].apply(len))
atc_concept_df = atc_concept_df.loc[atc_concept_df['len_atc'] >= 4]

n = len(atc_concept_df)
print("Number of rows in ATC concept df: %d" % n)

n = len(atc_concept_df['CUI'].unique())
print("Number of CUI in ATC concept df: %d" % n)

n = len(atc_concept_df['CODE'].unique())
print("Number of codes in ATC concept df: %d" % n)

print(atc_concept_df.head())
print("-"*50)

## LOINC (Laboratory tests):
lnc_concept_df = conso[
    (conso["TS"] == "P") &
    (conso["SAB"] == "LNC") &
    (conso["SUPPRESS"] != "O")
]

print(f"LOINC concepts before filtering: {lnc_concept_df.shape[0]}")

# Try to filter by LOINC parts
try:
    loinc_part_df = pd.read_csv("/data/horse/ws/arsi805e-finetune/Thesis/Part.csv")
    loinc_part_df = loinc_part_df.loc[loinc_part_df['Status'] == 'ACTIVE']
    
    n = len(loinc_part_df)
    print("Number of LOINC parts before filtering by part name: %d" % n)
    
    # Filter out less useful part types
    loinc_part_df = loinc_part_df.loc[~loinc_part_df['PartTypeName'].isin([
        'ADJUSTMENT', 'CHALLENGE', 'COUNT', 'PROPERTY', 'SCALE', 'SUPER SYSTEM', 'TIME', 'TIME MODIFIER'])]
    
    n = len(loinc_part_df)
    print("Number of LOINC parts after filtering: %d" % n)
    
    # Filter concept df by loinc part numbers
    lnc_concept_df = lnc_concept_df.loc[lnc_concept_df['CODE'].isin(loinc_part_df['PartNumber'])]
except Exception as e:
    print(f"Warning: Could not filter LOINC concepts using parts file: {e}")
    print("Using all LOINC concepts instead.")

# Try to add LOINC codes from MIMIC data
try:
    import pickle
    with open('mimic.pkl', 'rb') as f:
        mimic = pickle.load(f)
    
    # Pool all LOINC codes into one flat set
    all_loinc = set()
    for loinc_list in mimic['lab_test_loinc']:
        all_loinc.update(loinc_list)
    
    if 'nan' in all_loinc:
        all_loinc.remove('nan')
        
    print(f"Unique LOINC codes in MIMIC data: {len(all_loinc)}")
    
    # Map LOINC codes to CUIs
    conso_lnc = conso[conso['SAB'] == 'LNC']
    loinc_to_cui = {}
    for code in all_loinc:
        row = conso_lnc[conso_lnc['CODE'] == code]
        if not row.empty:
            cui = row.iloc[0]['CUI']
            loinc_to_cui[code] = cui
        else:
            loinc_to_cui[code] = None
            
    valid_lab_cuis = set(cui for cui in loinc_to_cui.values() if cui is not None)
    print(f"Total valid lab CUIs mapped from MIMIC: {len(valid_lab_cuis)}")
    
    # Add these concepts
    extra_conso_lnc = conso_lnc[conso_lnc['CUI'].isin(valid_lab_cuis)]
    merged_lnc_concept = pd.concat([lnc_concept_df, extra_conso_lnc], ignore_index=True)
    merged_lnc_concept = merged_lnc_concept.drop_duplicates(subset=['CUI', 'CODE'])
    lnc_concept_df = merged_lnc_concept
except Exception as e:
    print(f"Warning: Could not add MIMIC LOINC concepts: {e}")

n = len(lnc_concept_df)
print("Number of rows in LOINC concept df: %d" % n)

n = len(lnc_concept_df['CUI'].unique())
print("Number of CUI in LOINC concept df: %d" % n)

n = len(lnc_concept_df['CODE'].unique())
print("Number of codes in LOINC concept df: %d" % n)

print(lnc_concept_df.head())
print("-"*50)

## CPT (Current Procedural Terminology):
cpt_concept_df = conso[
    (conso["TS"] == "P") &
    (conso["SAB"] == "CPT") &
    (conso["SUPPRESS"] != "O")
]

print(f"CPT concepts before filtering: {cpt_concept_df.shape[0]}")

# Category I only: 00100–99499 (main procedure codes)
cpt_concept_df['CODE_INT'] = pd.to_numeric(cpt_concept_df['CODE'], errors='coerce')
cpt_concept_df = cpt_concept_df.loc[cpt_concept_df['CODE_INT'].between(100, 99499)]

# There are many CUIs mapped to individual CPT codes due to many detailed atoms
# To reduce excessive redundancies, we limit to only the preferred term ('PT')
cpt_concept_df = cpt_concept_df.loc[cpt_concept_df['TTY'] == 'PT']

n = len(cpt_concept_df)
print("Number of rows in CPT concept df: %d" % n)

n = len(cpt_concept_df['CUI'].unique())
print("Number of CUI in CPT concept df: %d" % n)

n = len(cpt_concept_df['CODE'].unique())
print("Number of codes in CPT concept df: %d" % n)

print(cpt_concept_df.head())
print("-"*50)

## ICD10PCS (Procedure Coding System):
icd10pcs_concept_df = conso[
    (conso["TS"] == "P") &
    (conso["SAB"] == "ICD10PCS") &
    (conso["SUPPRESS"] != "O")
]

print(f"ICD10PCS concepts before filtering: {icd10pcs_concept_df.shape[0]}")

# Try to filter using ICD9-to-ICD10PCS mapping
try:
    icd9_icd10_map = pd.read_csv('icd9toicd10pcsgem.csv')
    icd10pcs_codes = icd9_icd10_map['icd10cm'].unique()
    print(f'No. of unique ICD10PCS codes in map: {len(icd10pcs_codes)}')
    
    icd10pcs_concept_df = icd10pcs_concept_df[icd10pcs_concept_df['CODE'].isin(icd10pcs_codes)]
    icd10pcs_concept_df = icd10pcs_concept_df.drop_duplicates(subset=['CUI', 'CODE', 'STR'])
except Exception as e:
    print(f"Warning: Could not filter ICD10PCS concepts using mapping: {e}")
    print("Using all ICD10PCS concepts instead.")

n = len(icd10pcs_concept_df)
print("Number of rows in ICD10PCS concept df: %d" % n)

n = len(icd10pcs_concept_df['CUI'].unique())
print("Number of CUI in ICD10PCS concept df: %d" % n)

n = len(icd10pcs_concept_df['CODE'].unique())
print("Number of codes in ICD10PCS concept df: %d" % n)

print(icd10pcs_concept_df.head())
print("-"*50)

# Combine all CUIs from your vocabularies
all_cuis = set(icd9_concept_df['CUI']).union(
    icd10_concept_df['CUI']).union(
    snomed_concept_df['CUI']).union(
    atc_concept_df['CUI']).union(
    lnc_concept_df['CUI']).union(
    cpt_concept_df['CUI']).union(
    icd10pcs_concept_df['CUI'])

print("Total unique CUIs across all vocabularies: ", len(all_cuis))

# Create a combined concept dataframe
concept_df = pd.concat([
    icd9_concept_df, 
    icd10_concept_df, 
    snomed_concept_df, 
    lnc_concept_df, 
    atc_concept_df, 
    cpt_concept_df, 
    icd10pcs_concept_df
])

print("Combined concept dataframe shape: ", concept_df.shape)

# Filter relationships to include only those between our concepts
rel_edges = rel[(rel['CUI1'].isin(all_cuis)) & (rel['CUI2'].isin(all_cuis))]
print("Relationship edges after filtering by CUIs: ", rel_edges.shape)

# Further filter by source vocabularies
sab_vocabs = ['CPT', 'LNC', 'ATC', 'SNOMEDCT_US', 'ICD10CM', 'ICD9CM', 'ICD10PCS']
rel_edges = rel_edges[rel_edges['SAB'].isin(sab_vocabs)]
print("Relationship edges after filtering by vocabularies: ", rel_edges.shape)

# Create node dataframe for KG
atom_df = concept_df[['CUI', 'SAB', 'CODE', 'STR']].drop_duplicates()
print("Atom dataframe shape: ", atom_df.shape)

# Check for duplicates and save nodes
print("Node counts by vocabulary type:")
print(atom_df[['SAB', 'CODE']].groupby('SAB').count())

atom_df.to_csv(f"{DATA_DIR}/nodes_kg.csv", sep='\t', index=False)

# Create nodes with unique IDs
nodes = atom_df.copy()
nodes['node_id'] = nodes['CODE'] + ':' + nodes['SAB'].str.lower()
nodes['node_name'] = nodes['STR'].copy()
nodes['ntype'] = nodes['SAB'].copy()
nodes = nodes[['node_id', 'node_name', 'ntype', 'CUI']]
nodes['old_node_id'] = nodes['node_id'].str.split(':', expand=True)[0]
nodes = nodes.reset_index().drop(['index'], axis=1)
nodes['node_index'] = nodes.index

# Function to create relationship edges for each vocabulary
def make_kg_edges_for_vocab(vocab, rel_edges, concept_df, atom_df, wanted_rels):
    edges = rel_edges[
        (rel_edges['SAB'] == vocab) &
        (rel_edges['RELA'].isin(wanted_rels))
    ]
    
    edges = edges.merge(
        concept_df[['CUI', 'CODE', 'STR', 'SAB']],
        left_on='CUI1', right_on='CUI', how='left'
    ).rename(
        columns={'CODE': 'CODE_1', 'STR': 'STR_1', 'SAB': 'SAB_1'}
    ).drop('CUI', axis=1)
    
    edges = edges.merge(
        concept_df[['CUI', 'CODE', 'STR', 'SAB']],
        left_on='CUI2', right_on='CUI', how='left'
    ).rename(
        columns={'CODE': 'CODE_2', 'STR': 'STR_2', 'SAB': 'SAB_2'}
    ).drop('CUI', axis=1)
    
    edges = edges[edges['CODE_1'] != edges['CODE_2']].drop_duplicates()
    
    edges = edges.merge(atom_df[['CUI', 'CODE', 'STR', 'SAB']], left_on='CUI1', right_on='CUI', how='left')
    
    edges = edges.rename(columns={
        'CODE': 'CODE_1',
        'STR': 'STR_1',
        'SAB_y': 'SAB_1'
    })
    
    edges = edges[[
        'CODE_1', 'STR_1', 'CUI1', 'SAB_1', 'RELA', 'CODE_2', 'STR_2', 'CUI2', 'SAB_2'
    ]]
    edges = edges.loc[:, ~edges.columns.duplicated()]
    edges.columns = [
        'node_id_x', 'node_name_x', 'CUI_x','ntype_x',
        'relationship',
        'node_id_y', 'node_name_y', 'CUI_y', 'ntype_y'
    ]
    
    return edges

# Define relationships to keep for each vocabulary
vocab_relations = {
    'ATC': ['isa', 'member_of', 'member-of'],
    'CPT': ['associated_procedure_of', 'has_associated_procedure', 'has_procedure_site', 
            'procedure_site_of', 'has_pathology', 'pathology_of', 'add_on_code_for', 
            'has_add_on_code'],
    'LNC': ['has_expanded_form', 'expanded_form_of', 'mth_has_expanded_form', 'mth_expanded_form_of'],
    'SNOMEDCT_US': ['cause_of', 'due_to', 'definitional_manifestation_of', 
                    'has_definitional_manifestation', 'occurs_after', 
                    'occurs_before', 'occurs_in', 'associated_with'],
    'ICD10PCS': ['expanded_form_of', 'has_expanded_form'],
    'ICD9CM': ['has_finding_site', 'finding_site_of'],
    'ICD10CM': ['has_finding_site', 'finding_site_of']
}

# Create edges for each vocabulary
edge_dfs = []
for vocab, rels in vocab_relations.items():
    print(f"Creating edges for {vocab}...")
    edges = make_kg_edges_for_vocab(vocab, rel_edges, concept_df, atom_df, rels)
    edges['node_id_x'] = edges['node_id_x'] + ':' + edges['ntype_x'].str.lower()
    edges['node_id_y'] = edges['node_id_y'] + ':' + edges['ntype_y'].str.lower()
    edge_dfs.append(edges)
    print(f"Created {len(edges)} edges for {vocab}")

# Try to add SNOMED-ICD10CM mappings
try:
    print("Adding SNOMED to ICD10CM mappings...")
    snomed_map_df = pd.read_csv("/data/horse/ws/arsi805e-finetune/Thesis/SNOMED/subset_SNOMED/Refset/Map/der2_iisssccRefset_ExtendedMapSnapshot_INT_20250401.txt", sep='\t')
    snomed_map_df['referencedComponentId'] = snomed_map_df['referencedComponentId'].astype(str)
    snomed_map_df['mapTarget'] = snomed_map_df['mapTarget'].astype(str)

    snomed_map_df = snomed_map_df.loc[(snomed_map_df['referencedComponentId'].isin(nodes.loc[nodes['ntype'] == 'SNOMEDCT_US']['old_node_id'].tolist())) & 
                      (snomed_map_df['mapTarget'].isin(nodes.loc[nodes['ntype']=='ICD10CM']['old_node_id'].tolist()))]

    snomed_map_df = snomed_map_df.loc[(snomed_map_df['active'] == 1)]

    snomed_map_df = snomed_map_df[['referencedComponentId', 'mapTarget']]
    snomed_map_df.columns = ['node_id_x', 'node_id_y']
    snomed_map_df = snomed_map_df.assign(relationship='snomed_icd')
    snomed_map_df = snomed_map_df.assign(ntype_x='SNOMEDCT_US')
    snomed_map_df = snomed_map_df.assign(ntype_y='ICD10CM')

    snomed_map_df = snomed_map_df.merge(snomed_concept_df[['CODE', 'STR']], left_on='node_id_x', right_on='CODE').drop(['CODE'], axis=1)
    snomed_map_df = snomed_map_df.merge(icd10_concept_df[['CODE', 'STR']], left_on='node_id_y', right_on='CODE').drop(['CODE'], axis=1)

    snomed_map_df['node_id_x'] = snomed_map_df['node_id_x'] + ':' + 'snomedct_us'
    snomed_map_df['node_id_y'] = snomed_map_df['node_id_y'] + ':' + 'icd10cm'
    snomed_map_df.columns = ['node_id_x', 'node_id_y', 'relationship', 'ntype_x', 'ntype_y', 'node_name_x', 'node_name_y']
    
    edge_dfs.append(snomed_map_df)
    print(f"Added {len(snomed_map_df)} SNOMED-ICD10CM edges")
except Exception as e:
    print(f"Warning: Could not add SNOMED to ICD10CM mappings: {e}")

# Try to add ATC4-ATC5 edges
try:
    print("Adding ATC4-ATC5 mappings...")
    atc5_df = atc_concept_df.loc[atc_concept_df['len_atc'] == 7]
    atc4_df = atc_concept_df.loc[atc_concept_df['len_atc'] == 5]

    atc_rows = []
    for i, parent_atc in atc4_df.iterrows():
        atc = parent_atc['CODE']
        name = parent_atc['STR']
        cui = parent_atc['CUI']
        df = atc5_df.loc[atc5_df['CODE'].str.startswith(atc)][['CODE', 'STR', 'SAB', 'CUI']]
        df.columns = ['node_id_y', 'node_name_y', 'ntype_y', 'CUI_y']
        df = df.assign(node_id_x=[atc]*len(df))
        df = df.assign(node_name_x=[name]*len(df))
        df = df.assign(ntype_x=['ATC']*len(df))
        df = df.assign(CUI_x=[cui]*len(df))
        atc_rows.append(df)

    if atc_rows:
        atc45_relation_df = pd.concat(atc_rows)
        atc45_relation_df['relationship'] = 'has_drug'
        atc45_relation_df['node_id_x'] = atc45_relation_df['node_id_x'] + ':atc'
        atc45_relation_df['node_id_y'] = atc45_relation_df['node_id_y'] + ':atc'
        edge_dfs.append(atc45_relation_df)
        print(f"Added {len(atc45_relation_df)} ATC4-ATC5 edges")
except Exception as e:
    print(f"Warning: Could not add ATC4-ATC5 mappings: {e}")

# Function to standardize edge columns
def standardize_edge_columns(df):
    """
    Ensure the DataFrame columns are in the canonical order for KG edges.
    Missing columns are added with None values.
    """
    canonical_cols = [
        'node_id_x', 'node_name_x', 'CUI_x', 'ntype_x',
        'relationship',
        'node_id_y', 'node_name_y', 'CUI_y', 'ntype_y'
    ]
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = None
    return df[canonical_cols]

# Standardize and combine all edge dataframes
print("Combining edge dataframes...")
edge_dfs = [standardize_edge_columns(df).loc[:, ~df.columns.duplicated()] for df in edge_dfs]
edge_df = pd.concat(edge_dfs, ignore_index=True)
print(f"Combined {len(edge_df)} edges from all vocabularies")

# Create bidirectional edges
print("Creating bidirectional edges...")
edge_x_list = edge_df['node_id_x'].tolist() + edge_df['node_id_y'].tolist()
edge_y_list = edge_df['node_id_y'].tolist() + edge_df['node_id_x'].tolist()
edge_relationship_list = edge_df['relationship'].tolist() + edge_df['relationship'].tolist()
edge_CUI_x = edge_df['CUI_x'].tolist() + edge_df['CUI_y'].tolist()
edge_CUI_y = edge_df['CUI_y'].tolist() + edge_df['CUI_x'].tolist()
edge_ntype_x_list = edge_df['ntype_x'].tolist() + edge_df['ntype_y'].tolist()
edge_ntype_y_list = edge_df['ntype_y'].tolist() + edge_df['ntype_x'].tolist()

new_edge_df = pd.DataFrame({
    'node_id_x': edge_x_list, 
    'CUI_x': edge_CUI_x, 
    'ntype_x': edge_ntype_x_list,
    'relationship': edge_relationship_list,
    'node_id_y': edge_y_list, 
    'CUI_y': edge_CUI_y,
    'ntype_y': edge_ntype_y_list
}).drop_duplicates(['node_id_x', 'relationship', 'node_id_y'])

print(f"Final edge dataframe shape after bidirectionalization: {new_edge_df.shape}")

# Standardize node IDs format
for col in ['node_id_x', 'node_id_y']:
    new_edge_df[col] = new_edge_df[col].str.replace(
        r':([A-Za-z0-9]+)$',
        lambda m: ':' + m.group(1).lower(),
        regex=True
    )

# Remove edges with null relationships
new_edge_df.dropna(subset=['relationship'], inplace=True)
print(f"Final edge dataframe shape after removing null relationships: {new_edge_df.shape}")

# Save edges
new_edge_df.to_csv(f"{DATA_DIR}/edges.csv", sep='\t', index=False)

# Create combined edges with node indices
valid_nodes = set(nodes['node_id'])
df = new_edge_df[new_edge_df['node_id_x'].isin(valid_nodes) & new_edge_df['node_id_y'].isin(valid_nodes)]

combined_edge_df = df.merge(nodes[['node_id', 'node_index']], left_on='node_id_x', right_on='node_id').drop(['node_id'], axis=1)
combined_edge_df = combined_edge_df.merge(nodes[['node_id', 'node_index']], left_on='node_id_y', right_on='node_id').drop(['node_id'], axis=1)
combined_edge_df.dropna(subset=['relationship'], inplace=True)

# Save final edge data with node indices
combined_edge_df = combined_edge_df.reset_index().drop(['index'], axis=1)
combined_edge_df = combined_edge_df.assign(edge_index=combined_edge_df.index)
combined_edge_df.to_csv(f"{DATA_DIR}/edge_kg.csv", sep='\t', index=False)

# Generate adjacency list for network analysis
combined_edge_df[['node_index_x', 'node_index_y']].to_csv(f"{DATA_DIR}/adj_list.csv", sep=' ', index=False, header=False)

print("Building NetworkX graph...")
# Create a NetworkX graph representation
G = nx.MultiDiGraph()  # Directed graph with possible multiple edges

# Add nodes
for _, row in tqdm(nodes.iterrows(), total=len(nodes), desc="Adding nodes"):
    G.add_node(
        row['node_id'],
        node_name=row['node_name'],
        ntype=row['ntype'],
        CUI=row['CUI'],
        old_node_id=row['old_node_id'],
        node_index=row['node_index']
    )

# Add edges
for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
    G.add_edge(
        row['node_id_x'],
        row['node_id_y'],
        relationship=row['relationship'],
        CUI_x=row['CUI_x'],
        ntype_x=row['ntype_x'],
        CUI_y=row['CUI_y'],
        ntype_y=row['ntype_y']
    )

# Find largest connected component for a more usable graph
print("Finding largest connected component...")
largest_cc = max(nx.weakly_connected_components(G), key=len)
G_connected = G.subgraph(largest_cc).copy()

print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Largest component: {G_connected.number_of_nodes()} nodes, {G_connected.number_of_edges()} edges")

# Build lookup dictionaries for easier access
print("Building lookup dictionaries...")

def build_lookups(conso_df):
    # CODE → CUI
    code2cui = pd.Series(conso_df['CUI'].values, index=conso_df['CODE']).to_dict()
    # CUI → list of codes
    cui2codes = conso_df.groupby('CUI')['CODE'].apply(list).to_dict()
    # CUI → list of strings
    cui2strs = conso_df.groupby('CUI')['STR'].apply(list).to_dict()
    # CUI → list of SABs
    cui2sabs = conso_df.groupby('CUI')['SAB'].apply(list).to_dict()
    # CUI → all codes+vocab+desc
    cui2records = conso_df.groupby('CUI').apply(lambda g: g[['CODE','SAB','STR']].to_dict('records')).to_dict()
    return code2cui, cui2codes, cui2strs, cui2sabs, cui2records

# Build lookups
code2cui, cui2codes, cui2strs, cui2sabs, cui2records = build_lookups(concept_df)

# Save the graph and lookup data
import pickle
print("Saving graph and lookups...")

with open(f"{DATA_DIR}/clinical_kg_networkx.pkl", "wb") as f:
    pickle.dump(G, f)
print(f"Saved full KG graph to {DATA_DIR}/clinical_kg_networkx.pkl")

with open(f"{DATA_DIR}/clinical_kg_connected_networkx.pkl", "wb") as f:
    pickle.dump(G_connected, f)
print(f"Saved connected component KG graph to {DATA_DIR}/clinical_kg_connected_networkx.pkl")

with open(f"{DATA_DIR}/kg_lookups.pkl", "wb") as f:
    pickle.dump({
        'code2cui': code2cui,
        'cui2codes': cui2codes, 
        'cui2strs': cui2strs,
        'cui2sabs': cui2sabs,
        'cui2records': cui2records
    }, f)
print(f"Saved lookup dictionaries to {DATA_DIR}/kg_lookups.pkl")

print("Knowledge Graph building complete!")
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Connected component: {G_connected.number_of_nodes()} nodes, {G_connected.number_of_edges()} edges")

print("Done.....")