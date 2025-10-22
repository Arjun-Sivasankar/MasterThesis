import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

# Define the ICD9 formatting functions from common_textgen.py
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

def is_valid_icd9(code: str) -> bool:
    if not code: return False
    c = code.upper()
    if c[0].isdigit(): return bool(re.match(r"^\d{3}(\.\d{1,2})?$", c))
    if c[0]=="V":      return bool(re.match(r"^V\d{2}(\.\d{1,2})?$", c))
    if c[0]=="E":      return bool(re.match(r"^E\d{3}(\.\d{1})?$", c))
    return False

# Set paths
MIMIC_ICD9_PATH = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl'
TOP50_PATH = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/top_50_codes.csv'
BOTTOM50_PATH = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/icd9_analysis_improved/bottom_50_codes.csv'
KG_DIR = '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG'
OUTPUT_DIR = os.path.join(KG_DIR, 'coverage_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ICD-9 codes from MIMIC dataset
print("Loading ICD-9 codes from MIMIC dataset...")
with open(MIMIC_ICD9_PATH, 'rb') as f:
    mimic_icd9_df = pickle.load(f)

print(f"Loaded MIMIC ICD-9 dataframe with shape: {mimic_icd9_df.shape}")
print("Sample rows from MIMIC ICD-9 DataFrame:")
print(mimic_icd9_df.head())

# Load Top 50 and Bottom 50 codes
print("\nLoading Top 50 and Bottom 50 ICD-9 codes...")
try:
    top50_df = pd.read_csv(TOP50_PATH)
    print(f"Loaded Top 50 codes dataframe with shape: {top50_df.shape}")
    print("Sample from Top 50 codes:")
    print(top50_df.head())
    
    bottom50_df = pd.read_csv(BOTTOM50_PATH)
    print(f"Loaded Bottom 50 codes dataframe with shape: {bottom50_df.shape}")
    print("Sample from Bottom 50 codes:")
    print(bottom50_df.head())
    
    # Extract and format the code columns
    if 'icd_code' in top50_df.columns:
        top50_codes = {format_icd9(code) for code in top50_df['icd_code']}
    else:
        # Try to find the column that contains ICD codes
        code_columns = [col for col in top50_df.columns if 'code' in col.lower()]
        if code_columns:
            top50_codes = {format_icd9(code) for code in top50_df[code_columns[0]]}
        else:
            # Assume first column contains codes
            top50_codes = {format_icd9(code) for code in top50_df.iloc[:, 0]}
    
    if 'icd_code' in bottom50_df.columns:
        bottom50_codes = {format_icd9(code) for code in bottom50_df['icd_code']}
    else:
        # Try to find the column that contains ICD codes
        code_columns = [col for col in bottom50_df.columns if 'code' in col.lower()]
        if code_columns:
            bottom50_codes = {format_icd9(code) for code in bottom50_df[code_columns[0]]}
        else:
            # Assume first column contains codes
            bottom50_codes = {format_icd9(code) for code in bottom50_df.iloc[:, 0]}
    
    print(f"Extracted {len(top50_codes)} unique top 50 codes")
    print(f"Extracted {len(bottom50_codes)} unique bottom 50 codes")
    
except Exception as e:
    print(f"Error loading Top/Bottom 50 files: {e}")
    top50_codes = set()
    bottom50_codes = set()

# Format all MIMIC ICD9 codes
mimic_icd9_df['formatted_code'] = mimic_icd9_df['icd_code'].apply(format_icd9)

# Load the UMLS concepts
print("\nLoading KG lookup dictionaries...")
try:
    with open(os.path.join(KG_DIR, 'data_files/kg_lookups.pkl'), 'rb') as f:
        lookups = pickle.load(f)
    code2cui = lookups.get('code2cui', {})
    cui2codes = lookups.get('cui2codes', {})
    cui2sabs = lookups.get('cui2sabs', {})
    
    # Extract ICD9CM codes from lookups
    umls_icd9_codes = []
    umls_icd9_titles = []
    
    print("Extracting ICD9CM codes from lookups...")
    for cui, sabs in cui2sabs.items():
        if 'ICD9CM' in sabs:
            codes = cui2codes.get(cui, [])
            icd9_codes = [code for code in codes if code in code2cui and code2cui[code] == cui]
            for code in icd9_codes:
                # Format the code properly
                formatted_code = format_icd9(code)
                if is_valid_icd9(formatted_code):
                    umls_icd9_codes.append(formatted_code)
                    
except FileNotFoundError:
    print("KG lookups pickle not found. Trying to load nodes CSV...")
    try:
        nodes_df = pd.read_csv(os.path.join(KG_DIR, 'data_files/nodes_kg.csv'), sep='\t')
        # Filter for ICD9CM codes
        umls_icd9_df = nodes_df[nodes_df['SAB'] == 'ICD9CM'][['CODE', 'STR']].copy()
        umls_icd9_codes = [format_icd9(code) for code in umls_icd9_df['CODE']]
        umls_icd9_codes = [code for code in umls_icd9_codes if is_valid_icd9(code)]
    except FileNotFoundError:
        print("Nodes CSV not found. Using buildKG.py to extract ICD9 codes...")
        # Try to access icd9_concept_df directly
        try:
            # Load necessary modules
            import importlib.util
            spec = importlib.util.spec_from_file_location("buildKG", 
                   "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/buildKG.py")
            buildKG = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(buildKG)
            
            # Format codes from icd9_concept_df
            umls_icd9_codes = [format_icd9(code) for code in buildKG.icd9_concept_df['CODE']]
            umls_icd9_codes = [code for code in umls_icd9_codes if is_valid_icd9(code)]
        except Exception as e:
            print(f"Error loading buildKG.py: {e}")
            print("Loading pickled graph to extract ICD9 codes...")
            try:
                import networkx as nx
                with open(os.path.join(KG_DIR, 'clinical_kg_networkx.pkl'), 'rb') as f:
                    G = pickle.load(f)
                
                # Extract ICD9CM nodes
                umls_icd9_codes = []
                for node, attrs in G.nodes(data=True):
                    if attrs.get('ntype') == 'ICD9CM':
                        code = attrs.get('old_node_id', '')
                        formatted_code = format_icd9(code)
                        if is_valid_icd9(formatted_code):
                            umls_icd9_codes.append(formatted_code)
            except Exception as e:
                print(f"Error loading graph: {e}")
                umls_icd9_codes = []

# Clean and standardize code formats for comparison
print("Standardizing code formats for comparison...")
mimic_codes = set(mimic_icd9_df['formatted_code'])
mimic_valid_codes = {code for code in mimic_codes if is_valid_icd9(code)}
umls_codes = set(umls_icd9_codes)

# Report the number of codes before and after validation
print(f"MIMIC codes before validation: {len(mimic_codes)}")
print(f"MIMIC codes after validation: {len(mimic_valid_codes)}")
print(f"UMLS codes: {len(umls_codes)}")

# Find overlap and differences
codes_in_both = mimic_valid_codes & umls_codes
codes_only_in_mimic = mimic_valid_codes - umls_codes
codes_only_in_umls = umls_codes - mimic_valid_codes

# Print summary statistics
print("\n===== ICD-9-CM Code Coverage Analysis =====")
print(f"Total valid ICD-9 codes in MIMIC: {len(mimic_valid_codes)}")
print(f"Total ICD-9 codes in UMLS concept df: {len(umls_codes)}")
print(f"Codes present in BOTH: {len(codes_in_both)} ({len(codes_in_both)/len(mimic_valid_codes)*100:.2f}% of MIMIC codes)")
print(f"Codes ONLY in MIMIC (missing from UMLS): {len(codes_only_in_mimic)} ({len(codes_only_in_mimic)/len(mimic_valid_codes)*100:.2f}% of MIMIC codes)")
print(f"Codes ONLY in UMLS (not in MIMIC): {len(codes_only_in_umls)} ({len(codes_only_in_umls)/len(umls_codes)*100:.2f}% of UMLS codes)")

# Analyze Top 50 and Bottom 50 codes coverage
if top50_codes:
    top50_valid = {code for code in top50_codes if is_valid_icd9(code)}
    top50_in_umls = top50_valid & umls_codes
    top50_missing = top50_valid - umls_codes
    
    print("\n===== Top 50 ICD-9-CM Code Coverage =====")
    print(f"Total valid codes in Top 50: {len(top50_valid)}")
    print(f"Top 50 codes present in KG: {len(top50_in_umls)} ({len(top50_in_umls)/len(top50_valid)*100:.2f}% coverage)")
    print(f"Top 50 codes missing from KG: {len(top50_missing)} ({len(top50_missing)/len(top50_valid)*100:.2f}% missing)")
    
    if top50_missing:
        print("\nMissing Top 50 codes:")
        for code in sorted(top50_missing):
            matching_rows = mimic_icd9_df[mimic_icd9_df['formatted_code'] == code]
            if not matching_rows.empty:
                title = matching_rows.iloc[0]['long_title']
                print(f"  {code} - {title}")
            else:
                print(f"  {code} - (description not available)")

if bottom50_codes:
    bottom50_valid = {code for code in bottom50_codes if is_valid_icd9(code)}
    bottom50_in_umls = bottom50_valid & umls_codes
    bottom50_missing = bottom50_valid - umls_codes
    
    print("\n===== Bottom 50 ICD-9-CM Code Coverage =====")
    print(f"Total valid codes in Bottom 50: {len(bottom50_valid)}")
    print(f"Bottom 50 codes present in KG: {len(bottom50_in_umls)} ({len(bottom50_in_umls)/len(bottom50_valid)*100:.2f}% coverage)")
    print(f"Bottom 50 codes missing from KG: {len(bottom50_missing)} ({len(bottom50_missing)/len(bottom50_valid)*100:.2f}% missing)")
    
    if bottom50_missing:
        print("\nMissing Bottom 50 codes:")
        for code in sorted(bottom50_missing):
            matching_rows = mimic_icd9_df[mimic_icd9_df['formatted_code'] == code]
            if not matching_rows.empty:
                title = matching_rows.iloc[0]['long_title']
                print(f"  {code} - {title}")
            else:
                print(f"  {code} - (description not available)")

# Analyze long tail problem - compare coverage percentages between top and bottom
if top50_codes and bottom50_codes:
    top50_coverage = len(top50_in_umls)/len(top50_valid)*100
    bottom50_coverage = len(bottom50_in_umls)/len(bottom50_valid)*100
    
    print("\n===== Long Tail Analysis =====")
    print(f"Top 50 codes coverage: {top50_coverage:.2f}%")
    print(f"Bottom 50 codes coverage: {bottom50_coverage:.2f}%")
    print(f"Coverage difference (Top - Bottom): {top50_coverage - bottom50_coverage:.2f}%")
    
    if top50_coverage > bottom50_coverage:
        print("The knowledge graph has better coverage of frequent codes than rare codes,")
        print("which may reinforce the long-tail problem.")
    elif top50_coverage < bottom50_coverage:
        print("Surprisingly, the knowledge graph has better coverage of rare codes than frequent codes.")
    else:
        print("The knowledge graph has similar coverage for both frequent and rare codes.")

# Analyze code format patterns
print("\n=== Code Format Analysis ===")
# Extract patterns from codes only in MIMIC to understand why they might be missing
mimic_only_patterns = {}
for code in codes_only_in_mimic:
    # Look for patterns like length, decimal points, leading zeros, etc.
    length = len(code)
    has_decimal = '.' in code
    has_letter = any(c.isalpha() for c in code)
    pattern = f"Length:{length}, Decimal:{has_decimal}, Letter:{has_letter}"
    
    if pattern not in mimic_only_patterns:
        mimic_only_patterns[pattern] = []
    
    mimic_only_patterns[pattern].append(code)

print("Patterns in MIMIC-only codes:")
for pattern, codes in sorted(mimic_only_patterns.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"{pattern}: {len(codes)} codes (e.g., {codes[:3]})")

# Analyze code patterns by chapter/category
print("\n=== Analyzing ICD-9 chapters/categories ===")

def get_icd9_chapter(code):
    """Map ICD-9-CM code to its chapter/category"""
    try:
        # Remove decimal point if present
        clean_code = code.replace('.', '')
        
        if clean_code.startswith('V'):
            return "V codes - Supplementary"
        elif clean_code.startswith('E'):
            return "E codes - External causes"
        
        numeric_part = ''.join(c for c in clean_code if c.isdigit())
        if not numeric_part:
            return "Invalid/Unknown Format"
            
        num = int(numeric_part[:3])
        
        if 1 <= num <= 139:
            return "001-139: Infectious and Parasitic Diseases"
        elif 140 <= num <= 239:
            return "140-239: Neoplasms"
        elif 240 <= num <= 279:
            return "240-279: Endocrine, Nutritional, Metabolic, Immunity"
        elif 280 <= num <= 289:
            return "280-289: Blood and Blood-Forming Organs"
        elif 290 <= num <= 319:
            return "290-319: Mental Disorders"
        elif 320 <= num <= 389:
            return "320-389: Nervous System and Sense Organs"
        elif 390 <= num <= 459:
            return "390-459: Circulatory System"
        elif 460 <= num <= 519:
            return "460-519: Respiratory System"
        elif 520 <= num <= 579:
            return "520-579: Digestive System"
        elif 580 <= num <= 629:
            return "580-629: Genitourinary System"
        elif 630 <= num <= 679:
            return "630-679: Pregnancy, Childbirth, Puerperium"
        elif 680 <= num <= 709:
            return "680-709: Skin and Subcutaneous Tissue"
        elif 710 <= num <= 739:
            return "710-739: Musculoskeletal and Connective Tissue"
        elif 740 <= num <= 759:
            return "740-759: Congenital Anomalies"
        elif 760 <= num <= 779:
            return "760-779: Perinatal Period Conditions"
        elif 780 <= num <= 799:
            return "780-799: Symptoms, Signs, Ill-Defined Conditions"
        elif 800 <= num <= 999:
            return "800-999: Injury and Poisoning"
        else:
            return "Other/Unknown"
    except:
        return "Invalid/Unknown Format"

# Analyze chapter distribution for MIMIC-only codes
mimic_only_chapters = {}
for code in codes_only_in_mimic:
    chapter = get_icd9_chapter(code)
    mimic_only_chapters[chapter] = mimic_only_chapters.get(chapter, 0) + 1

# Analyze chapter distribution for Top 50 and Bottom 50 missing codes
if top50_missing:
    top50_missing_chapters = {}
    for code in top50_missing:
        chapter = get_icd9_chapter(code)
        top50_missing_chapters[chapter] = top50_missing_chapters.get(chapter, 0) + 1
    
    print("\nMissing Top 50 codes by chapter:")
    for chapter, count in sorted(top50_missing_chapters.items(), key=lambda x: x[1], reverse=True):
        print(f"{chapter}: {count} codes ({count/len(top50_missing)*100:.2f}%)")

if bottom50_missing:
    bottom50_missing_chapters = {}
    for code in bottom50_missing:
        chapter = get_icd9_chapter(code)
        bottom50_missing_chapters[chapter] = bottom50_missing_chapters.get(chapter, 0) + 1
    
    print("\nMissing Bottom 50 codes by chapter:")
    for chapter, count in sorted(bottom50_missing_chapters.items(), key=lambda x: x[1], reverse=True):
        print(f"{chapter}: {count} codes ({count/len(bottom50_missing)*100:.2f}%)")

# Sort and print overall chapter distribution
sorted_chapters = sorted(mimic_only_chapters.items(), key=lambda x: x[1], reverse=True)
print("\nICD-9 chapters for codes found ONLY in MIMIC:")
for chapter, count in sorted_chapters:
    print(f"{chapter}: {count} codes ({count/len(codes_only_in_mimic)*100:.2f}%)")

# Save detailed data to CSV for further analysis
print("\nSaving detailed data to CSV files...")

# Save codes in both datasets
both_codes_df = mimic_icd9_df[mimic_icd9_df['formatted_code'].isin(codes_in_both)].copy()
both_codes_df['source'] = 'both'
both_codes_df.to_csv(os.path.join(OUTPUT_DIR, 'icd9_codes_in_both.csv'), index=False)

# Save MIMIC-only codes
mimic_only_df = mimic_icd9_df[mimic_icd9_df['formatted_code'].isin(codes_only_in_mimic)].copy()
mimic_only_df['source'] = 'mimic_only'
mimic_only_df.to_csv(os.path.join(OUTPUT_DIR, 'icd9_codes_only_in_mimic.csv'), index=False)

# Save Top 50 and Bottom 50 coverage analysis
if top50_codes:
    top50_coverage_df = pd.DataFrame({
        'code': list(top50_valid),
        'in_umls': [code in umls_codes for code in top50_valid]
    })
    top50_coverage_df.to_csv(os.path.join(OUTPUT_DIR, 'top50_coverage.csv'), index=False)

if bottom50_codes:
    bottom50_coverage_df = pd.DataFrame({
        'code': list(bottom50_valid),
        'in_umls': [code in umls_codes for code in bottom50_valid]
    })
    bottom50_coverage_df.to_csv(os.path.join(OUTPUT_DIR, 'bottom50_coverage.csv'), index=False)

# Create visualizations
print("Creating visualizations...")

# Overall code coverage
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
labels = ['MIMIC Valid\nCodes', 'UMLS\nCodes', 'In Both', 'Only in MIMIC', 'Only in UMLS']
values = [len(mimic_valid_codes), len(umls_codes), len(codes_in_both), len(codes_only_in_mimic), len(codes_only_in_umls)]
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

plt.bar(labels, values, color=colors)
plt.title('ICD-9-CM Code Coverage Analysis', fontsize=16)
plt.ylabel('Number of Codes', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

for i, v in enumerate(values):
    plt.text(i, v + 0.5, str(v), ha='center')

plt.savefig(os.path.join(OUTPUT_DIR, 'icd9_code_coverage.png'), dpi=300, bbox_inches='tight')

# Top 50 vs Bottom 50 Coverage Comparison
if top50_codes and bottom50_codes:
    plt.figure(figsize=(10, 6))
    comparison_data = {
        'Category': ['Top 50', 'Bottom 50'],
        'Present in KG': [len(top50_in_umls), len(bottom50_in_umls)],
        'Missing from KG': [len(top50_missing), len(bottom50_missing)]
    }
    
    df = pd.DataFrame(comparison_data)
    df_plot = df.set_index('Category')
    
    ax = df_plot.plot(kind='bar', stacked=True, figsize=(10, 6), 
                     color=['#2ecc71', '#e74c3c'])
    
    plt.title('Coverage Comparison: Top 50 vs Bottom 50 ICD-9 Codes', fontsize=16)
    plt.ylabel('Number of Codes', fontsize=14)
    plt.tight_layout()
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height > 0:  # Only add label if there's space
            if i < 2:  # First category (Present in KG)
                percentage = height / (df_plot.iloc[i//2].sum()) * 100
                label = f'{int(height)} ({percentage:.1f}%)'
            else:  # Second category (Missing)
                percentage = height / (df_plot.iloc[(i-2)//2].sum()) * 100
                label = f'{int(height)} ({percentage:.1f}%)'
            
            ax.text(x + width/2, y + height/2, label, ha='center', va='center')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_vs_bottom_coverage.png'), dpi=300, bbox_inches='tight')

# Visualize chapter distribution for missing codes
plt.figure(figsize=(14, 8))
chapters, counts = zip(*sorted_chapters[:10])  # Top 10 chapters
plt.barh(chapters, counts, color=sns.color_palette("viridis", len(chapters)))
plt.title('Top 10 ICD-9 Chapters Missing from UMLS (MIMIC-only codes)', fontsize=16)
plt.xlabel('Number of Codes', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'icd9_missing_chapters_top10.png'), dpi=300, bbox_inches='tight')

print(f"\nAnalysis complete! Detailed reports saved to {OUTPUT_DIR}")
print("\n===== Summary of Results =====")
print(f"MIMIC coverage in KG: {len(codes_in_both)/len(mimic_valid_codes)*100:.2f}% of valid MIMIC codes")
print(f"KG utilization in MIMIC: {len(codes_in_both)/len(umls_codes)*100:.2f}% of KG ICD9 codes")

if top50_codes and bottom50_codes:
    print(f"\nLong tail analysis:")
    print(f"Top 50 codes coverage: {len(top50_in_umls)/len(top50_valid)*100:.2f}%")
    print(f"Bottom 50 codes coverage: {len(bottom50_in_umls)/len(bottom50_valid)*100:.2f}%")
    
    if len(top50_in_umls)/len(top50_valid)*100 > len(bottom50_in_umls)/len(bottom50_valid)*100:
        print("The knowledge graph has better coverage of frequent codes.")
        print("This suggests the KG could help with the common codes but may not")
        print("provide as much assistance with the long tail of rare codes.")
    else:
        print("The knowledge graph has good coverage of rare codes.")
        print("This suggests the KG may be particularly helpful in addressing")
        print("the long tail problem by providing knowledge for rare codes.")