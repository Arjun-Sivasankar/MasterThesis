import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Tuple
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

# Import necessary functions from the finetuning script
import sys
sys.path.append('/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen')
from finetune_llama_gen_difftrainsize_improved import (
    format_icd9_properly, is_valid_icd9, get_icd9_parent
)

def get_args():
    ap = argparse.ArgumentParser(description="Improved ICD-9 Code Analysis for MIMIC Dataset")
    ap.add_argument("--train_pickle", default=None, help="Path to training data pickle")
    ap.add_argument("--val_pickle", default=None, help="Path to validation data pickle")
    ap.add_argument("--test_pickle", default=None, help="Path to test data pickle")
    ap.add_argument("--data_pickle", default=None, help="Path to combined data pickle")
    ap.add_argument("--icd9_pickle", default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl",
                    help="Path to ICD-9 code reference")
    ap.add_argument("--label_col", default="icd_code", help="Column containing ICD codes")
    ap.add_argument("--output_dir", default="icd9_analysis_improved",
                    help="Directory to save analysis outputs")
    ap.add_argument("--top_n", type=int, default=50, 
                    help="Number of top codes for detailed analysis (increased for better coverage)")
    return ap.parse_args()

def load_and_prepare_data(args):
    """Load data and extract all ICD-9 codes with proper formatting."""
    if args.train_pickle and args.val_pickle and args.test_pickle:
        train_df = pickle.load(open(args.train_pickle, "rb"))
        val_df = pickle.load(open(args.val_pickle, "rb"))
        test_df = pickle.load(open(args.test_pickle, "rb"))
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        print(f"Loaded data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    elif args.data_pickle:
        all_df = pickle.load(open(args.data_pickle, "rb"))
        print(f"Loaded combined data: {len(all_df)} samples")
    else:
        raise ValueError("Provide either --data_pickle OR all of --train_pickle/--val_pickle/--test_pickle")
    
    # Extract and format codes once here
    all_df['formatted_codes'] = all_df[args.label_col].apply(
        lambda codes: [format_icd9_properly(str(c)) for c in codes if is_valid_icd9(format_icd9_properly(str(c)))]
    )
    return all_df

def get_code_distribution(df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    """Count frequency of each ICD-9 code across all visits."""
    all_codes = [code for code_list in df['formatted_codes'] for code in code_list]
    code_counts = Counter(all_codes)
    
    # Ensure all frequencies are integers
    for code in code_counts:
        code_counts[code] = int(code_counts[code])
    
    return code_counts, all_codes

def analyze_basic_statistics(df: pd.DataFrame, code_counts: Dict[str, int], all_codes: List[str], output_dir: str):
    """Compute and save basic statistics: unique codes, occurrences, codes per visit."""
    os.makedirs(output_dir, exist_ok=True)
    
    total_unique = len(code_counts)
    total_occurrences = sum(code_counts.values())
    codes_per_visit = df['formatted_codes'].apply(len)
    
    stats = {
        'total_unique_codes': total_unique,
        'total_occurrences': total_occurrences,
        'mean_frequency': np.mean(list(code_counts.values())),
        'median_frequency': np.median(list(code_counts.values())),
        'mean_codes_per_visit': codes_per_visit.mean(),
        'median_codes_per_visit': codes_per_visit.median(),
        'min_codes_per_visit': codes_per_visit.min(),
        'max_codes_per_visit': codes_per_visit.max(),
    }
    
    pd.DataFrame([stats]).to_csv(os.path.join(output_dir, "basic_statistics.csv"), index=False)
    
    # Combined distribution plot: frequency histogram + cumulative
    fig, ax1 = plt.subplots(figsize=(12, 6))
    frequencies = list(code_counts.values())
    ax1.hist(frequencies, bins=50, log=True, alpha=0.7, color='blue')
    ax1.set_xlabel('Code Frequency')
    ax1.set_ylabel('Number of Unique Codes (log scale)', color='blue')
    
    ax2 = ax1.twinx()
    sorted_freqs = sorted(frequencies, reverse=True)
    cumulative_percent = np.cumsum(sorted_freqs) / sum(sorted_freqs) * 100
    ax2.plot(range(1, len(sorted_freqs) + 1), cumulative_percent, 'r-', linewidth=2)
    ax2.set_ylabel('Cumulative Percentage', color='red')
    
    plt.title('ICD-9 Code Frequency Distribution and Cumulative Coverage')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frequency_and_cumulative.png"), dpi=300)
    plt.close()
    
    # Codes per visit histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(codes_per_visit, bins=30, kde=True)
    plt.title('Distribution of ICD-9 Codes per Visit')
    plt.xlabel('Number of Codes')
    plt.ylabel('Number of Visits')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "codes_per_visit_hist.png"), dpi=300)
    plt.close()
    
    # Boxplot for codes per visit
    plt.figure(figsize=(6, 8))
    sns.boxplot(y=codes_per_visit)
    plt.title('Boxplot of ICD-9 Codes per Visit')
    plt.ylabel('Number of Codes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "codes_per_visit_box.png"), dpi=300)
    plt.close()

def analyze_top_codes(code_counts: Dict[str, int], top_n: int, icd9_pickle: str, output_dir: str):
    """Analyze top N codes with descriptions and save as CSV/plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        icd9_df = pd.read_pickle(icd9_pickle)
        code_to_desc = dict(zip(
            icd9_df['icd_code'].apply(format_icd9_properly),
            icd9_df['long_title']
        ))
    except Exception as e:
        print(f"Could not load ICD-9 descriptions: {e}")
        code_to_desc = {}
    
    top_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    total_occurrences = sum(code_counts.values())
    
    top_df = pd.DataFrame({
        'ICD9_Code': [code for code, _ in top_codes],
        'Frequency': [freq for _, freq in top_codes],
        'Percentage': [(freq / total_occurrences * 100) for _, freq in top_codes],
        'Description': [code_to_desc.get(code, "N/A") for code, _ in top_codes]
    })
    top_df.to_csv(os.path.join(output_dir, f"top_{top_n}_codes.csv"), index=False)
    
    # Bar plot for top codes
    plt.figure(figsize=(14, 10))
    sns.barplot(data=top_df, x='Frequency', y='ICD9_Code', orient='h')
    plt.title(f'Top {top_n} ICD-9 Codes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"top_{top_n}_codes_bar.png"), dpi=300)
    plt.close()
    
    # Pie plot for top 10 percentages
    if len(top_df) >= 10:
        top10_df = top_df.head(10)
        plt.figure(figsize=(10, 10))
        plt.pie(top10_df['Percentage'], labels=top10_df['ICD9_Code'], autopct='%1.1f%%', textprops={'fontsize': 8})
        plt.title('Percentage Distribution of Top 10 ICD-9 Codes')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_10_codes_pie.png"), dpi=300)
        plt.close()

def get_icd9_chapter(code: str) -> str:
    """Return the official ICD-9-CM chapter name for a given code."""
    if not code:
        return "Unknown"
    
    if code.startswith('V'):
        return "V: Supplementary Classification"
    
    if code.startswith('E'):
        return "E: External Causes of Injury"
    
    if code[0].isdigit():
        first_three = code[:3] if len(code) >= 3 else code
        try:
            first_three_num = int(first_three)
            if 1 <= first_three_num <= 139:
                return "001-139: Infectious and Parasitic Diseases"
            elif 140 <= first_three_num <= 239:
                return "140-239: Neoplasms"
            elif 240 <= first_three_num <= 279:
                return "240-279: Endocrine, Nutritional, Metabolic, Immunity"
            elif 280 <= first_three_num <= 289:
                return "280-289: Blood and Blood-Forming Organs"
            elif 290 <= first_three_num <= 319:
                return "290-319: Mental Disorders"
            elif 320 <= first_three_num <= 389:
                return "320-389: Nervous System and Sense Organs"
            elif 390 <= first_three_num <= 459:
                return "390-459: Circulatory System"
            elif 460 <= first_three_num <= 519:
                return "460-519: Respiratory System"
            elif 520 <= first_three_num <= 579:
                return "520-579: Digestive System"
            elif 580 <= first_three_num <= 629:
                return "580-629: Genitourinary System"
            elif 630 <= first_three_num <= 679:
                return "630-679: Pregnancy, Childbirth, Puerperium"
            elif 680 <= first_three_num <= 709:
                return "680-709: Skin and Subcutaneous Tissue"
            elif 710 <= first_three_num <= 739:
                return "710-739: Musculoskeletal and Connective Tissue"
            elif 740 <= first_three_num <= 759:
                return "740-759: Congenital Anomalies"
            elif 760 <= first_three_num <= 779:
                return "760-779: Perinatal Conditions"
            elif 780 <= first_three_num <= 799:
                return "780-799: Symptoms, Signs, Ill-Defined Conditions"
            elif 800 <= first_three_num <= 999:
                return "800-999: Injury and Poisoning"
            else:
                return "Other Diagnosis Codes"
        except ValueError:
            pass
    
    return "Unknown"

def analyze_hierarchy(all_codes: List[str], output_dir: str):
    """Analyze chapter distribution and top subcategories in major chapters with more plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    chapters = [get_icd9_chapter(code) for code in all_codes]
    chapter_counts = Counter(chapters)
    total = sum([int(count) for count in chapter_counts.values()])
    
    chapter_df = pd.DataFrame({
        'Chapter': list(chapter_counts.keys()),
        'Count': [int(count) for count in chapter_counts.values()],
        'Percentage': [int(count) / total * 100 for count in chapter_counts.values()]
    }).sort_values('Count', ascending=False)
    chapter_df.to_csv(os.path.join(output_dir, "chapter_distribution.csv"), index=False)
    
    # Pie chart for chapters
    plt.figure(figsize=(12, 10))
    plt.pie(chapter_df['Count'], labels=chapter_df['Chapter'], autopct='%1.1f%%', textprops={'fontsize': 8})
    plt.title('ICD-9 Chapter Distribution (Pie)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chapter_distribution_pie.png"), dpi=300)
    plt.close()
    
    # Bar plot for chapters
    plt.figure(figsize=(14, 10))
    sns.barplot(data=chapter_df, x='Count', y='Chapter', orient='h')
    plt.title('ICD-9 Chapter Distribution (Bar)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chapter_distribution_bar.png"), dpi=300)
    plt.close()
    
    # Top subcategories in top 3 chapters with plots
    top_chapters = chapter_df['Chapter'][:3].tolist()
    for chapter in top_chapters:
        chapter_codes = [code for code, ch in zip(all_codes, chapters) if ch == chapter]
        subcats = Counter([get_icd9_parent(code) for code in chapter_codes])
        # Ensure integer counts
        for code in subcats:
            subcats[code] = int(subcats[code])
        
        top_subcats = pd.DataFrame(subcats.most_common(10), columns=['Subcategory', 'Count'])
        top_subcats.to_csv(os.path.join(output_dir, f"top_subcats_{chapter.split(':')[0]}.csv"), index=False)
        
        # Bar plot for top subcategories
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_subcats, x='Count', y='Subcategory', orient='h')
        plt.title(f'Top 10 Subcategories in {chapter}')
        plt.tight_layout()
        safe_chapter = chapter.split(':')[0].replace('-', '_')
        plt.savefig(os.path.join(output_dir, f"top_subcats_{safe_chapter}_bar.png"), dpi=300)
        plt.close()

def calculate_jaccard_similarity(matrix):
    """Calculate Jaccard similarity for sparse binary matrix."""
    # Convert to dense if small enough, otherwise use sparse computation
    if matrix.shape[1] <= 50:  # For small matrices, dense is fine
        dense_matrix = matrix.toarray()
        similarity = 1 - pairwise_distances(dense_matrix.T, metric='jaccard')
    else:
        # Sparse Jaccard similarity calculation
        n = matrix.shape[0]
        similarity = np.zeros((matrix.shape[1], matrix.shape[1]))
        
        for i in range(matrix.shape[1]):
            for j in range(i + 1, matrix.shape[1]):
                # Jaccard = intersection / union
                intersection = matrix[:, i].multiply(matrix[:, j]).sum()
                union = matrix[:, i].multiply(~matrix[:, j]).sum() + matrix[:, j].multiply(~matrix[:, i]).sum() + intersection
                jaccard = intersection / union if union > 0 else 0
                similarity[i, j] = 1 - jaccard  # Distance
                similarity[j, i] = 1 - jaccard
            similarity[i, i] = 0  # Distance to self is 0
    
    return similarity

def analyze_cooccurrence(df: pd.DataFrame, code_counts: Dict[str, int], top_n: int, output_dir: str):
    """Analyze co-occurrence between top codes using Jaccard similarity."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top codes (limit to 30 for reasonable heatmap size)
    effective_top_n = min(top_n, 30)
    top_codes = [code for code, _ in sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:effective_top_n]]
    code_to_idx = {code: idx for idx, code in enumerate(top_codes)}
    
    print(f"Computing co-occurrence for top {effective_top_n} codes...")
    
    # Create binary matrix: rows=visits, cols=top_codes
    matrix = np.zeros((len(df), len(top_codes)), dtype=bool)
    for visit_idx, codes in enumerate(df['formatted_codes']):
        for code in codes:
            if code in code_to_idx:
                matrix[visit_idx, code_to_idx[code]] = True
    
    sparse_matrix = csr_matrix(matrix)
    
    # Calculate Jaccard similarity using our custom function
    similarity = calculate_jaccard_similarity(sparse_matrix)
    
    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity, xticklabels=top_codes, yticklabels=top_codes, cmap='YlGnBu', cbar_kws={'label': 'Jaccard Distance'})
    plt.title(f'Jaccard Distance Matrix of Top {effective_top_n} ICD-9 Codes')
    plt.xlabel('ICD-9 Codes')
    plt.ylabel('ICD-9 Codes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cooccurrence_heatmap.png"), dpi=300)
    plt.close()
    
    # Save top co-occurrences (similarity matrix)
    cooc_df = pd.DataFrame(similarity, index=top_codes, columns=top_codes)
    cooc_df.to_csv(os.path.join(output_dir, "cooccurrence_matrix.csv"))
    
    # Additional analysis: Top pairwise co-occurrences
    upper_triangle = np.triu(cooc_df.values, k=1)
    pairs = np.where(upper_triangle < 0.1)  # Top co-occurring pairs (low distance = high similarity)
    top_pairs = []
    for i, j in zip(pairs[0], pairs[1]):
        if i != j:
            pair_score = 1 - upper_triangle[i, j]  # Convert back to similarity
            top_pairs.append((cooc_df.index[i], cooc_df.columns[j], pair_score))
    
    # Sort by similarity (highest first)
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs_df = pd.DataFrame(top_pairs[:20], columns=['Code1', 'Code2', 'Jaccard_Similarity'])
    top_pairs_df.to_csv(os.path.join(output_dir, "top_cooccurring_pairs.csv"), index=False)
    print(f"Top 20 co-occurring pairs saved to {output_dir}/top_cooccurring_pairs.csv")

def analyze_sparsity_and_imbalance(code_counts: Dict[str, int], all_codes: List[str], icd9_pickle: str, output_dir: str):
    """Analyze sparsity (coverage) and class imbalance metrics with more plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        icd9_df = pd.read_pickle(icd9_pickle)
        all_possible = set(icd9_df['icd_code'].apply(format_icd9_properly))
        all_possible = {c for c in all_possible if is_valid_icd9(c)}
        actual_codes = set(all_codes)
        coverage_pct = (len(actual_codes) / len(all_possible)) * 100
        
        stats = {
            'used_codes': len(actual_codes),
            'total_possible': len(all_possible),
            'coverage_pct': coverage_pct
        }
        pd.DataFrame([stats]).to_csv(os.path.join(output_dir, "sparsity_stats.csv"), index=False)
        
        # Pie plot for used vs unused
        plt.figure(figsize=(8, 8))
        plt.pie([len(actual_codes), len(all_possible) - len(actual_codes)], labels=['Used', 'Unused'], autopct='%1.1f%%')
        plt.title('ICD-9 Code Coverage')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sparsity_pie.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not analyze sparsity: {e}")
    
    # Imbalance: Gini coefficient
    frequencies = np.array(list(code_counts.values()))
    frequencies = frequencies / frequencies.sum()
    gini = 1 - np.sum(frequencies**2)
    
    # Long-tail: codes needed for coverage thresholds
    sorted_freqs = sorted(code_counts.values(), reverse=True)
    cumsum = np.cumsum(sorted_freqs) / sum(sorted_freqs)
    thresholds = [0.5, 0.8, 0.9, 0.95]
    codes_needed = [np.where(cumsum >= t)[0][0] + 1 for t in thresholds]
    
    imbalance_df = pd.DataFrame({
        'Metric': ['Gini Coefficient'] + [f'Codes for {int(t*100)}% Coverage' for t in thresholds],
        'Value': [gini] + codes_needed
    })
    imbalance_df.to_csv(os.path.join(output_dir, "imbalance_metrics.csv"), index=False)
    
    # Log-log plot for power law distribution
    ranks = range(1, len(sorted_freqs) + 1)
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, sorted_freqs, 'b-', alpha=0.7)
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Power Law Distribution of ICD-9 Code Frequencies')
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power_law_distribution.png"), dpi=300)
    plt.close()

def analyze_category_level(all_codes: List[str], top_n: int, icd9_pickle: str, output_dir: str):
    """Analyze top K codes at the 3-digit category level of hierarchy with more plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parent categories (assuming get_icd9_parent returns 3-digit code)
    parent_codes = [get_icd9_parent(code) for code in all_codes]
    parent_counts = Counter(parent_codes)
    
    # Ensure integer counts
    for code in parent_counts:
        parent_counts[code] = int(parent_counts[code])
    
    # Get top K parents
    top_parents = sorted(parent_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    total_occurrences = sum(parent_counts.values())
    
    # Load descriptions if available (use parent code for lookup)
    try:
        icd9_df = pd.read_pickle(icd9_pickle)
        code_to_desc = dict(zip(
            icd9_df['icd_code'].apply(format_icd9_properly),
            icd9_df['long_title']
        ))
    except Exception as e:
        print(f"Could not load ICD-9 descriptions: {e}")
        code_to_desc = {}
    
    top_df = pd.DataFrame({
        'Category_Code': [parent for parent, _ in top_parents],
        'Aggregated_Frequency': [freq for _, freq in top_parents],
        'Percentage': [(freq / total_occurrences * 100) for _, freq in top_parents],
        'Description': [code_to_desc.get(parent, "N/A") for parent, _ in top_parents],
        'Chapter': [get_icd9_chapter(parent) for parent, _ in top_parents]
    })
    top_df.to_csv(os.path.join(output_dir, f"top_{top_n}_category_levels.csv"), index=False)
    
    # Bar plot for top categories
    plt.figure(figsize=(14, 10))
    sns.barplot(data=top_df, x='Aggregated_Frequency', y='Category_Code', orient='h')
    plt.title(f'Top {top_n} ICD-9 Codes at 3-Digit Category Level (Bar)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"top_{top_n}_category_levels_bar.png"), dpi=300)
    plt.close()
    
    # Pie plot for top 10 categories
    if len(top_df) >= 10:
        top10_df = top_df.head(10)
        plt.figure(figsize=(10, 10))
        plt.pie(top10_df['Percentage'], labels=top10_df['Category_Code'], autopct='%1.1f%%', textprops={'fontsize': 8})
        plt.title('Percentage Distribution of Top 10 3-Digit Categories')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_10_categories_pie.png"), dpi=300)
        plt.close()

def get_4_digit_parent(code: str) -> str:
    """Get the 4-digit level parent for grouping (XXX.X)."""
    if '.' in code:
        base, sub = code.split('.')
        if len(sub) > 1:
            return base + '.' + sub[0]
        else:
            return code
    return code

def analyze_subcategory_level(all_codes: List[str], top_n: int, icd9_pickle: str, output_dir: str):
    """Analyze top K codes at the 4-digit subcategory level of hierarchy."""
    os.makedirs(output_dir, exist_ok=True)
    
    subcat_codes = [get_4_digit_parent(code) for code in all_codes]
    subcat_counts = Counter(subcat_codes)
    
    # Ensure integer counts
    for code in subcat_counts:
        subcat_counts[code] = int(subcat_counts[code])
    
    # Get top K
    top_subcats = sorted(subcat_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    total_occurrences = sum(subcat_counts.values())
    
    # Load descriptions
    try:
        icd9_df = pd.read_pickle(icd9_pickle)
        code_to_desc = dict(zip(
            icd9_df['icd_code'].apply(format_icd9_properly),
            icd9_df['long_title']
        ))
    except Exception as e:
        print(f"Could not load ICD-9 descriptions: {e}")
        code_to_desc = {}
    
    top_df = pd.DataFrame({
        'Subcategory_Code': [subcat for subcat, _ in top_subcats],
        'Aggregated_Frequency': [freq for _, freq in top_subcats],
        'Percentage': [(freq / total_occurrences * 100) for _, freq in top_subcats],
        'Description': [code_to_desc.get(subcat, "N/A") for subcat, _ in top_subcats],
        'Chapter': [get_icd9_chapter(subcat) for subcat, _ in top_subcats]
    })
    top_df.to_csv(os.path.join(output_dir, f"top_{top_n}_subcategory_levels.csv"), index=False)
    
    # Bar plot
    plt.figure(figsize=(14, 10))
    sns.barplot(data=top_df, x='Aggregated_Frequency', y='Subcategory_Code', orient='h')
    plt.title(f'Top {top_n} ICD-9 Codes at 4-Digit Subcategory Level')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"top_{top_n}_subcategory_levels_bar.png"), dpi=300)
    plt.close()
    
    # Pie plot for top 10
    if len(top_df) >= 10:
        top10_df = top_df.head(10)
        plt.figure(figsize=(10, 10))
        plt.pie(top10_df['Percentage'], labels=top10_df['Subcategory_Code'], autopct='%1.1f%%', textprops={'fontsize': 8})
        plt.title('Percentage Distribution of Top 10 4-Digit Subcategories')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_10_subcategories_pie.png"), dpi=300)
        plt.close()

def analyze_codes_per_chapter(df: pd.DataFrame, all_codes: List[str], output_dir: str):
    """Analyze distribution of codes per chapter per visit."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get chapters for all codes
    chapters = [get_icd9_chapter(code) for code in all_codes]
    
    # Per visit: count codes per chapter
    chapter_per_visit = []
    for codes in df['formatted_codes']:
        visit_chapters = [get_icd9_chapter(code) for code in codes]
        chapter_per_visit.append(Counter(visit_chapters))
    
    # Create DF with chapters as columns, visits as rows
    unique_chapters = list(set(chapters))
    per_visit_df = pd.DataFrame(0, index=range(len(df)), columns=unique_chapters)
    for i, counter in enumerate(chapter_per_visit):
        for ch, count in counter.items():
            per_visit_df.at[i, ch] = int(count)
    
    # Save stats
    chapter_stats = per_visit_df.describe().T
    chapter_stats.to_csv(os.path.join(output_dir, "codes_per_chapter_stats.csv"))
    
    # Boxplot for codes per chapter across visits
    plt.figure(figsize=(16, 10))
    melt_df = per_visit_df.melt(var_name='Chapter', value_name='Count')
    sns.boxplot(data=melt_df, x='Count', y='Chapter', orient='h')
    plt.title('Distribution of Codes per Chapter per Visit')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "codes_per_chapter_box.png"), dpi=300)
    plt.close()

def analyze_rare_codes(code_counts: Dict[str, int], top_n: int, output_dir: str):
    """Analyze rare codes: bottom N, singletons, etc."""
    os.makedirs(output_dir, exist_ok=True)
    
    rare_codes = sorted(code_counts.items(), key=lambda x: x[1])[:top_n]  # Least frequent
    singletons = sum(1 for _, freq in code_counts.items() if freq == 1)
    low_freq = sum(1 for _, freq in code_counts.items() if freq <= 5)
    
    stats = {
        'singletons': singletons,
        'low_freq_leq5': low_freq,
        'total_unique': len(code_counts)
    }
    pd.DataFrame([stats]).to_csv(os.path.join(output_dir, "rare_codes_stats.csv"), index=False)
    
    rare_df = pd.DataFrame({
        'Code': [code for code, _ in rare_codes],
        'Frequency': [freq for _, freq in rare_codes]
    })
    rare_df.to_csv(os.path.join(output_dir, f"bottom_{top_n}_codes.csv"), index=False)
    
    # Bar plot for bottom N
    plt.figure(figsize=(14, 10))
    sns.barplot(data=rare_df, x='Frequency', y='Code', orient='h')
    plt.title(f'Bottom {top_n} Least Frequent ICD-9 Codes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"bottom_{top_n}_codes_bar.png"), dpi=300)
    plt.close()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    all_df = load_and_prepare_data(args)
    
    # Get distributions
    code_counts, all_codes = get_code_distribution(all_df)
    print(f"Unique ICD-9 codes: {len(code_counts)}")
    print(f"Total occurrences: {sum(code_counts.values())}")
    
    # Analyses
    analyze_basic_statistics(all_df, code_counts, all_codes, args.output_dir)
    analyze_top_codes(code_counts, args.top_n, args.icd9_pickle, args.output_dir)
    analyze_hierarchy(all_codes, args.output_dir)
    analyze_cooccurrence(all_df, code_counts, args.top_n, args.output_dir)
    analyze_sparsity_and_imbalance(code_counts, all_codes, args.icd9_pickle, args.output_dir)
    analyze_category_level(all_codes, args.top_n, args.icd9_pickle, args.output_dir)
    analyze_subcategory_level(all_codes, args.top_n, args.icd9_pickle, args.output_dir)
    analyze_codes_per_chapter(all_df, all_codes, args.output_dir)
    analyze_rare_codes(code_counts, args.top_n, args.output_dir)
    
    print(f"Improved analysis complete. Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()

    