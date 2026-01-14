"""
Streamlined ICD-9 Code Analysis for MIMIC Dataset
Generates essential plots and CSV files for thesis reporting
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

# Set professional plot style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_palette("husl")

# Import necessary functions from the finetuning script
sys.path.append('/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/pipeline')
from common_textgen import (
    format_icd9, is_valid_icd9, get_icd9_parent, clean_dataframe
)

def get_args():
    ap = argparse.ArgumentParser(description="Streamlined ICD-9 Code Analysis")
    ap.add_argument("--train_pickle", default=None, help="Path to training data pickle")
    ap.add_argument("--val_pickle", default=None, help="Path to validation data pickle")
    ap.add_argument("--test_pickle", default=None, help="Path to test data pickle")
    ap.add_argument("--data_pickle", default=None, help="Path to combined data pickle")
    ap.add_argument("--icd9_pickle", default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl",
                    help="Path to ICD-9 code reference")
    ap.add_argument("--label_col", default="icd_code", help="Column containing ICD codes")
    ap.add_argument("--output_dir", default="icd9_analysis",
                    help="Directory to save analysis outputs")
    ap.add_argument("--top_n", type=int, default=50, help="Number of top/bottom codes for analysis")
    ap.add_argument("--head_n", type=int, default=50, help="Number of head codes to highlight")
    ap.add_argument("--tail_n", type=int, default=50, help="Number of tail codes to highlight")
    return ap.parse_args()

def load_and_prepare_data(args):
    """Load data and extract all ICD-9 codes with proper formatting."""
    train_df = val_df = test_df = None
    
    if args.train_pickle and args.val_pickle and args.test_pickle:
        print("Loading train/val/test splits...")
        train_df = pickle.load(open(args.train_pickle, "rb"))
        val_df = pickle.load(open(args.val_pickle, "rb"))
        test_df = pickle.load(open(args.test_pickle, "rb"))
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        print(f"Loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
        train_df = clean_dataframe(train_df)
        val_df = clean_dataframe(val_df)
        test_df = clean_dataframe(test_df)
    elif args.data_pickle:
        print("Loading combined data...")
        all_df = pickle.load(open(args.data_pickle, "rb"))
        print(f"Loaded combined data: {len(all_df)} samples")
    else:
        raise ValueError("Provide either --data_pickle OR all of --train_pickle/--val_pickle/--test_pickle")
    
    print("Formatting ICD-9 codes...")
    all_df['formatted_codes'] = all_df[args.label_col].apply(
        lambda codes: [format_icd9(str(c)) for c in codes if is_valid_icd9(format_icd9(str(c)))]
    )
    
    if train_df is not None:
        train_df['formatted_codes'] = train_df[args.label_col].apply(
            lambda codes: [format_icd9(str(c)) for c in codes if is_valid_icd9(format_icd9(str(c)))]
        )
        val_df['formatted_codes'] = val_df[args.label_col].apply(
            lambda codes: [format_icd9(str(c)) for c in codes if is_valid_icd9(format_icd9(str(c)))]
        )
        test_df['formatted_codes'] = test_df[args.label_col].apply(
            lambda codes: [format_icd9(str(c)) for c in codes if is_valid_icd9(format_icd9(str(c)))]
        )
    
    return all_df, train_df, val_df, test_df

def get_code_distribution(df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    """Count frequency of each ICD-9 code across all visits."""
    all_codes = [code for code_list in df['formatted_codes'] for code in code_list]
    code_counts = Counter(all_codes)
    
    for code in code_counts:
        code_counts[code] = int(code_counts[code])
    
    return code_counts, all_codes

def plot_diagnosis_frequency_rank(code_counts: Dict[str, int], output_dir: str, head_n: int = 50, tail_n: int = 50):
    """Rank plot showing head-tail structure."""
    os.makedirs(output_dir, exist_ok=True)
    
    sorted_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)
    ranks = np.arange(1, len(sorted_codes) + 1)
    frequencies = [freq for _, freq in sorted_codes]
    
    total_occurrences = sum(frequencies)
    head_coverage = sum(frequencies[:head_n]) / total_occurrences * 100
    tail_coverage = sum(frequencies[-tail_n:]) / total_occurrences * 100
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(ranks, frequencies, 'b-', linewidth=1.5, alpha=0.8, label='All Codes')
    
    ax.axvspan(1, head_n, alpha=0.25, color='green', label=f'Head-{head_n} ({head_coverage:.1f}% coverage)')
    
    tail_start = len(sorted_codes) - tail_n + 1
    ax.axvspan(tail_start, len(sorted_codes), alpha=0.25, color='red', label=f'Tail-{tail_n} ({tail_coverage:.2f}% coverage)')
    
    ax.set_yscale('log')
    ax.set_xlabel('Diagnosis Rank', fontweight='bold')
    ax.set_ylabel('Global Frequency (log scale)', fontweight='bold')
    ax.set_title('Diagnosis Frequency Rank Plot with Head–Tail Structure', fontweight='bold', pad=15)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diag_frequency_rankplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    stats_df = pd.DataFrame({
        'Metric': [f'Head-{head_n} frequency range', f'Tail-{tail_n} frequency range', 
                   f'Head-{head_n} coverage (%)', f'Tail-{tail_n} coverage (%)'],
        'Value': [f"{frequencies[0]} to {frequencies[head_n-1]}", 
                  f"{frequencies[tail_start-1]} to {frequencies[-1]}",
                  f"{head_coverage:.2f}%", f"{tail_coverage:.2f}%"]
    })
    stats_df.to_csv(os.path.join(output_dir, "head_tail_statistics.csv"), index=False)
    
    print(f"[SAVED] Head-Tail rank plot: diag_frequency_rankplot.png")

def plot_power_law_distribution(code_counts: Dict[str, int], output_dir: str):
    """Log-log plot showing power-law distribution."""
    os.makedirs(output_dir, exist_ok=True)
    
    sorted_freqs = sorted(code_counts.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freqs) + 1)
    
    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_freqs)
    coeffs = np.polyfit(log_ranks, log_freqs, 1)
    fitted_freqs = np.exp(coeffs[1]) * ranks**coeffs[0]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(ranks, sorted_freqs, 'b.', markersize=4, alpha=0.6, label='Actual Data')
    ax.loglog(ranks, fitted_freqs, 'r--', linewidth=2.5, 
              label=f'Power-law Fit: y = {np.exp(coeffs[1]):.1f} * x^{coeffs[0]:.2f}')
    
    ax.set_xlabel('Code Rank (log scale)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontweight='bold', fontsize=12)
    ax.set_title('Power-Law Distribution of ICD-9 Codes (Zipf\'s Law)', fontweight='bold', pad=15, fontsize=14)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power_law_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Power-law distribution plot: power_law_distribution.png (exponent: {coeffs[0]:.3f})")

def export_top_codes(code_counts: Dict[str, int], top_n: int, icd9_pickle: str, output_dir: str):
    """Export top N codes to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(icd9_pickle, 'rb') as f:
            icd9_data = pickle.load(f)
            if isinstance(icd9_data, dict):
                code_to_desc = icd9_data
            else:
                code_to_desc = {code: desc for code, desc in icd9_data}
    except Exception as e:
        print(f"Warning: Could not load ICD-9 descriptions: {e}")
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
    
    print(f"[SAVED] Top {top_n} codes CSV: top_{top_n}_codes.csv")

def export_bottom_codes(code_counts: Dict[str, int], bottom_n: int, icd9_pickle: str, output_dir: str):
    """Export bottom N codes to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(icd9_pickle, 'rb') as f:
            icd9_data = pickle.load(f)
            if isinstance(icd9_data, dict):
                code_to_desc = icd9_data
            else:
                code_to_desc = {code: desc for code, desc in icd9_data}
    except Exception as e:
        print(f"Warning: Could not load ICD-9 descriptions: {e}")
        code_to_desc = {}
    
    bottom_codes = sorted(code_counts.items(), key=lambda x: x[1])[:bottom_n]
    total_occurrences = sum(code_counts.values())
    
    bottom_df = pd.DataFrame({
        'ICD9_Code': [code for code, _ in bottom_codes],
        'Frequency': [freq for _, freq in bottom_codes],
        'Percentage': [(freq / total_occurrences * 100) for _, freq in bottom_codes],
        'Description': [code_to_desc.get(code, "N/A") for code, _ in bottom_codes]
    })
    bottom_df.to_csv(os.path.join(output_dir, f"bottom_{bottom_n}_codes.csv"), index=False)
    
    singleton_count = sum(1 for freq in bottom_df['Frequency'] if freq == 1)
    print(f"[SAVED] Bottom {bottom_n} codes CSV: bottom_{bottom_n}_codes.csv (singletons: {singleton_count})")

def export_category_level_analysis(all_codes: List[str], top_n: int, icd9_pickle: str, output_dir: str):
    """Export category-level (3-digit) analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    parent_codes = [get_icd9_parent(code) for code in all_codes]
    parent_counts = Counter(parent_codes)
    parent_counts = {code: int(count) for code, count in parent_counts.items()}
    
    top_parents = sorted(parent_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    total_occurrences = sum(parent_counts.values())
    
    try:
        with open(icd9_pickle, 'rb') as f:
            icd9_data = pickle.load(f)
            if isinstance(icd9_data, dict):
                code_to_desc = icd9_data
            else:
                code_to_desc = {code: desc for code, desc in icd9_data}
    except Exception as e:
        print(f"Warning: Could not load ICD-9 descriptions: {e}")
        code_to_desc = {}
    
    top_df = pd.DataFrame({
        'Category_Code': [parent for parent, _ in top_parents],
        'Aggregated_Frequency': [freq for _, freq in top_parents],
        'Percentage': [(freq / total_occurrences * 100) for _, freq in top_parents],
        'Description': [code_to_desc.get(parent, "N/A") for parent, _ in top_parents]
    })
    top_df.to_csv(os.path.join(output_dir, f"top_{top_n}_category_levels.csv"), index=False)
    
    print(f"[SAVED] Top {top_n} category-level CSV: top_{top_n}_category_levels.csv")

def plot_diagnoses_per_admission(df: pd.DataFrame, output_dir: str):
    """Histogram of codes per admission."""
    os.makedirs(output_dir, exist_ok=True)
    
    codes_per_visit = df['formatted_codes'].apply(len)
    median_val = codes_per_visit.median()
    mean_val = codes_per_visit.mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(codes_per_visit, bins=40, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax.axvline(median_val, color='red', linestyle='--', linewidth=2.5, 
               label=f'Median = {median_val:.0f}', alpha=0.9)
    ax.axvline(mean_val, color='orange', linestyle='--', linewidth=2.5, 
               label=f'Mean = {mean_val:.1f}', alpha=0.9)
    
    ax.set_xlabel('Number of Diagnoses per Admission', fontweight='bold')
    ax.set_ylabel('Number of Admissions', fontweight='bold')
    ax.set_title('Distribution of Diagnoses per Admission', fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnoses_per_admission_hist.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Diagnoses per admission histogram: diagnoses_per_admission_hist.png")

def create_summary_statistics(df: pd.DataFrame, code_counts: Dict[str, int], output_dir: str):
    """Create summary statistics CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    frequencies = list(code_counts.values())
    codes_per_visit = df['formatted_codes'].apply(len)
    
    sorted_freqs = sorted(frequencies, reverse=True)
    cumsum = np.cumsum(sorted_freqs)
    total = cumsum[-1]
    
    idx_50 = np.argmax(cumsum >= total * 0.5) + 1
    idx_80 = np.argmax(cumsum >= total * 0.8) + 1
    
    summary = {
        'Metric': [
            'Total Admissions',
            'Total Unique Codes',
            'Total Code Occurrences',
            'Mean Codes per Admission',
            'Median Codes per Admission',
            'Singleton Codes (freq=1)',
            'Rare Codes (freq<=10)',
            'Codes for 50% Coverage',
            'Codes for 80% Coverage',
        ],
        'Value': [
            len(df),
            len(code_counts),
            sum(frequencies),
            f"{codes_per_visit.mean():.2f}",
            int(codes_per_visit.median()),
            sum(1 for f in frequencies if f == 1),
            sum(1 for f in frequencies if f <= 10),
            idx_50,
            idx_80,
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)
    
    print(f"[SAVED] Summary statistics CSV: summary_statistics.csv")

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("STREAMLINED ICD-9 CODE ANALYSIS")
    print("="*60 + "\n")
    
    # Load data
    all_df, train_df, val_df, test_df = load_and_prepare_data(args)
    
    # Get distributions
    code_counts, all_codes = get_code_distribution(all_df)
    print(f"\n[INFO] Dataset Overview:")
    print(f"  Unique ICD-9 codes: {len(code_counts):,}")
    print(f"  Total occurrences: {sum(code_counts.values()):,}")
    print(f"  Total admissions: {len(all_df):,}\n")
    
    print("Generating essential analysis outputs...\n")
    
    # 1. Rank plot
    plot_diagnosis_frequency_rank(code_counts, args.output_dir, args.head_n, args.tail_n)
    
    # 2. Power-law plot
    plot_power_law_distribution(code_counts, args.output_dir)
    
    # 3. Diagnoses per admission
    plot_diagnoses_per_admission(all_df, args.output_dir)
    
    # 4. Top codes CSV
    export_top_codes(code_counts, args.top_n, args.icd9_pickle, args.output_dir)
    
    # 5. Bottom codes CSV
    export_bottom_codes(code_counts, args.top_n, args.icd9_pickle, args.output_dir)
    
    # 6. Category-level CSV
    export_category_level_analysis(all_codes, args.top_n, args.icd9_pickle, args.output_dir)
    
    # 7. Summary statistics
    create_summary_statistics(all_df, code_counts, args.output_dir)
    
    print("\n" + "="*60)
    print(f"[COMPLETE] Analysis finished successfully!")
    print(f"  All outputs saved to: {args.output_dir}/")
    print("="*60 + "\n")
    
    print("Generated Files:")
    print("  [PLOTS]")
    print("     - diag_frequency_rankplot.png")
    print("     - power_law_distribution.png")
    print("     - diagnoses_per_admission_hist.png")
    print("\n  [CSV FILES]")
    print("     - top_50_codes.csv")
    print("     - bottom_50_codes.csv")
    print("     - top_50_category_levels.csv")
    print("     - head_tail_statistics.csv")
    print("     - summary_statistics.csv\n")

if __name__ == "__main__":
    main()