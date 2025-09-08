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

# Import necessary functions from the finetuning script
import sys
sys.path.append('/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen')
from finetune_llama_gen_difftrainsize_improved import (
    format_icd9_properly, is_valid_icd9, get_icd9_parent
)

def get_args():
    ap = argparse.ArgumentParser(description="Analyze ICD-9 code distribution in MIMIC dataset")
    ap.add_argument("--train_pickle", default=None, help="Path to training data pickle")
    ap.add_argument("--val_pickle", default=None, help="Path to validation data pickle")
    ap.add_argument("--test_pickle", default=None, help="Path to test data pickle")
    ap.add_argument("--data_pickle", default=None, help="Path to combined data pickle")
    ap.add_argument("--icd9_pickle", default="MasterThesis/dataset/codes/icd9.pkl",
                    help="Path to ICD-9 code reference")
    ap.add_argument("--label_col", default="icd_code", help="Column containing ICD codes")
    ap.add_argument("--output_dir", default="icd9_analysis2",
                    help="Directory to save analysis outputs")
    ap.add_argument("--top_n", type=int, default=20, 
                    help="Number of top codes to display")
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
        train_df = all_df  # Just for consistency
        print(f"Loaded combined data: {len(all_df)} samples")
    else:
        raise ValueError("Provide either --data_pickle OR all of --train_pickle/--val_pickle/--test_pickle")
    
    return train_df, all_df

def extract_all_codes(df: pd.DataFrame, label_col: str) -> List[str]:
    """Extract all ICD-9 codes from the dataset with proper formatting."""
    all_codes = []
    for code_list in df[label_col]:
        formatted_codes = []
        for code in code_list:
            formatted = format_icd9_properly(str(code))
            if is_valid_icd9(formatted):
                formatted_codes.append(formatted)
        all_codes.extend(formatted_codes)
    return all_codes

def get_code_distribution(codes: List[str]) -> Dict[str, int]:
    """Count frequency of each ICD-9 code."""
    return Counter(codes)

def plot_code_frequency_histogram(code_counts: Dict[str, int], output_dir: str):
    """Plot histogram of code frequencies."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get frequency values
    frequencies = list(code_counts.values())
    
    plt.figure(figsize=(12, 6))
    plt.hist(frequencies, bins=50, log=True, alpha=0.7)
    plt.xlabel('Code Frequency')
    plt.ylabel('Number of Unique ICD-9 Codes (log scale)')
    plt.title('Distribution of ICD-9 Code Frequencies')
    plt.grid(True, alpha=0.3)
    
    # Add statistics to plot
    total_codes = len(code_counts)
    total_occurrences = sum(frequencies)
    median_freq = np.median(frequencies)
    mean_freq = np.mean(frequencies)
    
    stats_text = f"Total unique codes: {total_codes}\n"
    stats_text += f"Total code occurrences: {total_occurrences}\n"
    stats_text += f"Mean frequency: {mean_freq:.1f}\n"
    stats_text += f"Median frequency: {median_freq:.1f}"
    
    plt.annotate(stats_text, xy=(0.70, 0.75), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "icd9_frequency_histogram.png"), dpi=300)
    plt.close()
    
    # Also create cumulative distribution plot
    plt.figure(figsize=(12, 6))
    sorted_freqs = sorted(frequencies, reverse=True)
    cumulative_percent = np.cumsum(sorted_freqs) / sum(sorted_freqs) * 100
    
    plt.plot(range(1, len(sorted_freqs) + 1), cumulative_percent, 'b-', linewidth=2)
    plt.xlabel('Number of Top Codes')
    plt.ylabel('Cumulative Percentage of All Occurrences')
    plt.title('Cumulative Distribution of ICD-9 Codes')
    plt.grid(True, alpha=0.3)
    
    # Mark key percentages
    for pct in [50, 80, 90, 95]:
        idx = np.where(cumulative_percent >= pct)[0][0]
        plt.scatter(idx + 1, pct, color='red', s=50)
        plt.annotate(f"{pct}% ({idx + 1} codes)", 
                     xy=(idx + 1, pct),
                     xytext=(idx + 50, pct + 2),
                     arrowprops=dict(arrowstyle="->", color="black"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "icd9_cumulative_distribution.png"), dpi=300)
    plt.close()

def plot_top_codes(code_counts: Dict[str, int], top_n: int, icd9_pickle: str, output_dir: str):
    """Plot the top N most frequent ICD-9 codes with descriptions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to load ICD-9 descriptions if available
    try:
        icd9_df = pd.read_pickle(icd9_pickle)
        code_to_desc = dict(zip(
            icd9_df['icd_code'].apply(format_icd9_properly),
            icd9_df['long_title']
        ))
    except Exception as e:
        print(f"Could not load ICD-9 descriptions: {e}")
        code_to_desc = {}
    
    # Get top N codes
    top_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    codes = [code for code, _ in top_codes]
    freqs = [freq for _, freq in top_codes]
    
    # Get descriptions (use code itself if no description available)
    descriptions = []
    for code in codes:
        if code in code_to_desc:
            # Truncate long descriptions
            desc = code_to_desc[code]
            if len(desc) > 50:
                desc = desc[:47] + "..."
            descriptions.append(f"{code}: {desc}")
        else:
            descriptions.append(code)
    
    # Calculate total percentage these top codes represent
    total_occurrences = sum(code_counts.values())
    top_occurrences = sum(freqs)
    top_percentage = (top_occurrences / total_occurrences) * 100
    
    # Create horizontal bar chart
    plt.figure(figsize=(14, 10))
    bars = plt.barh(range(len(freqs)), freqs, align='center', alpha=0.7)
    plt.yticks(range(len(freqs)), descriptions, fontsize=10)
    plt.xlabel('Frequency')
    plt.title(f'Top {top_n} Most Common ICD-9 Codes\n'
              f'(Representing {top_percentage:.1f}% of all code occurrences)')
    
    # Add frequency values at the end of each bar
    for i, (bar, freq) in enumerate(zip(bars, freqs)):
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                 f"{freq} ({(freq/total_occurrences*100):.1f}%)",
                 va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"top_{top_n}_icd9_codes.png"), dpi=300)
    plt.close()
    
    # Save top codes as CSV for reference
    top_df = pd.DataFrame({
        'ICD9_Code': codes,
        'Frequency': freqs,
        'Percentage': [(f/total_occurrences*100) for f in freqs],
        'Description': [code_to_desc.get(code, "N/A") for code in codes]
    })
    top_df.to_csv(os.path.join(output_dir, f"top_{top_n}_icd9_codes.csv"), index=False)

def analyze_codes_per_visit(df: pd.DataFrame, label_col: str, output_dir: str):
    """Analyze and plot the distribution of codes per patient visit."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Count codes per visit
    codes_per_visit = df[label_col].apply(lambda x: len([
        c for c in x if is_valid_icd9(format_icd9_properly(str(c)))
    ]))
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(codes_per_visit, bins=30, kde=True)
    plt.xlabel('Number of ICD-9 Codes per Visit')
    plt.ylabel('Number of Visits')
    plt.title('Distribution of ICD-9 Codes per Patient Visit')
    
    # Add statistics
    stats_text = f"Mean: {codes_per_visit.mean():.1f} codes\n"
    stats_text += f"Median: {codes_per_visit.median():.1f} codes\n"
    stats_text += f"Min: {codes_per_visit.min()} codes\n"
    stats_text += f"Max: {codes_per_visit.max()} codes"
    
    plt.annotate(stats_text, xy=(0.70, 0.75), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "codes_per_visit_distribution.png"), dpi=300)
    plt.close()
    
    # Create table of percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(codes_per_visit, percentiles)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(percentiles)), percentile_values)
    plt.xlabel('Percentile')
    plt.ylabel('Number of Codes')
    plt.title('Number of ICD-9 Codes per Visit at Different Percentiles')
    plt.xticks(range(len(percentiles)), [f"{p}th" for p in percentiles])
    
    for i, val in enumerate(percentile_values):
        plt.text(i, val + 0.5, f"{val:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "codes_percentiles.png"), dpi=300)
    plt.close()

def get_icd9_chapter(code: str) -> str:
    """Return the official ICD-9-CM chapter name for a given code."""
    if not code:
        return "Unknown"
    
    # V codes (Supplementary)
    if code.startswith('V'):
        return "V: Supplementary Classification"
    
    # E codes (External Causes)
    if code.startswith('E'):
        return "E: External Causes of Injury"
    
    # Regular diagnosis codes (numbered chapters)
    if code[0].isdigit():
        first_three = code[:3] if len(code) >= 3 else code
        try:
            first_three_num = int(first_three)
            
            # Match official ICD-9-CM chapter ranges
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

def analyze_code_hierarchy(all_codes: List[str], output_dir: str):
    """Analyze the distribution of ICD-9 code categories/families using official chapters."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get chapters for each code
    chapters = [get_icd9_chapter(code) for code in all_codes]
    chapter_counts = Counter(chapters)
    
    # Also get parent codes for more detailed analysis
    parent_codes = [get_icd9_parent(code) for code in all_codes]
    parent_counts = Counter(parent_codes)
    
    # Plot chapters
    sorted_chapters = sorted(chapter_counts.items(), key=lambda x: x[1], reverse=True)
    chapter_names = [chapter for chapter, _ in sorted_chapters]
    chapter_values = [count for _, count in sorted_chapters]
    
    plt.figure(figsize=(16, 10))
    bars = plt.bar(range(len(chapter_names)), chapter_values, color='skyblue')
    plt.xticks(range(len(chapter_names)), chapter_names, rotation=45, ha='right', fontsize=9)
    plt.xlabel('ICD-9-CM Chapters')
    plt.ylabel('Frequency')
    plt.title('Distribution of ICD-9 Codes by Official Chapter')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add percentage values on top of bars
    total_codes = sum(chapter_values)
    for i, (bar, count) in enumerate(zip(bars, chapter_values)):
        percentage = (count / total_codes) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{percentage:.1f}%", ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "icd9_chapters.png"), dpi=300)
    plt.close()
    
    # Create pie chart of chapter distribution (easier to see proportions)
    plt.figure(figsize=(12, 10))
    wedges, texts, autotexts = plt.pie(
        chapter_values, 
        labels=None,
        autopct='%1.1f%%', 
        startangle=90,
        shadow=False
    )
    
    # Enhance the appearance
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')
    
    # Add a legend
    plt.legend(
        wedges, 
        [f"{name} ({count})" for name, count in zip(chapter_names, chapter_values)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8
    )
    
    plt.title('Distribution of ICD-9 Codes by Official Chapter')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "icd9_chapters_pie.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot top parent codes for more granular analysis
    top_parents = parent_counts.most_common(20)
    parent_labels = [code for code, _ in top_parents]
    parent_values = [count for _, count in top_parents]
    
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(parent_labels)), parent_values)
    plt.xticks(range(len(parent_labels)), parent_labels, rotation=45, ha='right')
    plt.xlabel('Parent Codes')
    plt.ylabel('Frequency')
    plt.title('Top 20 Most Common ICD-9 Parent Categories')
    
    # Add percentage labels
    total_parent_codes = sum(parent_values)
    for i, value in enumerate(parent_values):
        percentage = (value / total_parent_codes) * 100
        plt.text(i, value + (max(parent_values) * 0.01), 
                f"{percentage:.1f}%", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_parent_codes.png"), dpi=300)
    plt.close()
    
    # Generate a more detailed subcategory analysis for top chapters
    try:
        # Get the top 3 chapters
        top_chapters = [chapter for chapter, _ in sorted_chapters[:3]]
        
        for chapter in top_chapters:
            # Get all codes in this chapter
            chapter_codes = [code for code, ch in zip(all_codes, chapters) if ch == chapter]
            
            if not chapter_codes:
                continue
                
            # Group by parent codes within this chapter
            subcategories = Counter([get_icd9_parent(code) for code in chapter_codes])
            top_subcats = subcategories.most_common(15)
            
            if not top_subcats:
                continue
                
            # Create visualization
            subcat_labels = [code for code, _ in top_subcats]
            subcat_values = [count for _, count in top_subcats]
            
            plt.figure(figsize=(14, 8))
            plt.bar(range(len(subcat_labels)), subcat_values)
            plt.xticks(range(len(subcat_labels)), subcat_labels, rotation=45, ha='right')
            plt.xlabel('Subcategory Code')
            plt.ylabel('Frequency')
            plt.title(f'Top 15 Subcategories in "{chapter}"')
            plt.tight_layout()
            
            # Format chapter name for filename
            safe_chapter = chapter.split(':')[0].replace('-', 'to')
            plt.savefig(os.path.join(output_dir, f"subcategories_{safe_chapter}.png"), dpi=300)
            plt.close()
            
    except Exception as e:
        print(f"Could not generate subcategory analysis: {e}")

def analyze_code_sparsity(all_codes: List[str], icd9_pickle: str, output_dir: str):
    """Analyze how many potential ICD-9 codes are actually used in the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference ICD-9 codes
    try:
        icd9_df = pd.read_pickle(icd9_pickle)
        all_possible_codes = set(icd9_df['icd_code'].apply(format_icd9_properly))
        all_possible_codes = {c for c in all_possible_codes if is_valid_icd9(c)}
        
        # Get actual codes used in dataset
        actual_codes = set(all_codes)
        
        # Calculate coverage
        coverage_pct = (len(actual_codes) / len(all_possible_codes)) * 100
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.bar(["Used", "Unused"], 
                [len(actual_codes), len(all_possible_codes) - len(actual_codes)],
                color=['green', 'red'])
        plt.ylabel('Number of ICD-9 Codes')
        plt.title(f'ICD-9 Code Coverage\n{len(actual_codes)} of {len(all_possible_codes)} codes used ({coverage_pct:.1f}%)')
        
        # Add values on bars
        plt.text(0, len(actual_codes)/2, f"{len(actual_codes)}\n({coverage_pct:.1f}%)", 
                 ha='center', color='white', fontweight='bold')
        plt.text(1, (len(all_possible_codes) - len(actual_codes))/2, 
                 f"{len(all_possible_codes) - len(actual_codes)}\n({100-coverage_pct:.1f}%)", 
                 ha='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "icd9_coverage.png"), dpi=300)
        plt.close()
        
        # Create visualization showing chapter-wise coverage
        chapter_coverage = {}
        
        # Group all possible codes by chapter
        for code in all_possible_codes:
            chapter = get_icd9_chapter(code)
            if chapter not in chapter_coverage:
                chapter_coverage[chapter] = {'total': 0, 'used': 0}
            chapter_coverage[chapter]['total'] += 1
        
        # Count used codes by chapter
        for code in actual_codes:
            chapter = get_icd9_chapter(code)
            if chapter in chapter_coverage:
                chapter_coverage[chapter]['used'] += 1
        
        # Calculate coverage percentage and prepare for plotting
        chapter_names = []
        coverage_percentages = []
        used_counts = []
        total_counts = []
        
        for chapter, counts in chapter_coverage.items():
            if counts['total'] > 0:  # Avoid division by zero
                chapter_names.append(chapter)
                coverage_pct = (counts['used'] / counts['total']) * 100
                coverage_percentages.append(coverage_pct)
                used_counts.append(counts['used'])
                total_counts.append(counts['total'])
        
        # Sort by coverage percentage
        sorted_indices = np.argsort(coverage_percentages)[::-1]  # Descending order
        chapter_names = [chapter_names[i] for i in sorted_indices]
        coverage_percentages = [coverage_percentages[i] for i in sorted_indices]
        used_counts = [used_counts[i] for i in sorted_indices]
        total_counts = [total_counts[i] for i in sorted_indices]
        
        # Plot chapter-wise coverage
        plt.figure(figsize=(14, 10))
        bars = plt.bar(range(len(chapter_names)), coverage_percentages)
        plt.xticks(range(len(chapter_names)), 
                   [c.split(':')[0] for c in chapter_names],  # Use shorter names for clarity
                   rotation=45, ha='right')
        plt.ylabel('Coverage Percentage')
        plt.title('ICD-9 Code Coverage by Chapter')
        
        # Add counts as text
        for i, (bar, used, total, pct) in enumerate(zip(bars, used_counts, total_counts, coverage_percentages)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f"{used}/{total}\n({pct:.1f}%)", 
                     ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "icd9_chapter_coverage.png"), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Could not analyze code sparsity: {e}")

def analyze_long_tail(code_counts: Dict[str, int], output_dir: str):
    """Analyze the long tail distribution of ICD-9 codes."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by frequency
    sorted_counts = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate cumulative coverage
    total_occurrences = sum(code_counts.values())
    running_sum = 0
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    code_counts_at_threshold = {}
    
    for i, (_, count) in enumerate(sorted_counts):
        running_sum += count
        percent = running_sum / total_occurrences
        
        for t in thresholds:
            if percent >= t and t not in code_counts_at_threshold:
                code_counts_at_threshold[t] = i + 1
    
    # Plot
    plt.figure(figsize=(10, 6))
    thresholds_pct = [t*100 for t in thresholds]
    code_counts_list = [code_counts_at_threshold.get(t, 0) for t in thresholds]
    
    plt.bar(thresholds_pct, code_counts_list, width=5)
    plt.xlabel('Coverage Percentage')
    plt.ylabel('Number of Codes Needed')
    plt.title('Number of Most Frequent ICD-9 Codes Needed for Different Coverage Levels')
    
    for i, (pct, count) in enumerate(zip(thresholds_pct, code_counts_list)):
        plt.text(pct, count + 10, f"{count}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "long_tail_analysis.png"), dpi=300)
    plt.close()
    
    # Create table with statistics
    stats_df = pd.DataFrame({
        'Coverage': [f"{t*100:.1f}%" for t in thresholds],
        'Codes_Required': code_counts_list,
        'Percentage_of_Total_Codes': [f"{count/len(code_counts)*100:.2f}%" for count in code_counts_list]
    })
    stats_df.to_csv(os.path.join(output_dir, "coverage_statistics.csv"), index=False)
    
    # Additional plot: frequency by code rank (log scale)
    ranks = range(1, len(sorted_counts) + 1)
    frequencies = [count for _, count in sorted_counts]
    
    plt.figure(figsize=(12, 6))
    plt.loglog(ranks, frequencies, 'b-', alpha=0.7)
    plt.xlabel('Code Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('ICD-9 Code Frequency by Rank (Power Law Distribution)')
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power_law_distribution.png"), dpi=300)
    plt.close()
    
    # Plot frequency distribution by ranges
    freq_ranges = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100), (101, 500), (501, 1000), (1001, float('inf'))]
    range_labels = ['1', '2-5', '6-10', '11-50', '51-100', '101-500', '501-1000', '1000+']
    range_counts = [0] * len(freq_ranges)
    
    for _, count in code_counts.items():
        for i, (min_f, max_f) in enumerate(freq_ranges):
            if min_f <= count <= max_f:
                range_counts[i] += 1
                break
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(range_labels)), range_counts)
    plt.xticks(range(len(range_labels)), range_labels)
    plt.xlabel('Frequency Range')
    plt.ylabel('Number of Codes')
    plt.title('Distribution of ICD-9 Codes by Frequency Range')
    
    # Add count and percentage labels
    total_codes = sum(range_counts)
    for i, count in enumerate(range_counts):
        percentage = (count / total_codes) * 100
        plt.text(i, count + 5, f"{count}\n({percentage:.1f}%)", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frequency_ranges.png"), dpi=300)
    plt.close()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_df, all_df = load_and_prepare_data(args)
    
    # Extract all codes with proper formatting
    all_codes = extract_all_codes(all_df, args.label_col)
    print(f"Total ICD-9 code occurrences: {len(all_codes)}")
    
    # Get code distribution
    code_counts = get_code_distribution(all_codes)
    print(f"Unique ICD-9 codes: {len(code_counts)}")
    
    # Generate all plots
    plot_code_frequency_histogram(code_counts, args.output_dir)
    plot_top_codes(code_counts, args.top_n, args.icd9_pickle, args.output_dir)
    analyze_codes_per_visit(all_df, args.label_col, args.output_dir)
    analyze_code_hierarchy(all_codes, args.output_dir)
    analyze_code_sparsity(all_codes, args.icd9_pickle, args.output_dir)
    analyze_long_tail(code_counts, args.output_dir)
    
    print(f"Analysis complete. Results saved to {args.output_dir}/")
    
    # Print summary findings
    rare_codes = sum(1 for freq in code_counts.values() if freq <= 5)
    rare_pct = (rare_codes / len(code_counts)) * 100
    
    print("\n===== ICD-9 Code Distribution Summary =====")
    print(f"Total unique ICD-9 codes: {len(code_counts)}")
    print(f"Rare codes (≤5 occurrences): {rare_codes} ({rare_pct:.1f}%)")
    print(f"Top 20 codes cover: {sum(sorted(code_counts.values(), reverse=True)[:20]) / len(all_codes) * 100:.1f}% of all occurrences")
    print(f"Average codes per patient visit: {all_df[args.label_col].apply(len).mean():.1f}")
    
    # Class imbalance metrics
    gini = 1 - sum((freq/len(all_codes))**2 for freq in code_counts.values())
    print(f"Gini impurity (measure of class imbalance): {gini:.3f} (higher = more imbalanced)")
    
    # Chapter distribution
    print("\n===== ICD-9 Chapter Distribution =====")
    chapters = [get_icd9_chapter(code) for code in all_codes]
    chapter_counts = Counter(chapters)
    for chapter, count in sorted(chapter_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(all_codes) * 100
        print(f"{chapter}: {count} codes ({pct:.1f}%)")

if __name__ == "__main__":
    main()