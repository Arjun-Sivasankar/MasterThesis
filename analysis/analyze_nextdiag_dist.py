#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze distribution of current vs. next diagnoses to inform prompt design.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def to_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        it = x.tolist() if hasattr(x, "tolist") else x
        return [v for v in it if v is not None and str(v).strip() and str(v).lower() not in ("nan","none")]
    return []

def analyze_distributions(pkl_path, output_prefix):
    """Analyze code count distributions."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {pkl_path}")
    print(f"{'='*80}\n")
    
    df = pd.read_pickle(pkl_path)
    print(f"Total samples: {len(df)}")
    
    # Extract code counts
    icd_counts = [len(to_list(row)) for row in df['icd_code']]
    next_6m_counts = [len(to_list(row)) for row in df['NEXT_DIAG_6M']]
    
    # Compute overlap (new diagnoses only)
    new_only_counts = []
    for _, row in df.iterrows():
        current = set(to_list(row.get('icd_code', [])))
        next_dx = set(to_list(row.get('NEXT_DIAG_6M', [])))
        new_only = next_dx - current
        new_only_counts.append(len(new_only))
    
    # Statistics
    stats = {
        'Current ICD (index visit)': {
            'mean': np.mean(icd_counts),
            'median': np.median(icd_counts),
            'std': np.std(icd_counts),
            'min': np.min(icd_counts),
            'max': np.max(icd_counts),
            'p25': np.percentile(icd_counts, 25),
            'p75': np.percentile(icd_counts, 75),
            'p90': np.percentile(icd_counts, 90),
            'p95': np.percentile(icd_counts, 95),
        },
        'NEXT_DIAG_6M (all)': {
            'mean': np.mean(next_6m_counts),
            'median': np.median(next_6m_counts),
            'std': np.std(next_6m_counts),
            'min': np.min(next_6m_counts),
            'max': np.max(next_6m_counts),
            'p25': np.percentile(next_6m_counts, 25),
            'p75': np.percentile(next_6m_counts, 75),
            'p90': np.percentile(next_6m_counts, 90),
            'p95': np.percentile(next_6m_counts, 95),
        },
        'NEXT_DIAG_6M (emergent only)': {
            'mean': np.mean(new_only_counts),
            'median': np.median(new_only_counts),
            'std': np.std(new_only_counts),
            'min': np.min(new_only_counts),
            'max': np.max(new_only_counts),
            'p25': np.percentile(new_only_counts, 25),
            'p75': np.percentile(new_only_counts, 75),
            'p90': np.percentile(new_only_counts, 90),
            'p95': np.percentile(new_only_counts, 95),
        }
    }
    
    # Print statistics
    for category, vals in stats.items():
        print(f"\n{category}:")
        print(f"  Mean:   {vals['mean']:.2f}")
        print(f"  Median: {vals['median']:.1f}")
        print(f"  Std:    {vals['std']:.2f}")
        print(f"  Range:  [{vals['min']:.0f}, {vals['max']:.0f}]")
        print(f"  P25/P75: [{vals['p25']:.0f}, {vals['p75']:.0f}]")
        print(f"  P90/P95: [{vals['p90']:.0f}, {vals['p95']:.0f}]")
    
    # Overlap statistics
    overlap_counts = [len(set(to_list(r['icd_code'])) & set(to_list(r['NEXT_DIAG_6M']))) 
                      for _, r in df.iterrows()]
    print(f"\n\nOverlap Statistics (codes in both current and next):")
    print(f"  Mean overlap:   {np.mean(overlap_counts):.2f} codes")
    print(f"  Median overlap: {np.median(overlap_counts):.1f} codes")
    print(f"  % with overlap: {100 * np.mean([x > 0 for x in overlap_counts]):.1f}%")
    
    # Ratio analysis
    ratios = [n / max(c, 1) for c, n in zip(icd_counts, next_6m_counts)]
    print(f"\n\nRatio (NEXT / CURRENT):")
    print(f"  Mean ratio: {np.mean(ratios):.2f}x")
    print(f"  Median ratio: {np.median(ratios):.2f}x")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram: Current ICD
    axes[0, 0].hist(icd_counts, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(icd_counts), color='red', linestyle='--', label=f'Mean: {np.mean(icd_counts):.1f}')
    axes[0, 0].axvline(np.median(icd_counts), color='blue', linestyle='--', label=f'Median: {np.median(icd_counts):.1f}')
    axes[0, 0].set_title('Current ICD Code Count Distribution')
    axes[0, 0].set_xlabel('Number of Codes')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram: Next 6M (all)
    axes[0, 1].hist(next_6m_counts, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(np.mean(next_6m_counts), color='red', linestyle='--', label=f'Mean: {np.mean(next_6m_counts):.1f}')
    axes[0, 1].axvline(np.median(next_6m_counts), color='blue', linestyle='--', label=f'Median: {np.median(next_6m_counts):.1f}')
    axes[0, 1].set_title('NEXT_DIAG_6M Code Count Distribution (All)')
    axes[0, 1].set_xlabel('Number of Codes')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram: Next 6M (emergent only)
    axes[1, 0].hist(new_only_counts, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(np.mean(new_only_counts), color='red', linestyle='--', label=f'Mean: {np.mean(new_only_counts):.1f}')
    axes[1, 0].axvline(np.median(new_only_counts), color='blue', linestyle='--', label=f'Median: {np.median(new_only_counts):.1f}')
    axes[1, 0].set_title('NEXT_DIAG_6M Code Count Distribution (Emergent Only)')
    axes[1, 0].set_xlabel('Number of Codes')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter: Current vs Next
    axes[1, 1].scatter(icd_counts, next_6m_counts, alpha=0.3, s=10)
    axes[1, 1].plot([0, max(icd_counts)], [0, max(icd_counts)], 'r--', label='y=x', linewidth=2)
    axes[1, 1].set_title('Current ICD vs NEXT_DIAG_6M')
    axes[1, 1].set_xlabel('Current ICD Count')
    axes[1, 1].set_ylabel('NEXT_DIAG_6M Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analysis/{output_prefix}_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n\nPlot saved: {output_prefix}_distribution.png")
    
    return stats

if __name__ == "__main__":
    import sys
    
    pkl_6m = "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/mimic_diag_6m.pkl"
    pkl_12m = "/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/mimic_diag_12m.pkl"
    
    print("\n" + "="*80)
    print("CODE DISTRIBUTION ANALYSIS FOR NEXT DIAGNOSIS PREDICTION")
    print("="*80)
    
    stats_6m = analyze_distributions(pkl_6m, "nextdiag_6m")
    stats_12m = analyze_distributions(pkl_12m, "nextdiag_12m")