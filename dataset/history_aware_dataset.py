import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- config ---
mimic_lab   = "dataset/mimic_with_lab_mappings.pkl"
subject_col = "subject_id_x"
min_visits  = 2     
ratio_single = 0.15 
test_size   = 0.10
val_size    = 0.10
seed        = 42

# --- load ---
print("Loading dataset...")
df = pd.read_pickle(mimic_lab)
print(f"Original dataset: {len(df):,} rows, {df[subject_col].nunique():,} patients")

# --- filter out rows with NaN Chief Complaint ---
print("\nFiltering out rows with NaN Chief Complaint...")
df_with_cc = df[df['Chief Complaint'].notna()].copy()
print(f"After filtering NaN Chief Complaint: {len(df_with_cc):,} rows ({len(df_with_cc)/len(df)*100:.1f}% of original)")

# --- SEPARATE COHORTS ---
print("\nSeparating cohorts...")
visit_counts = df_with_cc[subject_col].value_counts()

# Cohort: Multi-Visit
multi_visit_ids = visit_counts[visit_counts >= min_visits].index
df_multi = df_with_cc[df_with_cc[subject_col].isin(multi_visit_ids)].copy()
s_visit_ids = visit_counts[visit_counts == 1].index

# --- SAMPLE SINGLE VISIT COHORT ---
target_single_count = int(len(multi_visit_ids) * ratio_single)

# Randomly sample subjects
np.random.seed(seed)
sampled_single_ids = np.random.choice(s_visit_ids, size=target_single_count, replace=False)
df_single = df_with_cc[df_with_cc[subject_col].isin(sampled_single_ids)].copy()

print(f"\n=== Cohort Stats ===")
print(f"1. Multi-Visit (Core):   {len(df_multi):,} rows | {len(multi_visit_ids):,} patients")
print(f"2. Single-Visit (Noise): {len(df_single):,} rows | {len(sampled_single_ids):,} patients")
print(f"   (Sampled {ratio_single*100}% of multi-visit patient count)")
print(f"Total Dataset Size:      {len(df_multi) + len(df_single):,} rows")

# --- subject-based split function ---
def subject_splits(df: pd.DataFrame, subject_col: str,
                   test_size=0.10, val_size=0.10, seed=42):
    """
    Split dataframe by subjects (patients) to prevent data leakage.
    """
    subs = df[subject_col].dropna().unique()
    
    # Split subjects first
    train_subs, test_subs = train_test_split(
        subs, test_size=test_size, random_state=seed
    )
    train_subs, val_subs = train_test_split(
        train_subs, test_size=val_size/(1-test_size), random_state=seed
    )

    # Filter dataframe by subject sets
    tr = df[df[subject_col].isin(train_subs)].copy()
    va = df[df[subject_col].isin(val_subs)].copy()
    te = df[df[subject_col].isin(test_subs)].copy()

    # Safety: no subject overlaps
    set_tr, set_va, set_te = set(train_subs), set(val_subs), set(test_subs)
    assert not (set(train_subs) & set(val_subs)), "Leakage between TRAIN and VAL."
    assert not (set(train_subs) & set(test_subs)), "Leakage between TRAIN and TEST."
    assert not (set(val_subs) & set(test_subs)),  "Leakage between VAL and TEST."

    # detailed report
    total_rows = len(df)
    total_subs = len(subs)
    n_tr, n_va, n_te = len(tr), len(va, ), len(te)
    n_tr_subs, n_va_subs, n_te_subs = len(set_tr), len(set_va), len(set_te)

    print("=== Subject-based Split Report ===")
    print(f"Total rows:     {total_rows:,}")
    print(f"Total subjects: {total_subs:,}\n")
    print(f"TRAIN: rows={n_tr:,}  ({n_tr/total_rows:6.2%}) | subjects={n_tr_subs:,}  ({n_tr_subs/total_subs:6.2%})")
    print(f"VAL:   rows={n_va:,}  ({n_va/total_rows:6.2%}) | subjects={n_va_subs:,}  ({n_va_subs/total_subs:6.2%})")
    print(f"TEST:  rows={n_te:,}  ({n_te/total_rows:6.2%}) | subjects={n_te_subs:,}  ({n_te_subs/total_subs:6.2%})")

    return tr, va, te

# --- PERFORM SPLITS SEPARATELY ---
# This ensures we have both history-rich and cold-start patients in Train, Val, AND Test
print("\nSplitting Multi-Visit Cohort...")
tr_m, va_m, te_m = subject_splits(df_multi, subject_col, test_size, val_size, seed)

print("Splitting Single-Visit Cohort...")
tr_s, va_s, te_s = subject_splits(df_single, subject_col, test_size, val_size, seed)

# --- MERGE ---
print("\nMerging Cohorts...")
train_df = pd.concat([tr_m, tr_s]).sample(frac=1, random_state=seed).reset_index(drop=True)
val_df   = pd.concat([va_m, va_s]).sample(frac=1, random_state=seed).reset_index(drop=True)
test_df  = pd.concat([te_m, te_s]).sample(frac=1, random_state=seed).reset_index(drop=True)

# Safety Check: Ensure no subject leakage across splits
train_subs = set(train_df[subject_col])
val_subs = set(val_df[subject_col])
test_subs = set(test_df[subject_col])

assert len(train_subs.intersection(val_subs)) == 0, "Leakage Train-Val"
assert len(train_subs.intersection(test_subs)) == 0, "Leakage Train-Test"
assert len(val_subs.intersection(test_subs)) == 0,   "Leakage Val-Test"

# --- REPORT ---
total_rows = len(train_df) + len(val_df) + len(test_df)
print("\n=== Final Combined Split Report ===")
print(f"TRAIN: {len(train_df):,} rows")
print(f"VAL:   {len(val_df):,} rows")
print(f"TEST:  {len(test_df):,} rows")
print(f"Total: {total_rows:,} rows")

# --- save to history_aware_data/ ---
outdir = Path("dataset/history_aware_data")
outdir.mkdir(parents=True, exist_ok=True)

print(f"\nSaving datasets to {outdir.resolve()}...")
train_df.to_pickle(outdir / "train_df.pkl")
val_df.to_pickle(outdir / "val_df.pkl")
test_df.to_pickle(outdir / "test_df.pkl")

# --- verify no NaN Chief Complaint ---
print("\n=== Verification ===")
print(f"Train - Chief Complaint NaN count: {train_df['Chief Complaint'].isna().sum()}")
print(f"Val   - Chief Complaint NaN count: {val_df['Chief Complaint'].isna().sum()}")
print(f"Test  - Chief Complaint NaN count: {test_df['Chief Complaint'].isna().sum()}")
