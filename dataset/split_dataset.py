import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- config ---
mimic_lab   = "dataset/mimic_with_lab_mappings.pkl"
subject_col = "subject_id_x"   
test_size   = 0.10
val_size    = 0.10
seed        = 42

# --- load ---
df = pd.read_pickle(mimic_lab)

def subject_splits(df: pd.DataFrame, subject_col: str,
                   test_size=0.10, val_size=0.10, seed=42):
    if subject_col not in df.columns:
        raise KeyError(f"'{subject_col}' not found in DataFrame columns.")
    subs = df[subject_col].dropna().unique()
    if len(subs) < 3:
        raise ValueError(f"Need at least 3 unique subjects; found {len(subs)}.")

    train_subs, test_subs = train_test_split(subs, test_size=test_size, random_state=seed)
    train_subs, val_subs  = train_test_split(train_subs, test_size=val_size/(1-test_size), random_state=seed)

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

# --- split ---
train_df, val_df, test_df = subject_splits(
    df, subject_col=subject_col, test_size=test_size, val_size=val_size, seed=seed
)

# --- save to final_data/ ---
outdir = Path("dataset/final_data")
outdir.mkdir(parents=True, exist_ok=True)

train_df.to_pickle(outdir / "train_df.pkl")
val_df.to_pickle(outdir / "val_df.pkl")
test_df.to_pickle(outdir / "test_df.pkl")

print(f"Saved to: {outdir.resolve()}")


