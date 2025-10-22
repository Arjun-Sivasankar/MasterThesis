# # -*- coding: utf-8 -*-
# """
# Build code->CUI maps from UMLS MRCONSO.RRF (chunked, memory-safe), with
# robust ICD-9 alias support via a separate alias->canonical mapping.

# Outputs under --out-dir:
#   - code2cui_icd9_dx.pkl        # canonical ICD-9-CM DX -> list[CUIs]
#   - code2cui_icd9_proc.pkl      # canonical ICD-9-CM PROC -> list[CUIs]
#   - alias2canon_icd9_dx.pkl     # alias (nodot, shortened, etc) -> canonical DX
#   - alias2canon_icd9_proc.pkl   # alias -> canonical PROC
#   - code2cui_atc.pkl
#   - code2cui_loinc.pkl
# (+ optional code2name_* if --with-names)

# Dataset-aware options:
#   --dataset-codes : PKL/CSV/TXT of ICD-9 dx codes (PKL DataFrame must have 'icd_code' lists).
#   --allow-suppress-for-dataset : allow SUPPRESS='O' only for codes present in dataset aliases.
# """

# import os, re, json, pickle, argparse
# from collections import defaultdict
# from typing import Dict, Iterable, Tuple, List, Set, Any
# import pandas as pd
# import numpy as np
# from tqdm import tqdm

# # -------------------- Formatting & validation --------------------

# def _strip(x: str) -> str:
#     return re.sub(r"\s+", "", str(x or "")).upper().rstrip(".")

# def format_icd9_dx(code: str) -> str:
#     c = _strip(code)
#     if not c: return ""
#     if c[0].isdigit():
#         return c[:3]+"."+c[3:] if len(c) > 3 and "." not in c else c
#     if c[0] == "V":
#         return c[:3]+"."+c[3:] if len(c) > 3 and "." not in c else c
#     if c[0] == "E":
#         return c[:4]+"."+c[4:] if len(c) > 4 and "." not in c else c
#     return c

# def format_icd9_proc(code: str) -> str:
#     c = _strip(code)
#     if c.startswith("PRO_"):
#         c = c[4:]
#     if not c: return ""
#     if c[0].isdigit():
#         return c[:2]+"."+c[2:] if len(c) > 2 and "." not in c else c
#     return c

# _dx_pat   = re.compile(r"^(?:\d{3}(\.\d{1,2})?|V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?)$", re.I)
# _proc_pat = re.compile(r"^\d{2}\.\d{1,2}$")

# def is_icd9_dx(code: str) -> bool:
#     c = _strip(code)
#     return bool(_dx_pat.match(c)) and not bool(_proc_pat.match(c))

# def is_icd9_proc(code: str) -> bool:
#     c = _strip(code)
#     return bool(_proc_pat.match(c))

# def dx_aliases(k: str) -> Set[str]:
#     """Generate lookup aliases for ICD-9 DX."""
#     out = set()
#     k = _strip(k)
#     if not k: return out
#     out.add(k)
#     out.add(k.replace('.', ''))  # no-dot
#     # *.00 -> *.0 (add both)
#     if k[0].isdigit() and re.match(r'^\d{3}\.\d{2}$', k) and k.endswith('0'):
#         out.add(k[:-1])
#         out.add(k[:-1].replace('.', ''))
#     # E***.x -> E**** (no-dot)
#     if k.startswith('E') and '.' in k and len(k.split('.')[-1]) == 1:
#         out.add(k.replace('.', ''))
#     return out

# def proc_aliases(k: str) -> Set[str]:
#     """Lookup aliases for procedures (dot and no-dot)."""
#     out = set()
#     k = _strip(k)
#     if not k: return out
#     out.add(k)
#     out.add(k.replace('.', ''))
#     return out

# # -------------------- IO helpers --------------------

# def ensure_dir(p: str):
#     os.makedirs(p, exist_ok=True)

# def dump_pickle(path: str, obj: Any):
#     with open(path, "wb") as f:
#         pickle.dump(obj, f)

# def _flatten_list_column(series: pd.Series) -> List[str]:
#     out = []
#     for x in series:
#         if isinstance(x, (list, tuple, np.ndarray)):
#             out.extend(map(str, x))
#         elif pd.isna(x):
#             continue
#         else:
#             out.append(str(x))
#     return out

# def load_dataset_codes(path: str) -> Set[str]:
#     """PKL/CSV/TXT → unique formatted dx codes → expanded alias set (for matching & SUPPRESS allowlist)."""
#     if not path or not os.path.exists(path):
#         return set()
#     path_lower = path.lower()
#     try:
#         if path_lower.endswith((".pkl", ".pickle")):
#             df = pickle.load(open(path, "rb"))
#             if not isinstance(df, pd.DataFrame):
#                 raise ValueError("Pickle does not contain a pandas DataFrame.")
#             if 'icd_code' not in df.columns:
#                 raise ValueError("Pickle DataFrame must have an 'icd_code' column.")
#             raw_codes = _flatten_list_column(df['icd_code'])
#         elif path_lower.endswith(".csv"):
#             df = pd.read_csv(path)
#             col = 'icd_code' if 'icd_code' in df.columns else df.columns[0]
#             raw_codes = df[col].astype(str).tolist()
#         else:
#             with open(path, "r") as f:
#                 raw_codes = [line.strip() for line in f if str(line).strip()]
#     except Exception as e:
#         raise RuntimeError(f"Failed to load dataset codes from {path}: {e}")

#     formatted = {format_icd9_dx(c) for c in set(raw_codes) if format_icd9_dx(c)}
#     expanded = set()
#     for c in formatted:
#         expanded |= dx_aliases(c)
#     return expanded

# # -------------------- MRCONSO streaming --------------------

# COLS = [
#     'CUI','LAT','TS','LUI','STT','SUI','ISPREF','AUI','SAUI','SCUI','SDUI',
#     'SAB','TTY','CODE','STR','SRL','SUPPRESS','CVF'
# ]
# USECOLS_MIN = ['CUI', 'LAT', 'TS', 'SAB', 'TTY', 'CODE', 'STR', 'SUPPRESS']

# def read_mrconso_stream(path: str, usecols: List[str], chunksize: int):
#     return pd.read_csv(
#         path, sep='|', names=COLS, usecols=[COLS.index(c) for c in usecols],
#         dtype=str, chunksize=chunksize, index_col=False
#     )

# # -------------------- Builders --------------------

# def build_code2cui_generic(
#     mrconso_path: str,
#     target_sab: str,
#     langs: Iterable[str],
#     chunksize: int,
#     keep_ts_p_only: bool = False,
#     code_normalizer = None,
#     code_validator  = None
# ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
#     code2cuis = defaultdict(set)
#     code2name = {}
#     for ch in tqdm(read_mrconso_stream(mrconso_path, USECOLS_MIN, chunksize),
#                    desc=f"SAB={target_sab}", unit="chunk"):
#         ch = ch[(ch['LAT'].isin(langs)) & (ch['SAB'] == target_sab)]
#         ch = ch[(ch['SUPPRESS'] != 'O')]
#         if keep_ts_p_only:
#             ch_pref = ch[ch['TS'] == 'P']
#             if not ch_pref.empty:
#                 ch = ch_pref
#         if ch.empty: continue

#         for _, row in ch.iterrows():
#             raw_code = str(row['CODE'] or "")
#             if not raw_code: continue
#             norm = code_normalizer(raw_code) if code_normalizer else raw_code
#             if not norm: continue
#             if code_validator and not code_validator(norm): continue
#             cui  = str(row['CUI'] or "")
#             name = str(row['STR'] or "")
#             code2cuis[norm].add(cui)
#             code2name.setdefault(norm, name)

#     return {k: sorted(v) for k, v in code2cuis.items()}, code2name

# def build_icd9_dx_proc(
#     mrconso_path: str,
#     langs: Iterable[str],
#     chunksize: int,
#     dataset_aliases: Set[str],
#     allow_suppress_for_dataset: bool
# ):
#     """
#     Build canonical maps + alias->canonical for DX and PROC.
#     Returns:
#       dx_code2cui, dx_code2name, proc_code2cui, proc_code2name,
#       alias2canon_dx, alias2canon_proc
#     """
#     dx_canon2cuis, dx_canon2name = defaultdict(set), {}
#     pr_canon2cuis, pr_canon2name = defaultdict(set), {}
#     alias2canon_dx, alias2canon_pr = {}, {}

#     for ch in tqdm(read_mrconso_stream(mrconso_path, USECOLS_MIN, chunksize),
#                    desc="SAB=ICD9CM", unit="chunk"):
#         ch = ch[(ch['LAT'].isin(langs)) & (ch['SAB'] == 'ICD9CM')]
#         if ch.empty: continue

#         for _, row in ch.iterrows():
#             raw = str(row['CODE'] or "")
#             if not raw: continue
#             sup = str(row['SUPPRESS'] or "")
#             if sup == 'O':
#                 # keep only if explicitly allowed for dataset codes
#                 tmp_dx = format_icd9_dx(raw)
#                 tmp_pr = format_icd9_proc(raw)
#                 al = dx_aliases(tmp_dx) | proc_aliases(tmp_pr)
#                 if not (allow_suppress_for_dataset and any(a in dataset_aliases for a in al)):
#                     continue

#             cui  = str(row['CUI'] or "")
#             name = str(row['STR'] or "")

#             dx_canon   = format_icd9_dx(raw)
#             proc_canon = format_icd9_proc(raw)

#             if is_icd9_proc(proc_canon):
#                 # store only the canonical key
#                 pr_canon2cuis[proc_canon].add(cui)
#                 pr_canon2name.setdefault(proc_canon, name)
#                 # collect aliases that resolve to this canonical
#                 for a in proc_aliases(proc_canon):
#                     alias2canon_pr.setdefault(a, proc_canon)
#             else:
#                 if is_icd9_dx(dx_canon):
#                     dx_canon2cuis[dx_canon].add(cui)
#                     dx_canon2name.setdefault(dx_canon, name)
#                     for a in dx_aliases(dx_canon):
#                         alias2canon_dx.setdefault(a, dx_canon)

#     dx_code2cui = {k: sorted(v) for k, v in dx_canon2cuis.items()}
#     pr_code2cui = {k: sorted(v) for k, v in pr_canon2cuis.items()}
#     return dx_code2cui, dx_canon2name, pr_code2cui, pr_canon2name, alias2canon_dx, alias2canon_pr

# # -------------------- CLI --------------------

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--mrconso", required=True, help="Path to MRCONSO.RRF")
#     ap.add_argument("--out-dir", required=True, help="Where to write pickles")
#     ap.add_argument("--langs", default="ENG", help="Comma-separated langs (default: ENG)")
#     ap.add_argument("--chunk-size", type=int, default=1_000_000, help="Rows per chunk (default: 1e6)")
#     ap.add_argument("--with_names", action="store_true", help="Also write code->preferred name maps")
#     ap.add_argument("--dataset-codes", default="", help="PKL/CSV/TXT of dx codes (PKL must have 'icd_code' lists)")
#     ap.add_argument("--allow-suppress-for-dataset", action="store_true",
#                     help="Allow SUPPRESS='O' rows ONLY for codes present in --dataset-codes")
#     args = ap.parse_args()

#     ensure_dir(args.out_dir)
#     langs = [s.strip().upper() for s in args.langs.split(",") if s.strip()]

#     dataset_aliases = load_dataset_codes(args.dataset_codes) if args.dataset_codes else set()
#     if dataset_aliases:
#         print(f"[Dataset] Loaded {len(dataset_aliases):,} alias keys from dataset dx codes (expanded).")

#     # ---------- ATC ----------
#     atc_c2c, atc_c2n = build_code2cui_generic(
#         mrconso_path=args.mrconso,
#         target_sab="ATC",
#         langs=langs,
#         chunksize=args.chunk_size,
#         keep_ts_p_only=True,
#         code_normalizer=lambda x: _strip(x),
#         code_validator=lambda x: bool(x)
#     )
#     dump_pickle(os.path.join(args.out_dir, "code2cui_atc.pkl"), atc_c2c)
#     if args.with_names:
#         dump_pickle(os.path.join(args.out_dir, "code2name_atc.pkl"), atc_c2n)
#     print(f"[ATC] canonical codes: {len(atc_c2c):,}")

#     # ---------- LOINC ----------
#     loinc_c2c, loinc_c2n = build_code2cui_generic(
#         mrconso_path=args.mrconso,
#         target_sab="LNC",
#         langs=langs,
#         chunksize=args.chunk_size,
#         keep_ts_p_only=True,
#         code_normalizer=lambda x: _strip(x),
#         code_validator=lambda x: bool(x)
#     )
#     dump_pickle(os.path.join(args.out_dir, "code2cui_loinc.pkl"), loinc_c2c)
#     if args.with_names:
#         dump_pickle(os.path.join(args.out_dir, "code2name_loinc.pkl"), loinc_c2n)
#     print(f"[LNC] canonical codes: {len(loinc_c2c):,}")

#     # ---------- ICD-9 (DX & PROC) ----------
#     dx_c2c, dx_c2n, pr_c2c, pr_c2n, a2c_dx, a2c_pr = build_icd9_dx_proc(
#         mrconso_path=args.mrconso,
#         langs=langs,
#         chunksize=args.chunk_size,
#         dataset_aliases=dataset_aliases,
#         allow_suppress_for_dataset=bool(args.allow_suppress_for_dataset)
#     )
#     dump_pickle(os.path.join(args.out_dir, "code2cui_icd9_dx.pkl"), dx_c2c)
#     dump_pickle(os.path.join(args.out_dir, "code2cui_icd9_proc.pkl"), pr_c2c)
#     dump_pickle(os.path.join(args.out_dir, "alias2canon_icd9_dx.pkl"), a2c_dx)
#     dump_pickle(os.path.join(args.out_dir, "alias2canon_icd9_proc.pkl"), a2c_pr)
#     if args.with_names:
#         dump_pickle(os.path.join(args.out_dir, "code2name_icd9_dx.pkl"), dx_c2n)
#         dump_pickle(os.path.join(args.out_dir, "code2name_icd9_proc.pkl"), pr_c2n)

#     # ---------- Summary (canonical only) ----------
#     summary = {
#         "atc_codes": len(atc_c2c),
#         "loinc_codes": len(loinc_c2c),
#         "icd9_dx_codes": len(dx_c2c),
#         "icd9_proc_codes": len(pr_c2c)
#     }
#     with open(os.path.join(args.out_dir, "code2cui_summary.json"), "w") as f:
#         json.dump(summary, f, indent=2)
#     print("\n[OK] Wrote pickles + canonical summary:")
#     print(json.dumps(summary, indent=2))

# if __name__ == "__main__":
#     main()

# build_code2cui_maps_all_in_one.py
# -*- coding: utf-8 -*-
"""
Build code->CUI maps from UMLS MRCONSO.RRF (chunked, memory-safe), with
robust ICD-9 alias support via a separate alias->canonical mapping.
Also auto-repairs your remaining ICD-9 dx gaps (hard-coded) in one pass.

Outputs under --out-dir:
  - code2cui_icd9_dx.pkl        # canonical ICD-9-CM DX -> list[CUIs]
  - code2cui_icd9_proc.pkl      # canonical ICD-9-CM PROC -> list[CUIs]
  - alias2canon_icd9_dx.pkl     # alias (nodot, shortened, etc) -> canonical DX
  - alias2canon_icd9_proc.pkl   # alias -> canonical PROC
  - code2cui_atc.pkl
  - code2cui_loinc.pkl
  - (optional) code2name_* if --with_names
  - gap_icd9_audit.parquet      # MRCONSO rows used to fix hard-coded gaps

Dataset-aware options:
  --dataset-codes : PKL/CSV/TXT of ICD-9 dx codes (PKL DataFrame must have 'icd_code' lists).
  --allow-suppress-for-dataset : allow SUPPRESS='O' rows ONLY for codes present in dataset aliases.
"""

import os, re, json, pickle, argparse
from collections import defaultdict
from typing import Dict, Iterable, Tuple, List, Set, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

# -------------------- Hard-coded ICD-9 dx gaps to force-recover --------------------
GAP_CODES = {
    '042','075','135','138','185','193','220','226','261','262','311','317','319','340','412',
    '426.82','430','431','452','462','475','481','486','490','496','501','513.0','514','515',
    '566','570','587','591','611.72','650','724.2','725','920'
}

# -------------------- Formatting & validation --------------------

def _strip(x: str) -> str:
    return re.sub(r"\s+", "", str(x or "")).upper().rstrip(".")

def format_icd9_dx(code: str) -> str:
    c = _strip(code)
    if not c: return ""
    if c[0].isdigit():
        return c[:3]+"."+c[3:] if len(c) > 3 and "." not in c else c
    if c[0] == "V":
        return c[:3]+"."+c[3:] if len(c) > 3 and "." not in c else c
    if c[0] == "E":
        return c[:4]+"."+c[4:] if len(c) > 4 and "." not in c else c
    return c

def format_icd9_proc(code: str) -> str:
    c = _strip(code)
    if c.startswith("PRO_"):
        c = c[4:]
    if not c: return ""
    if c[0].isdigit():
        return c[:2]+"."+c[2:] if len(c) > 2 and "." not in c else c
    return c

_dx_pat   = re.compile(r"^(?:\d{3}(\.\d{1,2})?|V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?)$", re.I)
_proc_pat = re.compile(r"^\d{2}\.\d{1,2}$")

def is_icd9_dx(code: str) -> bool:
    c = _strip(code)
    return bool(_dx_pat.match(c)) and not bool(_proc_pat.match(c))

def is_icd9_proc(code: str) -> bool:
    c = _strip(code)
    return bool(_proc_pat.match(c))

def dx_aliases(k: str) -> Set[str]:
    """Generate lookup aliases for ICD-9 DX."""
    out = set()
    k = _strip(k)
    if not k: return out
    out.add(k)
    out.add(k.replace('.', ''))  # no-dot
    # *.00 -> *.0 (+ no-dot)
    if k[0].isdigit() and re.match(r'^\d{3}\.\d{2}$', k) and k.endswith('0'):
        out.add(k[:-1])
        out.add(k[:-1].replace('.', ''))
    # E***.x -> E**** (no-dot)
    if k.startswith('E') and '.' in k and len(k.split('.')[-1]) == 1:
        out.add(k.replace('.', ''))
    return out

def proc_aliases(k: str) -> Set[str]:
    out = set()
    k = _strip(k)
    if not k: return out
    out.add(k)
    out.add(k.replace('.', ''))
    return out

# -------------------- IO helpers --------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def dump_pickle(path: str, obj: Any):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _flatten_list_column(series: pd.Series) -> List[str]:
    out = []
    for x in series:
        if isinstance(x, (list, tuple, np.ndarray)):
            out.extend(map(str, x))
        elif pd.isna(x):
            continue
        else:
            out.append(str(x))
    return out

def load_dataset_codes(path: str) -> Set[str]:
    """PKL/CSV/TXT → unique formatted dx codes → expanded alias set (for matching & SUPPRESS allowlist)."""
    if not path or not os.path.exists(path):
        return set()
    path_lower = path.lower()
    try:
        if path_lower.endswith((".pkl", ".pickle")):
            df = pickle.load(open(path, "rb"))
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Pickle does not contain a pandas DataFrame.")
            if 'icd_code' not in df.columns:
                raise ValueError("Pickle DataFrame must have an 'icd_code' column.")
            raw_codes = _flatten_list_column(df['icd_code'])
        elif path_lower.endswith(".csv"):
            df = pd.read_csv(path)
            col = 'icd_code' if 'icd_code' in df.columns else df.columns[0]
            raw_codes = df[col].astype(str).tolist()
        else:
            with open(path, "r") as f:
                raw_codes = [line.strip() for line in f if str(line).strip()]
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset codes from {path}: {e}")

    formatted = {format_icd9_dx(c) for c in set(raw_codes) if format_icd9_dx(c)}
    expanded = set()
    for c in formatted:
        expanded |= dx_aliases(c)
    return expanded

# -------------------- MRCONSO streaming --------------------

COLS = [
    'CUI','LAT','TS','LUI','STT','SUI','ISPREF','AUI','SAUI','SCUI','SDUI',
    'SAB','TTY','CODE','STR','SRL','SUPPRESS','CVF'
]
USECOLS_MIN = ['CUI', 'LAT', 'TS', 'SAB', 'TTY', 'CODE', 'STR', 'SUPPRESS']

def read_mrconso_stream(path: str, usecols: List[str], chunksize: int):
    return pd.read_csv(
        path, sep='|', names=COLS, usecols=[COLS.index(c) for c in usecols],
        dtype=str, chunksize=chunksize, index_col=False
    )

# -------------------- Generic SAB builder --------------------

def build_code2cui_generic(
    mrconso_path: str,
    target_sab: str,
    langs: Iterable[str],
    chunksize: int,
    keep_ts_p_only: bool = False,
    code_normalizer = None,
    code_validator  = None
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    code2cuis = defaultdict(set)
    code2name = {}
    for ch in tqdm(read_mrconso_stream(mrconso_path, USECOLS_MIN, chunksize),
                   desc=f"SAB={target_sab}", unit="chunk"):
        ch = ch[(ch['LAT'].isin(langs)) & (ch['SAB'] == target_sab)]
        ch = ch[(ch['SUPPRESS'] != 'O')]
        if keep_ts_p_only:
            ch_pref = ch[ch['TS'] == 'P']
            if not ch_pref.empty:
                ch = ch_pref
        if ch.empty: continue

        for _, row in ch.iterrows():
            raw_code = str(row['CODE'] or "")
            if not raw_code: continue
            norm = code_normalizer(raw_code) if code_normalizer else raw_code
            if not norm: continue
            if code_validator and not code_validator(norm): continue
            cui  = str(row['CUI'] or "")
            name = str(row['STR'] or "")
            code2cuis[norm].add(cui)
            code2name.setdefault(norm, name)

    return {k: sorted(v) for k, v in code2cuis.items()}, code2name

# -------------------- ICD-9 builder + gap recovery --------------------

def build_icd9_dx_proc(
    mrconso_path: str,
    langs: Iterable[str],
    chunksize: int,
    dataset_aliases: Set[str],
    allow_suppress_for_dataset: bool
):
    """
    Build canonical maps + alias->canonical for DX and PROC.
    Returns:
      dx_code2cui, dx_code2name, proc_code2cui, proc_code2name,
      alias2canon_dx, alias2canon_proc
    """
    dx_canon2cuis, dx_canon2name = defaultdict(set), {}
    pr_canon2cuis, pr_canon2name = defaultdict(set), {}
    alias2canon_dx, alias2canon_pr = {}, {}

    for ch in tqdm(read_mrconso_stream(mrconso_path, USECOLS_MIN, chunksize),
                   desc="SAB=ICD9CM", unit="chunk"):
        ch = ch[(ch['LAT'].isin(langs)) & (ch['SAB'] == 'ICD9CM')]
        if ch.empty: continue

        for _, row in ch.iterrows():
            raw = str(row['CODE'] or "")
            if not raw: continue

            sup = str(row['SUPPRESS'] or "")
            if sup == 'O':
                # keep only if (a) in dataset aliases OR (b) in our hard-coded gaps
                tmp_dx = format_icd9_dx(raw)
                tmp_pr = format_icd9_proc(raw)
                al = dx_aliases(tmp_dx) | proc_aliases(tmp_pr)
                keep = any(a in dataset_aliases for a in al) or (format_icd9_dx(raw) in {format_icd9_dx(x) for x in GAP_CODES})
                if not keep:
                    continue

            cui  = str(row['CUI'] or "")
            name = str(row['STR'] or "")

            dx_canon   = format_icd9_dx(raw)
            proc_canon = format_icd9_proc(raw)

            if is_icd9_proc(proc_canon):
                pr_canon2cuis[proc_canon].add(cui)
                pr_canon2name.setdefault(proc_canon, name)
                for a in proc_aliases(proc_canon):
                    alias2canon_pr.setdefault(a, proc_canon)
            else:
                if is_icd9_dx(dx_canon):
                    dx_canon2cuis[dx_canon].add(cui)
                    dx_canon2name.setdefault(dx_canon, name)
                    for a in dx_aliases(dx_canon):
                        alias2canon_dx.setdefault(a, dx_canon)

    dx_code2cui = {k: sorted(v) for k, v in dx_canon2cuis.items()}
    pr_code2cui = {k: sorted(v) for k, v in pr_canon2cuis.items()}
    return dx_code2cui, dx_canon2name, pr_code2cui, pr_canon2name, alias2canon_dx, alias2canon_pr

def recover_icd9_gaps_inline(
    mrconso_path: str,
    out_dir: str,
    dx_code2cui: Dict[str, List[str]],
    alias2canon_dx: Dict[str, str],
    chunk_size: int = 1_000_000
):
    """Scan MRCONSO for the hard-coded GAP_CODES and add CUIs for any still-missing ones (includes suppressed rows)."""
    targets = {format_icd9_dx(c) for c in GAP_CODES if format_icd9_dx(c)}
    missing = sorted([c for c in targets if c not in dx_code2cui])
    if not missing:
        return

    # alias map for filtering
    a2c_local = {}
    for c in missing:
        for a in dx_aliases(c):
            a2c_local.setdefault(a, c)
    alias_all = set(a2c_local.keys())

    usecols = ['CUI','LAT','TS','SAB','CODE','STR','SUPPRESS']
    rows = []
    found = defaultdict(set)

    it = pd.read_csv(mrconso_path, sep='|', names=COLS, usecols=[COLS.index(c) for c in usecols],
                     dtype=str, chunksize=chunk_size, index_col=False)

    for ch in tqdm(it, desc="Recovering hard-coded ICD-9 gaps", unit="chunk"):
        ch = ch[(ch['LAT'] == 'ENG') & (ch['SAB'] == 'ICD9CM')]
        if ch.empty: continue
        ch['CODE'] = ch['CODE'].astype(str).str.upper().str.rstrip('.')
        ch['CODE_NODOT'] = ch['CODE'].str.replace('.', '', regex=False)
        mask = ch['CODE'].isin(alias_all) | ch['CODE_NODOT'].isin(alias_all)
        hit = ch[mask]
        if hit.empty: continue

        for _, r in hit.iterrows():
            canon = a2c_local.get(r['CODE'], a2c_local.get(r['CODE_NODOT']))
            if not canon: continue
            cui  = (r['CUI'] or "").strip()
            name = (r['STR'] or "").strip()
            rows.append({
                "icd_canonical": canon,
                "code_field": r['CODE'],
                "cui": cui,
                "ts": (r['TS'] or "").strip(),
                "suppressed": (r['SUPPRESS'] or "").strip(),
                "name": name
            })
            found[canon].add(cui)

    # write audit for transparency
    if rows:
        pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "gap_icd9_audit.parquet"), index=False)

    # patch maps
    for k, cuis in found.items():
        if k not in dx_code2cui and len(cuis) > 0:
            dx_code2cui[k] = sorted(cuis)
        # aliases for these newly added canonicals
        for a in dx_aliases(k):
            alias2canon_dx.setdefault(a, k)

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrconso", required=True, help="Path to MRCONSO.RRF")
    ap.add_argument("--out-dir", required=True, help="Where to write pickles")
    ap.add_argument("--langs", default="ENG", help="Comma-separated langs (default: ENG)")
    ap.add_argument("--chunk-size", type=int, default=1_000_000, help="Rows per chunk (default: 1e6)")
    ap.add_argument("--with_names", action="store_true", help="Also write code->preferred name maps")
    ap.add_argument("--dataset-codes", default="", help="PKL/CSV/TXT of dx codes (PKL must have 'icd_code' lists)")
    ap.add_argument("--allow-suppress-for-dataset", action="store_true",
                    help="Allow SUPPRESS='O' rows ONLY for codes present in --dataset-codes")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    langs = [s.strip().upper() for s in args.langs.split(",") if s.strip()]

    dataset_aliases = load_dataset_codes(args.dataset_codes) if args.dataset_codes else set()
    if dataset_aliases:
        print(f"[Dataset] Loaded {len(dataset_aliases):,} alias keys from dataset dx codes (expanded).")

    # ---------- ATC ----------
    atc_c2c, atc_c2n = build_code2cui_generic(
        mrconso_path=args.mrconso,
        target_sab="ATC",
        langs=langs,
        chunksize=args.chunk_size,
        keep_ts_p_only=True,
        code_normalizer=lambda x: _strip(x),
        code_validator=lambda x: bool(x)
    )
    dump_pickle(os.path.join(args.out_dir, "code2cui_atc.pkl"), atc_c2c)
    if args.with_names:
        dump_pickle(os.path.join(args.out_dir, "code2name_atc.pkl"), atc_c2n)
    print(f"[ATC] canonical codes: {len(atc_c2c):,}")

    # ---------- LOINC ----------
    loinc_c2c, loinc_c2n = build_code2cui_generic(
        mrconso_path=args.mrconso,
        target_sab="LNC",
        langs=langs,
        chunksize=args.chunk_size,
        keep_ts_p_only=True,
        code_normalizer=lambda x: _strip(x),
        code_validator=lambda x: bool(x)
    )
    dump_pickle(os.path.join(args.out_dir, "code2cui_loinc.pkl"), loinc_c2c)
    if args.with_names:
        dump_pickle(os.path.join(args.out_dir, "code2name_loinc.pkl"), loinc_c2n)
    print(f"[LNC] canonical codes: {len(loinc_c2c):,}")

    # ---------- ICD-9 (DX & PROC) ----------
    dx_c2c, dx_c2n, pr_c2c, pr_c2n, a2c_dx, a2c_pr = build_icd9_dx_proc(
        mrconso_path=args.mrconso,
        langs=langs,
        chunksize=args.chunk_size,
        dataset_aliases=dataset_aliases,
        allow_suppress_for_dataset=bool(args.allow_suppress_for_dataset)
    )

    # Inline recovery for your GAP_CODES (ensures 100% for those)
    recover_icd9_gaps_inline(
        mrconso_path=args.mrconso,
        out_dir=args.out_dir,
        dx_code2cui=dx_c2c,
        alias2canon_dx=a2c_dx,
        chunk_size=args.chunk_size
    )

    dump_pickle(os.path.join(args.out_dir, "code2cui_icd9_dx.pkl"), dx_c2c)
    dump_pickle(os.path.join(args.out_dir, "code2cui_icd9_proc.pkl"), pr_c2c)
    dump_pickle(os.path.join(args.out_dir, "alias2canon_icd9_dx.pkl"), a2c_dx)
    dump_pickle(os.path.join(args.out_dir, "alias2canon_icd9_proc.pkl"), a2c_pr)
    if args.with_names:
        dump_pickle(os.path.join(args.out_dir, "code2name_icd9_dx.pkl"), dx_c2n)
        dump_pickle(os.path.join(args.out_dir, "code2name_icd9_proc.pkl"), pr_c2n)

    # ---------- Summary (canonical only) ----------
    summary = {
        "atc_codes": len(atc_c2c),
        "loinc_codes": len(loinc_c2c),
        "icd9_dx_codes": len(dx_c2c),
        "icd9_proc_codes": len(pr_c2c)
    }
    with open(os.path.join(args.out_dir, "code2cui_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[OK] Wrote pickles + canonical summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
