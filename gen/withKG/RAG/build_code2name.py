#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_code2name.py
Create code→name map for:
- ATC:<code>        (from SAB == 'ATC')
- LNC:<code>        (LOINC; from SAB == 'LNC')
- PROC:<code>       (ICD-9-Proc; from SAB == 'ICD9CM' & looks like NN.NN)

Outputs:
  kg_recommender/code2name.json
  kg_recommender/code2name.pkl
"""

import os, json, pickle, argparse
import pandas as pd
from kg_utils import load_nodes, is_icd9_proc

def build_code2name(nodes_csv: str) -> dict:
    nodes = load_nodes(nodes_csv)
    code2name = {}

    # ATC
    atc = nodes[(nodes["sab"]=="ATC") & nodes["code"].ne("") & nodes["name"].ne("")]
    if not atc.empty:
        m = (atc.groupby("code")["name"]
                .apply(lambda s: " | ".join(sorted(set(map(str, s)))))
                .to_dict())
        for code, nm in m.items():
            code2name[f"ATC:{code}"] = nm

    # LOINC (LNC)
    lnc = nodes[(nodes["sab"]=="LNC") & nodes["code"].ne("") & nodes["name"].ne("")]
    if not lnc.empty:
        m = (lnc.groupby("code")["name"]
                .apply(lambda s: " | ".join(sorted(set(map(str, s)))))
                .to_dict())
        for code, nm in m.items():
            code2name[f"LNC:{code}"] = nm

    # ICD-9 Procedures (from ICD9CM, code looks like proc)
    icd = nodes[(nodes["sab"]=="ICD9CM") & nodes["code"].ne("") & nodes["name"].ne("")]
    if not icd.empty:
        icd_proc = icd[icd["code"].apply(is_icd9_proc)]
        if not icd_proc.empty:
            m = (icd_proc.groupby("code")["name"]
                        .apply(lambda s: " | ".join(sorted(set(map(str, s)))))
                        .to_dict())
            for code, nm in m.items():
                code2name[f"PROC:{code}"] = nm

    return code2name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg_nodes_csv", required=True)
    ap.add_argument("--out_json", default="kg_recommender/code2name.json")
    ap.add_argument("--out_pkl",  default="kg_recommender/code2name.pkl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    code2name = build_code2name(args.kg_nodes_csv)
    with open(args.out_json, "w") as f:
        json.dump(code2name, f, ensure_ascii=False, indent=2)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(code2name, f)
    print(f"Wrote {len(code2name):,} entries → {args.out_json} & {args.out_pkl}")

if __name__ == "__main__":
    main()
