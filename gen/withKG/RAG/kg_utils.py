#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kg_utils.py
Load UMLS-like KG from CSVs into a NetworkX DiGraph with useful attributes.
Also provides small helpers to detect ICD-9 procedures vs diagnoses.
"""

from __future__ import annotations
import pandas as pd
import networkx as nx
import re
from typing import Tuple, Dict, Set

PROC_RE = re.compile(r"^\d{2}\.\d{1,2}$")  # e.g., 54.91

def is_icd9_proc(code: str) -> bool:
    """Heuristic: ICD-9 procedures look like NN.NN (2 digits '.' 1-2 digits)."""
    if not code: return False
    return bool(PROC_RE.match(str(code).strip()))

def load_nodes(nodes_csv: str) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_csv)
    for col in ["cui", "name", "sab", "code", "semantic_type"]:
        if col not in nodes.columns:
            raise ValueError(f"Missing column '{col}' in {nodes_csv}")
    # normalize
    for c in ["cui", "name", "sab", "code"]:
        nodes[c] = nodes[c].astype(str)
    return nodes

def load_edges(edges_csv: str) -> pd.DataFrame:
    edges = pd.read_csv(edges_csv)
    for col in ["cui_start","cui_target","rel","rela"]:
        if col not in edges.columns:
            raise ValueError(f"Missing column '{col}' in {edges_csv}")
    for c in ["cui_start","cui_target","rel","rela"]:
        edges[c] = edges[c].astype(str)
    return edges

def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    # add nodes
    for _, r in nodes.iterrows():
        G.add_node(r["cui"], **{
            "name": r.get("name",""),
            "sab":  r.get("sab",""),
            "code": r.get("code",""),
            "semantic_type": r.get("semantic_type",""),
        })
    # add edges
    for _, r in edges.iterrows():
        G.add_edge(r["cui_start"], r["cui_target"], **{
            "rel":  r.get("rel",""),
            "rela": r.get("rela",""),
            "sab_relation": r.get("sab_relation",""),
        })
    return G

def icd9_dx_title_and_cuis(nodes: pd.DataFrame) -> Tuple[Dict[str,str], Dict[str,Set[str]]]:
    """
    Returns:
      code_to_title: ICD-9 **diagnosis** code -> one preferred title
      code_to_cuis:  ICD-9 **diagnosis** code -> set of CUIs backing it
    """
    df = nodes[(nodes["sab"]=="ICD9CM") & nodes["code"].ne("") & nodes["name"].ne("")].copy()
    df = df[~df["code"].apply(is_icd9_proc)]  # filter out procedures
    # title (pick one)
    code_to_title = (df.groupby("code")["name"]
                       .apply(lambda s: sorted(set(map(str, s)))[0])
                       .to_dict())
    # CUI set
    code_to_cuis = (df.groupby("code")["cui"]
                      .apply(lambda s: set(s))
                      .to_dict())
    return code_to_title, code_to_cuis
