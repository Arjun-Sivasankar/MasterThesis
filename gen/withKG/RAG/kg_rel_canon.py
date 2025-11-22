#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Canonicalize UMLS 'rela' relations into compact buckets + scores (no argparse).

Edit the variables below and run:
  python kg_rel_canon.py
"""

import json, math
from collections import Counter
import pandas as pd

# ========= CONFIG (edit me) =========
IN_EDGES_CSV  = "KG/kg_output4/kg_edges.csv"
OUT_EDGES_CSV = "KG/kg_output4/kg_edges_canon.csv"
DUMP_STATS    = "KG/kg_output4/rela_stats.json"   # set "" to skip stats file
MIN_SCORE     = 0.2      # keep rows with rela_score >= MIN_SCORE (0 keeps all)
PREFER_RELA   = True    # if True, prefer 'rela' over 'rel' when both present
# ====================================

# ---- Canonical buckets ----
REL_CANON_EXACT = {
    # Ontology / hierarchy
    "isa": "isa",
    "inverse_isa": "isa",
    "was_a": "isa",
    "inverse_was_a": "isa",
    "parent_group_of": "isa",
    "has_parent_group": "isa",
    "class_of": "isa",
    "has_class": "isa",

    # Anatomy, morphology, pathology
    "finding_site_of": "finding_site",
    "has_finding_site": "finding_site",
    "associated_morphology_of": "morphology",
    "has_associated_morphology": "morphology",
    "pathological_process_of": "pathology",
    "has_pathological_process": "pathology",

    # Causality / etiology
    "causative_agent_of": "etiology",
    "has_causative_agent": "etiology",
    "due_to": "etiology",
    "associated_etiologic_finding_of": "etiology",
    "has_associated_etiologic_finding": "etiology",

    # Procedure / site / method / device
    "procedure_site_of": "proc_site",
    "direct_procedure_site_of": "proc_site",
    "indirect_procedure_site_of": "proc_site",
    "has_procedure_site": "proc_site",
    "has_direct_procedure_site": "proc_site",
    "has_indirect_procedure_site": "proc_site",

    "method_of": "proc_method",
    "has_method": "proc_method",
    "finding_method_of": "proc_method",
    "has_finding_method": "proc_method",
    "measurement_method_of": "proc_method",
    "has_measurement_method": "proc_method",
    "technique_of": "proc_method",
    "has_technique": "proc_method",

    "instrumentation_of": "proc_device",
    "procedure_device_of": "proc_device",
    "has_procedure_device": "proc_device",
    "uses_device": "proc_device",
    "device_used_by": "proc_device",
    "access_instrument_of": "proc_device",
    "has_access_instrument": "proc_device",
    "direct_device_of": "proc_device",
    "has_direct_device": "proc_device",

    # Labs / measurements bridge
    "measures": "measurement",
    "measured_by": "measurement",
    "has_measured_component": "measurement",
    "measured_component_of": "measurement",
    "interpretation_of": "measurement",
    "is_interpreted_by": "measurement",
    "has_interpretation": "measurement",
    "scale_of": "measurement",
    "has_scale": "measurement",
    "property_of": "measurement",
    "has_property": "measurement",
    "analyzes": "measurement",
    "analyzed_by": "measurement",

    # Temporal
    "occurs_before": "temporal",
    "occurs_after": "temporal",
    "temporally_follows": "temporal",
    "temporally_followed_by": "temporal",
    "temporally_related_to": "temporal",

    # Mapping / equivalence
    "same_as": "equivalent",
    "possibly_equivalent_to": "equivalent",
    "partially_equivalent_to": "equivalent",
    "mapped_to": "equivalent",
    "mapped_from": "equivalent",
    "common_name_of": "equivalent",
    "has_common_name": "equivalent",
    "alternative_of": "equivalent",
    "has_alternative": "equivalent",
    "replaces": "equivalent",
    "replaced_by": "equivalent",
    "possibly_replaces": "equivalent",
    "possibly_replaced_by": "equivalent",

    # Generic association
    "associated_with": "assoc",
    "characterized_by": "assoc",
    "characterizes": "assoc",

    # Location / inherent
    "inherent_location_of": "location",
    "has_inherent_location": "location",
    "location_of": "location",

    # Intent/priority/severity/course
    "intent_of": "intent",
    "has_intent": "intent",
    "priority_of": "priority",
    "has_priority": "priority",
    "severity_of": "severity",
    "has_severity": "severity",
    "clinical_course_of": "course",
    "has_clinical_course": "course",
    "course_of": "course",

    # Metadata-ish
    "has_loinc_number": "meta",
    "loinc_number_of": "meta",
    "mth_has_expanded_form": "meta",
    "mth_expanded_form_of": "meta",
    "mth_xml_form_of": "meta",
    "mth_plain_text_form_of": "meta",
    "mth_has_plain_text_form": "meta",
}

# Substring fallbacks for robustness
REL_CANON_SUBSTR = [
    (("etiologic", "causative_agent", "due_to"), "etiology"),
    (("finding_site",), "finding_site"),
    (("morphology",), "morphology"),
    (("patholog",), "pathology"),
    (("procedure_site", "direct_procedure_site", "indirect_procedure_site"), "procedure_site"),
    (("method", "technique", "measurement_method", "finding_method"), "procedure_method"),
    (("device", "instrument"), "procedure_device"),
    (("measure", "measured_component", "interpret", "analyz", "scale", "property"), "measurement"),
    (("occurs_before", "occurs_after", "temporally"), "temporal"),
    (("same_as", "equivalent", "mapped", "common_name", "alternative", "replace"), "equivalent"),
    (("inherent_location", "location_of"), "location"),
    (("intent",), "intent"),
    (("priority",), "priority"),
    (("severity",), "severity"),
    (("course", "clinical_course"), "clinical_course"),
    (("associated_with", "characterized_by", "characterizes"), "assoc"),
    (("loinc", "mth_", "xml_form", "plain_text_form"), "meta"),
    (("isa", "was_a", "parent_group", "has_parent_group", "class_of", "has_class"), "isa"),
]

REL_SCORE = {
    "etiology":      3.0,
    "finding_site":  2.8,
    "morphology":    2.6,
    "pathology":     2.6,
    "proc_site":     2.0,
    "proc_method":   1.6,
    "proc_device":   1.4,
    "measurement":   1.2,
    "isa":           1.0,
    "location":      1.0,
    "equivalent":    0.8,
    "temporal":      0.6,
    "intent":        0.4,
    "priority":      0.3,
    "severity":      0.8,
    "course":        0.5,
    "assoc":         0.3,
    "meta":          0.2,
    "other":         0.2,
}

# ---------- helpers ----------
def norm_any(x) -> str:
    """Robust normalizer: None/NaN → '', lowercased, stripped."""
    if x is None:
        return ""
    try:
        import math
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()

def to_canon_key(raw: str) -> str:
    r = norm_any(raw)
    if not r:
        return "other"
    if r in REL_CANON_EXACT:
        return REL_CANON_EXACT[r]
    for subs, bucket in REL_CANON_SUBSTR:
        if any(s in r for s in subs):
            return bucket
    return "other"

def to_score(bucket: str) -> float:
    return float(REL_SCORE.get(bucket, 0.2))

# ---------- main work ----------
def run():
    print('[INFO] Loading KG edges dataset')
    df = pd.read_csv(IN_EDGES_CSV, dtype={"rel": "object", "rela": "object"})

    # Choose relation per row with fallback precedence
    def pick_relation(row) -> str:
        a = norm_any(row.get("rela", ""))
        b = norm_any(row.get("rel", ""))
        if PREFER_RELA:
            return a if a else b
        else:
            return b if b else a

    df["rela_raw"]   = df.apply(pick_relation, axis=1)
    df["rela_final"] = df["rela_raw"]  # already normalized
    df["rela_canon"] = df["rela_final"].map(to_canon_key)
    df["rela_score"] = df["rela_canon"].map(to_score)

    print("[INFO] Canonicalization complete....doing stats!")

    # Stats (before filter)
    total = len(df)
    num_empty = int((df["rela_final"] == "").sum())
    unmapped = int((df["rela_canon"] == "other").sum())
    freq_before = Counter(df["rela_canon"].tolist())

    # Optional filter by score
    if MIN_SCORE > 0:
        df = df[df["rela_score"] >= MIN_SCORE].reset_index(drop=True)

    print(f"[INFO] Saving canon edges.....")

    df.to_csv(OUT_EDGES_CSV, index=False)
    kept = len(df)

    # Dump stats
    if DUMP_STATS:
        out = {
            "input_rows": total,
            "kept_rows": kept,
            "empty_relation_rows": num_empty,
            "unmapped_other_rows_before_filter": unmapped,
            "bucket_scores": REL_SCORE,
            "bucket_freq_before_filter": dict(freq_before),
            "bucket_freq_after_filter": Counter(df["rela_canon"].tolist()),
            "config": {
                "IN_EDGES_CSV": IN_EDGES_CSV,
                "OUT_EDGES_CSV": OUT_EDGES_CSV,
                "MIN_SCORE": MIN_SCORE,
                "PREFER_RELA": PREFER_RELA,
            }
        }
        with open(DUMP_STATS, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] Stats written to {DUMP_STATS}")

    print(f"[OK] Canonicalized {total:,} edges → kept {kept:,} (min_score={MIN_SCORE})")
    print(f"[OK] Written: {OUT_EDGES_CSV}")

if __name__ == "__main__":
    run()
