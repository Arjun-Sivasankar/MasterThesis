import json
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import numpy as np

sys.path.insert(0, '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/withKG/withRAG/')

from retrieval_utils import (
    MedicalFactIndex,
    CombinedFactIndex,
    SapBERTEncoder,
    retrieve_facts,
    get_relationship_weights_for_facts
)

def load_unweighted_paths(jsonl_path):
    """Load unweighted retrievals as dict keyed by (patient_id, hadm_id)."""
    data = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            key = (str(obj.get("patient_id")), str(obj.get("hadm_id")))
            data[key] = obj.get("paths", [])
    return data

def build_query(sample, note_sections):
    return "\n".join([sample.get(sec, "") for sec in note_sections if sample.get(sec, "")]).strip()

def weighted_retrieve_and_compare(
    input_jsonl,
    unweighted_jsonl,
    output_jsonl,
    fact_index,
    encoder,
    k=50,
    note_sections=None
):
    # Load unweighted retrievals
    unweighted_dict = load_unweighted_paths(unweighted_jsonl)
    weighted_dict = {}
    comparison_stats = []
    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"Weighted retrieval for {Path(input_jsonl).stem}"):
            sample = json.loads(line)
            key = (str(sample.get("patient_id")), str(sample.get("admission_id", sample.get("hadm_id"))))
            query = build_query(sample, note_sections)
            if not query:
                weighted_paths = []
            else:
                weighted_paths = retrieve_facts(
                    query_text=query,
                    fact_index=fact_index,
                    encoder=encoder,
                    k=k,
                    use_weighting=True,
                    alpha=0.3,
                    h1_ratio=0.5,
                    rel_aggregation='sum',
                    debug=False
                )
            # Save weighted
            out = {
                "patient_id": key[0],
                "hadm_id": key[1],
                "paths": weighted_paths
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            weighted_dict[key] = weighted_paths

            # For comparison
            unweighted_paths = unweighted_dict.get(key, [])
            weighted_wts = get_relationship_weights_for_facts(weighted_paths, fact_index, aggregation='sum')
            unweighted_wts = get_relationship_weights_for_facts(unweighted_paths, fact_index, aggregation='sum')
            comparison_stats.append({
                "patient_id": key[0],
                "hadm_id": key[1],
                "avg_weight_weighted": float(np.mean(weighted_wts)) if weighted_wts else 0.0,
                "avg_weight_unweighted": float(np.mean(unweighted_wts)) if unweighted_wts else 0.0,
                "overlap": len(set(weighted_paths) & set(unweighted_paths)),
                "n_weighted": len(weighted_paths),
                "n_unweighted": len(unweighted_paths)
            })
    return comparison_stats

def summarize_comparison(comparison_stats, split, mode):
    weighted_avgs = [x["avg_weight_weighted"] for x in comparison_stats if x["n_weighted"] > 0]
    unweighted_avgs = [x["avg_weight_unweighted"] for x in comparison_stats if x["n_unweighted"] > 0]
    overlap = [x["overlap"] for x in comparison_stats]
    n = len(comparison_stats)
    summary = {
        "split": split,
        "mode": mode,
        "n_samples": n,
        "mean_weight_weighted": float(np.mean(weighted_avgs)) if weighted_avgs else 0.0,
        "mean_weight_unweighted": float(np.mean(unweighted_avgs)) if unweighted_avgs else 0.0,
        "mean_overlap": float(np.mean(overlap)) if overlap else 0.0,
        "overlap_pct": float(100 * np.mean(overlap) / 50) if overlap else 0.0,  # assuming k=50
        "weight_boost_pct": float(
            100 * (np.mean(weighted_avgs) - np.mean(unweighted_avgs)) / np.mean(unweighted_avgs)
        ) if unweighted_avgs and np.mean(unweighted_avgs) > 0 else 0.0
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--mode", type=str, required=True, choices=["h1", "h2", "combined"])
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    base_dir = Path("/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis")
    data_dir = base_dir / "dataset/history_aware_data4/jsonl_output_final"

    input_jsonl = str(data_dir / f"{args.split}_modular.jsonl")
    note_sections = [
        "note_chief_complaint",
        "note_history_present_illness",
        "note_past_medical_history",
        "note_family_history",
        "note_physical_exam",
        "note_medications_on_admission"
    ]

    encoder_model = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    print("Loading SapBERT encoder...")
    encoder = SapBERTEncoder(encoder_model)

    if args.mode == "h1":
        print("Loading H1 index...")
        fact_index = MedicalFactIndex(str(base_dir / "fact_indexes_history_aware/h1_index"), index_type="h1")
    elif args.mode == "h2":
        print("Loading H2 index...")
        fact_index = MedicalFactIndex(str(base_dir / "fact_indexes_history_aware/h2_index"), index_type="h2")
    elif args.mode == "combined":
        print("Loading Combined index...")
        h1_index_dir = str(base_dir / "fact_indexes_history_aware/h1_index")
        h2_index_dir = str(base_dir / "fact_indexes_history_aware/h2_index")
        fact_index = CombinedFactIndex(h1_index_dir, h2_index_dir)
    else:
        raise ValueError("Unknown mode")

    # Unweighted paths file
    unweighted_jsonl = str(data_dir / f"{args.split}_{args.mode}.jsonl")
    # Weighted output file
    weighted_jsonl = str(data_dir / f"{args.split}_{args.mode}_weighted.jsonl")

    # Run weighted retrieval and compare
    stats = weighted_retrieve_and_compare(
        input_jsonl=input_jsonl,
        unweighted_jsonl=unweighted_jsonl,
        output_jsonl=weighted_jsonl,
        fact_index=fact_index,
        encoder=encoder,
        k=args.k,
        note_sections=note_sections
    )

    # Summarize and save
    summary = summarize_comparison(stats, args.split, args.mode)
    print(f"\n=== {args.split.upper()} {args.mode.upper()} WEIGHTED vs UNWEIGHTED ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    # Save stats
    stats_path = str(data_dir / f"{args.split}_{args.mode}_weighted_vs_unweighted_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"summary": summary, "per_sample": stats}, f, indent=2)
    print(f"Saved comparison stats to {stats_path}")

if __name__ == "__main__":
    main()