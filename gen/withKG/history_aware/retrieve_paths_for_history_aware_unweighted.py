import json
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

sys.path.insert(0, '/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/gen/withKG/')

from retrieval_utils import (
    MedicalFactIndex,
    CombinedFactIndex,
    SapBERTEncoder,
    retrieve_facts
)

NOTE_SECTIONS = [
    "note_chief_complaint",
    "note_history_present_illness",
    "note_past_medical_history",
    "note_family_history",
    "note_physical_exam",
    "note_medications_on_admission"
]

def build_query(sample):
    return "\n".join([sample.get(sec, "") for sec in NOTE_SECTIONS if sample.get(sec, "")]).strip()

def retrieve_and_save_paths(
    input_jsonl,
    output_jsonl,
    fact_index,
    encoder,
    k=50,
    n_samples=None
):
    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:
        for i, line in enumerate(tqdm(fin, desc=f"Retrieving paths for {Path(input_jsonl).stem}")):
            if n_samples is not None and i >= n_samples:
                break
            sample = json.loads(line)
            query = build_query(sample)
            if not query:
                paths = []
            else:
                paths = retrieve_facts(
                    query_text=query,
                    fact_index=fact_index,
                    encoder=encoder,
                    k=k,
                    use_weighting=False,
                    alpha=0.3,
                    h1_ratio=0.5,
                    rel_aggregation='sum',
                    debug=False
                )
            out = {
                "patient_id": sample.get("patient_id", None),
                "hadm_id": sample.get("admission_id", sample.get("hadm_id", None)),
                "paths": paths
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--h1", action="store_true", help="Run H1 retrieval")
    parser.add_argument("--h2", action="store_true", help="Run H2 retrieval")
    parser.add_argument("--combined", action="store_true", help="Run combined retrieval")
    parser.add_argument("--k", type=int, default=50, help="Number of paths to retrieve")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to process (for smoke test)")
    args = parser.parse_args()

    base_dir = Path("/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis")
    data_dir = base_dir / "dataset/history_aware_data/jsonl_output_final"
    out_dir = data_dir

    input_jsonl = str(data_dir / f"{args.split}_modular.jsonl")
    encoder_model = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    print("Loading SapBERT encoder...")
    encoder = SapBERTEncoder(encoder_model)

    if args.h1:
        print("Loading H1 index...")
        h1_index = MedicalFactIndex(str(base_dir / "fact_indexes_history_aware/h1_index"), index_type="h1")
        output_jsonl = str(out_dir / f"{args.split}_h1.jsonl")
        retrieve_and_save_paths(input_jsonl, output_jsonl, h1_index, encoder, k=args.k, n_samples=args.n_samples)

    if args.h2:
        print("Loading H2 index...")
        h2_index = MedicalFactIndex(str(base_dir / "fact_indexes_history_aware/h2_index"), index_type="h2")
        output_jsonl = str(out_dir / f"{args.split}_h2.jsonl")
        retrieve_and_save_paths(input_jsonl, output_jsonl, h2_index, encoder, k=args.k, n_samples=args.n_samples)

    if args.combined:
        print("Loading Combined index...")
        h1_index_dir = str(base_dir / "fact_indexes_history_aware/h1_index")
        h2_index_dir = str(base_dir / "fact_indexes_history_aware/h2_index")
        combined_index = CombinedFactIndex(h1_index_dir, h2_index_dir)
        output_jsonl = str(out_dir / f"{args.split}_combined.jsonl")
        retrieve_and_save_paths(input_jsonl, output_jsonl, combined_index, encoder, k=args.k, n_samples=args.n_samples)

if __name__ == "__main__":
    main()