import json
import csv
from pathlib import Path
import argparse
from transformers import AutoTokenizer

def load_paths_jsonl(paths_jsonl):
    """Load paths from *_h1.jsonl or *_h2.jsonl as a dict keyed by (patient_id, hadm_id)."""
    data = {}
    with open(paths_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            key = (str(obj.get("patient_id")), str(obj.get("hadm_id")))
            data[key] = obj.get("paths", [])
    return data

def read_tsv(tsv_path):
    """Read TSV as list of dicts."""
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)

def write_tsv(rows, out_path):
    """Write list of dicts to TSV."""
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'target'], delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def enforce_token_budget(kg_facts, tokenizer, max_tokens):
    """Return as many facts as fit in the token budget."""
    selected = []
    total_tokens = 0
    for fact in kg_facts:
        tokens = len(tokenizer.tokenize(fact))
        if total_tokens + tokens > max_tokens:
            break
        selected.append(fact)
        total_tokens += tokens
    return selected

def insert_kg_facts(prompt, kg_facts, tokenizer, max_kg_tokens=1500):
    """Insert [KNOWLEDGE GRAPH FACTS] before [TASK], add facts up to token budget."""
    kg_tag = "\n[KNOWLEDGE GRAPH FACTS]"
    task_tag = "\nTask"
    # Enforce token budget
    kg_facts = enforce_token_budget(kg_facts, tokenizer, max_kg_tokens)
    facts_str = kg_tag + "\n"
    if kg_facts:
        facts_str += "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(kg_facts))
    else:
        facts_str += "(none)"
    # Insert before [TASK]
    if task_tag in prompt:
        before, after = prompt.split(task_tag, 1)
        return before + facts_str + "\n" + task_tag + after
    else:
        # If [TASK] not found, append at end
        return prompt + "\n" + facts_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, help="Input prompt TSV (baseline)")
    parser.add_argument("--paths_jsonl", required=True, help="KG paths JSONL (h1 or h2)")
    parser.add_argument("--modular_jsonl", required=True, help="Original modular JSONL (for patient/admission IDs)")
    parser.add_argument("--out_tsv", required=True, help="Output TSV with KG facts")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct", help="HF tokenizer for token budget")
    parser.add_argument("--max_kg_tokens", type=int, default=1500, help="Token budget for KG facts")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load mapping from modular_jsonl: row index → (patient_id, hadm_id)
    idx_to_ids = []
    with open(args.modular_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            idx_to_ids.append((str(obj.get("patient_id")), str(obj.get("admission_id", obj.get("hadm_id")))))

    # Load KG paths
    paths_dict = load_paths_jsonl(args.paths_jsonl)

    # Read baseline TSV
    rows = read_tsv(args.tsv)
    assert len(rows) == len(idx_to_ids), "TSV and modular JSONL must have same number of rows"

    # Build new rows with KG facts inserted
    new_rows = []
    for i, row in enumerate(rows):
        key = idx_to_ids[i]
        kg_facts = paths_dict.get(key, [])
        new_prompt = insert_kg_facts(row['prompt'], kg_facts, tokenizer, args.max_kg_tokens)
        new_rows.append({'prompt': new_prompt, 'target': row['target']})

    # Write output TSV
    write_tsv(new_rows, args.out_tsv)
    print(f"Wrote {len(new_rows)} rows to {args.out_tsv}")

if __name__ == "__main__":
    main()