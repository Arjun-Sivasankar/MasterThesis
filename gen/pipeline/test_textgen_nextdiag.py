# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# test_textgen_nextdiag.py — Testing script for next diagnosis prediction models.

# Extended features:
# - Tests models trained on NEXT_DIAG_6M or NEXT_DIAG_12M
# - Includes index visit icd_code in prompts (if --include_index_icd)
# - Can evaluate emergent diagnoses only (if --new_only)
# - Supports both single-GPU and multi-GPU distributed inference
# """

# import os, json, time, argparse, logging, pickle, sys, glob
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

# import torch
# import torch.distributed as dist
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# from common_textgen_nextdiag import (
#     log, is_main_process, world_size, local_rank,
#     serialize_structured_readable, serialize_notes,
#     build_generate_kwargs, generate_terms,
#     ICDMapper, to_list, format_icd9, is_valid_icd9,
#     restrict_to, multihot, eval_pack, add_parent_macro_f1,
#     get_icd9_parent
# )

# # ---------- minimal dist utils ----------
# def maybe_init_dist():
#     if world_size() > 1 and not (dist.is_available() and dist.is_initialized()):
#         dist.init_process_group(backend="nccl", timeout=torch.distributed.elastic.utils.get_default_timeout())
#     return dist.is_initialized()

# def shard_indices(N:int, rank:int, W:int):
#     return list(range(rank, N, W))

# def barrier():
#     if dist.is_available() and dist.is_initialized():
#         try: dist.barrier()
#         except Exception: pass

# def cleanup_dist():
#     if dist.is_available() and dist.is_initialized():
#         try: dist.destroy_process_group()
#         except Exception: pass

# # ---------- prompt builder for NEXT diagnosis ----------
# def build_nextdiag_prompt(row: pd.Series, N_max_terms: int, 
#                           future_window: str = "6M",
#                           include_index_icd: bool = False) -> str:
#     """
#     Build prompt for next diagnosis prediction.
    
#     Args:
#         row: DataFrame row
#         N_max_terms: Max terms to generate
#         future_window: "6M" or "12M" for prompt wording
#         include_index_icd: Whether to include current visit diagnoses
#     """
#     s = []
#     s.append(f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}")
#     s.append(serialize_structured_readable(row, include_index_icd=include_index_icd))
    
#     notes = serialize_notes(row)
#     if notes: s.append(notes)
    
#     # Task description based on future window
#     if future_window == "6M":
#         task = "[TASK] Based on this admission, predict diagnoses likely to occur in the next 6 months."
#     elif future_window == "12M":
#         task = "[TASK] Based on this admission, predict diagnoses likely to occur in the next 12 months."
#     else:
#         task = f"[TASK] Based on this admission, predict future diagnoses in the next {future_window}."
    
#     s.append(task)
#     s.append("[FORMAT]")
#     s.append("- One diagnosis per line")
#     s.append("- Avoid abbreviations if possible")
#     s.append("- No ICD codes or explanations")
#     s.append(f"- Maximum: {N_max_terms} lines")
#     s.append("[OUTPUT]")
#     return "\n".join([x for x in s if x])

# def extract_codes(df, label_col, index_col="icd_code", new_only=False):
#     """
#     Extract codes from dataframe.
    
#     Args:
#         df: DataFrame
#         label_col: Column with target codes (e.g., NEXT_DIAG_6M)
#         index_col: Column with index visit codes (for new_only mode)
#         new_only: If True, subtract index codes from target codes
#     """
#     out = []
#     for _, r in df.iterrows():
#         lst = to_list(r.get(label_col, []))
#         lst = [format_icd9(c) for c in lst if c]
#         lst = [c for c in lst if is_valid_icd9(c)]
        
#         if new_only:
#             # Subtract index visit codes (emergent diagnoses only)
#             index_codes = to_list(r.get(index_col, []))
#             index_codes = set([format_icd9(c) for c in index_codes if c and is_valid_icd9(format_icd9(c))])
#             lst = [c for c in lst if c not in index_codes]
        
#         out.append(lst)
#     return out

# # ---------- CSV helpers (for code lists) ----------
# def _read_first_col_codes(path) -> list:
#     if not path: return []
#     try:
#         df = pd.read_csv(path)
#         if df.shape[1] == 0: return []
#         col = df.columns[0]
#         vals = [format_icd9(x) for x in df[col].tolist()]
#         vals = [v for v in vals if is_valid_icd9(v)]
#         return sorted(set(vals))
#     except Exception as e:
#         log.warning(f"Could not read codes from {path}: {e}")
#         return []

# def _read_first_col_parents(path) -> list:
#     if not path: return []
#     try:
#         df = pd.read_csv(path)
#         if df.shape[1] == 0: return []
#         col = df.columns[0]
#         raw = [format_icd9(x) for x in df[col].tolist()]
#         parents = sorted(set([get_icd9_parent(x) for x in raw if x]))
#         return parents
#     except Exception as e:
#         log.warning(f"Could not read parent codes from {path}: {e}")
#         return []

# # ---------- parent metrics (extended) ----------
# def add_parent_metrics_full(metrics_dict, gold_lists, pred_lists):
#     """Adds parent-level micro/macro/samples P/R/F1."""
#     g = [[get_icd9_parent(c) for c in lst] for lst in gold_lists]
#     p = [[get_icd9_parent(c) for c in lst] for lst in pred_lists]
#     labels = sorted({x for lst in g for x in lst})
#     Yg = multihot(g, labels); Yp = multihot(p, labels)
#     metrics_dict.update({
#         "precision_macro_parent": float(precision_score(Yg, Yp, average="macro", zero_division=0)),
#         "recall_macro_parent":    float(recall_score(Yg, Yp, average="macro", zero_division=0)),
#         "f1_macro_parent":        float(f1_score(Yg, Yp, average="macro", zero_division=0)),
#         "precision_micro_parent": float(precision_score(Yg, Yp, average="micro", zero_division=0)),
#         "recall_micro_parent":    float(recall_score(Yg, Yp, average="micro", zero_division=0)),
#         "f1_micro_parent":        float(f1_score(Yg, Yp, average="micro", zero_division=0)),
#         "precision_samples_parent": float(precision_score(Yg, Yp, average="samples", zero_division=0)),
#         "recall_samples_parent":    float(recall_score(Yg, Yp, average="samples", zero_division=0)),
#         "f1_samples_parent":        float(f1_score(Yg, Yp, average="samples", zero_division=0)),
#     })
#     return labels, Yg, Yp

# # ---------- per-label table ----------
# def per_label_table(y_true, y_pred, labels, out_csv_path=None):
#     p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
#     df = pd.DataFrame({
#         "code": labels,
#         "precision": p,
#         "recall": r,
#         "f1": f1,
#         "support": support
#     })
#     if out_csv_path:
#         os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
#         df.to_csv(out_csv_path, index=False)
#     return df

# # ---------- sample-level set metrics ----------
# def sample_set_prf(gold, pred):
#     vals = []
#     for g, p in zip(gold, pred):
#         G, P = set(g), set(p)
#         tp = len(G & P); fp = len(P - G); fn = len(G - P)
#         prec = tp / (tp+fp) if tp+fp>0 else 0.0
#         rec  = tp / (tp+fn) if tp+fn>0 else 0.0
#         f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
#         vals.append((prec, rec, f1))
#     arr = np.array(vals) if vals else np.zeros((0,3))
#     return float(arr[:,0].mean() if arr.size else 0.0), float(arr[:,1].mean() if arr.size else 0.0), float(arr[:,2].mean() if arr.size else 0.0)

# # ---------- pretty printer ----------
# def _pretty_print_block(title: str, d: dict):
#     log.info(f"\n{'='*60}")
#     log.info(f"  {title}")
#     log.info(f"{'='*60}")
#     for k in sorted(d.keys()):
#         v = d[k]
#         if isinstance(v, float):
#             log.info(f"  {k:>32s}: {v:.6f}")
#         else:
#             log.info(f"  {k:>32s}: {v}")
#     log.info(f"{'='*60}\n")

# def main():
#     ap = argparse.ArgumentParser()
#     # data
#     ap.add_argument("--data_pickle", required=True)
#     ap.add_argument("--subject_col", default="subject_id_x")
#     ap.add_argument("--label_col", default="NEXT_DIAG_6M",
#                     help="Target column: NEXT_DIAG_6M or NEXT_DIAG_12M")
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--test_only", action="store_true", 
#                     help="Use entire file as test set (skip internal split)")

#     # next-diagnosis specific
#     ap.add_argument("--future_window", default="6M",
#                     help="Prediction window for prompt wording: 6M or 12M")
#     ap.add_argument("--include_index_icd", action="store_true",
#                     help="Include index visit icd_code in prompt")
#     ap.add_argument("--new_only", action="store_true",
#                     help="Evaluate only emergent diagnoses (exclude index visit codes)")

#     # prompts/generation
#     ap.add_argument("--N_max_terms", type=int, default=12)
#     ap.add_argument("--max_len", type=int, default=3072)
#     ap.add_argument("--gen_max_new", type=int, default=128)
#     ap.add_argument("--gen_batch_size", type=int, default=8)

#     # decoding flags
#     ap.add_argument("--decoding", choices=["greedy","beam","sample"], default="greedy")
#     ap.add_argument("--num_beams", type=int, default=2)
#     ap.add_argument("--temperature", type=float, default=1.0)
#     ap.add_argument("--top_p", type=float, default=0.95)
#     ap.add_argument("--top_k", type=int, default=50)
#     ap.add_argument("--no_repeat_ngram", type=int, default=0)

#     # model/adapter
#     ap.add_argument("--base_model", required=True, help="Base model used during training")
#     ap.add_argument("--adapter_dir", required=True, help="Directory with saved PEFT adapter")
#     ap.add_argument("--use_bf16", action="store_true")

#     # mapper
#     ap.add_argument("--icd_index_dir", required=True)
#     ap.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
#     ap.add_argument("--faiss_rows", type=int, default=50)
#     ap.add_argument("--tau_cos", type=float, default=0.40)
#     ap.add_argument("--tau_final", type=float, default=0.60)
#     ap.add_argument("--w_cos", type=float, default=0.6)
#     ap.add_argument("--w_fuz", type=float, default=0.4)

#     # label space eval
#     ap.add_argument("--labels_space", choices=["full","head"], default="full")
#     ap.add_argument("--labels_head_k", type=int, default=0)
#     ap.add_argument("--print_samples", type=int, default=5)

#     # extra code lists for per-label metrics
#     ap.add_argument("--top_codes_csv", default="")
#     ap.add_argument("--bottom_codes_csv", default="")
#     ap.add_argument("--top_parent_csv", default="")

#     # multi-gpu infer
#     ap.add_argument("--distributed", action="store_true", help="Use torchrun multi-GPU sharding")
#     ap.add_argument("--tmp_dir", default="runs_textgen_nextdiag/test_shards")
#     ap.add_argument("--out_metrics", default="runs_textgen_nextdiag/test_metrics.json")

#     args = ap.parse_args()

#     # ---------- data ----------
#     if is_main_process():
#         log.info(f"Loading data: {args.data_pickle}")
#         log.info(f"Target column: {args.label_col}")
#         log.info(f"Future window: {args.future_window}")
#         log.info(f"Include index ICD: {args.include_index_icd}")
#         log.info(f"Emergent diagnoses only: {args.new_only}")

#     try:
#         df = pd.read_pickle(args.data_pickle)
#     except Exception:
#         with open(args.data_pickle, "rb") as f: 
#             df = pickle.load(f)

#     if args.test_only:
#         test_df = df.copy()
#     else:
#         subs = df[args.subject_col].dropna().unique()
#         tr_subs, te_subs = train_test_split(subs, test_size=0.10, random_state=args.seed)
#         test_df = df[df[args.subject_col].isin(te_subs)].copy()

#     # Extract gold codes (with optional new_only filtering)
#     gold_codes = extract_codes(test_df, args.label_col, 
#                                index_col="icd_code", 
#                                new_only=args.new_only)

#     # prepare label space for evaluation
#     labels_full = sorted({c for lst in gold_codes for c in lst})
#     labels_eval = labels_full
#     head_name = None
#     if args.labels_space == "head" and args.labels_head_k > 0:
#         from collections import Counter
#         cnt = Counter([c for lst in gold_codes for c in lst])
#         labels_eval = [c for c,_ in cnt.most_common(args.labels_head_k)]
#         head_name = f"HEAD_{args.labels_head_k}"

#     if is_main_process():
#         log.info(f"Test size: {len(test_df)}")
#         log.info(f"Eval label space: {len(labels_eval)} codes ({'FULL' if head_name is None else head_name})")
#         if args.new_only:
#             log.info(f"Evaluating emergent diagnoses only (excluding index visit codes)")

#     # ---------- model + adapter ----------
#     dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) \
#            else (torch.float16 if torch.cuda.is_available() else torch.float32)

#     tok_src = args.adapter_dir if os.path.exists(os.path.join(args.adapter_dir,"tokenizer_config.json")) else args.base_model
#     tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
#     if tok.pad_token_id is None: tok.pad_token = tok.eos_token
#     tok.padding_side = "right"

#     if is_main_process():
#         log.info(f"Loading base model: {args.base_model}")
#         log.info(f"Loading adapter: {args.adapter_dir}")

#     base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
#     model = PeftModel.from_pretrained(base, args.adapter_dir)
#     model.config.use_cache = True  # speed decode
#     dev = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
#     model.to(dev).eval()

#     # ---------- decoding kwargs ----------
#     gen_kwargs = build_generate_kwargs(
#         decoding=args.decoding, max_new=args.gen_max_new,
#         eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
#         num_beams=args.num_beams, temperature=args.temperature,
#         top_p=args.top_p, top_k=args.top_k, no_repeat_ngram=args.no_repeat_ngram
#     )

#     # ---------- prompts (NEXT diagnosis version) ----------
#     prompts = [build_nextdiag_prompt(r, args.N_max_terms, 
#                                      future_window=args.future_window,
#                                      include_index_icd=args.include_index_icd) 
#                for _, r in test_df.iterrows()]

#     # ---------- distributed sharding ----------
#     if args.distributed:
#         maybe_init_dist()
#         rank = int(os.environ.get("RANK", "0"))
#         W = world_size()
#         idxs = shard_indices(len(prompts), rank, W)
#     else:
#         rank, W = 0, 1
#         idxs = list(range(len(prompts)))

#     # ---------- run generation on shard ----------
#     shard_prompts = [prompts[i] for i in idxs]
#     shard_gold    = [gold_codes[i] for i in idxs]

#     terms_lists = []
#     bs = max(1, int(args.gen_batch_size))
#     t0 = time.time()
    
#     if is_main_process() or rank == 0:
#         log.info(f"Starting generation for {len(shard_prompts)} samples on rank {rank}...")
    
#     for i in range(0, len(shard_prompts), bs):
#         batch = shard_prompts[i:i+bs]
#         terms = generate_terms(model, tok, batch, max_len=args.max_len, gen_kwargs=gen_kwargs, batch_size=bs)
#         terms_lists.extend(terms)
        
#         if (i // bs) % 10 == 0 and (is_main_process() or rank == 0):
#             log.info(f"[Rank {rank}] Processed {i+len(batch)}/{len(shard_prompts)} samples")
    
#     elapsed = time.time() - t0
#     if is_main_process() or rank == 0:
#         per = elapsed / max(1, len(idxs))
#         log.info(f"Generation done: {elapsed:.1f}s total, {per:.2f}s/sample on rank {rank}")

#     # ---------- mapping ----------
#     if is_main_process() or rank == 0:
#         log.info(f"[Rank {rank}] Starting ICD-9 mapping...")
    
#     mapper = ICDMapper(
#         index_dir=args.icd_index_dir,
#         encoder_model_cli=args.encoder_model,
#         tau_cos=args.tau_cos, tau_final=args.tau_final,
#         w_cos=args.w_cos, w_fuz=args.w_fuz,
#         faiss_rows=args.faiss_rows
#     )
#     mapped_codes = mapper.map_terms(terms_lists)

#     if is_main_process() or rank == 0:
#         log.info(f"[Rank {rank}] Mapping complete")

#     # ---------- persist shard ----------
#     os.makedirs(args.tmp_dir, exist_ok=True)
#     shard_path = os.path.join(args.tmp_dir, f"shard_{rank:03d}_of_{W:03d}.pkl")
#     with open(shard_path, "wb") as f:
#         pickle.dump({
#             "idxs": idxs,
#             "free_text": terms_lists,
#             "mapped": mapped_codes,
#             "gold": shard_gold,
#         }, f)
#     log.info(f"[Rank {rank}] Wrote shard to {shard_path}")
#     barrier()

#     # ---------- rank-0 merge & metrics ----------
#     if rank == 0:
#         log.info("="*80)
#         log.info("MERGING SHARDS AND COMPUTING METRICS")
#         log.info("="*80)
        
#         shards = sorted(glob.glob(os.path.join(args.tmp_dir, f"shard_*_of_{W:03d}.pkl")))
#         log.info(f"Found {len(shards)} shard files")
        
#         all_idx, all_free, all_map, all_gold = [], [], [], []
#         for sp in shards:
#             with open(sp, "rb") as f:
#                 D = pickle.load(f)
#             all_idx.extend(D["idxs"]); all_free.extend(D["free_text"])
#             all_map.extend(D["mapped"]); all_gold.extend(D["gold"])

#         # restore order
#         order = np.argsort(np.array(all_idx))
#         free_text = [all_free[i] for i in order]
#         pred_codes = [all_map[i] for i in order]
#         gold_all   = [all_gold[i] for i in order]

#         # restrict to eval set
#         gold_eval = restrict_to(gold_all, labels_eval)
#         pred_eval = restrict_to(pred_codes, labels_eval)

#         log.info(f"Computing metrics for {len(gold_eval)} samples...")

#         # base metrics (code level)
#         Yt = multihot(gold_eval, labels_eval)
#         Yp = multihot(pred_eval, labels_eval)
#         metrics = eval_pack(Yt, Yp)  # micro/macro/samples

#         # parent metrics (extended)
#         parent_metrics = {}
#         _parent_labels, _Yg_par, _Yp_par = add_parent_metrics_full(parent_metrics, gold_eval, pred_eval)
#         metrics.update(parent_metrics)

#         # also keep legacy parent macro F1 for compatibility
#         add_parent_macro_f1(metrics, gold_eval, pred_eval)

#         # diagnostics from mapper
#         if mapper.last_stats:
#             n_terms = np.array([n for (n,m) in mapper.last_stats], dtype=np.float32)
#             n_map   = np.array([m for (n,m) in mapper.last_stats], dtype=np.float32)
#             metrics["mean_terms_per_visit"] = float(n_terms.mean())
#             metrics["mean_mapped_terms_per_visit"] = float(n_map.mean())
#             metrics["unmappable_term_rate"] = float(np.mean(np.where(n_terms>0, 1.0 - (n_map/np.maximum(n_terms,1)), 0.0)))

#         # sample-level set metrics (explicit)
#         ps, rs, fs = sample_set_prf(gold_eval, pred_eval)
#         metrics["precision_samples_set"] = ps
#         metrics["recall_samples_set"] = rs
#         metrics["f1_samples_set"] = fs

#         # per-label CSV: FULL
#         out_dir = os.path.dirname(os.path.abspath(args.out_metrics))
#         per_label_table(Yt, Yp, labels_eval, os.path.join(out_dir, "per_label_FULL.csv"))
#         log.info(f"Saved per-label metrics: {os.path.join(out_dir, 'per_label_FULL.csv')}")

#         # Optional HEAD-K per-label CSV (if used)
#         head_labels = None
#         if head_name is not None:
#             head_labels = labels_eval
#             Yt_h = multihot(gold_eval, head_labels); Yp_h = multihot(pred_eval, head_labels)
#             per_label_table(Yt_h, Yp_h, head_labels, os.path.join(out_dir, f"per_label_{head_name}.csv"))
#             log.info(f"Saved head metrics: {os.path.join(out_dir, f'per_label_{head_name}.csv')}")

#         # Top/Bottom/Parent CSVs (if provided)
#         top_codes = _read_first_col_codes(args.top_codes_csv)
#         bottom_codes = _read_first_col_codes(args.bottom_codes_csv)
#         top_parents = _read_first_col_parents(args.top_parent_csv)

#         results_ext = {}

#         if top_codes:
#             log.info(f"Computing metrics for TOP_{len(top_codes)} codes...")
#             g = restrict_to(gold_all, top_codes); p = restrict_to(pred_codes, top_codes)
#             Yg = multihot(g, top_codes); Yp2 = multihot(p, top_codes)
#             results_ext[f"TOP_{len(top_codes)}_CODES"] = eval_pack(Yg, Yp2)
#             per_label_table(Yg, Yp2, top_codes, os.path.join(out_dir, f"per_label_TOP_{len(top_codes)}_CODES.csv"))

#         if bottom_codes:
#             log.info(f"Computing metrics for BOTTOM_{len(bottom_codes)} codes...")
#             g = restrict_to(gold_all, bottom_codes); p = restrict_to(pred_codes, bottom_codes)
#             Yg = multihot(g, bottom_codes); Yp2 = multihot(p, bottom_codes)
#             results_ext[f"BOTTOM_{len(bottom_codes)}_CODES"] = eval_pack(Yg, Yp2)
#             per_label_table(Yg, Yp2, bottom_codes, os.path.join(out_dir, f"per_label_BOTTOM_{len(bottom_codes)}_CODES.csv"))

#         if top_parents:
#             log.info(f"Computing metrics for TOP_{len(top_parents)} parent categories...")
#             g_par = [[get_icd9_parent(c) for c in lst] for lst in gold_all]
#             p_par = [[get_icd9_parent(c) for c in lst] for lst in pred_codes]
#             g_par_r = restrict_to(g_par, top_parents)
#             p_par_r = restrict_to(p_par, top_parents)
#             YgP = multihot(g_par_r, top_parents); YpP = multihot(p_par_r, top_parents)
#             results_ext[f"TOP_{len(top_parents)}_PARENTS"] = eval_pack(YgP, YpP)
#             per_label_table(YgP, YpP, top_parents, os.path.join(out_dir, f"per_label_TOP_{len(top_parents)}_PARENTS.csv"))

#         # print samples (with per-sample P/R/F1)
#         n_show = min(args.print_samples, len(free_text))
#         log.info("\n" + "="*80)
#         log.info("SAMPLE PREDICTIONS")
#         log.info("="*80)
#         for i in range(n_show):
#             G = set(gold_all[i]); P = set(pred_codes[i])
#             tp = len(G & P); fp = len(P - G); fn = len(G - P)
#             prec = tp / (tp+fp) if tp+fp>0 else 0.0
#             rec  = tp / (tp+fn) if tp+fn>0 else 0.0
#             f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
#             log.info(f"\n[Sample {i+1}/{n_show}]")
#             log.info(f"  GOLD codes ({len(G)}): {', '.join(sorted(G)) if G else '(none)'}")
#             log.info(f"  FREE-TEXT terms ({len(free_text[i])}):")
#             for t in free_text[i][:args.N_max_terms]:
#                 log.info(f"    - {t}")
#             log.info(f"  MAPPED ICD-9 ({len(P)}): {', '.join(sorted(P)) if P else '(none)'}")
#             log.info(f"  Sample metrics: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

#         # write metrics JSON
#         payload = {
#             "prediction_task": f"next_diagnosis_{args.future_window}",
#             "label_column": args.label_col,
#             "include_index_icd": args.include_index_icd,
#             "emergent_only": args.new_only,
#             "label_space": ("FULL" if head_name is None else head_name),
#             "num_samples": len(free_text),
#             "num_labels_eval": len(labels_eval),
#             "metrics": metrics,
#             **results_ext
#         }
#         os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
#         with open(args.out_metrics, "w") as f:
#             json.dump(payload, f, indent=2)
#         log.info(f"\nMetrics saved to: {args.out_metrics}")

#         # ---- PRINT METRICS AT THE END ----
#         _pretty_print_block("OVERALL METRICS (code-level)", metrics)
#         for key in sorted(results_ext.keys()):
#             _pretty_print_block(f"{key} METRICS", results_ext[key])

#         log.info("\n" + "="*80)
#         log.info("TESTING COMPLETE")
#         log.info("="*80)

#     cleanup_dist()
#     return 0

# if __name__ == "__main__":
#     sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_textgen_nextdiag.py — Testing script for next diagnosis prediction models.

Extended features:
- Tests models trained on NEXT_DIAG_6M or NEXT_DIAG_12M
- Includes index visit icd_code in prompts (if --include_index_icd)
- Can evaluate emergent diagnoses only (if --new_only)
- Supports both single-GPU and multi-GPU distributed inference
- Adaptive vs static max_terms modes
- Uses pre-split test data
"""

import os, json, time, argparse, logging, pickle, sys, glob
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from common_textgen_nextdiag import (
    log, is_main_process, world_size, local_rank,
    serialize_structured_readable, serialize_notes,
    build_generate_kwargs, generate_terms,
    ICDMapper, to_list, format_icd9, is_valid_icd9,
    restrict_to, multihot, eval_pack, add_parent_macro_f1,
    get_icd9_parent, load_kg_mappings
)

# ---------- minimal dist utils ----------
def maybe_init_dist():
    if world_size() > 1 and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl", timeout=torch.distributed.elastic.utils.get_default_timeout())
    return dist.is_initialized()

def shard_indices(N:int, rank:int, W:int):
    return list(range(rank, N, W))

def barrier():
    if dist.is_available() and dist.is_initialized():
        try: dist.barrier()
        except Exception: pass

def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        try: dist.destroy_process_group()
        except Exception: pass

# ---------- prompt builder for NEXT diagnosis ----------
def build_nextdiag_prompt(row: pd.Series, N_max_terms: int, 
                          future_window: str = "6M",
                          include_index_icd: bool = False,
                          adaptive_max_terms: bool = False,
                          kg_map: Dict[str, str] = None  # NEW PARAMETER
                          ) -> str:
    """
    Build prompt for next diagnosis prediction.
    
    Analysis results:
    - 6M: median=14 (all), median=9 (emergent), ratio=1.14x
    - 12M: median=12 (all), median=7 (emergent), ratio=1.00x
    """
    s = []
    s.append(f"[VISIT] subject_id={row.get('subject_id_x','?')} hadm_id={row.get('hadm_id','?')}")
    s.append(serialize_structured_readable(row, include_index_icd=include_index_icd, kg_map=kg_map))
    
    notes = serialize_notes(row)
    if notes: s.append(notes)
    
    # Compute target count (adaptive or static)
    if adaptive_max_terms and include_index_icd:
        current_codes = to_list(row.get('icd_code', []))
        n_current = len(current_codes)
        
        if future_window == "6M":
            # 6M: expect 1.15x current codes
            expected_next = max(10, int(n_current * 1.15))
        elif future_window == "12M":
            # 12M: expect 1.0x current codes
            expected_next = max(8, int(n_current * 1.0))
        else:
            expected_next = max(10, int(n_current * 1.1))
        
        expected_next = min(expected_next, N_max_terms)
    else:
        # Static mode: use median from analysis
        expected_next = 14 if future_window == "6M" else 12
    
    # Task description based on future window
    if future_window == "6M":
        task = "[TASK] Based on this admission, predict diagnoses likely to occur in the next 6 months."
        if adaptive_max_terms:
            guidance = f"Expected: ~{expected_next} diagnoses. Include both persistent and emergent conditions."
        else:
            guidance = "Expected: 10-19 diagnoses (typically ~14). Include both persistent and emergent conditions."
    elif future_window == "12M":
        task = "[TASK] Based on this admission, predict diagnoses likely to occur in the next 12 months."
        if adaptive_max_terms:
            guidance = f"Expected: ~{expected_next} diagnoses. Include both persistent and emergent conditions."
        else:
            guidance = "Expected: 8-18 diagnoses (typically ~12). Include both persistent and emergent conditions."
    else:
        task = f"[TASK] Based on this admission, predict future diagnoses in the next {future_window}."
        guidance = f"Expected: ~{expected_next} diagnoses."
    
    s.append(task)
    s.append(guidance)
    s.append("[FORMAT]")
    s.append("- One diagnosis per line")
    s.append("- Avoid abbreviations if possible")
    s.append("- No ICD codes or explanations")
    # s.append(f"- Generate close to {expected_next} lines (maximum: {N_max_terms})")
    s.append(f"- Target: around {expected_next} lines (maximum: {N_max_terms})")
    s.append("[OUTPUT]")
    return "\n".join([x for x in s if x])

def extract_codes(df, label_col, index_col="icd_code", new_only=False):
    """Extract codes from dataframe."""
    out = []
    for _, r in df.iterrows():
        lst = to_list(r.get(label_col, []))
        lst = [format_icd9(c) for c in lst if c]
        lst = [c for c in lst if is_valid_icd9(c)]
        
        if new_only:
            index_codes = to_list(r.get(index_col, []))
            index_codes = set([format_icd9(c) for c in index_codes if c and is_valid_icd9(format_icd9(c))])
            lst = [c for c in lst if c not in index_codes]
        
        out.append(lst)
    return out

# ---------- CSV helpers ----------
def _read_first_col_codes(path) -> list:
    if not path: return []
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 0: return []
        col = df.columns[0]
        vals = [format_icd9(x) for x in df[col].tolist()]
        vals = [v for v in vals if is_valid_icd9(v)]
        return sorted(set(vals))
    except Exception as e:
        log.warning(f"Could not read codes from {path}: {e}")
        return []

def _read_first_col_parents(path) -> list:
    if not path: return []
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 0: return []
        col = df.columns[0]
        raw = [format_icd9(x) for x in df[col].tolist()]
        parents = sorted(set([get_icd9_parent(x) for x in raw if x]))
        return parents
    except Exception as e:
        log.warning(f"Could not read parent codes from {path}: {e}")
        return []

# ---------- parent metrics ----------
def add_parent_metrics_full(metrics_dict, gold_lists, pred_lists):
    g = [[get_icd9_parent(c) for c in lst] for lst in gold_lists]
    p = [[get_icd9_parent(c) for c in lst] for lst in pred_lists]
    labels = sorted({x for lst in g for x in lst})
    Yg = multihot(g, labels); Yp = multihot(p, labels)
    metrics_dict.update({
        "precision_macro_parent": float(precision_score(Yg, Yp, average="macro", zero_division=0)),
        "recall_macro_parent":    float(recall_score(Yg, Yp, average="macro", zero_division=0)),
        "f1_macro_parent":        float(f1_score(Yg, Yp, average="macro", zero_division=0)),
        "precision_micro_parent": float(precision_score(Yg, Yp, average="micro", zero_division=0)),
        "recall_micro_parent":    float(recall_score(Yg, Yp, average="micro", zero_division=0)),
        "f1_micro_parent":        float(f1_score(Yg, Yp, average="micro", zero_division=0)),
        "precision_samples_parent": float(precision_score(Yg, Yp, average="samples", zero_division=0)),
        "recall_samples_parent":    float(recall_score(Yg, Yp, average="samples", zero_division=0)),
        "f1_samples_parent":        float(f1_score(Yg, Yp, average="samples", zero_division=0)),
    })
    return labels, Yg, Yp

# ---------- per-label table ----------
def per_label_table(y_true, y_pred, labels, out_csv_path=None):
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    df = pd.DataFrame({
        "code": labels,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support
    })
    if out_csv_path:
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        df.to_csv(out_csv_path, index=False)
    return df

# ---------- sample-level metrics ----------
def sample_set_prf(gold, pred):
    vals = []
    for g, p in zip(gold, pred):
        G, P = set(g), set(p)
        tp = len(G & P); fp = len(P - G); fn = len(G - P)
        prec = tp / (tp+fp) if tp+fp>0 else 0.0
        rec  = tp / (tp+fn) if tp+fn>0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        vals.append((prec, rec, f1))
    arr = np.array(vals) if vals else np.zeros((0,3))
    return float(arr[:,0].mean() if arr.size else 0.0), float(arr[:,1].mean() if arr.size else 0.0), float(arr[:,2].mean() if arr.size else 0.0)

# ---------- pretty printer ----------
def _pretty_print_block(title: str, d: dict):
    log.info(f"\n{'='*60}")
    log.info(f"  {title}")
    log.info(f"{'='*60}")
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, float):
            log.info(f"  {k:>32s}: {v:.6f}")
        else:
            log.info(f"  {k:>32s}: {v}")
    log.info(f"{'='*60}\n")

def main():
    ap = argparse.ArgumentParser()
    # data - now expects pre-split test pickle
    ap.add_argument("--test_pickle", required=True, help="Path to test data pickle")
    ap.add_argument("--label_col", default="NEXT_DIAG_6M")
    ap.add_argument("--seed", type=int, default=42)

    # next-diagnosis specific
    ap.add_argument("--future_window", default="6M")
    ap.add_argument("--include_index_icd", action="store_true")
    ap.add_argument("--new_only", action="store_true")
    ap.add_argument("--adaptive_max_terms", action="store_true",
                    help="Enable adaptive target counts based on current visit")

    # prompts/generation
    ap.add_argument("--N_max_terms", type=int, default=22)
    ap.add_argument("--max_len", type=int, default=3072)
    ap.add_argument("--gen_max_new", type=int, default=384)
    ap.add_argument("--gen_batch_size", type=int, default=8)

    # decoding
    ap.add_argument("--decoding", choices=["greedy","beam","sample"], default="greedy")
    ap.add_argument("--num_beams", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--no_repeat_ngram", type=int, default=3)

    # model/adapter
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--use_bf16", action="store_true")

    # mapper
    ap.add_argument("--icd_index_dir", required=True)
    ap.add_argument("--encoder_model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--faiss_rows", type=int, default=50)
    ap.add_argument("--tau_cos", type=float, default=0.40)
    ap.add_argument("--tau_final", type=float, default=0.60)
    ap.add_argument("--w_cos", type=float, default=0.6)
    ap.add_argument("--w_fuz", type=float, default=0.4)

    # eval
    ap.add_argument("--labels_space", choices=["full","head"], default="full")
    ap.add_argument("--labels_head_k", type=int, default=0)
    ap.add_argument("--print_samples", type=int, default=5)

    # extra CSVs
    ap.add_argument("--top_codes_csv", default="")
    ap.add_argument("--bottom_codes_csv", default="")
    ap.add_argument("--top_parent_csv", default="")

    # multi-gpu
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--tmp_dir", default="runs_textgen_nextdiag/test_shards")
    ap.add_argument("--out_metrics", default="runs_textgen_nextdiag/test_metrics.json")

    # Add KG path argument
    ap.add_argument("--kg_path", default="", 
                    help="Path to kg_nodes.csv for code-to-description mapping")

    args = ap.parse_args()

    if is_main_process():
        log.info(f"Loading test data: {args.test_pickle}")
        log.info(f"Target column: {args.label_col}")
        log.info(f"Future window: {args.future_window}")
        log.info(f"Include index ICD: {args.include_index_icd}")
        log.info(f"Emergent only: {args.new_only}")
        log.info(f"Adaptive max terms: {args.adaptive_max_terms}")

    # Load pre-split test data
    try:
        test_df = pd.read_pickle(args.test_pickle)
    except Exception:
        with open(args.test_pickle, "rb") as f: 
            test_df = pickle.load(f)

    if is_main_process():
        log.info(f"Test data loaded: {len(test_df)} samples")

    gold_codes = extract_codes(test_df, args.label_col, index_col="icd_code", new_only=args.new_only)

    labels_full = sorted({c for lst in gold_codes for c in lst})
    labels_eval = labels_full
    head_name = None
    if args.labels_space == "head" and args.labels_head_k > 0:
        from collections import Counter
        cnt = Counter([c for lst in gold_codes for c in lst])
        labels_eval = [c for c,_ in cnt.most_common(args.labels_head_k)]
        head_name = f"HEAD_{args.labels_head_k}"

    if is_main_process():
        log.info(f"Test size: {len(test_df)}")
        log.info(f"Eval label space: {len(labels_eval)} codes ({'FULL' if head_name is None else head_name})")

    # load model
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) \
           else (torch.float16 if torch.cuda.is_available() else torch.float32)

    tok_src = args.adapter_dir if os.path.exists(os.path.join(args.adapter_dir,"tokenizer_config.json")) else args.base_model
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if is_main_process():
        log.info(f"Loading base model: {args.base_model}")
        log.info(f"Loading adapter: {args.adapter_dir}")

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.config.use_cache = True
    dev = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    gen_kwargs = build_generate_kwargs(
        decoding=args.decoding, max_new=args.gen_max_new,
        eos_id=tok.eos_token_id, pad_id=tok.pad_token_id,
        num_beams=args.num_beams, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, no_repeat_ngram=args.no_repeat_ngram
    )

    log.info("Loading KG mappings...")
    kg_map = load_kg_mappings(args.kg_path) if args.kg_path else None

    # build prompts
    prompts = [build_nextdiag_prompt(r, args.N_max_terms, 
                                     future_window=args.future_window,
                                     include_index_icd=args.include_index_icd,
                                     adaptive_max_terms=args.adaptive_max_terms,
                                     kg_map=kg_map) 
               for _, r in test_df.iterrows()]

    print(prompts[0])

    # distributed sharding
    if args.distributed:
        maybe_init_dist()
        rank = int(os.environ.get("RANK", "0"))
        W = world_size()
        idxs = shard_indices(len(prompts), rank, W)
    else:
        rank, W = 0, 1
        idxs = list(range(len(prompts)))

    shard_prompts = [prompts[i] for i in idxs]
    shard_gold    = [gold_codes[i] for i in idxs]

    terms_lists = []
    bs = max(1, int(args.gen_batch_size))
    t0 = time.time()
    
    if is_main_process() or rank == 0:
        log.info(f"Starting generation for {len(shard_prompts)} samples on rank {rank}...")
    
    for i in range(0, len(shard_prompts), bs):
        batch = shard_prompts[i:i+bs]
        terms = generate_terms(model, tok, batch, max_len=args.max_len, gen_kwargs=gen_kwargs, batch_size=bs)
        terms_lists.extend(terms)
        
        if (i // bs) % 10 == 0 and (is_main_process() or rank == 0):
            log.info(f"[Rank {rank}] Processed {i+len(batch)}/{len(shard_prompts)} samples")
    
    elapsed = time.time() - t0
    if is_main_process() or rank == 0:
        per = elapsed / max(1, len(idxs))
        log.info(f"Generation done: {elapsed:.1f}s total, {per:.2f}s/sample on rank {rank}")

    # mapping
    if is_main_process() or rank == 0:
        log.info(f"[Rank {rank}] Starting ICD-9 mapping...")
    
    mapper = ICDMapper(
        index_dir=args.icd_index_dir,
        encoder_model_cli=args.encoder_model,
        tau_cos=args.tau_cos, tau_final=args.tau_final,
        w_cos=args.w_cos, w_fuz=args.w_fuz,
        faiss_rows=args.faiss_rows
    )
    mapped_codes = mapper.map_terms(terms_lists)

    if is_main_process() or rank == 0:
        log.info(f"[Rank {rank}] Mapping complete")

    # persist shard
    os.makedirs(args.tmp_dir, exist_ok=True)
    shard_path = os.path.join(args.tmp_dir, f"shard_{rank:03d}_of_{W:03d}.pkl")
    with open(shard_path, "wb") as f:
        pickle.dump({
            "idxs": idxs,
            "free_text": terms_lists,
            "mapped": mapped_codes,
            "gold": shard_gold,
        }, f)
    log.info(f"[Rank {rank}] Wrote shard to {shard_path}")
    barrier()

    # rank-0 merge & metrics
    if rank == 0:
        log.info("="*80)
        log.info("MERGING SHARDS AND COMPUTING METRICS")
        log.info("="*80)
        
        shards = sorted(glob.glob(os.path.join(args.tmp_dir, f"shard_*_of_{W:03d}.pkl")))
        log.info(f"Found {len(shards)} shard files")
        
        all_idx, all_free, all_map, all_gold = [], [], [], []
        for sp in shards:
            with open(sp, "rb") as f:
                D = pickle.load(f)
            all_idx.extend(D["idxs"]); all_free.extend(D["free_text"])
            all_map.extend(D["mapped"]); all_gold.extend(D["gold"])

        order = np.argsort(np.array(all_idx))
        free_text = [all_free[i] for i in order]
        pred_codes = [all_map[i] for i in order]
        gold_all   = [all_gold[i] for i in order]

        gold_eval = restrict_to(gold_all, labels_eval)
        pred_eval = restrict_to(pred_codes, labels_eval)

        log.info(f"Computing metrics for {len(gold_eval)} samples...")

        Yt = multihot(gold_eval, labels_eval)
        Yp = multihot(pred_eval, labels_eval)
        metrics = eval_pack(Yt, Yp)

        parent_metrics = {}
        _parent_labels, _Yg_par, _Yp_par = add_parent_metrics_full(parent_metrics, gold_eval, pred_eval)
        metrics.update(parent_metrics)

        add_parent_macro_f1(metrics, gold_eval, pred_eval)

        if mapper.last_stats:
            n_terms = np.array([n for (n,m) in mapper.last_stats], dtype=np.float32)
            n_map   = np.array([m for (n,m) in mapper.last_stats], dtype=np.float32)
            metrics["mean_terms_per_visit"] = float(n_terms.mean())
            metrics["mean_mapped_terms_per_visit"] = float(n_map.mean())
            metrics["unmappable_term_rate"] = float(np.mean(np.where(n_terms>0, 1.0 - (n_map/np.maximum(n_terms,1)), 0.0)))

        ps, rs, fs = sample_set_prf(gold_eval, pred_eval)
        metrics["precision_samples_set"] = ps
        metrics["recall_samples_set"] = rs
        metrics["f1_samples_set"] = fs

        out_dir = os.path.dirname(os.path.abspath(args.out_metrics))
        per_label_table(Yt, Yp, labels_eval, os.path.join(out_dir, "per_label_FULL.csv"))
        log.info(f"Saved per-label metrics: {os.path.join(out_dir, 'per_label_FULL.csv')}")

        head_labels = None
        if head_name is not None:
            head_labels = labels_eval
            Yt_h = multihot(gold_eval, head_labels); Yp_h = multihot(pred_eval, head_labels)
            per_label_table(Yt_h, Yp_h, head_labels, os.path.join(out_dir, f"per_label_{head_name}.csv"))
            log.info(f"Saved head metrics: {os.path.join(out_dir, f'per_label_{head_name}.csv')}")

        top_codes = _read_first_col_codes(args.top_codes_csv)
        bottom_codes = _read_first_col_codes(args.bottom_codes_csv)
        top_parents = _read_first_col_parents(args.top_parent_csv)

        results_ext = {}

        if top_codes:
            log.info(f"Computing metrics for TOP_{len(top_codes)} codes...")
            g = restrict_to(gold_all, top_codes); p = restrict_to(pred_codes, top_codes)
            Yg = multihot(g, top_codes); Yp2 = multihot(p, top_codes)
            results_ext[f"TOP_{len(top_codes)}_CODES"] = eval_pack(Yg, Yp2)
            per_label_table(Yg, Yp2, top_codes, os.path.join(out_dir, f"per_label_TOP_{len(top_codes)}_CODES.csv"))

        if bottom_codes:
            log.info(f"Computing metrics for BOTTOM_{len(bottom_codes)} codes...")
            g = restrict_to(gold_all, bottom_codes); p = restrict_to(pred_codes, bottom_codes)
            Yg = multihot(g, bottom_codes); Yp2 = multihot(p, bottom_codes)
            results_ext[f"BOTTOM_{len(bottom_codes)}_CODES"] = eval_pack(Yg, Yp2)
            per_label_table(Yg, Yp2, bottom_codes, os.path.join(out_dir, f"per_label_BOTTOM_{len(bottom_codes)}_CODES.csv"))

        if top_parents:
            log.info(f"Computing metrics for TOP_{len(top_parents)} parent categories...")
            g_par = [[get_icd9_parent(c) for c in lst] for lst in gold_all]
            p_par = [[get_icd9_parent(c) for c in lst] for lst in pred_codes]
            g_par_r = restrict_to(g_par, top_parents)
            p_par_r = restrict_to(p_par, top_parents)
            YgP = multihot(g_par_r, top_parents); YpP = multihot(p_par_r, top_parents)
            results_ext[f"TOP_{len(top_parents)}_PARENTS"] = eval_pack(YgP, YpP)
            per_label_table(YgP, YpP, top_parents, os.path.join(out_dir, f"per_label_TOP_{len(top_parents)}_PARENTS.csv"))

        n_show = min(args.print_samples, len(free_text))
        log.info("="*80)
        log.info("SAMPLE PREDICTIONS")
        log.info("="*80)
        for i in range(n_show):
            G = set(gold_all[i]); P = set(pred_codes[i])
            tp = len(G & P); fp = len(P - G); fn = len(G - P)
            prec = tp / (tp+fp) if tp+fp>0 else 0.0
            rec  = tp / (tp+fn) if tp+fn>0 else 0.0
            f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
            log.info(f"\n[Sample {i+1}/{n_show}]")
            log.info(f"  GOLD codes ({len(G)}): {', '.join(sorted(G)) if G else '(none)'}")
            log.info(f"  FREE-TEXT terms ({len(free_text[i])}):")
            for t in free_text[i][:args.N_max_terms]:
                log.info(f"    - {t}")
            log.info(f"  MAPPED ICD-9 ({len(P)}): {', '.join(sorted(P)) if P else '(none)'}")
            log.info(f"  Sample metrics: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        mode_str = "ADAPTIVE" if args.adaptive_max_terms else "STATIC"
        payload = {
            "prediction_task": f"next_diagnosis_{args.future_window}",
            "label_column": args.label_col,
            "include_index_icd": args.include_index_icd,
            "emergent_only": args.new_only,
            "max_terms_mode": mode_str,
            "N_max_terms": args.N_max_terms,
            "label_space": ("FULL" if head_name is None else head_name),
            "num_samples": len(free_text),
            "num_labels_eval": len(labels_eval),
            "metrics": metrics,
            **results_ext
        }
        os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
        with open(args.out_metrics, "w") as f:
            json.dump(payload, f, indent=2)
        log.info(f"\nMetrics saved to: {args.out_metrics}")

        _pretty_print_block("OVERALL METRICS (code-level)", metrics)
        for key in sorted(results_ext.keys()):
            _pretty_print_block(f"{key} METRICS", results_ext[key])

        log.info("="*80)
        log.info("TESTING COMPLETE")
        log.info("="*80)

    cleanup_dist()
    return 0

if __name__ == "__main__":
    sys.exit(main())