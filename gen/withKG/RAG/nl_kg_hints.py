# nl_kg_hints.py
# Render short, readable [KG HINTS] blocks within a token budget.

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Set

# --- relation → natural language ---
def rel2text(rel: str, rela: str) -> str:
    r = (rela or rel or "").strip().lower()
    table = {
        "isa": "is a type of",
        "inverse_isa": "is a type of",
        "same_as": "is the same as",
        "mapped_to": "is mapped to",
        "associated_with": "is associated with",
        "has_associated_finding": "is associated with",
        "finding_site_of": "affects",
        "associated_morphology_of": "has morphology",
        "due_to": "can be due to",
        "causative_agent_of": "may be caused by",
        "treated_by": "is treated with",
        "managed_by": "is managed by",
        "complicates": "can complicate",
        "has_complication": "can be complicated by",
    }
    return table.get(r, "is related to")

# --- token helpers ---
def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)

def count_tokens(tok, text: str) -> int:
    if tok is None:
        return _approx_token_count(text)
    try:
        return int(tok(text, add_special_tokens=False, return_length=True)["length"][0])
    except Exception:
        return _approx_token_count(text)

def trim_to_token_budget(tok, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text: return ""
    if count_tokens(tok, text) <= max_tokens: return text
    lo, hi = 0, len(text); best = ""
    while lo <= hi:
        mid = (lo + hi)//2
        cand = text[:mid]
        if count_tokens(tok, cand) <= max_tokens:
            best = cand; lo = mid + 1
        else:
            hi = mid - 1
    return best

# --- verbalizers ---
def verbalize_visit_evidence(src_keys: Iterable[str],
                             code2name: Dict[str, str],
                             max_items: int = 6) -> List[str]:
    """
    src_keys like ['ATC:B01A','PROC:54.91','LNC:1015-1'].
    code2name maps those keys to human names when possible.
    """
    out = []
    for k in list(src_keys)[:max_items]:
        typ, code = (k.split(":",1)+[""])[:2]
        nm = (code2name.get(k,"") or "").strip()
        if typ == "ATC":
            out.append(f"{(nm or 'Pharmacologic therapy')} (ATC {code})")
        elif typ in {"LNC","LOINC"}:
            out.append(f"{(nm or 'Laboratory assay')} (LOINC {code})")
        elif typ == "PROC":
            out.append(f"{(nm or 'Procedure performed')} (ICD-9-Proc {code})")
        else:
            out.append(f"{(nm or typ)} ({code})")
    return out

def verbalize_neighbor_edges(G,
                             seed_cuis: Set[str],
                             hops: int = 1,
                             rel_whitelist: Optional[Set[str]] = None,
                             rela_whitelist: Optional[Set[str]] = None,
                             max_lines: int = 6) -> List[str]:
    """Return short sentences like 'Troponin I is associated with myocardial injury.'"""
    out = []
    seen=set(seed_cuis); frontier=set(seed_cuis)
    for _ in range(hops):
        nxt=set()
        for u in list(frontier):
            if u not in G: continue
            subj = (G.nodes[u].get("name") or "").strip()
            for v in G.successors(u):
                d = G[u][v]
                rel, rela = (d.get("rel") or ""), (d.get("rela") or "")
                if rel_whitelist and rel not in rel_whitelist:   continue
                if rela_whitelist and rela not in rela_whitelist: continue
                if v in seen: continue
                obj = (G.nodes[v].get("name") or "").strip()
                if subj and obj and len(out) < max_lines:
                    out.append(f"{subj} {rel2text(rel, rela)} {obj}.")
                nxt.add(v)
        frontier = nxt - seen
        seen |= nxt
        if len(out) >= max_lines or not frontier:
            break
    # de-dup
    uniq=[]; seen_txt=set()
    for s in out:
        t = s.lower()
        if t not in seen_txt:
            uniq.append(s); seen_txt.add(t)
    return uniq[:max_lines]

def build_nl_kg_hints(tok,
                      evidence_lines: List[str],
                      edge_lines: List[str],
                      candidate_names: List[str],
                      budget_tokens: int = 600) -> str:
    """Assemble the [KG HINTS] block within a token budget."""
    def assemble(evs, eds, cands):
        parts = ["[KG HINTS]"]
        if evs:
            parts.append("Evidence from this visit:")
            parts += [f"- {x}" for x in evs]
        if eds:
            parts.append("Graph context:")
            parts += [f"- {x}" for x in eds]
        if cands:
            parts.append("Likely diagnoses suggested by the graph:")
            parts += [f"- {x}" for x in cands]
        return "\n".join(parts)

    evs=list(evidence_lines or [])
    eds=list(edge_lines or [])
    cands=list(candidate_names or [])
    text = assemble(evs, eds, cands)

    while count_tokens(tok, text) > budget_tokens:
        if cands and len(cands) > 4:
            cands = cands[:-2]
        elif eds and len(eds) > 3:
            eds = eds[:-1]
        elif evs and len(evs) > 2:
            evs = evs[:-1]
        else:
            text = trim_to_token_budget(tok, text, budget_tokens)
            break
        text = assemble(evs, eds, cands)
    return text
