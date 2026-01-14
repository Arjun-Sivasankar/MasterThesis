import os
import json
import torch
import numpy as np
import faiss
import re
from rapidfuzz.fuzz import token_set_ratio
from transformers import AutoTokenizer, AutoModel

# Reusing the functions from the original file
def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class HFMeanEncoder:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device).eval()

    def _mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        s = torch.sum(last_hidden * mask, dim=1)
        d = torch.clamp(mask.sum(dim=1), min=1e-9)
        return s / d

    @torch.no_grad()
    def encode(self, texts: list, batch_size=32) -> np.ndarray:
        if not texts: return np.zeros((0,768), dtype=np.float32)
        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            out = self.model(**enc)
            emb = self._mean_pool(out.last_hidden_state, enc.attention_mask)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            chunks.append(emb.cpu().numpy())
        return np.vstack(chunks)

class ICDMapper:
    def __init__(self, index_dir, encoder_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 tau_cos=0.40, tau_final=0.60, w_cos=0.6, w_fuz=0.4, faiss_rows=20):
        self.dir = index_dir
        self.faiss_rows = faiss_rows
        self.tau_cos = tau_cos
        self.tau_final = tau_final
        self.w_cos = w_cos
        self.w_fuz = w_fuz

        idx_path = os.path.join(self.dir, "icd.faiss")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"FAISS index not found: {idx_path}")
        self.index = faiss.read_index(idx_path)

        rows_path = os.path.join(self.dir, "rows.json")
        if not os.path.exists(rows_path):
            raise FileNotFoundError(f"rows.json not found: {rows_path}")
        with open(rows_path, "r") as f:
            self.rows = json.load(f)
        
        meta = {}
        meta_path = os.path.join(self.dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path,"r") as f:
                meta = json.load(f)
        self.metric = meta.get("metric","ip").lower()
        
        print(f"FAISS index loaded: {self.index.ntotal} rows (metric={self.metric})")
        print(f"Using encoder model: {encoder_model}")
        self.encoder = HFMeanEncoder(encoder_model)

    def map_terms(self, terms_list):
        """Map a list of diagnosis terms to ICD codes with scores."""
        if not terms_list:
            return []
            
        embs = self.encoder.encode(terms_list)
        results = []
        
        for t_idx, term in enumerate(terms_list):
            if not term or len(term) < 3: 
                results.append([])
                continue
                
            norm_t = norm_text(term)
            if len(norm_t) < 3:
                results.append([])
                continue

            D, I = self.index.search(embs[t_idx:t_idx+1], self.faiss_rows)
            D, I = D[0], I[0]

            matches = []
            for j, row_idx in enumerate(I):
                if row_idx < 0: continue
                entry = self.rows[row_idx]
                cand_text = entry.get("text","")
                cand_code = entry.get("code","")
                if not cand_text or not cand_code: continue

                if self.metric == "ip":
                    cos = float(D[j])
                else:
                    cos = 1.0 - float(D[j]) / 2.0

                if cos < self.tau_cos: continue

                fuzzy = token_set_ratio(norm_t, norm_text(cand_text)) / 100.0
                score = self.w_cos * cos + self.w_fuz * fuzzy
                
                if score >= self.tau_final:
                    matches.append((cand_code, score, cand_text))
            
            # Sort by score descending
            matches.sort(key=lambda x: x[1], reverse=True)
            results.append(matches)
            
        return results

def main():
    # Path to the index directory
    index_dir = "./gen/TextGen/icd_index_v9"
    
    # Check if index exists or provide instructions
    if not os.path.exists(index_dir):
        print(f"Error: Index directory not found at {index_dir}")
        print("Please provide the correct path to the icd_index_v9 directory.")
        return

    # Initialize the mapper
    mapper = ICDMapper(
        index_dir=index_dir,
        encoder_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        tau_cos=0.40, 
        tau_final=0.60,
        w_cos=0.6, 
        w_fuz=0.4,
        faiss_rows=50
    )
    
    print("\nICD Code Mapper Test")
    print("===================")
    print("Enter diagnosis text (or 'quit' to exit):")
    
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
            
        if not user_input.strip():
            continue
            
        # Process the input
        results = mapper.map_terms([user_input])
        
        if results and results[0]:
            print(f"\nFound {len(results[0])} potential ICD codes:")
            for code, score, text in results[0]:
                print(f"{code} - {text} (score: {score:.3f})")
        else:
            print("No matching ICD codes found.")

if __name__ == "__main__":
    main()