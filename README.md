<!-- 1. For Data: 
- Download MIMIC IV data - hosp and note
- Download GameNet repo

2. For KG:
- Download UMLS/2025AA/META/MRCONSO.RRF
- Download UMLS/2025AA/META/MRREL.RRF
- Download UMLS/2025AA/META/MRSTY.RRF
- Download LOINC concept map from : https://pages.doit.wisc.edu/JLMARTIN22/mimic-code/-/tree/lab_results/mimic-iv/concepts/concept_map
- Download other files for KG (mention.....)

3. Run analysis for insights and bucket eval csv's:
- run scripts/analysis.sh

4. To Build KG:
- run generate_umls_mappings.py -> creates mappings dir with cui_to_icd9_EXACT.json for KG building
- run buildKG.sh to get KG related files
- run canonicalize_rel.py -> kg_edges_canon.csv
- run graph_from_canon_edges.py -> builds medical_knowledge_graph2.pkl from kg_edges_canon.csv

5. Train Codegen:
- confugure LLM to be finetuned in the slurm script file
- run ./gen/train_codegen.sh (runs gen/train_codegen.py) -> runs_gen/<timestamp_id>

6. Test Codegen:
- run test_codegen.sh (runs gen/test_codegen.py) -> configure the adapter to be used based on training
(Test trained version and base model as well)

7. Semantic Mapper (ICD Mapper)
- run gen/pipeline/build_icd_index.sh -> builds ICD9 mapper => faiss index, embeddings etc.

8. Train Textgen:
- run gen/pipeline/preprocess_baseline.sh 
- run train_textgen_baseline.sh

9. Test Textgen:
- run test_textgen_baseline.sh

10. For KG-augmented RAG based preprocessing:
- run gen/withKG/build_medical_fact_index.sh -> builds h1 and h2 index
- run gen/withKG/preprocess_data_rag.sh -> to build unweighted and weighted datasets in jsonl

11. KG augmented Finetuning - Training:
- run gen/withKG/train_textgen_ragKG.sh

12. KG augmented Finetuning - Testing:
- run gen/withKG/test_textgen_ragKG.sh

13. History aware:
- run history_aware_dataset.sh -> creates dataset
- run analysis_HA.sh -> for insights and bucket eval csv's
- run build_medical_fact_index.sh -> creates index for HA data
- run history_aware_create_jsonl.sh -> creates train,val,test jsonls
- run retrieve_paths_for_history_aware_unweighted.sh -> for different splits across h1 and h2 modes => for unweighted jsonls
- run retrieve_paths_for_history_aware_weighted.sh -> for different splits across h1 and h2 modes => for weighted jsonls
- run history_aware_prompt_builder.sh -> creates TSV files with prompts for baseline (creates prompts folder)
- run build_history_aware_prompts_with_kg.sh -> creates TSV files with prompts for KG-augmented modes

- Training: run train_textgen_history_aware.sh -> for desired mode
- Testing: run test_textgen_history_aware.sh -> for desired mode -->

## Table of Contents

- [Data Preparation](#data-preparation)
- [Knowledge Graph Construction](#knowledge-graph-construction)
- [Analysis](#analysis)
- [Codegen Model Training & Testing](#codegen-model-training--testing)
- [Semantic Mapper (ICD Mapper)](#semantic-mapper-icd-mapper)
- [Textgen Model Training & Testing](#textgen-model-training--testing)
- [KG-Augmented RAG Preprocessing & Training](#kg-augmented-rag-preprocessing--training)
- [History-Aware Pipeline](#history-aware-pipeline)

---

## Data Preparation

1. **Download Required Data:**
   - [MIMIC-IV](https://physionet.org/content/mimiciv) (hosp and note files)
   - [GameNet repository](https://github.com/sjy1203/GAMENet)

2. **Preprocessing:**
   - Use scripts in `dataset/` and provided shell scripts to process and merge data as needed.
   - run `codes.sh`
   - run `run_data_preproc_str.sh`
   - run `run_data_preproc_unstr.sh`
   - run `run_merged_data.sh`
   - run `run_labtestloincmap.sh`
   - run `run_split_data.sh`

---

## Knowledge Graph Construction

1. **Download KG Resources:**
   - UMLS files: `MRCONSO.RRF`, `MRREL.RRF`, `MRSTY.RRF` from `UMLS/2025AA/META/`
   - LOINC concept map: [Download here](https://pages.doit.wisc.edu/JLMARTIN22/mimic-code/-/tree/lab_results/mimic-iv/concepts/concept_map)
   - Additional KG files as required

2. **Build the KG:**
   - Run `generate_umls_mappings.py` to create mappings (`cui_to_icd9_EXACT.json`)
   - Execute `buildKG.sh` to generate KG files
   - Canonicalize relations: `canonicalize_rel.py` → `kg_edges_canon.csv`
   - Build the graph: `graph_from_canon_edges.py` → `medical_knowledge_graph2.pkl`

---

## Analysis

- Run `scripts/analysis.sh` to generate insights and evaluation CSVs.

---

## Codegen Model Training & Testing

1. **Training:**
   - Configure the LLM in the SLURM script
   - Run `./gen/train_codegen.sh` (calls `gen/train_codegen.py`)
   - Outputs to `runs_gen/<timestamp_id>`

2. **Testing:**
   - Run `test_codegen.sh` (calls `gen/test_codegen.py`)
   - Configure model as needed for trained or base eval (uncomment base_llm)

---

## Semantic Mapper (ICD Mapper)

- Build the ICD9 semantic mapper index:
  - Run `gen/pipeline/build_icd_index.sh` to create the FAISS index and embeddings

---

## Textgen Model Training & Testing

1. **Preprocessing:**
   - Run `gen/pipeline/preprocess_baseline.sh`

2. **Training:**
   - Run `train_textgen_baseline.sh`

3. **Testing:**
   - Run `test_textgen_baseline.sh`

---

## KG-Augmented RAG Preprocessing & Training

1. **Preprocessing:**
   - Build fact indices: `gen/withKG/build_medical_fact_index.sh`
   - Preprocess data: `gen/withKG/preprocess_data_rag.sh` (creates unweighted/weighted datasets)

2. **Training:**
   - Run `gen/withKG/train_textgen_ragKG.sh`

3. **Testing:**
   - Run `gen/withKG/test_textgen_ragKG.sh`

---

## History-Aware Pipeline

1. **Dataset Creation:**
   - Run `history_aware_dataset.sh`

2. **Analysis:**
   - Run `analysis_HA.sh` ==> for bucket eval csv's

3. **Indexing & Preprocessing:**
   - Build index: `build_medical_fact_index.sh`
   - Create splits: `history_aware_create_jsonl.sh`
   - Retrieve paths: 
     - Unweighted: `retrieve_paths_for_history_aware_unweighted.sh`
     - Weighted: `retrieve_paths_for_history_aware_weighted.sh`

4. **Prompt Building:**
   - Baseline prompts: `history_aware_prompt_builder.sh` (outputs to `prompts/`)
   - KG-augmented prompts: `build_history_aware_prompts_with_kg.sh`

5. **Training & Testing:**
   - Train: `train_textgen_history_aware.sh`
   - Test: `test_textgen_history_aware.sh`