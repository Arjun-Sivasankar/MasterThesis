import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Dict, Set
import json
from tqdm import tqdm
import re
import numpy as np
from datetime import datetime

# ------------------ Code Description & Formatting ------------------
def get_code_descriptions(kg, codes: List[str]) -> Dict[str, str]:
    """
    Get descriptions for given codes from the medical knowledge graph.
    
    Args:
        kg: NetworkX DiGraph - the loaded medical knowledge graph
        codes: List of codes to look up
    
    Returns:
        Dict mapping code -> description
    """
    code_to_desc = {}
    
    for node, data in kg.nodes(data=True):
        if 'code' in data:
            node_code = str(data['code']).strip()
            if node_code in codes:
                description = data.get('name', 'No description available')
                code_to_desc[node_code] = description
    
    return code_to_desc

def format_icd9(code: str) -> str:
    """Format ICD-9 code with proper decimal placement."""
    code = re.sub(r"\s+", "", str(code)).upper().rstrip(".")
    if not code:
        return ""
    if code[0].isdigit():
        if len(code) > 3 and "." not in code:
            return code[:3] + "." + code[3:]
        return code
    if code[0] == "V":
        if len(code) > 3 and "." not in code:
            return code[:3] + "." + code[3:]
        return code
    if code[0] == "E":
        if len(code) > 4 and "." not in code:
            return code[:4] + "." + code[4:]
        return code
    return code

def format_icd9_proc_from_pro(c: str) -> str:
    """Format procedure code from pro_code column."""
    s = str(c or "").strip().upper().replace(" ", "")
    if s.startswith("PRO_"):
        s = s[4:]
    s = re.sub(r"[^0-9]", "", s)
    if not s:
        return ""
    if len(s) >= 3:
        return s[:2] + "." + s[2:]
    return s

def clean_text(x) -> str:
    """Clean text content."""
    if x is None:
        return ""
    try:
        s = " ".join(map(str, x.tolist())) if isinstance(x, (np.ndarray, pd.Series)) else str(x)
    except Exception:
        s = str(x)
    s = s.replace("\x00", " ").replace("\r", " ")
    s = re.sub(r"_+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def to_list(x) -> List[str]:
    """Convert various formats to list of strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        it = x.tolist() if hasattr(x, "tolist") else x
        out = []
        for v in it:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            sv = str(v).strip()
            if sv and sv.lower() not in ("nan", "none"):
                out.append(sv)
        return out
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [str(t).strip() for t in v if str(t).strip()]
        except Exception:
            pass
    return [t for t in re.split(r"[,\s]+", s) if t]

def extract_date_from_charttime(charttime) -> str:
    """Extract date from charttime column (format: '2167-06-21 00:00:00')."""
    if pd.isna(charttime):
        return ""
    
    try:
        # Convert to string and extract date portion
        dt_str = str(charttime).strip()
        # Parse and format as YYYY-MM-DD
        if ' ' in dt_str:
            date_part = dt_str.split(' ')[0]
        else:
            date_part = dt_str
        
        # Validate it's a proper date
        datetime.strptime(date_part, '%Y-%m-%d')
        return date_part
    except Exception:
        return ""

# ------------------ Parsing Functions ------------------
def parse_codes(code_str, code_type: str = "icd") -> Set[str]:
    """
    Parse code string into a set of formatted codes.
    
    Args:
        code_str: Raw code string (can be string, list, array, or other formats)
        code_type: 'icd', 'proc', 'ndc', or 'lab'
    
    Returns:
        Set of formatted code strings
    """
    # Handle None, empty, or NaN values first
    if code_str is None:
        return set()
    
    # Check for various empty conditions
    try:
        # For scalar values
        if isinstance(code_str, (str, float, int)):
            if pd.isna(code_str) or str(code_str).strip() == '':
                return set()
        # For arrays/lists - check if empty
        elif isinstance(code_str, (list, tuple, np.ndarray)):
            if len(code_str) == 0:
                return set()
    except (ValueError, TypeError):
        # If any comparison fails, try to process anyway
        pass
    
    # Use to_list to handle various input formats
    codes_list = to_list(code_str)
    
    if not codes_list:
        return set()
    
    # Format codes based on type
    formatted_codes = set()
    
    if code_type == "icd":
        for c in codes_list:
            formatted = format_icd9(c)
            if formatted:
                formatted_codes.add(formatted)
    
    elif code_type == "proc":
        for c in codes_list:
            formatted = format_icd9_proc_from_pro(c)
            if formatted:
                formatted_codes.add(formatted)
    
    else:  # 'ndc' or 'lab'
        # Keep as-is, just clean up
        for c in codes_list:
            cleaned = str(c).strip()
            if cleaned and cleaned.lower() not in ('nan', 'none', ''):
                formatted_codes.add(cleaned)
    
    return formatted_codes

# ------------------ Individual Text Section Extraction ------------------
def extract_text_section(row: pd.Series, column_name: str) -> str:
    """Extract and clean a single text section from the row."""
    if column_name not in row.index or pd.isna(row[column_name]):
        return ""
    
    content = clean_text(row[column_name])
    if content and len(content) > 5:  # Filter out very short content
        return content
    return ""

# ------------------ Code Formatting Functions ------------------
def codes_to_code_string(codes: Set[str]) -> str:
    """Convert a set of codes to a comma-separated string of codes."""
    if not codes:
        return ""
    return ", ".join(sorted(codes))

def codes_to_description_string(kg: nx.DiGraph, codes: Set[str]) -> str:
    """Convert a set of codes to a comma-separated string of descriptions."""
    if not codes:
        return ""
    
    code_desc_map = get_code_descriptions(kg, list(codes))
    
    descriptions = []
    for code in sorted(codes):  # Sort for consistency
        desc = code_desc_map.get(code)
        if desc and desc != 'No description available':
            descriptions.append(desc)
        else:
            # Fallback: use code itself if no description
            descriptions.append(f"Code {code}")
    
    if not descriptions:
        return ""
    
    return ", ".join(descriptions)

# ------------------ Patient Visit Processing ------------------
def process_patient_visits(patient_df: pd.DataFrame, kg: nx.DiGraph, subject_id_x: str) -> List[Dict]:
    """
    Process all visits for a single patient using rolling window approach.
    
    Args:
        patient_df: DataFrame containing all visits for one patient (sorted by date)
        kg: Knowledge graph for code descriptions
        subject_id_x: Patient identifier
    
    Returns:
        List of sample dictionaries (one per visit starting from visit 1)
    """
    samples = []
    
    # Initialize history buffers
    historical_icd = set()  # Pool diagnoses from 0 to N-1
    
    # Track previous visit data (N-1)
    previous_proc = set()
    previous_meds = set()
    previous_labs = set()
    
    for visit_idx, (_, row) in enumerate(patient_df.iterrows()):
        hadm_id = str(row['hadm_id'])
        
        # Parse current visit codes using the corrected parse_codes function
        current_icd = parse_codes(row.get('icd_code', ''), code_type='icd')
        current_proc = parse_codes(row.get('pro_code', ''), code_type='proc')
        current_ndc = parse_codes(row.get('ndc', ''), code_type='ndc')
        current_labs = parse_codes(row.get('lab_test_loinc', ''), code_type='lab')
        
        # Visit 0: Only build history, don't create sample
        if visit_idx == 0:
            historical_icd.update(current_icd)
            previous_proc = current_proc.copy()
            previous_meds = current_ndc.copy()
            previous_labs = current_labs.copy()
            continue
        
        # Visit N (N > 0): Create training sample
        
        # === Patient Demographics ===
        section_patient_id = subject_id_x
        section_admission_id = hadm_id
        section_visit_number = str(visit_idx)
        section_gender = str(row.get('gender', '')).strip()
        section_age = str(row.get('age', '')).strip()
        
        # Extract date from charttime
        section_admission_date = extract_date_from_charttime(row.get('charttime', ''))
        
        # === Historical Medical History ===
        # Diagnoses: pooled from visits 0 to N-1
        history_diagnoses_codes = codes_to_code_string(historical_icd)
        history_diagnoses_descriptions = codes_to_description_string(kg, historical_icd)
        
        # Structured data from last visit (N-1)
        history_last_procedures_codes = codes_to_code_string(previous_proc)
        history_last_procedures_descriptions = codes_to_description_string(kg, previous_proc)
        
        history_last_medications_codes = codes_to_code_string(previous_meds)
        history_last_medications_descriptions = codes_to_description_string(kg, previous_meds)
        
        history_last_lab_tests_codes = codes_to_code_string(previous_labs)
        history_last_lab_tests_descriptions = codes_to_description_string(kg, previous_labs)
        
        # === Current Visit (N) - Clinical Notes ===
        note_chief_complaint = extract_text_section(row, "Chief Complaint")
        note_history_present_illness = extract_text_section(row, "History of Present Illness")
        note_past_medical_history = extract_text_section(row, "Past Medical History")
        note_family_history = extract_text_section(row, "Family History")
        note_physical_exam = extract_text_section(row, "Physical Exam")
        note_pertinent_results = extract_text_section(row, "Pertinent Results")
        note_brief_hospital_course = extract_text_section(row, "Brief Hospital Course")
        note_medications_on_admission = extract_text_section(row, "Medications on Admission")
        
        # === Target Outputs (Current Visit N) ===
        target_icd_codes = codes_to_code_string(current_icd)
        target_icd_descriptions = codes_to_description_string(kg, current_icd)
        
        # Only create sample if we have meaningful target
        if target_icd_codes and target_icd_descriptions:
            sample = {
                # ===== Patient Identification =====
                "patient_id": section_patient_id,
                "admission_id": section_admission_id,
                "visit_number": section_visit_number,
                
                # ===== Demographics =====
                "gender": section_gender,
                "age": section_age,
                "admission_date": section_admission_date,
                
                # ===== Historical Medical History =====
                # Diagnoses pooled from visits 0 to N-1
                "history_diagnoses_codes": history_diagnoses_codes,
                "history_diagnoses_descriptions": history_diagnoses_descriptions,
                
                # Structured data from last visit (N-1)
                "history_last_procedures_codes": history_last_procedures_codes,
                "history_last_procedures_descriptions": history_last_procedures_descriptions,
                
                "history_last_medications_codes": history_last_medications_codes,
                "history_last_medications_descriptions": history_last_medications_descriptions,
                
                "history_last_lab_tests_codes": history_last_lab_tests_codes,
                "history_last_lab_tests_descriptions": history_last_lab_tests_descriptions,
                
                # ===== Current Visit (N) - Clinical Notes =====
                "note_chief_complaint": note_chief_complaint,
                "note_history_present_illness": note_history_present_illness,
                "note_past_medical_history": note_past_medical_history,
                "note_family_history": note_family_history,
                "note_physical_exam": note_physical_exam,
                "note_pertinent_results": note_pertinent_results,
                "note_brief_hospital_course": note_brief_hospital_course,
                "note_medications_on_admission": note_medications_on_admission,
                
                # ===== Target Outputs (Current Visit N) =====
                "target_icd_codes": target_icd_codes,
                "target_icd_descriptions": target_icd_descriptions
            }
            
            samples.append(sample)
        
        # Update history buffers for next iteration
        # Add current diagnoses to pooled history
        historical_icd.update(current_icd)
        
        # Update last visit structured data (for next iteration's N-1)
        previous_proc = current_proc.copy()
        previous_meds = current_ndc.copy()
        previous_labs = current_labs.copy()
    
    return samples

# ------------------ Main Processing ------------------
def create_modular_jsonl(
    df: pd.DataFrame,
    kg: nx.DiGraph,
    output_path: str,
    split_name: str
):
    """
    Convert DataFrame to modular JSONL format with history-aware context.
    
    Args:
        df: Input DataFrame
        kg: Knowledge graph for code descriptions
        output_path: Path to save JSONL file
        split_name: Name of split (train/val/test) for logging
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split...")
    print(f"{'='*60}")
    
    # Sort by subject and admission date (using charttime)
    df = df.sort_values(['subject_id_x', 'charttime']).reset_index(drop=True)
    
    all_samples = []
    patient_ids = df['subject_id_x'].unique()
    
    print(f"Total patients: {len(patient_ids)}")
    
    for subject_id_x in tqdm(patient_ids, desc=f"Processing {split_name} patients"):
        patient_df = df[df['subject_id_x'] == subject_id_x]
        
        # Process this patient's visits
        patient_samples = process_patient_visits(patient_df, kg, str(subject_id_x))
        all_samples.extend(patient_samples)
    
    # Write to JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n{split_name.upper()} Statistics:")
    print(f"  - Total samples: {len(all_samples)}")
    print(f"  - Patients: {len(patient_ids)}")
    if len(patient_ids) > 0:
        print(f"  - Avg samples/patient: {len(all_samples)/len(patient_ids):.2f}")
    print(f"  - Output: {output_file}")
    
    # Print sample structure
    if all_samples:
        print(f"\nSample JSON structure (first record):")
        sample_str = json.dumps(all_samples[0], indent=2, ensure_ascii=False)
        if len(sample_str) > 1500:
            print(sample_str[:1500] + "\n  ... [truncated]")
        else:
            print(sample_str)

def main():
    # Paths
    base_path = Path("/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis")
    data_dir = base_path / "dataset/history_aware_data2"
    kg_path = base_path / "KG/kg_output4/medical_knowledge_graph2.pkl"
    output_dir = data_dir / "jsonl_output"
    
    # Load knowledge graph
    print("Loading knowledge graph...")
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)
    print(f"Knowledge graph loaded: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        input_file = data_dir / f"{split}_df.pkl"
        output_file = output_dir / f"{split}_modular.jsonl"
        
        if not input_file.exists():
            print(f"WARNING: {input_file} not found, skipping...")
            continue
        
        # Load DataFrame
        print(f"\nLoading {split}_df.pkl...")
        with open(input_file, 'rb') as f:
            df = pickle.load(f)
        
        print(f"Loaded {len(df)} rows")
        
        # Create JSONL
        create_modular_jsonl(df, kg, output_file, split)
    
    print("\n" + "="*60)
    print("✓ All splits processed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()

import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Dict, Set
import json
from tqdm import tqdm
import re
import numpy as np
from datetime import datetime
import argparse

# ------------------ Code Description & Formatting ------------------
def get_code_descriptions_kg(kg, codes: List[str]) -> Dict[str, str]:
    code_to_desc = {}
    for node, data in kg.nodes(data=True):
        if 'code' in data:
            node_code = str(data['code']).strip()
            if node_code in codes:
                description = data.get('name', 'No description available')
                code_to_desc[node_code] = description
    return code_to_desc

def get_code_descriptions_df(df: pd.DataFrame, codes: List[str]) -> Dict[str, str]:
    """Map codes to long_title using a DataFrame with columns 'icd_code' and 'long_title'."""
    code_map = {}
    df = df.drop_duplicates(subset=["icd_code"])
    lookup = dict(zip(df["icd_code"].astype(str).str.upper().str.strip(), df["long_title"]))
    for code in codes:
        code_map[code] = lookup.get(code.upper().strip(), f"Code {code}")
    return code_map

def format_icd9(code: str) -> str:
    code = re.sub(r"\s+", "", str(code)).upper().rstrip(".")
    if not code:
        return ""
    if code[0].isdigit():
        if len(code) > 3 and "." not in code:
            return code[:3] + "." + code[3:]
        return code
    if code[0] == "V":
        if len(code) > 3 and "." not in code:
            return code[:3] + "." + code[3:]
        return code
    if code[0] == "E":
        if len(code) > 4 and "." not in code:
            return code[:4] + "." + code[4:]
        return code
    return code

def format_icd9_proc_from_pro(c: str) -> str:
    s = str(c or "").strip().upper().replace(" ", "")
    if s.startswith("PRO_"):
        s = s[4:]
    s = re.sub(r"[^0-9]", "", s)
    if not s:
        return ""
    if len(s) >= 3:
        return s[:2] + "." + s[2:]
    return s

def clean_text(x) -> str:
    if x is None:
        return ""
    try:
        s = " ".join(map(str, x.tolist())) if isinstance(x, (np.ndarray, pd.Series)) else str(x)
    except Exception:
        s = str(x)
    s = s.replace("\x00", " ").replace("\r", " ")
    s = re.sub(r"_+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def to_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        it = x.tolist() if hasattr(x, "tolist") else x
        out = []
        for v in it:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            sv = str(v).strip()
            if sv and sv.lower() not in ("nan", "none"):
                out.append(sv)
        return out
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [str(t).strip() for t in v if str(t).strip()]
        except Exception:
            pass
    return [t for t in re.split(r"[,\s]+", s) if t]

def extract_date_from_charttime(charttime) -> str:
    if pd.isna(charttime):
        return ""
    try:
        dt_str = str(charttime).strip()
        if ' ' in dt_str:
            date_part = dt_str.split(' ')[0]
        else:
            date_part = dt_str
        datetime.strptime(date_part, '%Y-%m-%d')
        return date_part
    except Exception:
        return ""

def parse_codes(code_str, code_type: str = "icd") -> Set[str]:
    if code_str is None:
        return set()
    try:
        if isinstance(code_str, (str, float, int)):
            if pd.isna(code_str) or str(code_str).strip() == '':
                return set()
        elif isinstance(code_str, (list, tuple, np.ndarray)):
            if len(code_str) == 0:
                return set()
    except (ValueError, TypeError):
        pass
    codes_list = to_list(code_str)
    if not codes_list:
        return set()
    formatted_codes = set()
    if code_type == "icd":
        for c in codes_list:
            formatted = format_icd9(c)
            if formatted:
                formatted_codes.add(formatted)
    elif code_type == "proc":
        for c in codes_list:
            formatted = format_icd9_proc_from_pro(c)
            if formatted:
                formatted_codes.add(formatted)
    else:  # 'ndc' or 'lab'
        for c in codes_list:
            cleaned = str(c).strip()
            if cleaned and cleaned.lower() not in ('nan', 'none', ''):
                formatted_codes.add(cleaned)
    return formatted_codes

def extract_text_section(row: pd.Series, column_name: str) -> str:
    if column_name not in row.index or pd.isna(row[column_name]):
        return ""
    content = clean_text(row[column_name])
    if content and len(content) > 5:
        return content
    return ""

def codes_to_code_string(codes: Set[str]) -> str:
    if not codes:
        return ""
    return ", ".join(sorted(codes))

def codes_to_description_string(
    codes: Set[str],
    desc_df: pd.DataFrame = None,
    kg: nx.DiGraph = None,
    code_type: str = None,
    sep: str = ", "
) -> str:
    if not codes:
        return ""
    codes = sorted(codes)
    if code_type == "icd" and desc_df is not None:
        code_desc_map = get_code_descriptions_df(desc_df, codes)
    elif code_type == "proc" and desc_df is not None:
        code_desc_map = get_code_descriptions_df(desc_df, codes)
    elif kg is not None:
        code_desc_map = get_code_descriptions_kg(kg, codes)
    else:
        code_desc_map = {c: f"Code {c}" for c in codes}
    descriptions = []
    for code in codes:
        desc = code_desc_map.get(code)
        if desc and desc != 'No description available':
            descriptions.append(desc)
        else:
            descriptions.append(f"Code {code}")
    return sep.join(descriptions) if descriptions else ""

def process_patient_visits(
    patient_df: pd.DataFrame,
    kg: nx.DiGraph,
    subject_id_x: str,
    icd9dx_df: pd.DataFrame,
    icd9proc_df: pd.DataFrame
) -> List[Dict]:
    samples = []
    historical_icd = set()
    previous_proc = set()
    previous_meds = set()
    previous_labs = set()
    for visit_idx, (_, row) in enumerate(patient_df.iterrows()):
        hadm_id = str(row['hadm_id'])
        current_icd = parse_codes(row.get('icd_code', ''), code_type='icd')
        current_proc = parse_codes(row.get('pro_code', ''), code_type='proc')
        current_ndc = parse_codes(row.get('ndc', ''), code_type='ndc')
        current_labs = parse_codes(row.get('lab_test_loinc', ''), code_type='lab')

        section_patient_id = subject_id_x
        section_admission_id = hadm_id
        section_visit_number = str(visit_idx)
        section_gender = str(row.get('gender', '')).strip()
        section_age = str(row.get('age', '')).strip()
        section_admission_date = extract_date_from_charttime(row.get('charttime', ''))

        if visit_idx == 0:
            history_diagnoses_codes = "No History"
            history_diagnoses_descriptions = "No History"
            history_last_procedures_codes = "No History"
            history_last_procedures_descriptions = "No History"
            history_last_medications_codes = "No History"
            history_last_medications_descriptions = "No History"
            history_last_lab_tests_codes = "No History"
            history_last_lab_tests_descriptions = "No History"
        else:
            history_diagnoses_codes = codes_to_code_string(historical_icd)
            # Use " | " as separator for descriptions
            history_diagnoses_descriptions = codes_to_description_string(
                historical_icd, desc_df=icd9dx_df, code_type="icd", sep=" | "
            )
            history_last_procedures_codes = codes_to_code_string(previous_proc)
            # Use " | " as separator for descriptions
            history_last_procedures_descriptions = codes_to_description_string(
                previous_proc, desc_df=icd9proc_df, code_type="proc", sep=" | "
            )
            history_last_medications_codes = codes_to_code_string(previous_meds)
            # Use " | " as separator for descriptions
            history_last_medications_descriptions = codes_to_description_string(
                previous_meds, kg=kg, sep=" | "
            )
            history_last_lab_tests_codes = codes_to_code_string(previous_labs)
            # Use " | " as separator for descriptions
            history_last_lab_tests_descriptions = codes_to_description_string(
                previous_labs, kg=kg, sep=" | "
            )

        note_chief_complaint = extract_text_section(row, "Chief Complaint")
        note_history_present_illness = extract_text_section(row, "History of Present Illness")
        note_past_medical_history = extract_text_section(row, "Past Medical History")
        note_family_history = extract_text_section(row, "Family History")
        note_physical_exam = extract_text_section(row, "Physical Exam")
        note_pertinent_results = extract_text_section(row, "Pertinent Results")
        note_brief_hospital_course = extract_text_section(row, "Brief Hospital Course")
        note_medications_on_admission = extract_text_section(row, "Medications on Admission")
        target_icd_codes = codes_to_code_string(current_icd)
        # Use " | " as separator for target_icd_descriptions
        target_icd_descriptions = codes_to_description_string(
            current_icd, desc_df=icd9dx_df, code_type="icd", sep=" | "
        )
        # Only create sample if we have meaningful target
        if target_icd_codes and target_icd_descriptions:
            sample = {
                "patient_id": section_patient_id,
                "admission_id": section_admission_id,
                "visit_number": section_visit_number,
                "gender": section_gender,
                "age": section_age,
                "admission_date": section_admission_date,
                "history_diagnoses_codes": history_diagnoses_codes,
                "history_diagnoses_descriptions": history_diagnoses_descriptions,
                "history_last_procedures_codes": history_last_procedures_codes,
                "history_last_procedures_descriptions": history_last_procedures_descriptions,
                "history_last_medications_codes": history_last_medications_codes,
                "history_last_medications_descriptions": history_last_medications_descriptions,
                "history_last_lab_tests_codes": history_last_lab_tests_codes,
                "history_last_lab_tests_descriptions": history_last_lab_tests_descriptions,
                "note_chief_complaint": note_chief_complaint,
                "note_history_present_illness": note_history_present_illness,
                "note_past_medical_history": note_past_medical_history,
                "note_family_history": note_family_history,
                "note_physical_exam": note_physical_exam,
                "note_pertinent_results": note_pertinent_results,
                "note_brief_hospital_course": note_brief_hospital_course,
                "note_medications_on_admission": note_medications_on_admission,
                "target_icd_codes": target_icd_codes,
                "target_icd_descriptions": target_icd_descriptions
            }
            samples.append(sample)
        historical_icd.update(current_icd)
        previous_proc = current_proc.copy()
        previous_meds = current_ndc.copy()
        previous_labs = current_labs.copy()
    return samples

def create_modular_jsonl(
    df: pd.DataFrame,
    kg: nx.DiGraph,
    output_path: str,
    split_name: str,
    icd9dx_df: pd.DataFrame,
    icd9proc_df: pd.DataFrame
):
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split...")
    print(f"{'='*60}")
    df = df.sort_values(['subject_id_x', 'charttime']).reset_index(drop=True)
    all_samples = []
    patient_ids = df['subject_id_x'].unique()
    print(f"Total patients: {len(patient_ids)}")
    for subject_id_x in tqdm(patient_ids, desc=f"Processing {split_name} patients"):
        patient_df = df[df['subject_id_x'] == subject_id_x]
        patient_samples = process_patient_visits(
            patient_df, kg, str(subject_id_x), icd9dx_df, icd9proc_df
        )
        all_samples.extend(patient_samples)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"\n{split_name.upper()} Statistics:")
    print(f"  - Total samples: {len(all_samples)}")
    print(f"  - Patients: {len(patient_ids)}")
    if len(patient_ids) > 0:
        print(f"  - Avg samples/patient: {len(all_samples)/len(patient_ids):.2f}")
    print(f"  - Output: {output_file}")
    if all_samples:
        print(f"\nSample JSON structure (first record):")
        sample_str = json.dumps(all_samples[0], indent=2, ensure_ascii=False)
        if len(sample_str) > 1500:
            print(sample_str[:1500] + "\n  ... [truncated]")
        else:
            print(sample_str)

def main():
    arg_parser = argparse.ArgumentParser(description="Convert DataFrame to modular JSONL format with history-aware context.")
    arg_parser.add_argument("--data_dir", type=str, default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/history_aware_data2", help="Directory containing the dataset.")
    arg_parser.add_argument("--kg_path", type=str, default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/KG/kg_output4/medical_knowledge_graph2.pkl", help="Path to the knowledge graph pickle file.")
    arg_parser.add_argument("--output_dir", type=str, default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/history_aware_data2/jsonl_output_final", help="Directory to save the output JSONL files.")
    arg_parser.add_argument("--icd9dx_path", type=str, default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9.pkl", help="Path to the ICD9 diagnosis mapping pickle file.")
    arg_parser.add_argument("--icd9proc_path", type=str, default="/data/horse/ws/arsi805e-finetune/Thesis/MasterThesis/dataset/codes/icd9proc.pkl", help="Path to the ICD9 procedure mapping pickle file.")
    args = arg_parser.parse_args()

    data_dir = Path(args.data_dir)
    kg_path = Path(args.kg_path)
    output_dir = Path(args.output_dir)
    icd9dx_path = Path(args.icd9dx_path)
    icd9proc_path = Path(args.icd9proc_path)

    print("Loading knowledge graph...")
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)
    print(f"Knowledge graph loaded: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    print("Loading ICD9 diagnosis mapping...")
    icd9dx_df = pd.read_pickle(icd9dx_path)
    icd9dx_df = icd9dx_df.copy()
    icd9dx_df["icd_code"] = icd9dx_df["icd_code"].astype(str).apply(format_icd9)
    print(f"Loaded {len(icd9dx_df)} ICD9 diagnosis codes")
    print("Loading ICD9 procedure mapping...")
    icd9proc_df = pd.read_pickle(icd9proc_path)
    icd9proc_df = icd9proc_df.copy()
    icd9proc_df["icd_code"] = icd9proc_df["icd_code"].astype(str).apply(format_icd9_proc_from_pro)
    print(f"Loaded {len(icd9proc_df)} ICD9 procedure codes")
    splits = ['train', 'val', 'test']
    for split in splits:
        input_file = data_dir / f"{split}_df.pkl"
        output_file = output_dir / f"{split}_modular.jsonl"
        if not input_file.exists():
            print(f"WARNING: {input_file} not found, skipping...")
            continue
        print(f"\nLoading {split}_df.pkl...")
        with open(input_file, 'rb') as f:
            df = pickle.load(f)
        print(f"Loaded {len(df)} rows")
        create_modular_jsonl(
            df, kg, output_file, split, icd9dx_df, icd9proc_df
        )
    print("\n" + "="*60)
    print("✓ All splits processed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()