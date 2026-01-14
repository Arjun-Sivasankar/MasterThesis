import json
from pathlib import Path
import argparse

def make_prompt(sample):
    prompt = f"""You are a medical expert. Given the following patient's longitudinal electronic health record (EHR) history and current visit notes, provide the most likely clinical diagnoses for the current admission.

Patient Demographics:
- Gender: {sample.get('gender', 'Unknown')}
- Age: {sample.get('age', 'Unknown')}
- Admission Date: {sample.get('admission_date', 'Unknown')}

Past Medical History:
- Diagnoses: {sample.get('history_diagnoses_descriptions', 'No History')}
- Procedures (last visit): {sample.get('history_last_procedures_descriptions', 'No History')}
- Medications (last visit): {sample.get('history_last_medications_descriptions', 'No History')}
- Lab Tests (last visit): {sample.get('history_last_lab_tests_descriptions', 'No History')}

Visit Notes:
Chief Complaint: {sample.get('note_chief_complaint', '')}
History of Present Illness: {sample.get('note_history_present_illness', '')}
Past Medical History: {sample.get('note_past_medical_history', '')}
Family History: {sample.get('note_family_history', '')}
Physical Exam: {sample.get('note_physical_exam', '')}
Medications on Admission: {sample.get('note_medications_on_admission', '')}

Task: As a clinician, analyze the above information and predict the most probable clinical diagnoses for this admission. List all relevant diagnoses separated by a vertical bar (|).
"""
    return prompt.strip()

def jsonl_to_prompt_tsv(jsonl_path, tsv_path):
    with open(jsonl_path, 'r', encoding='utf-8') as fin, open(tsv_path, 'w', encoding='utf-8') as fout:
        fout.write("prompt\ttarget\n")
        for line in fin:
            sample = json.loads(line)
            prompt = make_prompt(sample)
            target = sample.get("target_icd_descriptions", "")
            # Escape tabs and newlines in prompt/target
            prompt = prompt.replace('\t', ' ').replace('\n', '\\n')
            target = target.replace('\t', ' ').replace('\n', '\\n')
            fout.write(f"{prompt}\t{target}\n")

def main():
    argparser = argparse.ArgumentParser(description="Convert JSONL to Prompt TSV")
    argparser.add_argument("--jsonl_path", type=str, required=True, help="Path to input JSONL file")
    argparser.add_argument("--tsv_path", type=str, required=True, help="Path to output TSV file")

    args = argparser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    tsv_path = Path(args.tsv_path)
    jsonl_to_prompt_tsv(jsonl_path, tsv_path)

if __name__ == "__main__":
    main()