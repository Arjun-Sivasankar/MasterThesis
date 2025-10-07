## to generate icd9.pkl (for codes) --> run icd_code.sh (icd_code.py)

## Use struct --> to get structured data (struct_icd9.pkl)
## Use unstruct --> to get notes.pkl and (raw) merged.pkl
## Use code_format_validity --> to get cleaned version of merged.pkl (used for train, val, test). This also generates a report for the codes accepted, rejected and their examples along with the report about the parents.

## To build icd9 index:
sbatch gen/TextGen/build_icd_index.sh

## To run experiments for pipeline:

sbatch --export=ALL,TASK=textgen,STAGE=train gen/pipeline/run_pipeline.sh
sbatch --export=ALL,TASK=textgen,STAGE=test,DISTRIBUTED_TEST=1 gen/pipeline/run_pipeline.sh
sbatch --export=ALL,TASK=codegen,STAGE=train gen/pipeline/run_pipeline.sh
sbatch --export=ALL,TASK=codegen,STAGE=test gen/pipeline/run_pipeline.sh