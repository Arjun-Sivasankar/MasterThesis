import sys
import gc
import os
import copy
import json
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import torch.nn.functional as F
import time
from datetime import timedelta
from typing import Dict, List, Tuple, Set, Optional, Union


class MIMICPreprocessor:
    """
    Class for preprocessing MIMIC-IV dataset into a structured format for
    healthcare predictive modeling tasks.
    """
    
    def __init__(self, base_path: str, use_icd10: bool = False, gamenet_path: str = None, verbose: bool = True):
        """
        Initialize MIMIC preprocessor with file paths and configuration.
        
        Args:
            base_path: Root directory containing MIMIC-IV data files
            use_icd10: If True, use ICD-10 codes, otherwise use ICD-9 codes
            gamenet_path: Path to GAMENet repo containing drug code mappings
            verbose: Whether to print detailed progress messages
        """
        self.data_path = base_path
        self.icd10 = use_icd10
        self.gamenet_path = gamenet_path or "/data/horse/ws/arsi805e-finetune/Thesis/GAMENet"
        self.verbose = verbose
        self.timing_stats = {}  # Store timing statistics for each step
        
        # Set up file paths
        self.med_file = os.path.join(self.data_path, "prescriptions.csv.gz")
        self.procedure_file = os.path.join(self.data_path, "procedures_icd.csv.gz")
        self.diag_file = os.path.join(self.data_path, "diagnoses_icd.csv.gz")
        self.admission_file = os.path.join(self.data_path, "admissions.csv.gz")
        self.lab_test_file = os.path.join(self.data_path, "labevents.csv.gz")
        self.patient_file = os.path.join(self.data_path, "patients.csv.gz")
        
        # GAMENet mapping files
        self.ndc2atc_file = f'{self.gamenet_path}/data/ndc2atc_level4.csv' 
        self.cid_atc = f'{self.gamenet_path}/data/drug-atc.csv'
        self.ndc2rxnorm_file = f'{self.gamenet_path}/data/ndc2rxnorm_mapping.txt'
        
        if self.verbose:
            print(f"MIMIC-IV data path: {self.data_path}")
            print(f"Using {'ICD-10' if self.icd10 else 'ICD-9'} codes")

    def _time_operation(self, operation_name):
        """Context manager to time operations and store results"""
        class TimerContextManager:
            def __init__(self, processor, name):
                self.processor = processor
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                if self.processor.verbose:
                    print(f"Starting {self.name}...")
                return self
            
            def __exit__(self, *args):
                duration = time.time() - self.start_time
                self.processor.timing_stats[self.name] = duration
                if self.processor.verbose:
                    print(f"Completed {self.name} in {duration:.2f} seconds")
                
        return TimerContextManager(self, operation_name)

    def process_medications(self) -> pd.DataFrame:
        """
        Process medication data from MIMIC-IV prescriptions file.
        
        Returns:
            DataFrame with processed medication data
        """
        with self._time_operation("Processing medications"):
            med_pd = pd.read_csv(self.med_file, dtype={'ndc': 'category'})
            
            # Remove unnecessary columns
            med_pd.drop(columns=['pharmacy_id', 'poe_id', 'poe_seq',
                'order_provider_id', 'stoptime', 'drug_type', 'drug',
                'formulary_drug_cd', 'gsn', 'prod_strength', 'form_rx',
                'dose_val_rx', 'dose_unit_rx', 'form_val_disp', 'form_unit_disp',
                'doses_per_24_hrs', 'route'], axis=1, inplace=True)
            
            # Clean data
            med_pd.drop(index=med_pd[med_pd['ndc'] == '0'].index, axis=0, inplace=True)
            med_pd.fillna(method='pad', inplace=True)
            med_pd.dropna(inplace=True)
            med_pd.drop_duplicates(inplace=True)
            
            # Convert timestamp and sort
            med_pd['starttime'] = pd.to_datetime(med_pd['starttime'], format='%Y-%m-%d %H:%M:%S')    
            med_pd.sort_values(by=['subject_id', 'hadm_id', 'starttime'], inplace=True)
            med_pd = med_pd.reset_index(drop=True)
            
            # Filter for first 24 hours
            med_pd = self._filter_first24hour_med(med_pd)
            med_pd = med_pd.drop_duplicates()
            
            if self.verbose:
                print(f"Medication data shape: {med_pd.shape}")
            
            return med_pd.reset_index(drop=True)
    
    def _filter_first24hour_med(self, med_pd: pd.DataFrame) -> pd.DataFrame:
        """Helper method to filter medications to first 24 hours of admission"""
        with self._time_operation("Filtering medications to first 24 hours"):
            med_pd_new = med_pd.drop(columns=['ndc'])
            med_pd_new = med_pd_new.groupby(by=['subject_id','hadm_id']).head(1).reset_index(drop=True)
            med_pd_new = pd.merge(med_pd_new, med_pd, on=['subject_id','hadm_id','starttime'])
            med_pd_new = med_pd_new.drop(columns=['starttime'])
            return med_pd_new

    def process_procedures(self) -> pd.DataFrame:
        """
        Process procedure data from MIMIC-IV procedures file.
        
        Returns:
            DataFrame with processed procedure data
        """
        with self._time_operation("Processing procedures"):
            pro_pd = pd.read_csv(self.procedure_file, dtype={'icd_code': 'category'})
            
            # Filter by ICD version
            if self.icd10:
                pro_pd = pro_pd[pro_pd['icd_version'] == 10]
            else:
                pro_pd = pro_pd[pro_pd['icd_version'] == 9]
            
            if self.verbose:
                print(f"Using procedure codes from ICD version: {pro_pd['icd_version'].unique()}")
                print(f"First 5 procedure codes: {pro_pd['icd_code'].head().tolist()}")
            
            # Clean and format data
            pro_pd.drop(columns=['chartdate', 'icd_version'], inplace=True)
            pro_pd.drop_duplicates(inplace=True)
            pro_pd.sort_values(by=['subject_id', 'hadm_id', 'seq_num'], inplace=True)
            pro_pd.drop(columns=['seq_num'], inplace=True)
            pro_pd["icd_code"] = "PRO_" + pro_pd["icd_code"].astype(str)
            pro_pd.drop_duplicates(inplace=True)
            
            if self.verbose:
                print(f"Procedure data shape: {pro_pd.shape}")
            
            return pro_pd.reset_index(drop=True)

    def process_diagnoses(self) -> pd.DataFrame:
        """
        Process diagnosis data from MIMIC-IV diagnoses file.
        
        Returns:
            DataFrame with processed diagnosis data
        """
        with self._time_operation("Processing diagnoses"):
            diag_pd = pd.read_csv(self.diag_file)
            diag_pd.dropna(inplace=True)
            
            # Filter by ICD version
            if self.icd10:
                diag_pd = diag_pd[diag_pd['icd_version'] == 10]
            else:
                diag_pd = diag_pd[diag_pd['icd_version'] == 9]
            
            if self.verbose:
                print(f"Using diagnosis codes from ICD version: {diag_pd['icd_version'].unique()}")
                print(f"First 5 diagnosis codes: {diag_pd['icd_code'].head().tolist()}")
            
            # Clean and format data
            diag_pd.drop(columns=['seq_num', 'icd_version'], inplace=True)
            diag_pd.drop_duplicates(inplace=True)
            diag_pd.sort_values(by=['subject_id', 'hadm_id'], inplace=True)
            
            if self.verbose:
                print(f"Diagnosis data shape: {diag_pd.shape}")
            
            return diag_pd.reset_index(drop=True)

    def process_admissions(self) -> pd.DataFrame:
        """
        Process admission data from MIMIC-IV admissions file.
        
        Returns:
            DataFrame with processed admission data
        """
        with self._time_operation("Processing admissions"):
            ad_pd = pd.read_csv(self.admission_file)
            patient_pd = pd.read_csv(self.patient_file)
            
            # Remove unnecessary columns
            ad_pd.drop(columns=['admission_type', 'admit_provider_id', 'admission_location',
                              'discharge_location', 'insurance', 'language', 'marital_status', 
                              'race', 'edregtime', 'edouttime', 'hospital_expire_flag'], 
                      axis=1, inplace=True)
            patient_pd.drop(columns=['anchor_year', 'anchor_year_group'], axis=1, inplace=True)
            
            # Parse timestamps
            ad_pd["admittime"] = pd.to_datetime(ad_pd['admittime'], format='%Y-%m-%d %H:%M:%S')
            ad_pd["dischtime"] = pd.to_datetime(ad_pd['dischtime'], format='%Y-%m-%d %H:%M:%S')
            
            # Merge with patient data
            ad_pd = ad_pd.merge(patient_pd, on=['subject_id'], how='inner')
            
            # Create derived features
            ad_pd = self._create_admission_features(ad_pd)
            
            if self.verbose:
                print(f"Admission data shape: {ad_pd.shape}")
            
            return ad_pd.reset_index(drop=True)
    
    def _create_admission_features(self, ad_pd: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for admission data"""
        with self._time_operation("Creating admission features"):
            # Age binning
            ad_pd["age"] = ad_pd['anchor_age']
            age = ad_pd["age"]
            bins = np.linspace(age.min(), age.max(), 20 + 1)
            ad_pd['age'] = pd.cut(age, bins=bins, labels=False, include_lowest=True)
            ad_pd['age'] = "age_" + ad_pd["age"].astype("str")
            
            # Death indicator
            ad_pd["death"] = ad_pd["dod"].notna()
            
            # Length of stay
            ad_pd["stay_days"] = (ad_pd["dischtime"] - ad_pd["admittime"]).dt.days
            
            # Readmission indicator
            ad_pd['admittime'] = ad_pd['admittime'].astype(str)
            ad_pd.sort_values(by=['subject_id', 'hadm_id', 'admittime'], inplace=True)
            ad_pd['next_visit'] = ad_pd.groupby('subject_id')['hadm_id'].shift(-1)
            ad_pd['readmission'] = ad_pd['next_visit'].notnull().astype(int)
            ad_pd.drop('next_visit', axis=1, inplace=True)
            
            # Drop unnecessary columns
            ad_pd.drop(columns=['dischtime', 'dod', 'deathtime', 'anchor_age'], axis=1, inplace=True)
            ad_pd.drop_duplicates(inplace=True)
            
            return ad_pd

    def process_lab_tests(self, n_bins: int = 5) -> pd.DataFrame:
        """
        Process laboratory test data from MIMIC-IV labevents file.
        
        Args:
            n_bins: Number of bins to discretize continuous lab values
            
        Returns:
            DataFrame with processed lab test data
        """
        with self._time_operation("Processing lab tests"):
            lab_pd = pd.read_csv(self.lab_test_file)
            
            # Only keep first value for each test
            lab_pd = lab_pd.groupby(by=['subject_id', 'itemid']).head(1).reset_index(drop=True)
            lab_pd = lab_pd[lab_pd["valuenum"].notna()]
            lab_pd = lab_pd[lab_pd["hadm_id"].notna()]
            
            # Bin lab values
            lab_pd = self._bin_lab_values(lab_pd, n_bins)
            
            # Clean up columns
            lab_pd.drop(columns=['labevent_id', 'specimen_id', 'itemid',
                  'order_provider_id', 'charttime', 'storetime', 'value', 'valuenum',
                  'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag', 'priority',
                  'comments', 'value_bin'], axis=1, inplace=True)
            lab_pd.drop_duplicates(inplace=True)
            
            if self.verbose:
                print(f"Lab test data shape: {lab_pd.shape}")
            
            return lab_pd.reset_index(drop=True)
    
    def _bin_lab_values(self, lab_pd: pd.DataFrame, n_bins: int) -> pd.DataFrame:
        """Helper to bin lab values into discrete categories"""
        with self._time_operation("Binning lab values"):
            def contains_text(group):
                for item in group:
                    try:
                        float(item)
                    except ValueError:
                        return True
                return False
            
            unique_items = lab_pd['itemid'].unique()
            if self.verbose:
                print(f"Binning {len(unique_items)} unique lab tests...")
            
            for i, itemid in enumerate(unique_items):
                if self.verbose and i % 1000 == 0 and i > 0:
                    print(f"Processed {i}/{len(unique_items)} lab tests...")
                    
                group = lab_pd[lab_pd['itemid'] == itemid]['value']
                # If the lab test contains text value then directly copy the value
                if contains_text(group):
                    lab_pd.loc[lab_pd['itemid'] == itemid, 'value_bin'] = group
                else:
                    # Convert to numeric
                    values_numeric = pd.to_numeric(group, errors='coerce')
                    if len(values_numeric.dropna()) < n_bins:
                        lab_pd.loc[lab_pd['itemid'] == itemid, 'value_bin'] = group
                    else:
                        # Quantile-based binning
                        lab_pd.loc[lab_pd['itemid'] == itemid, 'value_bin'] = pd.qcut(
                            values_numeric, q=n_bins, labels=False, duplicates='drop')
            
            lab_pd["itemid"] = lab_pd["itemid"].astype(str)
            lab_pd["value_bin"] = lab_pd["value_bin"].astype(str)
            lab_pd["lab_test"] = lab_pd[["itemid", "value_bin"]].apply("-".join, axis=1)
            
            return lab_pd

    def map_ndc_to_atc4(self, med_pd: pd.DataFrame) -> pd.DataFrame:
        """
        Map NDC (National Drug Code) to ATC4 (Anatomical Therapeutic Chemical) codes.
        
        Args:
            med_pd: DataFrame with medication data containing NDC codes
            
        Returns:
            DataFrame with NDC codes converted to ATC4 codes
        """
        with self._time_operation("Mapping NDC to ATC4 codes"):
            # Load NDC to RxNorm mapping
            with open(self.ndc2rxnorm_file, 'r') as f:
                ndc2rxnorm = eval(f.read())
            med_pd['RXCUI'] = med_pd['ndc'].map(ndc2rxnorm)
            med_pd.dropna(inplace=True)

            # Load RxNorm to ATC mapping
            rxnorm2atc = pd.read_csv(self.ndc2atc_file)
            
            # Inspect columns before dropping - this will help identify what's available
            if self.verbose:
                print(f"Columns in rxnorm2atc: {rxnorm2atc.columns.tolist()}")
            
            # Check if columns exist before dropping them
            cols_to_drop = []
            for col in ['YEAR', 'MONTH', 'ndc']:
                if col in rxnorm2atc.columns:
                    cols_to_drop.append(col)
            
            # Only drop columns that actually exist
            if cols_to_drop:
                rxnorm2atc = rxnorm2atc.drop(columns=cols_to_drop)
                
            rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
            
            # Remove empty RxCUI entries
            med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)
            
            # Convert and merge
            med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
            med_pd = med_pd.reset_index(drop=True)
            med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
            
            # Clean up and extract ATC4 code
            med_pd.drop(columns=['ndc', 'RXCUI'], inplace=True)
            med_pd = med_pd.rename(columns={'ATC4':'ndc'})
            med_pd['ndc'] = med_pd['ndc'].map(lambda x: x[:4])
            med_pd = med_pd.drop_duplicates()
            
            if self.verbose:
                print(f"Medication data after mapping shape: {med_pd.shape}")
                print(f"Unique ATC4 codes: {len(med_pd['ndc'].unique())}")

            return med_pd.reset_index(drop=True)
        

    def filter_most_frequent(self, 
                          diag_pd: pd.DataFrame, 
                          pro_pd: pd.DataFrame, 
                          lab_pd: pd.DataFrame,
                          diag_threshold: int = 2000,
                          pro_threshold: int = 800,
                          lab_threshold: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Filter to keep only the most frequent diagnoses, procedures, and lab tests.
        
        Args:
            diag_pd: Diagnosis DataFrame
            pro_pd: Procedure DataFrame
            lab_pd: Lab test DataFrame
            diag_threshold: Number of most frequent diagnoses to keep
            pro_threshold: Number of most frequent procedures to keep
            lab_threshold: Number of most frequent lab tests to keep
            
        Returns:
            Tuple of filtered DataFrames (diagnoses, procedures, lab tests)
        """
        with self._time_operation("Filtering to most frequent codes"):
            # Capture initial counts
            if self.verbose:
                initial_diag_count = len(diag_pd['icd_code'].unique())
                initial_pro_count = len(pro_pd['icd_code'].unique())
                initial_lab_count = len(lab_pd['lab_test'].unique())
                
            # Filter diagnoses
            diag_count = diag_pd.groupby(by=['icd_code']).size().reset_index() \
                        .rename(columns={0: 'count'}) \
                        .sort_values(by=['count'], ascending=False).reset_index(drop=True)
            diag_pd = diag_pd[diag_pd['icd_code'].isin(diag_count.loc[:diag_threshold, 'icd_code'])]
            
            # Filter procedures
            pro_count = pro_pd.groupby(by=['icd_code']).size().reset_index() \
                      .rename(columns={0: 'count'}) \
                      .sort_values(by=['count'], ascending=False).reset_index(drop=True)
            pro_pd = pro_pd[pro_pd['icd_code'].isin(pro_count.loc[:pro_threshold, 'icd_code'])]
            
            # Filter lab tests
            lab_count = lab_pd.groupby(by=['lab_test']).size().reset_index() \
                      .rename(columns={0: 'count'}) \
                      .sort_values(by=['count'], ascending=False).reset_index(drop=True)
            lab_pd = lab_pd[lab_pd['lab_test'].isin(lab_count.loc[:lab_threshold, 'lab_test'])]
            
            if self.verbose:
                print(f"Diagnosis codes reduced: {initial_diag_count} -> {len(diag_pd['icd_code'].unique())}")
                print(f"Procedure codes reduced: {initial_pro_count} -> {len(pro_pd['icd_code'].unique())}")
                print(f"Lab test codes reduced: {initial_lab_count} -> {len(lab_pd['lab_test'].unique())}")
            
            return (diag_pd.reset_index(drop=True), 
                   pro_pd.reset_index(drop=True), 
                   lab_pd.reset_index(drop=True))

    def process_dataset(self) -> pd.DataFrame:
        """
        Process the entire MIMIC-IV dataset, combining all modalities.
        
        Returns:
            Processed DataFrame ready for machine learning tasks
        """
        start_time = time.time()
        local_time = time.ctime(start_time)
        print(f"Starting processing at: {local_time}\n")
        
        # Process individual tables
        med_pd = self.process_medications()
        med_pd = self.map_ndc_to_atc4(med_pd)
        
        diag_pd = self.process_diagnoses()
        pro_pd = self.process_procedures()
        ad_pd = self.process_admissions()
        lab_pd = self.process_lab_tests()
        
        # Filter to most frequent codes
        diag_pd, pro_pd, lab_pd = self.filter_most_frequent(diag_pd, pro_pd, lab_pd)
        
        # Check for sample codes
        if self.verbose:
            print(f"Sample diagnosis codes: {diag_pd['icd_code'].unique()[:5]}")
            print(f"Sample procedure codes: {pro_pd['icd_code'].unique()[:5]}")
        
        # Filter to common patients/admissions across all tables
        data = self._combine_data(med_pd, diag_pd, pro_pd, lab_pd, ad_pd)
        
        # Calculate processing time
        end_time = time.time()
        total_time = end_time - start_time
        self.timing_stats["Total processing time"] = total_time
        
        print(f"\nProcessing completed in {total_time:.2f} seconds ({timedelta(seconds=int(total_time))})")
        print("\nTiming breakdown:")
        for step, duration in self.timing_stats.items():
            if step != "Total processing time":
                pct = (duration / total_time) * 100
                print(f"  {step}: {duration:.2f} seconds ({pct:.1f}%)")
        
        return data
    
    def _combine_data(self, 
                    med_pd: pd.DataFrame, 
                    diag_pd: pd.DataFrame, 
                    pro_pd: pd.DataFrame, 
                    lab_pd: pd.DataFrame, 
                    ad_pd: pd.DataFrame) -> pd.DataFrame:
        """Combine all data sources into a single DataFrame"""
        with self._time_operation("Combining data from all sources"):
            # Capture initial counts
            if self.verbose:
                initial_patients = set(med_pd['subject_id']).union(
                    set(diag_pd['subject_id'])).union(
                    set(pro_pd['subject_id'])).union(
                    set(lab_pd['subject_id'])).union(
                    set(ad_pd['subject_id']))
                initial_admissions = set(zip(med_pd['subject_id'], med_pd['hadm_id'])).union(
                    set(zip(diag_pd['subject_id'], diag_pd['hadm_id']))).union(
                    set(zip(pro_pd['subject_id'], pro_pd['hadm_id']))).union(
                    set(zip(lab_pd['subject_id'], lab_pd['hadm_id']))).union(
                    set(zip(ad_pd['subject_id'], ad_pd['hadm_id'])))
                print(f"Initial patients: {len(initial_patients)}")
                print(f"Initial admissions: {len(initial_admissions)}")
            
            # Get common keys across all tables
            med_pd_key = med_pd[['subject_id', 'hadm_id']].drop_duplicates()
            diag_pd_key = diag_pd[['subject_id', 'hadm_id']].drop_duplicates()
            pro_pd_key = pro_pd[['subject_id', 'hadm_id']].drop_duplicates()
            lab_pd_key = lab_pd[['subject_id', 'hadm_id']].drop_duplicates()
            ad_pd_key = ad_pd[['subject_id', 'hadm_id']].drop_duplicates()
            
            # Filter to common keys
            combined_key = med_pd_key.merge(diag_pd_key, on=['subject_id', 'hadm_id'], how='inner')
            combined_key = combined_key.merge(pro_pd_key, on=['subject_id', 'hadm_id'], how='inner')
            combined_key = combined_key.merge(lab_pd_key, on=['subject_id', 'hadm_id'], how='inner')
            combined_key = combined_key.merge(ad_pd_key, on=['subject_id', 'hadm_id'], how='inner')
            
            if self.verbose:
                print(f"Common patients across all tables: {len(combined_key['subject_id'].unique())}")
                print(f"Common admissions across all tables: {len(combined_key)}")
            
            # Apply filter to all tables
            diag_pd = diag_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
            med_pd = med_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
            pro_pd = pro_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
            lab_pd = lab_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
            ad_pd = ad_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
            
            # Group and flatten data
            with self._time_operation("Grouping and flattening data"):
                diag_pd = diag_pd.groupby(by=['subject_id', 'hadm_id'])['icd_code'].unique().reset_index()  
                med_pd = med_pd.groupby(by=['subject_id', 'hadm_id'])['ndc'].unique().reset_index()
                pro_pd = pro_pd.groupby(by=['subject_id', 'hadm_id'])['icd_code'].unique().reset_index() \
                        .rename(columns={'icd_code': 'pro_code'})  
                lab_pd = lab_pd.groupby(by=['subject_id', 'hadm_id'])['lab_test'].unique().reset_index()
                
                # Convert to lists
                med_pd['ndc'] = med_pd['ndc'].map(lambda x: list(x))
                pro_pd['pro_code'] = pro_pd['pro_code'].map(lambda x: list(x))
                lab_pd['lab_test'] = lab_pd['lab_test'].map(lambda x: list(x))
            
            # Merge all data
            with self._time_operation("Merging all data"):
                data = diag_pd.merge(med_pd, on=['subject_id', 'hadm_id'], how='inner')
                data = data.merge(pro_pd, on=['subject_id', 'hadm_id'], how='inner')
                data = data.merge(lab_pd, on=['subject_id', 'hadm_id'], how='inner')
                data = data.merge(ad_pd, on=['subject_id', 'hadm_id'], how='inner')
            
            # Sort by patient and admission time
            data = data.sort_values(by=['subject_id', 'admittime'])
            
            # Create readmission features
            data = self._create_readmission_features(data)
            
            if self.verbose:
                print(f"Final dataset shape: {data.shape}")
                print(f"Final number of patients: {len(data['subject_id'].unique())}")
            
            return data.reset_index(drop=True)
    
    def _create_readmission_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create readmission and next diagnosis features"""
        with self._time_operation("Creating readmission features"):
            # Convert admission time to datetime
            data['admittime'] = pd.to_datetime(data['admittime'])
            
            # Create readmission features
            data['READMISSION_1M'] = data.groupby('subject_id')['admittime'].shift(-1) - data['admittime']
            data['READMISSION_3M'] = data['READMISSION_1M'].apply(lambda x: 1 if x <= timedelta(days=90) else 0)
            data['READMISSION_1M'] = data['READMISSION_1M'].apply(lambda x: 1 if x <= timedelta(days=30) else 0)
            
            # Create future diagnosis features
            with self._time_operation("Creating future diagnosis features"):
                data['NEXT_DIAG_6M'] = data.apply(lambda x: data[(data['subject_id'] == x['subject_id']) & 
                                              (data['admittime'] > x['admittime']) & 
                                              (data['admittime'] <= x['admittime'] + timedelta(days=180))]['icd_code'].tolist(), axis=1)
                data['NEXT_DIAG_12M'] = data.apply(lambda x: data[(data['subject_id'] == x['subject_id']) & 
                                               (data['admittime'] > x['admittime']) & 
                                               (data['admittime'] <= x['admittime'] + timedelta(days=365))]['icd_code'].tolist(), axis=1)
            
            # Process next diagnosis lists
            data['NEXT_DIAG_6M'] = data['NEXT_DIAG_6M'].apply(lambda x: x[0] if x else float('nan'))
            data['NEXT_DIAG_12M'] = data['NEXT_DIAG_12M'].apply(lambda x: x[0] if x else float('nan'))
            
            # Drop admission time column as it's no longer needed
            data.drop(columns=['admittime'], axis=1, inplace=True)
            
            return data
        
    def calculate_statistics(self, data: pd.DataFrame) -> None:
        """
        Calculate and print dataset statistics.
        
        Args:
            data: Processed MIMIC-IV dataset
        """
        with self._time_operation("Calculating dataset statistics"):
            print("\n===== Dataset Statistics =====")
            print(f"Number of patients: {data['subject_id'].unique().shape[0]}")
            print(f"Number of clinical events: {len(data)}")
            
            # Extract all codes
            diag = data['icd_code'].values
            med = data['ndc'].values
            pro = data['pro_code'].values
            lab_test = data['lab_test'].values
            
            # Count unique codes
            unique_diag = set([j for i in diag for j in list(i)])
            unique_med = set([j for i in med for j in list(i)])
            unique_pro = set([j for i in pro for j in list(i)])
            unique_lab = set([j for i in lab_test for j in list(i)])
            
            print(f"Number of unique diagnoses: {len(unique_diag)}")
            print(f"Number of unique medications: {len(unique_med)}")
            print(f"Number of unique procedures: {len(unique_pro)}")
            print(f"Number of unique lab tests: {len(unique_lab)}")
            
            # Calculate averages and maximums
            avg_diag = avg_med = avg_pro = avg_lab = 0
            max_diag = max_med = max_pro = max_lab = 0
            cnt = max_visit = avg_visit = 0

            for subject_id in tqdm(data['subject_id'].unique(), desc="Calculating patient statistics"):
                item_data = data[data['subject_id'] == subject_id]
                x, y, z, k = [], [], [], []
                visit_cnt = 0
                for _, row in item_data.iterrows():
                    visit_cnt += 1
                    cnt += 1
                    x.extend(list(row['icd_code']))
                    y.extend(list(row['ndc']))
                    z.extend(list(row['pro_code']))
                    k.extend(list(row['lab_test']))
                x, y, z, k = set(x), set(y), set(z), set(k)
                avg_diag += len(x)
                avg_med += len(y)
                avg_pro += len(z)
                avg_lab += len(k)
                avg_visit += visit_cnt
                max_diag = max(max_diag, len(x))
                max_med = max(max_med, len(y))
                max_pro = max(max_pro, len(z))
                max_lab = max(max_lab, len(k))
                max_visit = max(max_visit, visit_cnt)

            n_patients = len(data['subject_id'].unique())
            print(f"Average diagnoses per patient: {avg_diag/cnt:.2f}")
            print(f"Average medications per patient: {avg_med/cnt:.2f}")
            print(f"Average procedures per patient: {avg_pro/cnt:.2f}")
            print(f"Average lab tests per patient: {avg_lab/cnt:.2f}")
            print(f"Average visits per patient: {avg_visit/n_patients:.2f}")
            print(f"Maximum diagnoses per patient: {max_diag}")
            print(f"Maximum medications per patient: {max_med}")
            print(f"Maximum procedures per patient: {max_pro}")
            print(f"Maximum lab tests per patient: {max_lab}")
            print(f"Maximum visits per patient: {max_visit}")
    
    def save_timing_report(self, output_path: str = None) -> None:
        """
        Save timing statistics to a JSON file.
        
        Args:
            output_path: Path where to save the timing report
        """
        if output_path is None:
            icd_version = "icd10" if self.icd10 else "icd9"
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = f"./timing_report_{icd_version}_{timestamp}.json"
        
        # Add percentage to timing stats
        total_time = self.timing_stats.get("Total processing time", 0)
        timing_with_pct = {}
        
        for step, duration in self.timing_stats.items():
            timing_with_pct[step] = {
                "duration_seconds": duration,
                "duration_formatted": str(timedelta(seconds=int(duration))),
                "percentage": 0 if total_time == 0 else (duration / total_time) * 100
            }
        
        with open(output_path, 'w') as f:
            json.dump(timing_with_pct, f, indent=2)
        
        print(f"Timing report saved to: {output_path}")


def main():
    """Main function to run the MIMIC-IV preprocessing pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process MIMIC-IV dataset')
    parser.add_argument('--data_path', type=str, default='mimic-iv-3.1/hosp',
                        help='Path to MIMIC-IV data directory')
    parser.add_argument('--use_icd10', type=int, default=0,
                        help='Whether to use ICD-10 codes (1) or ICD-9 codes (0)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0: minimal, 1: detailed)')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                        help='Output directory for processed data')
    parser.add_argument('--gamenet_path', type=str, 
                        default='/data/horse/ws/arsi805e-finetune/Thesis/GAMENet',
                        help='Path to GAMENet repo with drug mappings')
    
    args = parser.parse_args()
    
    # Configuration from command line arguments
    data_path = args.data_path
    use_icd10 = bool(args.use_icd10)
    verbose = bool(args.verbose)
    output_dir = args.output_dir
    gamenet_path = args.gamenet_path
    
    print(f"Starting preprocessing with configuration:")
    print(f"  - Data path: {data_path}")
    print(f"  - Using ICD-10: {use_icd10}")
    print(f"  - Verbose: {verbose}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - GAMENet path: {gamenet_path}")
    
    # Initialize preprocessor
    preprocessor = MIMICPreprocessor(
        base_path=data_path,
        use_icd10=use_icd10,
        verbose=verbose
    )
    
    # Process the dataset
    data = preprocessor.process_dataset()
    
    # Print statistics
    preprocessor.calculate_statistics(data)
    
    # Save processed data
    store_dir = "./dataset"
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    
    # Determine output file path based on ICD version
    version_dir = "icd10" if use_icd10 else "icd9"
    output_path = os.path.join(store_dir, version_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, "mimic.pkl")
    
    # Save the dataset
    with open(output_file, "wb") as outfile:
        pickle.dump(data, outfile)
    
    # Save timing report
    preprocessor.save_timing_report(os.path.join(output_path, "timing_report.json"))
    
    print(f"Dataset successfully processed and saved to: {output_file}")
    print(f"Final dataset shape: {data.shape}")
    print(f"Memory usage: {sys.getsizeof(data) / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()