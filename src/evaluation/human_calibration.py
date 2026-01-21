import yaml
import os
import json
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from natsort import natsorted
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from claim_verification.claim_verifier import ClaimVerifier
from claim_extractor.claim_extractor import ClaimExtractor
from file_content_extraction.ingestion_pipeline import IngestionPipeline
from file_content_extraction.data_loader import DataLoader
from common.result_aggregator import ResultAggregator
from common.schemes import MultiCompanyReport, ResultSummary, ClaimsResponse
from quality_checker.quality_checker import QualityChecker, QualityReport

# In calibration, the root is the evaluation folder
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
MANUAL_ANNOTATION_PATH = Path(__file__).parent / "calibration_resources" / "Manual Annotation.xlsx"
LLM_ANNOTATION_PATH = Path(__file__).parent / "calibration_resources" / "LLM Annotations.xlsx"
VERIFICATIONS_PATH = Path(__file__).parent / "calibration_resources"

load_dotenv()

def load_configuration():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    return config

def llm_annotation():
    verification_files = [f for f in VERIFICATIONS_PATH.glob("verification_*.json")]
    
    if not verification_files:
        raise FileNotFoundError(f"No verification files found in {VERIFICATIONS_PATH}")
    
    sorted_files = natsorted(verification_files, key=lambda x: int(x.stem.split("_")[-1]))
    
    llm_annotations = []

    """
    llm_annotations =
    [
        {
            "claim_id": "Pay33-C1",
            "verdict": "INSUFFICIENT_EVIDENCE",
            "certainty": 0.3,
        }
        ...
    ]
    """
    
    for verification_file in sorted_files:
        with open(verification_file, "r", encoding="utf-8") as f:
            verifications = json.load(f)
        
        for claim in verifications["verification_results"]:
            llm_annotations.append({
                "claim_id": claim["claim_id"],
                "verdict": claim["verdict"],
                "certainty": claim["certainty"]
            })

    df = pd.DataFrame(llm_annotations)
    df.to_excel(LLM_ANNOTATION_PATH, index=False, engine="openpyxl")

    return llm_annotations

def basic_statistics():
    llm_annotations = pd.read_excel(LLM_ANNOTATION_PATH)
    human_annotations = pd.read_excel(MANUAL_ANNOTATION_PATH)

    COL_LLM_VERDICT = "verdict"
    COL_LLM_CERTAINTY = "certainty"
    COL_HUMAN_VERDICT = "Human Verdict"
    COL_HUMAN_CERTAINTY = "Human Certainty"

    human_annotations = human_annotations[[COL_HUMAN_VERDICT, COL_HUMAN_CERTAINTY]].copy()
    llm_annotations = llm_annotations[[COL_LLM_VERDICT, COL_LLM_CERTAINTY]].copy()

    llm_annotations[COL_LLM_VERDICT] = llm_annotations[COL_LLM_VERDICT].str.upper()
    llm_annotations[COL_LLM_CERTAINTY] = llm_annotations[COL_LLM_CERTAINTY].astype(float)
    human_annotations[COL_HUMAN_VERDICT] = human_annotations[COL_HUMAN_VERDICT].str.upper()
    human_annotations[COL_HUMAN_CERTAINTY] = human_annotations[COL_HUMAN_CERTAINTY].astype(float)

    p_true = human_annotations[COL_HUMAN_VERDICT].to_numpy()
    p_pred = llm_annotations[COL_LLM_VERDICT].to_numpy()

    # Calculate accuracy
    accuracy = accuracy_score(p_true, p_pred)

    # Calculate precision, recall, and F1 score
    report = classification_report(p_true, p_pred, output_dict=True)
    # precision = report["macro avg"]["precision"]
    # recall = report["macro avg"]["recall"]
    # f1_score = report["macro avg"]["f1-score"]

    # Calculate confusion matrix
    labels = sorted(set(p_true) | set(p_pred))
    cm = confusion_matrix(p_true, p_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Human:{l}" for l in labels], columns=[f"LLM:{l}" for l in labels])

    print("===== Overall Accuracy =====")
    print(f"{accuracy:.3f}\n")

    print("===== Classification Report (per-class + macro) =====")
    print(report)

    print("===== Confusion Matrix =====")
    print(cm_df)

if __name__ == "__main__":
    # llm_annotation()
    basic_statistics()
