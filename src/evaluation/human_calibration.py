import yaml
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from natsort import natsorted
from pathlib import Path
from datetime import datetime


from dotenv import load_dotenv

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
    return p_true, p_pred, labels

def plot_confusion(p_true, p_pred, labels, normalize=None, title="Confusion Matrix"):
    """
    normalize: None | "true" (row-normalized) | "pred" | "all"
    """
    cm = confusion_matrix(p_true, p_pred, labels=labels, normalize=normalize)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")  # 默认 colormap
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Human (True)",
        xlabel="LLM (Pred)",
        title=title + ("" if normalize is None else f" (normalize={normalize})"),
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # 在格子里标注数值
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    return fig, ax

    from sklearn.metrics import precision_recall_fscore_support

def plot_per_class_prf(p_true, p_pred, labels, title="Per-class Precision/Recall/F1"):
    precision, recall, f1, support = precision_recall_fscore_support(
        p_true, p_pred, labels=labels, zero_division=0
    )

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x,         recall,    width, label="Recall")
    ax.bar(x + width, f1,        width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()

    # 标注 support（每类样本数）
    for i, s in enumerate(support):
        ax.text(i, 1.02, f"n={int(s)}", ha="center", va="bottom")

    fig.tight_layout()
    return fig, ax

from sklearn.metrics import accuracy_score

def plot_accuracy_slices(df, title="Accuracy by Slice"):
    # df 需要列：Correct（0/1）、Human Certainty（0-1）
    thr = float(df["Human Certainty"].median())
    high = df[df["Human Certainty"] >= thr]
    low  = df[df["Human Certainty"] < thr]

    values = [
        float(df["Correct"].mean()),
        float(high["Correct"].mean()) if len(high) else np.nan,
        float(low["Correct"].mean()) if len(low) else np.nan,
    ]
    counts = [len(df), len(high), len(low)]
    labels = [f"All (n={counts[0]})", f"High HC (n={counts[1]})", f"Low HC (n={counts[2]})"]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title + f" (threshold=median={thr:.2f})")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # 顶部标值
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    return fig, ax

# 用法：
# fig, ax = plot_accuracy_slices(df)
# plt.show()



if __name__ == "__main__":
    # llm_annotation()
    p_pred, p_true, labels = basic_statistics()
    #fig, ax = plot_confusion(p_true, p_pred, labels, normalize=None, title="Confusion (counts)")
    #fig, ax = plot_confusion(p_true, p_pred, labels, normalize="true", title="Confusion (row-normalized)")
    fig, ax = plot_per_class_prf(p_true, p_pred, labels)
    #fig, ax = plot_accuracy_slices(df)
    plt.show()
