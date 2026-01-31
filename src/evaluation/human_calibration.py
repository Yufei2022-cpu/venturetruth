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
from matplotlib.gridspec import GridSpec


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

def plot_confusion(p_true, p_pred, labels=None, normalize=None, title="Confusion Matrix",
                   figsize=(7.2, 6.0), dpi=220, save_path=None, show_percent=True,
                   rotate_xticks=12, wrap_underscore=True, wrap_len=14):
    """
    Plot a multi-class confusion matrix (NxN).

    Args:
        p_true, p_pred: list/array of true/pred labels
        labels: list of class names in desired order. If None, inferred from data.
        normalize: None | "true" | "pred" | "all"  (sklearn semantics)
        title: figure title
        figsize, dpi: figure size and resolution
        save_path: if provided, save figure to this path (png/pdf/...)
        show_percent: if True and normalize is not None, display percentage in cells.
                      Also shows counts (recommended).
        rotate_xticks: rotation angle for x tick labels
    Returns:
        fig, ax
    """

    def wrap_label(s: str) -> str:
        s = str(s)
        if not wrap_underscore:
            return s
        if "_" in s and len(s) >= wrap_len:
            parts = s.split("_")
            mid = len(parts) // 2
            return "_".join(parts[:mid]) + "\n" + "_".join(parts[mid:])
        return s

    # --- decide label order ---
    if labels is None:
        labels = sorted(set(p_true) | set(p_pred))
    else:
        labels = list(labels)

    disp_labels = [wrap_label(x) for x in labels]

    # --- compute matrices ---
    cm_counts = confusion_matrix(p_true, p_pred, labels=labels, normalize=None)
    cm = confusion_matrix(p_true, p_pred, labels=labels, normalize=normalize)

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(cm, interpolation="nearest")  # default colormap
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title + ("" if normalize is None else f" (normalize={normalize})"), pad=12)
    ax.set_xlabel("Pred", labelpad=10)
    ax.set_ylabel("True", labelpad=10)

    tick_pos = np.arange(len(labels))
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(disp_labels)
    ax.set_yticklabels(disp_labels)
    plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right", rotation_mode="anchor")

    # optional grid lines to separate cells
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # --- annotate cells ---
    # If normalize is None -> show counts only
    # If normalize is not None -> show counts + percent (from normalized cm)
    if normalize is None:
        fmt_main = "d"
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(
                    j, i, format(int(val), fmt_main),
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="black" if val > thresh else "white"
                )
    else:
        # normalized -> cm in [0,1]
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cnt = int(cm_counts[i, j])
                val = float(cm[i, j])
                color = "black" if val > thresh else "white"

                # counts on first line
                ax.text(j, i - 0.12, f"{cnt:d}",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

                # percentage on second line (optional)
                if show_percent:
                    ax.text(j, i + 0.20, f"{val*100:.1f}%",
                            ha="center", va="center",
                            fontsize=11, color=color)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax

def poster_binary_confusions(
    y_true,
    y_pred,
    categories=None,
    normalize=None,                 # None | "true" | "pred" | "all"
    title_prefix="One-vs-Rest Confusion",
    pos_label_fmt="{c}",
    neg_label_fmt="NOT {c}",
    figsize=(5.2, 4.6),
    dpi=240,
    save_dir=None,
    save_formats=("png",),          # e.g. ("png","pdf")
    font_scale=1.0,
):
    """
    For each category c, plot a 2x2 one-vs-rest confusion matrix as a standalone figure.

    - If normalize is None: cell shows count
    - If normalize is set: cell shows both count and normalized value (percent)
    """
    if categories is None:
        categories = sorted(set(y_true) | set(y_pred))
    else:
        categories = list(categories)

    results = {}

    # Poster-ish font sizes
    fs_title = int(15 * font_scale)
    fs_axis  = int(12 * font_scale)
    fs_tick  = int(11 * font_scale)
    fs_cell  = int(13 * font_scale)
    fs_cell2 = int(10 * font_scale)

    for c in categories:
        pos = pos_label_fmt.format(c=c)
        neg = neg_label_fmt.format(c=c)
        labels_bin = [pos, neg]

        # Build binary vectors (clear + readable, no nested list comprehensions)
        y_true_bin = []
        y_pred_bin = []
        for t, p in zip(y_true, y_pred):
            y_true_bin.append(pos if t == c else neg)
            y_pred_bin.append(pos if p == c else neg)

        # Compute counts matrix (always useful for annotations)
        cm_counts = confusion_matrix(y_true_bin, y_pred_bin, labels=labels_bin, normalize=None)

        # Compute displayed matrix (maybe normalized)
        cm_disp = confusion_matrix(y_true_bin, y_pred_bin, labels=labels_bin, normalize=normalize)

        # Figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(cm_disp, interpolation="nearest")  # keep default cmap for now

        # Title
        norm_txt = "" if normalize is None else f" (normalize={normalize})"
        ax.set_title(f"{title_prefix}: {c}{norm_txt}", fontsize=fs_title, pad=12)

        # Axes labels
        ax.set_xlabel("Predicted (LLM)", fontsize=fs_axis, labelpad=10)
        ax.set_ylabel("True (Human)", fontsize=fs_axis, labelpad=10)

        # Ticks
        ticks = np.arange(2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels_bin, fontsize=fs_tick, rotation=18, ha="right")
        ax.set_yticklabels(labels_bin, fontsize=fs_tick)

        # Make squares & add subtle grid
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Annotate each cell
        # If normalized: show "count\nxx.x%"
        # Else: show "count"
        maxv = cm_disp.max() if cm_disp.size else 0.0
        thresh = maxv / 2.0 if maxv else 0.0

        for i in range(2):
            for j in range(2):
                cnt = cm_counts[i, j]
                val = cm_disp[i, j]

                if normalize is None:
                    main = f"{cnt:d}"
                    sub = ""
                else:
                    main = f"{cnt:d}"
                    sub = f"{val*100:.1f}%"

                color = "black" if val > thresh else "white"

                # main line (count)
                ax.text(j, i - 0.08, main, ha="center", va="center",
                        fontsize=fs_cell, fontweight="bold", color=color)
                # sub line (percent)
                if sub:
                    ax.text(j, i + 0.22, sub, ha="center", va="center",
                            fontsize=fs_cell2, color=color)

        # Small footer: support (how many true positives exist in this class)
        # TP+FN = number of true positives in one-vs-rest sense => row0 sum
        # support = int(cm_counts[0, :].sum())
        # total = int(cm_counts.sum())
        # ax.text(0.5, -0.14, f"Support (True {pos}) = {support} / Total = {total}",
                 # transform=ax.transAxes, ha="center", va="top", fontsize=int(10*font_scale))

        fig.tight_layout()

        # Save
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            suffix = "counts" if normalize is None else f"norm_{normalize}"
            safe_c = str(c).replace(" ", "_").replace("/", "_")
            for fmt in save_formats:
                out_path = os.path.join(save_dir, f"confusion_{safe_c}_{suffix}.{fmt}")
                fig.savefig(out_path, bbox_inches="tight")

        results[c] = (fig, ax, cm_counts, cm_disp)

    return results

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

def per_class_metrics(y_true, y_pred, labels=None):
    """
    返回 dataframe:
    class | precision | recall | f1 | support
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    dfm = pd.DataFrame({
        "class": labels,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": sup
    })
    return dfm


def plot_metric_points(
    metrics_df,
    x="recall",
    y="precision",
    title=None,
    annotate=True,
    figsize=(6.2, 4.6),
    dpi=220,
    save_path=None
):
    """
    metrics_df: per_class_metrics 
    x,y: "precision" | "recall" | "f1" （x != y）
    """
    assert x in ["precision", "recall", "f1"]
    assert y in ["precision", "recall", "f1"]
    assert x != y

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def smart_offset(x, y):
        # 靠近上边界就往下偏移；靠近右边界就往左偏移；否则默认右上
        dx = -0.015 if x > 0.85 else 0.015
        dy = -0.015 if y > 0.90 else 0.015
        ha = "right" if dx < 0 else "left"
        va = "top" if dy < 0 else "bottom"
        return dx, dy, ha, va

    # make the points more readable
    for _, row in metrics_df.iterrows():
        cls = row["class"]
        xv = float(row[x])
        yv = float(row[y])

        ax.scatter(xv, yv, s=140, linewidths=1.5, label=str(cls))
        dx, dy, ha, va = smart_offset(xv, yv)

        # annotate the points directly on the plot
        if annotate:
            ax.text(xv + dx, yv + dy, str(cls), fontsize=11, va=va, ha=ha)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(x.capitalize(), fontsize=13)
    ax.set_ylabel(y.capitalize(), fontsize=13)

    if title is None:
        title = f"{y.capitalize()} vs {x.capitalize()} (per-class points)"
    ax.set_title(title, fontsize=14, pad=10)

    # add grid for to be more readable
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    # automatic legent position
    ax.legend(loc="best", frameon=True, fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax

def plot_all_pairs(metrics_df, prefix="per_class_points", out_dir=None):
    pairs = [("recall", "precision"), ("precision", "f1"), ("recall", "f1")]
    figs = []
    for x, y in pairs:
        save_path = None
        if out_dir is not None:
            import os
            os.makedirs(out_dir, exist_ok=True)
            save_path = f"{out_dir}/{prefix}_{y}_vs_{x}.png"

        fig, ax = plot_metric_points(
            metrics_df, x=x, y=y,
            title=f"{y.capitalize()} vs {x.capitalize()}",
            annotate=True,
            save_path=save_path
        )
        figs.append(fig)
    return figs

def plot_ovr_confusions_poster_compact(
    y_true,
    y_pred,
    categories,
    normalize="true",
    title="One-vs-Rest Confusion Matrices",
    dpi=240,
    # 关键：通过画布比例让矩阵更大（右侧留一点给 colorbar）
    figsize=(13.2, 6.2),
    font_scale=1.0,
    short_not=True,
    wrap_long_labels=True,
    # 子图间距（越小越紧凑）
    wspace=0.0,
    hspace=0.30,
    # colorbar 在右下角的相对宽度
    cbar_width_ratio=0.1,
    save_path=None
):
    assert len(categories) == 3, "按 3 类设计"

    fs_title = int(22 * font_scale)
    fs_sub   = int(16 * font_scale)
    fs_axis  = int(14 * font_scale)
    fs_tick  = int(13 * font_scale)

    # 让“格子数字”更清晰，同时不要挤占空间
    fs_cell1 = int(18 * font_scale)  # count
    fs_cell2 = int(12 * font_scale)  # percent

    def maybe_wrap(s: str) -> str:
        s = str(s)
        if not wrap_long_labels:
            return s
        if "_" in s and len(s) > 14:
            parts = s.split("_")
            mid = len(parts) // 2
            return "_".join(parts[:mid]) + "\n" + "_".join(parts[mid:])
        return s

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # GridSpec：2行2列，但第二列可以更窄（右下角放 colorbar）
    # 让第一列宽一些、第二列略窄，整体矩阵会变大
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.0, 1.0 - cbar_width_ratio],
        height_ratios=[1.0, 1.0],
        wspace=wspace,
        hspace=hspace
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    cax_host = fig.add_subplot(gs[1, 1])  # 用来承载 colorbar
    cax_host.axis("off")

    ax_list = [ax1, ax2, ax3]

    mats = []
    for c in categories:
        pos = maybe_wrap(c)
        neg = "NOT" if short_not else maybe_wrap(f"NOT {c}")
        labels_bin = [pos, neg]

        y_true_bin = [pos if t == c else neg for t in y_true]
        y_pred_bin = [pos if p == c else neg for p in y_pred]

        cm_counts = confusion_matrix(y_true_bin, y_pred_bin, labels=labels_bin, normalize=None)
        cm_disp   = confusion_matrix(y_true_bin, y_pred_bin, labels=labels_bin, normalize=normalize)
        mats.append((c, labels_bin, cm_counts, cm_disp))

    if normalize is None:
        vmin, vmax = 0, max(m[3].max() for m in mats)
    else:
        vmin, vmax = 0.0, 1.0

    last_im = None
    for ax, (c, labels_bin, cm_counts, cm_disp) in zip(ax_list, mats):
        im = ax.imshow(cm_disp, interpolation="nearest", vmin=vmin, vmax=vmax)
        last_im = im

        # ax.set_title(maybe_wrap(c), fontsize=fs_sub, pad=10)

        ticks = np.arange(2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # 让 tick label 更省空间
        ax.set_xticklabels(labels_bin, fontsize=fs_tick, rotation=10, ha="right")
        ax.set_yticklabels(labels_bin, fontsize=fs_tick)

        ax.set_aspect("equal")

        # 细网格线
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        # annotate
        thresh = (cm_disp.max() / 2.0) if cm_disp.size else 0.0
        for i in range(2):
            for j in range(2):
                cnt = int(cm_counts[i, j])
                val = float(cm_disp[i, j])
                color = "black" if val > thresh else "white"

                ax.text(j, i - 0.10, f"{cnt:d}",
                        ha="center", va="center",
                        fontsize=fs_cell1, fontweight="bold", color=color)

                if normalize is not None:
                    ax.text(j, i + 0.24, f"{val*100:.1f}%",
                            ha="center", va="center",
                            fontsize=fs_cell2, color=color)

    # 只在左侧显示 y label，底部显示 x label（更干净）
    ax1.set_ylabel("True", fontsize=fs_axis)
    ax3.set_ylabel("True", fontsize=fs_axis)
    ax3.set_xlabel("Pred", fontsize=fs_axis)

    # Colorbar：在右下角格子里再建一个更窄更靠右的轴
    if last_im is not None:
        bbox = cax_host.get_position()
        # 让 colorbar 更窄 + 更靠右（你说的“左右紧凑”）
        bar_ax = fig.add_axes([
            bbox.x0 + 0.40 * bbox.width,   # 往右挪
            bbox.y0 + 0.06 * bbox.height,  # 往上抬一点
            0.20 * bbox.width,             # 更窄
            0.76 * bbox.height             # 更高
        ])
        cb = fig.colorbar(last_im, cax=bar_ax)
        cb.ax.tick_params(labelsize=fs_tick)

    norm_txt = "" if normalize is None else f" (normalize={normalize})"
    fig.suptitle(title + norm_txt, fontsize=fs_title, y=0.98)

    # tight_layout 只负责排子图，colorbar 是 add_axes 手工放的，不受它影响
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig




if __name__ == "__main__":
    #llm_annotation()
    p_true, p_pred, labels = basic_statistics()

    labels = ["SUPPORTED", "INSUFFICIENT_EVIDENCE", "CONTRADICTED"]


    
     #plot_ovr_confusions_poster_compact(p_true, p_pred, labels, normalize="true", save_path="./poster_figs/ovr_compact.png")



    # m = per_class_metrics(p_true, p_pred, labels=labels)

    # # Precision vs Recall
    # plot_metric_points(m, x="recall", y="precision", title="Precision vs Recall (per-class)")

    # # ("recall", "precision"), ("precision", "f1"), ("recall", "f1")
    # plot_all_pairs(m, out_dir="./poster_figs")


    
    # poster_binary_confusions(p_true, p_pred, normalize="true", title_prefix="Confusion (One-vs-Rest)", save_dir="./poster_figs", save_formats=("png","pdf"), font_scale=1.1)
    
    
    
    fig, ax = plot_confusion(p_true, p_pred, labels=labels, normalize="true",
                         title="Confusion Matrix", save_path="./poster_figs/cm3x3_norm_true.png")
    # fig, ax = plot_confusion(p_true, p_pred, labels, normalize=None, title="Confusion (counts)")
    #fig, ax = plot_per_class_prf(p_true, p_pred, labels)
    #fig, ax = plot_accuracy_slices(df)
    plt.show()
