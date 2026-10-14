#!/usr/bin/env python3
"""M6 – Model Evaluation and Combined 5x2 Cross-Validation F-Test Module.

Based on the Thesis Proposal (Chapter I pp. 21, 27–28; Chapter III pp. 62–66, 68–71)
and System Architecture Diagram (`System_Architecture_Diagram_CANVA.pdf.pdf`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from m3_common import find_project_root


def compute_confusion_matrix_4x4(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    """Compute a 4x4 confusion matrix where rows are Actual grades and columns are Predicted grades."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_binary_roc_auc(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    """Compute exact binary ROC-AUC via the Mann-Whitney U / Wilcoxon rank-sum statistic."""
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y_true_bin == 1))
    n_neg = int(np.sum(y_true_bin == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = stats.rankdata(y_score, method="average")
    sum_pos_ranks = float(np.sum(ranks[y_true_bin == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_multiclass_ovo_auc_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 4,
) -> float:
    """Compute Macro One-vs-One (OvO) Multi-Class AUC-ROC (Hand & Till, 2001)."""
    try:
        from sklearn.metrics import roc_auc_score

        return float(
            roc_auc_score(
                y_true, y_prob, multi_class="ovo", average="macro", labels=list(range(num_classes))
            )
        )
    except Exception:
        pair_aucs: list[float] = []
        y_true_arr = np.asarray(y_true, dtype=int)
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                mask_ij = (y_true_arr == i) | (y_true_arr == j)
                if not np.any(y_true_arr == i) or not np.any(y_true_arr == j):
                    continue
                sub_y = y_true_arr[mask_ij]
                sub_p = y_prob_arr[mask_ij]
                denom = np.maximum(sub_p[:, i] + sub_p[:, j], 1e-12)
                a_ij = compute_binary_roc_auc((sub_y == i).astype(int), sub_p[:, i] / denom)
                a_ji = compute_binary_roc_auc((sub_y == j).astype(int), sub_p[:, j] / denom)
                pair_aucs.append(0.5 * (a_ij + a_ji))
        return float(np.mean(pair_aucs)) if pair_aucs else 0.5


def compute_fold_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    num_classes: int = 4,
) -> dict[str, Any]:
    """Compute Precision, Recall, Macro F1, Per-Class metrics, Accuracy, OvO AUC-ROC, and 4x4 Confusion Matrix."""
    cm = compute_confusion_matrix_4x4(y_true, y_pred, num_classes=num_classes)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    accuracy = float(np.trace(cm) / max(1, np.sum(cm)))
    ovo_auc = (
        compute_multiclass_ovo_auc_roc(y_true, y_prob, num_classes=num_classes)
        if y_prob is not None
        else np.nan
    )

    return {
        "macro_f1": float(np.mean(f1s)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "accuracy": accuracy,
        "ovo_auc_roc": ovo_auc,
        "f1_grade_0": f1s[0],
        "f1_grade_1": f1s[1],
        "f1_grade_2": f1s[2],
        "f1_grade_3": f1s[3],
        "precision_grade_0": precisions[0],
        "precision_grade_1": precisions[1],
        "precision_grade_2": precisions[2],
        "precision_grade_3": precisions[3],
        "recall_grade_0": recalls[0],
        "recall_grade_1": recalls[1],
        "recall_grade_2": recalls[2],
        "recall_grade_3": recalls[3],
        "confusion_matrix": cm,
    }


def evaluate_single_model_5x2(
    predictions_df: pd.DataFrame,
    model_name: str,
    m1_folds_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Evaluate a model's predictions across all 10 Stratified 5x2 CV configurations."""
    if m1_folds_df is not None:
        test_folds = m1_folds_df[m1_folds_df["split"] == "test"].copy()
        if len(predictions_df) == len(test_folds):
            merged = test_folds[
                ["configuration", "repeat", "fold", "filename", "authoritative_grade"]
            ].merge(
                predictions_df.drop(columns=["authoritative_grade"], errors="ignore"),
                on=["configuration", "repeat", "fold", "filename"],
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(test_folds):
                raise ValueError(
                    f"Prediction–Label alignment mismatch for {model_name}: {len(merged)} vs {len(test_folds)}"
                )
            predictions_df = merged

    fold_rows: list[dict[str, Any]] = []
    total_cm = np.zeros((4, 4), dtype=int)

    for config_idx, group in predictions_df.groupby("configuration", sort=True):
        y_true = group["authoritative_grade"].to_numpy(dtype=int)
        y_pred = group["predicted_grade"].to_numpy(dtype=int)
        prob_cols = ["prob_grade_0", "prob_grade_1", "prob_grade_2", "prob_grade_3"]
        y_prob = (
            group[prob_cols].to_numpy(dtype=np.float64)
            if set(prob_cols).issubset(group.columns)
            else None
        )
        metrics = compute_fold_classification_metrics(y_true, y_pred, y_prob)
        cm = metrics.pop("confusion_matrix")
        total_cm += cm

        latency_ms = (
            float(group["inference_latency_ms"].mean())
            if "inference_latency_ms" in group.columns
            else np.nan
        )
        memory_mb = (
            float(group["peak_memory_mb"].mean())
            if "peak_memory_mb" in group.columns
            else np.nan
        )

        fold_rows.append(
            {
                "model": model_name,
                "configuration": int(config_idx),
                "repeat": int(group["repeat"].iloc[0]),
                "fold": int(group["fold"].iloc[0]),
                "test_n": len(group),
                **metrics,
                "inference_latency_ms": latency_ms,
                "peak_memory_mb": memory_mb,
            }
        )

    fold_df = pd.DataFrame(fold_rows).sort_values("configuration").reset_index(drop=True)
    metric_keys = [
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "accuracy",
        "ovo_auc_roc",
        "f1_grade_0",
        "f1_grade_1",
        "f1_grade_2",
        "f1_grade_3",
        "inference_latency_ms",
    ]
    summary = {
        "model": model_name,
        "configurations_evaluated": len(fold_df),
        "total_test_predictions": int(fold_df["test_n"].sum()),
    }
    for k in metric_keys:
        if k in fold_df.columns and fold_df[k].notna().any():
            summary[f"{k}_mean"] = float(fold_df[k].mean())
            summary[f"{k}_std"] = float(fold_df[k].std(ddof=1)) if len(fold_df) > 1 else 0.0

    return {
        "model": model_name,
        "fold_metrics_df": fold_df,
        "summary": summary,
        "confusion_matrix": total_cm,
    }


def combined_5x2_cv_f_test(
    rf_fold_df: pd.DataFrame,
    rcnn_fold_df: pd.DataFrame,
    metric: str = "macro_f1",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Perform Alpaydın's (1999) Combined 5x2 Cross-Validation F-Test."""
    merged = rf_fold_df[["configuration", "repeat", "fold", metric]].merge(
        rcnn_fold_df[["configuration", "repeat", "fold", metric]],
        on=["configuration", "repeat", "fold"],
        suffixes=("_rf", "_rcnn"),
        validate="one_to_one",
    )
    if len(merged) != 10:
        raise ValueError(
            f"Combined 5x2 CV F-test requires all 10 configurations (5 repeats x 2 folds); got {len(merged)}"
        )

    numerator = 0.0
    denom_variances = 0.0
    repeat_details: list[dict[str, Any]] = []

    for repeat_idx in range(1, 6):
        rep_rows = merged[merged["repeat"] == repeat_idx].sort_values("fold")
        p1 = float(rep_rows.iloc[0][f"{metric}_rf"] - rep_rows.iloc[0][f"{metric}_rcnn"])
        p2 = float(rep_rows.iloc[1][f"{metric}_rf"] - rep_rows.iloc[1][f"{metric}_rcnn"])
        p_bar = 0.5 * (p1 + p2)
        s_sq = (p1 - p_bar) ** 2 + (p2 - p_bar) ** 2
        numerator += (p1 ** 2) + (p2 ** 2)
        denom_variances += s_sq
        repeat_details.append(
            {
                "repeat": repeat_idx,
                "diff_fold_1": round(p1, 6),
                "diff_fold_2": round(p2, 6),
                "repeat_mean_diff": round(p_bar, 6),
                "repeat_variance_s2": round(s_sq, 8),
            }
        )

    denominator = 2.0 * denom_variances
    if denominator <= 1e-15:
        f_stat = float("inf") if numerator > 0 else 0.0
        p_value = 0.0 if numerator > 0 else 1.0
    else:
        f_stat = float(numerator / denominator)
        p_value = float(stats.f.sf(f_stat, dfn=10, dfd=5))

    f_critical = float(stats.f.ppf(1.0 - alpha, dfn=10, dfd=5))
    rf_mean = float(merged[f"{metric}_rf"].mean())
    rcnn_mean = float(merged[f"{metric}_rcnn"].mean())
    mean_diff = rf_mean - rcnn_mean
    reject_h0 = bool(p_value < alpha)

    if reject_h0 and mean_diff > 0:
        adoption_decision = (
            "ADOPT PROPOSED MODEL: Statistically significant superiority of the Chromatic+GLCM "
            "Random Forest over the Faster R-CNN lesion-counting baseline (Reject H0 of Hypothesis 2)."
        )
    elif mean_diff > 0:
        adoption_decision = (
            "ADOPT PROPOSED MODEL (EFFICIENCY & INTERPRETABILITY ADVANTAGE): Proposed Random Forest "
            "achieves higher Macro F1 with lower computational overhead, though F-test p >= alpha."
        )
    else:
        adoption_decision = (
            "RETAIN BASELINE / FURTHER FEATURE TUNING NEEDED: Proposed Random Forest did not "
            "exceed baseline Macro F1."
        )

    return {
        "metric": metric,
        "dfn": 10,
        "dfd": 5,
        "alpha": alpha,
        "f_statistic": round(f_stat, 6),
        "f_critical_alpha_005": round(f_critical, 6),
        "p_value": round(p_value, 8),
        "reject_null_hypothesis_h0": reject_h0,
        "rf_mean": round(rf_mean, 6),
        "rcnn_mean": round(rcnn_mean, 6),
        "mean_difference_rf_minus_rcnn": round(mean_diff, 6),
        "repeat_breakdown": repeat_details,
        "model_adoption_decision": adoption_decision,
    }


def plot_confusion_matrices_and_comparison(
    rf_eval: dict[str, Any] | None,
    rcnn_eval: dict[str, Any] | None,
    f_test_result: dict[str, Any] | None,
    output_dir: Path,
) -> Path:
    """Render publication-quality confusion matrices and comparative metric bar charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    short_labels = ["Mild\n(G0)", "Moderate\n(G1)", "Severe\n(G2)", "Very Severe\n(G3)"]

    def _draw_cm(ax: Any, cm: np.ndarray, title: str, cmap: str) -> None:
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        norm_cm = cm.astype(np.float64) / row_sums * 100.0
        im = ax.imshow(norm_cm, cmap=cmap, vmin=0, vmax=100)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(short_labels, fontsize=8.5)
        ax.set_yticklabels(short_labels, fontsize=8.5)
        ax.set_xlabel("Predicted Hayashi Grade", fontsize=9.5)
        ax.set_ylabel("Actual Hayashi Grade", fontsize=9.5)
        for r in range(4):
            for c in range(4):
                val = cm[r, c]
                pct = norm_cm[r, c]
                txt_color = "white" if pct > 52 else "#1a1a1a"
                ax.text(
                    c,
                    r,
                    f"{val}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=8.5,
                    fontweight="bold" if r == c else "normal",
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row %")

    if rf_eval is not None:
        _draw_cm(
            axes[0],
            rf_eval["confusion_matrix"],
            f"Proposed Random Forest (M5)\nMacro F1 = {rf_eval['summary']['macro_f1_mean']:.3f} ± {rf_eval['summary']['macro_f1_std']:.3f}",
            "Blues",
        )
    else:
        axes[0].axis("off")

    if rcnn_eval is not None:
        _draw_cm(
            axes[1],
            rcnn_eval["confusion_matrix"],
            f"Baseline Faster R-CNN (M2)\nMacro F1 = {rcnn_eval['summary']['macro_f1_mean']:.3f} ± {rcnn_eval['summary']['macro_f1_std']:.3f}",
            "Oranges",
        )
    else:
        axes[1].axis("off")

    ax3 = axes[2]
    metrics_to_plot = [
        ("macro_f1_mean", "Macro F1"),
        ("macro_precision_mean", "Precision"),
        ("macro_recall_mean", "Recall"),
        ("ovo_auc_roc_mean", "OvO AUC-ROC"),
        ("accuracy_mean", "Accuracy"),
    ]
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    if rf_eval is not None:
        rf_vals = [rf_eval["summary"].get(k, 0.0) for k, _ in metrics_to_plot]
        ax3.bar(x - width / 2, rf_vals, width, label="Proposed RF (M5)", color="#1f77b4")
    if rcnn_eval is not None:
        rcnn_vals = [rcnn_eval["summary"].get(k, 0.0) for k, _ in metrics_to_plot]
        ax3.bar(
            x + width / 2,
            rcnn_vals,
            width,
            label="Baseline R-CNN (M2)",
            color="#ff7f0e",
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels([lbl for _, lbl in metrics_to_plot], fontsize=9)
    ax3.set_ylim(0.0, 1.08)
    ax3.set_ylabel("Score", fontsize=9.5)
    subtitle = ""
    if f_test_result is not None:
        subtitle = (
            f"\nCombined 5×2 CV F = {f_test_result['f_statistic']:.2f} "
            f"(p = {f_test_result['p_value']:.4g})"
        )
    ax3.set_title(
        f"5×2 Cross-Validation Metric Comparison{subtitle}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax3.legend(loc="lower right", fontsize=8.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = output_dir / "m6_comparative_evaluation.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def run_m6_evaluation(
    project_root: Path,
    rf_predictions_path: Path | None = None,
    rcnn_predictions_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run M6 Model Evaluation for Random Forest (M5), Faster R-CNN (M2), and Combined 5x2 F-Test."""
    m1_folds_path = project_root / "M1_outputs" / "m1_5x2_fold_assignments.csv"
    m1_folds_df = pd.read_csv(m1_folds_path) if m1_folds_path.is_file() else None

    rf_csv = rf_predictions_path or (
        project_root / "M5_outputs" / "m5_rf_5x2_oof_predictions.csv"
    )
    rcnn_csv = rcnn_predictions_path or (
        project_root / "M2_outputs" / "m2_rcnn_5x2_oof_predictions.csv"
    )
    out_root = output_dir or (project_root / "M6_outputs")
    out_root.mkdir(parents=True, exist_ok=True)

    rf_eval = None
    if rf_csv.is_file():
        rf_eval = evaluate_single_model_5x2(
            pd.read_csv(rf_csv), "Proposed_Random_Forest_M5", m1_folds_df
        )
        rf_eval["fold_metrics_df"].to_csv(
            out_root / "m6_rf_fold_metrics.csv", index=False
        )
        pd.DataFrame(
            rf_eval["confusion_matrix"],
            index=[f"Actual_G{i}" for i in range(4)],
            columns=[f"Pred_G{i}" for i in range(4)],
        ).to_csv(out_root / "m6_rf_confusion_matrix.csv")
        print(
            f"[M6] Proposed Random Forest (M5): Macro F1 = {rf_eval['summary']['macro_f1_mean']:.4f} "
            f"± {rf_eval['summary']['macro_f1_std']:.4f} | "
            f"OvO AUC-ROC = {rf_eval['summary']['ovo_auc_roc_mean']:.4f} | "
            f"Acc = {rf_eval['summary']['accuracy_mean']:.2%}"
        )

    rcnn_eval = None
    if rcnn_csv.is_file():
        rcnn_eval = evaluate_single_model_5x2(
            pd.read_csv(rcnn_csv), "Baseline_Faster_RCNN_M2", m1_folds_df
        )
        rcnn_eval["fold_metrics_df"].to_csv(
            out_root / "m6_rcnn_fold_metrics.csv", index=False
        )
        pd.DataFrame(
            rcnn_eval["confusion_matrix"],
            index=[f"Actual_G{i}" for i in range(4)],
            columns=[f"Pred_G{i}" for i in range(4)],
        ).to_csv(out_root / "m6_rcnn_confusion_matrix.csv")
        print(
            f"[M6] Baseline Faster R-CNN (M2):  Macro F1 = {rcnn_eval['summary']['macro_f1_mean']:.4f} "
            f"± {rcnn_eval['summary']['macro_f1_std']:.4f} | "
            f"OvO AUC-ROC = {rcnn_eval['summary']['ovo_auc_roc_mean']:.4f} | "
            f"Acc = {rcnn_eval['summary']['accuracy_mean']:.2%}"
        )

    f_test_macro_f1 = None
    f_test_accuracy = None
    if rf_eval is not None and rcnn_eval is not None:
        f_test_macro_f1 = combined_5x2_cv_f_test(
            rf_eval["fold_metrics_df"], rcnn_eval["fold_metrics_df"], metric="macro_f1"
        )
        f_test_accuracy = combined_5x2_cv_f_test(
            rf_eval["fold_metrics_df"], rcnn_eval["fold_metrics_df"], metric="accuracy"
        )
        print(
            f"[M6] Alpaydın (1999) Combined 5x2 CV F-Test (Macro F1): "
            f"F(10, 5) = {f_test_macro_f1['f_statistic']:.4f}, "
            f"p = {f_test_macro_f1['p_value']:.6g} "
            f"(F_crit = {f_test_macro_f1['f_critical_alpha_005']:.4f})"
        )
        print(f"[M6] Decision: {f_test_macro_f1['model_adoption_decision']}")

    fig_path = plot_confusion_matrices_and_comparison(
        rf_eval, rcnn_eval, f_test_macro_f1, out_root
    )

    summary_rows = []
    if rf_eval is not None:
        summary_rows.append(rf_eval["summary"])
    if rcnn_eval is not None:
        summary_rows.append(rcnn_eval["summary"])
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            out_root / "m6_model_evaluation_summary.csv", index=False
        )

    report_payload = {
        "random_forest_summary": rf_eval["summary"] if rf_eval else None,
        "faster_rcnn_summary": rcnn_eval["summary"] if rcnn_eval else None,
        "combined_5x2_f_test_macro_f1": f_test_macro_f1,
        "combined_5x2_f_test_accuracy": f_test_accuracy,
        "comparison_figure": str(fig_path),
    }
    (out_root / "m6_evaluation_report.json").write_text(
        json.dumps(report_payload, indent=2), encoding="utf-8"
    )
    return report_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M6 Model Evaluation (Precision, Recall, Macro F1, Confusion Matrix, OvO AUC-ROC, 5x2 F-Test)"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--rf-predictions", type=Path, default=None)
    parser.add_argument("--rcnn-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = find_project_root(args.project_root, require_jpeg_dir=False)
    run_m6_evaluation(
        project_root=root,
        rf_predictions_path=(
            args.rf_predictions.resolve() if args.rf_predictions else None
        ),
        rcnn_predictions_path=(
            args.rcnn_predictions.resolve() if args.rcnn_predictions else None
        ),
        output_dir=args.output_dir.resolve() if args.output_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
