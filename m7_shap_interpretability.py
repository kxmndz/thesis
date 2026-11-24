#!/usr/bin/env python3
"""M7 – SHAP TreeExplainer Global and Local Interpretability Module.

Strictly uses `shap.TreeExplainer` on the trained Random Forest model and real M4 feature matrix.
Generates both compact (Top-12) and complete (All-Features) global SHAP outputs:
  - m7_shap_feature_importance.csv (full ranked feature importance metrics)
  - m7_all_features_importance_table.csv (separate complete feature importance table CSV)
  - m7_all_features_importance_table.png (separate publication-ready table figure for all features)
  - m7_global_shap_bar.png (Top-12 stacked multi-class mean |SHAP| bar plot)
  - m7_global_shap_beeswarm.png (Top-12 directional beeswarm plot for Severe Grades 2+3)
  - m7_global_shap_beeswarm_all_features.png (separate directional beeswarm plot for ALL features)
  - local_explanations/local_shap_*_{waterfall,force}.png (per-grade local explanations)
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m1_data_loader import assert_no_watermarked_filenames
from m3_common import find_project_root
from m4_feature_extraction import get_feature_columns

LOGGER = logging.getLogger(__name__)


def classify_feature_family(feature_name: str) -> str:
    """Map an M4 feature column name to its dermatological feature family."""
    if feature_name.startswith("a_glcm_"):
        return "CIELAB a* Chromatic GLCM"
    if feature_name.startswith("glcm_"):
        return "CLAHE L* Luminance GLCM"
    if feature_name.startswith("a_star_"):
        return "CIELAB a* Chromatic"
    return "Other"


def _robust_normalize_feature(f_vals: np.ndarray) -> np.ndarray:
    """Normalize feature values to [0, 1] using 2nd-98th percentile clipping for beeswarm color mapping."""
    vals = np.asarray(f_vals, dtype=np.float64)
    lo, hi = np.percentile(vals, [2.0, 98.0])
    if hi - lo <= 1e-12:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo <= 1e-12:
        return np.full(vals.shape, 0.5, dtype=np.float64)
    return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)


def compute_tree_shap_values(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact TreeSHAP values (n_samples, n_features, 4) and base_values (4,) via shap.TreeExplainer."""
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    ev = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
    if isinstance(sv, list):
        sv_arr = np.stack(sv, axis=-1).astype(np.float64)
    else:
        sv_arr = np.asarray(sv, dtype=np.float64)

    n_samples, n_features = X.shape
    n_classes = int(len(ev))
    if sv_arr.shape != (n_samples, n_features, n_classes):
        if sv_arr.ndim == 3 and sv_arr.shape == (n_classes, n_samples, n_features):
            sv_arr = np.transpose(sv_arr, (1, 2, 0))
        else:
            raise ValueError(
                f"Unexpected TreeSHAP array shape {sv_arr.shape}; "
                f"expected ({n_samples}, {n_features}, {n_classes})."
            )
    return sv_arr, ev


def build_feature_importance_table(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Build complete multi-class SHAP feature importance table across ALL features."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)  # (n_features, n_classes)
    total_imp = np.sum(mean_abs, axis=1)
    sv_severe = shap_values[:, :, 2] + shap_values[:, :, 3]
    severe_imp = np.mean(np.abs(sv_severe), axis=0)
    denom = max(float(np.sum(total_imp)), 1e-12)

    df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "feature_family": [classify_feature_family(f) for f in feature_names],
            "mean_abs_shap_grade_0": mean_abs[:, 0],
            "mean_abs_shap_grade_1": mean_abs[:, 1],
            "mean_abs_shap_grade_2": mean_abs[:, 2],
            "mean_abs_shap_grade_3": mean_abs[:, 3],
            "mean_abs_shap_total": total_imp,
            "mean_abs_shap_severe_g2_g3": severe_imp,
            "relative_importance_pct": (total_imp / denom) * 100.0,
        }
    ).sort_values("mean_abs_shap_total", ascending=False).reset_index(drop=True)

    df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=int))
    df["cumulative_importance_pct"] = df["relative_importance_pct"].cumsum()
    return df


def plot_global_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_k: int | None = 12,
) -> None:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total_imp = np.sum(mean_abs, axis=1)
    n_total = len(feature_names)
    k = n_total if (top_k is None or top_k >= n_total) else int(top_k)
    order = np.argsort(total_imp)[-k:]

    fig_h = max(5.5, 0.34 * k + 1.8) if k > 12 else 5.5
    fig, ax = plt.subplots(figsize=(9.5, fig_h), dpi=150)
    left = np.zeros(len(order))
    colors = ["#4caf50", "#ffb300", "#fb8c00", "#e53935"]
    for g in range(4):
        vals = mean_abs[order, g]
        ax.barh(np.arange(len(order)), vals, left=left, color=colors[g], label=f"Grade {g}")
        left += vals
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8.5 if k > 20 else 9.5)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output Probability)")
    scope_tag = f"All {n_total} Features" if k == n_total else f"Top {k} of {n_total} Features"
    ax.set_title(f"M7 Global SHAP Feature Importance — {scope_tag} (Random Forest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_global_shap_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_k: int | None = 12,
) -> None:
    sv_severe = shap_values[:, :, 2] + shap_values[:, :, 3]
    imp = np.mean(np.abs(sv_severe), axis=0)
    n_total = len(feature_names)
    k = n_total if (top_k is None or top_k >= n_total) else int(top_k)
    order = np.argsort(imp)[-k:]

    fig_w = 10.5 if k > 20 else 9.2
    fig_h = max(5.5, 0.34 * k + 2.0) if k > 12 else 5.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160 if k > 20 else 140)
    rng = np.random.default_rng(42)

    if k > 15:
        for row_pos in range(k):
            if row_pos % 2 == 1:
                ax.axhspan(row_pos - 0.5, row_pos + 0.5, color="#f4f6f8", zorder=0)

    sc = None
    point_size = 7 if k > 20 else 10
    jitter_sd = 0.14 if k > 20 else 0.12
    for row_pos, feat_idx in enumerate(order):
        f_vals = X[:, feat_idx]
        norm_f = _robust_normalize_feature(f_vals)
        jitter = rng.normal(0.0, jitter_sd, size=len(f_vals))
        sc = ax.scatter(
            sv_severe[:, feat_idx],
            np.full(len(f_vals), row_pos, dtype=np.float64) + jitter,
            c=norm_f,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=point_size,
            alpha=0.72,
            edgecolors="none",
            zorder=2,
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9, zorder=3)
    ax.set_ylim(-0.7, k - 0.3)
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8.2 if k > 20 else 9.5)
    ax.set_xlabel("SHAP Value (Directional Impact Toward Severe Acne: Grade 2 + Grade 3)")
    scope_tag = f"All {n_total} Features" if k == n_total else f"Top {k} of {n_total} Features"
    ax.set_title(f"M7 Global Directional SHAP Beeswarm Summary Plot ({scope_tag})")
    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=40 if k > 20 else 25)
        cbar.set_label("Normalized Feature Value (2nd–98th Pct: Low -> High)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_table(
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render a publication-ready table figure of SHAP feature importance across ALL features."""
    n_rows = len(importance_df)
    family_shares = (
        importance_df.groupby("feature_family")["relative_importance_pct"].sum().to_dict()
    )
    share_summary = "  |  ".join(
        f"{fam}: {pct:.1f}%"
        for fam, pct in sorted(family_shares.items(), key=lambda kv: kv[1], reverse=True)
    )

    col_labels = [
        "Rank",
        "Feature",
        "Feature Family",
        "G0 |SHAP|",
        "G1 |SHAP|",
        "G2 |SHAP|",
        "G3 |SHAP|",
        "Total |SHAP|",
        "Severe (G2+G3)",
        "Rel. (%)",
        "Cum. (%)",
    ]

    cell_rows: list[list[str]] = []
    for _, row in importance_df.iterrows():
        cell_rows.append(
            [
                f"{int(row['rank'])}",
                str(row["feature"]),
                str(row["feature_family"]),
                f"{float(row['mean_abs_shap_grade_0']):.4f}",
                f"{float(row['mean_abs_shap_grade_1']):.4f}",
                f"{float(row['mean_abs_shap_grade_2']):.4f}",
                f"{float(row['mean_abs_shap_grade_3']):.4f}",
                f"{float(row['mean_abs_shap_total']):.4f}",
                f"{float(row['mean_abs_shap_severe_g2_g3']):.4f}",
                f"{float(row['relative_importance_pct']):.2f}%",
                f"{float(row['cumulative_importance_pct']):.2f}%",
            ]
        )

    # Append summary totals row
    cell_rows.append(
        [
            "—",
            f"TOTAL ({n_rows} Features)",
            "All Families Combined",
            f"{float(importance_df['mean_abs_shap_grade_0'].sum()):.4f}",
            f"{float(importance_df['mean_abs_shap_grade_1'].sum()):.4f}",
            f"{float(importance_df['mean_abs_shap_grade_2'].sum()):.4f}",
            f"{float(importance_df['mean_abs_shap_grade_3'].sum()):.4f}",
            f"{float(importance_df['mean_abs_shap_total'].sum()):.4f}",
            f"{float(importance_df['mean_abs_shap_severe_g2_g3'].sum()):.4f}",
            "100.00%",
            "100.00%",
        ]
    )

    fig_h = max(6.0, 0.30 * (n_rows + 2) + 1.4)
    fig, ax = plt.subplots(figsize=(14.5, fig_h), dpi=180)
    ax.axis("off")

    col_widths = [0.045, 0.215, 0.185, 0.068, 0.068, 0.068, 0.068, 0.076, 0.085, 0.061, 0.061]
    table = ax.table(
        cellText=cell_rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="right",
        bbox=[0.0, 0.0, 1.0, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)

    header_bg = "#1f3a52"
    total_bg = "#e3edf7"
    family_tints = {
        "CIELAB a* Chromatic": "#fff3e0",
        "CIELAB a* Chromatic GLCM": "#fce4ec",
        "CLAHE L* Luminance GLCM": "#e8f5e9",
    }

    total_row_idx = n_rows + 1
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color="white", weight="bold", ha="center")
        elif r == total_row_idx:
            cell.set_facecolor(total_bg)
            cell.set_text_props(color="#102a43", weight="bold", ha="left" if c in (1, 2) else ("center" if c == 0 else "right"))
        else:
            base_bg = "#ffffff" if r % 2 == 1 else "#f7f9fb"
            if c == 2:
                fam_text = cell_rows[r - 1][2]
                cell.set_facecolor(family_tints.get(fam_text, base_bg))
            else:
                cell.set_facecolor(base_bg)

            if c == 0:
                cell.set_text_props(ha="center", weight="bold", color="#334e68")
            elif c in (1, 2):
                cell.set_text_props(ha="left", color="#102a43")
            elif c == 7:
                cell.set_text_props(ha="right", weight="bold", color="#102a43")
            else:
                cell.set_text_props(ha="right", color="#243b53")

    ax.text(
        0.5,
        0.985,
        f"M7 Complete Global SHAP Feature Importance Table (All {n_rows} Features)",
        ha="center",
        va="top",
        fontsize=12.5,
        weight="bold",
        color="#102a43",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.958,
        f"Family Share of Total Mean |SHAP| — {share_summary}",
        ha="center",
        va="top",
        fontsize=9.0,
        color="#486581",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_local_shap_waterfall(
    sample_shap_1d: np.ndarray,
    base_value: float,
    feature_values: np.ndarray,
    feature_names: list[str],
    target_grade: int,
    image_name: str,
    output_path: Path,
    max_display: int = 10,
    ax: Any | None = None,
) -> None:
    order = np.argsort(np.abs(sample_shap_1d))[::-1][:max_display][::-1]
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    vals = sample_shap_1d[order]
    labels = [f"{feature_names[i]} = {feature_values[i]:.2f}" for i in order]
    colors = ["#e53935" if v >= 0 else "#1e88e5" for v in vals]
    ax.barh(np.arange(len(order)), vals, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    fx = base_value + float(np.sum(sample_shap_1d))
    ax.set_title(
        f"Local SHAP Waterfall: {image_name} (Grade {target_grade}: E[f(X)]={base_value:.3f} -> f(x)={fx:.3f})"
    )
    if own_fig:
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def plot_local_shap_force(
    sample_shap_1d: np.ndarray,
    base_value: float,
    feature_values: np.ndarray,
    feature_names: list[str],
    target_grade: int,
    image_name: str,
    output_path: Path,
    top_k: int = 6,
) -> None:
    order = np.argsort(np.abs(sample_shap_1d))[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=140)
    pos_vals = [(feature_names[i], sample_shap_1d[i]) for i in order if sample_shap_1d[i] >= 0]
    neg_vals = [(feature_names[i], sample_shap_1d[i]) for i in order if sample_shap_1d[i] < 0]
    fx = base_value + float(np.sum(sample_shap_1d))
    cur = base_value
    for _name, v in pos_vals:
        ax.barh([0], [v], left=[cur], color="#e53935", edgecolor="white", height=0.45)
        cur += v
    cur = base_value
    for _name, v in neg_vals:
        ax.barh([0], [v], left=[cur], color="#1e88e5", edgecolor="white", height=0.45)
        cur += v
    ax.axvline(base_value, color="gray", linestyle="--", label=f"Base={base_value:.3f}")
    ax.axvline(fx, color="black", linewidth=1.5, label=f"f(x)={fx:.3f}")
    ax.set_yticks([])
    ax.set_xlabel(f"Predicted Probability Contribution for Grade {target_grade}")
    ax.set_title(f"Local SHAP Force Plot: {image_name} (Grade {target_grade})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_m7_shap_analysis(
    project_root: Path,
    best_model_path: Path | None = None,
    feature_matrix_path: Path | None = None,
    output_dir: Path | None = None,
    sample_filenames: list[str] | None = None,
) -> dict[str, Any]:
    out_root = output_dir or (project_root / "M7_outputs")
    local_dir = out_root / "local_explanations"
    local_dir.mkdir(parents=True, exist_ok=True)

    model_pkl = best_model_path or (project_root / "M5_outputs" / "models" / "rf_best_model.pkl")
    feat_csv = feature_matrix_path or (project_root / "M4_outputs" / "m4_feature_matrix.csv")
    if not model_pkl.is_file():
        raise FileNotFoundError(f"Missing trained Random Forest model: {model_pkl}. Run M5 first.")
    if not feat_csv.is_file():
        raise FileNotFoundError(f"Missing M4 feature matrix: {feat_csv}. Run M4 first.")

    with model_pkl.open("rb") as f_in:
        bundle = pickle.load(f_in)
    model = bundle["model"]
    feature_cols: list[str] = list(bundle.get("feature_columns", get_feature_columns()))
    feature_df = pd.read_csv(feat_csv)
    assert_no_watermarked_filenames(feature_df["filename"], context="M7 SHAP feature matrix")

    # Guard against stale model vs. upgraded M4 feature matrix mismatch
    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != len(feature_cols):
        raise ValueError(
            f"[M7] Model expects {model.n_features_in_} features, but bundle specifies {len(feature_cols)}."
        )
    missing_cols = [c for c in feature_cols if c not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"[M7] Feature matrix {feat_csv.name} is missing model features: {missing_cols}")

    expected_m4 = get_feature_columns()
    unmodeled_m4 = [c for c in expected_m4 if c in feature_df.columns and c not in feature_cols]
    if unmodeled_m4:
        LOGGER.warning(
            "[M7] Trained model (%s) uses %d features, while %s contains %d additional M4 features (%s...). "
            "Re-run M5 if you want those included in SHAP.",
            model_pkl.name,
            len(feature_cols),
            feat_csv.name,
            len(unmodeled_m4),
            ", ".join(unmodeled_m4[:3]),
        )

    LOGGER.info(
        "[M7] Computing TreeSHAP across all %d model features on %d images...",
        len(feature_cols),
        len(feature_df),
    )

    X = feature_df[feature_cols].to_numpy(dtype=np.float64)
    shap_values, base_values = compute_tree_shap_values(model, X)
    imp_df = build_feature_importance_table(shap_values, feature_cols)

    # Save CSVs (existing importance CSV + dedicated all-features table CSV)
    imp_csv_path = out_root / "m7_shap_feature_importance.csv"
    all_table_csv_path = out_root / "m7_all_features_importance_table.csv"
    imp_df.to_csv(imp_csv_path, index=False)
    imp_df.to_csv(all_table_csv_path, index=False)

    # Compact Top-12 global plots (existing outputs)
    bar_path = out_root / "m7_global_shap_bar.png"
    beeswarm_path = out_root / "m7_global_shap_beeswarm.png"
    plot_global_shap_bar(shap_values, feature_cols, bar_path, top_k=12)
    plot_global_shap_beeswarm(shap_values, X, feature_cols, beeswarm_path, top_k=12)

    # Separate ALL-features outputs (table image + all-features beeswarm plot)
    all_table_png_path = out_root / "m7_all_features_importance_table.png"
    all_beeswarm_path = out_root / "m7_global_shap_beeswarm_all_features.png"
    plot_feature_importance_table(imp_df, all_table_png_path)
    plot_global_shap_beeswarm(shap_values, X, feature_cols, all_beeswarm_path, top_k=None)

    if not sample_filenames:
        sample_filenames = [
            str(feature_df.loc[feature_df["authoritative_grade"] == g, "filename"].iloc[0])
            for g in (0, 1, 2, 3)
            if (feature_df["authoritative_grade"] == g).any()
        ]

    local_plots = []
    for fname in sample_filenames:
        matches = feature_df.index[feature_df["filename"] == fname].tolist()
        if not matches:
            continue
        row_idx = matches[0]
        x_row = X[row_idx]
        pred_grade = int(model.predict(x_row[np.newaxis, :])[0])
        stem = Path(fname).stem
        wf_path = local_dir / f"local_shap_{stem}_waterfall.png"
        fc_path = local_dir / f"local_shap_{stem}_force.png"
        plot_local_shap_waterfall(
            shap_values[row_idx, :, pred_grade],
            float(base_values[pred_grade]),
            x_row,
            feature_cols,
            target_grade=pred_grade,
            image_name=fname,
            output_path=wf_path,
        )
        plot_local_shap_force(
            shap_values[row_idx, :, pred_grade],
            float(base_values[pred_grade]),
            x_row,
            feature_cols,
            target_grade=pred_grade,
            image_name=fname,
            output_path=fc_path,
        )
        local_plots.append(
            {
                "filename": fname,
                "predicted_grade": str(pred_grade),
                "waterfall_plot": str(wf_path),
                "force_plot": str(fc_path),
            }
        )

    LOGGER.info(
        "[M7] Complete (%d features) | Top-12 Bar & Beeswarm + All-Features Table (%s) + All-Features Beeswarm (%s)",
        len(feature_cols),
        all_table_png_path.name,
        all_beeswarm_path.name,
    )

    return {
        "n_features": int(len(feature_cols)),
        "feature_columns": list(feature_cols),
        "importance_df": imp_df,
        "importance_csv": imp_csv_path,
        "all_features_table_csv": all_table_csv_path,
        "all_features_table_image": all_table_png_path,
        "global_bar_plot": bar_path,
        "global_beeswarm_plot": beeswarm_path,
        "global_beeswarm_all_features_plot": all_beeswarm_path,
        "local_plots": local_plots,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M7 SHAP TreeExplainer Global and Local Interpretability Module")
    parser.add_argument("--project-root", "--root-dir", dest="project_root", type=Path, default=None)
    parser.add_argument("--best-model", type=Path, default=None)
    parser.add_argument("--feature-matrix", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--filenames", type=str, nargs="*", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = find_project_root(args.project_root, require_jpeg_dir=False)
    run_m7_shap_analysis(
        project_root=root,
        best_model_path=args.best_model.resolve() if args.best_model else None,
        feature_matrix_path=args.feature_matrix.resolve() if args.feature_matrix else None,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        sample_filenames=args.filenames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
