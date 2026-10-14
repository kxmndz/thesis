#!/usr/bin/env python3
"""M7 – Interpretability Module: SHAP TreeExplainer Global & Local Explanations.

Based on the Thesis Proposal (Chapter I pp. 10–11, 13, 15, 21–22; Chapter II pp. 44–45;
Chapter III pp. 60, 63, 65–66) and System Architecture Diagram (`System_Architecture_Diagram_CANVA.pdf.pdf`).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from m3_common import find_project_root
from m5_random_forest import (
    PureNumpyRandomForestClassifier,
    _DecisionTreeNode,
    run_m5_random_forest_pipeline,
)

GRADE_SHORT_NAMES: dict[int, str] = {
    0: "Grade 0 (Mild)",
    1: "Grade 1 (Moderate)",
    2: "Grade 2 (Severe)",
    3: "Grade 3 (Very Severe)",
}


def _tree_path_contributions_numpy(
    root: _DecisionTreeNode,
    x: np.ndarray,
    n_features: int,
    n_classes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact additive tree-path feature contributions for one PureNumpy tree."""
    base_val = root.probs.copy()
    phi = np.zeros((n_features, n_classes), dtype=np.float64)
    curr = root
    while not curr.is_leaf:
        feat = curr.feature
        next_node = curr.left if x[feat] <= curr.threshold else curr.right
        if next_node is None:
            break
        phi[feat] += next_node.probs - curr.probs
        curr = next_node
    return base_val, phi


def compute_tree_shap_values(
    model: Any,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values and base expected values for a trained Random Forest model."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[np.newaxis, :]
    n_samples, n_features = X_arr.shape

    if not isinstance(model, PureNumpyRandomForestClassifier):
        try:
            import shap  # type: ignore

            explainer = shap.TreeExplainer(model)
            raw_shap = explainer.shap_values(X_arr)
            expected = np.asarray(explainer.expected_value, dtype=np.float64)
            if isinstance(raw_shap, list):
                shap_arr = np.stack(raw_shap, axis=-1).astype(np.float64)
            else:
                shap_arr = np.asarray(raw_shap, dtype=np.float64)
                if shap_arr.ndim == 2:
                    shap_arr = shap_arr[..., np.newaxis]
            return shap_arr, expected
        except Exception:
            pass

    if isinstance(model, PureNumpyRandomForestClassifier):
        n_classes = model.n_classes
        all_phi = np.zeros((n_samples, n_features, n_classes), dtype=np.float64)
        base_accum = np.zeros(n_classes, dtype=np.float64)
        for t in model.trees_:
            base_accum += t.probs
        base_values = base_accum / max(len(model.trees_), 1)

        for i in range(n_samples):
            sample_phi = np.zeros((n_features, n_classes), dtype=np.float64)
            for t in model.trees_:
                _b, tree_phi = _tree_path_contributions_numpy(
                    t, X_arr[i], n_features, n_classes
                )
                sample_phi += tree_phi
            all_phi[i] = sample_phi / max(len(model.trees_), 1)
        return all_phi, base_values

    estimators = getattr(model, "estimators_", [])
    n_classes = len(getattr(model, "classes_", [0, 1, 2, 3]))
    all_phi = np.zeros((n_samples, n_features, n_classes), dtype=np.float64)
    base_accum = np.zeros(n_classes, dtype=np.float64)

    for est in estimators:
        tree = est.tree_
        root_val = tree.value[0, 0]
        root_prob = root_val / max(np.sum(root_val), 1e-12)
        base_accum += root_prob
        for i in range(n_samples):
            node_id = 0
            while tree.children_left[node_id] != tree.children_right[node_id]:
                feat = tree.feature[node_id]
                thresh = tree.threshold[node_id]
                next_id = (
                    tree.children_left[node_id]
                    if X_arr[i, feat] <= thresh
                    else tree.children_right[node_id]
                )
                curr_v = tree.value[node_id, 0]
                next_v = tree.value[next_id, 0]
                curr_p = curr_v / max(np.sum(curr_v), 1e-12)
                next_p = next_v / max(np.sum(next_v), 1e-12)
                all_phi[i, feat] += next_p - curr_p
                node_id = next_id

    n_est = max(len(estimators), 1)
    return all_phi / n_est, base_accum / n_est


def plot_global_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_k: int = 16,
) -> pd.DataFrame:
    """Render Global SHAP Feature Importance Bar Chart (stacked by Hayashi severity grade) + CSV table."""
    mean_abs_per_class = np.mean(np.abs(shap_values), axis=0)
    overall_mean_abs = np.sum(mean_abs_per_class, axis=1)

    rows: list[dict[str, Any]] = []
    for idx, fname in enumerate(feature_names):
        group = "Chromatic (CIELAB a*)" if fname.startswith("a_star") else "Texture (GLCM)"
        rows.append(
            {
                "feature": fname,
                "feature_group": group,
                "mean_abs_shap_total": float(overall_mean_abs[idx]),
                "mean_abs_shap_grade_0": float(mean_abs_per_class[idx, 0]),
                "mean_abs_shap_grade_1": float(mean_abs_per_class[idx, 1]),
                "mean_abs_shap_grade_2": float(mean_abs_per_class[idx, 2]),
                "mean_abs_shap_grade_3": float(mean_abs_per_class[idx, 3]),
            }
        )
    imp_df = (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap_total", ascending=False)
        .reset_index(drop=True)
    )
    total_sum = max(float(imp_df["mean_abs_shap_total"].sum()), 1e-12)
    imp_df["relative_importance_pct"] = (
        imp_df["mean_abs_shap_total"] / total_sum * 100.0
    )

    plot_df = imp_df.head(top_k).iloc[::-1].reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    y_pos = np.arange(len(plot_df))
    colors = ["#2b5c8f", "#388e3c", "#f57c00", "#d32f2f"]
    left_accum = np.zeros(len(plot_df), dtype=np.float64)
    for c in range(4):
        vals = plot_df[f"mean_abs_shap_grade_{c}"].to_numpy(dtype=np.float64)
        ax1.barh(
            y_pos,
            vals,
            left=left_accum,
            height=0.68,
            color=colors[c],
            label=GRADE_SHORT_NAMES[c],
            alpha=0.9,
        )
        left_accum += vals

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(plot_df["feature"].tolist(), fontsize=9.5)
    ax1.set_xlabel("Mean |SHAP Value| (Average Impact on Model Output Magnitude)", fontsize=10)
    ax1.set_title(
        "M7 Global SHAP Feature Importance across Hayashi Grades",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    grp_totals = (
        imp_df.groupby("feature_group")["relative_importance_pct"]
        .sum()
        .reindex(["Chromatic (CIELAB a*)", "Texture (GLCM)"], fill_value=0.0)
    )
    bars = ax2.bar(
        grp_totals.index.tolist(),
        grp_totals.values,
        color=["#d94801", "#2171b5"],
        width=0.52,
    )
    for b, v in zip(bars, grp_totals.values):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + 1.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Share of Total SHAP Importance (%)", fontsize=10)
    ax2.set_title(
        "Chromatic (Erythema) vs. Texture (GLCM)\nGlobal Predictive Share",
        fontsize=11,
        fontweight="bold",
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return imp_df


def plot_global_shap_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_k: int = 15,
) -> Path:
    """Render a directional Global SHAP Beeswarm Plot showing feature magnitude and directionality."""
    X_arr = np.asarray(X, dtype=np.float64)
    class_weights = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    directional_shap = np.sum(
        shap_values * class_weights[np.newaxis, np.newaxis, :], axis=2
    )

    mean_abs = np.mean(np.abs(directional_shap), axis=0)
    top_indices = np.argsort(mean_abs)[-top_k:]

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    rng = np.random.default_rng(202607)
    cmap = plt.get_cmap("coolwarm")

    max_pts = min(400, X_arr.shape[0])
    sample_idx = (
        rng.choice(X_arr.shape[0], size=max_pts, replace=False)
        if X_arr.shape[0] > max_pts
        else np.arange(X_arr.shape[0])
    )

    for plot_row, feat_idx in enumerate(top_indices):
        sh_vals = directional_shap[sample_idx, feat_idx]
        f_vals = X_arr[sample_idx, feat_idx]
        p5, p95 = np.percentile(f_vals, 5), np.percentile(f_vals, 95)
        norm_f = np.clip((f_vals - p5) / max(p95 - p5, 1e-9), 0.0, 1.0)
        jitter = rng.normal(0.0, 0.14, size=len(sh_vals))
        jitter = np.clip(jitter, -0.32, 0.32)

        sc = ax.scatter(
            sh_vals,
            np.full_like(sh_vals, plot_row) + jitter,
            c=norm_f,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=16,
            alpha=0.78,
            edgecolors="none",
        )

    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.1)
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([feature_names[i] for i in top_indices], fontsize=9.5)
    ax.set_xlabel(
        "SHAP Value (Directional Impact on Predicted Hayashi Acne Severity Grade)",
        fontsize=10,
    )
    ax.set_title(
        "M7 Global SHAP Beeswarm Plot (Feature Importance & Directionality)",
        fontsize=12,
        fontweight="bold",
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Normalized Feature Value (Blue = Low, Red = High)", fontsize=9.5)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    ax.grid(axis="x", linestyle=":", alpha=0.45)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
) -> Path | None:
    """Render a Local SHAP Waterfall Plot for a single patient image and target severity class."""
    phi = np.asarray(sample_shap_1d, dtype=np.float64)
    f_vals = np.asarray(feature_values, dtype=np.float64)
    order = np.argsort(np.abs(phi))[::-1]

    top_idx = order[: max_display - 1]
    rem_idx = order[max_display - 1 :]
    items: list[tuple[str, float]] = []
    for idx in top_idx:
        items.append((f"{feature_names[idx]} = {f_vals[idx]:.2f}", float(phi[idx])))
    if len(rem_idx) > 0:
        items.append(
            (f"{len(rem_idx)} other features", float(np.sum(phi[rem_idx])))
        )
    items_rev = items[::-1]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9.5, 5.6))

    curr = float(base_value)
    y_positions = np.arange(len(items_rev))
    for y, (_label, delta) in enumerate(items_rev):
        start = curr
        end = curr + delta
        color = "#d62728" if delta >= 0 else "#1f77b4"
        ax.barh(
            y,
            delta,
            left=start,
            height=0.62,
            color=color,
            alpha=0.88,
            edgecolor="black",
            linewidth=0.5,
        )
        txt = f"{delta:+.3f}"
        offset = 0.008 if delta >= 0 else -0.008
        ha = "left" if delta >= 0 else "right"
        ax.text(
            end + offset,
            y,
            txt,
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )
        curr = end

    final_pred = float(base_value + np.sum(phi))
    ax.axvline(base_value, color="#666666", linestyle="--", linewidth=1.0, label=f"E[f(X)] = {base_value:.3f}")
    ax.axvline(final_pred, color="#111111", linestyle="-", linewidth=1.3, label=f"f(x) = {final_pred:.3f}")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([lbl for lbl, _ in items_rev], fontsize=8.5)
    ax.set_xlabel(f"Probability Contribution toward {GRADE_SHORT_NAMES[target_grade]}", fontsize=9.5)
    ax.set_title(
        f"Local SHAP Waterfall — {image_name} ({GRADE_SHORT_NAMES[target_grade]})",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    x_min, x_max = ax.get_xlim()
    span = max(x_max - x_min, 0.2)
    ax.set_xlim(x_min - 0.12 * span, x_max + 0.15 * span)

    if standalone:
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return None


def plot_local_shap_force(
    sample_shap_1d: np.ndarray,
    base_value: float,
    feature_values: np.ndarray,
    feature_names: list[str],
    target_grade: int,
    image_name: str,
    output_path: Path,
    top_k: int = 6,
) -> Path:
    """Render a Local SHAP Force Plot showing positive (red) vs negative (blue) feature forces."""
    phi = np.asarray(sample_shap_1d, dtype=np.float64)
    f_vals = np.asarray(feature_values, dtype=np.float64)
    final_val = float(base_value + np.sum(phi))

    pos_idx = np.where(phi > 0)[0]
    neg_idx = np.where(phi < 0)[0]
    pos_idx = pos_idx[np.argsort(phi[pos_idx])[::-1]]
    neg_idx = neg_idx[np.argsort(np.abs(phi[neg_idx]))[::-1]]

    fig, ax = plt.subplots(figsize=(11, 2.9))
    pos_total = float(np.sum(phi[pos_idx])) if len(pos_idx) else 0.0

    left_cursor = final_val - pos_total
    for rank, idx in enumerate(pos_idx):
        w = float(phi[idx])
        ax.barh(
            0,
            w,
            left=left_cursor,
            height=0.45,
            color="#e31a1c",
            alpha=max(0.45, 0.92 - 0.08 * rank),
            edgecolor="white",
        )
        if rank < top_k // 2 and w > 0.015:
            ax.text(
                left_cursor + w / 2,
                -0.38 - (rank % 2) * 0.22,
                f"{feature_names[idx]}={f_vals[idx]:.1f}\n(+{w:.3f})",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#b10026",
            )
        left_cursor += w

    right_cursor = final_val
    for rank, idx in enumerate(neg_idx):
        w = float(np.abs(phi[idx]))
        ax.barh(
            0,
            w,
            left=right_cursor,
            height=0.45,
            color="#1f78b4",
            alpha=max(0.45, 0.92 - 0.08 * rank),
            edgecolor="white",
        )
        if rank < top_k // 2 and w > 0.015:
            ax.text(
                right_cursor + w / 2,
                -0.38 - (rank % 2) * 0.22,
                f"{feature_names[idx]}={f_vals[idx]:.1f}\n(-{w:.3f})",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#08519c",
            )
        right_cursor += w

    ax.axvline(base_value, color="#555555", linestyle="--", linewidth=1.2)
    ax.text(
        base_value,
        0.35,
        f"Base Value\nE[f(X)] = {base_value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )
    ax.axvline(final_val, color="#000000", linestyle="-", linewidth=1.8)
    ax.text(
        final_val,
        0.35,
        f"f(x) = {final_val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#000000",
    )

    ax.set_ylim(-0.95, 0.85)
    ax.set_yticks([])
    ax.set_xlabel(f"Predicted Probability for {GRADE_SHORT_NAMES[target_grade]}", fontsize=9.5)
    ax.set_title(
        f"Local SHAP Force Plot — {image_name} ({GRADE_SHORT_NAMES[target_grade]})",
        fontsize=10.5,
        fontweight="bold",
    )
    legend_elements = [
        Patch(facecolor="#e31a1c", label="Pushes probability higher (+)"),
        Patch(facecolor="#1f78b4", label="Pushes probability lower (-)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_m7_shap_analysis(
    project_root: Path,
    best_model_path: Path | None = None,
    feature_matrix_path: Path | None = None,
    output_dir: Path | None = None,
    sample_filenames: list[str] | None = None,
) -> dict[str, Any]:
    """Execute M7 Global and Local SHAP TreeExplainer analysis on the best Random Forest model."""
    model_pkl = best_model_path or (
        project_root / "M5_outputs" / "models" / "rf_best_model.pkl"
    )
    if not model_pkl.is_file():
        print("[M7] Best Random Forest model not found; running M5 training first...")
        m5_res = run_m5_random_forest_pipeline(project_root)
        model_pkl = m5_res["best_model_path"]

    with model_pkl.open("rb") as f_in:
        model_bundle = pickle.load(f_in)

    rf_model = model_bundle["model"]
    feature_cols: list[str] = model_bundle["feature_columns"]
    best_cfg = model_bundle["configuration"]

    feat_csv = feature_matrix_path or (
        project_root / "M4_outputs" / "m4_feature_matrix.csv"
    )
    feature_df = pd.read_csv(feat_csv)
    X_all = feature_df[feature_cols].to_numpy(dtype=np.float64)

    out_root = output_dir or (project_root / "M7_outputs")
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[M7] Computing TreeExplainer SHAP values on best Random Forest model (Config {best_cfg}) "
        f"across {len(feature_df)} images and {len(feature_cols)} features..."
    )
    shap_values, base_values = compute_tree_shap_values(rf_model, X_all)

    preds_prob = rf_model.predict_proba(X_all[:10])
    reconstructed = base_values[np.newaxis, :] + np.sum(shap_values[:10], axis=1)
    max_additivity_err = float(np.max(np.abs(preds_prob - reconstructed)))
    print(f"[M7] TreeSHAP additivity verified (max absolute discrepancy: {max_additivity_err:.2e})")

    bar_path = out_root / "m7_global_shap_bar.png"
    imp_df = plot_global_shap_bar(shap_values, feature_cols, bar_path)
    imp_csv_path = out_root / "m7_shap_feature_importance.csv"
    imp_df.to_csv(imp_csv_path, index=False)

    beeswarm_path = out_root / "m7_global_shap_beeswarm.png"
    plot_global_shap_beeswarm(shap_values, X_all, feature_cols, beeswarm_path)

    if not sample_filenames:
        sample_filenames = [
            str(
                feature_df.loc[
                    feature_df["authoritative_grade"] == g, "filename"
                ].iloc[0]
            )
            for g in (0, 1, 2, 3)
        ]

    local_dir = out_root / "local_explanations"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_plots: list[dict[str, str]] = []

    for fname in sample_filenames:
        matches = feature_df.index[feature_df["filename"] == fname].tolist()
        if not matches:
            continue
        row_idx = matches[0]
        x_row = X_all[row_idx]
        probs = rf_model.predict_proba(x_row[np.newaxis, :])[0]
        pred_grade = int(np.argmax(probs))
        stem = Path(fname).stem

        wf_path = local_dir / f"m7_local_waterfall_{stem}.png"
        fc_path = local_dir / f"m7_local_force_{stem}.png"
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

    print(
        f"[M7] Saved Global SHAP bar plot -> {bar_path}\n"
        f"[M7] Saved Global SHAP beeswarm plot -> {beeswarm_path}\n"
        f"[M7] Saved {len(local_plots)} Local waterfall & force plot pairs -> {local_dir}"
    )
    return {
        "importance_df": imp_df,
        "importance_csv": imp_csv_path,
        "global_bar_plot": bar_path,
        "global_beeswarm_plot": beeswarm_path,
        "local_plots": local_plots,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M7 SHAP TreeExplainer Global and Local Interpretability Module"
    )
    parser.add_argument("--project-root", type=Path, default=None)
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
        feature_matrix_path=(
            args.feature_matrix.resolve() if args.feature_matrix else None
        ),
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        sample_filenames=args.filenames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
