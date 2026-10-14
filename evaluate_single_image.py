#!/usr/bin/env python3
"""Single-Image End-to-End Pipeline & Dual-Model Evaluator (`evaluate_single_image.py`)."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from m2_baseline_rcnn import (
    HAYASHI_GRADE_NAMES,
    detect_lesions_single_image,
    draw_lesion_detections,
    load_ground_truth_lesion_counts,
)
from m3_bisenet_pipeline import get_m3_artifact_for_filename, load_m3_output
from m3_common import find_project_root, load_authoritative_labels_dict
from m4_feature_extraction import (
    extract_m4_features_for_single_image,
    extract_m4_features_from_arrays,
    quantize_to_32_levels,
)
from m5_random_forest import run_m5_random_forest_pipeline
from m7_shap_interpretability import (
    compute_tree_shap_values,
    plot_local_shap_force,
    plot_local_shap_waterfall,
)


def generate_synthetic_demo_face(output_path: Path, grade: int = 2) -> Path:
    """Create a realistic synthetic facial skin test image when running in offline demo mode."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(202610 + grade)
    h, w = 600, 480
    yy, xx = np.mgrid[0:h, 0:w]
    face_ellipse = (((xx - w / 2) / (w * 0.36)) ** 2 + ((yy - h / 2) / (h * 0.42)) ** 2) <= 1.0
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = [25, 28, 34]

    illum = 1.0 - 0.15 * np.sqrt(((xx - w * 0.45) / w) ** 2 + ((yy - h * 0.42) / h) ** 2)
    skin_r = np.clip((218 * illum + rng.normal(0, 4, size=(h, w))), 140, 245)
    skin_g = np.clip((174 * illum + rng.normal(0, 4, size=(h, w))), 110, 210)
    skin_b = np.clip((148 * illum + rng.normal(0, 4, size=(h, w))), 90, 190)
    rgb[face_ellipse, 0] = skin_r[face_ellipse].astype(np.uint8)
    rgb[face_ellipse, 1] = skin_g[face_ellipse].astype(np.uint8)
    rgb[face_ellipse, 2] = skin_b[face_ellipse].astype(np.uint8)

    n_lesions = {0: 3, 1: 12, 2: 28, 3: 60}.get(grade, 15)
    img = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(img)
    valid_ys, valid_xs = np.where(face_ellipse)
    for _ in range(n_lesions):
        idx = int(rng.integers(0, len(valid_xs)))
        cx, cy = int(valid_xs[idx]), int(valid_ys[idx])
        r = int(rng.integers(4, 10))
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(215, 72, 68),
            outline=(185, 50, 48),
        )
    img.save(output_path, quality=95)
    return output_path


def render_preprocessing_cielab_sheet(
    image_name: str,
    arrays: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    """Render an 8-panel visual breakdown of M3 Resolution Standardization, Masking, CIELAB, and CLAHE."""
    std_rgb = arrays["standardized_rgb"]
    orig_rgb = arrays.get("original_rgb", std_rgb)
    content_mask = arrays["content_mask"]
    roi_mask = arrays["final_roi_mask"]
    masked_rgb = np.where(roi_mask[..., None], std_rgb, 0)

    l_star = arrays["lab_l_star"]
    a_star = arrays["lab_a_star"]
    b_star = arrays["lab_b_star"]
    clahe_l = arrays["clahe_l_star"]
    quantized_32 = quantize_to_32_levels(clahe_l, (0.0, 100.0), levels=32)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))

    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title(f"1. Original Input\n({orig_rgb.shape[1]}×{orig_rgb.shape[0]} px)", fontsize=10, fontweight="bold")

    axes[0, 1].imshow(std_rgb)
    axes[0, 1].set_title("2. Standardized RGB\n(512×512 Letterbox Canvas)", fontsize=10, fontweight="bold")

    mask_overlay = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_overlay[content_mask & ~roi_mask] = [45, 55, 75]
    mask_overlay[roi_mask] = [40, 215, 115]
    axes[0, 2].imshow(mask_overlay)
    roi_pct = 100.0 * np.count_nonzero(roi_mask) / max(1, np.count_nonzero(content_mask))
    axes[0, 2].set_title(f"3. Skin + Nose ROI Mask\n({np.count_nonzero(roi_mask):,} px | {roi_pct:.1f}% of content)", fontsize=10, fontweight="bold")

    axes[0, 3].imshow(masked_rgb)
    axes[0, 3].set_title("4. Isolated Facial Skin ROI\n(Non-Skin & Padding Excluded)", fontsize=10, fontweight="bold")

    im_l = axes[1, 0].imshow(np.where(roi_mask, l_star, np.nan), cmap="gray", vmin=0, vmax=100)
    axes[1, 0].set_title("5. CIELAB L* (Luminance)\n[0–100 Scale in ROI]", fontsize=10, fontweight="bold")
    plt.colorbar(im_l, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_a = axes[1, 1].imshow(np.where(roi_mask, a_star, np.nan), cmap="RdYlGn_r")
    axes[1, 1].set_title("6. CIELAB a* (Erythema / Redness)\n[Unmodified Chromatic Channel]", fontsize=10, fontweight="bold")
    plt.colorbar(im_a, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_b = axes[1, 2].imshow(np.where(roi_mask, b_star, np.nan), cmap="PuOr_r")
    axes[1, 2].set_title("7. CIELAB b* (Yellow–Blue Axis)\n[Unmodified Chromatic Channel]", fontsize=10, fontweight="bold")
    plt.colorbar(im_b, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im_q = axes[1, 3].imshow(
        np.where(roi_mask, quantized_32, np.nan), cmap="viridis", vmin=0, vmax=31
    )
    axes[1, 3].set_title(
        "8. CLAHE L* (clip=2.0, 8×8)\nQuantized to 32 GLCM Levels",
        fontsize=10,
        fontweight="bold",
    )
    plt.colorbar(im_q, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"M3 Image Preprocessing, ROI Masking & CIELAB/CLAHE Breakdown — {image_name}",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate_single_image_pipeline(
    project_root: Path,
    image_path: Path | str | None = None,
    filename: str | None = None,
    output_dir: Path | None = None,
    rf_model_path: Path | None = None,
    rcnn_model_path: Path | None = None,
    score_threshold: float = 0.50,
) -> dict[str, Any]:
    """Run one individual image through M3 preprocessing, M4 feature extraction, M5 RF, M2 R-CNN, and M7 SHAP."""
    out_root = output_dir or (project_root / "Single_Image_Evaluation")
    out_root.mkdir(parents=True, exist_ok=True)

    auth_labels = load_authoritative_labels_dict(
        project_root / "Classification", verify_disk_images=False
    )
    gt_lesions = load_ground_truth_lesion_counts(project_root)

    resolved_img_path: Path | None = None
    if image_path is not None:
        resolved_img_path = Path(image_path).resolve()
        fname = resolved_img_path.name
    elif filename is not None:
        fname = filename
        candidate = project_root / "Classification" / "JPEGImages" / fname
        if candidate.is_file():
            resolved_img_path = candidate
    else:
        fname = "levle2_93.jpg"
        candidate = project_root / "Classification" / "JPEGImages" / fname
        if candidate.is_file():
            resolved_img_path = candidate

    true_grade = auth_labels.get(fname)
    true_lesion_count = gt_lesions.get(fname)

    m3_candidates = [
        project_root
        / "M3_outputs"
        / "M3_bisenet_operational_final_20260928"
        / "m3_bisenet_operational_manifest.csv",
        project_root
        / "M3_outputs"
        / "M3_bisenet_final"
        / "m3_bisenet_full_manifest.csv",
    ]
    m3_manifest_file = next((p for p in m3_candidates if p.is_file()), None)
    arrays = None
    if m3_manifest_file is not None and fname in auth_labels:
        try:
            m3_df = pd.read_csv(m3_manifest_file)
            npz_p = get_m3_artifact_for_filename(fname, m3_df, project_root)
            arrays = load_m3_output(npz_p)
        except Exception:
            arrays = None

    if arrays is None:
        if resolved_img_path is None or not resolved_img_path.is_file():
            demo_grade = true_grade if true_grade is not None else 2
            resolved_img_path = generate_synthetic_demo_face(
                out_root / f"demo_{Path(fname).stem}.jpg", grade=demo_grade
            )
            print(
                f"[Single-Image Eval] Source JPEG not on disk; synthesized test image -> {resolved_img_path}"
            )
        onnx_path = project_root / "models" / "bisenet_resnet18_celebamaskhq.onnx"
        session = None
        if onnx_path.is_file():
            from m3c_face_parsing import create_face_parsing_session

            session = create_face_parsing_session(onnx_path)
        features, arrays, _m3_res = extract_m4_features_for_single_image(
            resolved_img_path, onnx_session=session
        )
    else:
        features = extract_m4_features_from_arrays(arrays)
        if resolved_img_path is None or not resolved_img_path.is_file():
            resolved_img_path = out_root / f"{Path(fname).stem}_standardized.jpg"
            Image.fromarray(arrays["standardized_rgb"]).save(resolved_img_path)

    stem = Path(fname).stem

    prep_sheet_path = out_root / f"single_image_{stem}_preprocessing_cielab.png"
    render_preprocessing_cielab_sheet(fname, arrays, prep_sheet_path)

    best_rf_pkl = rf_model_path or (
        project_root / "M5_outputs" / "models" / "rf_best_model.pkl"
    )
    if not best_rf_pkl.is_file():
        m5_res = run_m5_random_forest_pipeline(project_root)
        best_rf_pkl = m5_res["best_model_path"]

    with best_rf_pkl.open("rb") as f_in:
        rf_bundle = pickle.load(f_in)
    rf_model = rf_bundle["model"]
    feature_cols: list[str] = rf_bundle["feature_columns"]
    x_vec = np.array([features[c] for c in feature_cols], dtype=np.float64)

    t0_rf = time.perf_counter()
    rf_probs = rf_model.predict_proba(x_vec[np.newaxis, :])[0]
    rf_pred_grade = int(np.argmax(rf_probs))
    rf_latency_ms = (time.perf_counter() - t0_rf) * 1000.0

    rcnn_res = detect_lesions_single_image(
        resolved_img_path,
        roi_mask=arrays["final_roi_mask"],
        score_threshold=score_threshold,
        model_path=rcnn_model_path,
    )
    rcnn_overlay = draw_lesion_detections(
        arrays["standardized_rgb"],
        rcnn_res["boxes"],
        title=f"R-CNN Detections: {rcnn_res['lesion_count']} lesions -> Grade {rcnn_res['predicted_grade']}",
    )

    shap_vals, base_vals = compute_tree_shap_values(rf_model, x_vec[np.newaxis, :])
    waterfall_path = out_root / f"single_image_{stem}_shap_waterfall.png"
    force_path = out_root / f"single_image_{stem}_shap_force.png"
    plot_local_shap_waterfall(
        shap_vals[0, :, rf_pred_grade],
        float(base_vals[rf_pred_grade]),
        x_vec,
        feature_cols,
        target_grade=rf_pred_grade,
        image_name=fname,
        output_path=waterfall_path,
    )
    plot_local_shap_force(
        shap_vals[0, :, rf_pred_grade],
        float(base_vals[rf_pred_grade]),
        x_vec,
        feature_cols,
        target_grade=rf_pred_grade,
        image_name=fname,
        output_path=force_path,
    )

    dashboard_path = out_root / f"single_image_{stem}_evaluation_dashboard.png"
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11))

    axes[0, 0].imshow(rcnn_overlay)
    axes[0, 0].set_title(
        f"Baseline Faster R-CNN (M2)\nDetected Lesions: {rcnn_res['lesion_count']} "
        f"→ Predicted {HAYASHI_GRADE_NAMES[rcnn_res['predicted_grade']]} "
        f"({rcnn_res['inference_latency_ms']:.1f} ms)",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[0, 0].axis("off")

    ax_prob = axes[0, 1]
    grades_x = np.arange(4)
    w_bar = 0.36
    rcnn_probs = [
        rcnn_res["prob_grade_0"],
        rcnn_res["prob_grade_1"],
        rcnn_res["prob_grade_2"],
        rcnn_res["prob_grade_3"],
    ]
    ax_prob.bar(
        grades_x - w_bar / 2,
        rf_probs,
        w_bar,
        label=f"Proposed RF (Pred: G{rf_pred_grade}, {rf_latency_ms:.2f} ms)",
        color="#1f77b4",
    )
    ax_prob.bar(
        grades_x + w_bar / 2,
        rcnn_probs,
        w_bar,
        label=f"Baseline R-CNN (Pred: G{rcnn_res['predicted_grade']}, {rcnn_res['inference_latency_ms']:.1f} ms)",
        color="#ff7f0e",
    )
    if true_grade is not None:
        ax_prob.axvline(
            true_grade,
            color="#2ca02c",
            linestyle="--",
            linewidth=2.0,
            label=f"Authoritative Ground Truth: Grade {true_grade}",
        )
    ax_prob.set_xticks(grades_x)
    ax_prob.set_xticklabels(
        ["Grade 0\n(Mild)", "Grade 1\n(Moderate)", "Grade 2\n(Severe)", "Grade 3\n(Very Severe)"],
        fontsize=9,
    )
    ax_prob.set_ylim(0, 1.08)
    ax_prob.set_ylabel("Predicted Probability", fontsize=9.5)
    ax_prob.set_title(
        "Dual-Model Severity Grade Prediction Comparison",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_prob.legend(loc="upper right", fontsize=8.5)
    ax_prob.grid(axis="y", linestyle="--", alpha=0.4)

    ax_tbl = axes[1, 0]
    ax_tbl.axis("off")
    table_rows = [
        ["a* Mean (Erythema)", f"{features['a_star_mean']:.3f}"],
        ["a* Median", f"{features['a_star_median']:.3f}"],
        ["a* Variance", f"{features['a_star_variance']:.3f}"],
        ["a* 90th Percentile", f"{features['a_star_p90']:.3f}"],
        ["GLCM Contrast (Mean d=1,3,5,10)", f"{features['glcm_contrast_mean']:.4f}"],
        ["GLCM Correlation (Mean d=1,3,5,10)", f"{features['glcm_correlation_mean']:.4f}"],
        ["GLCM Energy (Mean d=1,3,5,10)", f"{features['glcm_energy_mean']:.4f}"],
        ["GLCM Homogeneity (Mean d=1,3,5,10)", f"{features['glcm_homogeneity_mean']:.4f}"],
    ]
    tbl = ax_tbl.table(
        cellText=table_rows,
        colLabels=["Extracted M4 Visual Feature", "Observed Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.62, 0.28],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.55)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f4e78")
            cell.set_text_props(color="white", fontweight="bold")
        elif r <= 4:
            cell.set_facecolor("#fff5f0")
        else:
            cell.set_facecolor("#f7fbff")
    ax_tbl.set_title(
        "M4 Explicit Chromatic (CIELAB a*) & Texture (32-Level GLCM) Features",
        fontsize=10.5,
        fontweight="bold",
    )

    plot_local_shap_waterfall(
        shap_vals[0, :, rf_pred_grade],
        float(base_vals[rf_pred_grade]),
        x_vec,
        feature_cols,
        target_grade=rf_pred_grade,
        image_name=fname,
        output_path=waterfall_path,
        max_display=9,
        ax=axes[1, 1],
    )

    gt_text = (
        f"Authoritative Ground Truth: Grade {true_grade} ({true_lesion_count} lesions)"
        if true_grade is not None
        else "Custom External Image"
    )
    fig.suptitle(
        f"Single-Image End-to-End Pipeline Evaluation — {fname} | {gt_text}",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(dashboard_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "filename": fname,
        "authoritative_grade": true_grade,
        "ground_truth_lesion_count": true_lesion_count,
        "random_forest_m5": {
            "predicted_grade": rf_pred_grade,
            "predicted_grade_label": HAYASHI_GRADE_NAMES[rf_pred_grade],
            "probabilities": {f"grade_{i}": round(float(rf_probs[i]), 4) for i in range(4)},
            "inference_latency_ms": round(rf_latency_ms, 3),
        },
        "faster_rcnn_baseline_m2": {
            "predicted_lesion_count": rcnn_res["lesion_count"],
            "predicted_grade": rcnn_res["predicted_grade"],
            "predicted_grade_label": rcnn_res["predicted_grade_label"],
            "probabilities": {
                "grade_0": round(rcnn_res["prob_grade_0"], 4),
                "grade_1": round(rcnn_res["prob_grade_1"], 4),
                "grade_2": round(rcnn_res["prob_grade_2"], 4),
                "grade_3": round(rcnn_res["prob_grade_3"], 4),
            },
            "inference_latency_ms": rcnn_res["inference_latency_ms"],
            "backend": rcnn_res["backend"],
        },
        "m4_extracted_features": {k: round(float(v), 5) for k, v in features.items()},
        "generated_visual_files": {
            "preprocessing_cielab_sheet": str(prep_sheet_path),
            "evaluation_dashboard": str(dashboard_path),
            "local_shap_waterfall": str(waterfall_path),
            "local_shap_force": str(force_path),
        },
    }
    report_json = out_root / f"single_image_{stem}_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"[Single-Image Eval] {fname}:\n"
        f"  - Ground Truth Grade:     {true_grade}\n"
        f"  - M5 Random Forest Pred:  Grade {rf_pred_grade} ({HAYASHI_GRADE_NAMES[rf_pred_grade]}, {rf_latency_ms:.2f} ms)\n"
        f"  - M2 Faster R-CNN Pred:   Grade {rcnn_res['predicted_grade']} ({rcnn_res['lesion_count']} lesions, {rcnn_res['inference_latency_ms']:.2f} ms)\n"
        f"  - Preprocessing Sheet:    {prep_sheet_path}\n"
        f"  - Evaluation Dashboard:   {dashboard_path}"
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-Image End-to-End Evaluator: Preprocessing, CIELAB/Mask Visualization, "
            "M4 Features, M5 Random Forest vs M2 Faster R-CNN Evaluation, and Local SHAP."
        )
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rf-model", type=Path, default=None)
    parser.add_argument("--rcnn-model", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = find_project_root(args.project_root, require_jpeg_dir=False)
    evaluate_single_image_pipeline(
        project_root=root,
        image_path=args.image,
        filename=args.filename,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        rf_model_path=args.rf_model.resolve() if args.rf_model else None,
        rcnn_model_path=args.rcnn_model.resolve() if args.rcnn_model else None,
        score_threshold=args.score_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
