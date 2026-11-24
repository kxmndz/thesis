#!/usr/bin/env python3
"""Single-Image End-to-End Pipeline & Dual-Model Evaluator (Random Forest vs. Faster R-CNN + Local SHAP).

Strictly evaluates a real image file from the dataset or user-provided path.
Contains zero synthetic face generation.
"""

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

from m2_baseline_rcnn import (
    HAYASHI_GRADE_NAMES,
    detect_lesions_single_image,
    draw_lesion_detections,
    load_ground_truth_lesion_counts,
)
from m3_common import find_project_root, load_authoritative_labels_dict
from m3c_face_parsing import create_face_parsing_session
from m4_feature_extraction import (
    extract_m4_features_for_single_image,
    quantize_to_32_levels,
)
from m7_shap_interpretability import (
    compute_tree_shap_values,
    plot_local_shap_force,
    plot_local_shap_waterfall,
)


def render_preprocessing_cielab_sheet(
    image_name: str,
    arrays: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=140)
    roi = arrays["final_roi_mask"]
    std_rgb = arrays["standardized_rgb"]
    masked_rgb = np.where(roi[..., None], std_rgb, 0)
    q32 = quantize_to_32_levels(arrays["clahe_l_star"], value_range=(0.0, 100.0), levels=32)

    panels = [
        ("1. Standardized 512x512 RGB", std_rgb, None),
        ("2. Facial Skin + Nose ROI Mask", roi.astype(np.uint8) * 255, "gray"),
        ("3. Masked Facial Skin ROI", masked_rgb, None),
        ("4. CIELAB L* (Luminance)", arrays["lab_l_star"], "gray"),
        ("5. CIELAB a* (Erythema / Redness)", arrays["lab_a_star"], "RdYlGn_r"),
        ("6. CIELAB b* (Yellow-Blue)", arrays["lab_b_star"], "viridis"),
        ("7. CLAHE-Enhanced L*", arrays["clahe_l_star"], "gray"),
        ("8. 32-Level Quantized GLCM Input", q32, "magma"),
    ]
    for ax, (title, arr, cmap) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(arr)
        else:
            ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Single-Image Pipeline Preprocessing, Masking & CIELAB Decomposition: {image_name}", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
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
    out_root = output_dir or (project_root / "Single_Image_Evaluation")
    out_root.mkdir(parents=True, exist_ok=True)

    auth_labels = load_authoritative_labels_dict(project_root, include_excluded=True)
    gt_lesions = load_ground_truth_lesion_counts(project_root, include_excluded=True)
    image_dir = project_root / "Classification" / "JPEGImages"

    resolved_img_path: Path | None = None
    if image_path is not None:
        resolved_img_path = Path(image_path).resolve()
        if not resolved_img_path.is_file():
            raise FileNotFoundError(f"Specified image_path does not exist: {resolved_img_path}")
        fname = resolved_img_path.name
    elif filename is not None:
        fname = filename if Path(filename).suffix else f"{filename}.jpg"
        candidate = image_dir / fname
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Requested image '{fname}' was not found in {image_dir}. "
                f"Choose a training-eligible image filename from Layered-Labs/ACNE04."
            )
        resolved_img_path = candidate
    else:
        available_jpegs = sorted(image_dir.glob("*.jpg")) if image_dir.is_dir() else []
        if not available_jpegs:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}.")
        default_cand = image_dir / "levle2_93.jpg"
        resolved_img_path = default_cand if default_cand.is_file() else available_jpegs[0]
        fname = resolved_img_path.name

    true_grade = auth_labels.get(fname)
    true_lesion_count = gt_lesions.get(fname)

    onnx_path = project_root / "models" / "bisenet_resnet18_celebamaskhq.onnx"
    session = (
        create_face_parsing_session(onnx_path, verify_sha256=False)
        if onnx_path.is_file()
        else None
    )

    features, arrays, _m3_res = extract_m4_features_for_single_image(
        resolved_img_path, onnx_session=session
    )
    stem = Path(fname).stem
    prep_sheet_path = out_root / f"single_image_{stem}_preprocessing_cielab.png"
    render_preprocessing_cielab_sheet(fname, arrays, prep_sheet_path)

    best_rf_pkl = rf_model_path or (project_root / "M5_outputs" / "models" / "rf_best_model.pkl")
    if not best_rf_pkl.is_file():
        raise FileNotFoundError(f"Missing trained Random Forest model at {best_rf_pkl}. Run M5 first.")

    with best_rf_pkl.open("rb") as f_in:
        rf_bundle = pickle.load(f_in)
    rf_model = rf_bundle["model"]
    feature_cols: list[str] = rf_bundle["feature_columns"]
    x_vec = np.array([features[c] for c in feature_cols], dtype=np.float64)

    t0_rf = time.perf_counter()
    rf_probs = rf_model.predict_proba(x_vec[np.newaxis, :])[0]
    rf_pred_grade = int(np.argmax(rf_probs))
    rf_latency_ms = (time.perf_counter() - t0_rf) * 1000.0

    if rcnn_model_path is None:
        ckpt_dir = project_root / "M2_outputs" / "checkpoints"
        best_ckpts = sorted(ckpt_dir.glob("*_best.pt")) if ckpt_dir.is_dir() else []
        if best_ckpts:
            rcnn_model_path = best_ckpts[0]
        else:
            pb_candidates = list((project_root / "models").rglob("*.pb")) if (project_root / "models").is_dir() else []
            if pb_candidates:
                rcnn_model_path = pb_candidates[0]

    rcnn_res = detect_lesions_single_image(
        resolved_img_path,
        roi_mask=arrays["final_roi_mask"],
        score_threshold=score_threshold,
        model_path=rcnn_model_path,
    )
    rcnn_overlay = draw_lesion_detections(
        arrays["standardized_rgb"],
        rcnn_res["boxes"],
        title=f"R-CNN: {rcnn_res['lesion_count']} lesions -> Grade {rcnn_res['predicted_grade']}",
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
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140)
    axes[0].imshow(rcnn_overlay)
    axes[0].set_title(f"M2 Baseline Faster R-CNN\nCount={rcnn_res['lesion_count']} -> Grade {rcnn_res['predicted_grade']}")
    axes[0].axis("off")

    axes[1].bar(["Grade 0", "Grade 1", "Grade 2", "Grade 3"], rf_probs, color=["#4caf50", "#ffb300", "#fb8c00", "#e53935"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"M5 Random Forest Probabilities\nPredicted: Grade {rf_pred_grade} ({HAYASHI_GRADE_NAMES[rf_pred_grade]})")

    plot_local_shap_waterfall(
        shap_vals[0, :, rf_pred_grade],
        float(base_vals[rf_pred_grade]),
        x_vec,
        feature_cols,
        target_grade=rf_pred_grade,
        image_name=fname,
        output_path=waterfall_path,
        max_display=8,
        ax=axes[2],
    )
    fig.tight_layout()
    fig.savefig(dashboard_path, bbox_inches="tight")
    plt.close(fig)

    report = {
        "filename": fname,
        "authoritative_grade": true_grade,
        "ground_truth_lesion_count": true_lesion_count,
        "random_forest_m5": {
            "predicted_grade": rf_pred_grade,
            "predicted_grade_label": HAYASHI_GRADE_NAMES[rf_pred_grade],
            "inference_latency_ms": round(rf_latency_ms, 3),
        },
        "faster_rcnn_baseline_m2": {
            "predicted_lesion_count": rcnn_res["lesion_count"],
            "predicted_grade": rcnn_res["predicted_grade"],
            "inference_latency_ms": rcnn_res["inference_latency_ms"],
        },
        "generated_visual_files": {
            "preprocessing_cielab_sheet": str(prep_sheet_path),
            "evaluation_dashboard": str(dashboard_path),
            "local_shap_waterfall": str(waterfall_path),
            "local_shap_force": str(force_path),
        },
    }
    (out_root / f"single_image_{stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-Image End-to-End Evaluator")
    parser.add_argument("--project-root", "--root-dir", dest="project_root", type=Path, default=None)
    parser.add_argument("--image", "--image-path", dest="image", type=Path, default=None)
    parser.add_argument("--filename", "--image-id", dest="filename", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rf-model", type=Path, default=None)
    parser.add_argument("--rcnn-model", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = find_project_root(args.project_root, require_jpeg_dir=True)
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
