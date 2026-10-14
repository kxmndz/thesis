#!/usr/bin/env python3
"""Unified CLI Pipeline for the Complete M1–M7 Acne Severity Grading System.

Integrates all modules of the thesis architecture (`System_Architecture_Diagram_CANVA.pdf.pdf`):
- M1 (`m1_data_loader.py`):          Stratified 5x2 Cross-Validation Splitter & Dataset Manifest
- M2 (`m2_baseline_rcnn.py`):        Baseline Lesion-Counting Module (Wen et al., 2022 Faster R-CNN ResNet-101)
                                     with pretrained download or checkpointed 5x2 retraining
- M3 (`m3_common.py`, `m3a...`,      Image Preprocessing Module (512x512 Standardization, ROI Masking,
      `m3b...`, `m3c...`,            Objective Mask Validation, CIELAB + Grayscale, L*-only CLAHE)
      `m3_bisenet_pipeline.py`):
- M4 (`m4_feature_extraction.py`):   Chromatic (CIELAB a*: mean, median, variance, 90th pctl) &
                                     Texture (Masked 32-level GLCM: contrast, correlation, energy, homogeneity)
- M5 (`m5_random_forest.py`):        Random Forest Classifier (Balanced Class Weights + Nested CV GridSearch)
- M6 (`m6_model_evaluation.py`):     Model Evaluation (Precision, Recall, Macro F1, Confusion Matrix,
                                     OvO AUC-ROC, and Alpaydın 1999 Combined 5x2 CV F-Test)
- M7 (`m7_shap_interpretability.py`):SHAP TreeExplainer Global (Bar & Directional Beeswarm) and
                                     Local (Per-Image Waterfall & Force) Plots
- Single-Image Evaluator (`evaluate_single_image.py`):
                                     End-to-end single-image preprocessing, CIELAB/masking visualization,
                                     Random Forest + Faster R-CNN evaluation, and local SHAP explanation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_single_image import evaluate_single_image_pipeline
from m1_data_loader import run_m1_pipeline
from m2_baseline_rcnn import (
    WENH06_PRETRAINED_MODELS,
    setup_wenh06_repo_and_pretrained,
    train_or_evaluate_m2_5x2,
)
from m3_bisenet_pipeline import (
    DEFAULT_REPRODUCTION_RUN_ID,
    process_operational_dataset,
    run_bisenet_pilot,
    run_fixed_parity,
    verify_m4_handoff,
)
from m3_common import find_project_root
from m3a_mediapipe_audit import (
    run_dimension_audit,
    run_initial_mask_audit,
    run_landmark_audit,
)
from m3b_skin_retention import export_m3b_dataset, run_m3_mediapipe_pipeline
from m3c_face_parsing import (
    export_m3c_dataset,
    run_m3c_expanded_validation,
    run_m3c_pilot_comparison,
    summarize_m3c_manual_ratings,
)
from m4_feature_extraction import DEFAULT_GLCM_DISTANCES, run_m4_feature_extraction
from m5_random_forest import run_m5_random_forest_pipeline
from m6_model_evaluation import run_m6_evaluation
from m7_shap_interpretability import run_m7_shap_analysis


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline_cli.py",
        description=(
            "Complete M1–M7 Acne Severity Grading CLI Pipeline: Data Partitioning (M1), "
            "Faster R-CNN Baseline (M2), Image Preprocessing (M3), Chromatic+GLCM Feature "
            "Extraction (M4), Random Forest Training (M5), Statistical Evaluation & 5x2 F-Test (M6), "
            "SHAP TreeExplainer Interpretability (M7), and Single-Image Evaluator."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. End-to-End Full M1 -> M7 Pipeline
    p_all = subparsers.add_parser(
        "run-pipeline",
        help="Run the complete M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7 pipeline end-to-end.",
    )
    p_all.add_argument("--project-root", type=Path, default=None)
    p_all.add_argument("--run-id", type=str, default=DEFAULT_REPRODUCTION_RUN_ID)
    p_all.add_argument(
        "--m2-mode",
        choices=("infer-pretrained", "train-5x2"),
        default="infer-pretrained",
        help="Whether M2 uses pretrained Faster R-CNN weights or retrains across the 5x2 split.",
    )
    p_all.add_argument("--skip-m3", action="store_true")
    p_all.add_argument("--overwrite", action="store_true")

    # 2. M1 Data Partitioning
    p_m1 = subparsers.add_parser(
        "m1",
        help="M1: Data Partitioning and Stratified 5x2 Cross-Validation Splitter.",
    )
    p_m1.add_argument("--project-root", type=Path, default=None)
    p_m1.add_argument("--output-dir", type=Path, default=None)
    p_m1.add_argument("--skip-disk-jpeg-check", action="store_true")
    p_m1.add_argument("--overwrite", action="store_true")
    p_m1.add_argument("--seed", type=int, default=42)

    # 3. M2 Baseline Faster R-CNN Lesion-Counting Module
    p_m2 = subparsers.add_parser(
        "m2",
        help="M2: Baseline Lesion-Counting Faster R-CNN ResNet-101 (download pretrained or checkpointed 5x2 retraining).",
    )
    p_m2.add_argument("--project-root", type=Path, default=None)
    p_m2.add_argument(
        "--mode",
        choices=("download-pretrained", "infer-pretrained", "train-5x2"),
        default="infer-pretrained",
    )
    p_m2.add_argument(
        "--model-key",
        choices=tuple(WENH06_PRETRAINED_MODELS.keys()),
        default="faster_rcnn_resnet101",
    )
    p_m2.add_argument("--pretrained-model-path", type=Path, default=None)
    p_m2.add_argument("--checkpoint-dir", type=Path, default=None)
    p_m2.add_argument("--output-dir", type=Path, default=None)
    p_m2.add_argument("--epochs", type=int, default=10)
    p_m2.add_argument("--batch-size", type=int, default=2)
    p_m2.add_argument("--lr", type=float, default=0.002)
    p_m2.add_argument("--score-threshold", type=float, default=0.50)
    p_m2.add_argument("--configs", type=int, nargs="*", default=None)
    p_m2.add_argument("--no-resume", action="store_true")

    # 4. M3A MediaPipe Audits
    p_m3a = subparsers.add_parser(
        "m3a-audit",
        help="M3A: Image dimension, MediaPipe segmentation, and Face Landmarker audits.",
    )
    p_m3a.add_argument("--project-root", type=Path, default=None)
    p_m3a.add_argument("--output-dir", type=Path, default=None)
    p_m3a.add_argument(
        "--mode",
        choices=("dimensions", "mask-audit", "landmark-audit", "all"),
        default="all",
    )

    # 5. M3B Skin Retention & MediaPipe Final Baseline
    p_m3b = subparsers.add_parser(
        "m3b",
        help="M3B: Skin-retention mask export or frozen M3 MediaPipe+Landmark NPZ pipeline.",
    )
    p_m3b.add_argument("--project-root", type=Path, default=None)
    p_m3b.add_argument("--output-dir", type=Path, default=None)
    p_m3b.add_argument(
        "--mode",
        choices=("m3b-png-export", "mediapipe-pilot", "mediapipe-full"),
        default="mediapipe-full",
    )

    # 6. M3C Face-Parsing Alternative & Expanded Validation
    p_m3c = subparsers.add_parser(
        "m3c",
        help="M3C: 19-class BiSeNet 11-image pilot, 40-image stratified validation, or PNG export.",
    )
    p_m3c.add_argument("--project-root", type=Path, default=None)
    p_m3c.add_argument("--output-dir", type=Path, default=None)
    p_m3c.add_argument(
        "--mode",
        choices=("pilot-11", "expanded-40", "summarize-ratings", "export-pngs"),
        default="expanded-40",
    )
    p_m3c.add_argument("--rated-csv", type=Path, default=None)

    # 7. M3 Final BiSeNet Preprocessing
    p_bisenet = subparsers.add_parser(
        "m3-bisenet",
        help="M3: Final BiSeNet preprocessing (13-image pilot, 1,457-image operational run, or 7-image parity).",
    )
    p_bisenet.add_argument("--project-root", type=Path, default=None)
    p_bisenet.add_argument(
        "--mode",
        choices=("pilot", "operational", "parity-7"),
        default="operational",
    )
    p_bisenet.add_argument("--run-id", type=str, default=DEFAULT_REPRODUCTION_RUN_ID)
    p_bisenet.add_argument("--output-dir", type=Path, default=None)
    p_bisenet.add_argument("--m3-manifest", type=Path, default=None)
    p_bisenet.add_argument("--overwrite", action="store_true")

    # 8. M4 Handoff Verification
    p_handoff = subparsers.add_parser(
        "m4-handoff",
        help="M3->M4: Verify M1->M3->M4 filename resolution, 11-key NPZ contract, and 14,570 fold memberships.",
    )
    p_handoff.add_argument("--project-root", type=Path, default=None)
    p_handoff.add_argument("--m3-manifest", type=Path, required=True)
    p_handoff.add_argument("--m1-manifest", type=Path, default=None)
    p_handoff.add_argument("--m1-folds", type=Path, default=None)

    # 9. M4 Feature Extraction
    p_m4 = subparsers.add_parser(
        "m4",
        help="M4: Extract Chromatic (CIELAB a*) and Texture (Masked 32-level GLCM) features.",
    )
    p_m4.add_argument("--project-root", type=Path, default=None)
    p_m4.add_argument("--m3-manifest", type=Path, default=None)
    p_m4.add_argument("--output-dir", type=Path, default=None)
    p_m4.add_argument(
        "--texture-source",
        choices=("clahe_l_star", "grayscale", "lab_l_star"),
        default="clahe_l_star",
    )
    p_m4.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=list(DEFAULT_GLCM_DISTANCES),
    )

    # 10. M5 Random Forest Classifier Training
    p_m5 = subparsers.add_parser(
        "m5",
        help="M5: Train Random Forest with balanced class weights and Nested CV across 10 5x2 folds.",
    )
    p_m5.add_argument("--project-root", type=Path, default=None)
    p_m5.add_argument("--feature-matrix", type=Path, default=None)
    p_m5.add_argument("--output-dir", type=Path, default=None)
    p_m5.add_argument("--checkpoint-dir", type=Path, default=None)
    p_m5.add_argument("--inner-folds", type=int, default=5)
    p_m5.add_argument("--no-resume", action="store_true")

    # 11. M6 Model Evaluation & Combined 5x2 CV F-Test
    p_m6 = subparsers.add_parser(
        "m6",
        help="M6: Evaluate Random Forest & Baseline Faster R-CNN and run Combined 5x2 CV F-Test.",
    )
    p_m6.add_argument("--project-root", type=Path, default=None)
    p_m6.add_argument("--rf-predictions", type=Path, default=None)
    p_m6.add_argument("--rcnn-predictions", type=Path, default=None)
    p_m6.add_argument("--output-dir", type=Path, default=None)

    # 12. M7 SHAP TreeExplainer Interpretability
    p_m7 = subparsers.add_parser(
        "m7",
        help="M7: Run SHAP TreeExplainer global (bar + beeswarm) and local (waterfall + force) plots.",
    )
    p_m7.add_argument("--project-root", type=Path, default=None)
    p_m7.add_argument("--best-model", type=Path, default=None)
    p_m7.add_argument("--feature-matrix", type=Path, default=None)
    p_m7.add_argument("--output-dir", type=Path, default=None)
    p_m7.add_argument("--filenames", type=str, nargs="*", default=None)

    # 13. Single-Image End-to-End Evaluator
    p_eval_img = subparsers.add_parser(
        "evaluate-image",
        help="Run one individual image through preprocessing, CIELAB/masking visualization, RF + R-CNN, and local SHAP.",
    )
    p_eval_img.add_argument("--project-root", type=Path, default=None)
    p_eval_img.add_argument("--image", type=Path, default=None)
    p_eval_img.add_argument("--filename", type=str, default=None)
    p_eval_img.add_argument("--output-dir", type=Path, default=None)
    p_eval_img.add_argument("--rf-model", type=Path, default=None)
    p_eval_img.add_argument("--rcnn-model", type=Path, default=None)
    p_eval_img.add_argument("--score-threshold", type=float, default=0.50)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "run-pipeline":
        root = find_project_root(args.project_root, require_jpeg_dir=False)
        has_jpegs = (root / "Classification" / "JPEGImages").is_dir()
        print("=== [Stage 1/6] M1 Data Partitioning & Stratified 5x2 CV ===")
        run_m1_pipeline(
            project_root=root,
            verify_disk_jpegs=has_jpegs,
            overwrite=args.overwrite,
        )

        if not args.skip_m3 and has_jpegs and (root / "models" / "bisenet_resnet18_celebamaskhq.onnx").is_file():
            print("\n=== [Stage 2/6] M3 BiSeNet Operational Preprocessing ===")
            process_operational_dataset(
                project_root=root,
                run_id=args.run_id,
                overwrite=args.overwrite,
            )
        else:
            print("\n=== [Stage 2/6] Skipping M3 BiSeNet full inference (using existing M3 artifacts or direct extraction) ===")

        print("\n=== [Stage 3/6] M2 Baseline Faster R-CNN Lesion-Counting Evaluation ===")
        train_or_evaluate_m2_5x2(project_root=root, mode=args.m2_mode)

        print("\n=== [Stage 4/6] M4 Chromatic (CIELAB a*) & Texture (32-Level GLCM) Feature Extraction ===")
        run_m4_feature_extraction(project_root=root)

        print("\n=== [Stage 5/6] M5 Random Forest Training (Balanced Weights + Nested CV) ===")
        run_m5_random_forest_pipeline(project_root=root)

        print("\n=== [Stage 6/6] M6 Model Evaluation (5x2 F-Test) & M7 SHAP TreeExplainer ===")
        run_m6_evaluation(project_root=root)
        run_m7_shap_analysis(project_root=root)
        return 0

    if args.command == "m1":
        run_m1_pipeline(
            project_root=args.project_root,
            output_dir=args.output_dir,
            verify_disk_jpegs=not args.skip_disk_jpeg_check,
            overwrite=args.overwrite,
            random_state=args.seed,
        )
        return 0

    if args.command == "m2":
        root = find_project_root(args.project_root, require_jpeg_dir=False)
        if args.mode == "download-pretrained":
            status = setup_wenh06_repo_and_pretrained(
                root, model_key=args.model_key, attempt_download=True
            )
            print(json.dumps(status, indent=2))
            return 0
        train_or_evaluate_m2_5x2(
            project_root=root,
            mode=args.mode,
            checkpoint_dir=args.checkpoint_dir.resolve() if args.checkpoint_dir else None,
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            epochs_per_fold=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            score_threshold=args.score_threshold,
            pretrained_model_path=args.pretrained_model_path,
            configs_to_run=args.configs,
            resume=not args.no_resume,
        )
        return 0

    if args.command == "m3a-audit":
        root = find_project_root(args.project_root, require_jpeg_dir=True)
        out_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else (root / "M3_outputs" / "M3A_audit")
        )
        if args.mode in ("dimensions", "all"):
            run_dimension_audit(root / "Classification" / "JPEGImages", out_dir)
        mask_df = None
        if args.mode in ("mask-audit", "all"):
            mask_df = run_initial_mask_audit(root, out_dir)
        if args.mode in ("landmark-audit", "all"):
            run_landmark_audit(root, initial_mask_audit=mask_df, output_dir=out_dir)
        return 0

    if args.command == "m3b":
        root = find_project_root(args.project_root, require_jpeg_dir=True)
        out_dir = args.output_dir.resolve() if args.output_dir else None
        if args.mode == "m3b-png-export":
            export_m3b_dataset(root, out_dir)
        elif args.mode == "mediapipe-pilot":
            run_m3_mediapipe_pipeline(root, mode="pilot", output_root=out_dir)
        else:
            run_m3_mediapipe_pipeline(root, mode="full", output_root=out_dir)
        return 0

    if args.command == "m3c":
        root = find_project_root(
            args.project_root,
            require_jpeg_dir=(args.mode != "summarize-ratings"),
        )
        out_dir = args.output_dir.resolve() if args.output_dir else None
        if args.mode == "pilot-11":
            run_m3c_pilot_comparison(root, out_dir)
        elif args.mode == "expanded-40":
            run_m3c_expanded_validation(root, out_dir)
        elif args.mode == "summarize-ratings":
            csv_path = args.rated_csv or (
                root
                / "M3_outputs"
                / "M3C_expanded_validation"
                / "m3c_expanded_validation_review.csv"
            )
            summarize_m3c_manual_ratings(csv_path)
        elif args.mode == "export-pngs":
            export_m3c_dataset(root, out_dir)
        return 0

    if args.command == "m3-bisenet":
        root = find_project_root(args.project_root, require_jpeg_dir=True)
        out_dir = args.output_dir.resolve() if args.output_dir else None
        if args.mode == "pilot":
            run_bisenet_pilot(root, output_root=out_dir, overwrite=args.overwrite)
        elif args.mode == "operational":
            process_operational_dataset(
                root,
                run_id=args.run_id,
                output_root=out_dir,
                overwrite=args.overwrite,
            )
        elif args.mode == "parity-7":
            ref_manifest = args.m3_manifest or (
                root
                / "M3_outputs"
                / "M3_bisenet_final"
                / "m3_bisenet_full_manifest.csv"
            )
            run_fixed_parity(root, ref_manifest)
        return 0

    if args.command == "m4-handoff":
        root = find_project_root(args.project_root, require_jpeg_dir=False)
        verify_m4_handoff(
            project_root=root,
            m3_manifest_path=args.m3_manifest.resolve(),
            m1_manifest_path=args.m1_manifest.resolve() if args.m1_manifest else None,
            m1_folds_path=args.m1_folds.resolve() if args.m1_folds else None,
        )
        return 0

    if args.command == "m4":
        root = find_project_root(args.project_root, require_jpeg_dir=False)
        run_m4_feature_extraction(
            project_root=root,
            m3_manifest_path=args.m3_manifest.resolve() if args.m3_manifest else None,
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            texture_source=args.texture_source,
            distances=tuple(args.distances),
        )
        return 0

    if args.command == "m5":
        root = find_project_root(args.project_root, require_jpeg_dir=False)
        run_m5_random_forest_pipeline(
            project_root=root,
            feature_matrix_path=(
                args.feature_matrix.resolve() if args.feature_matrix else None
            ),
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            checkpoint_dir=(
                args.checkpoint_dir.resolve() if args.checkpoint_dir else None
            ),
            inner_cv_folds=args.inner_folds,
            resume=not args.no_resume,
        )
        return 0

    if args.command == "m6":
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

    if args.command == "m7":
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

    if args.command == "evaluate-image":
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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
