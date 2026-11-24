#!/usr/bin/env python3
"""Unified CLI Pipeline for the Complete M1–M7 Acne Severity Grading System (Layered-Labs/ACNE04)."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_single_image import evaluate_single_image_pipeline
from m1_data_loader import DEFAULT_HF_DATASET_ID, run_m1_pipeline
from m2_baseline_rcnn import train_or_evaluate_m2_5x2
from m3_common import find_project_root
from m4_feature_extraction import run_m4_feature_extraction
from m5_random_forest import run_m5_random_forest_pipeline
from m6_model_evaluation import run_m6_evaluation
from m7_shap_interpretability import run_m7_shap_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline_cli.py")
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ("run-pipeline", "m1", "m2", "m4", "m5", "m6", "m7", "evaluate-image"):
        p = sub.add_parser(cmd)
        p.add_argument("--project-root", "--root-dir", dest="project_root", type=Path, default=None)
        if cmd in ("m1", "run-pipeline"):
            p.add_argument("--dataset-id", type=str, default=DEFAULT_HF_DATASET_ID)
            p.add_argument("--parquet-path", type=Path, default=None)
            p.add_argument("--target-size", type=int, default=512)
    args, _ = parser.parse_known_args(argv)
    root = find_project_root(args.project_root, require_jpeg_dir=False)
    if args.command == "m1":
        run_m1_pipeline(
            project_root=root,
            dataset_id=args.dataset_id,
            parquet_path=args.parquet_path,
            target_size=args.target_size,
        )
    elif args.command == "m2":
        train_or_evaluate_m2_5x2(project_root=root)
    elif args.command == "m4":
        run_m4_feature_extraction(project_root=root)
    elif args.command == "m5":
        run_m5_random_forest_pipeline(project_root=root)
    elif args.command == "m6":
        run_m6_evaluation(project_root=root)
    elif args.command == "m7":
        run_m7_shap_analysis(project_root=root)
    elif args.command == "evaluate-image":
        evaluate_single_image_pipeline(project_root=root)
    elif args.command == "run-pipeline":
        run_m1_pipeline(
            project_root=root,
            dataset_id=args.dataset_id,
            parquet_path=args.parquet_path,
            target_size=args.target_size,
        )
        train_or_evaluate_m2_5x2(project_root=root)
        run_m4_feature_extraction(project_root=root)
        run_m5_random_forest_pipeline(project_root=root)
        run_m6_evaluation(project_root=root)
        run_m7_shap_analysis(project_root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
