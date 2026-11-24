#!/usr/bin/env python3
"""M5 – Balanced Random Forest Classifier Training with Nested CV Grid Search,
Live Progress Logging, and Two-Tier (Outer Fold + Inner Candidate) Checkpointing.

Uses scikit-learn's `RandomForestClassifier` (with balanced class weights), `StratifiedKFold`,
and `ParameterGrid` directly with zero synthetic or custom tree fallbacks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from m1_data_loader import assert_no_watermarked_filenames
from m3_common import find_project_root
from m4_feature_extraction import get_feature_columns

LOGGER = logging.getLogger(__name__)


def _safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa across Hayashi grades 0..3 safely."""
    val = float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3], weights="quadratic"))
    return 0.0 if np.isnan(val) else val


DEFAULT_RF_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [300],
    "max_depth": [6, 10, 15, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": ["balanced", "balanced_subsample"],
    "criterion": ["gini", "entropy"],
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so Colab/Drive disconnects never corrupt checkpoint files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        tmp_path.replace(path)
    except PermissionError:
        if path.exists():
            path.unlink()
        tmp_path.replace(path)


def _canonical_param_key(params: dict[str, Any]) -> str:
    """Deterministic JSON string key for a Random Forest hyperparameter dictionary."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def fit_nested_cv_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_folds: int = 3,
    random_state: int = 42,
    param_grid: dict[str, list[Any]] | None = None,
    config_idx: int | None = None,
    total_configs: int = 10,
    log_every_candidate: bool = False,
    grid_checkpoint_path: Path | None = None,
    resume: bool = True,
    grid_checkpoint_every: int = 10,
) -> tuple[RandomForestClassifier, dict[str, Any], float]:
    """Run inner stratified CV grid search strictly on X_train, y_train optimizing Macro F1.

    Supports candidate-level checkpointing via `grid_checkpoint_path` so disconnects
    mid-fold never lose completed hyperparameter evaluations, while displaying a live
    `tqdm` progress bar and emitting timestamped `logging.info` updates.
    """
    from tqdm.auto import tqdm

    active_grid = param_grid if param_grid is not None else DEFAULT_RF_PARAM_GRID
    candidates = list(ParameterGrid(active_grid))
    if not candidates:
        raise ValueError("Random Forest hyperparameter grid produced 0 candidate configurations.")

    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    splits = list(skf.split(X_train, y_train))

    prefix = (
        f"M5 Config {config_idx:02d}/{total_configs:02d}"
        if config_idx is not None
        else "M5 Nested CV"
    )

    evaluated: dict[str, dict[str, Any]] = {}
    if grid_checkpoint_path is not None and resume and grid_checkpoint_path.is_file():
        try:
            ckpt = json.loads(grid_checkpoint_path.read_text(encoding="utf-8"))
            if (
                int(ckpt.get("inner_folds", -1)) == int(inner_folds)
                and int(ckpt.get("random_state", -1)) == int(random_state)
                and int(ckpt.get("n_train_samples", -1)) == int(len(X_train))
                and int(ckpt.get("n_features", -1)) == int(X_train.shape[1])
                and isinstance(ckpt.get("evaluated"), dict)
            ):
                evaluated = dict(ckpt["evaluated"])
        except Exception as exc:
            LOGGER.warning(
                "[%s] Ignoring unreadable grid checkpoint %s (%s); starting fresh.",
                prefix,
                grid_checkpoint_path.name,
                exc,
            )

    best_score = -1.0
    best_params: dict[str, Any] = dict(candidates[0])
    pre_completed = 0
    for params in candidates:
        key = _canonical_param_key(params)
        if key in evaluated:
            pre_completed += 1
            sc = float(evaluated[key]["inner_cv_macro_f1"])
            if sc > best_score:
                best_score = sc
                best_params = dict(params)

    if pre_completed > 0:
        LOGGER.info(
            "[%s] Resuming inner %d-fold Grid Search: %d/%d candidates loaded from checkpoint "
            "(%d remaining) | current best InnerCV Macro-F1=%.4f",
            prefix,
            inner_folds,
            pre_completed,
            len(candidates),
            len(candidates) - pre_completed,
            best_score,
        )
    else:
        LOGGER.info(
            "[%s] Starting inner %d-fold Grid Search (%d hyperparameter candidates)...",
            prefix,
            inner_folds,
            len(candidates),
        )

    def _save_grid_ckpt() -> None:
        if grid_checkpoint_path is None:
            return
        _atomic_write_json(
            grid_checkpoint_path,
            {
                "config_idx": config_idx,
                "inner_folds": int(inner_folds),
                "random_state": int(random_state),
                "n_train_samples": int(len(X_train)),
                "n_features": int(X_train.shape[1]),
                "completed_candidates": int(len(evaluated)),
                "best_score": round(float(best_score), 6),
                "best_params": best_params,
                "updated_at": datetime.now().isoformat(),
                "evaluated": evaluated,
            },
        )

    newly_evaluated = 0
    resumed_seen = 0
    ckpt_interval = max(1, int(grid_checkpoint_every))

    pbar = tqdm(
        enumerate(candidates, start=1),
        total=len(candidates),
        desc=f"{prefix} GridSearch",
        leave=True,
    )
    for step, params in pbar:
        key = _canonical_param_key(params)
        if key in evaluated:
            mean_score = float(evaluated[key]["inner_cv_macro_f1"])
            resumed_seen += 1
            pbar.set_postfix(
                cur_f1=f"{mean_score:.4f}",
                best_f1=f"{best_score:.4f}",
                resumed=resumed_seen,
            )
            continue

        fold_scores: list[float] = []
        for tr_idx, val_idx in splits:
            clf = RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
                **params,
            )
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            preds = clf.predict(X_train[val_idx])
            fold_scores.append(
                float(f1_score(y_train[val_idx], preds, average="macro", zero_division=0))
            )

        mean_score = float(np.mean(fold_scores))
        evaluated[key] = {
            "params": dict(params),
            "inner_cv_macro_f1": round(mean_score, 6),
            "fold_scores": [round(s, 6) for s in fold_scores],
        }
        newly_evaluated += 1

        is_new_best = mean_score > best_score
        if is_new_best:
            best_score = mean_score
            best_params = dict(params)
            LOGGER.info(
                "[%s] Step %d/%d NEW BEST InnerCV Macro-F1=%.4f | params=%s",
                prefix,
                step,
                len(candidates),
                best_score,
                best_params,
            )
            _save_grid_ckpt()
        else:
            if log_every_candidate:
                LOGGER.info(
                    "[%s] Step %d/%d InnerCV Macro-F1=%.4f | best=%.4f | params=%s",
                    prefix,
                    step,
                    len(candidates),
                    mean_score,
                    best_score,
                    dict(params),
                )
            if newly_evaluated % ckpt_interval == 0:
                _save_grid_ckpt()

        pbar.set_postfix(
            cur_f1=f"{mean_score:.4f}",
            best_f1=f"{best_score:.4f}",
            resumed=resumed_seen,
        )

    if newly_evaluated > 0 or (grid_checkpoint_path is not None and not grid_checkpoint_path.is_file()):
        _save_grid_ckpt()

    LOGGER.info(
        "[%s] Refitting final Random Forest on full outer training fold | best_params=%s",
        prefix,
        best_params,
    )
    final_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(X_train, y_train)
    return final_model, best_params, best_score


def run_m5_random_forest_pipeline(
    project_root: Path,
    feature_matrix_path: Path | None = None,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    inner_cv_folds: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
    log_every_candidate: bool = False,
    grid_checkpoint_every: int = 10,
    resume: bool = True,
) -> dict[str, Any]:
    """Train and evaluate the M5 Balanced Random Forest across all 10 5x2 CV configurations."""
    m1_folds_path = project_root / "M1_outputs" / "m1_5x2_fold_assignments.csv"
    if not m1_folds_path.is_file():
        raise FileNotFoundError(
            f"Missing M1 fold assignments: {m1_folds_path}. Run M1 first."
        )

    feat_path = feature_matrix_path or (project_root / "M4_outputs" / "m4_feature_matrix.csv")
    if not feat_path.is_file():
        raise FileNotFoundError(
            f"Missing M4 feature matrix: {feat_path}. Run M4 feature extraction first."
        )
    feature_df = pd.read_csv(feat_path)
    folds_df = pd.read_csv(m1_folds_path)
    assert_no_watermarked_filenames(feature_df["filename"], context="M5 feature matrix")
    assert_no_watermarked_filenames(folds_df["filename"], context="M5 5x2 CV folds")
    feature_cols = get_feature_columns()
    feat_indexed = feature_df.set_index("filename")

    m5_out = output_dir or (project_root / "M5_outputs")
    ckpt_root = checkpoint_dir or (m5_out / "checkpoints")
    models_dir = m5_out / "models"
    preds_dir = m5_out / "predictions"
    for d in (m5_out, ckpt_root, models_dir, preds_dir):
        d.mkdir(parents=True, exist_ok=True)

    state_path = ckpt_root / "m5_training_state.json"
    schema_compatible = False
    state: dict[str, Any] = {}
    if resume and state_path.is_file():
        loaded_state = json.loads(state_path.read_text(encoding="utf-8"))
        saved_cols = loaded_state.get("feature_columns")
        if saved_cols == feature_cols:
            state = loaded_state
            schema_compatible = True
        else:
            LOGGER.warning(
                "[M5] Existing checkpoint %s was trained on %s features (current schema has %d features); "
                "invalidating stale M5 checkpoints and retraining cleanly.",
                state_path.name,
                len(saved_cols) if isinstance(saved_cols, list) else "legacy/unknown",
                len(feature_cols),
            )

    if not schema_compatible:
        for stale_ckpt in ckpt_root.glob("m5_config_*_grid_checkpoint.json"):
            stale_ckpt.unlink(missing_ok=True)
        state = {
            "n_features": int(len(feature_cols)),
            "feature_columns": list(feature_cols),
            "completed_configurations": [],
            "completed_configs": {},
            "best_configuration": None,
            "best_macro_f1": -1.0,
        }

    all_preds: list[pd.DataFrame] = []
    sorted_configs = [int(c) for c in sorted(folds_df["configuration"].unique())]
    total_configs = len(sorted_configs)

    for config_idx in sorted_configs:
        pred_csv = preds_dir / f"rf_config_{config_idx:02d}_predictions.csv"
        if resume and config_idx in state.get("completed_configurations", []) and pred_csv.is_file():
            LOGGER.info(
                "[M5] Configuration %02d/%02d already completed in checkpoint; loading %s",
                config_idx,
                total_configs,
                pred_csv.name,
            )
            resumed_df = pd.read_csv(pred_csv)
            cfg_meta = state.get("completed_configs", {}).get(str(config_idx))
            if isinstance(cfg_meta, dict) and "test_qwk" not in cfg_meta:
                cfg_meta["test_qwk"] = round(
                    _safe_qwk(
                        resumed_df["authoritative_grade"].to_numpy(dtype=int),
                        resumed_df["predicted_grade"].to_numpy(dtype=int),
                    ),
                    4,
                )
            all_preds.append(resumed_df)
            continue

        cfg_df = folds_df[folds_df["configuration"] == config_idx]
        repeat_num = int(cfg_df["repeat"].iloc[0])
        fold_num = int(cfg_df["fold"].iloc[0])
        train_sub = cfg_df[cfg_df["split"] == "train"].sort_values("dataset_index")
        test_sub = cfg_df[cfg_df["split"] == "test"].sort_values("dataset_index")

        X_train = feat_indexed.loc[train_sub["filename"], feature_cols].to_numpy(dtype=np.float64)
        y_train = train_sub["authoritative_grade"].to_numpy(dtype=int)
        X_test = feat_indexed.loc[test_sub["filename"], feature_cols].to_numpy(dtype=np.float64)
        y_test = test_sub["authoritative_grade"].to_numpy(dtype=int)

        grid_ckpt_path = ckpt_root / f"m5_config_{config_idx:02d}_grid_checkpoint.json"
        t0 = time.perf_counter()
        model, best_params, inner_cv_f1 = fit_nested_cv_random_forest(
            X_train,
            y_train,
            inner_folds=inner_cv_folds,
            random_state=42 + config_idx,
            param_grid=param_grid,
            config_idx=config_idx,
            total_configs=total_configs,
            log_every_candidate=log_every_candidate,
            grid_checkpoint_path=grid_ckpt_path,
            resume=resume,
            grid_checkpoint_every=grid_checkpoint_every,
        )
        train_seconds = time.perf_counter() - t0

        tracemalloc.start()
        t_inf = time.perf_counter()
        probs = model.predict_proba(X_test)
        preds = np.argmax(probs, axis=1).astype(int)
        latency_ms_per_image = ((time.perf_counter() - t_inf) * 1000.0) / max(1, len(X_test))
        _cur, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem_mb = peak_bytes / (1024.0 * 1024.0)

        test_macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        test_acc = float(np.mean(preds == y_test))
        test_qwk = _safe_qwk(y_test, preds)

        fold_df = pd.DataFrame(
            {
                "configuration": config_idx,
                "repeat": repeat_num,
                "fold": fold_num,
                "dataset_index": test_sub["dataset_index"].to_numpy(dtype=int),
                "filename": test_sub["filename"].tolist(),
                "authoritative_grade": y_test,
                "predicted_grade": preds,
                "prob_grade_0": probs[:, 0],
                "prob_grade_1": probs[:, 1],
                "prob_grade_2": probs[:, 2],
                "prob_grade_3": probs[:, 3],
                "inference_latency_ms": round(latency_ms_per_image, 4),
                "peak_memory_mb": round(peak_mem_mb, 4),
            }
        )
        fold_df.to_csv(pred_csv, index=False)
        all_preds.append(fold_df)

        payload = {
            "configuration": config_idx,
            "repeat": repeat_num,
            "fold": fold_num,
            "model": model,
            "best_params": best_params,
            "feature_columns": feature_cols,
            "inner_cv_macro_f1": inner_cv_f1,
            "test_macro_f1": test_macro_f1,
            "test_qwk": test_qwk,
        }
        with (models_dir / f"rf_config_{config_idx:02d}_model.pkl").open("wb") as f_out:
            pickle.dump(payload, f_out)

        if config_idx not in state["completed_configurations"]:
            state["completed_configurations"].append(config_idx)
        state["completed_configs"][str(config_idx)] = {
            "configuration": config_idx,
            "repeat": repeat_num,
            "fold": fold_num,
            "inner_cv_macro_f1": round(inner_cv_f1, 4),
            "test_macro_f1": round(test_macro_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "test_qwk": round(test_qwk, 4),
            "best_params": best_params,
            "train_seconds": round(train_seconds, 3),
            "inference_latency_ms_per_image": round(latency_ms_per_image, 4),
            "peak_memory_mb": round(peak_mem_mb, 4),
        }
        if test_macro_f1 > float(state.get("best_macro_f1", -1.0)):
            state["best_macro_f1"] = round(test_macro_f1, 6)
            state["best_configuration"] = config_idx
            with (models_dir / "rf_best_model.pkl").open("wb") as f_best:
                pickle.dump(payload, f_best)

        _atomic_write_json(state_path, state)

        LOGGER.info(
            "[M5] Config %02d/%02d (Repeat %d, Fold %d) complete: InnerCV F1=%.4f | "
            "Test Macro-F1=%.4f | QWK=%.4f | Acc=%.2f%% | Train=%.1fs | Latency=%.3f ms/img",
            config_idx,
            total_configs,
            repeat_num,
            fold_num,
            inner_cv_f1,
            test_macro_f1,
            test_qwk,
            test_acc * 100.0,
            train_seconds,
            latency_ms_per_image,
        )

    combined_df = pd.concat(all_preds, ignore_index=True)
    combined_csv = m5_out / "m5_rf_5x2_oof_predictions.csv"
    combined_df.to_csv(combined_csv, index=False)
    return {
        "predictions_df": combined_df,
        "predictions_path": combined_csv,
        "best_model_path": models_dir / "rf_best_model.pkl",
        "state": state,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M5 Random Forest Classifier Training with Nested CV Grid Search and Two-Tier Checkpointing"
    )
    parser.add_argument("--project-root", "--root-dir", dest="project_root", type=Path, default=None)
    parser.add_argument("--feature-matrix", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--inner-folds", "--inner-cv", dest="inner_folds", type=int, default=3)
    parser.add_argument("--log-every-candidate", action="store_true")
    parser.add_argument("--grid-checkpoint-every", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = find_project_root(args.project_root, require_jpeg_dir=False)
    run_m5_random_forest_pipeline(
        project_root=root,
        feature_matrix_path=args.feature_matrix.resolve() if args.feature_matrix else None,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        checkpoint_dir=args.checkpoint_dir.resolve() if args.checkpoint_dir else None,
        inner_cv_folds=args.inner_folds,
        log_every_candidate=args.log_every_candidate,
        grid_checkpoint_every=args.grid_checkpoint_every,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
