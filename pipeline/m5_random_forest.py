#!/usr/bin/env python3
"""M5 – Random Forest Classifier Module with Balanced Weights & Nested Cross-Validation.

Based on the Thesis Proposal (Chapter I pp. 20, 26–27; Chapter III pp. 58, 61, 64–67)
and System Architecture Diagram (`System_Architecture_Diagram_CANVA.pdf.pdf`).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m3_common import find_project_root
from m4_feature_extraction import get_feature_columns, run_m4_feature_extraction


def compute_macro_f1_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    """Compute exact 4-class Macro F1-score in pure NumPy."""
    f1s: list[float] = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def stratified_kfold_indices(
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified k-fold (train_idx, val_idx) splits without requiring scikit-learn."""
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y, dtype=int)
    fold_bins: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in sorted(np.unique(y_arr)):
        cls_indices = np.where(y_arr == cls)[0].copy()
        rng.shuffle(cls_indices)
        for i, idx in enumerate(cls_indices):
            fold_bins[i % n_splits].append(int(idx))

    all_idx = np.arange(len(y_arr))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        val_idx = np.array(sorted(fold_bins[k]), dtype=int)
        train_mask = np.ones(len(y_arr), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
        splits.append((train_idx, val_idx))
    return splits


class _DecisionTreeNode:
    """Internal decision tree node storing class distributions for exact TreeSHAP traversal."""

    __slots__ = (
        "feature",
        "threshold",
        "left",
        "right",
        "probs",
        "n_samples",
        "weighted_n_samples",
    )

    def __init__(
        self,
        probs: np.ndarray,
        n_samples: int,
        weighted_n_samples: float,
        feature: int = -1,
        threshold: float = 0.0,
        left: _DecisionTreeNode | None = None,
        right: _DecisionTreeNode | None = None,
    ) -> None:
        self.probs = probs
        self.n_samples = n_samples
        self.weighted_n_samples = weighted_n_samples
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.feature < 0 or self.left is None or self.right is None


class PureNumpyRandomForestClassifier:
    """Balanced-weight Random Forest implementation in NumPy (used when scikit-learn is absent)."""

    def __init__(
        self,
        n_estimators: int = 35,
        max_depth: int = 6,
        min_samples_leaf: int = 5,
        max_features: str | int = "sqrt",
        class_weight: str = "balanced",
        random_state: int = 42,
        n_classes: int = 4,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_classes = n_classes
        self.trees_: list[_DecisionTreeNode] = []
        self.feature_importances_: np.ndarray = np.array([])
        self.classes_: np.ndarray = np.arange(n_classes)

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        class_w = np.where(counts > 0, n / (self.n_classes * np.maximum(counts, 1.0)), 1.0)
        return class_w[y]

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        depth: int,
        rng: np.random.Generator,
        imp_accum: np.ndarray,
    ) -> _DecisionTreeNode:
        n_samples, n_features = X.shape
        w_classes = np.bincount(y, weights=w, minlength=self.n_classes)
        w_total = float(np.sum(w_classes))
        probs = (
            w_classes / w_total
            if w_total > 0
            else np.full(self.n_classes, 1.0 / self.n_classes)
        )
        parent_gini = 1.0 - float(np.sum(probs ** 2))

        if (
            depth >= self.max_depth
            or n_samples < 2 * self.min_samples_leaf
            or parent_gini <= 1e-7
        ):
            return _DecisionTreeNode(
                probs=probs, n_samples=n_samples, weighted_n_samples=w_total
            )

        if self.max_features == "sqrt":
            k_feat = max(1, int(round(np.sqrt(n_features))))
        elif self.max_features == "log2":
            k_feat = max(1, int(round(np.log2(n_features))))
        elif isinstance(self.max_features, int):
            k_feat = min(n_features, max(1, self.max_features))
        else:
            k_feat = n_features

        feat_subset = rng.choice(n_features, size=k_feat, replace=False)
        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0
        best_left_mask = None

        for f in feat_subset:
            col = X[:, f]
            quantiles = np.unique(np.percentile(col, [20, 40, 60, 80]))
            for thresh in quantiles:
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                w_left_vec = np.bincount(
                    y[left_mask], weights=w[left_mask], minlength=self.n_classes
                )
                w_right_vec = w_classes - w_left_vec
                w_l = float(np.sum(w_left_vec))
                w_r = float(np.sum(w_right_vec))
                if w_l <= 0 or w_r <= 0:
                    continue
                gini_l = 1.0 - float(np.sum((w_left_vec / w_l) ** 2))
                gini_r = 1.0 - float(np.sum((w_right_vec / w_r) ** 2))
                gain = parent_gini - (w_l / w_total) * gini_l - (w_r / w_total) * gini_r
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thresh = float(thresh)
                    best_left_mask = left_mask

        if best_feat < 0 or best_left_mask is None:
            return _DecisionTreeNode(
                probs=probs, n_samples=n_samples, weighted_n_samples=w_total
            )

        imp_accum[best_feat] += best_gain * w_total
        right_mask = ~best_left_mask
        left_child = self._build_tree(
            X[best_left_mask],
            y[best_left_mask],
            w[best_left_mask],
            depth + 1,
            rng,
            imp_accum,
        )
        right_child = self._build_tree(
            X[right_mask],
            y[right_mask],
            w[right_mask],
            depth + 1,
            rng,
            imp_accum,
        )
        return _DecisionTreeNode(
            probs=probs,
            n_samples=n_samples,
            weighted_n_samples=w_total,
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> PureNumpyRandomForestClassifier:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int)
        n_samples, n_features = X_arr.shape
        sample_w = self._compute_sample_weights(y_arr)
        rng = np.random.default_rng(self.random_state)
        imp_accum = np.zeros(n_features, dtype=np.float64)
        self.trees_ = []

        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            tree = self._build_tree(
                X_arr[boot_idx],
                y_arr[boot_idx],
                sample_w[boot_idx],
                depth=0,
                rng=rng,
                imp_accum=imp_accum,
            )
            self.trees_.append(tree)

        total_imp = float(np.sum(imp_accum))
        self.feature_importances_ = (
            imp_accum / total_imp
            if total_imp > 0
            else np.full(n_features, 1.0 / n_features)
        )
        return self

    def _predict_tree_proba(self, node: _DecisionTreeNode, x: np.ndarray) -> np.ndarray:
        curr = node
        while not curr.is_leaf:
            if x[curr.feature] <= curr.threshold:
                curr = curr.left  # type: ignore[assignment]
            else:
                curr = curr.right  # type: ignore[assignment]
        return curr.probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[np.newaxis, :]
        out = np.zeros((len(X_arr), self.n_classes), dtype=np.float64)
        for i in range(len(X_arr)):
            row_probs = np.mean(
                [self._predict_tree_proba(t, X_arr[i]) for t in self.trees_], axis=0
            )
            out[i] = row_probs
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(int)


def fit_nested_cv_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_folds: int = 5,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any], float]:
    """Perform Nested Cross-Validation + Grid Search strictly on the training fold."""
    param_candidates = [
        {"n_estimators": 25, "max_depth": 5, "min_samples_leaf": 6, "max_features": "sqrt"},
        {"n_estimators": 40, "max_depth": 7, "min_samples_leaf": 4, "max_features": "sqrt"},
    ]

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [8, 12, None],
            "min_samples_leaf": [2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", "balanced_subsample"],
        }
        inner_cv = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=random_state
        )
        base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        grid = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, dict(grid.best_params_), float(grid.best_score_)
    except ImportError:
        splits = stratified_kfold_indices(
            y_train, n_splits=inner_folds, random_state=random_state
        )
        best_score = -1.0
        best_params = param_candidates[0]
        for cand in param_candidates:
            fold_scores: list[float] = []
            for tr_idx, val_idx in splits:
                clf = PureNumpyRandomForestClassifier(
                    n_estimators=cand["n_estimators"],
                    max_depth=cand["max_depth"],
                    min_samples_leaf=cand["min_samples_leaf"],
                    max_features=cand["max_features"],
                    class_weight="balanced",
                    random_state=random_state,
                )
                clf.fit(X_train[tr_idx], y_train[tr_idx])
                preds = clf.predict(X_train[val_idx])
                fold_scores.append(compute_macro_f1_numpy(y_train[val_idx], preds))
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = cand

        final_model = PureNumpyRandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            class_weight="balanced",
            random_state=random_state,
        )
        final_model.fit(X_train, y_train)
        return final_model, {**best_params, "class_weight": "balanced"}, best_score


def run_m5_random_forest_pipeline(
    project_root: Path,
    feature_matrix_path: Path | None = None,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    inner_cv_folds: int = 5,
    resume: bool = True,
) -> dict[str, Any]:
    """Train and evaluate Random Forest across all 10 Stratified 5x2 CV configurations with checkpointing."""
    m1_folds_path = project_root / "M1_outputs" / "m1_5x2_fold_assignments.csv"
    if not m1_folds_path.is_file():
        raise FileNotFoundError(
            f"Missing M1 fold assignments: {m1_folds_path}. Run `pipeline_cli.py m1` first."
        )

    feat_csv = feature_matrix_path or (
        project_root / "M4_outputs" / "m4_feature_matrix.csv"
    )
    if not feat_csv.is_file():
        print("[M5] M4 feature matrix not found; running M4 feature extraction first...")
        feature_df = run_m4_feature_extraction(project_root)
    else:
        feature_df = pd.read_csv(feat_csv)

    feature_cols = [
        c
        for c in get_feature_columns()
        if c in feature_df.columns
    ]
    if not feature_cols:
        raise ValueError(f"No valid M4 feature columns found in {feat_csv}")

    folds_df = pd.read_csv(m1_folds_path)
    feat_indexed = feature_df.set_index("filename")

    m5_out = output_dir or (project_root / "M5_outputs")
    ckpt_root = checkpoint_dir or (m5_out / "checkpoints")
    models_dir = m5_out / "models"
    preds_dir = m5_out / "predictions"
    for d in (m5_out, ckpt_root, models_dir, preds_dir):
        d.mkdir(parents=True, exist_ok=True)

    state_path = ckpt_root / "m5_training_state.json"
    if resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "completed_configurations": [],
            "fold_summaries": {},
            "best_configuration": None,
            "best_macro_f1": -1.0,
        }

    all_preds: list[pd.DataFrame] = []
    for config_idx in sorted(folds_df["configuration"].unique()):
        config_idx = int(config_idx)
        pred_csv = preds_dir / f"rf_config_{config_idx:02d}_predictions.csv"
        model_pkl = models_dir / f"rf_config_{config_idx:02d}.pkl"

        if (
            resume
            and config_idx in state.get("completed_configurations", [])
            and pred_csv.is_file()
            and model_pkl.is_file()
        ):
            print(
                f"[M5] Configuration {config_idx:02d}/10 already completed in checkpoint; loading {pred_csv.name}"
            )
            all_preds.append(pd.read_csv(pred_csv))
            continue

        cfg_df = folds_df[folds_df["configuration"] == config_idx]
        repeat_num = int(cfg_df["repeat"].iloc[0])
        fold_num = int(cfg_df["fold"].iloc[0])
        train_sub = cfg_df[cfg_df["split"] == "train"].sort_values("dataset_index")
        test_sub = cfg_df[cfg_df["split"] == "test"].sort_values("dataset_index")

        X_train = feat_indexed.loc[train_sub["filename"], feature_cols].to_numpy(
            dtype=np.float64
        )
        y_train = train_sub["authoritative_grade"].to_numpy(dtype=int)
        X_test = feat_indexed.loc[test_sub["filename"], feature_cols].to_numpy(
            dtype=np.float64
        )
        y_test = test_sub["authoritative_grade"].to_numpy(dtype=int)

        tracemalloc.start()
        t_train_start = time.perf_counter()
        best_model, best_params, inner_cv_f1 = fit_nested_cv_random_forest(
            X_train,
            y_train,
            inner_folds=inner_cv_folds,
            random_state=42 + config_idx,
        )
        train_seconds = time.perf_counter() - t_train_start

        t_infer_start = time.perf_counter()
        y_prob = best_model.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1).astype(int)
        infer_seconds = time.perf_counter() - t_infer_start
        _current_mem, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latency_ms_per_image = (infer_seconds * 1000.0) / max(len(X_test), 1)
        peak_mem_mb = peak_mem_bytes / (1024.0 * 1024.0)
        test_macro_f1 = compute_macro_f1_numpy(y_test, y_pred)
        test_acc = float(np.mean(y_test == y_pred))

        payload = {
            "configuration": config_idx,
            "repeat": repeat_num,
            "fold": fold_num,
            "feature_columns": feature_cols,
            "best_params": best_params,
            "inner_cv_macro_f1": inner_cv_f1,
            "test_macro_f1": test_macro_f1,
            "test_accuracy": test_acc,
            "model": best_model,
        }
        with model_pkl.open("wb") as f_out:
            pickle.dump(payload, f_out)

        fold_pred_df = pd.DataFrame(
            {
                "configuration": config_idx,
                "repeat": repeat_num,
                "fold": fold_num,
                "dataset_index": test_sub["dataset_index"].to_numpy(dtype=int),
                "filename": test_sub["filename"].tolist(),
                "authoritative_grade": y_test,
                "predicted_grade": y_pred,
                "prob_grade_0": y_prob[:, 0],
                "prob_grade_1": y_prob[:, 1],
                "prob_grade_2": y_prob[:, 2],
                "prob_grade_3": y_prob[:, 3],
                "inference_latency_ms": round(latency_ms_per_image, 4),
                "peak_memory_mb": round(peak_mem_mb, 4),
            }
        )
        fold_pred_df.to_csv(pred_csv, index=False)
        all_preds.append(fold_pred_df)

        if config_idx not in state["completed_configurations"]:
            state["completed_configurations"].append(config_idx)
        state["fold_summaries"][str(config_idx)] = {
            "configuration": config_idx,
            "repeat": repeat_num,
            "fold": fold_num,
            "best_params": best_params,
            "inner_cv_macro_f1": round(inner_cv_f1, 4),
            "test_macro_f1": round(test_macro_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "train_seconds": round(train_seconds, 3),
            "inference_latency_ms_per_image": round(latency_ms_per_image, 4),
            "peak_memory_mb": round(peak_mem_mb, 4),
        }
        if test_macro_f1 > float(state.get("best_macro_f1", -1.0)):
            state["best_macro_f1"] = round(test_macro_f1, 6)
            state["best_configuration"] = config_idx
            best_model_pkl = models_dir / "rf_best_model.pkl"
            with best_model_pkl.open("wb") as f_best:
                pickle.dump(payload, f_best)

        tmp_state = state_path.with_suffix(".json.tmp")
        tmp_state.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp_state.replace(state_path)

        print(
            f"[M5] Config {config_idx:02d}/10 (Repeat {repeat_num}, Fold {fold_num}): "
            f"InnerCV F1={inner_cv_f1:.4f} | Test Macro-F1={test_macro_f1:.4f} | "
            f"Acc={test_acc:.2%} | Latency={latency_ms_per_image:.3f} ms/img"
        )

    combined_df = pd.concat(all_preds, ignore_index=True)
    combined_csv = m5_out / "m5_rf_5x2_oof_predictions.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(
        f"[M5] Completed all 10 Random Forest folds -> {combined_csv} "
        f"(Best Config: {state['best_configuration']} with Macro-F1={state['best_macro_f1']:.4f})"
    )
    return {
        "predictions_df": combined_df,
        "predictions_path": combined_csv,
        "best_model_path": models_dir / "rf_best_model.pkl",
        "state": state,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M5 Random Forest Classifier Training with Nested CV and Balanced Class Weights"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--feature-matrix", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
