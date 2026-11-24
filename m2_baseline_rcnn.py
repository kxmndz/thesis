#!/usr/bin/env python3
"""M2 – Baseline Lesion-Counting Module (Wen et al., 2022 Faster R-CNN ResNet-101).

Strictly consumes Layered-Labs/ACNE04 512x512 standardized images, M1 Parquet-extracted
512x512 lesion bounding boxes (`M1_outputs/m1_lesion_boxes_512.json`), M1 Stratified 5x2 CV
folds (`M1_outputs/m1_5x2_fold_assignments.csv`), and M3 BiSeNet ROI masks
(`M3_outputs/roi_masks/*.png`). Contains zero legacy VOC XML, NNEW, Kaggle, or heuristic box fallbacks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import tracemalloc
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from m1_data_loader import assert_no_watermarked_filenames, load_m1_lesion_boxes_512
from m3_common import find_project_root, standardize_image


def _compute_m2_fold_qwk_and_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Compute Quadratic Weighted Kappa and Macro F1 across Hayashi grades 0..3."""
    from sklearn.metrics import cohen_kappa_score, f1_score

    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3], weights="quadratic"))
    if np.isnan(qwk):
        qwk = 0.0
    return qwk, mf1


WENH06_REPO_URL: str = "https://github.com/wenh06/acne_detection.git"
WENH06_PRETRAINED_MODELS: dict[str, dict[str, str]] = {
    "faster_rcnn_resnet101": {
        "name": "faster_rcnn_resnet101_large",
        "mega_url": "https://mega.nz/file/AZgDXS6Y#ampDThpnc0Gs8MOMlstLZSukGSDy_j_ThC9OxOVWfLs",
        "config_file": "faster_rcnn_resnet101_coco.config",
        "reported_ap50": "0.536",
        "reported_mae_025": "3.49",
        "reported_mae_050": "3.50",
    },
    "faster_rcnn_inception_v2": {
        "name": "faster_rcnn_inception_v2_large",
        "mega_url": "https://mega.nz/file/ocpBxSzL#-oCR-Ovj5s4aQD02cI1zexUA4RT7ath2tE3742GHfMc",
        "config_file": "faster_rcnn_inception_v2_coco.config",
        "reported_ap50": "0.434",
        "reported_mae_025": "3.57",
        "reported_mae_050": "3.63",
    },
}

HAYASHI_GRADE_NAMES: dict[int, str] = {
    0: "Mild (0-5 lesions)",
    1: "Moderate (6-20 lesions)",
    2: "Severe (21-50 lesions)",
    3: "Very Severe (>50 lesions)",
}

_PYTORCH_MODEL_CACHE: dict[str, Any] = {}
_M2_STD_IMAGE_CACHE: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}


def lesion_count_to_hayashi_grade(lesion_count: int | float) -> int:
    """Convert a detected inflammatory lesion count into a Hayashi severity grade (0-3)."""
    count = int(round(max(0.0, float(lesion_count))))
    if count <= 5:
        return 0
    if count <= 20:
        return 1
    if count <= 50:
        return 2
    return 3


def lesion_count_to_grade_probabilities(lesion_count: int | float) -> np.ndarray:
    """Compute smooth 4-class probability scores from a lesion count for ROC-AUC evaluation."""
    count = max(0.0, float(lesion_count))
    true_grade = lesion_count_to_hayashi_grade(count)
    centers = np.array([2.5, 13.0, 35.5, 65.0], dtype=np.float64)
    scales = np.array([3.5, 8.0, 15.5, 20.0], dtype=np.float64)
    logits = -((count - centers) / scales) ** 2
    logits[true_grade] = max(logits[true_grade], float(np.max(logits)) + 0.25)
    exp_logits = np.exp(logits - np.max(logits))
    return (exp_logits / np.sum(exp_logits)).astype(np.float64)


def setup_wenh06_repo_and_pretrained(
    project_root: Path,
    model_key: str = "faster_rcnn_resnet101",
    target_dir: Path | None = None,
    attempt_download: bool = False,
) -> dict[str, Any]:
    """Optional helper for downloading pretrained Wen et al. (2022) inference weights only."""
    info = WENH06_PRETRAINED_MODELS.get(model_key, WENH06_PRETRAINED_MODELS["faster_rcnn_resnet101"])
    ext_root = target_dir or (project_root / "models" / "wenh06_acne_detection")
    weights_dir = ext_root / info["name"]
    weights_dir.mkdir(parents=True, exist_ok=True)

    pb_path = weights_dir / "frozen_inference_graph.pb"
    ckpt_files = list(weights_dir.glob("*.pb")) + list(weights_dir.glob("*.pt")) + list(weights_dir.glob("*.pth"))
    status: dict[str, Any] = {
        "model_key": model_key,
        "model_name": info["name"],
        "mega_url": info["mega_url"],
        "weights_dir": str(weights_dir),
        "weights_downloaded": bool(pb_path.is_file() or ckpt_files),
        "frozen_graph_path": str(pb_path),
    }

    if attempt_download and not status["weights_downloaded"]:
        megadl_bin = shutil.which("megadl")
        if megadl_bin:
            try:
                subprocess.run(
                    [megadl_bin, "--path", str(weights_dir), info["mega_url"]],
                    check=True,
                    timeout=300,
                )
                status["weights_downloaded"] = True
            except Exception as exc:
                status["download_note"] = f"megadl failed: {exc}"

    (weights_dir / "pretrained_model_manifest.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    return status


def load_ground_truth_lesion_counts(
    project_root: Path,
    include_excluded: bool = False,
) -> dict[str, int]:
    """Load ground-truth lesion counts strictly from M1's Layered-Labs/ACNE04 Parquet manifest."""
    manifest_name = "m1_full_dataset_manifest.csv" if include_excluded else "m1_dataset_manifest.csv"
    manifest_path = project_root / "M1_outputs" / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing M1 manifest: {manifest_path}. Run M1 on Layered-Labs/ACNE04 first."
        )
    df = pd.read_csv(manifest_path)
    count_col = (
        "lesion_count_label"
        if "lesion_count_label" in df.columns
        else "num_lesions_annotated"
    )
    return {
        str(fn): int(cnt)
        for fn, cnt in zip(df["filename"], df[count_col])
        if pd.notna(cnt)
    }


def _load_standardized_image_and_roi(
    project_root: Path,
    img_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load 512x512 standardized RGB and M3 BiSeNet ROI mask (`M3_outputs/roi_masks/<stem>.png`)."""
    img_key = str(img_path)
    if img_key in _M2_STD_IMAGE_CACHE:
        return _M2_STD_IMAGE_CACHE[img_key]

    _orig, std_rgb, content_mask, meta = standardize_image(img_path)
    roi_mask = content_mask

    m3_mask_png = project_root / "M3_outputs" / "roi_masks" / f"{img_path.stem}.png"
    if m3_mask_png.is_file():
        mask_u8 = cv2.imread(str(m3_mask_png), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is not None and mask_u8.shape == content_mask.shape:
            m3_roi = (mask_u8 > 0) & content_mask
            if np.count_nonzero(m3_roi) >= 500:
                roi_mask = m3_roi

    if len(_M2_STD_IMAGE_CACHE) < 1600:
        _M2_STD_IMAGE_CACHE[img_key] = (std_rgb, roi_mask, meta)
    return std_rgb, roi_mask, meta


def get_training_boxes_512(
    boxes_map_512: dict[str, list[list[float]]],
    fname: str,
) -> list[list[float]]:
    """Return exact 512x512 lesion boxes [x1, y1, x2, y2] parsed from Layered-Labs/ACNE04."""
    return boxes_map_512.get(fname, [])


def kmeans_anchor_optimization(
    boxes_wh: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Run K-means clustering with IoU distance on training-fold bounding box (w, h) pairs."""
    if len(boxes_wh) < n_clusters:
        default_sizes = [[8], [16], [24], [36], [52]]
        default_ratios = [[0.5, 1.0, 2.0] for _ in default_sizes]
        return {
            "anchor_sizes": default_sizes,
            "aspect_ratios": default_ratios,
            "centroids_wh": [[8.0, 8.0], [16.0, 16.0], [24.0, 24.0], [36.0, 36.0], [52.0, 52.0]],
            "num_training_boxes": int(len(boxes_wh)),
            "anchor_source": "wenh06_faster_rcnn_resnet101_coco_config_defaults",
        }

    rng = np.random.default_rng(random_state)
    wh = np.clip(np.asarray(boxes_wh, dtype=np.float64), 1.0, None)
    centroids = wh[rng.choice(len(wh), size=n_clusters, replace=False)].copy()

    for _ in range(max_iter):
        w_min = np.minimum(wh[:, None, 0], centroids[None, :, 0])
        h_min = np.minimum(wh[:, None, 1], centroids[None, :, 1])
        intersection = w_min * h_min
        union = (wh[:, 0] * wh[:, 1])[:, None] + (centroids[:, 0] * centroids[:, 1])[None, :] - intersection
        assignments = np.argmin(1.0 - (intersection / np.clip(union, 1e-9, None)), axis=1)

        new_centroids = centroids.copy()
        for k in range(n_clusters):
            cluster_pts = wh[assignments == k]
            if len(cluster_pts) > 0:
                new_centroids[k] = np.median(cluster_pts, axis=0)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    order = np.argsort(centroids[:, 0] * centroids[:, 1])
    centroids = centroids[order]
    scales = np.sqrt(centroids[:, 0] * centroids[:, 1])
    ratios = centroids[:, 0] / np.clip(centroids[:, 1], 1e-6, None)
    unique_ratios = sorted(
        {
            round(float(np.clip(r, 0.5, 2.0)), 2)
            for r in [np.percentile(ratios, 20), np.median(ratios), np.percentile(ratios, 80)]
        }
    )
    if len(unique_ratios) < 3:
        unique_ratios = [0.75, 1.0, 1.33]

    anchor_sizes = [[max(4, int(round(s)))] for s in scales]
    aspect_ratios = [unique_ratios for _ in anchor_sizes]
    return {
        "anchor_sizes": anchor_sizes,
        "aspect_ratios": aspect_ratios,
        "centroids_wh": centroids.round(2).tolist(),
        "num_training_boxes": int(len(wh)),
        "anchor_source": "training_fold_layered_labs_acne04_parquet_kmeans_iou",
    }


def collect_training_fold_boxes(
    boxes_map_512: dict[str, list[list[float]]],
    train_filenames: list[str],
) -> np.ndarray:
    """Collect (width, height) pairs in 512x512 coordinates strictly from training-fold Parquet boxes."""
    boxes_wh: list[tuple[float, float]] = []
    for fname in train_filenames:
        for x1, y1, x2, y2 in get_training_boxes_512(boxes_map_512, fname):
            boxes_wh.append((x2 - x1, y2 - y1))
    if not boxes_wh:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(boxes_wh, dtype=np.float64)


def build_pytorch_faster_rcnn_resnet101(
    anchor_sizes: list[list[int]] | None = None,
    aspect_ratios: list[list[float]] | None = None,
    num_classes: int = 2,
):
    """Construct Faster R-CNN with ResNet-101 FPN backbone and custom K-means anchors."""
    import torchvision
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

    backbone = resnet_fpn_backbone(
        backbone_name="resnet101", weights="DEFAULT", trainable_layers=3
    )
    sizes = (
        tuple(tuple(s) for s in anchor_sizes)
        if anchor_sizes
        else ((8,), (16,), (24,), (36,), (52,))
    )
    ratios = (
        tuple(tuple(r) for r in aspect_ratios)
        if aspect_ratios
        else ((0.75, 1.0, 1.33),) * len(sizes)
    )
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )


def detect_lesions_single_image(
    image_path: Path | str,
    roi_mask: np.ndarray | None = None,
    score_threshold: float = 0.50,
    model_path: Path | str | None = None,
    loaded_model: Any | None = None,
) -> dict[str, Any]:
    """Detect acne lesions on a single standardized image using Faster R-CNN."""
    t0 = time.perf_counter()
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Cannot run Faster R-CNN inference; image not found: {image_path}")

    project_root = find_project_root(image_path.parent, require_jpeg_dir=False)
    std_rgb, cached_roi_mask, _meta = _load_standardized_image_and_roi(project_root, image_path)

    active_mask = (
        (np.asarray(roi_mask, dtype=bool) & cached_roi_mask)
        if roi_mask is not None
        else cached_roi_mask
    )

    boxes_scores: list[tuple[float, float, float, float, float]] = []
    backend_used: str | None = None

    if model_path is not None and Path(model_path).is_file() and str(model_path).endswith(".pb"):
        import tensorflow as tf  # type: ignore

        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(str(model_path), "rb") as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name="")
        with tf.compat.v1.Session(graph=graph) as sess:
            boxes_out, scores_out, _num = sess.run(
                [
                    graph.get_tensor_by_name("detection_boxes:0"),
                    graph.get_tensor_by_name("detection_scores:0"),
                    graph.get_tensor_by_name("num_detections:0"),
                ],
                feed_dict={graph.get_tensor_by_name("image_tensor:0"): std_rgb[np.newaxis, ...]},
            )
            h, w = std_rgb.shape[:2]
            for box, score in zip(boxes_out[0], scores_out[0]):
                if float(score) < score_threshold:
                    continue
                ymin, xmin, ymax, xmax = (
                    float(box[0] * h),
                    float(box[1] * w),
                    float(box[2] * h),
                    float(box[3] * w),
                )
                cx = int(np.clip((xmin + xmax) / 2, 0, w - 1))
                cy = int(np.clip((ymin + ymax) / 2, 0, h - 1))
                if active_mask[cy, cx]:
                    boxes_scores.append((xmin, ymin, xmax, ymax, float(score)))
            backend_used = "tensorflow_frozen_graph"

    if backend_used is None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = loaded_model
        if model is None and model_path is not None and Path(model_path).is_file():
            cache_key = f"{Path(model_path).resolve()}:{Path(model_path).stat().st_mtime}:{device}"
            if cache_key not in _PYTORCH_MODEL_CACHE:
                ckpt = torch.load(model_path, map_location=device)
                if "model_state_dict" in ckpt:
                    _m = build_pytorch_faster_rcnn_resnet101(
                        anchor_sizes=ckpt.get("anchor_sizes"),
                        aspect_ratios=ckpt.get("aspect_ratios"),
                    ).to(device)
                    _m.load_state_dict(ckpt["model_state_dict"])
                    _m.eval()
                    _PYTORCH_MODEL_CACHE.clear()
                    _PYTORCH_MODEL_CACHE[cache_key] = _m
            model = _PYTORCH_MODEL_CACHE.get(cache_key)

        if model is None:
            raise FileNotFoundError(
                f"No trained Faster R-CNN checkpoint found (model_path={model_path}). "
                f"Run M2 with mode='train-5x2' first."
            )

        model.eval()
        tensor_img = torch.from_numpy(std_rgb).permute(2, 0, 1).float().to(device) / 255.0
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    preds = model([tensor_img])[0]
            else:
                preds = model([tensor_img])[0]
        for box, score in zip(
            preds["boxes"].detach().cpu().numpy(),
            preds["scores"].detach().cpu().numpy(),
        ):
            if float(score) < score_threshold:
                continue
            xmin, ymin, xmax, ymax = map(float, box)
            cx = int(np.clip((xmin + xmax) / 2, 0, 511))
            cy = int(np.clip((ymin + ymax) / 2, 0, 511))
            if active_mask[cy, cx]:
                boxes_scores.append((xmin, ymin, xmax, ymax, float(score)))
        backend_used = "pytorch_faster_rcnn_resnet101"

    lesion_count = len(boxes_scores)
    predicted_grade = lesion_count_to_hayashi_grade(lesion_count)
    probs = lesion_count_to_grade_probabilities(lesion_count)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "filename": image_path.name,
        "backend": backend_used,
        "score_threshold": score_threshold,
        "lesion_count": lesion_count,
        "predicted_grade": predicted_grade,
        "predicted_grade_label": HAYASHI_GRADE_NAMES[predicted_grade],
        "prob_grade_0": float(probs[0]),
        "prob_grade_1": float(probs[1]),
        "prob_grade_2": float(probs[2]),
        "prob_grade_3": float(probs[3]),
        "boxes": boxes_scores,
        "standardized_rgb": std_rgb,
        "inference_latency_ms": round(latency_ms, 3),
    }


def draw_lesion_detections(
    standardized_rgb: np.ndarray,
    boxes_scores: list[tuple[float, float, float, float, float]],
    output_path: Path | str | None = None,
    title: str | None = None,
) -> np.ndarray:
    """Draw green lesion bounding boxes on a 512x512 RGB image."""
    canvas = Image.fromarray(np.asarray(standardized_rgb, dtype=np.uint8), "RGB")
    draw = ImageDraw.Draw(canvas)
    for xmin, ymin, xmax, ymax, _score in boxes_scores:
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 230, 64), width=2)
    if title:
        draw.rectangle([0, 0, 512, 26], fill=(0, 0, 0))
        draw.text((8, 6), title, fill=(0, 255, 100))
    annotated = np.asarray(canvas, dtype=np.uint8)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    return annotated


def save_atomic_checkpoint(state: dict[str, Any], checkpoint_path: Path) -> None:
    """Write a checkpoint atomically so Colab disconnects never corrupt state."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.{os.getpid()}.tmp")
    if checkpoint_path.suffix == ".json":
        tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    else:
        import torch

        torch.save(state, tmp_path)
    try:
        tmp_path.replace(checkpoint_path)
    except PermissionError:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        tmp_path.replace(checkpoint_path)


def train_or_evaluate_m2_5x2(
    project_root: Path,
    mode: str = "train-5x2",
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
    epochs_per_fold: int = 5,
    batch_size: int = 2,
    learning_rate: float = 0.002,
    score_threshold: float = 0.50,
    pretrained_model_path: Path | None = None,
    configs_to_run: list[int] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute M2 across all 10 Stratified 5x2 CV folds strictly on Layered-Labs/ACNE04."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm

    m1_folds_path = project_root / "M1_outputs" / "m1_5x2_fold_assignments.csv"
    if not m1_folds_path.is_file():
        raise FileNotFoundError(f"Missing M1 fold assignments: {m1_folds_path}. Run M1 first.")

    folds_df = pd.read_csv(m1_folds_path)
    assert_no_watermarked_filenames(folds_df["filename"], context="M2 Faster R-CNN 5x2 folds")

    boxes_map_512 = load_m1_lesion_boxes_512(project_root)
    gt_lesion_counts = load_ground_truth_lesion_counts(project_root, include_excluded=False)
    image_dir = project_root / "Classification" / "JPEGImages"

    m2_out = output_dir or (project_root / "M2_outputs")
    ckpt_root = checkpoint_dir or (m2_out / "checkpoints")
    pred_root = m2_out / "predictions"
    anchor_root = m2_out / "anchors"
    for d in (m2_out, ckpt_root, pred_root, anchor_root):
        d.mkdir(parents=True, exist_ok=True)

    state_file = ckpt_root / "m2_training_state.json"
    if resume and state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = {
            "mode": mode,
            "score_threshold": score_threshold,
            "epochs_per_fold": epochs_per_fold,
            "completed_configurations": [],
            "fold_summaries": {},
        }

    if mode == "infer-pretrained" and pretrained_model_path is None:
        setup_info = setup_wenh06_repo_and_pretrained(project_root, attempt_download=False)
        default_pb = Path(setup_info["frozen_graph_path"])
        if default_pb.is_file():
            pretrained_model_path = default_pb
        else:
            ckpt_candidates = sorted(ckpt_root.glob("*_best.pt"))
            if ckpt_candidates:
                pretrained_model_path = ckpt_candidates[0]
            else:
                raise FileNotFoundError(
                    f"Pretrained weights not found at {default_pb} and no checkpoints exist in {ckpt_root}."
                )

    all_configs = sorted(int(c) for c in folds_df["configuration"].unique().tolist())
    target_configs = (
        [c for c in all_configs if c in set(configs_to_run)] if configs_to_run else all_configs
    )

    all_fold_predictions: list[pd.DataFrame] = []

    for config_idx in target_configs:
        pred_csv_path = pred_root / f"m2_config_{config_idx:02d}_predictions.csv"
        if (
            resume
            and config_idx in state.get("completed_configurations", [])
            and pred_csv_path.is_file()
        ):
            print(f"[M2] Config {config_idx:02d}/10 already completed; loading {pred_csv_path.name}")
            resumed_df = pd.read_csv(pred_csv_path)
            fold_meta = state.get("fold_summaries", {}).get(str(config_idx))
            if isinstance(fold_meta, dict) and "quadratic_weighted_kappa" not in fold_meta:
                r_qwk, r_f1 = _compute_m2_fold_qwk_and_f1(
                    resumed_df["authoritative_grade"].to_numpy(dtype=int),
                    resumed_df["predicted_grade"].to_numpy(dtype=int),
                )
                fold_meta["macro_f1"] = round(r_f1, 4)
                fold_meta["quadratic_weighted_kappa"] = round(r_qwk, 4)
            all_fold_predictions.append(resumed_df)
            continue

        cfg_df = folds_df[folds_df["configuration"] == config_idx]
        repeat_num = int(cfg_df["repeat"].iloc[0])
        fold_num = int(cfg_df["fold"].iloc[0])
        train_df = cfg_df[cfg_df["split"] == "train"].sort_values("dataset_index")
        test_df = cfg_df[cfg_df["split"] == "test"].sort_values("dataset_index")

        train_boxes_wh = collect_training_fold_boxes(boxes_map_512, train_df["filename"].tolist())
        anchor_info = kmeans_anchor_optimization(
            train_boxes_wh, n_clusters=5, random_state=42 + config_idx
        )
        (anchor_root / f"m2_config_{config_idx:02d}_kmeans_anchors.json").write_text(
            json.dumps(anchor_info, indent=2), encoding="utf-8"
        )

        fold_model_path = pretrained_model_path
        active_fold_model = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mode == "train-5x2":
            fold_ckpt_path = ckpt_root / f"m2_config_{config_idx:02d}_checkpoint.pt"
            fold_best_path = ckpt_root / f"m2_config_{config_idx:02d}_best.pt"
            fold_model_path = fold_best_path if fold_best_path.is_file() else fold_ckpt_path

            class ACNE04LesionDataset(Dataset):
                def __init__(self, filenames: list[str]):
                    missing = [f for f in filenames if not (image_dir / f).is_file()]
                    if missing:
                        raise FileNotFoundError(
                            f"[M2] {len(missing)} training images missing from {image_dir} "
                            f"(first: {missing[0]}). Re-run M1 on Layered-Labs/ACNE04."
                        )
                    self.filenames = list(filenames)

                def __len__(self) -> int:
                    return len(self.filenames)

                def __getitem__(self, i: int):
                    fname = self.filenames[i]
                    rgb, _roi_m, _meta = _load_standardized_image_and_roi(
                        project_root, image_dir / fname
                    )
                    scaled_boxes = get_training_boxes_512(boxes_map_512, fname)
                    if scaled_boxes:
                        boxes_t = torch.tensor(scaled_boxes, dtype=torch.float32)
                        labels_t = torch.ones((len(scaled_boxes),), dtype=torch.int64)
                    else:
                        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                        labels_t = torch.zeros((0,), dtype=torch.int64)
                    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    return img_t, {"boxes": boxes_t, "labels": labels_t}

            model = build_pytorch_faster_rcnn_resnet101(
                anchor_sizes=anchor_info["anchor_sizes"],
                aspect_ratios=anchor_info["aspect_ratios"],
            ).to(device)
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=learning_rate,
                momentum=0.9,
                weight_decay=0.0005,
            )
            use_amp = device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            start_epoch = 1
            best_loss = float("inf")
            if resume and fold_ckpt_path.is_file():
                ckpt_data = torch.load(fold_ckpt_path, map_location=device)
                model.load_state_dict(ckpt_data["model_state_dict"])
                optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
                start_epoch = int(ckpt_data["epoch"]) + 1
                best_loss = float(ckpt_data.get("best_loss", float("inf")))

            train_ds = ACNE04LesionDataset(train_df["filename"].tolist())
            if start_epoch <= epochs_per_fold:
                print(
                    f"[M2] Config {config_idx:02d}/10: Training Faster R-CNN on {len(train_ds)} images "
                    f"({anchor_info['num_training_boxes']} Parquet lesion boxes, anchors={anchor_info['anchor_sizes']})",
                    flush=True,
                )
                train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=lambda batch: tuple(zip(*batch)),
                )
                for epoch in range(start_epoch, epochs_per_fold + 1):
                    model.train()
                    epoch_loss = 0.0
                    steps = 0
                    pbar = tqdm(
                        train_loader,
                        desc=f"M2 Config {config_idx:02d}/10 | Epoch {epoch}/{epochs_per_fold}",
                        leave=True,
                    )
                    for images, targets in pbar:
                        images = [img.to(device, non_blocking=True) for img in images]
                        targets = [
                            {k: v.to(device, non_blocking=True) for k, v in t.items()}
                            for t in targets
                        ]
                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            loss_dict = model(images, targets)
                            losses = sum(loss for loss in loss_dict.values())
                        scaler.scale(losses).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        epoch_loss += float(losses.item())
                        steps += 1
                        if steps % 25 == 0 or steps == len(train_loader):
                            pbar.set_postfix(loss=f"{epoch_loss / steps:.4f}")
                    avg_loss = epoch_loss / max(steps, 1)
                    ckpt_payload = {
                        "configuration": config_idx,
                        "repeat": repeat_num,
                        "fold": fold_num,
                        "epoch": epoch,
                        "avg_loss": avg_loss,
                        "best_loss": min(best_loss, avg_loss),
                        "anchor_sizes": anchor_info["anchor_sizes"],
                        "aspect_ratios": anchor_info["aspect_ratios"],
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    save_atomic_checkpoint(ckpt_payload, fold_ckpt_path)
                    if avg_loss <= best_loss:
                        best_loss = avg_loss
                        save_atomic_checkpoint(ckpt_payload, fold_best_path)

            if fold_best_path.is_file():
                best_ckpt = torch.load(fold_best_path, map_location=device)
                if "model_state_dict" in best_ckpt:
                    model.load_state_dict(best_ckpt["model_state_dict"])
            model.eval()
            active_fold_model = model
            fold_model_path = fold_best_path if fold_best_path.is_file() else fold_ckpt_path

        rows: list[dict[str, Any]] = []
        t_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tracemalloc.start()

        for row in tqdm(
            list(test_df.itertuples(index=False)),
            desc=f"M2 Config {config_idx:02d}/10 Test Inference",
            leave=True,
        ):
            fname = str(row.filename)
            true_grade = int(row.authoritative_grade)
            gt_count = gt_lesion_counts.get(fname, np.nan)
            img_path = image_dir / fname
            if not img_path.is_file():
                raise FileNotFoundError(
                    f"[M2] Test image '{fname}' missing at {img_path}. Re-run M1."
                )

            det = detect_lesions_single_image(
                img_path,
                score_threshold=score_threshold,
                model_path=fold_model_path,
                loaded_model=active_fold_model,
            )
            rows.append(
                {
                    "configuration": config_idx,
                    "repeat": repeat_num,
                    "fold": fold_num,
                    "dataset_index": int(row.dataset_index),
                    "filename": fname,
                    "authoritative_grade": true_grade,
                    "ground_truth_lesion_count": gt_count,
                    "predicted_lesion_count": int(det["lesion_count"]),
                    "predicted_grade": int(det["predicted_grade"]),
                    "prob_grade_0": det["prob_grade_0"],
                    "prob_grade_1": det["prob_grade_1"],
                    "prob_grade_2": det["prob_grade_2"],
                    "prob_grade_3": det["prob_grade_3"],
                    "inference_latency_ms": float(det["inference_latency_ms"]),
                    "backend": str(det["backend"]),
                }
            )

        _cur_mem, peak_ram_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = float(peak_ram_bytes) / (1024.0 * 1024.0)
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
        measured_fold_mem_mb = round(peak_ram_mb + peak_vram_mb, 4)

        fold_pred_df = pd.DataFrame(rows)
        fold_pred_df["peak_memory_mb"] = measured_fold_mem_mb
        fold_pred_df.to_csv(pred_csv_path, index=False)
        all_fold_predictions.append(fold_pred_df)

        valid_cnt_mask = fold_pred_df["ground_truth_lesion_count"].notna()
        mae = (
            float(
                np.mean(
                    np.abs(
                        fold_pred_df.loc[valid_cnt_mask, "predicted_lesion_count"]
                        - fold_pred_df.loc[valid_cnt_mask, "ground_truth_lesion_count"]
                    )
                )
            )
            if valid_cnt_mask.any()
            else float("nan")
        )
        acc = float(
            np.mean(fold_pred_df["predicted_grade"] == fold_pred_df["authoritative_grade"])
        )
        fold_qwk, fold_macro_f1 = _compute_m2_fold_qwk_and_f1(
            fold_pred_df["authoritative_grade"].to_numpy(dtype=int),
            fold_pred_df["predicted_grade"].to_numpy(dtype=int),
        )
        if config_idx not in state["completed_configurations"]:
            state["completed_configurations"].append(config_idx)
        state["fold_summaries"][str(config_idx)] = {
            "configuration": config_idx,
            "repeat": repeat_num,
            "fold": fold_num,
            "test_images": len(fold_pred_df),
            "lesion_count_mae": round(mae, 4),
            "grade_accuracy": round(acc, 4),
            "macro_f1": round(fold_macro_f1, 4),
            "quadratic_weighted_kappa": round(fold_qwk, 4),
            "peak_memory_mb": measured_fold_mem_mb,
            "elapsed_seconds": round(time.perf_counter() - t_start, 2),
        }
        save_atomic_checkpoint(state, state_file)

    combined_df = pd.concat(all_fold_predictions, ignore_index=True)
    combined_path = m2_out / "m2_rcnn_5x2_oof_predictions.csv"
    combined_df.to_csv(combined_path, index=False)
    return {
        "predictions_df": combined_df,
        "predictions_path": combined_path,
        "state_path": state_file,
        "state": state,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M2 Baseline Lesion-Counting Module (Layered-Labs/ACNE04)"
    )
    parser.add_argument("--project-root", "--root-dir", dest="project_root", type=Path, default=None)
    parser.add_argument("--mode", choices=("download-pretrained", "infer-pretrained", "train-5x2"), default="train-5x2")
    parser.add_argument("--model-key", choices=tuple(WENH06_PRETRAINED_MODELS.keys()), default="faster_rcnn_resnet101")
    parser.add_argument("--pretrained-model-path", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--configs", type=int, nargs="*", default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = find_project_root(args.project_root, require_jpeg_dir=True)
    if args.mode == "download-pretrained":
        print(json.dumps(setup_wenh06_repo_and_pretrained(root, model_key=args.model_key, attempt_download=True), indent=2))
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


if __name__ == "__main__":
    raise SystemExit(main())
