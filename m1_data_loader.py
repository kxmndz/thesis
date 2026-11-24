#!/usr/bin/env python3
"""M1 – Layered-Labs/ACNE04 Parquet Data Loader, 512x512 Resolution Standardization,
Watermark Audit/Exclusion, Live Progress Tracking, and Stratified 5x2 Cross-Validation Splitter.

Exclusively ingests https://huggingface.co/datasets/Layered-Labs/ACNE04 (Parquet format),
parsing the `image`, `level`, `file_name`, and `lesion_boxes` columns.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import RepeatedStratifiedKFold

LOGGER = logging.getLogger(__name__)

DEFAULT_HF_DATASET_ID: str = "Layered-Labs/ACNE04"
DEFAULT_TARGET_SIZE: int = 512
DEFAULT_RANDOM_STATE: int = 42

HAYASHI_GRADE_NAMES: dict[int, str] = {
    0: "Mild",
    1: "Moderate",
    2: "Severe",
    3: "Very Severe",
}

HAYASHI_GRADE_DESCRIPTIONS: dict[int, str] = {
    0: "Grade 0 (Mild: 0-5)",
    1: "Grade 1 (Moderate: 6-20)",
    2: "Grade 2 (Severe: 21-50)",
    3: "Grade 3 (Very Severe: >50)",
}

# Exact 51 watermarked images loaded into the full dataset inventory
# but strictly excluded from training and Stratified 5x2 CV evaluation.
WATERMARK_EXCLUDED_FILENAMES: frozenset[str] = frozenset(
    {
        # Level 0 (6)
        "levle0_0.jpg",
        "levle0_18.jpg",
        "levle0_224.jpg",
        "levle0_532.jpg",
        "levle0_5.jpg",
        "levle0_6.jpg",
        # Level 1 (14)
        "levle1_10.jpg",
        "levle1_13.jpg",
        "levle1_14.jpg",
        "levle1_1.jpg",
        "levle1_255.jpg",
        "levle1_256.jpg",
        "levle1_26.jpg",
        "levle1_27.jpg",
        "levle1_2.jpg",
        "levle1_3.jpg",
        "levle1_4.jpg",
        "levle1_5.jpg",
        "levle1_642.jpg",
        "levle1_6.jpg",
        # Level 2 (9)
        "levle2_0.jpg",
        "levle2_10.jpg",
        "levle2_116.jpg",
        "levle2_17.jpg",
        "levle2_22.jpg",
        "levle2_23.jpg",
        "levle2_3.jpg",
        "levle2_4.jpg",
        "levle2_5.jpg",
        # Level 3 (22)
        "levle3_104.jpg",
        "levle3_10.jpg",
        "levle3_11.jpg",
        "levle3_123.jpg",
        "levle3_12.jpg",
        "levle3_137.jpg",
        "levle3_13.jpg",
        "levle3_15.jpg",
        "levle3_16.jpg",
        "levle3_17.jpg",
        "levle3_18.jpg",
        "levle3_19.jpg",
        "levle3_20.jpg",
        "levle3_22.jpg",
        "levle3_48.jpg",
        "levle3_59.jpg",
        "levle3_60.jpg",
        "levle3_61.jpg",
        "levle3_6.jpg",
        "levle3_7.jpg",
        "levle3_8.jpg",
        "levle3_9.jpg",
    }
)

_WATERMARK_EXCLUDED_LOWER: frozenset[str] = frozenset(
    name.lower() for name in WATERMARK_EXCLUDED_FILENAMES
)


def is_watermarked_filename(filename: str | Path) -> bool:
    """Return True if filename is in the 51 watermark-excluded ACNE04 images."""
    return Path(filename).name.lower() in _WATERMARK_EXCLUDED_LOWER


def assert_no_watermarked_filenames(
    filenames: Iterable[str | Path],
    context: str = "training/evaluation",
) -> None:
    """Raise RuntimeError if any watermark-excluded filename leaks into downstream training/evaluation."""
    leaked = sorted({Path(fn).name for fn in filenames if is_watermarked_filename(fn)})
    if leaked:
        raise RuntimeError(
            f"[Watermark Guard] {len(leaked)} excluded watermarked image(s) detected in {context}: "
            f"{leaked[:10]}"
        )


@dataclass(frozen=True)
class StandardizationMetadata:
    orig_width: int
    orig_height: int
    std_width: int
    std_height: int
    resized_width: int
    resized_height: int
    scale_factor: float
    left_padding: int
    top_padding: int
    was_letterboxed: bool


def compute_standardization_metadata(
    orig_w: int,
    orig_h: int,
    target_size: int = DEFAULT_TARGET_SIZE,
    resize_mode: Literal["letterbox", "stretch"] = "letterbox",
) -> StandardizationMetadata:
    """Compute 512x512 standardization geometry deterministically from EXIF-oriented dimensions."""
    if orig_w == target_size and orig_h == target_size:
        return StandardizationMetadata(
            orig_width=orig_w,
            orig_height=orig_h,
            std_width=target_size,
            std_height=target_size,
            resized_width=target_size,
            resized_height=target_size,
            scale_factor=1.0,
            left_padding=0,
            top_padding=0,
            was_letterboxed=False,
        )

    if resize_mode == "stretch" or orig_w == orig_h:
        return StandardizationMetadata(
            orig_width=orig_w,
            orig_height=orig_h,
            std_width=target_size,
            std_height=target_size,
            resized_width=target_size,
            resized_height=target_size,
            scale_factor=target_size / float(max(orig_w, orig_h)),
            left_padding=0,
            top_padding=0,
            was_letterboxed=False,
        )

    scale = min(target_size / float(orig_w), target_size / float(orig_h))
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    left_pad = (target_size - new_w) // 2
    top_pad = (target_size - new_h) // 2
    return StandardizationMetadata(
        orig_width=orig_w,
        orig_height=orig_h,
        std_width=target_size,
        std_height=target_size,
        resized_width=new_w,
        resized_height=new_h,
        scale_factor=scale,
        left_padding=left_pad,
        top_padding=top_pad,
        was_letterboxed=(new_w != target_size or new_h != target_size),
    )


def _get_exif_oriented_size(pil_img: Image.Image) -> tuple[int, int]:
    """Return `(width, height)` after EXIF orientation without forcing a pixel copy/resize."""
    w, h = pil_img.size
    try:
        orientation = int(pil_img.getexif().get(274, 1))
    except Exception:
        orientation = 1
    if orientation in {5, 6, 7, 8}:
        return h, w
    return w, h


def standardize_pil_image(
    pil_img: Image.Image,
    target_size: int = DEFAULT_TARGET_SIZE,
    resize_mode: Literal["letterbox", "stretch"] = "letterbox",
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, StandardizationMetadata]:
    """Standardize a facial image to a uniform target_size x target_size RGB canvas (512x512).

    Uses EXIF orientation correction and aspect-ratio-preserving centered letterboxing
    (LANCZOS for downscaling, BICUBIC for upscaling) so non-square Layered-Labs/ACNE04
    images undergo zero geometric distortion. Square images receive zero padding.
    """
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    orig_w, orig_h = img.size
    meta = compute_standardization_metadata(
        orig_w=orig_w,
        orig_h=orig_h,
        target_size=target_size,
        resize_mode=resize_mode,
    )

    if orig_w == target_size and orig_h == target_size:
        return img, meta

    if resize_mode == "stretch" or orig_w == orig_h:
        resample = (
            Image.Resampling.LANCZOS
            if max(orig_w, orig_h) > target_size
            else Image.Resampling.BICUBIC
        )
        return img.resize((target_size, target_size), resample=resample), meta

    resample = Image.Resampling.LANCZOS if meta.scale_factor < 1.0 else Image.Resampling.BICUBIC
    resized = img.resize((meta.resized_width, meta.resized_height), resample=resample)
    canvas = Image.new("RGB", (target_size, target_size), color=pad_color)
    canvas.paste(resized, (meta.left_padding, meta.top_padding))
    return canvas, meta


def _decode_parquet_image(image_cell: Any) -> tuple[Image.Image, str | None]:
    """Decode the Layered-Labs/ACNE04 `image` Parquet column (HF PIL Image or Parquet struct)."""
    if isinstance(image_cell, Image.Image):
        return image_cell, getattr(image_cell, "filename", None) or None

    if isinstance(image_cell, dict):
        embedded_path = image_cell.get("path")
        raw_bytes = image_cell.get("bytes")
        if raw_bytes is not None:
            with Image.open(io.BytesIO(raw_bytes)) as opened:
                return opened.copy(), embedded_path
        raise ValueError(
            f"Parquet `image` struct missing `bytes` payload; keys={list(image_cell.keys())}"
        )

    if isinstance(image_cell, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image_cell))) as opened:
            return opened.copy(), None

    raise TypeError(f"Unsupported Parquet `image` column type: {type(image_cell).__name__}")


def _parse_parquet_level(raw_level: Any) -> int:
    """Strictly parse the Layered-Labs/ACNE04 integer `level` column (0, 1, 2, or 3)."""
    if raw_level is None or (isinstance(raw_level, float) and np.isnan(raw_level)):
        raise ValueError("Missing value in Parquet `level` column.")
    grade = int(raw_level)
    if grade not in HAYASHI_GRADE_NAMES:
        raise ValueError(f"Invalid Hayashi `level`={grade}; expected one of 0, 1, 2, 3.")
    return grade


def _parse_parquet_lesion_boxes(raw_boxes: Any) -> list[tuple[float, float, float, float]]:
    """Parse Layered-Labs/ACNE04 `lesion_boxes` struct: {xmin, ymin, xmax, ymax}."""
    if not isinstance(raw_boxes, dict):
        return []
    xmins = raw_boxes.get("xmin")
    ymins = raw_boxes.get("ymin")
    xmaxs = raw_boxes.get("xmax")
    ymaxs = raw_boxes.get("ymax")
    if xmins is None or ymins is None or xmaxs is None or ymaxs is None:
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in zip(xmins, ymins, xmaxs, ymaxs):
        fx1, fy1, fx2, fy2 = float(x1), float(y1), float(x2), float(y2)
        if fx2 > fx1 and fy2 > fy1:
            boxes.append((fx1, fy1, fx2, fy2))
    return boxes


def _transform_boxes_to_standardized(
    boxes: list[tuple[float, float, float, float]],
    meta: StandardizationMetadata,
) -> list[list[float]]:
    """Transform raw Layered-Labs/ACNE04 pixel boxes into standardized 512x512 coordinates."""
    sx = meta.resized_width / float(meta.orig_width)
    sy = meta.resized_height / float(meta.orig_height)
    out: list[list[float]] = []
    for x1, y1, x2, y2 in boxes:
        bx1 = float(np.clip(x1 * sx + meta.left_padding, 0.0, meta.std_width - 2.0))
        by1 = float(np.clip(y1 * sy + meta.top_padding, 0.0, meta.std_height - 2.0))
        bx2 = float(np.clip(x2 * sx + meta.left_padding, bx1 + 1.5, meta.std_width - 1.0))
        by2 = float(np.clip(y2 * sy + meta.top_padding, by1 + 1.5, meta.std_height - 1.0))
        if bx2 - bx1 >= 1.0 and by2 - by1 >= 1.0:
            out.append([round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)])
    return out


def load_layered_labs_acne04_stream(
    dataset_id: str = DEFAULT_HF_DATASET_ID,
    parquet_path: str | Path | None = None,
    hf_token: str | None = None,
) -> tuple[int, Iterable[tuple[str, dict[str, Any]]]]:
    """Return `(total_rows, row_iterator)` for the FULL Layered-Labs/ACNE04 Parquet dataset."""
    if parquet_path is not None:
        p = Path(parquet_path)
        parquet_files = [p] if p.is_file() else (sorted(p.rglob("*.parquet")) if p.is_dir() else [])
        if not parquet_files:
            raise FileNotFoundError(f"No Layered-Labs/ACNE04 .parquet files found at: {p}")

        loaded_frames: list[tuple[str, pd.DataFrame]] = []
        total_rows = 0
        for pq_file in parquet_files:
            df_pq = pd.read_parquet(pq_file)
            missing = {"image", "level"} - set(df_pq.columns)
            if missing:
                raise KeyError(
                    f"Parquet file {pq_file.name} missing required columns {sorted(missing)}; "
                    f"found {list(df_pq.columns)}"
                )
            loaded_frames.append((pq_file.stem, df_pq))
            total_rows += len(df_pq)

        def _pq_iter() -> Iterable[tuple[str, dict[str, Any]]]:
            for split_stem, df_pq in loaded_frames:
                for rec in df_pq.to_dict(orient="records"):
                    yield split_stem, rec

        return total_rows, _pq_iter()

    from datasets import Dataset, DatasetDict, load_dataset

    LOGGER.info("Loading full dataset '%s' from Hugging Face Hub...", dataset_id)
    ds_obj = load_dataset(dataset_id, token=hf_token)
    if isinstance(ds_obj, DatasetDict):
        splits = list(ds_obj.items())
    elif isinstance(ds_obj, Dataset):
        splits = [("train", ds_obj)]
    else:
        raise TypeError(f"Unexpected dataset object type: {type(ds_obj).__name__}")

    total_rows = 0
    for split_name, split_ds in splits:
        missing = {"image", "level"} - set(split_ds.column_names)
        if missing:
            raise KeyError(
                f"Split '{split_name}' in '{dataset_id}' missing required columns {sorted(missing)}; "
                f"found {split_ds.column_names}"
            )
        total_rows += len(split_ds)

    def _hf_iter() -> Iterable[tuple[str, dict[str, Any]]]:
        for split_name, split_ds in splits:
            for row in split_ds:
                yield split_name, row

    return total_rows, _hf_iter()


def iter_layered_labs_acne04_rows(
    dataset_id: str = DEFAULT_HF_DATASET_ID,
    parquet_path: str | Path | None = None,
    hf_token: str | None = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield `(split_name, row_dict)` for the FULL Layered-Labs/ACNE04 Parquet dataset."""
    _total, row_iter = load_layered_labs_acne04_stream(
        dataset_id=dataset_id,
        parquet_path=parquet_path,
        hf_token=hf_token,
    )
    return row_iter


def resolve_grade_column(df: pd.DataFrame) -> str:
    """Return the canonical severity grade column from an M1 manifest DataFrame."""
    for col in ("authoritative_grade", "level"):
        if col in df.columns:
            return col
    raise KeyError(
        f"Manifest DataFrame missing 'authoritative_grade'/'level' column; found {list(df.columns)}"
    )


def normalize_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `level` and `authoritative_grade` integer columns exist consistently."""
    out = df.copy()
    grade_col = resolve_grade_column(out)
    grades = out[grade_col].astype(int)
    out["level"] = grades
    out["authoritative_grade"] = grades
    return out


def load_m1_lesion_boxes_512(project_root: Path | str) -> dict[str, list[list[float]]]:
    """Load the 512x512 standardized lesion bounding boxes exported from Layered-Labs/ACNE04."""
    boxes_path = Path(project_root) / "M1_outputs" / "m1_lesion_boxes_512.json"
    if not boxes_path.is_file():
        raise FileNotFoundError(
            f"Missing M1 lesion boxes file: {boxes_path}. Run M1 on Layered-Labs/ACNE04 first."
        )
    return json.loads(boxes_path.read_text(encoding="utf-8"))


def build_5x2_cv_splits(
    filenames: list[str],
    y_labels: np.ndarray,
    random_state: int = DEFAULT_RANDOM_STATE,
    exclude_watermarked: bool = True,
) -> list[dict[str, Any]]:
    """Build 5 repetitions x 2 complementary stratified folds (configurations 1..10)."""
    if exclude_watermarked:
        assert_no_watermarked_filenames(filenames, context="M1 5x2 CV split generation")
    x_dummy = np.arange(len(filenames), dtype=int)
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random_state)
    cv_splits: list[dict[str, Any]] = []
    for configuration_index, (train_idx, test_idx) in enumerate(
        rskf.split(x_dummy, y_labels), start=1
    ):
        repeat_number = (configuration_index - 1) // 2 + 1
        fold_number = (configuration_index - 1) % 2 + 1
        cv_splits.append(
            {
                "configuration_index": configuration_index,
                "repeat_number": repeat_number,
                "fold_number": fold_number,
                "train_idx": train_idx.copy(),
                "test_idx": test_idx.copy(),
                "train_labels": y_labels[train_idx].copy(),
                "test_labels": y_labels[test_idx].copy(),
            }
        )
    return cv_splits


def _list_nonempty_filenames(directory: Path) -> set[str]:
    """List non-empty filenames in a directory with minimal Drive FUSE metadata calls."""
    if not directory.is_dir():
        return set()
    names: set[str] = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                try:
                    if entry.stat().st_size > 0:
                        names.add(entry.name)
                except OSError:
                    continue
    return names


def _try_resume_complete_m1(
    root: Path,
    m1_output_root: Path,
    jpg_train_dir: Path,
    target_size: int,
    resize_mode: str,
    exclude_watermarked: bool,
    random_state: int,
) -> dict[str, Any] | None:
    """Return cached M1 outputs immediately if all manifests/boxes and staged images are intact."""
    full_manifest_path = m1_output_root / "m1_full_dataset_manifest.csv"
    excluded_path = m1_output_root / "m1_excluded_watermarks.csv"
    manifest_path = m1_output_root / "m1_dataset_manifest.csv"
    folds_path = m1_output_root / "m1_5x2_fold_assignments.csv"
    boxes_path = m1_output_root / "m1_lesion_boxes_512.json"
    summary_path = m1_output_root / "m1_run_summary.json"

    required_files = (
        full_manifest_path,
        excluded_path,
        manifest_path,
        folds_path,
        boxes_path,
        summary_path,
    )
    if not all(p.is_file() for p in required_files):
        return None

    try:
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_payload.get("target_resolution") != f"{target_size}x{target_size}":
            return None
        if summary_payload.get("resize_mode") != resize_mode:
            return None
        cached_excl = summary_payload.get(
            "exclude_watermarked",
            int(summary_payload.get("excluded_watermarked_images", 0)) > 0,
        )
        if bool(cached_excl) != bool(exclude_watermarked):
            return None
        if int(summary_payload.get("random_state", DEFAULT_RANDOM_STATE)) != int(random_state):
            return None

        df_full = pd.read_csv(full_manifest_path)
        df_excluded = pd.read_csv(excluded_path)
        df_manifest = pd.read_csv(manifest_path)
        df_folds = pd.read_csv(folds_path)

        if df_manifest.empty or df_folds.empty:
            return None
        if exclude_watermarked:
            assert_no_watermarked_filenames(df_manifest["filename"], context="resumed m1_dataset_manifest.csv")
            assert_no_watermarked_filenames(df_folds["filename"], context="resumed m1_5x2_fold_assignments.csv")

        existing_train = _list_nonempty_filenames(jpg_train_dir)
        expected_train = set(df_manifest["filename"].astype(str))
        if not expected_train.issubset(existing_train):
            missing_cnt = len(expected_train - existing_train)
            LOGGER.info(
                "[M1] Found existing manifests but %d training image(s) missing on disk; continuing staging...",
                missing_cnt,
            )
            return None
    except Exception as exc:
        LOGGER.warning("[M1] Could not resume cached M1 manifests (%s); regenerating...", exc)
        return None

    LOGGER.info(
        "[M1] Resumed complete M1 outputs from %s (%d loaded | %d excluded | %d training-eligible | %d 5x2 CV configs) — skipping Parquet re-export.",
        m1_output_root,
        len(df_full),
        len(df_excluded),
        len(df_manifest),
        int(df_folds["configuration"].nunique()),
    )
    return {
        "project_root": root,
        "full_manifest": df_full,
        "excluded_manifest": df_excluded,
        "dataset_manifest": df_manifest,
        "fold_assignments": df_folds,
        "full_manifest_path": full_manifest_path,
        "excluded_manifest_path": excluded_path,
        "dataset_manifest_path": manifest_path,
        "fold_assignments_path": folds_path,
        "lesion_boxes_path": boxes_path,
        "summary": summary_payload,
        "resumed_from_cache": True,
    }


def run_m1_pipeline(
    project_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    dataset_id: str = DEFAULT_HF_DATASET_ID,
    parquet_path: Path | str | None = None,
    target_size: int = DEFAULT_TARGET_SIZE,
    resize_mode: Literal["letterbox", "stretch"] = "letterbox",
    exclude_watermarked: bool = True,
    hf_token: str | None = None,
    jpeg_quality: int = 95,
    overwrite: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Load the full Layered-Labs/ACNE04 Parquet dataset, standardize to 512x512,
    isolate the 51 watermarked images from training, and export M1 manifests & 5x2 CV splits.

    When `overwrite=False` (default):
      - If all M1 manifests, lesion boxes, and staged 512x512 images already exist, resumes instantly.
      - Otherwise skips re-saving any already-staged non-empty JPEGs and displays a live `tqdm` bar.
    """
    from tqdm.auto import tqdm

    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    jpg_train_dir = root / "Classification" / "JPEGImages"
    jpg_excl_dir = root / "Classification" / "JPEGImages_excluded"
    m1_output_root = Path(output_dir).resolve() if output_dir else (root / "M1_outputs")

    for d in (jpg_train_dir, jpg_excl_dir, m1_output_root):
        d.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        cached = _try_resume_complete_m1(
            root=root,
            m1_output_root=m1_output_root,
            jpg_train_dir=jpg_train_dir,
            target_size=target_size,
            resize_mode=resize_mode,
            exclude_watermarked=exclude_watermarked,
            random_state=random_state,
        )
        if cached is not None:
            return cached

    # Remove any stale excluded files from the active training directory
    if exclude_watermarked:
        for stale in WATERMARK_EXCLUDED_FILENAMES:
            stale_path = jpg_train_dir / stale
            if stale_path.exists():
                stale_path.unlink()

    existing_train_files = set() if overwrite else _list_nonempty_filenames(jpg_train_dir)
    existing_excl_files = set() if overwrite else _list_nonempty_filenames(jpg_excl_dir)

    total_rows, row_iter = load_layered_labs_acne04_stream(
        dataset_id=dataset_id,
        parquet_path=parquet_path,
        hf_token=hf_token,
    )
    LOGGER.info(
        "[M1] Standardizing & staging %d images (%dx%d, mode=%s, overwrite=%s; %d already on disk)...",
        total_rows,
        target_size,
        target_size,
        resize_mode,
        overwrite,
        len(existing_train_files) + len(existing_excl_files),
    )

    full_records: list[dict[str, Any]] = []
    lesion_boxes_512: dict[str, list[list[float]]] = {}
    seen_lower: set[str] = set()
    matched_excluded_lower: set[str] = set()
    saved_new = 0
    skipped_existing = 0

    pbar = tqdm(
        enumerate(row_iter),
        total=total_rows,
        desc=f"M1 Standardizing {target_size}x{target_size}",
        leave=True,
    )
    for row_idx, (hf_split, row) in pbar:
        pil_img, embedded_path = _decode_parquet_image(row["image"])
        level = _parse_parquet_level(row["level"])

        raw_fname = row.get("file_name") or embedded_path
        if not raw_fname or not str(raw_fname).strip():
            raise KeyError(
                f"Row {row_idx} in Layered-Labs/ACNE04 is missing `file_name` and image path metadata."
            )
        filename = Path(str(raw_fname).strip()).name
        if not Path(filename).suffix:
            filename = f"{filename}.jpg"

        fname_lower = filename.lower()
        if fname_lower in seen_lower:
            continue
        seen_lower.add(fname_lower)

        is_excluded = bool(exclude_watermarked and is_watermarked_filename(filename))
        if is_excluded:
            matched_excluded_lower.add(fname_lower)

        dest_dir = jpg_excl_dir if is_excluded else jpg_train_dir
        existing_set = existing_excl_files if is_excluded else existing_train_files
        dest_img_path = dest_dir / filename

        if not overwrite and filename in existing_set:
            orig_w, orig_h = _get_exif_oriented_size(pil_img)
            meta = compute_standardization_metadata(
                orig_w=orig_w,
                orig_h=orig_h,
                target_size=target_size,
                resize_mode=resize_mode,
            )
            skipped_existing += 1
        else:
            std_img, meta = standardize_pil_image(
                pil_img=pil_img,
                target_size=target_size,
                resize_mode=resize_mode,
            )
            std_img.save(dest_img_path, format="JPEG", quality=jpeg_quality, subsampling=0)
            saved_new += 1

        raw_boxes = _parse_parquet_lesion_boxes(row.get("lesion_boxes"))
        std_boxes = _transform_boxes_to_standardized(raw_boxes, meta)
        lesion_boxes_512[filename] = std_boxes

        raw_lesion_count = row.get("lesion_count_label")
        lesion_count_label = (
            int(raw_lesion_count)
            if raw_lesion_count is not None and not pd.isna(raw_lesion_count)
            else len(std_boxes)
        )
        num_lesions_annotated = int(row.get("num_lesions_annotated") or len(std_boxes))

        full_records.append(
            {
                "filename": filename,
                "level": level,
                "authoritative_grade": level,
                "source_path": dest_img_path.relative_to(root).as_posix(),
                "hf_split": str(row.get("split") or hf_split),
                "lesion_count_label": lesion_count_label,
                "num_lesions_annotated": num_lesions_annotated,
                "orig_width": meta.orig_width,
                "orig_height": meta.orig_height,
                "std_width": meta.std_width,
                "std_height": meta.std_height,
                "resized_width": meta.resized_width,
                "resized_height": meta.resized_height,
                "scale_factor": round(meta.scale_factor, 8),
                "left_padding": meta.left_padding,
                "top_padding": meta.top_padding,
                "was_letterboxed": meta.was_letterboxed,
                "excluded_from_training": is_excluded,
                "exclusion_reason": "watermark" if is_excluded else "",
            }
        )
        pbar.set_postfix(saved=saved_new, skipped=skipped_existing, excl=len(matched_excluded_lower))

    if not full_records:
        raise RuntimeError(f"[M1] Loaded 0 records from {parquet_path or dataset_id}.")

    df_full = (
        pd.DataFrame(full_records)
        .sort_values(by=["authoritative_grade", "filename"])
        .reset_index(drop=True)
    )
    df_full.insert(0, "full_dataset_index", np.arange(len(df_full), dtype=int))

    df_excluded = df_full[df_full["excluded_from_training"]].reset_index(drop=True)
    df_manifest = (
        df_full[~df_full["excluded_from_training"]]
        .drop(columns=["full_dataset_index"])
        .reset_index(drop=True)
    )
    df_manifest.insert(0, "dataset_index", np.arange(len(df_manifest), dtype=int))

    unmatched_exclusions = sorted(_WATERMARK_EXCLUDED_LOWER - matched_excluded_lower)
    if exclude_watermarked and unmatched_exclusions:
        LOGGER.warning(
            "[M1] %d exclusion filename(s) not found in dataset: %s",
            len(unmatched_exclusions),
            unmatched_exclusions,
        )

    y_train_eligible = df_manifest["authoritative_grade"].to_numpy(dtype=int)
    grade_counts = {g: int(np.sum(y_train_eligible == g)) for g in (0, 1, 2, 3)}
    if any(cnt < 2 for cnt in grade_counts.values()):
        raise ValueError(f"Insufficient samples per Hayashi grade for 5x2 CV: {grade_counts}")

    filenames = df_manifest["filename"].tolist()
    source_paths = df_manifest["source_path"].tolist()
    cv_splits = build_5x2_cv_splits(
        filenames,
        y_train_eligible,
        random_state=random_state,
        exclude_watermarked=exclude_watermarked,
    )

    membership_rows: list[dict[str, Any]] = []
    all_indices = set(range(len(filenames)))
    for split_info in cv_splits:
        tr_set = set(map(int, split_info["train_idx"]))
        te_set = set(map(int, split_info["test_idx"]))
        assert tr_set.isdisjoint(te_set) and (tr_set | te_set) == all_indices
        for split_name in ("train", "test"):
            for idx in split_info[f"{split_name}_idx"]:
                i = int(idx)
                membership_rows.append(
                    {
                        "configuration": int(split_info["configuration_index"]),
                        "repeat": int(split_info["repeat_number"]),
                        "fold": int(split_info["fold_number"]),
                        "split": split_name,
                        "dataset_index": i,
                        "filename": filenames[i],
                        "level": int(y_train_eligible[i]),
                        "authoritative_grade": int(y_train_eligible[i]),
                        "source_path": source_paths[i],
                    }
                )

    df_folds = pd.DataFrame(membership_rows)
    if exclude_watermarked:
        assert_no_watermarked_filenames(df_manifest["filename"], context="m1_dataset_manifest.csv")
        assert_no_watermarked_filenames(df_folds["filename"], context="m1_5x2_fold_assignments.csv")

    full_manifest_path = m1_output_root / "m1_full_dataset_manifest.csv"
    excluded_path = m1_output_root / "m1_excluded_watermarks.csv"
    manifest_path = m1_output_root / "m1_dataset_manifest.csv"
    folds_path = m1_output_root / "m1_5x2_fold_assignments.csv"
    boxes_path = m1_output_root / "m1_lesion_boxes_512.json"
    summary_path = m1_output_root / "m1_run_summary.json"

    df_full.to_csv(full_manifest_path, index=False, lineterminator="\n")
    df_excluded.to_csv(excluded_path, index=False, lineterminator="\n")
    df_manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    df_folds.to_csv(folds_path, index=False, lineterminator="\n")
    boxes_path.write_text(json.dumps(lesion_boxes_512), encoding="utf-8")

    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset_source": str(parquet_path) if parquet_path is not None else dataset_id,
        "target_resolution": f"{target_size}x{target_size}",
        "resize_mode": resize_mode,
        "exclude_watermarked": bool(exclude_watermarked),
        "random_state": int(random_state),
        "total_loaded_images": int(len(df_full)),
        "saved_new_images": int(saved_new),
        "skipped_existing_images": int(skipped_existing),
        "excluded_watermarked_images": int(len(df_excluded)),
        "training_eligible_images": int(len(df_manifest)),
        "unmatched_exclusion_filenames": unmatched_exclusions,
        "letterboxed_images": int(df_full["was_letterboxed"].sum()),
        "cv_configurations": int(df_folds["configuration"].nunique()),
        "full_class_counts": {
            str(k): int(v) for k, v in df_full["level"].value_counts().sort_index().items()
        },
        "training_class_counts": {str(k): int(v) for k, v in grade_counts.items()},
        "excluded_class_counts": {
            str(k): int(v) for k, v in df_excluded["level"].value_counts().sort_index().items()
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOGGER.info(
        "[M1] Complete | Loaded: %d (saved_new=%d, skipped_existing=%d) | Excluded watermarks: %d | "
        "Training-eligible: %d | Letterboxed to %dx%d: %d | 5x2 CV configs: %d",
        len(df_full),
        saved_new,
        skipped_existing,
        len(df_excluded),
        len(df_manifest),
        target_size,
        target_size,
        int(df_full["was_letterboxed"].sum()),
        df_folds["configuration"].nunique(),
    )
    print("M1_5X2_REPRODUCIBILITY_VERIFIED")

    return {
        "project_root": root,
        "full_manifest": df_full,
        "excluded_manifest": df_excluded,
        "dataset_manifest": df_manifest,
        "fold_assignments": df_folds,
        "full_manifest_path": full_manifest_path,
        "excluded_manifest_path": excluded_path,
        "dataset_manifest_path": manifest_path,
        "fold_assignments_path": folds_path,
        "lesion_boxes_path": boxes_path,
        "summary": summary_payload,
        "resumed_from_cache": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M1: Layered-Labs/ACNE04 Parquet Loader, 512x512 Standardization & 5x2 CV Splitter"
    )
    parser.add_argument("--project-root", "--root-dir", dest="project_root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-id", type=str, default=DEFAULT_HF_DATASET_ID)
    parser.add_argument("--parquet-path", type=Path, default=None)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--resize-mode", choices=("letterbox", "stretch"), default="letterbox")
    parser.add_argument("--no-exclude-watermarks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    run_m1_pipeline(
        project_root=args.project_root,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        parquet_path=args.parquet_path,
        target_size=args.target_size,
        resize_mode=args.resize_mode,
        exclude_watermarked=not args.no_exclude_watermarks,
        overwrite=bool(args.overwrite),
        random_state=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
