#!/usr/bin/env python3
"""M3 – Final BiSeNet Image Preprocessing Pipeline and M1->M3->M4 Handoff Module.

Extracted from:
- M3_Final_BiSeNet_Preprocessing.ipynb
- M3_BiSeNet_Preprocessing_M4_Handoff.ipynb

Responsibilities:
1. Standardize each source JPEG to 512x512 RGB with centered black padding and content_mask.
2. Run frozen 19-class ResNet-18 BiSeNet ONNX inference (`skin` class 1 + `nose` class 10).
3. Perform objective technical ROI validation (nonzero ROI, zero padding leakage, exact class equation).
4. Compute CIELAB (L*, a*, b*), ordinary grayscale, and CLAHE on full-canvas L* (clipLimit=2.0, grid=(8,8)).
5. Persist one 11-key compressed NPZ archive per image and validate saved archives.
6. Support 13-image representative pilot, 1,457-image research full run (with 92-image visual selection),
   operational M3 run (`M3_bisenet_operational_final_20260928`), 7-image fixed parity check, and
   M1->M3->M4 artifact resolution (`get_m3_artifact_for_filename`, `load_m3_output`) + full-bank parity verification.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import statistics
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import __version__ as pillow_version

from m3_common import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    CLASS_NAMES,
    EXPECTED_GRADE_COUNTS,
    EXPECTED_MODEL_SHA256,
    EXPECTED_TOTAL_IMAGES,
    FACE_PARSING_MEAN,
    FACE_PARSING_STD,
    M3ValidationError,
    PADDING_COLOR,
    RETAINED_CLASS_IDS,
    TARGET_SIZE,
    find_project_root,
    load_authoritative_labels_dict,
    make_visual_panel,
    prepare_color_representations,
    sha256_file,
    standardize_image,
    write_csv,
)
from m3c_face_parsing import create_face_parsing_session

ARCHIVE_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "standardized_rgb": ((TARGET_SIZE, TARGET_SIZE, 3), np.dtype("uint8")),
    "content_mask": ((TARGET_SIZE, TARGET_SIZE), np.dtype("bool")),
    "face_parsing_class_map": ((TARGET_SIZE, TARGET_SIZE), np.dtype("uint8")),
    "skin_mask": ((TARGET_SIZE, TARGET_SIZE), np.dtype("bool")),
    "nose_mask": ((TARGET_SIZE, TARGET_SIZE), np.dtype("bool")),
    "final_roi_mask": ((TARGET_SIZE, TARGET_SIZE), np.dtype("bool")),
    "lab_l_star": ((TARGET_SIZE, TARGET_SIZE), np.dtype("float32")),
    "lab_a_star": ((TARGET_SIZE, TARGET_SIZE), np.dtype("float32")),
    "lab_b_star": ((TARGET_SIZE, TARGET_SIZE), np.dtype("float32")),
    "grayscale": ((TARGET_SIZE, TARGET_SIZE), np.dtype("uint8")),
    "clahe_l_star": ((TARGET_SIZE, TARGET_SIZE), np.dtype("float32")),
}

MANIFEST_COLUMNS: tuple[str, ...] = (
    "filename",
    "grade",
    "audit_reason",
    "source_sha256",
    "original_width",
    "original_height",
    "orientation",
    "exif_orientation",
    "resized_width",
    "resized_height",
    "scale_factor",
    "resize_operation",
    "left_padding",
    "top_padding",
    "right_padding",
    "bottom_padding",
    "valid_content_pixels",
    "skin_pixels",
    "nose_pixels",
    "roi_pixel_count",
    "roi_ratio",
    "padding_leak_pixels",
    "a_star_unchanged",
    "b_star_unchanged",
    "processing_status",
    "error_message",
    "array_archive",
)

M4_EXPOSED_M3_KEYS: tuple[str, ...] = (
    "final_roi_mask",
    "lab_l_star",
    "lab_a_star",
    "lab_b_star",
    "grayscale",
    "clahe_l_star",
)

PILOT_CASES: tuple[tuple[str, str], ...] = (
    ("levle0_1.jpg", "standard portrait"),
    ("levle0_19.jpg", "square image"),
    ("levle0_59.jpg", "MediaPipe fragmented ROI; BiSeNet recovery candidate"),
    ("levle0_73.jpg", "rotated face; known BiSeNet weakness"),
    ("levle1_74.jpg", "Grade 1 landscape; high historical ROI"),
    ("levle2_93.jpg", "small source image; low historical ROI"),
    ("levle3_112.jpg", "authoritative Grade 2 despite filename prefix"),
    ("levle2_4.jpg", "text overlay contamination"),
    ("levle3_125.jpg", "BiSeNet recovery candidate"),
    ("levle3_62.jpg", "both experimental masks fragmented"),
    ("levle3_63.jpg", "close-up; high historical ROI"),
    ("levle3_41.jpg", "watermark contamination"),
    ("levle3_120.jpg", "historical MediaPipe zero-ROI technical failure"),
)

PARITY_FILENAMES: tuple[str, ...] = (
    "levle0_1.jpg",
    "levle1_74.jpg",
    "levle2_93.jpg",
    "levle3_125.jpg",
    "levle3_112.jpg",
    "levle3_120.jpg",
    "levle3_43.jpg",
)

VISUAL_REVIEW_SEED: int = 20260927
DEFAULT_REPRODUCTION_RUN_ID: str = "M3_bisenet_operational_final_20260928"


def parse_face(
    standardized_rgb: np.ndarray,
    content_mask: np.ndarray,
    session: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run 19-class BiSeNet ONNX inference and return class_map, skin_mask, nose_mask, final_roi_mask."""
    normalized = (
        standardized_rgb.astype(np.float32) / 255.0 - FACE_PARSING_MEAN
    ) / FACE_PARSING_STD
    input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(
        np.float32
    )
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_tensor})[0]
    if logits.shape != (1, len(CLASS_NAMES), TARGET_SIZE, TARGET_SIZE):
        raise RuntimeError(f"Unexpected face-parsing output shape: {logits.shape}")
    if not np.isfinite(logits).all():
        raise RuntimeError("Face-parsing logits contain nonfinite values")
    class_map = np.argmax(logits[0], axis=0).astype(np.uint8)
    class_map = np.where(content_mask, class_map, 0).astype(np.uint8)
    skin_mask = class_map == 1
    nose_mask = class_map == 10
    final_roi_mask = (skin_mask | nose_mask) & content_mask
    return class_map, skin_mask, nose_mask, final_roi_mask


def validate_masks(
    standardized_rgb: np.ndarray,
    content_mask: np.ndarray,
    class_map: np.ndarray,
    skin_mask: np.ndarray,
    nose_mask: np.ndarray,
    final_roi_mask: np.ndarray,
) -> int:
    """Perform objective technical validation on BiSeNet masks."""
    if (
        standardized_rgb.shape != (TARGET_SIZE, TARGET_SIZE, 3)
        or standardized_rgb.dtype != np.uint8
    ):
        raise M3ValidationError("standardized_rgb_invalid_shape_or_dtype")
    for name, mask in (
        ("content_mask", content_mask),
        ("skin_mask", skin_mask),
        ("nose_mask", nose_mask),
        ("final_roi_mask", final_roi_mask),
    ):
        if mask.shape != (TARGET_SIZE, TARGET_SIZE) or mask.dtype != np.bool_:
            raise M3ValidationError(f"{name}_invalid_shape_or_dtype")
    if class_map.shape != (TARGET_SIZE, TARGET_SIZE) or class_map.dtype != np.uint8:
        raise M3ValidationError("class_map_invalid_shape_or_dtype")
    if np.any(class_map >= len(CLASS_NAMES)):
        raise M3ValidationError("class_map_out_of_range")
    if not np.array_equal(
        final_roi_mask, np.isin(class_map, RETAINED_CLASS_IDS) & content_mask
    ):
        raise M3ValidationError("final_roi_not_exact_skin_plus_nose")
    if np.any(final_roi_mask & ~content_mask):
        raise M3ValidationError("final_roi_includes_padding")
    roi_pixels = int(np.count_nonzero(final_roi_mask))
    if roi_pixels == 0:
        raise M3ValidationError("final_roi_has_zero_usable_pixels")
    return roi_pixels


def process_m3_image(
    image_path: Path,
    grade: int,
    session: Any,
    audit_reason: str = "",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Process one image through the approved M3 BiSeNet preprocessing pipeline."""
    result: dict[str, Any] = {
        "filename": image_path.name,
        "grade": int(grade),
        "audit_reason": audit_reason,
        "source_sha256": None,
        "processing_status": "error",
        "error_message": None,
        "array_archive": None,
        "valid_content_pixels": None,
        "skin_pixels": None,
        "nose_pixels": None,
        "roi_pixel_count": None,
        "roi_ratio": None,
        "padding_leak_pixels": None,
        "a_star_unchanged": None,
        "b_star_unchanged": None,
    }
    arrays: dict[str, np.ndarray] = {}
    try:
        result["source_sha256"] = sha256_file(image_path)
        original, standardized_rgb, content_mask, resize_metadata = standardize_image(
            image_path
        )
        result.update(resize_metadata)
        result["valid_content_pixels"] = int(np.count_nonzero(content_mask))
        arrays["original_rgb"] = np.asarray(original, dtype=np.uint8)
        arrays["standardized_rgb"] = standardized_rgb
        arrays["content_mask"] = content_mask

        class_map, skin_mask, nose_mask, final_roi_mask = parse_face(
            standardized_rgb, content_mask, session
        )
        arrays.update(
            {
                "face_parsing_class_map": class_map,
                "skin_mask": skin_mask,
                "nose_mask": nose_mask,
                "final_roi_mask": final_roi_mask,
            }
        )
        result["skin_pixels"] = int(np.count_nonzero(skin_mask))
        result["nose_pixels"] = int(np.count_nonzero(nose_mask))
        result["roi_pixel_count"] = int(np.count_nonzero(final_roi_mask))
        result["roi_ratio"] = (
            result["roi_pixel_count"] / result["valid_content_pixels"]
        )
        result["padding_leak_pixels"] = int(
            np.count_nonzero(final_roi_mask & ~content_mask)
        )
        validate_masks(
            standardized_rgb,
            content_mask,
            class_map,
            skin_mask,
            nose_mask,
            final_roi_mask,
        )
        colors = prepare_color_representations(standardized_rgb)
        arrays.update(colors)
        lab_check = cv2.cvtColor(
            standardized_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB
        )
        result["a_star_unchanged"] = bool(
            np.array_equal(colors["lab_a_star"], lab_check[:, :, 1])
        )
        result["b_star_unchanged"] = bool(
            np.array_equal(colors["lab_b_star"], lab_check[:, :, 2])
        )
        if not result["a_star_unchanged"] or not result["b_star_unchanged"]:
            raise M3ValidationError("chromatic_channel_was_modified")
        result["processing_status"] = "success"
    except Exception as error:
        result["error_message"] = f"{type(error).__name__}: {error}"
    return result, arrays


def save_array_archive(
    result: dict[str, Any],
    arrays: dict[str, np.ndarray],
    archive_path: Path,
) -> None:
    """Save the 11-key compressed NPZ archive for a successful M3 BiSeNet image."""
    if result["processing_status"] != "success":
        raise ValueError("Only successful images may have NPZ archives")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive_path, **{key: arrays[key] for key in ARCHIVE_SPEC})


def validate_saved_archive(archive_path: Path) -> list[str]:
    """Reopen and validate all 11 arrays in a saved M3 BiSeNet NPZ archive."""
    issues: list[str] = []
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            for key, (shape, dtype) in ARCHIVE_SPEC.items():
                if key not in archive:
                    issues.append(f"missing_key:{key}")
                    continue
                array = archive[key]
                if array.shape != shape or array.dtype != dtype:
                    issues.append(f"invalid_shape_or_dtype:{key}")
                if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                    issues.append(f"nonfinite:{key}")
            if not issues:
                content = archive["content_mask"]
                classes = archive["face_parsing_class_map"]
                roi = archive["final_roi_mask"]
                if not np.array_equal(
                    roi, np.isin(classes, RETAINED_CLASS_IDS) & content
                ):
                    issues.append("roi_not_exact_skin_plus_nose")
                if np.any(roi & ~content):
                    issues.append("padding_leakage")
                if not np.any(roi):
                    issues.append("zero_roi")
                if not np.array_equal(archive["skin_mask"], classes == 1):
                    issues.append("skin_mask_mismatch")
                if not np.array_equal(archive["nose_mask"], classes == 10):
                    issues.append("nose_mask_mismatch")
                lab = cv2.cvtColor(
                    archive["standardized_rgb"].astype(np.float32) / 255.0,
                    cv2.COLOR_RGB2LAB,
                )
                if not np.array_equal(archive["lab_a_star"], lab[:, :, 1]):
                    issues.append("a_star_modified")
                if not np.array_equal(archive["lab_b_star"], lab[:, :, 2]):
                    issues.append("b_star_modified")
                gray = cv2.cvtColor(
                    archive["standardized_rgb"], cv2.COLOR_RGB2GRAY
                )
                if not np.array_equal(archive["grayscale"], gray):
                    issues.append("ordinary_grayscale_mismatch")
    except Exception as error:
        issues.append(f"archive_read_error:{type(error).__name__}:{error}")
    return issues


def validate_m1_identity(
    x_paths: np.ndarray,
    y_labels: np.ndarray,
    authoritative_labels: dict[str, int],
    image_dir: Path,
) -> None:
    """Validate the M1 image-identity interface without using fold variables."""
    if len(x_paths) != len(y_labels) or len(x_paths) != EXPECTED_TOTAL_IMAGES:
        raise ValueError("M1 image-path and grade arrays must be parallel and complete")
    observed: dict[str, int] = {}
    for image_path, grade in zip(x_paths, y_labels):
        filename = Path(image_path).name
        if filename in observed:
            raise ValueError(f"Duplicate M1 filename: {filename}")
        resolved_candidate = (
            (image_dir.parents[1] / image_path).resolve()
            if not Path(image_path).is_absolute()
            else Path(image_path).resolve()
        )
        if resolved_candidate != (image_dir / filename).resolve():
            raise ValueError(
                f"M1 path does not refer to the current JPEG directory: {image_path}"
            )
        observed[filename] = int(grade)
    if observed != authoritative_labels:
        raise ValueError(
            "M1 filenames or authoritative grades do not match the M3 input set"
        )


def numeric_summary(values: list[int | float]) -> dict[str, float | str]:
    """Compute min, mean, median, and max for a list of numeric values."""
    return {
        "minimum": min(values) if values else "",
        "mean": statistics.mean(values) if values else "",
        "median": statistics.median(values) if values else "",
        "maximum": max(values) if values else "",
    }


def breakdown(records: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    """Build descriptive ROI statistics grouped by grade or orientation."""
    groups: Any = (
        range(4) if category == "grade" else ("portrait", "landscape", "square")
    )
    rows: list[dict[str, Any]] = []
    for group in groups:
        subset = [row for row in records if row.get(category) == group]
        successful = [
            row for row in subset if row["processing_status"] == "success"
        ]
        pixels = [int(row["roi_pixel_count"]) for row in successful]
        ratios = [float(row["roi_ratio"]) for row in successful]
        rows.append(
            {
                category: group,
                "total": len(subset),
                "success": len(successful),
                "failure": len(subset) - len(successful),
                "zero_roi": sum(row.get("roi_pixel_count") == 0 for row in subset),
                "mean_roi_pixels": statistics.mean(pixels) if pixels else "",
                "median_roi_pixels": statistics.median(pixels) if pixels else "",
                "mean_roi_ratio": statistics.mean(ratios) if ratios else "",
                "median_roi_ratio": statistics.median(ratios) if ratios else "",
            }
        )
    return rows


def make_full_selection(
    records: list[dict[str, Any]],
    pilot_cases: tuple[tuple[str, str], ...] = PILOT_CASES,
    seed: int = VISUAL_REVIEW_SEED,
) -> list[dict[str, Any]]:
    """Select the 92-image reproducible visual review subset from a full M3 run."""
    by_name = {row["filename"]: row for row in records}
    successful = sorted(
        (row for row in records if row["processing_status"] == "success"),
        key=lambda row: (float(row["roi_ratio"]), row["filename"]),
    )
    reasons: dict[str, set[str]] = {}

    def add(filename: str, reason: str) -> None:
        reasons.setdefault(filename, set()).add(reason)

    for row in successful[:20]:
        add(row["filename"], "lowest_20_nonzero_roi_ratio")
    for row in successful[-20:]:
        add(row["filename"], "highest_20_roi_ratio")
    rng = random.Random(seed)
    for grade in range(4):
        population = sorted(
            row["filename"] for row in records if row["grade"] == grade
        )
        for filename in rng.sample(population, min(10, len(population))):
            add(filename, f"random_stratified_grade_{grade}")
    for row in records:
        if row.get("orientation") == "square":
            add(row["filename"], "all_square_orientation")
    for orientation in ("portrait", "landscape"):
        if not any(by_name[name].get("orientation") == orientation for name in reasons):
            candidates = sorted(
                row["filename"]
                for row in records
                if row.get("orientation") == orientation
            )
            if candidates:
                add(candidates[0], f"orientation_coverage_{orientation}")
    for filename, _reason in pilot_cases:
        if filename in by_name:
            add(filename, "historical_difficult_pilot")
    for row in records:
        if row["processing_status"] == "error":
            add(row["filename"], "technical_failure")

    selection: list[dict[str, Any]] = []
    for filename in sorted(reasons):
        row = by_name[filename]
        selection.append(
            {
                "filename": filename,
                "grade": row["grade"],
                "orientation": row.get("orientation") or "unknown",
                "selection_reason": ";".join(sorted(reasons[filename])),
                "processing_status": row["processing_status"],
                "roi_pixel_count": row.get("roi_pixel_count"),
                "roi_ratio": row.get("roi_ratio"),
            }
        )
    return selection


def run_bisenet_pilot(
    project_root: Path,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the 13-image representative BiSeNet pilot and generate 10-panel visual sequences."""
    image_dir = project_root / "Classification" / "JPEGImages"
    classification_dir = project_root / "Classification"
    model_path = project_root / "models" / "bisenet_resnet18_celebamaskhq.onnx"
    pilot_root = output_root or (
        project_root / "M3_outputs" / "M3_bisenet_final_pilot"
    )
    if pilot_root.exists() and any(pilot_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite an existing M3 pilot run (pass --overwrite to allow): {pilot_root}"
        )

    authoritative_labels = load_authoritative_labels_dict(
        classification_dir, image_dir
    )
    session = create_face_parsing_session(model_path)

    pilot_root.mkdir(parents=True, exist_ok=True)
    array_dir = pilot_root / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = pilot_root / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    visual_records: list[dict[str, Any]] = []
    start_time = perf_counter()

    for number, (filename, audit_reason) in enumerate(PILOT_CASES, start=1):
        image_path = image_dir / filename
        result, arrays = process_m3_image(
            image_path, authoritative_labels[filename], session, audit_reason
        )
        if result["processing_status"] == "success":
            archive_path = array_dir / f"{image_path.stem}.npz"
            save_array_archive(result, arrays, archive_path)
            result["array_archive"] = str(archive_path.relative_to(project_root))
        records.append({col: result.get(col) for col in MANIFEST_COLUMNS})

        panel_path = visual_dir / f"{image_path.stem}_m3_sequence.png"
        created = make_visual_panel(result, arrays, panel_path)
        visual_records.append(
            {
                "filename": result["filename"],
                "grade": result["grade"],
                "processing_status": result["processing_status"],
                "visual_path": (
                    str(panel_path.relative_to(project_root)) if created else ""
                ),
                "review_status": "pending researcher visual review",
                "review_notes": "",
            }
        )
        print(f"{number:>2}/{len(PILOT_CASES)} {filename}: {result['processing_status']}")

    runtime_seconds = perf_counter() - start_time
    write_csv(
        pilot_root / "m3_bisenet_pilot_manifest.csv", records, MANIFEST_COLUMNS
    )
    write_csv(
        pilot_root / "m3_bisenet_pilot_visual_audit.csv",
        visual_records,
        (
            "filename",
            "grade",
            "processing_status",
            "visual_path",
            "review_status",
            "review_notes",
        ),
    )
    print(f"BiSeNet pilot completed in {runtime_seconds:.2f}s -> {pilot_root}")
    return {"records": records, "runtime_seconds": runtime_seconds}


def process_operational_dataset(
    project_root: Path,
    run_id: str = DEFAULT_REPRODUCTION_RUN_ID,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    """Process all 1,457 unique images into an operational M3 BiSeNet directory."""
    import onnxruntime as ort

    image_dir = project_root / "Classification" / "JPEGImages"
    classification_dir = project_root / "Classification"
    model_path = project_root / "models" / "bisenet_resnet18_celebamaskhq.onnx"
    m1_manifest_path = project_root / "M1_outputs" / "m1_dataset_manifest.csv"

    authoritative_labels = load_authoritative_labels_dict(
        classification_dir, image_dir
    )
    if m1_manifest_path.is_file():
        m1_df = pd.read_csv(m1_manifest_path)
        validate_m1_identity(
            m1_df["source_path"].to_numpy(dtype=object),
            m1_df["authoritative_grade"].to_numpy(dtype=int),
            authoritative_labels,
            image_dir,
        )

    session = create_face_parsing_session(model_path)
    model_sha256 = sha256_file(model_path)

    if output_root is None:
        if not run_id or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Set a valid alphanumeric run_id")
        out_root = (project_root / "M3_outputs" / run_id).resolve()
    else:
        out_root = Path(output_root).resolve()

    if out_root.exists() and any(out_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite an existing M3 directory (pass --overwrite to allow): {out_root}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    archive_dir = out_root / "arrays"
    archive_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    archive_errors: list[dict[str, str]] = []
    started_utc = datetime.now(timezone.utc).isoformat()
    started_mono = time.perf_counter()

    for filename, grade in authoritative_labels.items():
        result, arrays = process_m3_image(image_dir / filename, grade, session)
        if result["processing_status"] == "success":
            archive_path = archive_dir / f"{Path(filename).stem}.npz"
            try:
                save_array_archive(result, arrays, archive_path)
                issues = validate_saved_archive(archive_path)
                if issues:
                    archive_errors.extend(
                        {"filename": filename, "issue": issue} for issue in issues
                    )
                    raise M3ValidationError(
                        "saved_archive_validation_failed:" + ";".join(issues)
                    )
                result["array_archive"] = (
                    str(archive_path.relative_to(project_root))
                    if archive_path.is_relative_to(project_root)
                    else str(archive_path)
                )
            except Exception as error:
                if archive_path.exists():
                    archive_path.unlink()
                result["processing_status"] = "error"
                result["error_message"] = f"{type(error).__name__}: {error}"
                result["array_archive"] = None
        manifest_rows.append(
            {
                **result,
                "authoritative_grade": grade,
                "error": result["error_message"],
            }
        )
        if (
            len(manifest_rows) % 100 == 0
            or len(manifest_rows) == len(authoritative_labels)
        ):
            print(
                f"M3 progress: {len(manifest_rows)}/{len(authoritative_labels)} attempted; "
                f"{sum(row['processing_status'] == 'success' for row in manifest_rows)} successful",
                flush=True,
            )

    duration_seconds = time.perf_counter() - started_mono
    ended_utc = datetime.now(timezone.utc).isoformat()

    manifest_path = out_root / "m3_bisenet_operational_manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        (*MANIFEST_COLUMNS, "authoritative_grade", "error"),
    )
    # Also write standard m3_bisenet_full_manifest.csv for downstream compatibility
    write_csv(
        out_root / "m3_bisenet_full_manifest.csv",
        manifest_rows,
        MANIFEST_COLUMNS,
    )

    success = [
        row for row in manifest_rows if row["processing_status"] == "success"
    ]
    qa = {
        "attempted_1457": len(manifest_rows) == EXPECTED_TOTAL_IMAGES,
        "unique_filename_count_1457": (
            len({row["filename"] for row in manifest_rows}) == EXPECTED_TOTAL_IMAGES
        ),
        "no_silent_omissions": (
            {row["filename"] for row in manifest_rows} == set(authoritative_labels)
        ),
        "authoritative_grades_preserved": all(
            row["grade"] == authoritative_labels[row["filename"]]
            for row in manifest_rows
        ),
        "successful_rois_nonempty": all(
            int(row["roi_pixel_count"]) > 0 for row in success
        ),
        "no_padding_leakage": all(
            int(row["padding_leak_pixels"]) == 0 for row in success
        ),
        "chromatic_channels_preserved": all(
            row["a_star_unchanged"] and row["b_star_unchanged"] for row in success
        ),
        "successful_archives_validated": not archive_errors,
        "archive_count_matches_successes": (
            len(list(archive_dir.glob("*.npz"))) == len(success)
        ),
    }
    write_csv(
        out_root / "m3_bisenet_operational_qa.csv",
        ({"check": name, "passed": passed} for name, passed in qa.items()),
        ("check", "passed"),
    )
    write_csv(
        out_root / "m3_bisenet_operational_archive_errors.csv",
        archive_errors,
        ("filename", "issue"),
    )
    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "Pillow": pillow_version,
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
        "model_sha256": model_sha256,
        "target_size": TARGET_SIZE,
        "retained_class_ids": RETAINED_CLASS_IDS,
        "clahe_clip_limit": CLAHE_CLIP_LIMIT,
        "clahe_tile_grid_size": CLAHE_TILE_GRID_SIZE,
        "classification_trainval_sha256": sha256_file(
            classification_dir / "NNEW_trainval_0.txt"
        ),
        "classification_test_sha256": sha256_file(
            classification_dir / "NNEW_test_0.txt"
        ),
        "m1_fold_membership_used": False,
        "run_start_utc": started_utc,
        "run_end_utc": ended_utc,
        "processing_duration_seconds": duration_seconds,
    }
    write_csv(
        out_root / "m3_bisenet_operational_metadata.csv",
        ({"field": key, "value": value} for key, value in metadata.items()),
        ("field", "value"),
    )
    summary = {
        "attempted": len(manifest_rows),
        "success": len(success),
        "failure": len(manifest_rows) - len(success),
        "archives": len(list(archive_dir.glob("*.npz"))),
        "qa_passed": sum(qa.values()),
        "qa_total": len(qa),
        "duration_seconds": round(duration_seconds, 2),
    }
    print("Operational M3 summary:", summary)
    return manifest_rows, summary, out_root


def get_m3_artifact_for_filename(
    filename: str,
    manifest_df: pd.DataFrame,
    project_root: Path,
) -> Path:
    """Resolve exactly one successful M3 NPZ artifact using M1's exact filename."""
    matches = manifest_df.loc[manifest_df["filename"] == filename]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one M3 manifest record for {filename}; found {len(matches)}"
        )
    row = matches.iloc[0]
    if row["processing_status"] != "success" or pd.isna(row["array_archive"]):
        raise ValueError(
            f"M3 artifact is unavailable for {filename}: {row['processing_status']}"
        )
    raw_archive = str(row["array_archive"]).replace("\\", "/")
    candidate = Path(raw_archive)
    path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (project_root / candidate).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Invalid or missing M3 archive for {filename}: {path}"
        )
    return path


def load_m3_output(npz_path: Path | str) -> dict[str, np.ndarray]:
    """Load and verify the frozen 11-key M3 NPZ archive contract without pickle."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARCHIVE_SPEC):
            raise ValueError(
                f"M3 archive keys differ from the frozen contract: {npz_path}"
            )
        arrays = {key: archive[key] for key in ARCHIVE_SPEC}
    for key, (shape, dtype) in ARCHIVE_SPEC.items():
        if arrays[key].shape != shape or arrays[key].dtype != dtype:
            raise ValueError(f"M3 array contract mismatch: {npz_path}, {key}")
    if not np.array_equal(
        arrays["final_roi_mask"],
        np.isin(arrays["face_parsing_class_map"], RETAINED_CLASS_IDS)
        & arrays["content_mask"],
    ):
        raise ValueError(f"M3 ROI equation mismatch: {npz_path}")
    return arrays


def compare_m3_arrays(
    expected: np.ndarray,
    actual: np.ndarray,
) -> tuple[int | str, float | str]:
    """Compare two M3 arrays element-by-element."""
    if expected.shape != actual.shape:
        return "not_comparable", "not_comparable"
    different = int(np.count_nonzero(expected != actual))
    if not different:
        return 0, 0.0
    if np.issubdtype(expected.dtype, np.number) and np.issubdtype(
        actual.dtype, np.number
    ):
        maximum = float(
            np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64)))
        )
    else:
        maximum = "not_numeric"
    return different, maximum


def run_fixed_parity(
    project_root: Path,
    reference_manifest_path: Path,
    filenames: tuple[str, ...] = PARITY_FILENAMES,
) -> dict[str, Any]:
    """Run the 7-image fixed parity check against a validated M3 reference manifest."""
    image_dir = project_root / "Classification" / "JPEGImages"
    classification_dir = project_root / "Classification"
    model_path = project_root / "models" / "bisenet_resnet18_celebamaskhq.onnx"
    authoritative_labels = load_authoritative_labels_dict(
        classification_dir, image_dir
    )
    session = create_face_parsing_session(model_path)
    ref_df = pd.read_csv(reference_manifest_path)

    parity_dir = Path(tempfile.mkdtemp(prefix="corecss_m3_operational_parity_"))
    details: list[dict[str, Any]] = []
    for filename in filenames:
        grade = authoritative_labels[filename]
        result, arrays = process_m3_image(image_dir / filename, grade, session)
        if result["processing_status"] != "success":
            raise RuntimeError(
                f"New M3 processing failed for {filename}: {result['error_message']}"
            )
        new_archive = parity_dir / f"{Path(filename).stem}.npz"
        save_array_archive(result, arrays, new_archive)
        new_issues = validate_saved_archive(new_archive)
        if new_issues:
            raise RuntimeError(
                f"New M3 archive validation failed for {filename}: {new_issues}"
            )
        validated_archive = get_m3_artifact_for_filename(
            filename, ref_df, project_root
        )
        with np.load(validated_archive, allow_pickle=False) as old, np.load(
            new_archive, allow_pickle=False
        ) as new:
            for key in ARCHIVE_SPEC:
                old_array, new_array = old[key], new[key]
                differing, maximum = compare_m3_arrays(old_array, new_array)
                details.append(
                    {
                        "filename": filename,
                        "authoritative_grade": grade,
                        "array_key": key,
                        "expected_shape": str(old_array.shape),
                        "actual_shape": str(new_array.shape),
                        "expected_dtype": str(old_array.dtype),
                        "actual_dtype": str(new_array.dtype),
                        "differing_elements": differing,
                        "maximum_absolute_difference": maximum,
                        "exact_match": (
                            old_array.shape == new_array.shape
                            and old_array.dtype == new_array.dtype
                            and np.array_equal(old_array, new_array)
                        ),
                    }
                )

    write_csv(
        parity_dir / "m3_operational_parity_details.csv",
        details,
        tuple(details[0]),
    )
    matches = sum(row["exact_match"] for row in details)
    summary = {
        "temporary_directory": str(parity_dir),
        "filenames": list(filenames),
        "images": len(filenames),
        "arrays_compared": len(details),
        "exact_matches": matches,
        "mismatches": len(details) - matches,
    }
    (parity_dir / "m3_operational_parity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Parity: {matches}/{len(details)} exact array matches across {len(filenames)} images"
    )
    if matches != len(details):
        raise AssertionError("Operational M3 differs from the validated baseline")
    return summary


def verify_m4_handoff(
    project_root: Path,
    m3_manifest_path: Path,
    m1_manifest_path: Path | None = None,
    m1_folds_path: Path | None = None,
) -> dict[str, Any]:
    """Verify M1->M3->M4 handoff resolution across all 14,570 fold memberships and demo cases."""
    m1_manifest_file = m1_manifest_path or (
        project_root / "M1_outputs" / "m1_dataset_manifest.csv"
    )
    m1_folds_file = m1_folds_path or (
        project_root / "M1_outputs" / "m1_5x2_fold_assignments.csv"
    )
    m1_manifest_df = pd.read_csv(m1_manifest_file)
    m1_folds_df = pd.read_csv(m1_folds_file)
    operational_manifest_df = pd.read_csv(m3_manifest_path)

    handoff_filenames = [
        m1_manifest_df.loc[
            m1_manifest_df["authoritative_grade"] == grade, "filename"
        ].iloc[0]
        for grade in (0, 1, 2, 3)
    ] + ["levle3_120.jpg"]

    for demo_filename in handoff_filenames:
        assert demo_filename in set(m1_folds_df["filename"])
        assert (m1_manifest_df["filename"] == demo_filename).sum() == 1
        demo_npz = get_m3_artifact_for_filename(
            demo_filename, operational_manifest_df, project_root
        )
        demo_arrays = load_m3_output(demo_npz)
        assert set(demo_arrays) == set(ARCHIVE_SPEC)
        assert all(
            demo_arrays[key].shape == ARCHIVE_SPEC[key][0] for key in ARCHIVE_SPEC
        )
        grade_val = int(
            m1_manifest_df.loc[
                m1_manifest_df["filename"] == demo_filename, "authoritative_grade"
            ].iloc[0]
        )
        print(
            f"M4 handoff: {demo_filename} grade {grade_val} => {demo_npz} | "
            "11 keys and 512x512 shapes PASS"
        )

    manifest_counts = Counter(operational_manifest_df["filename"])
    by_filename = {
        str(row["filename"]): row for _, row in operational_manifest_df.iterrows()
    }
    fold_unresolved = 0
    for filename in m1_folds_df["filename"]:
        if manifest_counts.get(filename, 0) != 1:
            fold_unresolved += 1
        else:
            row = by_filename[filename]
            try:
                _ = get_m3_artifact_for_filename(
                    filename, operational_manifest_df, project_root
                )
            except Exception:
                fold_unresolved += 1
            if row["processing_status"] != "success":
                fold_unresolved += 1

    resolved_rows = len(m1_folds_df) - fold_unresolved
    print(
        f"M1->M3->M4 fold coverage: {resolved_rows}/{len(m1_folds_df)} memberships resolved"
    )
    print(
        "M4-ready M3 arrays:",
        ", ".join(M4_EXPOSED_M3_KEYS),
        "(no M4 features calculated)",
    )
    return {
        "tested_filenames": handoff_filenames,
        "fold_membership_rows": len(m1_folds_df),
        "rows_resolved": resolved_rows,
        "unresolved_rows": fold_unresolved,
        "exposed_m3_keys": list(M4_EXPOSED_M3_KEYS),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3 BiSeNet Final Preprocessing and M4 Handoff Pipeline"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to project root (auto-discovered if omitted).",
    )
    parser.add_argument(
        "--mode",
        choices=("pilot", "operational", "handoff-verify", "parity-7"),
        default="operational",
        help="Execution mode: 13-image pilot, 1,457-image operational run, M4 handoff verification, or 7-image parity check.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=DEFAULT_REPRODUCTION_RUN_ID,
        help=f"Run directory identifier under M3_outputs/ (default: {DEFAULT_REPRODUCTION_RUN_ID}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory override.",
    )
    parser.add_argument(
        "--m3-manifest",
        type=Path,
        default=None,
        help="Path to M3 manifest CSV (for handoff-verify or parity-7).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = find_project_root(
        args.project_root,
        require_jpeg_dir=(args.mode != "handoff-verify"),
    )
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
    elif args.mode == "handoff-verify":
        manifest_path = args.m3_manifest or (
            (out_dir or (root / "M3_outputs" / args.run_id))
            / "m3_bisenet_operational_manifest.csv"
        )
        verify_m4_handoff(root, manifest_path)
    elif args.mode == "parity-7":
        ref_manifest = args.m3_manifest or (
            root / "M3_outputs" / "M3_bisenet_final" / "m3_bisenet_full_manifest.csv"
        )
        run_fixed_parity(root, ref_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
