#!/usr/bin/env python3
"""M3B Skin-Retention Alternative and Frozen M3 MediaPipe+Landmark Baseline Pipeline.

Extracted from:
- M3B_Skin_Retention_Alternative.ipynb
- M3_Final_Image_Preprocessing.ipynb

Responsibilities:
1. Subtract verified MediaPipe Face Landmarker convex hulls (left/right eye,
   left/right eyebrow, mouth/lips) when landmarks are detected, or fall back to
   the raw MediaPipe category-3 facial-skin mask when landmarks are not detected.
2. Export M3B PNG masks (raw_face_skin_masks, feature_exclusion_masks, selected_skin_masks)
   and m3b_export_audit.csv.
3. Run the frozen M3 MediaPipe+Landmark-Fallback preprocessing pipeline (10-image pilot
   or 1,457-image full export) producing 8-key compressed NPZ archives, manifest CSVs,
   and 15-check technical QA.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from m3_common import (
    M3ValidationError,
    TARGET_SIZE,
    find_project_root,
    load_authoritative_labels_df,
    prepare_color_representations,
    standardize_image,
)
from m3a_mediapipe_audit import (
    _get_mediapipe,
    create_face_landmarker_options,
    create_segmenter_options,
    detect_face_landmarks,
    extract_initial_skin_mask,
)

M3_MEDIAPIPE_PILOT_CASES: list[tuple[str, str]] = [
    ("levle0_1.jpg", "standard portrait / landmark detected"),
    ("levle0_17.jpg", "landscape profile / landmark fallback"),
    ("levle0_19.jpg", "square image"),
    ("levle2_93.jpg", "smallest image / historically low ROI"),
    ("levle3_112.jpg", "low ROI / landmark detected"),
    ("levle0_59.jpg", "fragmented MediaPipe ROI / fallback"),
    ("levle3_63.jpg", "close-up / high ROI / fallback"),
    ("levle0_73.jpg", "rotated difficult case"),
    ("levle1_74.jpg", "grade 1 / high ROI / fallback"),
    ("levle3_120.jpg", "historical zero-ROI technical failure"),
]

M3_MEDIAPIPE_MANIFEST_COLUMNS: list[str] = [
    "filename",
    "grade",
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
    "initial_roi_pixels",
    "roi_pixel_count",
    "roi_ratio",
    "landmark_status",
    "landmark_count",
    "landmark_error",
    "selection_mode",
    "padding_leak_pixels",
    "a_star_unchanged",
    "b_star_unchanged",
    "processing_status",
    "error_message",
    "array_archive",
]

M3_MEDIAPIPE_ARCHIVE_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "standardized_rgb": ((TARGET_SIZE, TARGET_SIZE, 3), np.dtype("uint8")),
    "content_mask": ((TARGET_SIZE, TARGET_SIZE), np.dtype("bool")),
    "final_roi_mask": ((TARGET_SIZE, TARGET_SIZE), np.dtype("bool")),
    "lab_l_star": ((TARGET_SIZE, TARGET_SIZE), np.dtype("float32")),
    "lab_a_star": ((TARGET_SIZE, TARGET_SIZE), np.dtype("float32")),
    "lab_b_star": ((TARGET_SIZE, TARGET_SIZE), np.dtype("float32")),
    "grayscale": ((TARGET_SIZE, TARGET_SIZE), np.dtype("uint8")),
    "clahe_l_star": ((TARGET_SIZE, TARGET_SIZE), np.dtype("float32")),
}


def get_feature_connection_groups() -> dict[str, Any]:
    """Return MediaPipe's documented semantic connection groups for non-skin exclusion."""
    mp = _get_mediapipe()
    conns = mp.tasks.vision.FaceLandmarksConnections
    return {
        "left_eye": conns.FACE_LANDMARKS_LEFT_EYE,
        "right_eye": conns.FACE_LANDMARKS_RIGHT_EYE,
        "left_eyebrow": conns.FACE_LANDMARKS_LEFT_EYEBROW,
        "right_eyebrow": conns.FACE_LANDMARKS_RIGHT_EYEBROW,
        "mouth_and_lips": conns.FACE_LANDMARKS_LIPS,
    }


def connection_landmark_indices(connections: Iterable) -> list[int]:
    """Return the unique sorted landmark indices used by a connection group."""
    indices: set[int] = set()
    for connection in connections:
        indices.add(connection.start)
        indices.add(connection.end)
    return sorted(indices)


def convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Compute a 2-D monotonic-chain convex hull without external dependencies."""
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    def cross(origin, point_a, point_b):
        return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (
            point_a[1] - origin[1]
        ) * (point_b[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[int, int]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def build_feature_exclusion_mask(
    landmark_coordinates: list[tuple[int, int]],
    image_shape: tuple[int, ...],
) -> np.ndarray:
    """Fill documented eye, eyebrow, and lip/mouth convex hulls into a boolean mask."""
    image_height, image_width = image_shape[:2]
    exclusion_image = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(exclusion_image)

    for connections in get_feature_connection_groups().values():
        indices = connection_landmark_indices(connections)
        region_points = [
            landmark_coordinates[index]
            for index in indices
            if index < len(landmark_coordinates)
        ]
        region_hull = convex_hull(region_points)
        if len(region_hull) >= 3:
            draw.polygon(region_hull, fill=255)

    return np.asarray(exclusion_image, dtype=np.uint8) > 0


def build_skin_retention_result(
    standardized_image: np.ndarray | Image.Image,
    raw_face_skin_mask: np.ndarray,
    landmarker: Any,
) -> dict[str, Any]:
    """Create the provisional M3B selected mask (or raw fallback) for one image."""
    raw_mask = np.asarray(raw_face_skin_mask, dtype=bool)
    landmark_result = detect_face_landmarks(standardized_image, landmarker)

    if landmark_result["detected"]:
        feature_exclusion_mask = build_feature_exclusion_mask(
            landmark_result["landmark_coordinates"], raw_mask.shape
        )
        selected_mask = raw_mask & ~feature_exclusion_mask
        selection_mode = "landmark_refined_eyes_brows_mouth"
    else:
        feature_exclusion_mask = np.zeros_like(raw_mask, dtype=bool)
        selected_mask = raw_mask.copy()
        selection_mode = "raw_skin_fallback"

    raw_skin_pixels = int(np.count_nonzero(raw_mask))
    selected_skin_pixels = int(np.count_nonzero(selected_mask))
    removed_skin_pixels = raw_skin_pixels - selected_skin_pixels
    retained_fraction = (
        selected_skin_pixels / raw_skin_pixels if raw_skin_pixels > 0 else 0.0
    )

    return {
        "face_detected": landmark_result["detected"],
        "landmark_count": landmark_result["landmark_count"],
        "selection_mode": selection_mode,
        "raw_face_skin_mask": raw_mask,
        "feature_exclusion_mask": feature_exclusion_mask,
        "selected_skin_mask": selected_mask,
        "raw_skin_pixels": raw_skin_pixels,
        "selected_skin_pixels": selected_skin_pixels,
        "removed_skin_pixels": removed_skin_pixels,
        "retained_fraction": retained_fraction,
    }


def refine_skin_mask(
    standardized_image: np.ndarray | Image.Image,
    initial_roi_mask: np.ndarray,
    landmarker: Any,
) -> dict[str, Any]:
    """Refine initial MediaPipe skin mask by subtracting landmarks with exception fallback."""
    initial_roi = np.asarray(initial_roi_mask, dtype=bool)
    landmark_error = None
    try:
        landmark_result = detect_face_landmarks(standardized_image, landmarker)
    except Exception as error:
        landmark_result = {"detected": False, "landmark_count": 0}
        landmark_error = str(error)

    if landmark_result["detected"]:
        exclusion_mask = build_feature_exclusion_mask(
            landmark_result["landmark_coordinates"], initial_roi.shape
        )
        final_roi = initial_roi & ~exclusion_mask
        landmark_status = "detected"
        selection_mode = "landmark_refined_eyes_brows_mouth"
    else:
        exclusion_mask = np.zeros_like(initial_roi, dtype=bool)
        final_roi = initial_roi.copy()
        landmark_status = "fallback"
        selection_mode = "mediapipe_roi_fallback"

    return {
        "landmark_status": landmark_status,
        "landmark_count": int(landmark_result["landmark_count"]),
        "landmark_error": landmark_error,
        "selection_mode": selection_mode,
        "feature_exclusion_mask": exclusion_mask,
        "final_roi_mask": final_roi,
    }


def validate_mediapipe_masks(
    standardized_rgb: np.ndarray,
    content_mask: np.ndarray,
    final_roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate shapes, nonzero ROI, and zero padding leakage for MediaPipe M3."""
    rgb = np.asarray(standardized_rgb, dtype=np.uint8)
    if rgb.shape != (TARGET_SIZE, TARGET_SIZE, 3):
        raise M3ValidationError(f"standardized_rgb_invalid_shape:{rgb.shape}")

    content = np.asarray(content_mask) != 0
    roi = np.asarray(final_roi_mask) != 0
    if content.shape != (TARGET_SIZE, TARGET_SIZE):
        raise M3ValidationError(f"content_mask_invalid_shape:{content.shape}")
    if roi.shape != (TARGET_SIZE, TARGET_SIZE):
        raise M3ValidationError(f"final_roi_mask_invalid_shape:{roi.shape}")

    roi_pixels = int(np.count_nonzero(roi))
    padding_leak_pixels = int(np.count_nonzero(roi & ~content))
    if roi_pixels == 0:
        raise M3ValidationError("final_roi_has_zero_usable_pixels")
    if padding_leak_pixels != 0:
        raise M3ValidationError(f"final_roi_includes_padding:{padding_leak_pixels}")

    return content, roi, roi_pixels, padding_leak_pixels


def process_m3_mediapipe_image(
    image_path: Path,
    grade: int,
    segmenter: Any,
    landmarker: Any,
) -> dict[str, Any]:
    """Execute the one-image M3 MediaPipe+Landmark-Fallback pipeline."""
    result: dict[str, Any] = {
        "filename": image_path.name,
        "grade": int(grade),
        "processing_status": "error",
        "error_message": None,
        "landmark_status": "not_attempted",
        "landmark_count": 0,
        "landmark_error": None,
        "selection_mode": None,
        "valid_content_pixels": None,
        "initial_roi_pixels": None,
        "roi_pixel_count": None,
        "roi_ratio": None,
        "padding_leak_pixels": None,
        "a_star_unchanged": None,
        "b_star_unchanged": None,
    }
    try:
        original, standardized_rgb, content_mask, resize_metadata = standardize_image(
            image_path
        )
        result.update(resize_metadata)
        result["original_rgb"] = np.asarray(original, dtype=np.uint8)
        result["standardized_rgb"] = standardized_rgb
        result["content_mask"] = content_mask
        result["valid_content_pixels"] = int(np.count_nonzero(content_mask))

        segmentation = extract_initial_skin_mask(
            standardized_rgb, content_mask, segmenter
        )
        initial_roi = np.asarray(segmentation["face_skin_mask"], dtype=bool)
        result["initial_roi_mask"] = initial_roi
        result["hair_mask"] = np.asarray(segmentation["hair_mask"], dtype=bool)
        result["initial_roi_pixels"] = int(np.count_nonzero(initial_roi))

        refinement = refine_skin_mask(standardized_rgb, initial_roi, landmarker)
        result.update(refinement)

        content_mask, final_roi, roi_pixels, padding_leak_pixels = (
            validate_mediapipe_masks(
                standardized_rgb, content_mask, refinement["final_roi_mask"]
            )
        )
        result["content_mask"] = content_mask
        result["final_roi_mask"] = final_roi
        result["roi_pixel_count"] = roi_pixels
        result["padding_leak_pixels"] = padding_leak_pixels
        result["roi_ratio"] = (
            roi_pixels / result["valid_content_pixels"]
            if result["valid_content_pixels"] > 0
            else 0.0
        )

        colors = prepare_color_representations(standardized_rgb)
        lab_check = cv2.cvtColor(
            standardized_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB
        )
        a_unchanged = bool(np.array_equal(colors["lab_a_star"], lab_check[:, :, 1]))
        b_unchanged = bool(np.array_equal(colors["lab_b_star"], lab_check[:, :, 2]))
        if not a_unchanged or not b_unchanged:
            raise M3ValidationError("chromatic_channel_was_modified")

        result.update(
            {
                "l_star": colors["lab_l_star"],
                "a_star": colors["lab_a_star"],
                "b_star": colors["lab_b_star"],
                "grayscale": colors["grayscale"],
                "clahe_l_star": colors["clahe_l_star"],
                "a_star_unchanged": a_unchanged,
                "b_star_unchanged": b_unchanged,
                "processing_status": "success",
            }
        )
    except Exception as error:
        if (
            result.get("roi_pixel_count") is None
            and result.get("final_roi_mask") is not None
        ):
            result["roi_pixel_count"] = int(
                np.count_nonzero(result["final_roi_mask"])
            )
            if result.get("valid_content_pixels"):
                result["roi_ratio"] = (
                    result["roi_pixel_count"] / result["valid_content_pixels"]
                )
        result["error_message"] = f"{type(error).__name__}: {error}"
    return result


def save_m3_mediapipe_archive(result: dict[str, Any], output_path: Path) -> None:
    """Save the 8-key compressed NPZ archive for a successful M3 MediaPipe image."""
    if result["processing_status"] != "success":
        raise ValueError("Only successful M3 results may be archived.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        standardized_rgb=result["standardized_rgb"],
        content_mask=result["content_mask"],
        final_roi_mask=result["final_roi_mask"],
        lab_l_star=result["l_star"],
        lab_a_star=result["a_star"],
        lab_b_star=result["b_star"],
        grayscale=result["grayscale"],
        clahe_l_star=result["clahe_l_star"],
    )


def validate_saved_mediapipe_archive(archive_path: Path) -> list[str]:
    """Verify keys, shapes, dtypes, and finite values of a saved M3 MediaPipe NPZ."""
    issues: list[str] = []
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            for key, (expected_shape, expected_dtype) in M3_MEDIAPIPE_ARCHIVE_SPEC.items():
                if key not in archive:
                    issues.append(f"missing_key:{key}")
                    continue
                array = archive[key]
                if array.shape != expected_shape:
                    issues.append(f"{key}_shape:{array.shape}")
                if array.dtype != expected_dtype:
                    issues.append(f"{key}_dtype:{array.dtype}")
                if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                    issues.append(f"{key}_contains_nonfinite_values")
    except Exception as error:
        issues.append(f"archive_read_error:{type(error).__name__}:{error}")
    return issues


def save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save a boolean mask as an 8-bit black-and-white PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(output_path)


def export_m3b_dataset(
    project_root: Path,
    output_root: Path | None = None,
) -> pd.DataFrame:
    """Export raw, feature-exclusion, and selected M3B masks as PNGs without excluding images."""
    mp = _get_mediapipe()
    image_dir = project_root / "Classification" / "JPEGImages"
    m3b_root = output_root or (
        project_root / "M3_outputs" / "M3B_skin_retention_alternative"
    )
    raw_mask_dir = m3b_root / "raw_face_skin_masks"
    exclusion_dir = m3b_root / "feature_exclusion_masks"
    selected_mask_dir = m3b_root / "selected_skin_masks"
    for directory in (raw_mask_dir, exclusion_dir, selected_mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.jpg"))
    seg_opts = create_segmenter_options(
        project_root / "models" / "selfie_multiclass_256x256.tflite"
    )
    land_opts = create_face_landmarker_options(
        project_root / "models" / "face_landmarker.task"
    )

    export_records: list[dict[str, Any]] = []
    start_time = perf_counter()

    with mp.tasks.vision.ImageSegmenter.create_from_options(
        seg_opts
    ) as segmenter, mp.tasks.vision.FaceLandmarker.create_from_options(
        land_opts
    ) as landmarker:
        for image_number, image_path in enumerate(image_files, start=1):
            try:
                _orig, standardized, content_mask, _meta = standardize_image(image_path)
                segmentation_result = extract_initial_skin_mask(
                    standardized, content_mask, segmenter
                )
                retention_result = build_skin_retention_result(
                    standardized, segmentation_result["face_skin_mask"], landmarker
                )
                output_filename = f"{image_path.stem}.png"
                save_binary_mask(
                    retention_result["raw_face_skin_mask"],
                    raw_mask_dir / output_filename,
                )
                save_binary_mask(
                    retention_result["feature_exclusion_mask"],
                    exclusion_dir / output_filename,
                )
                save_binary_mask(
                    retention_result["selected_skin_mask"],
                    selected_mask_dir / output_filename,
                )
                valid_content_pixels = int(np.count_nonzero(content_mask))
                selected_ratio = (
                    retention_result["selected_skin_pixels"] / valid_content_pixels
                    if valid_content_pixels > 0
                    else 0.0
                )
                export_records.append(
                    {
                        "filename": image_path.name,
                        "selection_mode": retention_result["selection_mode"],
                        "face_detected": retention_result["face_detected"],
                        "landmark_count": retention_result["landmark_count"],
                        "valid_content_pixels": valid_content_pixels,
                        "raw_skin_pixels": retention_result["raw_skin_pixels"],
                        "removed_skin_pixels": retention_result["removed_skin_pixels"],
                        "selected_skin_pixels": retention_result["selected_skin_pixels"],
                        "retained_fraction": retention_result["retained_fraction"],
                        "selected_skin_ratio": selected_ratio,
                        "status": "success",
                        "error": None,
                    }
                )
            except Exception as error:
                export_records.append(
                    {
                        "filename": image_path.name,
                        "selection_mode": None,
                        "face_detected": False,
                        "landmark_count": 0,
                        "valid_content_pixels": None,
                        "raw_skin_pixels": None,
                        "removed_skin_pixels": None,
                        "selected_skin_pixels": None,
                        "retained_fraction": None,
                        "selected_skin_ratio": None,
                        "status": "error",
                        "error": str(error),
                    }
                )
            if image_number % 100 == 0 or image_number == len(image_files):
                print(f"Exported {image_number}/{len(image_files)} M3B images...")

    export_table = pd.DataFrame(export_records)
    audit_csv_path = m3b_root / "m3b_export_audit.csv"
    export_table.to_csv(audit_csv_path, index=False)
    print(
        f"M3B export completed in {(perf_counter() - start_time) / 60:.2f} min -> {audit_csv_path}"
    )
    return export_table


def run_m3_mediapipe_pipeline(
    project_root: Path,
    mode: str = "full",
    output_root: Path | None = None,
) -> pd.DataFrame:
    """Run the M3_Final_Image_Preprocessing (MediaPipe + Landmark Fallback) pipeline."""
    mp = _get_mediapipe()
    image_dir = project_root / "Classification" / "JPEGImages"
    classification_dir = project_root / "Classification"
    seg_model_path = project_root / "models" / "selfie_multiclass_256x256.tflite"
    land_model_path = project_root / "models" / "face_landmarker.task"

    labels_df = load_authoritative_labels_df(classification_dir, image_dir)
    label_lookup = labels_df.set_index("filename")["grade"].to_dict()

    if mode == "pilot":
        out_root = output_root or (project_root / "M3_outputs" / "M3_final_pilot")
        selection = [
            (image_dir / filename, reason)
            for filename, reason in M3_MEDIAPIPE_PILOT_CASES
        ]
        manifest_name = "m3_final_pilot_manifest.csv"
    else:
        out_root = output_root or (project_root / "M3_outputs" / "M3_final_full")
        selection = [
            (image_dir / filename, "") for filename in labels_df["filename"].tolist()
        ]
        manifest_name = "m3_preprocessing_manifest.csv"

    out_root.mkdir(parents=True, exist_ok=True)
    array_dir = out_root / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)

    seg_opts = create_segmenter_options(seg_model_path)
    land_opts = create_face_landmarker_options(land_model_path)
    records: list[dict[str, Any]] = []
    start_time = perf_counter()

    with mp.tasks.vision.ImageSegmenter.create_from_options(
        seg_opts
    ) as segmenter, mp.tasks.vision.FaceLandmarker.create_from_options(
        land_opts
    ) as landmarker:
        for index, (image_path, audit_reason) in enumerate(selection, start=1):
            result = process_m3_mediapipe_image(
                image_path, label_lookup[image_path.name], segmenter, landmarker
            )
            result["audit_reason"] = audit_reason
            archive_path = None
            if result["processing_status"] == "success":
                archive_path = array_dir / f"{image_path.stem}.npz"
                save_m3_mediapipe_archive(result, archive_path)
            record = {
                col: (
                    str(archive_path.relative_to(project_root))
                    if col == "array_archive" and archive_path is not None
                    else result.get(col)
                )
                for col in M3_MEDIAPIPE_MANIFEST_COLUMNS
            }
            records.append(record)
            if index % 100 == 0 or index == len(selection):
                print(f"M3 MediaPipe ({mode}): {index}/{len(selection)} images")

    manifest = pd.DataFrame(records, columns=M3_MEDIAPIPE_MANIFEST_COLUMNS)
    manifest_path = out_root / manifest_name
    manifest.to_csv(manifest_path, index=False)
    elapsed = perf_counter() - start_time
    print(
        f"M3 MediaPipe ({mode}) completed in {elapsed:.1f}s -> {manifest_path} "
        f"(success={(manifest['processing_status'] == 'success').sum()}/{len(manifest)})"
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3B Skin-Retention Alternative and Frozen M3 MediaPipe+Landmark Pipeline"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to project root (auto-discovered if omitted).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("m3b-png-export", "mediapipe-pilot", "mediapipe-full"),
        default="mediapipe-full",
        help="Execution mode: M3B PNG mask export, M3 MediaPipe 10-image pilot, or M3 MediaPipe 1,457-image full NPZ run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = find_project_root(args.project_root, require_jpeg_dir=True)
    out_dir = args.output_dir.resolve() if args.output_dir else None

    if args.mode == "m3b-png-export":
        export_m3b_dataset(root, out_dir)
    elif args.mode == "mediapipe-pilot":
        run_m3_mediapipe_pipeline(root, mode="pilot", output_root=out_dir)
    else:
        run_m3_mediapipe_pipeline(root, mode="full", output_root=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
