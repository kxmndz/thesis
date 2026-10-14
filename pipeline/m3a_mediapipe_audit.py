#!/usr/bin/env python3
"""M3A – Initial MediaPipe Facial-Skin Segmentation and Face Landmark Audits.

Extracted from:
- Module3_Image_Preprocessing.ipynb

Responsibilities:
1. Audit image dimensions and select representative test samples (common portrait,
   landscape, square, smallest image).
2. Perform Step 3A Initial Facial-Skin Segmentation using MediaPipe Multiclass Selfie
   Segmentation (category 3 = facial skin, category 1 = hair) intersected with content_mask.
3. Run Step 3B Dataset-Wide Initial Mask-Quality Audit and Step 3C Boundary (>= 0.01) /
   High-Ratio Inspection.
4. Run Step 4A Face Landmark Detection Test and Step 4B Dataset-Wide Face Landmark Audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from m3_common import (
    TARGET_SIZE,
    find_project_root,
    load_authoritative_labels_df,
    standardize_image,
)

CANDIDATE_MINIMUM_SKIN_RATIO: float = 0.01

LANDMARK_TEST_RECORDS: list[dict[str, str]] = [
    {"description": "Common portrait", "filename": "levle0_1.jpg"},
    {"description": "Landscape profile", "filename": "levle0_17.jpg"},
    {"description": "Valid low-ratio mask", "filename": "levle3_112.jpg"},
    {"description": "Fragmented mask", "filename": "levle0_59.jpg"},
    {"description": "Fragmented close-up", "filename": "levle3_62.jpg"},
    {"description": "High-ratio close-up", "filename": "levle3_63.jpg"},
]


def _get_mediapipe():
    """Import mediapipe lazily so non-MediaPipe CLI commands do not require it."""
    try:
        import mediapipe as mp
        return mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required for M3A/M3B commands. Install `mediapipe` in your Python environment."
        ) from exc


def create_segmenter_options(segmentation_model_path: Path | str):
    """Configure MediaPipe ImageSegmenterOptions for still-image multiclass segmentation."""
    mp = _get_mediapipe()
    if not Path(segmentation_model_path).is_file():
        raise FileNotFoundError(
            f"Missing MediaPipe segmentation model: {segmentation_model_path}"
        )
    return mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(segmentation_model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False,
    )


def create_face_landmarker_options(face_landmarker_model_path: Path | str):
    """Configure MediaPipe FaceLandmarkerOptions for single-face still-image detection."""
    mp = _get_mediapipe()
    if not Path(face_landmarker_model_path).is_file():
        raise FileNotFoundError(
            f"Missing MediaPipe face landmarker model: {face_landmarker_model_path}"
        )
    return mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(face_landmarker_model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )


def extract_initial_skin_mask(
    standardized_image: np.ndarray | Image.Image,
    content_mask: np.ndarray | Image.Image,
    segmenter: Any,
) -> dict[str, np.ndarray]:
    """Extract MediaPipe category 3 facial-skin mask intersected with content_mask."""
    mp = _get_mediapipe()
    image_array = np.ascontiguousarray(np.asarray(standardized_image, dtype=np.uint8))
    mediapipe_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_array,
    )
    segmentation_result = segmenter.segment(mediapipe_image)
    if segmentation_result.category_mask is None:
        raise RuntimeError("MediaPipe did not return a category mask.")

    category_mask = segmentation_result.category_mask.numpy_view().copy()
    if category_mask.ndim == 3 and category_mask.shape[-1] == 1:
        category_mask = np.squeeze(category_mask, axis=-1)

    valid_content = np.asarray(content_mask) > 0
    face_skin_mask = (category_mask == 3) & valid_content
    hair_mask = (category_mask == 1) & valid_content

    masked_skin_image = np.zeros_like(image_array)
    masked_skin_image[face_skin_mask] = image_array[face_skin_mask]

    return {
        "standardized_image": image_array,
        "category_mask": category_mask,
        "face_skin_mask": face_skin_mask,
        "hair_mask": hair_mask,
        "masked_skin_image": masked_skin_image,
    }


def detect_face_landmarks(
    standardized_image: np.ndarray | Image.Image,
    landmarker: Any,
) -> dict[str, Any]:
    """Detect facial landmarks in one standardized 512x512 RGB image."""
    mp = _get_mediapipe()
    image_array = np.ascontiguousarray(np.asarray(standardized_image, dtype=np.uint8))
    mediapipe_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_array,
    )
    detection_result = landmarker.detect(mediapipe_image)
    number_of_faces = len(detection_result.face_landmarks)

    if number_of_faces == 0:
        return {
            "detected": False,
            "number_of_faces": 0,
            "landmark_count": 0,
            "landmark_coordinates": [],
            "raw_result": detection_result,
        }

    detected_landmarks = detection_result.face_landmarks[0]
    image_height, image_width = image_array.shape[:2]
    landmark_coordinates: list[tuple[int, int]] = []
    for landmark in detected_landmarks:
        x_coord = int(
            np.clip(round(landmark.x * (image_width - 1)), 0, image_width - 1)
        )
        y_coord = int(
            np.clip(round(landmark.y * (image_height - 1)), 0, image_height - 1)
        )
        landmark_coordinates.append((x_coord, y_coord))

    return {
        "detected": True,
        "number_of_faces": number_of_faces,
        "landmark_count": len(landmark_coordinates),
        "landmark_coordinates": landmark_coordinates,
        "raw_result": detection_result,
    }


def run_dimension_audit(
    image_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit image dimensions and select representative test samples."""
    image_files = sorted(image_dir.glob("*.jpg"))
    dimension_counts: Counter[tuple[int, int]] = Counter()
    unreadable_images: list[tuple[str, str]] = []
    image_information: list[dict[str, Any]] = []

    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                dimension_counts[(image.width, image.height)] += 1
                corrected = ImageOps.exif_transpose(image)
                width, height = corrected.size
            image_information.append(
                {
                    "path": image_path,
                    "width": width,
                    "height": height,
                    "area": width * height,
                    "aspect_ratio": width / height,
                }
            )
        except Exception as error:
            unreadable_images.append((image_path.name, str(error)))

    square_images = sum(
        count for (w, h), count in dimension_counts.items() if w == h
    )
    non_square_images = len(image_files) - square_images

    print("Image folder:", image_dir)
    print("Total JPG images:", len(image_files))
    print("Square images:", square_images)
    print("Non-square images:", non_square_images)
    print("Unreadable images:", len(unreadable_images))

    test_samples: list[tuple[str, Path]] = []
    if image_information:
        size_counter = Counter(
            (item["width"], item["height"]) for item in image_information
        )
        most_common_size = size_counter.most_common(1)[0][0]
        common_portrait = next(
            item
            for item in image_information
            if (item["width"], item["height"]) == most_common_size
        )
        landscape = next(
            (item for item in image_information if item["aspect_ratio"] > 1.3),
            image_information[0],
        )
        square = next(
            (item for item in image_information if item["width"] == item["height"]),
            image_information[0],
        )
        smallest = min(image_information, key=lambda item: item["area"])
        test_samples = [
            ("Common portrait", common_portrait["path"]),
            ("Landscape", landscape["path"]),
            ("Square", square["path"]),
            ("Smallest image", smallest["path"]),
        ]

    if output_dir is not None and test_samples:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(
            len(test_samples), 3, figsize=(13, 4 * len(test_samples)), squeeze=False
        )
        for row, (description, img_path) in enumerate(test_samples):
            original, standardized, content_mask, _meta = standardize_image(img_path)
            axes[row, 0].imshow(original)
            axes[row, 0].set_title(f"{description}\nOriginal: {original.size}")
            axes[row, 1].imshow(standardized)
            axes[row, 1].set_title(f"Standardized: ({TARGET_SIZE}, {TARGET_SIZE})")
            axes[row, 2].imshow(
                content_mask.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255
            )
            axes[row, 2].set_title("Content Mask")
            for col in range(3):
                axes[row, col].axis("off")
        plt.tight_layout()
        fig.savefig(
            output_dir / "Module3_Step2_Resolution_Standardization.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    return {
        "total_images": len(image_files),
        "square_images": square_images,
        "non_square_images": non_square_images,
        "unreadable_images": unreadable_images,
        "dimension_counts": dimension_counts,
        "test_samples": test_samples,
    }


def run_initial_mask_audit(
    project_root: Path,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run Step 3B Initial Mask-Quality Audit and Step 3C Boundary/High-Ratio Inspection."""
    mp = _get_mediapipe()
    image_dir = project_root / "Classification" / "JPEGImages"
    classification_dir = project_root / "Classification"
    model_path = project_root / "models" / "selfie_multiclass_256x256.tflite"
    out_root = output_dir or (project_root / "M3_outputs" / "M3A_audit")
    out_root.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.jpg"))
    segmenter_options = create_segmenter_options(model_path)
    mask_audit_records: list[dict[str, Any]] = []
    audit_start = perf_counter()

    with mp.tasks.vision.ImageSegmenter.create_from_options(
        segmenter_options
    ) as segmenter:
        for image_number, image_path in enumerate(image_files, start=1):
            try:
                _orig, standardized, content_mask, resize_metadata = standardize_image(
                    image_path
                )
                result = extract_initial_skin_mask(
                    standardized, content_mask, segmenter
                )
                valid_content_pixels = int(np.count_nonzero(content_mask))
                face_skin_pixels = int(np.count_nonzero(result["face_skin_mask"]))
                hair_pixels = int(np.count_nonzero(result["hair_mask"]))
                skin_ratio = (
                    face_skin_pixels / valid_content_pixels
                    if valid_content_pixels > 0
                    else 0.0
                )
                mask_audit_records.append(
                    {
                        "filename": image_path.name,
                        "original_size": resize_metadata["original_size"],
                        "resized_content": resize_metadata["resized_content"],
                        "resize_operation": resize_metadata["operation"],
                        "scale_factor": resize_metadata["scale_factor"],
                        "valid_content_pixels": valid_content_pixels,
                        "face_skin_pixels": face_skin_pixels,
                        "hair_pixels": hair_pixels,
                        "skin_ratio": skin_ratio,
                        "status": "success",
                        "error": None,
                    }
                )
            except Exception as error:
                mask_audit_records.append(
                    {
                        "filename": image_path.name,
                        "original_size": None,
                        "resized_content": None,
                        "resize_operation": None,
                        "scale_factor": None,
                        "valid_content_pixels": None,
                        "face_skin_pixels": None,
                        "hair_pixels": None,
                        "skin_ratio": None,
                        "status": "error",
                        "error": str(error),
                    }
                )
            if image_number % 100 == 0 or image_number == len(image_files):
                print(f"M3A mask audit: {image_number}/{len(image_files)} images")

    audit_duration = perf_counter() - audit_start
    initial_mask_audit = pd.DataFrame(mask_audit_records)
    complete_labels = load_authoritative_labels_df(
        classification_dir, image_dir, verify_disk_images=False
    )
    mask_audit_with_labels = initial_mask_audit.merge(
        complete_labels, on="filename", how="left", validate="one_to_one"
    )
    mask_audit_with_labels.to_csv(out_root / "m3a_initial_mask_audit.csv", index=False)
    print(
        f"M3A initial segmentation audit completed in {audit_duration / 60:.2f} min."
    )
    return mask_audit_with_labels


def run_landmark_audit(
    project_root: Path,
    initial_mask_audit: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run Step 4B Dataset-Wide Face Landmark Audit across all standardized images."""
    mp = _get_mediapipe()
    image_dir = project_root / "Classification" / "JPEGImages"
    classification_dir = project_root / "Classification"
    landmarker_path = project_root / "models" / "face_landmarker.task"
    out_root = output_dir or (project_root / "M3_outputs" / "M3A_audit")
    out_root.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.jpg"))
    face_landmarker_options = create_face_landmarker_options(landmarker_path)
    landmark_audit_records: list[dict[str, Any]] = []
    start_time = perf_counter()

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        face_landmarker_options
    ) as landmarker:
        for image_number, image_path in enumerate(image_files, start=1):
            try:
                original, standardized, _content_mask, resize_metadata = (
                    standardize_image(image_path)
                )
                landmark_result = detect_face_landmarks(standardized, landmarker)
                original_width, original_height = original.size
                orientation = resize_metadata["orientation"]
                landmark_audit_records.append(
                    {
                        "filename": image_path.name,
                        "original_width": original_width,
                        "original_height": original_height,
                        "orientation": orientation,
                        "resize_operation": resize_metadata["operation"],
                        "scale_factor": resize_metadata["scale_factor"],
                        "face_detected": landmark_result["detected"],
                        "number_of_faces": landmark_result["number_of_faces"],
                        "landmark_count": landmark_result["landmark_count"],
                        "status": "success",
                        "error": None,
                    }
                )
            except Exception as error:
                landmark_audit_records.append(
                    {
                        "filename": image_path.name,
                        "original_width": None,
                        "original_height": None,
                        "orientation": None,
                        "resize_operation": None,
                        "scale_factor": None,
                        "face_detected": False,
                        "number_of_faces": 0,
                        "landmark_count": 0,
                        "status": "error",
                        "error": str(error),
                    }
                )
            if image_number % 100 == 0 or image_number == len(image_files):
                print(f"M3A landmark audit: {image_number}/{len(image_files)} images")

    duration = perf_counter() - start_time
    face_landmark_audit = pd.DataFrame(landmark_audit_records)
    complete_labels = load_authoritative_labels_df(
        classification_dir, image_dir, verify_disk_images=False
    )
    if initial_mask_audit is not None:
        face_landmark_audit = face_landmark_audit.merge(
            initial_mask_audit[["filename", "face_skin_pixels", "skin_ratio"]],
            on="filename",
            how="left",
            validate="one_to_one",
        )
    face_landmark_audit = face_landmark_audit.merge(
        complete_labels[["filename", "grade", "lesion_count"]],
        on="filename",
        how="left",
        validate="one_to_one",
    )
    face_landmark_audit.to_csv(out_root / "m3a_face_landmark_audit.csv", index=False)
    print(f"M3A face landmark audit completed in {duration / 60:.2f} min.")
    return face_landmark_audit


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3A MediaPipe Facial-Skin Segmentation and Face Landmark Audits"
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
        help="Directory for M3A audit outputs (defaults to <project_root>/M3_outputs/M3A_audit).",
    )
    parser.add_argument(
        "--mode",
        choices=("dimensions", "mask-audit", "landmark-audit", "all"),
        default="all",
        help="Which M3A audit step(s) to execute.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
