#!/usr/bin/env python3
"""Shared M3 preprocessing utilities, constants, and validation helpers.

Extracted from:
- Module3_Image_Preprocessing.ipynb
- M3_Final_Image_Preprocessing.ipynb
- M3_Final_BiSeNet_Preprocessing.ipynb
- M3_BiSeNet_Preprocessing_M4_Handoff.ipynb
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

TARGET_SIZE: int = 512
PADDING_COLOR: tuple[int, int, int] = (0, 0, 0)
RETAINED_CLASS_IDS: tuple[int, int] = (1, 10)
CLASS_NAMES: tuple[str, ...] = (
    "background",
    "skin",
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "eyeglasses",
    "left_ear",
    "right_ear",
    "earring",
    "nose",
    "inner_mouth",
    "upper_lip",
    "lower_lip",
    "neck",
    "necklace",
    "clothing",
    "hair",
    "hat",
)
FACE_PARSING_CLASSES: dict[int, str] = {
    idx: name for idx, name in enumerate(CLASS_NAMES)
}
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)
EXPECTED_MODEL_SHA256: str = (
    "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"
)
EXPECTED_GRADE_COUNTS: dict[int, int] = {0: 513, 1: 633, 2: 182, 3: 129}
EXPECTED_TOTAL_IMAGES: int = 1457

FACE_PARSING_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
FACE_PARSING_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PARSING_PALETTE = np.array(
    [
        (0, 0, 0),
        (255, 205, 160),
        (145, 50, 110),
        (195, 70, 100),
        (75, 90, 180),
        (85, 140, 210),
        (80, 210, 245),
        (220, 135, 70),
        (245, 150, 80),
        (180, 125, 180),
        (245, 190, 95),
        (200, 55, 70),
        (225, 80, 110),
        (235, 105, 130),
        (110, 180, 95),
        (95, 170, 115),
        (80, 125, 75),
        (80, 70, 55),
        (100, 100, 160),
    ],
    dtype=np.uint8,
)


class M3ValidationError(RuntimeError):
    """Raised when an M3 preprocessing step fails an objective technical check."""


def find_project_root(
    start_path: Path | str | None = None,
    require_jpeg_dir: bool = False,
) -> Path:
    """Discover the CORECS5 THESIS project root from the given path or cwd."""
    if start_path is not None:
        explicit = Path(start_path).resolve()
        if (explicit / "Classification" / "NNEW_trainval_0.txt").is_file():
            if not require_jpeg_dir or (explicit / "Classification" / "JPEGImages").is_dir():
                return explicit

    current = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    candidates = (current, current / "CORECS5 THESIS", *current.parents)
    for candidate in candidates:
        has_labels = (
            (candidate / "Classification" / "NNEW_trainval_0.txt").is_file()
            and (candidate / "Classification" / "NNEW_test_0.txt").is_file()
        )
        has_jpegs = (candidate / "Classification" / "JPEGImages").is_dir()
        if require_jpeg_dir and has_jpegs:
            return candidate
        if not require_jpeg_dir and (has_jpegs or has_labels):
            return candidate

    raise FileNotFoundError(
        "Could not locate project root containing Classification/NNEW_trainval_0.txt "
        "(and Classification/JPEGImages when required)."
    )


def sha256_file(path: Path | str) -> str:
    """Compute the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(
    path: Path | str,
    rows: Iterable[dict[str, Any]],
    columns: Iterable[str],
) -> None:
    """Write dictionary rows to a UTF-8 CSV with exact column order."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_classification_list(file_path: Path | str) -> pd.DataFrame:
    """Read filename, grade, and lesion_count from one Acne04 classification text file."""
    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["filename", "grade", "lesion_count"],
    )


def load_authoritative_labels_df(
    classification_dir: Path | str,
    image_dir: Path | str | None = None,
    verify_disk_images: bool = True,
) -> pd.DataFrame:
    """Load and validate the 1,457 authoritative labels as a sorted DataFrame."""
    c_dir = Path(classification_dir)
    labels = pd.concat(
        [
            read_classification_list(c_dir / "NNEW_trainval_0.txt"),
            read_classification_list(c_dir / "NNEW_test_0.txt"),
        ],
        ignore_index=True,
    )
    if labels["filename"].duplicated().any():
        duplicates = labels.loc[
            labels["filename"].duplicated(keep=False), "filename"
        ].tolist()
        raise ValueError(f"Duplicate authoritative labels: {duplicates[:10]}")

    labels["grade"] = labels["grade"].astype(int)
    if set(labels["grade"].unique()) != {0, 1, 2, 3}:
        raise ValueError("Authoritative labels must contain exactly grades 0-3.")

    if len(labels) != EXPECTED_TOTAL_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_IMAGES} authoritative records, got {len(labels)}"
        )

    observed_counts = dict(Counter(labels["grade"].tolist()))
    if observed_counts != EXPECTED_GRADE_COUNTS:
        raise ValueError(
            f"Authoritative grade distribution differs from approved dataset: {observed_counts}"
        )

    if verify_disk_images and image_dir is not None:
        i_dir = Path(image_dir)
        image_names = {
            path.name
            for path in i_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        }
        label_names = set(labels["filename"])
        missing_images = sorted(label_names - image_names)
        unlabeled_images = sorted(image_names - label_names)
        if missing_images or unlabeled_images:
            raise ValueError(
                f"Label/image mismatch: missing={len(missing_images)}, "
                f"unlabeled={len(unlabeled_images)}"
            )

    return labels.sort_values("filename").reset_index(drop=True)


def load_authoritative_labels_dict(
    classification_dir: Path | str,
    image_dir: Path | str | None = None,
    verify_disk_images: bool = True,
) -> dict[str, int]:
    """Load and validate the 1,457 authoritative labels as a filename->grade dictionary."""
    df = load_authoritative_labels_df(
        classification_dir=classification_dir,
        image_dir=image_dir,
        verify_disk_images=verify_disk_images,
    )
    return dict(zip(df["filename"], df["grade"].astype(int)))


def standardize_image(
    image_path: Path | str,
    target_size: int = TARGET_SIZE,
    padding_color: tuple[int, int, int] = PADDING_COLOR,
) -> tuple[Image.Image, np.ndarray, np.ndarray, dict[str, Any]]:
    """Standardize an image to a centered 512x512 RGB canvas while preserving aspect ratio.

    Returns
    -------
    original : PIL.Image.Image
        EXIF-transposed RGB PIL image at original dimensions.
    standardized_rgb : np.ndarray
        (target_size, target_size, 3) uint8 RGB array.
    content_mask : np.ndarray
        (target_size, target_size) bool array where True marks valid image pixels.
    metadata : dict[str, Any]
        Geometry and resize diagnostic fields.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        exif_orientation = int(image.getexif().get(274, 1))
        original = ImageOps.exif_transpose(image).convert("RGB")

    original_width, original_height = original.size
    scale = min(target_size / original_width, target_size / original_height)
    new_width = max(1, round(original_width * scale))
    new_height = max(1, round(original_height * scale))

    if scale < 1:
        interpolation, resize_operation = Image.Resampling.LANCZOS, "downscaled"
    elif scale > 1:
        interpolation, resize_operation = Image.Resampling.BICUBIC, "upscaled"
    else:
        interpolation, resize_operation = Image.Resampling.NEAREST, "unchanged"

    resized = original.resize((new_width, new_height), interpolation)
    standardized = Image.new("RGB", (target_size, target_size), padding_color)
    left_padding = (target_size - new_width) // 2
    top_padding = (target_size - new_height) // 2
    right_padding = target_size - new_width - left_padding
    bottom_padding = target_size - new_height - top_padding
    standardized.paste(resized, (left_padding, top_padding))

    content_mask = np.zeros((target_size, target_size), dtype=bool)
    content_mask[
        top_padding : top_padding + new_height,
        left_padding : left_padding + new_width,
    ] = True

    if original_width > original_height:
        orientation = "landscape"
    elif original_height > original_width:
        orientation = "portrait"
    else:
        orientation = "square"

    metadata = {
        "filename": image_path.name,
        "original_width": original_width,
        "original_height": original_height,
        "original_size": f"{original_width} × {original_height}",
        "resized_content": f"{new_width} × {new_height}",
        "final_size": f"{target_size} × {target_size}",
        "orientation": orientation,
        "exif_orientation": exif_orientation,
        "resized_width": new_width,
        "resized_height": new_height,
        "scale_factor": scale,
        "resize_operation": resize_operation,
        "operation": resize_operation,
        "left_padding": left_padding,
        "top_padding": top_padding,
        "right_padding": right_padding,
        "bottom_padding": bottom_padding,
    }
    return original, np.asarray(standardized, dtype=np.uint8), content_mask, metadata


def prepare_color_representations(
    standardized_rgb: np.ndarray,
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid_size: tuple[int, int] = CLAHE_TILE_GRID_SIZE,
    target_size: int = TARGET_SIZE,
) -> dict[str, np.ndarray]:
    """Compute CIELAB (L*, a*, b*), ordinary grayscale, and CLAHE-enhanced L*."""
    rgb_u8 = np.asarray(standardized_rgb, dtype=np.uint8)
    rgb_float = rgb_u8.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    l_star = lab[:, :, 0].copy()
    a_star = lab[:, :, 1].copy()
    b_star = lab[:, :, 2].copy()
    grayscale = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)

    l_star_u8 = np.clip(np.rint(l_star * (255.0 / 100.0)), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip_limit),
        tileGridSize=tuple(clahe_tile_grid_size),
    )
    clahe_l_star_u8 = clahe.apply(l_star_u8)
    clahe_l_star = clahe_l_star_u8.astype(np.float32) * (100.0 / 255.0)

    representations = {
        "lab_l_star": l_star,
        "lab_a_star": a_star,
        "lab_b_star": b_star,
        "grayscale": grayscale,
        "clahe_l_star": clahe_l_star,
    }
    if any(value.shape != (target_size, target_size) for value in representations.values()):
        raise M3ValidationError("color_representation_invalid_shape")
    if any(not np.isfinite(value).all() for value in representations.values()):
        raise M3ValidationError("color_representation_contains_nonfinite_values")
    if not np.array_equal(a_star, lab[:, :, 1]) or not np.array_equal(b_star, lab[:, :, 2]):
        raise M3ValidationError("chromatic_channel_was_modified")
    return representations


def tile_rgb(array: np.ndarray, size: int = 220) -> Image.Image:
    """Create a centered thumbnail tile for visual QA sheets."""
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), "RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (22, 22, 22))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def gray_rgb(array: np.ndarray) -> np.ndarray:
    """Convert a 2-D uint8 array into a 3-channel RGB uint8 array."""
    return np.repeat(np.asarray(array, dtype=np.uint8)[..., np.newaxis], 3, axis=2)


def make_visual_panel(
    result: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output_path: Path | str,
) -> bool:
    """Save a 10-panel M3 visual inspection sequence PNG."""
    standard = arrays.get("standardized_rgb")
    if standard is None:
        return False
    roi = arrays.get("final_roi_mask")
    available: list[tuple[str, np.ndarray]] = [
        ("Original", arrays.get("original_rgb", standard)),
        ("Standardized RGB", standard),
        ("Content mask", gray_rgb(arrays["content_mask"].astype(np.uint8) * 255)),
    ]
    if "face_parsing_class_map" in arrays:
        available.append(
            ("19-class parsing", PARSING_PALETTE[arrays["face_parsing_class_map"]])
        )
    if roi is not None:
        available.append(("Skin + nose ROI", gray_rgb(roi.astype(np.uint8) * 255)))
        available.append(("Masked RGB", np.where(roi[..., None], standard, 0)))
    if result.get("processing_status") == "success":
        available.extend(
            [
                (
                    "CIELAB L*",
                    gray_rgb(np.clip(np.rint(arrays["lab_l_star"] * 255 / 100), 0, 255)),
                ),
                (
                    "CIELAB a*",
                    gray_rgb(np.clip(np.rint(arrays["lab_a_star"] + 128), 0, 255)),
                ),
                ("Ordinary grayscale", gray_rgb(arrays["grayscale"])),
                (
                    "CLAHE L*",
                    gray_rgb(np.clip(np.rint(arrays["clahe_l_star"] * 255 / 100), 0, 255)),
                ),
            ]
        )
    columns, tile_size, label_height = 5, 220, 34
    rows = (len(available) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_size, rows * (tile_size + label_height) + 64),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    headline = (
        f"{result['filename']} | authoritative Grade {result['grade']} | "
        f"{result['processing_status']}"
    )
    draw.text((10, 9), headline, fill="black")
    if result.get("error_message"):
        draw.text((10, 32), str(result["error_message"])[:145], fill="red")
    for index, (label, array) in enumerate(available):
        x = (index % columns) * tile_size
        y = 64 + (index // columns) * (tile_size + label_height)
        sheet.paste(tile_rgb(array, tile_size), (x, y))
        draw.text((x + 6, y + tile_size + 5), label, fill="black")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True
