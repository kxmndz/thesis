#!/usr/bin/env python3
"""Shared M3 preprocessing utilities, constants, and validation helpers (Layered-Labs/ACNE04)."""

from __future__ import annotations

import csv
import hashlib
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
FACE_PARSING_CLASSES: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)
EXPECTED_MODEL_SHA256: str = (
    "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f"
)

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

_M1_PADDING_META_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


class M3ValidationError(RuntimeError):
    """Raised when an M3 preprocessing step fails an objective technical check."""


def find_project_root(
    start_path: Path | str | None = None,
    require_jpeg_dir: bool = False,
) -> Path:
    """Discover the acne_severity_pipeline project root strictly from M1_outputs or Classification/JPEGImages."""
    if start_path is not None:
        explicit = Path(start_path).resolve()
        if (explicit / "M1_outputs" / "m1_dataset_manifest.csv").is_file():
            if not require_jpeg_dir or (explicit / "Classification" / "JPEGImages").is_dir():
                return explicit

    current = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    for candidate in (current, *current.parents):
        has_manifest = (candidate / "M1_outputs" / "m1_dataset_manifest.csv").is_file()
        has_jpegs = (candidate / "Classification" / "JPEGImages").is_dir()
        if require_jpeg_dir and has_jpegs:
            return candidate
        if not require_jpeg_dir and (has_manifest or has_jpegs):
            return candidate

    return Path(start_path).resolve() if start_path is not None else current


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


def load_authoritative_labels_dict(
    project_or_classification_dir: Path | str,
    include_excluded: bool = False,
) -> dict[str, int]:
    """Load authoritative `level` labels strictly from M1's Layered-Labs/ACNE04 manifests."""
    base = Path(project_or_classification_dir).resolve()
    root = base.parent if base.name == "Classification" else base
    manifest_name = "m1_full_dataset_manifest.csv" if include_excluded else "m1_dataset_manifest.csv"
    m1_manifest = root / "M1_outputs" / manifest_name
    if not m1_manifest.is_file():
        raise FileNotFoundError(
            f"Missing M1 manifest: {m1_manifest}. Run M1 on Layered-Labs/ACNE04 first."
        )
    df = pd.read_csv(m1_manifest)
    grade_col = "authoritative_grade" if "authoritative_grade" in df.columns else "level"
    return {str(fn): int(g) for fn, g in zip(df["filename"], df[grade_col])}


def load_m1_padding_metadata(project_root: Path | str) -> dict[str, dict[str, Any]]:
    """Load per-filename 512x512 letterbox geometry exported by M1."""
    root = Path(project_root).resolve()
    key = str(root)
    if key in _M1_PADDING_META_CACHE:
        return _M1_PADDING_META_CACHE[key]

    meta_map: dict[str, dict[str, Any]] = {}
    for name in ("m1_full_dataset_manifest.csv", "m1_dataset_manifest.csv"):
        p = root / "M1_outputs" / name
        if p.is_file():
            df = pd.read_csv(p)
            required = {"filename", "left_padding", "top_padding", "resized_width", "resized_height"}
            if required.issubset(df.columns):
                for row in df.to_dict(orient="records"):
                    meta_map[str(row["filename"])] = row
                break
    _M1_PADDING_META_CACHE[key] = meta_map
    return meta_map


def standardize_image(
    image_path: Path | str,
    target_size: int = TARGET_SIZE,
    padding_color: tuple[int, int, int] = PADDING_COLOR,
    padding_meta: dict[str, Any] | None = None,
) -> tuple[Image.Image, np.ndarray, np.ndarray, dict[str, Any]]:
    """Standardize an image to a centered 512x512 RGB canvas while preserving aspect ratio.

    When reading a 512x512 image already standardized by M1, restores the exact non-padded
    `content_mask` using M1's recorded letterbox padding metadata.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        exif_orientation = int(image.getexif().get(274, 1))
        original = ImageOps.exif_transpose(image).convert("RGB")

    original_width, original_height = original.size

    if (
        padding_meta is None
        and original_width == target_size
        and original_height == target_size
        and image_path.parent.name in {"JPEGImages", "JPEGImages_excluded"}
    ):
        m1_root = image_path.parents[2]
        padding_meta = load_m1_padding_metadata(m1_root).get(image_path.name)

    if (
        original_width == target_size
        and original_height == target_size
        and padding_meta is not None
        and int(padding_meta.get("std_width", target_size)) == target_size
    ):
        left_padding = int(padding_meta["left_padding"])
        top_padding = int(padding_meta["top_padding"])
        new_width = int(padding_meta["resized_width"])
        new_height = int(padding_meta["resized_height"])
        orig_w = int(padding_meta.get("orig_width", target_size))
        orig_h = int(padding_meta.get("orig_height", target_size))
        scale = float(padding_meta.get("scale_factor", 1.0))
        standardized = original
        resize_operation = "pre_standardized_m1"
    else:
        orig_w, orig_h = original_width, original_height
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
        standardized.paste(resized, (left_padding, top_padding))

    right_padding = target_size - new_width - left_padding
    bottom_padding = target_size - new_height - top_padding

    content_mask = np.zeros((target_size, target_size), dtype=bool)
    content_mask[
        top_padding : top_padding + new_height,
        left_padding : left_padding + new_width,
    ] = True

    orientation = "landscape" if orig_w > orig_h else ("portrait" if orig_h > orig_w else "square")
    metadata = {
        "filename": image_path.name,
        "original_width": orig_w,
        "original_height": orig_h,
        "original_size": f"{orig_w} × {orig_h}",
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

    return {
        "lab_l_star": l_star,
        "lab_a_star": a_star,
        "lab_b_star": b_star,
        "grayscale": grayscale,
        "clahe_l_star": clahe_l_star,
    }


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
        available.append(("19-class parsing", PARSING_PALETTE[arrays["face_parsing_class_map"]]))
    if roi is not None:
        available.append(("Skin + nose ROI", gray_rgb(roi.astype(np.uint8) * 255)))
        available.append(("Masked RGB", np.where(roi[..., None], standard, 0)))
    if result.get("processing_status") == "success":
        available.extend(
            [
                ("CIELAB L*", gray_rgb(np.clip(np.rint(arrays["lab_l_star"] * 255 / 100), 0, 255))),
                ("CIELAB a*", gray_rgb(np.clip(np.rint(arrays["lab_a_star"] + 128), 0, 255))),
                ("Ordinary grayscale", gray_rgb(arrays["grayscale"])),
                ("CLAHE L*", gray_rgb(np.clip(np.rint(arrays["clahe_l_star"] * 255 / 100), 0, 255))),
            ]
        )
    columns, tile_size, label_height = 5, 220, 34
    rows = (len(available) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height) + 64), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (10, 9),
        f"{result['filename']} | authoritative Grade {result['grade']} | {result['processing_status']}",
        fill="black",
    )
    for index, (label, array) in enumerate(available):
        x = (index % columns) * tile_size
        y = 64 + (index // columns) * (tile_size + label_height)
        sheet.paste(tile_rgb(array, tile_size), (x, y))
        draw.text((x + 6, y + tile_size + 5), label, fill="black")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True
