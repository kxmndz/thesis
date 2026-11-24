#!/usr/bin/env python3
"""M4 – Explicit Chromatic (CIELAB a*), Luminance Texture (CLAHE L* GLCM),
and Scale-Normalized Chromatic Texture (CIELAB a* GLCM) Feature Extraction.

Extracts 40 explicit, interpretable visual features strictly from real images
and real M3 skin+nose ROI masks:
  - 4 whole-ROI CIELAB a* erythema statistics (mean, median, variance, p90)
  - 20 luminance masked 32-level GLCM features on CLAHE L* (4 distances x 4 descriptors + 4 means)
  - 16 tight-cropped, baseline-centered chromatic masked 32-level GLCM features on CIELAB a*
    (4 distances x 4 descriptors), addressing scale variance and red-lesion spatial clustering.

Contains zero synthetic data generation or simulated feature fallbacks.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

from m1_data_loader import assert_no_watermarked_filenames
from m3_bisenet_pipeline import process_m3_image
from m3_common import (
    find_project_root,
    prepare_color_representations,
    standardize_image,
)
from m3c_face_parsing import create_face_parsing_session

LOGGER = logging.getLogger(__name__)

GLCM_QUANTIZATION_LEVELS: int = 32
DEFAULT_GLCM_DISTANCES: tuple[int, ...] = (1, 3, 5, 10)
CHROMATIC_GLCM_ROI_SIZE: int = 384
CHROMATIC_A_CENTERED_RANGE: tuple[float, float] = (-8.0, 24.0)

GLCM_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "contrast",
    "correlation",
    "energy",
    "homogeneity",
)

CHROMATIC_FEATURE_NAMES: tuple[str, ...] = (
    "a_star_mean",
    "a_star_median",
    "a_star_variance",
    "a_star_p90",
)


def get_feature_columns(
    distances: Sequence[int] = DEFAULT_GLCM_DISTANCES,
    include_chromatic_glcm: bool = True,
) -> list[str]:
    """Return the ordered list of all 40 explicit M4 chromatic and GLCM texture feature names."""
    cols = list(CHROMATIC_FEATURE_NAMES)
    for d in distances:
        for prop in GLCM_DESCRIPTOR_NAMES:
            cols.append(f"glcm_d{d}_{prop}")
    for prop in GLCM_DESCRIPTOR_NAMES:
        cols.append(f"glcm_{prop}_mean")
    if include_chromatic_glcm:
        for d in distances:
            for prop in GLCM_DESCRIPTOR_NAMES:
                cols.append(f"a_glcm_d{d}_{prop}")
    return cols


def extract_chromatic_features(
    lab_a_star: np.ndarray,
    final_roi_mask: np.ndarray,
) -> dict[str, float]:
    """Extract mean, median, variance, and 90th percentile of CIELAB a* within the skin ROI."""
    roi = np.asarray(final_roi_mask, dtype=bool)
    a_values = np.asarray(lab_a_star, dtype=np.float64)[roi]
    if a_values.size == 0:
        raise ValueError("final_roi_mask contains zero pixels; cannot extract chromatic features.")

    return {
        "a_star_mean": float(np.mean(a_values)),
        "a_star_median": float(np.median(a_values)),
        "a_star_variance": float(np.var(a_values)),
        "a_star_p90": float(np.percentile(a_values, 90)),
    }


def quantize_to_32_levels(
    image_2d: np.ndarray,
    value_range: tuple[float, float] = (0.0, 100.0),
    levels: int = GLCM_QUANTIZATION_LEVELS,
) -> np.ndarray:
    """Quantize a 2-D plane into `levels` discrete integers (0 .. levels-1)."""
    low, high = value_range
    scaled = (np.asarray(image_2d, dtype=np.float64) - low) / max(high - low, 1e-9)
    clipped = np.clip(scaled, 0.0, 1.0 - 1e-12)
    return np.floor(clipped * levels).astype(np.int32)


def _compute_masked_glcm_single_offset(
    quantized: np.ndarray,
    roi_mask: np.ndarray,
    dy: int,
    dx: int,
    levels: int = GLCM_QUANTIZATION_LEVELS,
) -> np.ndarray:
    """Compute the symmetric, normalized (levels x levels) GLCM ignoring outside-mask pixel pairs."""
    h, w = quantized.shape
    y0_start, y0_end = max(0, -dy), min(h, h - dy)
    x0_start, x0_end = max(0, -dx), min(w, w - dx)
    y1_start, y1_end = y0_start + dy, y0_end + dy
    x1_start, x1_end = x0_start + dx, x0_end + dx

    m0 = roi_mask[y0_start:y0_end, x0_start:x0_end]
    m1 = roi_mask[y1_start:y1_end, x1_start:x1_end]
    valid_pairs = m0 & m1
    if not np.any(valid_pairs):
        return np.zeros((levels, levels), dtype=np.float64)

    q0 = quantized[y0_start:y0_end, x0_start:x0_end][valid_pairs]
    q1 = quantized[y1_start:y1_end, x1_start:x1_end][valid_pairs]

    flat_indices = q0 * levels + q1
    counts = np.bincount(flat_indices, minlength=levels * levels).reshape(
        (levels, levels)
    ).astype(np.float64)
    sym_counts = counts + counts.T
    total = float(np.sum(sym_counts))
    if total <= 0.0:
        return np.zeros((levels, levels), dtype=np.float64)
    return sym_counts / total


def _haralick_descriptors_from_glcm(glcm: np.ndarray) -> dict[str, float]:
    """Compute Contrast, Correlation, Energy, and Homogeneity from a normalized GLCM."""
    levels = glcm.shape[0]
    if float(np.sum(glcm)) <= 0.0:
        return {
            "contrast": 0.0,
            "correlation": 0.0,
            "energy": 0.0,
            "homogeneity": 0.0,
        }

    i_idx, j_idx = np.ogrid[0:levels, 0:levels]
    diff = (i_idx - j_idx).astype(np.float64)
    diff_sq = diff ** 2

    contrast = float(np.sum(glcm * diff_sq))
    energy = float(np.sqrt(np.sum(glcm ** 2)))
    homogeneity = float(np.sum(glcm / (1.0 + diff_sq)))

    p_i = np.sum(glcm, axis=1)
    p_j = np.sum(glcm, axis=0)
    mu_i = float(np.sum(np.arange(levels, dtype=np.float64) * p_i))
    mu_j = float(np.sum(np.arange(levels, dtype=np.float64) * p_j))
    sigma_i = float(
        np.sqrt(np.sum(((np.arange(levels, dtype=np.float64) - mu_i) ** 2) * p_i))
    )
    sigma_j = float(
        np.sqrt(np.sum(((np.arange(levels, dtype=np.float64) - mu_j) ** 2) * p_j))
    )
    if sigma_i * sigma_j <= 1e-12:
        correlation = 1.0
    else:
        corr_num = np.sum(((i_idx - mu_i) * (j_idx - mu_j)) * glcm)
        correlation = float(corr_num / (sigma_i * sigma_j))

    return {
        "contrast": contrast,
        "correlation": correlation,
        "energy": energy,
        "homogeneity": homogeneity,
    }


def extract_texture_features(
    texture_plane: np.ndarray,
    final_roi_mask: np.ndarray,
    value_range: tuple[float, float] = (0.0, 100.0),
    distances: Sequence[int] = DEFAULT_GLCM_DISTANCES,
    levels: int = GLCM_QUANTIZATION_LEVELS,
) -> dict[str, float]:
    """Compute rotation-invariant 32-level masked GLCM descriptors across distances (1, 3, 5, 10)."""
    roi = np.asarray(final_roi_mask, dtype=bool)
    quantized = quantize_to_32_levels(texture_plane, value_range=value_range, levels=levels)

    features: dict[str, float] = {}
    per_prop_across_d: dict[str, list[float]] = {
        prop: [] for prop in GLCM_DESCRIPTOR_NAMES
    }

    for d in distances:
        offsets = ((0, d), (-d, d), (-d, 0), (-d, -d))
        angle_props: dict[str, list[float]] = {
            prop: [] for prop in GLCM_DESCRIPTOR_NAMES
        }
        for dy, dx in offsets:
            glcm = _compute_masked_glcm_single_offset(
                quantized, roi, dy=dy, dx=dx, levels=levels
            )
            props = _haralick_descriptors_from_glcm(glcm)
            for prop in GLCM_DESCRIPTOR_NAMES:
                angle_props[prop].append(props[prop])

        for prop in GLCM_DESCRIPTOR_NAMES:
            rot_inv_val = float(np.mean(angle_props[prop]))
            features[f"glcm_d{d}_{prop}"] = rot_inv_val
            per_prop_across_d[prop].append(rot_inv_val)

    for prop in GLCM_DESCRIPTOR_NAMES:
        features[f"glcm_{prop}_mean"] = float(np.mean(per_prop_across_d[prop]))

    return features


def extract_chromatic_texture_features(
    lab_a_star: np.ndarray,
    final_roi_mask: np.ndarray,
    distances: Sequence[int] = DEFAULT_GLCM_DISTANCES,
    levels: int = GLCM_QUANTIZATION_LEVELS,
    roi_resize_dim: int | None = CHROMATIC_GLCM_ROI_SIZE,
    value_range: tuple[float, float] = CHROMATIC_A_CENTERED_RANGE,
) -> dict[str, float]:
    """Compute rotation-invariant 32-level masked chromatic GLCM descriptors on CIELAB a*.

    1. Subtracts the per-image median skin a* to remove baseline skin-tone/white-balance shift.
    2. Tightly crops to the facial skin ROI bounding box and resizes to a uniform scale (384x384)
       so GLCM pixel offsets (d=1, 3, 5, 10) capture consistent physical lesion scales across
       close-up and full-face ACNE04 images.
    3. Quantizes centered a* in [-8, +24] (~1 CIELAB unit/bin) into 32 discrete levels and
       computes masked Haralick descriptors strictly over in-mask skin pixel pairs.
    """
    roi = np.asarray(final_roi_mask, dtype=bool)
    a_star = np.asarray(lab_a_star, dtype=np.float32)
    a_values = a_star[roi]
    if a_values.size == 0:
        raise ValueError("final_roi_mask contains zero pixels; cannot extract chromatic GLCM features.")

    a_median = float(np.median(a_values))
    ys, xs = np.where(roi)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    a_crop = a_star[y0:y1, x0:x1] - a_median
    m_crop = roi[y0:y1, x0:x1]

    if roi_resize_dim is not None and roi_resize_dim > 0:
        a_input = cv2.resize(
            a_crop,
            (int(roi_resize_dim), int(roi_resize_dim)),
            interpolation=cv2.INTER_LINEAR,
        )
        m_input = (
            cv2.resize(
                m_crop.astype(np.uint8),
                (int(roi_resize_dim), int(roi_resize_dim)),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
    else:
        a_input = a_crop
        m_input = m_crop

    quantized = quantize_to_32_levels(a_input, value_range=value_range, levels=levels)
    features: dict[str, float] = {}

    for d in distances:
        offsets = ((0, d), (-d, d), (-d, 0), (-d, -d))
        angle_props: dict[str, list[float]] = {
            prop: [] for prop in GLCM_DESCRIPTOR_NAMES
        }
        for dy, dx in offsets:
            glcm = _compute_masked_glcm_single_offset(
                quantized, m_input, dy=dy, dx=dx, levels=levels
            )
            props = _haralick_descriptors_from_glcm(glcm)
            for prop in GLCM_DESCRIPTOR_NAMES:
                angle_props[prop].append(props[prop])

        for prop in GLCM_DESCRIPTOR_NAMES:
            features[f"a_glcm_d{d}_{prop}"] = float(np.mean(angle_props[prop]))

    return features


def extract_m4_features_from_arrays(
    arrays: dict[str, np.ndarray],
    texture_source: str = "clahe_l_star",
    distances: Sequence[int] = DEFAULT_GLCM_DISTANCES,
    include_chromatic_glcm: bool = True,
) -> dict[str, float]:
    """Extract the complete 40-feature M4 vector (4 a* + 20 luminance GLCM + 16 chromatic a* GLCM)."""
    roi = arrays["final_roi_mask"]
    chromatic = extract_chromatic_features(arrays["lab_a_star"], roi)

    if texture_source == "clahe_l_star":
        plane = arrays["clahe_l_star"]
        val_range = (0.0, 100.0)
    elif texture_source == "grayscale":
        plane = arrays["grayscale"]
        val_range = (0.0, 255.0)
    elif texture_source == "lab_l_star":
        plane = arrays["lab_l_star"]
        val_range = (0.0, 100.0)
    else:
        raise ValueError(f"Unsupported texture_source: {texture_source}")

    texture = extract_texture_features(
        plane, roi, value_range=val_range, distances=distances
    )
    if include_chromatic_glcm:
        chromatic_texture = extract_chromatic_texture_features(
            arrays["lab_a_star"], roi, distances=distances
        )
        return {**chromatic, **texture, **chromatic_texture}
    return {**chromatic, **texture}


def extract_m4_features_for_single_image(
    image_path: Path | str,
    onnx_session: Any | None = None,
    texture_source: str = "clahe_l_star",
    distances: Sequence[int] = DEFAULT_GLCM_DISTANCES,
    include_chromatic_glcm: bool = True,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, Any]]:
    """Run M3 preprocessing + M4 feature extraction on a single real image file."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Cannot extract M4 features; image file not found: {image_path}")

    saved_mask_path = image_path.parents[2] / "M3_outputs" / "roi_masks" / f"{image_path.stem}.png"
    if saved_mask_path.is_file():
        original, std_rgb, content_mask, meta = standardize_image(image_path)
        colors = prepare_color_representations(std_rgb)
        mask_u8 = cv2.imread(str(saved_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None or mask_u8.shape != content_mask.shape:
            raise RuntimeError(f"Invalid or corrupted M3 ROI mask at {saved_mask_path}")
        skin_roi = (mask_u8 > 0) & content_mask
        if int(np.count_nonzero(skin_roi)) == 0:
            raise ValueError(f"Saved M3 ROI mask contains 0 valid skin pixels: {saved_mask_path}")
        arrays = {
            "original_rgb": np.asarray(original, dtype=np.uint8),
            "standardized_rgb": std_rgb,
            "content_mask": content_mask,
            "face_parsing_class_map": skin_roi.astype(np.uint8),
            "skin_mask": skin_roi,
            "nose_mask": np.zeros_like(skin_roi, dtype=bool),
            "final_roi_mask": skin_roi,
            **colors,
        }
        result = {
            "filename": image_path.name,
            "grade": 0,
            "processing_status": "success",
            "error_message": None,
            "roi_pixel_count": int(np.count_nonzero(skin_roi)),
            "valid_content_pixels": int(np.count_nonzero(content_mask)),
            "roi_ratio": float(
                np.count_nonzero(skin_roi) / max(1, np.count_nonzero(content_mask))
            ),
            **meta,
        }
    elif onnx_session is not None:
        result, arrays = process_m3_image(image_path, grade=0, session=onnx_session)
    else:
        raise FileNotFoundError(
            f"Missing M3 mask ({saved_mask_path}) and no BiSeNet ONNX session provided for {image_path}."
        )

    features = extract_m4_features_from_arrays(
        arrays,
        texture_source=texture_source,
        distances=distances,
        include_chromatic_glcm=include_chromatic_glcm,
    )
    return features, arrays, result


def run_m4_feature_extraction(
    project_root: Path,
    output_dir: Path | None = None,
    texture_source: str = "clahe_l_star",
    distances: Sequence[int] = DEFAULT_GLCM_DISTANCES,
    include_chromatic_glcm: bool = True,
) -> pd.DataFrame:
    """Build the complete 40-feature M4 matrix strictly from Layered-Labs/ACNE04 M1 images and M3 ROI masks."""
    from tqdm.auto import tqdm

    m1_manifest_path = project_root / "M1_outputs" / "m1_dataset_manifest.csv"
    if not m1_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing M1 dataset manifest: {m1_manifest_path}. Run M1 first."
        )
    m1_df = pd.read_csv(m1_manifest_path)
    assert_no_watermarked_filenames(m1_df["filename"], context="M4 feature extraction")

    out_root = output_dir or (project_root / "M4_outputs")
    out_root.mkdir(parents=True, exist_ok=True)
    feature_csv_path = out_root / "m4_feature_matrix.csv"

    image_dir = project_root / "Classification" / "JPEGImages"
    onnx_path = project_root / "models" / "bisenet_resnet18_celebamaskhq.onnx"
    onnx_session = (
        create_face_parsing_session(onnx_path, verify_sha256=False)
        if onnx_path.is_file()
        else None
    )

    n_feats = len(get_feature_columns(distances, include_chromatic_glcm=include_chromatic_glcm))
    records: list[dict[str, Any]] = []
    start_time = perf_counter()

    pbar = tqdm(
        enumerate(m1_df.itertuples(index=False), start=1),
        total=len(m1_df),
        desc=f"M4 Feature Extraction ({n_feats} feat.)",
        leave=True,
    )
    for idx, row in pbar:
        fname = str(row.filename)
        grade = int(row.authoritative_grade)
        ds_index = int(row.dataset_index)

        img_file = image_dir / fname
        if not img_file.is_file():
            raise FileNotFoundError(
                f"[M4] Image '{fname}' in M1 manifest does not exist at {img_file}. "
                f"Re-run M1 on Layered-Labs/ACNE04."
            )
        feats, _arr, _res = extract_m4_features_for_single_image(
            img_file,
            onnx_session=onnx_session,
            texture_source=texture_source,
            distances=distances,
            include_chromatic_glcm=include_chromatic_glcm,
        )

        records.append(
            {
                "dataset_index": ds_index,
                "filename": fname,
                "authoritative_grade": grade,
                "feature_source": "m3_roi_mask_and_image",
                **feats,
            }
        )
        pbar.set_postfix(file=fname, grade=grade)
        if idx % 250 == 0 or idx == len(m1_df):
            LOGGER.info("[M4] Extracted %d features for %d/%d real images", n_feats, idx, len(m1_df))

    feature_df = pd.DataFrame(records)
    feature_df.to_csv(feature_csv_path, index=False)
    elapsed = perf_counter() - start_time
    LOGGER.info(
        "[M4] Feature matrix saved to %s (%d rows, %d features, %.2fs)",
        feature_csv_path,
        len(feature_df),
        n_feats,
        elapsed,
    )
    return feature_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M4 Chromatic (CIELAB a*), Luminance GLCM, and Chromatic a* GLCM Feature Extraction"
    )
    parser.add_argument("--project-root", "--root-dir", dest="project_root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--texture-source",
        choices=("clahe_l_star", "grayscale", "lab_l_star"),
        default="clahe_l_star",
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=list(DEFAULT_GLCM_DISTANCES),
    )
    parser.add_argument(
        "--no-chromatic-glcm",
        action="store_true",
        help="Disable the 16 scale-normalized CIELAB a* chromatic GLCM features (reverts to 24-feature baseline).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = find_project_root(args.project_root, require_jpeg_dir=True)
    run_m4_feature_extraction(
        project_root=root,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        texture_source=args.texture_source,
        distances=tuple(args.distances),
        include_chromatic_glcm=not args.no_chromatic_glcm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
