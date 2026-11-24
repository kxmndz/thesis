#!/usr/bin/env python3
"""M3 – Production BiSeNet Preprocessing & M4 Handoff Module (Layered-Labs/ACNE04)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from m3_common import prepare_color_representations, standardize_image
from m3c_face_parsing import extract_m3c_face_skin_mask

DEFAULT_REPRODUCTION_RUN_ID: str = "M3_bisenet_operational_final_20260928"


def parse_face(
    standardized_rgb: np.ndarray,
    content_mask: np.ndarray,
    session: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if session is None:
        raise RuntimeError("BiSeNet ONNX InferenceSession is required for M3 face parsing.")
    return extract_m3c_face_skin_mask(standardized_rgb, content_mask, session)


def process_m3_image(
    image_path: Path | str,
    grade: int = 0,
    session: Any | None = None,
    padding_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"M3 input image does not exist on disk: {image_path}")
    if session is None:
        raise RuntimeError("BiSeNet ONNX session is required to run process_m3_image.")

    original, std_rgb, content_mask, meta = standardize_image(
        image_path, padding_meta=padding_meta
    )
    colors = prepare_color_representations(std_rgb)
    class_map, skin_mask, nose_mask, final_roi = parse_face(std_rgb, content_mask, session)
    if int(np.count_nonzero(final_roi)) == 0:
        raise RuntimeError(f"BiSeNet produced an empty skin+nose ROI mask for {image_path.name}")

    arrays = {
        "original_rgb": np.asarray(original, dtype=np.uint8),
        "standardized_rgb": std_rgb,
        "content_mask": content_mask,
        "face_parsing_class_map": class_map,
        "skin_mask": skin_mask,
        "nose_mask": nose_mask,
        "final_roi_mask": final_roi,
        **colors,
    }
    result = {
        "filename": image_path.name,
        "grade": int(grade),
        "processing_status": "success",
        "error_message": None,
        "roi_pixel_count": int(np.count_nonzero(final_roi)),
        "valid_content_pixels": int(np.count_nonzero(content_mask)),
        "roi_ratio": float(np.count_nonzero(final_roi) / max(1, np.count_nonzero(content_mask))),
        **meta,
    }
    return result, arrays
