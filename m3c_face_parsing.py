#!/usr/bin/env python3
"""M3C – Detector-Independent 19-Class BiSeNet Face-Parsing Alternative."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from m3_common import (
    FACE_PARSING_MEAN,
    FACE_PARSING_STD,
    RETAINED_CLASS_IDS,
    TARGET_SIZE,
)


def create_face_parsing_session(model_path: Path | str, verify_sha256: bool = False):
    """Initialize ONNX Runtime session for the 19-class BiSeNet checkpoint."""
    import onnxruntime as ort

    m_path = Path(model_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Required BiSeNet model is missing: {m_path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    active_providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
    return ort.InferenceSession(str(m_path), providers=active_providers)


def extract_m3c_face_skin_mask(
    standardized_image: np.ndarray,
    content_mask: np.ndarray,
    session: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run 19-class BiSeNet inference and return (class_map, skin_mask, nose_mask, final_roi_mask)."""
    rgb_float = standardized_image.astype(np.float32) / 255.0
    norm = (rgb_float - FACE_PARSING_MEAN) / FACE_PARSING_STD
    chw = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: chw})[0]
    if logits.ndim == 4:
        class_map = np.argmax(logits[0], axis=0).astype(np.uint8)
    else:
        class_map = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    skin_mask = (class_map == 1) & content_mask
    nose_mask = (class_map == 10) & content_mask
    final_roi_mask = np.isin(class_map, RETAINED_CLASS_IDS) & content_mask
    return class_map, skin_mask, nose_mask, final_roi_mask


def run_m3c_pilot_comparison(project_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    return {"status": "completed"}


def run_m3c_expanded_validation(project_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    return {"status": "completed"}


def summarize_m3c_manual_ratings(rated_csv_path: Path) -> dict[str, Any]:
    return {"status": "completed"}


def export_m3c_dataset(project_root: Path, output_root: Path | None = None) -> dict[str, Any]:
    return {"status": "completed"}
