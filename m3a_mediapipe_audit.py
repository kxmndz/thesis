#!/usr/bin/env python3
"""M3A – MediaPipe Selfie Segmentation & Face Landmarker Audit Module."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd


def run_dimension_audit(image_dir: Path, output_dir: Path) -> pd.DataFrame:
    return pd.DataFrame()


def run_initial_mask_audit(project_root: Path, output_dir: Path) -> pd.DataFrame:
    return pd.DataFrame()


def run_landmark_audit(
    project_root: Path,
    initial_mask_audit: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    return pd.DataFrame()
