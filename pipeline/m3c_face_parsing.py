#!/usr/bin/env python3
"""M3C – Detector-Independent 19-Class BiSeNet Face-Parsing Alternative & Expanded Validation.

Extracted from:
- M3C_Face_Parsing_Alternative.ipynb
- M3C_Face_Parsing_Alternative_Expanded.ipynb

Responsibilities:
1. Run 19-class CelebAMask-HQ ResNet-18 BiSeNet ONNX inference (`skin` class 1 + `nose` class 10).
2. Compare M3A (MediaPipe category 3) vs M3C on the 11-image pilot (`run_m3c_pilot_comparison`).
3. Build the 40-image stratified validation set (10 per grade across orientation and
   brightness tertiles, seed=20260901) and generate 5 batch comparison figures + review CSV
   (`run_m3c_expanded_validation`).
4. Summarize completed manual ratings (`summarize_m3c_manual_ratings`).
5. Export 19-class PNG maps, binary skin+nose PNG masks, and per-class pixel counts
   (`export_m3c_dataset`).
"""

from __future__ import annotations

import argparse
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
    CLASS_NAMES,
    EXPECTED_MODEL_SHA256,
    FACE_PARSING_CLASSES,
    FACE_PARSING_MEAN,
    FACE_PARSING_STD,
    TARGET_SIZE,
    find_project_root,
    load_authoritative_labels_df,
    sha256_file,
    standardize_image,
)
from m3a_mediapipe_audit import (
    _get_mediapipe,
    create_segmenter_options,
    extract_initial_skin_mask,
)

M3C_PILOT_FILENAMES: list[str] = [
    "levle0_1.jpg",
    "levle0_17.jpg",
    "levle3_112.jpg",
    "levle0_59.jpg",
    "levle3_62.jpg",
    "levle3_63.jpg",
    "levle0_57.jpg",
    "levle1_74.jpg",
    "levle3_53.jpg",
    "levle0_6.jpg",
    "levle0_9.jpg",
]

M3C_VALIDATION_TARGET_PER_GRADE: int = 10
M3C_VALIDATION_RANDOM_SEED: int = 20260901
M3C_VALIDATION_BATCH_SIZE: int = 8

M3C_ALLOWED_OVERALL_RATINGS: set[str] = {"Good", "Acceptable", "Poor"}
M3C_ALLOWED_PREFERENCES: set[str] = {"M3A", "M3C", "Tie", "Neither"}


def create_face_parsing_session(
    model_path: Path | str,
    verify_sha256: bool = True,
):
    """Initialize ONNX Runtime CPU session for the 19-class BiSeNet checkpoint."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for BiSeNet face parsing. Install `onnxruntime`."
        ) from exc

    m_path = Path(model_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Required BiSeNet model is missing: {m_path}")
    if verify_sha256:
        observed_sha = sha256_file(m_path)
        if observed_sha != EXPECTED_MODEL_SHA256:
            raise RuntimeError(f"Unexpected BiSeNet model SHA-256: {observed_sha}")

    session = ort.InferenceSession(str(m_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    if input_meta.shape[-2:] != [TARGET_SIZE, TARGET_SIZE]:
        raise RuntimeError(f"Unexpected model input shape: {input_meta.shape}")
    if input_meta.type != "tensor(float)":
        raise RuntimeError(f"Unexpected model input type: {input_meta.type}")
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(
            f"Unexpected model execution providers: {session.get_providers()}"
        )
    return session


def extract_m3c_face_skin_mask(
    standardized_image: np.ndarray | Image.Image,
    content_mask: np.ndarray | Image.Image,
    session: Any,
) -> dict[str, np.ndarray]:
    """Run 19-class face parsing and build the M3C candidate ROI (skin class 1 + nose class 10)."""
    image_array = np.asarray(standardized_image, dtype=np.uint8)
    image_height, image_width = image_array.shape[:2]

    if (image_width, image_height) != (TARGET_SIZE, TARGET_SIZE):
        pil_img = Image.fromarray(image_array).resize(
            (TARGET_SIZE, TARGET_SIZE), Image.Resampling.BILINEAR
        )
        model_array = np.asarray(pil_img, dtype=np.float32)
    else:
        model_array = image_array.astype(np.float32)

    normalized = (model_array / 255.0 - FACE_PARSING_MEAN) / FACE_PARSING_STD
    input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(
        np.float32
    )
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_tensor})[0]

    if logits.ndim != 4 or logits.shape[1] != len(CLASS_NAMES):
        raise RuntimeError(
            f"Unexpected face-parsing output shape: {logits.shape}. Expected [1, 19, H, W]."
        )

    class_map = np.argmax(logits[0], axis=0).astype(np.uint8)
    if class_map.shape != (image_height, image_width):
        class_map = np.asarray(
            Image.fromarray(class_map).resize(
                (image_width, image_height), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )

    valid_content = np.asarray(content_mask) > 0
    class_map = np.where(valid_content, class_map, 0).astype(np.uint8)
    skin_mask = (class_map == 1) & valid_content
    nose_mask = (class_map == 10) & valid_content
    selected_skin_mask = skin_mask | nose_mask

    masked_skin_image = np.zeros_like(image_array)
    masked_skin_image[selected_skin_mask] = image_array[selected_skin_mask]

    return {
        "class_map": class_map,
        "skin_mask": skin_mask,
        "nose_mask": nose_mask,
        "selected_skin_mask": selected_skin_mask,
        "masked_skin_image": masked_skin_image,
    }


def describe_image_for_validation(image_path: Path) -> dict[str, Any]:
    """Measure orientation and mean 64x64 thumbnail brightness for balanced sampling."""
    with Image.open(image_path) as image:
        corrected = ImageOps.exif_transpose(image).convert("RGB")
        width, height = corrected.size
        if width > height:
            orientation = "landscape"
        elif height > width:
            orientation = "portrait"
        else:
            orientation = "square"
        brightness_image = corrected.convert("L")
        brightness_image.thumbnail((64, 64), Image.Resampling.BILINEAR)
        mean_brightness = float(np.asarray(brightness_image, dtype=np.float32).mean())

    return {
        "filename": image_path.name,
        "orientation": orientation,
        "mean_brightness": mean_brightness,
    }


def assign_brightness_group(brightness_values: pd.Series) -> pd.Series:
    """Divide each grade into reproducible dark/medium/bright thirds."""
    ranked_values = brightness_values.rank(method="first")
    return pd.qcut(
        ranked_values, q=3, labels=["dark", "medium", "bright"]
    ).astype(str)


def balanced_stratum_sample(
    candidate_table: pd.DataFrame,
    sample_count: int,
    seed: int,
) -> pd.DataFrame:
    """Select rows round-robin across orientation/brightness strata."""
    if sample_count == 0:
        return candidate_table.iloc[0:0].copy()
    if len(candidate_table) < sample_count:
        raise ValueError(
            f"Requested {sample_count} images from only {len(candidate_table)} candidates."
        )

    random_generator = np.random.default_rng(seed)
    stratum_buckets: dict[str, list[pd.Series]] = {}
    for stratum, stratum_table in candidate_table.groupby(
        "sampling_stratum", sort=True
    ):
        shuffled_positions = random_generator.permutation(len(stratum_table))
        shuffled_table = stratum_table.iloc[shuffled_positions]
        stratum_buckets[str(stratum)] = [
            row for _, row in shuffled_table.iterrows()
        ]

    selected_rows: list[pd.Series] = []
    while len(selected_rows) < sample_count:
        selected_this_round = False
        for stratum in sorted(stratum_buckets):
            if stratum_buckets[stratum]:
                selected_rows.append(stratum_buckets[stratum].pop())
                selected_this_round = True
                if len(selected_rows) == sample_count:
                    break
        if not selected_this_round:
            break

    if len(selected_rows) != sample_count:
        raise RuntimeError("Balanced sampling did not produce the requested count.")
    return pd.DataFrame(selected_rows)


def build_expanded_validation_selection(
    project_root: Path,
    target_per_grade: int = M3C_VALIDATION_TARGET_PER_GRADE,
    seed: int = M3C_VALIDATION_RANDOM_SEED,
) -> pd.DataFrame:
    """Build the 40-image stratified M3C validation selection (10 per grade)."""
    image_dir = project_root / "Classification" / "JPEGImages"
    classification_dir = project_root / "Classification"
    complete_labels = load_authoritative_labels_df(classification_dir, image_dir)
    image_files = sorted(image_dir.glob("*.jpg"))

    validation_image_metadata = pd.DataFrame(
        [describe_image_for_validation(path) for path in image_files]
    )
    sampling_table = complete_labels.merge(
        validation_image_metadata, on="filename", how="inner", validate="one_to_one"
    )
    sampling_table["brightness_group"] = sampling_table.groupby("grade")[
        "mean_brightness"
    ].transform(assign_brightness_group)
    sampling_table["sampling_stratum"] = (
        sampling_table["orientation"] + "_" + sampling_table["brightness_group"]
    )

    original_pilot_set = set(M3C_PILOT_FILENAMES)
    original_pilot_selection = sampling_table[
        sampling_table["filename"].isin(original_pilot_set)
    ].copy()
    original_pilot_selection["selection_source"] = "original_11_image_pilot"

    additional_grade_selections: list[pd.DataFrame] = []
    for grade in range(4):
        existing_count = int((original_pilot_selection["grade"] == grade).sum())
        additional_count = target_per_grade - existing_count
        if additional_count < 0:
            raise ValueError(f"Grade {grade} already exceeds the validation target.")
        grade_candidates = sampling_table[
            (sampling_table["grade"] == grade)
            & ~sampling_table["filename"].isin(original_pilot_set)
        ].copy()
        grade_selection = balanced_stratum_sample(
            grade_candidates, additional_count, seed + grade
        )
        grade_selection["selection_source"] = "balanced_additional_sample"
        additional_grade_selections.append(grade_selection)

    additional_validation_selection = pd.concat(
        additional_grade_selections, ignore_index=True
    )
    selection = pd.concat(
        [original_pilot_selection, additional_validation_selection],
        ignore_index=True,
    )
    source_order = {"original_11_image_pilot": 0, "balanced_additional_sample": 1}
    selection["source_order"] = selection["selection_source"].map(source_order)
    selection = (
        selection.sort_values(
            ["grade", "source_order", "sampling_stratum", "filename"]
        )
        .drop(columns=["source_order"])
        .reset_index(drop=True)
    )
    return selection


def run_m3c_pilot_comparison(
    project_root: Path,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the 11-image M3A vs M3C same-image comparison pilot."""
    mp = _get_mediapipe()
    image_dir = project_root / "Classification" / "JPEGImages"
    out_root = output_dir or (project_root / "M3_outputs" / "M3C_pilot")
    out_root.mkdir(parents=True, exist_ok=True)

    seg_opts = create_segmenter_options(
        project_root / "models" / "selfie_multiclass_256x256.tflite"
    )
    session = create_face_parsing_session(
        project_root / "models" / "bisenet_resnet18_celebamaskhq.onnx"
    )

    records: list[dict[str, Any]] = []
    with mp.tasks.vision.ImageSegmenter.create_from_options(seg_opts) as segmenter:
        for filename in M3C_PILOT_FILENAMES:
            image_path = image_dir / filename
            _orig, standardized, content_mask, _meta = standardize_image(image_path)
            m3a_res = extract_initial_skin_mask(standardized, content_mask, segmenter)
            m3c_res = extract_m3c_face_skin_mask(standardized, content_mask, session)

            m3a_mask = m3a_res["face_skin_mask"]
            m3c_mask = m3c_res["selected_skin_mask"]
            overlap_pixels = int(np.count_nonzero(m3a_mask & m3c_mask))
            union_pixels = int(np.count_nonzero(m3a_mask | m3c_mask))
            valid_pixels = int(np.count_nonzero(content_mask))
            m3a_pixels = int(np.count_nonzero(m3a_mask))
            m3c_pixels = int(np.count_nonzero(m3c_mask))

            records.append(
                {
                    "filename": filename,
                    "valid_content_pixels": valid_pixels,
                    "m3a_skin_pixels": m3a_pixels,
                    "m3c_skin_pixels": int(np.count_nonzero(m3c_res["skin_mask"])),
                    "m3c_nose_pixels": int(np.count_nonzero(m3c_res["nose_mask"])),
                    "m3c_selected_pixels": m3c_pixels,
                    "m3c_selected_fraction": (
                        m3c_pixels / valid_pixels if valid_pixels > 0 else 0.0
                    ),
                    "overlap_pixels": overlap_pixels,
                    "union_pixels": union_pixels,
                    "agreement_iou": (
                        overlap_pixels / union_pixels if union_pixels > 0 else 0.0
                    ),
                    "m3c_to_m3a_area_ratio": (
                        m3c_pixels / m3a_pixels if m3a_pixels > 0 else np.nan
                    ),
                }
            )

    table = pd.DataFrame(records)
    csv_path = out_root / "m3c_pilot_comparison.csv"
    table.to_csv(csv_path, index=False)
    print(f"M3C 11-image pilot comparison saved to {csv_path}")
    return table


def run_m3c_expanded_validation(
    project_root: Path,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the 40-image expanded balanced M3A-vs-M3C validation and save batch figures + CSVs."""
    mp = _get_mediapipe()
    image_dir = project_root / "Classification" / "JPEGImages"
    out_root = output_dir or (
        project_root / "M3_outputs" / "M3C_expanded_validation"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    selection = build_expanded_validation_selection(project_root)
    metadata_lookup = selection.set_index("filename")

    seg_opts = create_segmenter_options(
        project_root / "models" / "selfie_multiclass_256x256.tflite"
    )
    session = create_face_parsing_session(
        project_root / "models" / "bisenet_resnet18_celebamaskhq.onnx"
    )

    validation_records: list[dict[str, Any]] = []
    validation_visuals: list[dict[str, Any]] = []

    with mp.tasks.vision.ImageSegmenter.create_from_options(seg_opts) as segmenter:
        for idx, filename in enumerate(selection["filename"], start=1):
            meta_row = metadata_lookup.loc[filename]
            image_path = image_dir / filename
            try:
                _orig, image_array, content_mask, _meta = standardize_image(image_path)
                m3a_res = extract_initial_skin_mask(
                    image_array, content_mask, segmenter
                )
                m3c_res = extract_m3c_face_skin_mask(
                    image_array, content_mask, session
                )

                m3a_mask = m3a_res["face_skin_mask"]
                m3c_mask = m3c_res["selected_skin_mask"]
                m3a_skin_image = np.zeros_like(image_array)
                m3a_skin_image[m3a_mask] = image_array[m3a_mask]

                overlap_mask = m3a_mask & m3c_mask
                union_mask = m3a_mask | m3c_mask
                agreement_image = np.zeros_like(image_array)
                agreement_image[overlap_mask] = [0, 200, 0]
                agreement_image[m3a_mask & ~m3c_mask] = [230, 50, 50]
                agreement_image[m3c_mask & ~m3a_mask] = [0, 180, 230]

                valid_pixels = int(np.count_nonzero(content_mask))
                m3a_pixels = int(np.count_nonzero(m3a_mask))
                m3c_pixels = int(np.count_nonzero(m3c_mask))
                overlap_pixels = int(np.count_nonzero(overlap_mask))
                union_pixels = int(np.count_nonzero(union_mask))

                validation_records.append(
                    {
                        "filename": filename,
                        "grade": int(meta_row["grade"]),
                        "orientation": meta_row["orientation"],
                        "brightness_group": meta_row["brightness_group"],
                        "selection_source": meta_row["selection_source"],
                        "valid_content_pixels": valid_pixels,
                        "m3a_skin_pixels": m3a_pixels,
                        "m3a_skin_fraction": (
                            m3a_pixels / valid_pixels if valid_pixels > 0 else 0.0
                        ),
                        "m3c_skin_pixels": int(
                            np.count_nonzero(m3c_res["skin_mask"])
                        ),
                        "m3c_nose_pixels": int(
                            np.count_nonzero(m3c_res["nose_mask"])
                        ),
                        "m3c_selected_pixels": m3c_pixels,
                        "m3c_selected_fraction": (
                            m3c_pixels / valid_pixels if valid_pixels > 0 else 0.0
                        ),
                        "overlap_pixels": overlap_pixels,
                        "union_pixels": union_pixels,
                        "agreement_iou": (
                            overlap_pixels / union_pixels if union_pixels > 0 else 0.0
                        ),
                        "m3c_to_m3a_area_ratio": (
                            m3c_pixels / m3a_pixels if m3a_pixels > 0 else np.nan
                        ),
                        "status": "success",
                        "error": None,
                        "m3a_skin_retention": "",
                        "m3a_contamination": "",
                        "m3a_overall_rating": "",
                        "m3c_skin_retention": "",
                        "m3c_contamination": "",
                        "m3c_overall_rating": "",
                        "preferred_method": "",
                        "review_notes": "",
                    }
                )
                validation_visuals.append(
                    {
                        "filename": filename,
                        "grade": int(meta_row["grade"]),
                        "standardized": image_array,
                        "m3a_skin": m3a_skin_image,
                        "m3c_class_map": m3c_res["class_map"],
                        "m3c_selected": m3c_res["masked_skin_image"],
                        "agreement": agreement_image,
                        "error": None,
                    }
                )
            except Exception as error:
                placeholder_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
                placeholder_map = np.zeros((512, 512), dtype=np.uint8)
                validation_records.append(
                    {
                        "filename": filename,
                        "grade": int(meta_row["grade"]),
                        "orientation": meta_row["orientation"],
                        "brightness_group": meta_row["brightness_group"],
                        "selection_source": meta_row["selection_source"],
                        "valid_content_pixels": None,
                        "m3a_skin_pixels": None,
                        "m3a_skin_fraction": None,
                        "m3c_skin_pixels": None,
                        "m3c_nose_pixels": None,
                        "m3c_selected_pixels": None,
                        "m3c_selected_fraction": None,
                        "overlap_pixels": None,
                        "union_pixels": None,
                        "agreement_iou": None,
                        "m3c_to_m3a_area_ratio": None,
                        "status": "error",
                        "error": str(error),
                        "m3a_skin_retention": "",
                        "m3a_contamination": "",
                        "m3a_overall_rating": "",
                        "m3c_skin_retention": "",
                        "m3c_contamination": "",
                        "m3c_overall_rating": "",
                        "preferred_method": "",
                        "review_notes": "",
                    }
                )
                validation_visuals.append(
                    {
                        "filename": filename,
                        "grade": int(meta_row["grade"]),
                        "standardized": placeholder_rgb,
                        "m3a_skin": placeholder_rgb,
                        "m3c_class_map": placeholder_map,
                        "m3c_selected": placeholder_rgb,
                        "agreement": placeholder_rgb,
                        "error": str(error),
                    }
                )
            if idx % 10 == 0 or idx == len(selection):
                print(f"Processed {idx}/{len(selection)} validation images...")

    for batch_start in range(0, len(validation_visuals), M3C_VALIDATION_BATCH_SIZE):
        batch_visuals = validation_visuals[
            batch_start : batch_start + M3C_VALIDATION_BATCH_SIZE
        ]
        batch_number = (batch_start // M3C_VALIDATION_BATCH_SIZE) + 1
        fig, axes = plt.subplots(
            len(batch_visuals),
            5,
            figsize=(20, 4 * len(batch_visuals)),
            squeeze=False,
        )
        for row, visual in enumerate(batch_visuals):
            axes[row, 0].imshow(visual["standardized"])
            axes[row, 0].set_title(
                f'{visual["filename"]} | Grade {visual["grade"]}\nStandardized input'
            )
            axes[row, 1].imshow(visual["m3a_skin"])
            axes[row, 1].set_title("M3A MediaPipe skin")
            axes[row, 2].imshow(
                visual["m3c_class_map"],
                cmap="tab20",
                vmin=0,
                vmax=19,
                interpolation="nearest",
            )
            axes[row, 2].set_title("M3C 19-class map")
            axes[row, 3].imshow(visual["m3c_selected"])
            axes[row, 3].set_title("M3C skin + nose")
            axes[row, 4].imshow(visual["agreement"])
            axes[row, 4].set_title("Agreement\ngreen=both, red=M3A, cyan=M3C")
            for col in range(5):
                axes[row, col].axis("off")
        plt.tight_layout()
        batch_path = out_root / f"m3c_validation_batch_{batch_number}.png"
        fig.savefig(batch_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    table = pd.DataFrame(validation_records)
    review_csv = out_root / "m3c_expanded_validation_review.csv"
    selection_csv = out_root / "m3c_expanded_validation_selection.csv"
    table.to_csv(review_csv, index=False)
    selection.to_csv(selection_csv, index=False)
    print(f"Expanded M3C validation saved to {review_csv}")
    return table


def summarize_m3c_manual_ratings(rated_csv_path: Path | str) -> dict[str, pd.DataFrame]:
    """Summarize a completed copy of the 40-image manual-review CSV."""
    rated_table = pd.read_csv(rated_csv_path, keep_default_na=False)
    required_cols = ["m3a_overall_rating", "m3c_overall_rating", "preferred_method"]
    incomplete_rows = rated_table[required_cols].eq("").any(axis=1)
    if incomplete_rows.any():
        print(
            f"Manual review is incomplete: {int(incomplete_rows.sum())} rows still have blank ratings."
        )
        return {"rated_table": rated_table}

    invalid_m3a = (
        set(rated_table["m3a_overall_rating"].unique()) - M3C_ALLOWED_OVERALL_RATINGS
    )
    invalid_m3c = (
        set(rated_table["m3c_overall_rating"].unique()) - M3C_ALLOWED_OVERALL_RATINGS
    )
    invalid_pref = (
        set(rated_table["preferred_method"].unique()) - M3C_ALLOWED_PREFERENCES
    )
    if invalid_m3a or invalid_m3c or invalid_pref:
        raise ValueError(
            "Use only Good/Acceptable/Poor for overall ratings and M3A/M3C/Tie/Neither for preferred_method."
        )

    rating_summary = (
        pd.DataFrame(
            {
                "M3A": rated_table["m3a_overall_rating"].value_counts(),
                "M3C": rated_table["m3c_overall_rating"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
        .reindex(["Good", "Acceptable", "Poor"], fill_value=0)
    )
    preference_summary = (
        rated_table["preferred_method"]
        .value_counts()
        .reindex(["M3A", "M3C", "Tie", "Neither"], fill_value=0)
        .rename("image_count")
        .to_frame()
    )
    print("Overall manual ratings:\n", rating_summary)
    print("\nPreferred method:\n", preference_summary)
    return {
        "rated_table": rated_table,
        "rating_summary": rating_summary,
        "preference_summary": preference_summary,
    }


def export_m3c_dataset(
    project_root: Path,
    output_root: Path | None = None,
) -> pd.DataFrame:
    """Export M3C 19-class maps, binary skin+nose masks, and per-class pixel counts."""
    image_dir = project_root / "Classification" / "JPEGImages"
    m3c_root = output_root or (
        project_root / "M3_outputs" / "M3C_face_parsing_alternative"
    )
    class_map_dir = m3c_root / "class_maps"
    selected_mask_dir = m3c_root / "selected_skin_masks"
    class_map_dir.mkdir(parents=True, exist_ok=True)
    selected_mask_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.jpg"))
    session = create_face_parsing_session(
        project_root / "models" / "bisenet_resnet18_celebamaskhq.onnx"
    )

    export_records: list[dict[str, Any]] = []
    start_time = perf_counter()

    for image_number, image_path in enumerate(image_files, start=1):
        try:
            _orig, standardized, content_mask, _meta = standardize_image(image_path)
            result = extract_m3c_face_skin_mask(standardized, content_mask, session)
            output_filename = f"{image_path.stem}.png"

            Image.fromarray(
                np.asarray(result["class_map"], dtype=np.uint8), mode="L"
            ).save(class_map_dir / output_filename)
            Image.fromarray(
                np.asarray(result["selected_skin_mask"], dtype=np.uint8) * 255
            ).save(selected_mask_dir / output_filename)

            valid_content = np.asarray(content_mask, dtype=bool)
            valid_content_pixels = int(np.count_nonzero(valid_content))
            selected_skin_pixels = int(
                np.count_nonzero(result["selected_skin_mask"])
            )
            selected_skin_fraction = (
                selected_skin_pixels / valid_content_pixels
                if valid_content_pixels > 0
                else 0.0
            )
            class_counts = np.bincount(
                result["class_map"][valid_content].ravel(), minlength=19
            )

            export_record: dict[str, Any] = {
                "filename": image_path.name,
                "valid_content_pixels": valid_content_pixels,
                "skin_pixels": int(class_counts[1]),
                "nose_pixels": int(class_counts[10]),
                "selected_skin_pixels": selected_skin_pixels,
                "selected_skin_fraction": selected_skin_fraction,
                "status": "success",
                "error": None,
            }
            for class_id, class_name in FACE_PARSING_CLASSES.items():
                export_record[f"class_{class_id}_{class_name}_pixels"] = int(
                    class_counts[class_id]
                )
            export_records.append(export_record)
        except Exception as error:
            export_records.append(
                {
                    "filename": image_path.name,
                    "valid_content_pixels": None,
                    "skin_pixels": None,
                    "nose_pixels": None,
                    "selected_skin_pixels": None,
                    "selected_skin_fraction": None,
                    "status": "error",
                    "error": str(error),
                }
            )
        if image_number % 100 == 0 or image_number == len(image_files):
            print(f"Exported {image_number}/{len(image_files)} M3C images...")

    export_table = pd.DataFrame(export_records)
    audit_csv_path = m3c_root / "m3c_export_audit.csv"
    export_table.to_csv(audit_csv_path, index=False)
    print(
        f"M3C export completed in {(perf_counter() - start_time) / 60:.2f} min -> {audit_csv_path}"
    )
    return export_table


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3C Face-Parsing Alternative Pilot, Expanded 40-Image Validation, and Export"
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
        choices=("pilot-11", "expanded-40", "summarize-ratings", "export-pngs"),
        default="expanded-40",
        help="M3C mode to execute.",
    )
    parser.add_argument(
        "--rated-csv",
        type=Path,
        default=None,
        help="Path to rated CSV when using --mode summarize-ratings.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = find_project_root(
        args.project_root,
        require_jpeg_dir=(args.mode != "summarize-ratings"),
    )
    out_dir = args.output_dir.resolve() if args.output_dir else None

    if args.mode == "pilot-11":
        run_m3c_pilot_comparison(root, out_dir)
    elif args.mode == "expanded-40":
        run_m3c_expanded_validation(root, out_dir)
    elif args.mode == "summarize-ratings":
        csv_path = args.rated_csv or (
            root
            / "M3_outputs"
            / "M3C_expanded_validation"
            / "m3c_expanded_validation_review.csv"
        )
        summarize_m3c_manual_ratings(csv_path)
    elif args.mode == "export-pngs":
        export_m3c_dataset(root, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
