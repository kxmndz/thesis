#!/usr/bin/env python3
"""M1 – Data Partitioning and Stratified 5x2 Cross-Validation Module.

Extracted from:
- Module1_Data_Loader.ipynb

Responsibilities:
1. Load authoritative Classification records (NNEW_trainval_0.txt, NNEW_test_0.txt).
2. Validate all 1,457 unique images and authoritative grade counts.
3. Generate 5 repetitions x 2 complementary stratified folds (seed=42).
4. Export and verify M1_outputs/m1_dataset_manifest.csv and M1_outputs/m1_5x2_fold_assignments.csv.
5. Run Cross-Validation QA (disjointness, class balance, complementary roles, reproducibility).
6. Provide optional legacy 224x224 PyTorch Dataset/DataLoader definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m3_common import EXPECTED_GRADE_COUNTS, EXPECTED_TOTAL_IMAGES, find_project_root


def load_m1_records(
    project_root: Path,
) -> tuple[list[tuple[str, int, str]], np.ndarray, np.ndarray]:
    """Load authoritative records preserving source order across trainval_0 and test_0."""
    classification_dir = project_root / "Classification"
    image_dir = classification_dir / "JPEGImages"
    label_files = (
        classification_dir / "NNEW_trainval_0.txt",
        classification_dir / "NNEW_test_0.txt",
    )

    records: list[tuple[str, int, str]] = []
    for label_file in label_files:
        if not label_file.is_file():
            raise FileNotFoundError(f"Missing authoritative label file: {label_file}")
        with label_file.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 3:
                    raise ValueError(
                        f"Expected 3 fields in {label_file}:{line_number}; got {len(parts)}"
                    )
                filename, grade_text, lesion_count_text = parts
                if (
                    Path(filename).name != filename
                    or Path(filename).suffix.lower() not in {".jpg", ".jpeg"}
                ):
                    raise ValueError(
                        f"Unexpected image filename in {label_file}:{line_number}: {filename}"
                    )
                grade = int(grade_text)
                if grade not in (0, 1, 2, 3):
                    raise ValueError(
                        f"Unexpected grade in {label_file}:{line_number}: {grade}"
                    )
                float(lesion_count_text)  # Schema check only; grade is authoritative.
                records.append((filename, grade, str((image_dir / filename).resolve())))

    x_paths = np.asarray([record[2] for record in records], dtype=object)
    y_labels = np.asarray([record[1] for record in records], dtype=int)
    return records, x_paths, y_labels


def validate_m1_dataset(
    records: list[tuple[str, int, str]],
    x_paths: np.ndarray,
    y_labels: np.ndarray,
    image_dir: Path,
    verify_disk_jpegs: bool = True,
) -> dict[int, int]:
    """Validate record counts, grade counts, and one-to-one JPEG correspondence."""
    filenames = [record[0] for record in records]
    record_filenames = set(filenames)
    if len(records) != EXPECTED_TOTAL_IMAGES or len(record_filenames) != EXPECTED_TOTAL_IMAGES:
        raise AssertionError("Expected exactly 1,457 unique authoritative image records")

    if verify_disk_jpegs:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"JPEG directory not found: {image_dir}")
        disk_filenames = {
            path.name
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        }
        if disk_filenames != record_filenames:
            raise AssertionError(
                f"JPEG mismatch: missing={sorted(record_filenames - disk_filenames)}, "
                f"extra={sorted(disk_filenames - record_filenames)}"
            )
        if any(not Path(path).is_file() for path in x_paths):
            raise AssertionError("At least one source image path is missing")

    grade_counts = {grade: int(np.sum(y_labels == grade)) for grade in (0, 1, 2, 3)}
    if grade_counts != EXPECTED_GRADE_COUNTS:
        raise AssertionError(f"Unexpected authoritative grade counts: {grade_counts}")

    return grade_counts


def build_5x2_cv_splits(
    x_paths: np.ndarray,
    y_labels: np.ndarray,
    random_state: int = 42,
    fallback_folds_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Build 5 repetitions x 2 complementary stratified folds."""
    cv_splits: list[dict[str, Any]] = []
    try:
        from sklearn.model_selection import RepeatedStratifiedKFold

        rskf = RepeatedStratifiedKFold(
            n_splits=2, n_repeats=5, random_state=random_state
        )
        for configuration_index, (train_idx, test_idx) in enumerate(
            rskf.split(x_paths, y_labels), start=1
        ):
            repeat_number = (configuration_index - 1) // 2 + 1
            fold_number = (configuration_index - 1) % 2 + 1
            cv_splits.append(
                {
                    "configuration_index": configuration_index,
                    "repeat_number": repeat_number,
                    "fold_number": fold_number,
                    "train_idx": train_idx.copy(),
                    "test_idx": test_idx.copy(),
                    "train_paths": x_paths[train_idx].copy(),
                    "test_paths": x_paths[test_idx].copy(),
                    "train_labels": y_labels[train_idx].copy(),
                    "test_labels": y_labels[test_idx].copy(),
                }
            )
    except ImportError as exc:
        if fallback_folds_csv is not None and fallback_folds_csv.is_file():
            folds_df = pd.read_csv(fallback_folds_csv)
            for configuration_index, group in folds_df.groupby("configuration", sort=True):
                repeat_number = int(group["repeat"].iloc[0])
                fold_number = int(group["fold"].iloc[0])
                train_idx = (
                    group.loc[group["split"] == "train", "dataset_index"]
                    .to_numpy(dtype=int)
                )
                test_idx = (
                    group.loc[group["split"] == "test", "dataset_index"]
                    .to_numpy(dtype=int)
                )
                cv_splits.append(
                    {
                        "configuration_index": int(configuration_index),
                        "repeat_number": repeat_number,
                        "fold_number": fold_number,
                        "train_idx": train_idx.copy(),
                        "test_idx": test_idx.copy(),
                        "train_paths": x_paths[train_idx].copy(),
                        "test_paths": x_paths[test_idx].copy(),
                        "train_labels": y_labels[train_idx].copy(),
                        "test_labels": y_labels[test_idx].copy(),
                    }
                )
        else:
            raise RuntimeError(
                "scikit-learn is required to generate new 5x2 stratified splits when "
                "m1_5x2_fold_assignments.csv is not already present."
            ) from exc

    if len(cv_splits) != 10:
        raise AssertionError(f"Expected 10 CV configurations, got {len(cv_splits)}")
    return cv_splits


def save_or_verify_csv(
    frame: pd.DataFrame,
    output_path: Path,
    overwrite: bool = False,
) -> str:
    """Save CSV or verify that an existing file matches byte-for-byte."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if output_path.exists() and not overwrite:
        if output_path.read_bytes() != expected_bytes:
            raise FileExistsError(
                f"Existing output differs; inspect or pass --overwrite: {output_path}"
            )
        return "verified"
    output_path.write_bytes(expected_bytes)
    return "created"


def export_m1_manifests(
    records: list[tuple[str, int, str]],
    x_paths: np.ndarray,
    y_labels: np.ndarray,
    cv_splits: list[dict[str, Any]],
    output_root: Path,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Build and export m1_dataset_manifest.csv and m1_5x2_fold_assignments.csv."""
    filenames = [record[0] for record in records]
    portable_source_paths = [
        (Path("Classification") / "JPEGImages" / filename).as_posix()
        for filename in filenames
    ]
    dataset_manifest = pd.DataFrame(
        {
            "dataset_index": np.arange(len(x_paths), dtype=int),
            "filename": filenames,
            "authoritative_grade": y_labels,
            "source_path": portable_source_paths,
        }
    )

    membership_rows: list[dict[str, Any]] = []
    for split_info in cv_splits:
        for split_name in ("train", "test"):
            for index in split_info[f"{split_name}_idx"]:
                idx = int(index)
                membership_rows.append(
                    {
                        "configuration": split_info["configuration_index"],
                        "repeat": split_info["repeat_number"],
                        "fold": split_info["fold_number"],
                        "split": split_name,
                        "dataset_index": idx,
                        "filename": filenames[idx],
                        "authoritative_grade": int(y_labels[idx]),
                        "source_path": portable_source_paths[idx],
                    }
                )
    fold_assignments = pd.DataFrame(membership_rows)
    if len(dataset_manifest) != 1457 or len(fold_assignments) != 14570:
        raise AssertionError("Unexpected row counts in M1 manifests")

    dataset_manifest_path = output_root / "m1_dataset_manifest.csv"
    fold_assignments_path = output_root / "m1_5x2_fold_assignments.csv"

    status_manifest = save_or_verify_csv(
        dataset_manifest, dataset_manifest_path, overwrite=overwrite
    )
    status_folds = save_or_verify_csv(
        fold_assignments, fold_assignments_path, overwrite=overwrite
    )
    print(f"Dataset manifest ({status_manifest}): {dataset_manifest_path} ({len(dataset_manifest)} rows)")
    print(f"Fold assignments ({status_folds}): {fold_assignments_path} ({len(fold_assignments)} rows)")
    return dataset_manifest, fold_assignments, dataset_manifest_path, fold_assignments_path


def run_m1_qa(
    records: list[tuple[str, int, str]],
    x_paths: np.ndarray,
    y_labels: np.ndarray,
    grade_counts: dict[int, int],
    cv_splits: list[dict[str, Any]],
    dataset_manifest: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    dataset_manifest_path: Path,
    fold_assignments_path: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """Execute all M1 cross-validation QA checks and return the QA summary DataFrame."""
    filenames = [record[0] for record in records]
    record_filenames = set(filenames)
    all_indices = set(range(len(x_paths)))
    qa_rows: list[dict[str, Any]] = []

    for split_info in cv_splits:
        train_idx = split_info["train_idx"]
        test_idx = split_info["test_idx"]
        train_set, test_set = set(map(int, train_idx)), set(map(int, test_idx))
        assert len(train_set) == len(train_idx) and len(test_set) == len(test_idx)
        assert train_set.isdisjoint(test_set) and train_set | test_set == all_indices
        assert len(train_set) + len(test_set) == EXPECTED_TOTAL_IMAGES
        assert (
            set(Path(path).name for path in split_info["train_paths"])
            | set(Path(path).name for path in split_info["test_paths"])
            == record_filenames
        )
        train_counts = {
            grade: int(np.sum(split_info["train_labels"] == grade))
            for grade in (0, 1, 2, 3)
        }
        test_counts = {
            grade: int(np.sum(split_info["test_labels"] == grade))
            for grade in (0, 1, 2, 3)
        }
        assert all(
            train_counts[grade] + test_counts[grade] == grade_counts[grade]
            for grade in grade_counts
        )
        assert all(
            abs(train_counts[grade] - test_counts[grade]) <= 1
            for grade in grade_counts
        )
        qa_rows.append(
            {
                "configuration": split_info["configuration_index"],
                "repeat": split_info["repeat_number"],
                "fold": split_info["fold_number"],
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                **{f"train_g{grade}": train_counts[grade] for grade in grade_counts},
                **{f"test_g{grade}": test_counts[grade] for grade in grade_counts},
            }
        )

    for repeat_number in range(1, 6):
        first, second = cv_splits[2 * (repeat_number - 1) : 2 * repeat_number]
        assert first["repeat_number"] == second["repeat_number"] == repeat_number
        assert {first["fold_number"], second["fold_number"]} == {1, 2}
        assert set(map(int, first["test_idx"])) == set(map(int, second["train_idx"]))
        assert set(map(int, first["train_idx"])) == set(map(int, second["test_idx"]))

    try:
        from sklearn.model_selection import RepeatedStratifiedKFold

        second_pass = list(
            RepeatedStratifiedKFold(
                n_splits=2, n_repeats=5, random_state=random_state
            ).split(x_paths, y_labels)
        )
        assert len(second_pass) == len(cv_splits)
        assert all(
            np.array_equal(first["train_idx"], second[0])
            and np.array_equal(first["test_idx"], second[1])
            for first, second in zip(cv_splits, second_pass)
        )
    except ImportError:
        pass

    pd.testing.assert_frame_equal(
        pd.read_csv(dataset_manifest_path), dataset_manifest, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(fold_assignments_path), fold_assignments, check_dtype=False
    )
    assert not fold_assignments.duplicated(["configuration", "dataset_index"]).any()
    assert fold_assignments.groupby("configuration").size().to_dict() == {
        number: EXPECTED_TOTAL_IMAGES for number in range(1, 11)
    }
    for _configuration_index, group in fold_assignments.groupby("configuration"):
        assert group.sort_values("dataset_index")["filename"].tolist() == filenames
        assert (
            group.sort_values("dataset_index")["authoritative_grade"].tolist()
            == y_labels.tolist()
        )

    print("M1_5X2_REPRODUCIBILITY_VERIFIED")
    print("CSV identities, grades, and 14,570 memberships VERIFIED")
    return pd.DataFrame(qa_rows)


def build_legacy_224_loaders():
    """Return the historical PyTorch 224x224 transforms and AcneDataset class."""
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class AcneDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            label = int(self.labels[idx])
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)

    return train_transform, test_transform, AcneDataset


def run_m1_pipeline(
    project_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    verify_disk_jpegs: bool = True,
    overwrite: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the complete M1 data partitioning and cross-validation workflow."""
    root = find_project_root(project_root, require_jpeg_dir=verify_disk_jpegs)
    image_dir = root / "Classification" / "JPEGImages"
    m1_output_root = Path(output_dir).resolve() if output_dir else (root / "M1_outputs")

    print(f"Project root: {root}")
    records, x_paths, y_labels = load_m1_records(root)
    print(f"Loaded {len(x_paths)} classification records")

    grade_counts = validate_m1_dataset(
        records, x_paths, y_labels, image_dir, verify_disk_jpegs=verify_disk_jpegs
    )
    print(f"Authoritative grades: {grade_counts}")

    cv_splits = build_5x2_cv_splits(
        x_paths,
        y_labels,
        random_state=random_state,
        fallback_folds_csv=m1_output_root / "m1_5x2_fold_assignments.csv",
    )
    print(f"Preserved {len(cv_splits)} configurations across 5 repeats")

    (
        dataset_manifest,
        fold_assignments,
        dataset_manifest_path,
        fold_assignments_path,
    ) = export_m1_manifests(
        records,
        x_paths,
        y_labels,
        cv_splits,
        m1_output_root,
        overwrite=overwrite,
    )

    qa_table = run_m1_qa(
        records,
        x_paths,
        y_labels,
        grade_counts,
        cv_splits,
        dataset_manifest,
        fold_assignments,
        dataset_manifest_path,
        fold_assignments_path,
        random_state=random_state,
    )
    print(qa_table.to_string(index=False))

    return {
        "project_root": root,
        "X_paths": x_paths,
        "y_labels": y_labels,
        "cv_splits": cv_splits,
        "dataset_manifest": dataset_manifest,
        "fold_assignments": fold_assignments,
        "dataset_manifest_path": dataset_manifest_path,
        "fold_assignments_path": fold_assignments_path,
        "qa_table": qa_table,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M1 Data Partitioning and 5x2 Cross-Validation Pipeline"
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
        help="Directory for M1 CSV outputs (defaults to <project_root>/M1_outputs).",
    )
    parser.add_argument(
        "--skip-disk-jpeg-check",
        action="store_true",
        help="Skip verifying physical JPEG files on disk (useful when testing manifest generation only).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing M1 manifest CSVs if they differ.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for RepeatedStratifiedKFold (default: 42).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_m1_pipeline(
        project_root=args.project_root,
        output_dir=args.output_dir,
        verify_disk_jpegs=not args.skip_disk_jpeg_check,
        overwrite=args.overwrite,
        random_state=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
