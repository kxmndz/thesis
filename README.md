# Explainable Acne Severity Grading Pipeline (M1–M7)

A modular, end-to-end pipeline for acne severity classification and interpretability on the [`Layered-Labs/ACNE04`](https://huggingface.co/datasets/Layered-Labs/ACNE04) dataset. It pairs a **Balanced Random Forest** using 40 interpretable chromatic and GLCM texture features against a baseline **Faster R-CNN** lesion detector.

---

## 1. Setup & Installation

Ensure Python 3.10+ is installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. How to Run

You can run the pipeline either locally via the CLI or in the cloud using Google Colab.

### Option A: Command-Line Interface (`pipeline_cli.py`)

Run the complete pipeline from start to finish:

```bash
python pipeline_cli.py run-pipeline
```

Or execute individual stages independently:

```bash
# Stage 1: Download & standardize dataset, create 5x2 CV splits
python pipeline_cli.py m1

# Stage 2: Train / evaluate baseline Faster R-CNN
python pipeline_cli.py m2

# Stage 3: Extract 40 chromatic (CIELAB a*) and GLCM texture features
python pipeline_cli.py m4

# Stage 4: Train Balanced Random Forest with nested CV grid search
python pipeline_cli.py m5

# Stage 5: Run model evaluation, QWK, and efficiency benchmarks
python pipeline_cli.py m6

# Stage 6: Generate global and local SHAP explanations
python pipeline_cli.py m7

# Single-Image Evaluation: Run end-to-end demo on a test image
python pipeline_cli.py evaluate-image
```

*Optional flags:*
- `--project-root <PATH>`: Specify custom workspace root (defaults to current directory).
- `--parquet-path <PATH>`: Path to local ACNE04 parquet file (bypasses Hugging Face download).

---

### Option B: Google Colab Notebook (`Thesis_Colab.ipynb`)

For a managed, GPU-accelerated run (recommended: **T4 GPU** runtime):

1. Open `Thesis_Colab.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run the notebook sequentially (or select **Runtime > Run all**):
   - **Steps 0–2**: Mount Google Drive, install dependencies, and unpack code.
   - **Stages 1–7 (M1–M7)**: Run data prep, BiSeNet face masking, feature extraction, Faster R-CNN, Random Forest, evaluation benchmarks, and SHAP interpretability.
   - **Stage 8**: Generate a dual-model diagnostic dashboard for a single image.
   - **Stage 9**: Package all outputs into `Acne_Pipeline_Results_Bundle.zip` for download.

---

## 3. Expected Outputs

Running the pipeline populates the following directories:

| Directory | Module | Expected Output Files & Artifacts |
|---|---|---|
| `Classification/` | M1 | `JPEGImages/` (clean standardized 512×512 images), `JPEGImages_excluded/` (watermarked images) |
| `M1_outputs/` | M1 | `m1_dataset_manifest.csv`, `m1_5x2_fold_assignments.csv`, `m1_lesion_boxes_512.json` |
| `M3_outputs/` | M3 | `roi_masks/*.png` (19-class BiSeNet facial skin and nose segmentation masks) |
| `M4_outputs/` | M4 | `m4_feature_matrix.csv` (40 chromatic $a^*$ statistics + luminance/chromatic GLCM features) |
| `M2_outputs/` | M2 | `checkpoints/` (Faster R-CNN fold weights), `m2_rcnn_5x2_oof_predictions.csv` |
| `M5_outputs/` | M5 | `models/rf_best_model.pkl`, `m5_rf_5x2_oof_predictions.csv` |
| `M6_outputs/` | M6 | `m6_evaluation_report.json` (Macro F1, QWK $\kappa_w$, ROC-AUC), `m6_confusion_matrices.png`, latency & memory benchmark summaries |
| `M7_outputs/` | M7 | `m7_shap_feature_importance.csv`, `m7_global_shap_bar.png`, `m7_global_shap_beeswarm.png`, `local_plots/` |
| `Single_Image_Evaluation/` | Demo | `single_image_<filename>_dashboard.png`, `single_image_<filename>_preprocessing.png`, `single_image_<filename>_report.json` |
