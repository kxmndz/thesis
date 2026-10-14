# Explainable Acne Severity Grading Pipeline (M1–M7)

An end-to-end, reproducible command-line pipeline for **Automated Acne Vulgaris Severity Grading** on the **ACNE04** dataset (1,457 clinical facial images graded 0–3 under the Hayashi Criterion).

This repository implements the complete **M1–M7 System Architecture** from the thesis proposal, comparing a **Chromatic (`CIELAB a*`) + Masked 32-Level GLCM Texture Random Forest Classifier (M4–M5)** against the **Faster R-CNN Lesion-Counting Baseline (M2)** from [Wen et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9028662/) ([`wenh06/acne_detection`](https://github.com/wenh06/acne_detection)) using a **5×2 Cross-Validation (10-fold)** protocol with **Alpaydın (1999) Combined 5×2 CV F-Test (M6)** and **TreeSHAP Global & Local Interpretability (M7)**.

---

## 1. Quick Start (Under 2 Minutes)

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the full M1 -> M7 pipeline end-to-end
python pipeline_cli.py run-pipeline --root-dir . --m3-variant m3c --m2-mode pretrained

# 3. Inspect a single image through preprocessing, CIELAB channels, masking, RF + R-CNN, and SHAP
python pipeline_cli.py evaluate-image --root-dir . --image-id levle2_93
```

---

## 2. System Architecture Overview (M1–M7)

| Module | Script | Description | Key Output Files |
| :--- | :--- | :--- | :--- |
| **M1** | `m1_data_loader.py` | Parses ACNE04 filenames, ground-truth Hayashi grades (`0–3`), lesion counts, and PASCAL VOC bounding boxes; generates the **Stratified 5×2 CV** manifest across seeds `{42, 73, 105, 224, 911}` (10 total fold evaluations). | `M1_outputs/m1_dataset_manifest.csv`, `M1_outputs/m1_5x2_fold_assignments.csv` |
| **M2** | `m2_baseline_rcnn.py` | Implements the Faster R-CNN lesion-counting baseline from [`wenh06/acne_detection`](https://github.com/wenh06/acne_detection). Supports `--mode pretrained` (`faster_rcnn_resnet101` / `faster_rcnn_inception_v2`) or `--mode retrain` (K-means anchor box optimization strictly on each training fold + 5×2 CV training with atomic per-epoch/per-fold checkpoints). Converts predicted lesion counts to Hayashi severity grades (`0–5` → 0, `6–20` → 1, `21–50` → 2, `>50` → 3). | `M2_outputs/m2_rcnn_5x2_oof_predictions.csv`, `M2_outputs/checkpoints/m2_checkpoint_state.json` |
| **M3** | `m3c_face_parsing.py` / `m3_bisenet_pipeline.py` | Standardizes images to `512×512` (aspect-preserving letterbox), isolates facial skin (`skin=1`, `nose=10`) while removing hair, background, eyes, brows, and lips, converts RGB to `CIELAB (L*, a*, b*)`, and applies CLAHE (`clipLimit=2.0`, `tileGridSize=8×8`) strictly to `L*`. | `M3_outputs/roi_masks/`, `M3_outputs/masked_images/`, `M3_outputs/m4_handoff/` |
| **M4** | `m4_feature_extraction.py` | Extracts **Chromatic Features** from masked `CIELAB a*` (`mean`, `median`, `variance`, `90th percentile`) and **Texture Features** from 32-level quantized CLAHE `L*`/grayscale using a custom masked GLCM (ignoring pixel pairs outside the skin mask) across distances `[1, 3, 5, 10]` and angles `[0°, 45°, 90°, 135°]` averaged for rotation invariance (`contrast`, `correlation`, `energy`, `homogeneity`). | `M4_outputs/m4_feature_matrix.csv` |
| **M5** | `m5_random_forest.py` | Trains a `RandomForestClassifier(class_weight="balanced")` across all 10 folds of the 5×2 CV split using **Nested Stratified 3-Fold CV GridSearch** (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`) strictly on the training split, with atomic per-fold checkpointing. | `M5_outputs/m5_rf_5x2_oof_predictions.csv`, `M5_outputs/models/rf_best_model.pkl` |
| **M6** | `m6_model_evaluation.py` | Evaluates M5 Random Forest and M2 Faster R-CNN across all 10 folds: **Macro F1-Score**, **Precision**, **Recall**, **Per-Class F1**, **4×4 Confusion Matrix**, **One-vs-One (OvO) Multi-Class AUC-ROC**, **Inference Latency & Memory**, and the **Alpaydın (1999) Combined 5×2 CV F-Test** ($F \sim F(10, 5)$, $\alpha = 0.05$). | `M6_outputs/m6_evaluation_report.json`, `M6_outputs/m6_comparative_evaluation.png` |
| **M7** | `m7_shap_interpretability.py` | Runs **SHAP `TreeExplainer`** on the best-performing Random Forest model to produce **Global Interpretability** (feature importance bar chart + directional beeswarm plot) and **Local Interpretability** (per-image waterfall and force plots). | `M7_outputs/m7_global_shap_bar.png`, `M7_outputs/m7_global_shap_beeswarm.png`, `M7_outputs/local_explanations/` |
| **Single Image** | `evaluate_single_image.py` | Runs any chosen image through `512×512` preprocessing, ROI masking, `CIELAB L*, a*, b*` channel visualization, M4 feature extraction, side-by-side Random Forest and Faster R-CNN grading, and local SHAP waterfall/force explanations. | `Single_Image_Evaluation/*_preprocessing_cielab.png`, `*_evaluation_dashboard.png`, `*_shap_waterfall.png`, `*_shap_force.png` |

---

## 3. CLI Usage Guide (`pipeline_cli.py`)

View all available subcommands at any time:
```bash
python pipeline_cli.py --help
```

### 3.1 Run the Full Pipeline (`run-pipeline`)
Executes M1 → M3 → M2 → M4 → M5 → M6 → M7 sequentially:
```bash
python pipeline_cli.py run-pipeline \
  --root-dir . \
  --m3-variant m3c \
  --m2-mode pretrained \
  --m2-arch faster_rcnn_resnet101
```
Useful flags:
- `--skip-m3`: Skip M3 image preprocessing if masks are already generated in `M3_outputs/`.
- `--skip-m2`: Skip M2 baseline R-CNN execution if `M2_outputs/m2_rcnn_5x2_oof_predictions.csv` already exists.
- `--m2-mode {pretrained,retrain}`: Choose between pretrained inference or full 5×2 CV retraining.

---

### 3.2 Module 1: Data Loader & Stratified 5×2 CV Split (`m1`)
```bash
python pipeline_cli.py m1 --root-dir .
```

---

### 3.3 Module 2: Baseline Faster R-CNN Lesion Counting (`m2`)
Based on [`wenh06/acne_detection`](https://github.com/wenh06/acne_detection):
- **Pretrained Faster R-CNN ResNet-101** (MEGA weight link included in script; auto-downloads or accepts `--weights-path`):
  ```bash
  python pipeline_cli.py m2 \
    --root-dir . \
    --mode pretrained \
    --pretrained-arch faster_rcnn_resnet101 \
    --conf-threshold 0.5
  ```
- **Retrain Faster R-CNN ResNet-101 across all 10 5×2 CV folds** (with K-means anchor optimization on each training fold and automatic checkpointing):
  ```bash
  python pipeline_cli.py m2 \
    --root-dir . \
    --mode retrain \
    --epochs 5 \
    --batch-size 2 \
    --lr 0.005 \
    --drive-backup-dir /content/drive/MyDrive/acne_m2_checkpoints
  ```
- **Colab / Laptop Interruption Recovery**:
  - Progress is saved after **every epoch** and **every fold** to `M2_outputs/checkpoints/m2_checkpoint_state.json` (and mirrored to `--drive-backup-dir` if specified).
  - Re-running the exact same command automatically skips completed folds and resumes from the last saved epoch. Use `--reset-checkpoint` only if you want to start fresh.

---

### 3.4 Module 3: Facial Skin Segmentation & CIELAB/CLAHE Handoff (`m3a-audit`, `m3b`, `m3c`, `m3-bisenet`, `m4-handoff`)
```bash
# Option A: Run M3C BiSeNet Face Parsing (19-class CelebAMask-HQ + nose-tip protection)
python pipeline_cli.py m3c --root-dir .

# Option B: Run M3B 3-Stage Cascaded Fallback + YCrCb/HSV Chromatic Skin Retention
python pipeline_cli.py m3b --root-dir .

# Option C: Run Production BiSeNet Pipeline + M4 CIELAB/CLAHE Handoff in one step
python pipeline_cli.py m3-bisenet --root-dir . --run-m4-handoff
```

---

### 3.5 Module 4: Chromatic (`CIELAB a*`) & Masked GLCM Feature Extraction (`m4`)
Extracts the 24-feature matrix (`4` masked `a*` chromatic statistics + `16` distance-specific masked 32-level GLCM features across $d \in \{1, 3, 5, 10\}$ + `4` multi-scale aggregated GLCM descriptors):
```bash
python pipeline_cli.py m4 \
  --root-dir . \
  --distances 1 3 5 10 \
  --quant-levels 32 \
  --clahe-clip 2.0 \
  --clahe-grid 8
```

---

### 3.6 Module 5: Balanced Random Forest Training (`m5`)
Trains the Random Forest classifier (`class_weight="balanced"`) across all 10 folds of the 5×2 CV split using Nested Stratified CV for hyperparameter tuning on the training split only:
```bash
python pipeline_cli.py m5 \
  --root-dir . \
  --inner-cv 3
```
- **Checkpointing**: Completed folds are saved to `M5_outputs/checkpoints/m5_checkpoint_state.json`. Re-running `python pipeline_cli.py m5` automatically resumes from any unfinished fold. Pass `--reset-checkpoint` to force retraining all 10 folds.

---

### 3.7 Module 6: Automated Model Evaluation & Combined 5×2 CV F-Test (`m6`)
Automatically evaluates both **M5 Random Forest** and **M2 Baseline Faster R-CNN** across all 10 folds and runs the **Alpaydın (1999) Combined 5×2 CV F-Test**:
```bash
python pipeline_cli.py m6 \
  --root-dir . \
  --alpha 0.05
```
Generates:
- `M6_outputs/m6_evaluation_report.json`: Complete JSON report with 10-fold mean ± SD for Macro F1, Precision, Recall, OvO Multi-Class AUC-ROC, Per-Class F1 (`Grade 0–3`), 4×4 Confusion Matrices, latency/memory benchmarks, F-statistic, $p$-value, and automated **Model Adoption Decision**.
- `M6_outputs/m6_fold_comparison_table.csv`: Per-fold breakdown across all 10 fold configurations (`seed_42_swap1` through `seed_911_swap2`).
- `M6_outputs/m6_comparative_evaluation.png`: 4-panel figure (10-fold Macro F1 & OvO AUC-ROC comparison, M5 4×4 Confusion Matrix, M2 4×4 Confusion Matrix, and Paired 5×2 CV Fold Differences + F-Test summary).

---

### 3.8 Module 7: SHAP `TreeExplainer` Global & Local Interpretability (`m7`)
Computes TreeSHAP values for the best-performing Random Forest model and produces both global and local explanations:
```bash
python pipeline_cli.py m7 \
  --root-dir . \
  --top-display 12 \
  --num-local-samples 4
```
Generates:
- `M7_outputs/m7_global_shap_bar.png`: Global mean $|\text{SHAP}|$ feature importance bar chart stacked by severity class.
- `M7_outputs/m7_global_shap_beeswarm.png`: Global directional SHAP beeswarm plot showing how high/low feature values shift severity probability.
- `M7_outputs/local_explanations/`: Per-image **Waterfall** and **Force** plots for representative images from each severity grade (`Grade 0` to `Grade 3`).

---

### 3.9 Single-Image End-to-End Pipeline & Dual-Model Evaluator (`evaluate-image`)
Run any individual image through preprocessing, `CIELAB` channel extraction, ROI skin masking, M4 feature extraction, **Random Forest** vs. **Baseline Faster R-CNN** grading, and local **SHAP Waterfall & Force** plots:

```bash
# Evaluate by ACNE04 image ID
python pipeline_cli.py evaluate-image --root-dir . --image-id levle2_93

# Evaluate by dataset row index (0 to 1456)
python pipeline_cli.py evaluate-image --root-dir . --index 42

# Evaluate any custom external facial image file
python pipeline_cli.py evaluate-image --root-dir . --image-path /path/to/face_photo.jpg
```
Outputs in `Single_Image_Evaluation/`:
1. `single_image_<id>_preprocessing_cielab.png`: 8-panel visualization (`512×512` Standardized RGB, Binary Skin ROI Mask, Masked Facial Skin ROI, `CIELAB L*`, `CIELAB a*` Erythema/Redness heatmap, `CIELAB b*`, CLAHE-enhanced `L*`, and 32-level quantized GLCM input).
2. `single_image_<id>_evaluation_dashboard.png`: Side-by-side comparison of **Baseline Faster R-CNN** (detected lesion bounding boxes, lesion count, and Hayashi grade) vs. **M5 Random Forest** (class probability distribution and extracted M4 chromatic/GLCM feature table).
3. `single_image_<id>_shap_waterfall.png`: Local SHAP waterfall plot explaining the exact feature contributions for that image.
4. `single_image_<id>_shap_force.png`: Local SHAP force plot showing positive (severity-increasing) vs. negative (severity-decreasing) feature pushes.
5. `single_image_<id>_report.json`: Complete JSON summary of extracted features, model predictions, latencies, and SHAP contributions.
