---
title: Diabetes Risk Predictor
emoji: 🩺
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.3.0"
app_file: app.py
pinned: false
---

# 🩺 Diabetes Risk Predictor

An end-to-end Machine Learning system that predicts diabetes risk from clinical measurements. The pipeline combines **PyCaret AutoML**, a **Calibrated Soft-Voting Ensemble** of five tuned base learners, and a modern **Gradio** web interface deployed on Hugging Face Spaces.

---

## 🚀 Live Demo

| Resource | Link |
|---|---|
| 🤗 Hugging Face Space | [faisaltitu/Diabetics_Prediction](https://huggingface.co/spaces/faisaltitu/Diabetics_Prediction) |
| 💻 GitHub Repository | [faisal-titu/Diabetics_detection](https://github.com/faisal-titu/Diabetics_detection) |
---

## 📊 Model Performance

The final deployed model is a **Calibrated Soft-Voting Ensemble** selected after comparing three ensemble strategies:

| Strategy | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|---|
| Stacking Ensemble | 84.38% | 80.95% | 73.91% | 77.27% | 0.9128 |
| Hard Voting | 85.94% | 80.88% | 79.71% | 80.29% | — |
| Soft Voting | 86.98% | 82.35% | 81.16% | 81.75% | 0.9306 |
| **Calibrated Soft Voting ✅** | **90.10%** | **90.32%** | **81.16%** | **85.50%** | **0.9306** |

> The calibrated model achieves the best balance of precision and recall, minimising false negatives (missed diagnoses) while keeping false positives low.

---

## 🧠 Machine Learning Workflow

### 1. Dataset
- **Source:** Pima Indians Diabetes Dataset (768 samples, 8 features)
- **Target:** `Outcome` — `0` (No Diabetes) / `1` (Diabetes)

### 2. Data Preprocessing
- Medically impossible zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` replaced with **column means**
- Features scaled using **QuantileTransformer** (`n_quantiles=100`, `output_distribution='normal'`) for robust normalisation
- 80/20 stratified train-test split

### 3. Exploratory Analysis
- Correlation heatmaps, distribution plots, and pairplots
- **UMAP 2D & 3D** visualisations to inspect class separability in low-dimensional space

### 4. AutoML with PyCaret
- `compare_models()` benchmarked 15+ classifiers
- Top 5 base learners selected: **CatBoost, LightGBM, Random Forest, Extra Trees, Gradient Boosting**
- Each base learner independently tuned with `tune_model(optimize='AUC')`

### 5. Ensemble Strategies Compared

| Method | Description |
|---|---|
| `stack_models()` | Meta-learner stacks predictions from all base learners |
| `blend_models(method='hard')` | Majority vote across base learners |
| `blend_models(method='soft')` | Averages predicted probabilities — **best strategy** |

### 6. Calibration & Finalisation
- Best ensemble (Soft Voting) wrapped with **`CalibratedClassifierCV`** to produce well-calibrated probabilities
- `finalize_model()` retrains on the full dataset (train + test) before saving

### 7. Model Persistence

```python
import pickle
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
```

---

## 🖥️ Web Application

Built with **Gradio Blocks** for a fully custom two-column layout:

- **Left panel** — Eight sliders grouped by category (Personal Info, Blood Metrics, Body Measurements) with normal reference ranges info-box
- **Right panel** — Live result card with animated confidence bar, risk tier badge (Low / Moderate / High), and a probability breakdown with gradient progress bars
- **Quick Examples** — Five pre-filled patient profiles for instant testing
- **Clear button** — Resets all inputs to defaults in one click

### Input Features

| Feature | Unit | Typical Range |
|---|---|---|
| Pregnancies | count | 0 – 17 |
| Glucose | mg/dL | 70 – 99 |
| Blood Pressure | mmHg | 60 – 80 |
| Skin Thickness | mm | 10 – 40 |
| Insulin | µU/mL | 2 – 25 |
| BMI | kg/m² | 18.5 – 24.9 |
| Diabetes Pedigree Function | score | 0.08 – 2.5 |
| Age | years | 21 – 81 |

### Output

- **Prediction card** — ✅ No Diabetes / ⚠️ Diabetes Detected
- **Risk tier** — Low / Moderate / High based on predicted probability
- **Confidence score** with animated fill bar
- **Probability breakdown** — individual gradient bars for both classes

---

## 🗂️ Project Structure

```
Diabetics_detection/
├── app.py                                   # Gradio web app (Blocks UI)
├── diabetes_model.pkl                       # Saved calibrated ensemble pipeline
├── requirements.txt                         # Runtime dependencies (Hugging Face Spaces)
├── final_exam_diabetics_prediction.ipynb    # Baseline: Logistic Regression + sklearn Pipeline
├── diabetes-three-ensemble-models.ipynb     # Original ensemble training notebook
├── diabetes-three-ensemble-models_2.ipynb   # Final ensemble notebook (fully executed)
├── archive/
│   └── diabetes.csv                         # Pima Indians dataset (768 rows × 9 cols)
└── README.md
```

---

## ⚙️ How to Run Locally

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the App

```bash
python app.py
```

Open the local URL printed in the terminal (default: `http://127.0.0.1:7860`).

### Training Environment (optional)

The ensemble notebook requires additional packages not bundled in `requirements.txt`:

```bash
pip install pycaret catboost lightgbm shap umap-learn missingno
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | pandas, NumPy, Matplotlib, Seaborn, missingno |
| Dimensionality Reduction | UMAP |
| AutoML & Ensembling | PyCaret 3.3, scikit-learn 1.4 |
| Base Learners | CatBoost, LightGBM, Random Forest, Extra Trees, Gradient Boosting |
| Calibration | `CalibratedClassifierCV` (isotonic regression) |
| Model Persistence | pickle |
| Web Interface | Gradio 6.3 (Blocks API) |
| Deployment | Hugging Face Spaces |

---

> ⚕️ **Disclaimer:** This tool is for educational purposes only and is not a medical device. Always consult a qualified healthcare professional for diagnosis and treatment.
