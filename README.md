# 🩺 Diabetes Risk Prediction — Hybrid ML Framework

A Streamlit prototype that estimates an individual's diabetes risk (**Low / Moderate / High**)
from eight health parameters, using a hybrid **stacked ensemble** and a lightweight,
SHAP-style explainability layer.

> Built as a mock-up demonstration for the **Computational Biology 1** group project.
> ⚠️ **Not for clinical use** — for educational / demonstration purposes only.

---

## Overview

The app trains a small stacked ensemble on the Frankfurt Hospital diabetes dataset
(2,000 patients), then lets a user enter health parameters through interactive
sliders to receive:

- A **risk gauge** showing the predicted probability and a Low / Moderate / High label.
- A **contributing factors** chart, in plain language, showing which inputs push the
  risk up (red, right) or down (green, left), with the user's own value on each bar.
- A **"Why did I get this result?"** explanation that adapts to the risk level and
  names the values working against the user most.
- **Personalized prevention tips** for the factors raising the user's risk, plus
  general advice for everyone on how to lower diabetes risk.

### Pipeline

```
Input (health parameters)
        ↓
Preprocessing (StandardScaler)
        ↓
Hybrid stacked ensemble
  ├─ Random Forest
  ├─ Gradient Boosting
  └─ SVM
        ↓  (combined by a Logistic Regression meta-learner)
Risk prediction
        ↓
Explainability layer (SHAP-style local contributions)
```

---

## Features

- **Hybrid stacked ensemble** — Random Forest + Gradient Boosting + SVM, combined
  by a Logistic Regression meta-learner (`StackingClassifier`).
- **Resilient data loading** — uses a bundled local CSV first, falls back to a remote
  download, then to a 2,000-row synthetic dataset if neither is available.
- **Missing-value handling** — biologically impossible zeros (Glucose, BloodPressure,
  SkinThickness, Insulin, BMI) are treated as missing and imputed with the column median.
- **Cached training** — the model trains once per session (`@st.cache_resource`).
- **Interactive UI** — sliders for all eight features (with helpful tooltips, e.g.
  blood pressure is the *lower* number of a reading), a Plotly risk gauge, and a
  plain-language contributing-factors chart.
- **Plain-language explanations** — a "Why did I get this result?" summary plus
  personalized and general tips on how to lower diabetes risk.
- **Model metrics** — AUC, accuracy, and training sample count shown in the header.

---

## Input Parameters

| Feature | Range | Description |
|---|---|---|
| Pregnancies | 0–17 | Number of times pregnant |
| Glucose | 50–200 | Plasma glucose (mg/dL) |
| BloodPressure | 40–130 | Diastolic blood pressure — the **lower** number in a reading, e.g. the 80 in 120/80 (mm Hg) |
| SkinThickness | 0–99 | Triceps skin fold thickness (mm) |
| Insulin | 0–846 | 2-hour serum insulin (mu U/ml) |
| BMI | 15.0–60.0 | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | 0.05–2.5 | Family history score |
| Age | 18–90 | Age (years) |

---

## Installation

Requires **Python 3.9+**.

```bash
# (optional) create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run main.py
```

The app opens in your browser (default: http://localhost:8501). Set the patient
parameters on the left and click **Predict Risk**.

---

## Project Structure

```
Cb1_Project/
├── main.py                  # Streamlit app + model training + UI
├── frankfurt_diabetes.csv   # Bundled dataset (optional; auto-downloaded if absent)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## How It Works

1. **Data** — `load_data()` loads the bundled `frankfurt_diabetes.csv` if present,
   otherwise downloads it from a public mirror; if both fail, `synth_data()` generates
   a plausible synthetic dataset with a latent risk signal driven by glucose, BMI, age,
   and pedigree function. `clean_zeros()` then imputes impossible zeros with the column median.
2. **Training** — features are scaled, then fed into the stacked ensemble with 5-fold
   cross-validation for the meta-learner. AUC and accuracy are evaluated on a 20% hold-out.
3. **Prediction** — user inputs are passed through the same pipeline to produce a
   probability, mapped to Low (<33%), Moderate (33–66%), or High (≥66%) risk.
4. **Explainability** — `local_contributions()` approximates each feature's local
   effect by combining its global Random Forest importance with how far the user's
   value sits from the population mean (in standard-deviation units).
5. **Explanation & guidance** — the app turns those contributions into a plain-language
   "Why did I get this result?" summary (naming the biggest factors), then shows
   prevention tips: targeted advice for the factors raising the user's risk and a
   general healthy-living checklist.

---

## Notes & Limitations

- This is a **prototype / mock-up**, not a validated diagnostic tool.
- Some columns use `0` to encode missing values (e.g. Insulin, SkinThickness); these
  are imputed with the column median via `clean_zeros()`, but median imputation is a
  simplification and may not reflect true patient values.
- The contributing-factors chart is a **SHAP-style approximation**, not exact SHAP
  values. Feature importances are unsigned, so the displayed direction reflects only
  whether a value is above/below the population mean — interpret it as indicative.
- Do **not** use this for real medical decisions.
