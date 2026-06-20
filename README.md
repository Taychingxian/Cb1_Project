# 🩺 Diabetes Risk Prediction — Hybrid ML Framework

A Streamlit prototype that estimates an individual's diabetes risk (**Low / Moderate / High**)
from eight health parameters, using a hybrid **stacked ensemble** and a lightweight,
SHAP-style explainability layer.

> Built as a mock-up demonstration for the **Computational Biology 1** group project.
> ⚠️ **Not for clinical use** — for educational / demonstration purposes only.

---

## Overview

The app trains a small stacked ensemble on the PIMA Indians Diabetes dataset
(with a synthetic fallback if no internet is available), then lets a user enter
health parameters through interactive sliders to receive:

- A **risk gauge** showing the predicted probability and a Low / Moderate / High label.
- A **top contributing factors** chart indicating which inputs push the prediction
  toward higher (red) or lower (blue) risk.

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
- **Offline-safe** — automatically falls back to a 2,000-row synthetic dataset if the
  PIMA dataset can't be downloaded.
- **Cached training** — the model trains once per session (`@st.cache_resource`).
- **Interactive UI** — sliders for all eight features, a Plotly risk gauge, and a
  contributing-factors bar chart.
- **Model metrics** — AUC, accuracy, and training sample count shown in the header.

---

## Input Parameters

| Feature | Range | Description |
|---|---|---|
| Pregnancies | 0–17 | Number of times pregnant |
| Glucose | 50–200 | Plasma glucose (mg/dL) |
| BloodPressure | 40–130 | Diastolic blood pressure (mm Hg) |
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
├── main.py            # Streamlit app + model training + UI
├── requirements.txt   # Python dependencies
└── README.md
```

---

## How It Works

1. **Data** — `load_data()` fetches the PIMA dataset from a public mirror; if that
   fails, `synth_data()` generates a plausible synthetic dataset with a latent
   risk signal driven by glucose, BMI, age, and pedigree function.
2. **Training** — features are scaled, then fed into the stacked ensemble with 5-fold
   cross-validation for the meta-learner. AUC and accuracy are evaluated on a 20% hold-out.
3. **Prediction** — user inputs are passed through the same pipeline to produce a
   probability, mapped to Low (<33%), Moderate (33–66%), or High (≥66%) risk.
4. **Explainability** — `local_contributions()` approximates each feature's local
   effect by combining its global Random Forest importance with how far the user's
   value sits from the population mean (in standard-deviation units).

---

## Notes & Limitations

- This is a **prototype / mock-up**, not a validated diagnostic tool.
- The PIMA dataset uses `0` to encode some missing values (e.g. Insulin,
  SkinThickness); these are not imputed in this prototype.
- The contributing-factors chart is a **SHAP-style approximation**, not exact SHAP
  values. Feature importances are unsigned, so the displayed direction reflects only
  whether a value is above/below the population mean — interpret it as indicative.
- Do **not** use this for real medical decisions.
