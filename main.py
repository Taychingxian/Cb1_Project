import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
)

FEATURES = [
    ("Pregnancies", 0, 17, 1, "Number of times pregnant"),
    ("Glucose", 50, 200, 117, "Plasma glucose (mg/dL)"),
    ("BloodPressure", 40, 130, 72, "Diastolic blood pressure (mm Hg)"),
    ("SkinThickness", 0, 99, 23, "Triceps skin fold thickness (mm)"),
    ("Insulin", 0, 846, 30, "2-Hour serum insulin (mu U/ml)"),
    ("BMI", 15.0, 60.0, 32.0, "Body mass index (kg/m²)"),
    ("DiabetesPedigreeFunction", 0.05, 2.5, 0.37, "Family history score"),
    ("Age", 18, 90, 33, "Age (years)"),
]
FEATURE_NAMES = [f[0] for f in FEATURES]


# ----------------------------------------------------------------------------
# Data + model (cached so it only trains once)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    """Load PIMA dataset; fall back to a synthetic dataset if offline."""
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv"
    )
    cols = FEATURE_NAMES + ["Outcome"]
    try:
        df = pd.read_csv(url, header=None, names=cols)
        source = "PIMA Indians Diabetes dataset"
    except Exception:
        df = synth_data()
        source = "synthetic fallback dataset (offline)"
    return df, source


def synth_data(n=2000, seed=42):
    """Generate a plausible synthetic dataset if the real one can't be fetched."""
    rng = np.random.default_rng(seed)
    glucose = rng.normal(120, 30, n).clip(50, 200)
    bmi = rng.normal(32, 7, n).clip(15, 60)
    age = rng.integers(18, 90, n)
    bp = rng.normal(72, 12, n).clip(40, 130)
    skin = rng.normal(23, 10, n).clip(0, 99)
    insulin = rng.normal(120, 100, n).clip(0, 846)
    preg = rng.integers(0, 17, n)
    dpf = rng.gamma(2.0, 0.25, n).clip(0.05, 2.5)
    # latent risk
    z = (
        0.03 * (glucose - 120)
        + 0.05 * (bmi - 32)
        + 0.02 * (age - 33)
        + 0.8 * (dpf - 0.37)
        + rng.normal(0, 1, n)
    )
    prob = 1 / (1 + np.exp(-z))
    outcome = (prob > 0.5).astype(int)
    return pd.DataFrame({
        "Pregnancies": preg, "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age, "Outcome": outcome,
    })


@st.cache_resource(show_spinner=False)
def train_model():
    df, source = load_data()
    X = df[FEATURE_NAMES].values
    y = df["Outcome"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("gb", GradientBoostingClassifier(random_state=42)),
        ("svm", SVC(probability=True, random_state=42)),
    ]
    stack = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
    )
    model = Pipeline([("scaler", StandardScaler()), ("clf", stack)])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "auc": roc_auc_score(y_test, proba),
        "n": len(df),
        "source": source,
    }

    # simple global importance via the random forest base learner
    rf = model.named_steps["clf"].named_estimators_["rf"]
    importances = dict(zip(FEATURE_NAMES, rf.feature_importances_))
    return model, metrics, importances, df


def local_contributions(model, user_row, df, importances):
    """Lightweight SHAP-style local explanation.

    Approximates each feature's push by combining its global importance with
    how far the user's value sits from the population mean (in std units).
    """
    means = df[FEATURE_NAMES].mean()
    stds = df[FEATURE_NAMES].std().replace(0, 1)
    contribs = {}
    for f in FEATURE_NAMES:
        z = (user_row[f] - means[f]) / stds[f]
        contribs[f] = importances[f] * z
    return contribs


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🩺 Diabetes Risk Prediction")
st.caption(
    "Using a Hybrid Machine Learning Framework — Computational Biology 1 Prototype"
)

with st.spinner("Training hybrid stacked ensemble..."):
    model, metrics, importances, df = train_model()

# Header metrics
m1, m2, m3 = st.columns(3)
m1.metric("Model AUC", f"{metrics['auc']:.3f}")
m2.metric("Model Accuracy", f"{metrics['accuracy']*100:.1f}%")
m3.metric("Training samples", f"{metrics['n']:,}")
st.caption(f"Data source: {metrics['source']}. Prototype for demonstration only — not for clinical use.")

st.divider()

left, right = st.columns([1, 1.4])

# ---- Left: input form ----
with left:
    st.subheader("Patient Health Parameters")
    user_vals = {}
    for name, lo, hi, default, help_txt in FEATURES:
        if isinstance(default, float):
            user_vals[name] = st.slider(name, float(lo), float(hi), float(default), help=help_txt)
        else:
            user_vals[name] = st.slider(name, int(lo), int(hi), int(default), help=help_txt)
    predict = st.button("Predict Risk", type="primary", use_container_width=True)

# ---- Right: results ----
with right:
    st.subheader("Predicted Risk")
    if predict:
        row = pd.Series(user_vals)
        X_user = np.array([[user_vals[f] for f in FEATURE_NAMES]])
        prob = float(model.predict_proba(X_user)[0, 1])

        if prob < 0.33:
            level, color = "LOW", "#2ecc71"
        elif prob < 0.66:
            level, color = "MODERATE", "#f39c12"
        else:
            level, color = "HIGH", "#e74c3c"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": f"{level} RISK"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 33], "color": "#eafaf1"},
                    {"range": [33, 66], "color": "#fef5e7"},
                    {"range": [66, 100], "color": "#fdedec"},
                ],
            },
        ))
        gauge.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

        # ---- contributing factors ----
        st.subheader("Top Contributing Factors")
        contribs = local_contributions(model, row, df, importances)
        items = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        labels = [k for k, _ in items][::-1]
        values = [v for _, v in items][::-1]
        bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

        bar = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=bar_colors,
        ))
        bar.update_layout(
            height=320, margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="← lowers risk      raises risk →",
        )
        st.plotly_chart(bar, use_container_width=True)

        st.info(
            "Red bars push the prediction toward higher risk; blue bars toward "
            "lower risk. This is a SHAP-style approximation for the prototype."
        )
    else:
        st.write("Set the parameters on the left and click **Predict Risk**.")

st.divider()
with st.expander("About this prototype"):
    st.markdown(
        """
        **Framework:** Input (health parameters) → Preprocessing (scaling) →
        Feature contributions → Hybrid stacked ensemble
        (Random Forest + Gradient Boosting + SVM, combined by Logistic
        Regression meta-learner) → Risk prediction → Explainability layer.

        **Note:** This is a mock-up demonstration built for the Computational
        Biology 1 group project. The model is trained on a small public dataset
        and must **not** be used for real medical decisions.
        """
    )