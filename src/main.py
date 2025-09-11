import json
import os
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from utils.train_model import train

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Detector",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff4b4b;
        margin-bottom: 1rem;
    }
    .info-text {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff7070;
        color: white;
    }
    .risk-low {
        color: #2ecc71;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .feature-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def _compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _build_user_input_dataframe(user_input_dict: dict, feature_schema: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_input_dict])
    # enforce column order and presence
    ordered_cols = feature_schema["numeric"] + feature_schema["categorical"]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[ordered_cols]
    return df


def predict_with_pipeline(model, user_input: dict, feature_schema: dict):
    x_df = _build_user_input_dataframe(user_input, feature_schema)
    x_df = x_df.apply(pd.to_numeric, errors='coerce')
    x_df.fillna(x_df.median(numeric_only=True), inplace=True)
    proba = model.predict_proba(x_df)[0, 1]
    pred = int(proba >= 0.5)
    return pred, proba

# Create feature importance visualization
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        fig = px.bar(
            x=importance[indices][:10],
            y=[feature_names[i] for i in indices[:10]],
            orientation='h',
            title="Top 10 Most Important Features for Prediction",
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        fig.update_layout(showlegend=False)
        return fig
    return None

# Create risk factors visualization
def plot_risk_factors(user_input):
    # Define which factors contribute to risk
    risk_factors = {
        'High Cholesterol': user_input['chol'] > 240,
        'High Blood Pressure': user_input['trestbps'] > 140,
        'Low Max Heart Rate': user_input['thalach'] < 120,
        'ST Depression': user_input['oldpeak'] > 1.0,
        'Exercise Angina': user_input['exang'] == 1,
        'High Fasting Sugar': user_input['fbs'] == 1
    }
    
    factors = list(risk_factors.keys())
    values = [1 if risk_factors[factor] else 0 for factor in factors]
    
    fig = px.bar(
        x=values,
        y=factors,
        orientation='h',
        title="Your Risk Factors",
        labels={'x': 'Present', 'y': 'Risk Factor'}
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[0, 1], showticklabels=False)
    return fig

# Persistence helpers
@st.cache_resource(show_spinner=False)
def load_or_train_model(data_path: str, force_retrain: bool = False):
    models_dir = Path(__file__).parent.parent / "models"
    _ensure_dir(models_dir)

    model_path = models_dir / "model.joblib"
    meta_path = models_dir / "metadata.json"

    data_sha = _compute_file_sha256(data_path)

    if model_path.exists() and meta_path.exists() and not force_retrain:
        try:
            model = joblib.load(model_path)
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            return model, metadata
        except Exception:
            pass

    # Train fresh
    df, model, metrics, _, y_test, y_proba, feature_schema = train(data_path)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path,
        "data_sha256": data_sha,
        "metrics": metrics,
        "feature_schema": feature_schema,
        "notes": "Calibrated LogisticRegression with leakage-free preprocessing",
    }

    joblib.dump(model, model_path)
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Store eval artifacts for UI
    eval_dir = models_dir / "eval"
    _ensure_dir(eval_dir)
    joblib.dump({"y_test": y_test, "y_proba": y_proba}, eval_dir / "holdout.joblib")
    joblib.dump({"df": df}, eval_dir / "df.joblib")

    return model, metadata


# Streamlit app
def app(model, metadata):
    # Header section
    st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Detector</h1>', unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Health Dashboard", "Model Insights"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Enter Your Health Parameters</div>', unsafe_allow_html=True)

            with st.form("risk_form"):
                # Organize inputs into expandable sections
                with st.expander("Personal Information", expanded=True):
                    age = st.slider("Age", 18, 100, 50)
                    sex = st.radio("Sex", ["Female", "Male"], index=1)
                    sex = 1 if sex == "Male" else 0

                with st.expander("Heart Health Metrics"):
                    cp = st.selectbox("Chest Pain Type",
                                      ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                                      index=0)
                    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
                    cp = cp_map[cp]

                    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
                    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
                    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
                    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)

                with st.expander("Additional Health Indicators"):
                    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], index=0)
                    fbs = 1 if fbs == "Yes" else 0

                    restecg = st.selectbox("Resting ECG",
                                           ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                                           index=0)
                    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
                    restecg = restecg_map[restecg]

                    exang = st.radio("Exercise Induced Angina", ["No", "Yes"], index=0)
                    exang = 1 if exang == "Yes" else 0

                    slope = st.selectbox("Slope of Peak Exercise ST Segment",
                                         ["Upsloping", "Flat", "Downsloping"],
                                         index=1)
                    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
                    slope = slope_map[slope]

                    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)

                    thal = st.selectbox("Thalassemia",
                                        ["Normal", "Fixed Defect", "Reversible Defect"],
                                        index=0)
                    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
                    thal = thal_map[thal]

                threshold = st.slider("Classification threshold", 0.05, 0.95, 0.5, 0.01,
                                       help="Risk above this threshold is classified as High risk")

                submitted = st.form_submit_button("Assess My Heart Disease Risk")

            user_input = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
                "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }

        with col2:
            st.markdown('<div class="sub-header">Your Health Summary</div>', unsafe_allow_html=True)
            
            # Display health metrics in a visually appealing way
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.metric("Age", f"{age} years")
            st.metric("Blood Pressure", f"{trestbps} mm Hg", 
                     help="Normal: <120 mm Hg, Elevated: 120-129 mm Hg, High: ≥130 mm Hg")
            st.metric("Cholesterol", f"{chol} mg/dL", 
                     help="Desirable: <200 mg/dL, Borderline: 200-239 mg/dL, High: ≥240 mg/dL")
            st.metric("Max Heart Rate", f"{thalach} bpm", 
                     help="Average max heart rate: 220 - age")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk factors visualization
            risk_fig = plot_risk_factors(user_input)
            st.plotly_chart(risk_fig, use_container_width=True)
        
        # Prediction button
        if 'submitted' in locals() and submitted:
            with st.spinner("Analyzing your health data..."):
                time.sleep(0.6)
                pred_default, proba = predict_with_pipeline(model, user_input, metadata["feature_schema"])
                pred = int(proba >= threshold)
                risk_percentage = proba * 100

                # Animated result display
                if pred == 0:
                    st.markdown('<p class="risk-low">✅ Low risk of heart disease</p>', unsafe_allow_html=True)
                    st.success(f"Your estimated risk: {risk_percentage:.1f}%")
                else:
                    st.markdown('<p class="risk-high">⚠️ High risk of heart disease</p>', unsafe_allow_html=True)
                    st.error(f"Your estimated risk: {risk_percentage:.1f}%")

                # Progress bar showing risk level
                st.progress(min(max(proba, 0.0), 1.0))

                # Recommendations based on risk level
                st.markdown('<div class="sub-header">Recommendations</div>', unsafe_allow_html=True)
                if risk_percentage < 20:
                    st.info(
                        """
                        - Maintain your healthy lifestyle
                        - Continue regular exercise
                        - Schedule annual check-ups
                        """
                    )
                elif risk_percentage < 50:
                    st.warning(
                        """
                        - Consider dietary improvements
                        - Increase physical activity
                        - Monitor blood pressure regularly
                        - Consult with your doctor
                        """
                    )
                else:
                    st.error(
                        """
                        - Consult a cardiologist promptly
                        - Implement significant lifestyle changes
                        - Monitor health indicators closely
                        - Follow medical advice carefully
                        """
                    )

                # Structured prediction logging
                logs_dir = Path(__file__).parent.parent / "data" / "logs"
                _ensure_dir(logs_dir)
                log_row = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "risk_score": float(proba),
                    "pred": int(pred),
                    "threshold": float(threshold),
                    "model_trained_at": metadata.get("trained_at"),
                    "model_data_sha256": metadata.get("data_sha256"),
                    **user_input,
                }
                try:
                    with open(logs_dir / "predictions.ndjson", "a") as f:
                        f.write(json.dumps(log_row) + "\n")
                except Exception:
                    pass

                # Also maintain a CSV alongside the NDJSON for easy ingestion (e.g., Power BI)
                try:
                    ndjson_path = logs_dir / "predictions.ndjson"
                    csv_path = logs_dir / "predictions.csv"

                    # Define a stable column order
                    base_cols = [
                        "ts",
                        "risk_score",
                        "pred",
                        "threshold",
                        "model_trained_at",
                        "model_data_sha256",
                    ]
                    feature_cols = metadata.get("feature_schema", {}).get("numeric", []) + \
                                   metadata.get("feature_schema", {}).get("categorical", [])
                    ordered_cols = base_cols + feature_cols

                    if not csv_path.exists():
                        # Backfill: convert existing NDJSON to CSV and include the current row
                        rows = []
                        if ndjson_path.exists():
                            with open(ndjson_path, "r") as nf:
                                for line in nf:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        rows.append(json.loads(line))
                                    except Exception:
                                        continue
                        # If NDJSON didn't exist (edge case), include the current row
                        if not ndjson_path.exists():
                            rows.append(log_row)

                        df_csv = pd.DataFrame(rows)
                        # Ensure all expected columns exist
                        for c in ordered_cols:
                            if c not in df_csv.columns:
                                df_csv[c] = pd.NA
                        df_csv = df_csv[ordered_cols]
                        df_csv.to_csv(csv_path, index=False)
                    else:
                        # Append just the current row
                        df_row = pd.DataFrame([log_row])
                        for c in ordered_cols:
                            if c not in df_row.columns:
                                df_row[c] = pd.NA
                        df_row = df_row[ordered_cols]
                        df_row.to_csv(csv_path, mode="a", header=False, index=False)
                except Exception:
                    # CSV maintenance is best-effort; avoid breaking the UI on errors
                    pass
    
    
    
    with tab2:
        st.markdown('<div class="sub-header">Model Performance Insights</div>', unsafe_allow_html=True)

        metrics = metadata.get("metrics", {})
        col1, col2, col3, col4, col5 = st.columns(5)
        if metrics:
            col1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
            col2.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
            col3.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
            col4.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
            col5.metric("PR AUC", f"{metrics.get('pr_auc', 0):.3f}")

        # Load evaluation artifacts
        eval_art = joblib.load((Path(__file__).parent.parent / "models" / "eval" / "holdout.joblib"))
        y_test = eval_art["y_test"]
        y_proba = eval_art["y_proba"]
        y_pred = (y_proba >= 0.5).astype(int)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            text_auto=True,
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            x=['Low Risk', 'High Risk'], y=['Low Risk', 'High Risk']
        )
        fig.update_layout(title="Confusion Matrix (threshold=0.5)")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">About This Tool</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-text" style="color: black;">
        This Heart Disease Risk Predictor uses machine learning to estimate your risk of heart disease 
        based on health parameters. The model was trained on the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

        **Disclaimer:** This tool is for informational purposes only and is not a substitute for professional medical advice. 
        Always consult with healthcare providers for proper diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">How It Works</div>', unsafe_allow_html=True)
        st.write("""
        1. The model was trained using a Random Forest classifier
        2. Your inputs are preprocessed in the same way as the training data
        3. The model provides a probability score for heart disease risk
        4. Results are displayed with recommendations based on your risk level
        """)
        
        st.markdown('<div class="sub-header">Parameter Descriptions</div>', unsafe_allow_html=True)
        param_info = pd.DataFrame({
            'Parameter': ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 
                         'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate',
                         'Exercise Induced Angina', 'ST Depression', 'Slope', 
                         'Major Vessels', 'Thalassemia'],
            'Description': [
                'Age in years',
                'Biological sex (0 = female, 1 = male)',
                'Type of chest pain experienced',
                'Resting blood pressure in mm Hg',
                'Serum cholesterol in mg/dl',
                'Whether fasting blood sugar > 120 mg/dl',
                'Resting electrocardiographic results',
                'Maximum heart rate achieved during exercise',
                'Whether exercise induces angina',
                'ST depression induced by exercise relative to rest',
                'Slope of the peak exercise ST segment',
                'Number of major vessels colored by fluoroscopy',
                'Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversible defect)'
            ]
        })
        st.table(param_info)

    with tab3:
        st.markdown('<div class="sub-header">About This Tool</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-text" style="color: black;">
        This Heart Disease Risk Predictor uses a calibrated logistic regression model with a
        leakage-free preprocessing pipeline (scaling + one-hot encoding fit only on training data).

        <strong>Disclaimer:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice.
        Always consult with healthcare providers for proper diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sub-header">Model & Data</div>', unsafe_allow_html=True)
        st.json({
            "trained_at": metadata.get("trained_at"),
            "data_sha256": metadata.get("data_sha256"),
            "notes": metadata.get("notes"),
        })


# Main
def main():
    DATA_PATH = str(Path(__file__).parent.parent / "data" / "processed.cleveland.data")

    with st.sidebar:
        st.markdown("### Model Controls")
        retrain = st.button("Retrain model")

    model, metadata = load_or_train_model(DATA_PATH, force_retrain=retrain)

    # Inform if data hash changed
    current_sha = _compute_file_sha256(DATA_PATH)
    if metadata.get("data_sha256") != current_sha:
        st.sidebar.warning("Data file has changed since training. Consider retraining.")

    app(model, metadata)

if __name__ == "__main__":
    main()