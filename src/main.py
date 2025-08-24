import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
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

# Function to get the correct file path
def get_file_path(filename):
    """
    Get the absolute path to a file, checking multiple possible locations
    """
    # Check if file exists in the given path
    if os.path.exists(filename):
        return filename
    
    # Check relative to current directory
    current_dir = Path(__file__).parent
    possible_paths = [
        filename,
        current_dir / filename,
        current_dir / "data" / filename,
        current_dir / "../data" / filename,
        current_dir / "../../data" / filename,
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

# Preprocess user input
def preprocess_user_input(user_input_dict, oh_columns, scaler):
    df = pd.DataFrame([user_input_dict])
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.median(), inplace=True)

    # One-hot encode the same categorical features as training
    df = pd.get_dummies(df, columns=['cp', 'slope', 'ca', 'thal', 'restecg'])

    # Reindex to training columns
    df = df.reindex(columns=oh_columns, fill_value=0)

    # Scale
    scaled = scaler.transform(df)
    return scaled

# Predict
def predict_heart_disease(model, scaler, user_input, oh_columns):
    input_scaled = preprocess_user_input(user_input, oh_columns, scaler)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    return prediction, probability

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
        fig.update_traces(marker_line_color='black', marker_line_width=1)
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
    fig.update_traces(marker_line_color='black', marker_line_width=1)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[0, 1], showticklabels=False)
    return fig

# Streamlit app
def app(model, scaler, oh_columns, X, y_test, y_pred):
    # Header section
    st.markdown('<p class="main-header">❤️ Heart Disease Risk Predictor</p>', unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Assessment", "Health Dashboard", "Model Insights", "About"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Enter Your Health Parameters</div>', unsafe_allow_html=True)
            
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
        if st.button("Assess My Heart Disease Risk"):
            with st.spinner("Analyzing your health data..."):
                time.sleep(1.5)  # Simulate processing time
                pred, proba = predict_heart_disease(model, scaler, user_input, oh_columns)
                risk_percentage = proba[0][1] * 100
                
                # Animated result display
                if pred == 0:
                    st.markdown(f'<p class="risk-low">✅ Low risk of heart disease</p>', unsafe_allow_html=True)
                    st.success(f"Your estimated risk: {risk_percentage:.1f}%")
                else:
                    st.markdown(f'<p class="risk-high">⚠️ High risk of heart disease</p>', unsafe_allow_html=True)
                    st.error(f"Your estimated risk: {risk_percentage:.1f}%")
                
                # Progress bar showing risk level
                st.progress(risk_percentage/100)
                
                # Recommendations based on risk level
                st.markdown('<div class="sub-header">Recommendations</div>', unsafe_allow_html=True)
                if risk_percentage < 20:
                    st.info("""
                    - Maintain your healthy lifestyle
                    - Continue regular exercise
                    - Schedule annual check-ups
                    """)
                elif risk_percentage < 50:
                    st.warning("""
                    - Consider dietary improvements
                    - Increase physical activity
                    - Monitor blood pressure regularly
                    - Consult with your doctor
                    """)
                else:
                    st.error("""
                    - Consult a cardiologist promptly
                    - Implement significant lifestyle changes
                    - Monitor health indicators closely
                    - Follow medical advice carefully
                    """)
    
    with tab2:
        st.markdown('<div class="sub-header">Health Dashboard</div>', unsafe_allow_html=True)
        
        # Create a clean copy of the data for visualization
        X_clean = X.copy()
        
        # Filter out invalid values
        X_clean = X_clean[X_clean['age'] > 0]  # Remove negative/zero ages
        X_clean = X_clean[X_clean['age'] <= 120]  # Remove unrealistic ages
        X_clean = X_clean[X_clean['chol'] > 0]  # Remove negative cholesterol
        
        # Interactive health metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            if not X_clean.empty:
                fig = px.histogram(
                    X_clean, x='age', title='Age Distribution in Dataset',
                    labels={'age': 'Age'}, nbins=20
                )
                fig.update_traces(marker_line_color='black', marker_line_width=1)
                fig.update_xaxes(range=[0, 100])  # Set reasonable age range
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid age data available for visualization")
        
        with col2:
            if not X_clean.empty:
                fig = px.box(
                    X_clean, y='chol', title='Cholesterol Distribution',
                    labels={'chol': 'Cholesterol (mg/dl)'}
                )
                fig.update_traces(marker_line_color='black', marker_line_width=1)
                # Set reasonable cholesterol range (most adults are between 120-300)
                fig.update_yaxes(range=[100, 400])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid cholesterol data available for visualization")
        
        # Add data quality information
        st.markdown("---")
        st.caption(f"Visualizing {len(X_clean)} of {len(X)} records after filtering invalid values")
        
        # Health score calculator
        st.markdown('<div class="sub-header">Heart Health Score Calculator</div>', unsafe_allow_html=True)
        st.write("Based on your inputs, here's your heart health score:")
        
        # Simplified scoring (this is just for demonstration)
        health_score = 100
        if user_input['chol'] > 240: health_score -= 15
        if user_input['trestbps'] > 140: health_score -= 15
        if user_input['thalach'] < 120: health_score -= 10
        if user_input['oldpeak'] > 1.0: health_score -= 10
        if user_input['exang'] == 1: health_score -= 10
        if user_input['fbs'] == 1: health_score -= 5
        
        st.metric("Your Heart Health Score", f"{health_score}/100")
        st.progress(health_score/100)
    
    with tab3:
        st.markdown('<div class="sub-header">Model Performance Insights</div>', unsafe_allow_html=True)
        
        # Show model accuracy and metrics
        accuracy = np.mean(y_test == y_pred)
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy*100:.1f}%")
        col2.metric("Precision", f"{np.mean(y_pred[y_test==1] == 1)*100:.1f}%")
        col3.metric("Recall", f"{np.mean(y_test[y_pred==1] == 1)*100:.1f}%")
        
        # Feature importance plot
        feature_names = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])]
        importance_fig = plot_feature_importance(model, feature_names)
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Low Risk', 'High Risk'], y=['Low Risk', 'High Risk'])
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
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

# Main
def main():
    # Try to find the data file
    data_path = get_file_path("processed.cleveland.data")
    
    if data_path is None:
        st.error("""
        **Data file not found!**
        
        Please make sure the data file is available in one of these locations:
        - In the same directory as this script: `processed.cleveland.data`
        - In a 'data' subdirectory: `data/processed.cleveland.data`
        - Or upload it using the file uploader below.
        """)
        
        # Add file uploader as fallback
        uploaded_file = st.file_uploader("Upload your heart disease data file", type=["data", "csv", "txt"])
        
        if uploaded_file is not None:
            try:
                # For .data files (like Cleveland dataset)
                if uploaded_file.name.endswith('.data'):
                    df = pd.read_csv(uploaded_file, header=None, na_values='?')
                else:
                    # For CSV files
                    df = pd.read_csv(uploaded_file)
                
                # Process your data
                X, model, scaler, y_test, y_pred, oh_columns = train(df)
                app(model, scaler, oh_columns, X, y_test, y_pred)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.stop()
    else:
        try:
            X, model, scaler, y_test, y_pred, oh_columns = train(data_path)
            app(model, scaler, oh_columns, X, y_test, y_pred)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()