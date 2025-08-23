import streamlit as st
import numpy as np

# Function for prediction using the trained model
def predict_heart_disease(model, scaler, user_input):
    # Convert input to array
    input_array = np.array(user_input).reshape(1, -1)
    # Scale input same as training
    input_scaled = scaler.transform(input_array)
    # Predict
    prediction = model.predict(input_scaled)[0]
    return prediction


def app(model, scaler):
    st.title("❤️ Heart Disease Risk Predictor")

    st.write("Enter your health parameters below:")

    # Example features (Cleveland dataset has 13 features)
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)", [1, 2, 3])

    # Collect inputs
    user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]

    if st.button("Predict Risk"):
        prediction = predict_heart_disease(model, scaler, user_input)
        if prediction == 0:
            st.success("✅ Low risk of heart disease")
        else:
            st.error(f"⚠️ High risk: Class {prediction} (disease severity)")