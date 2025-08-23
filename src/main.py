import streamlit as st
import pandas as pd
from utils.train_model import train
from utils.test_model import test

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
    return prediction

# Streamlit app
def app(model, scaler, oh_columns):
    st.title("❤️ Heart Disease Risk Predictor")
    st.write("Enter your health parameters:")

    # Input fields
    age = st.number_input("Age", 18, 100, 50)
    sex = st.selectbox("Sex (0=female,1=male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1=Normal,2=Fixed,3=Reversible)", [1, 2, 3])

    user_input = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    if st.button("Predict Risk"):
        pred = predict_heart_disease(model, scaler, user_input, oh_columns)
        if pred == 0:
            st.success("✅ Low risk of heart disease")
        else:
            st.error("⚠️ High risk of heart disease")

# Main
def main():
    X, model, scaler, y_test, y_pred, oh_columns = train("../data/processed.cleveland.data")
    test(X, model, scaler, y_test, y_pred)
    app(model, scaler, oh_columns)

if __name__ == "__main__":
    main()
