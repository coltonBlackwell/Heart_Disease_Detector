import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

def train(input_csv):
    # Column names
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    df = pd.read_csv(input_csv, header=None, names=column_names)

    # Replace missing values
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df.fillna(df.median(), inplace=True)

    # Convert target to binary
    df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

    # Features / target
    X = df.drop('num', axis=1)
    y = df['num']

    # One-hot encode categorical
    categorical_cols = ['cp', 'slope', 'ca', 'thal', 'restecg']
    X = pd.get_dummies(X, columns=categorical_cols)

    # Save column order for app
    oh_columns = X.columns

    # Scale numeric features
    numeric_cols = [col for col in X.columns if col not in pd.get_dummies(df[categorical_cols]).columns]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Bagged Logistic Regression
    base_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    model = BaggingClassifier(
        base_model,
        n_estimators=150,
        max_samples=0.6,
        max_features=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return X, model, scaler, y_test, y_pred, oh_columns