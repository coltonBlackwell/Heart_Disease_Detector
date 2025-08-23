import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

## Use a Logistic Regression Model

## use data/processed.cleveland.data

def train(input_csv):

    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    df = pd.read_csv(input_csv, header=None, names=column_names)


    # Replace "?" with NaN
    df.replace('?', np.nan, inplace=True)

    # Convert all columns to numeric (necessary after replacing '?')
    df = df.apply(pd.to_numeric)

    # Fill missing values with median
    df.fillna(df.median(), inplace=True)


    # Features and target
    X = df.drop('num', axis=1)
    y = df['num']

    # One-hot encode categorical features
    categorical_cols = ['cp', 'slope', 'ca', 'thal', 'restecg']
    X = pd.get_dummies(X, columns=categorical_cols)

    # Scale numeric features
    numeric_cols = [col for col in X.columns if col not in pd.get_dummies(df[categorical_cols]).columns]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) ## , random_state=42

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    base_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs') # class_weight='balanced'
    model = BaggingClassifier(
        base_model,
        n_estimators=150,
        max_samples=0.6,
        max_features=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Bagged Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

    return X, model, scaler, y_test, y_pred

# # Example usage
# model, scaler = train("data/processed.cleveland.data")