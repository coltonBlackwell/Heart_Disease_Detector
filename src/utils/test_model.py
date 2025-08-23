import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

def test(X, model, scaler, y_test, y_pred):
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Bagged Logistic Regression Accuracy: {acc:.4f}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Predicted vs Actual distribution
    plt.figure(figsize=(8,5))
    plt.hist([y_test, y_pred], label=['Actual', 'Predicted'], bins=np.arange(-0.5, 5, 1), alpha=0.7, rwidth=0.8)
    plt.xticks(range(5))
    plt.xlabel("Heart Disease Class (0-4)")
    plt.ylabel("Count")
    plt.title("Actual vs Predicted Distribution")
    plt.legend()
    plt.show()
