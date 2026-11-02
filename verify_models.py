
# ==========================================
# STEP 19: VERIFY STREAMLIT MODEL FILES
# ==========================================

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score
)

# Load same CSV used in Streamlit
df = pd.read_csv("test_data.csv")

target = "Number of Doctors Visited"

X = df.drop(columns=[target])
y = df[target]

models = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

print("\nSTEP 19: SAVED MODEL VERIFICATION")

for name, filename in models.items():

    model = joblib.load("model/" + filename)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    accuracy = accuracy_score(y, predictions)

    auc = roc_auc_score(
        y, probabilities,
        multi_class="ovr",
        average="macro"
    )

    f1 = f1_score(
        y, predictions,
        average="macro",
        zero_division=0
    )

    print(f"\nModel: {name}")
    print("Accuracy:", round(accuracy, 4))
    print("AUC     :", round(auc, 4))
    print("F1      :", round(f1, 4))

print("\nVERIFICATION COMPLETED")
