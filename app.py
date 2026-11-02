
# ==========================================
# NPHA DOCTOR VISITS CLASSIFICATION APP
# ==========================================

import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# 1. Page configuration
st.set_page_config(
    page_title="NPHA Classification",
    page_icon="🏥",
    layout="wide"
)

st.title("NPHA Doctor Visits Classification")

st.write(
    "Prediction of Doctor Visit Categories "
    "using Five Machine Learning Models"
)

# 2. Available saved models
model_files = {
    "Logistic Regression (Balanced)": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

# 3. Select model
selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(model_files.keys())
)

# 4. Upload test CSV
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    # Read uploaded CSV
    test_df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(test_df.head())

    # Check target
    target = "Number of Doctors Visited"

    if target not in test_df.columns:
        st.error("Target column missing!")
        st.stop()

    # Separate X and y
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Check labels
    if not y_test.isin([1, 2, 3]).all():
        st.error("Target must contain classes 1, 2, or 3.")
        st.stop()

    # 5. Load saved model
    model_path = (
        Path(__file__).resolve().parent
        / "model"
        / model_files[selected_model]
    )

    if not model_path.exists():
        st.error("Saved model file not found!")
        st.stop()

    model = joblib.load(model_path)

    # 6. Predict without retraining
    try:
        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)

    except (ValueError, KeyError) as error:
        st.error(f"Prediction error: {error}")
        st.stop()

    # 7. Calculate six evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    if y_test.nunique() == 3:
        auc = roc_auc_score(
            y_test,
            probabilities,
            labels=model.classes_,
            multi_class="ovr",
            average="macro"
        )
    else:
        auc = None

    precision = precision_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, predictions)

    # 8. Display metrics
    st.subheader("Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric(
        "AUC",
        f"{auc:.4f}" if auc is not None else "N/A"
    )
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # 9. Confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[1, 2, 3]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 1", "Actual 2", "Actual 3"],
        columns=[
            "Predicted 1",
            "Predicted 2",
            "Predicted 3"
        ]
    )

    st.dataframe(cm_df)

    # 10. Classification report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        labels=[1, 2, 3],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).T)

    st.success(
        f"Evaluation completed using {selected_model}"
    )
