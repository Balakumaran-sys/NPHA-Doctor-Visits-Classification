
import pandas as pd

df = pd.read_csv("NPHA-doctor-visits.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

# ==========================================
# STEP 2: DATASET QUALITY CHECK
# ==========================================

# 1. Check missing values
print("\n1. Missing Values:")
print(df.isnull().sum())

# 2. Check duplicate records
print("\n2. Duplicate Records:")
print(df.duplicated().sum())

# 3. Check target class distribution
print("\n3. Target Class Distribution:")
print(df["Number of Doctors Visited"].value_counts().sort_index())

# 4. Check data types
print("\n4. Data Types:")
print(df.dtypes)

# ==========================================
# STEP 3: SEPARATE FEATURES AND TARGET
# ==========================================

# Define the target column
target_column = "Number of Doctors Visited"

# X contains all input features
X = df.drop(columns=[target_column])

# y contains the target values
y = df[target_column]

# Display shapes
print("\nSTEP 3: FEATURES AND TARGET")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# Display target categories
print("Target Classes:", sorted(y.unique()))

# Display input feature names
print("\nInput Features:")
print(X.columns.tolist())

# ==========================================
# STEP 4: TRAIN-TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nSTEP 4: TRAIN-TEST SPLIT")

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting Target Distribution:")
print(y_test.value_counts().sort_index())

# ==========================================
# STEP 5: DATA PREPROCESSING
# ==========================================

from sklearn.preprocessing import OneHotEncoder

# Create the One-Hot Encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Learn category values from training data
X_train_encoded = encoder.fit_transform(X_train)

# Apply the same encoding to testing data
X_test_encoded = encoder.transform(X_test)

print("\nSTEP 5: DATA PREPROCESSING")

print("Original Training Shape:", X_train.shape)
print("Encoded Training Shape:", X_train_encoded.shape)

print("Original Testing Shape:", X_test.shape)
print("Encoded Testing Shape:", X_test_encoded.shape)

print("\nFirst 10 Encoded Feature Names:")
print(encoder.get_feature_names_out()[:10])

# ==========================================
# STEP 6: LOGISTIC REGRESSION MODEL
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Create Logistic Regression model
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# 2. Train model using training data
lr_model.fit(X_train_encoded, y_train)

# 3. Predict classes on testing data
lr_predictions = lr_model.predict(X_test_encoded)

# 4. Calculate accuracy
lr_accuracy = accuracy_score(y_test, lr_predictions)

# 5. Display output
print("\nSTEP 6: LOGISTIC REGRESSION")

print("Model Training: Completed")

print("Actual Values (First 10):")
print(y_test.values[:10])

print("Predicted Values (First 10):")
print(lr_predictions[:10])

print("Logistic Regression Accuracy:",
      round(lr_accuracy, 4))

# ==========================================
# STEP 7: LOGISTIC REGRESSION EVALUATION
# ==========================================

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# 1. Predict class probabilities
lr_probabilities = lr_model.predict_proba(X_test_encoded)

# 2. Calculate all six metrics
lr_auc = roc_auc_score(
    y_test,
    lr_probabilities,
    multi_class="ovr",
    average="macro"
)

lr_precision = precision_score(
    y_test, lr_predictions,
    average="macro", zero_division=0
)

lr_recall = recall_score(
    y_test, lr_predictions,
    average="macro", zero_division=0
)

lr_f1 = f1_score(
    y_test, lr_predictions,
    average="macro", zero_division=0
)

lr_mcc = matthews_corrcoef(
    y_test, lr_predictions
)

# 3. Display all metrics
print("\nSTEP 7: LOGISTIC REGRESSION METRICS")

print("Accuracy :", round(lr_accuracy, 4))
print("AUC      :", round(lr_auc, 4))
print("Precision:", round(lr_precision, 4))
print("Recall   :", round(lr_recall, 4))
print("F1 Score :", round(lr_f1, 4))
print("MCC      :", round(lr_mcc, 4))

# 4. Check predicted class distribution
print("\nPredicted Class Distribution:")
print(pd.Series(lr_predictions).value_counts().sort_index())

# 5. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, lr_predictions, labels=[1, 2, 3]
))

# ==========================================
# STEP 8: DECISION TREE CLASSIFIER
# ==========================================

from sklearn.tree import DecisionTreeClassifier

# 1. Create Decision Tree model
dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# 2. Train the model
dt_model.fit(X_train_encoded, y_train)

# 3. Predict test classes
dt_predictions = dt_model.predict(X_test_encoded)

# 4. Predict class probabilities for AUC
dt_probabilities = dt_model.predict_proba(X_test_encoded)

# 5. Calculate evaluation metrics
dt_accuracy = accuracy_score(y_test, dt_predictions)

dt_auc = roc_auc_score(
    y_test, dt_probabilities,
    multi_class="ovr", average="macro"
)

dt_precision = precision_score(
    y_test, dt_predictions,
    average="macro", zero_division=0
)

dt_recall = recall_score(
    y_test, dt_predictions,
    average="macro", zero_division=0
)

dt_f1 = f1_score(
    y_test, dt_predictions,
    average="macro", zero_division=0
)

dt_mcc = matthews_corrcoef(
    y_test, dt_predictions
)

# 6. Display metrics
print("\nSTEP 8: DECISION TREE METRICS")

print("Accuracy :", round(dt_accuracy, 4))
print("AUC      :", round(dt_auc, 4))
print("Precision:", round(dt_precision, 4))
print("Recall   :", round(dt_recall, 4))
print("F1 Score :", round(dt_f1, 4))
print("MCC      :", round(dt_mcc, 4))

# 7. Display predicted classes
print("\nPredicted Class Distribution:")
print(pd.Series(dt_predictions).value_counts().sort_index())

# 8. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, dt_predictions, labels=[1, 2, 3]
))

# ==========================================
# STEP 9: K-NEAREST NEIGHBORS (KNN)
# ==========================================

from sklearn.neighbors import KNeighborsClassifier

# 1. Create KNN model using 5 neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)

# 2. Train the model
knn_model.fit(X_train_encoded, y_train)

# 3. Predict test classes
knn_predictions = knn_model.predict(X_test_encoded)

# 4. Predict class probabilities
knn_probabilities = knn_model.predict_proba(X_test_encoded)

# 5. Calculate evaluation metrics
knn_accuracy = accuracy_score(y_test, knn_predictions)

knn_auc = roc_auc_score(
    y_test, knn_probabilities,
    multi_class="ovr", average="macro"
)

knn_precision = precision_score(
    y_test, knn_predictions,
    average="macro", zero_division=0
)

knn_recall = recall_score(
    y_test, knn_predictions,
    average="macro", zero_division=0
)

knn_f1 = f1_score(
    y_test, knn_predictions,
    average="macro", zero_division=0
)

knn_mcc = matthews_corrcoef(
    y_test, knn_predictions
)

# 6. Display evaluation metrics
print("\nSTEP 9: KNN METRICS")

print("Accuracy :", round(knn_accuracy, 4))
print("AUC      :", round(knn_auc, 4))
print("Precision:", round(knn_precision, 4))
print("Recall   :", round(knn_recall, 4))
print("F1 Score :", round(knn_f1, 4))
print("MCC      :", round(knn_mcc, 4))

# 7. Display predicted class distribution
print("\nPredicted Class Distribution:")
print(pd.Series(knn_predictions).value_counts().sort_index())

# 8. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, knn_predictions, labels=[1, 2, 3]
))

# ==========================================
# STEP 10: GAUSSIAN NAIVE BAYES
# ==========================================

from sklearn.naive_bayes import GaussianNB

# 1. Create Gaussian Naive Bayes model
nb_model = GaussianNB()

# 2. Train the model
nb_model.fit(X_train_encoded, y_train)

# 3. Predict test classes
nb_predictions = nb_model.predict(X_test_encoded)

# 4. Predict class probabilities
nb_probabilities = nb_model.predict_proba(X_test_encoded)

# 5. Calculate evaluation metrics
nb_accuracy = accuracy_score(y_test, nb_predictions)

nb_auc = roc_auc_score(
    y_test, nb_probabilities,
    multi_class="ovr", average="macro"
)

nb_precision = precision_score(
    y_test, nb_predictions,
    average="macro", zero_division=0
)

nb_recall = recall_score(
    y_test, nb_predictions,
    average="macro", zero_division=0
)

nb_f1 = f1_score(
    y_test, nb_predictions,
    average="macro", zero_division=0
)

nb_mcc = matthews_corrcoef(
    y_test, nb_predictions
)

# 6. Display evaluation metrics
print("\nSTEP 10: NAIVE BAYES METRICS")

print("Accuracy :", round(nb_accuracy, 4))
print("AUC      :", round(nb_auc, 4))
print("Precision:", round(nb_precision, 4))
print("Recall   :", round(nb_recall, 4))
print("F1 Score :", round(nb_f1, 4))
print("MCC      :", round(nb_mcc, 4))

# 7. Display predicted class distribution
print("\nPredicted Class Distribution:")
print(pd.Series(nb_predictions).value_counts().sort_index())

# 8. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, nb_predictions, labels=[1, 2, 3]
))

# ==========================================
# STEP 11: SUPPORT VECTOR MACHINE (SVM)
# ==========================================

from sklearn.svm import SVC

# 1. Create SVM classifier
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,
    random_state=42
)

# 2. Train SVM model
svm_model.fit(X_train_encoded, y_train)

# 3. Predict test classes
svm_predictions = svm_model.predict(X_test_encoded)

# 4. Predict class probabilities
svm_probabilities = svm_model.predict_proba(X_test_encoded)

# 5. Calculate six evaluation metrics
svm_accuracy = accuracy_score(
    y_test, svm_predictions
)

svm_auc = roc_auc_score(
    y_test,
    svm_probabilities,
    multi_class="ovr",
    average="macro"
)

svm_precision = precision_score(
    y_test, svm_predictions,
    average="macro",
    zero_division=0
)

svm_recall = recall_score(
    y_test, svm_predictions,
    average="macro",
    zero_division=0
)

svm_f1 = f1_score(
    y_test, svm_predictions,
    average="macro",
    zero_division=0
)

svm_mcc = matthews_corrcoef(
    y_test, svm_predictions
)

# 6. Print evaluation metrics
print("\nSTEP 11: SVM METRICS")

print("Accuracy :", round(svm_accuracy, 4))
print("AUC      :", round(svm_auc, 4))
print("Precision:", round(svm_precision, 4))
print("Recall   :", round(svm_recall, 4))
print("F1 Score :", round(svm_f1, 4))
print("MCC      :", round(svm_mcc, 4))

# 7. Predicted class distribution
print("\nPredicted Class Distribution:")
print(
    pd.Series(svm_predictions)
    .value_counts()
    .sort_index()
)

# 8. Confusion matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        svm_predictions,
        labels=[1, 2, 3]
    )
)

# ==========================================
# STEP 12: COMPARE ALL FIVE ML MODELS
# ==========================================

# Create comparison table from actual results
comparison_data = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ],
    "Accuracy": [
        lr_accuracy, dt_accuracy, knn_accuracy,
        nb_accuracy, svm_accuracy
    ],
    "AUC": [
        lr_auc, dt_auc, knn_auc,
        nb_auc, svm_auc
    ],
    "Precision": [
        lr_precision, dt_precision, knn_precision,
        nb_precision, svm_precision
    ],
    "Recall": [
        lr_recall, dt_recall, knn_recall,
        nb_recall, svm_recall
    ],
    "F1": [
        lr_f1, dt_f1, knn_f1,
        nb_f1, svm_f1
    ],
    "MCC": [
        lr_mcc, dt_mcc, knn_mcc,
        nb_mcc, svm_mcc
    ]
}

# Convert results to DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Display all models
print("\nSTEP 12: MODEL COMPARISON")
print(comparison_df.round(4).to_string(index=False))

# Save results to CSV
comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")
print("File: model_comparison.csv")
class_weight="balanced"

# ==========================================
# STEP 13: CLASS BALANCING EXPERIMENT
# ==========================================

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# 1. Define balanced Logistic Regression
balanced_lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

# 2. Define 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 3. Evaluate original Logistic Regression
original_cv_scores = cross_val_score(
    LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    X_train_encoded,
    y_train,
    cv=cv,
    scoring="f1_macro"
)

# 4. Evaluate balanced Logistic Regression
balanced_cv_scores = cross_val_score(
    balanced_lr_model,
    X_train_encoded,
    y_train,
    cv=cv,
    scoring="f1_macro"
)

# 5. Display results
print("\nSTEP 13: CLASS BALANCING EXPERIMENT")

print("Original Logistic Regression CV F1 Scores:")
print(original_cv_scores.round(4))

print("Original Mean CV F1:",
      round(original_cv_scores.mean(), 4))

print("\nBalanced Logistic Regression CV F1 Scores:")
print(balanced_cv_scores.round(4))

print("Balanced Mean CV F1:",
      round(balanced_cv_scores.mean(), 4))

print("\nDifference in Mean CV F1:",
      round(
          balanced_cv_scores.mean()
          - original_cv_scores.mean(),
          4
      ))

# ==========================================
# STEP 14: PIPELINE CROSS-VALIDATION
# ==========================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

# 1. Original Logistic Regression pipeline
original_pipeline = Pipeline([
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# 2. Balanced Logistic Regression pipeline
balanced_pipeline = Pipeline([
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])

# 3. Cross-validation on ORIGINAL training features
original_scores = cross_val_score(
    original_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro"
)

balanced_scores = cross_val_score(
    balanced_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro"
)

# 4. Display results
print("\nSTEP 14: PIPELINE CROSS-VALIDATION")

print("Original Pipeline F1 Scores:")
print(original_scores.round(4))

print("Original Mean F1:",
      round(original_scores.mean(), 4))

print("\nBalanced Pipeline F1 Scores:")
print(balanced_scores.round(4))

print("Balanced Mean F1:",
      round(balanced_scores.mean(), 4))

# 5. Select preferred configuration using CV only
if balanced_scores.mean() > original_scores.mean():
    selected_lr_pipeline = balanced_pipeline
    selected_name = "Balanced Logistic Regression"
else:
    selected_lr_pipeline = original_pipeline
    selected_name = "Original Logistic Regression"

print("\nSelected Configuration:", selected_name)

# ==========================================
# STEP 15: SAVE BALANCED LOGISTIC REGRESSION
# ==========================================

import joblib
import os

# 1. Create model folder
os.makedirs("model", exist_ok=True)

# 2. Train selected pipeline
selected_lr_pipeline.fit(X_train, y_train)

# 3. Save trained pipeline
joblib.dump(
    selected_lr_pipeline,
    "model/logistic_regression.joblib"
)

print("\nSTEP 15: SAVE LOGISTIC REGRESSION")

print("Selected Model:", selected_name)
print("Training: Completed")
print("Model Saved: model/logistic_regression.joblib")

# 4. Predict on untouched test dataset
final_lr_predictions = selected_lr_pipeline.predict(X_test)

final_lr_probabilities = (
    selected_lr_pipeline.predict_proba(X_test)
)

# 5. Calculate final metrics
final_lr_accuracy = accuracy_score(
    y_test, final_lr_predictions
)

final_lr_auc = roc_auc_score(
    y_test,
    final_lr_probabilities,
    multi_class="ovr",
    average="macro"
)

final_lr_precision = precision_score(
    y_test, final_lr_predictions,
    average="macro",
    zero_division=0
)

final_lr_recall = recall_score(
    y_test, final_lr_predictions,
    average="macro",
    zero_division=0
)

final_lr_f1 = f1_score(
    y_test, final_lr_predictions,
    average="macro",
    zero_division=0
)

final_lr_mcc = matthews_corrcoef(
    y_test, final_lr_predictions
)

print("\nFINAL LOGISTIC REGRESSION METRICS")

print("Accuracy :", round(final_lr_accuracy, 4))
print("AUC      :", round(final_lr_auc, 4))
print("Precision:", round(final_lr_precision, 4))
print("Recall   :", round(final_lr_recall, 4))
print("F1 Score :", round(final_lr_f1, 4))
print("MCC      :", round(final_lr_mcc, 4))

print("\nPredicted Class Distribution:")
print(
    pd.Series(final_lr_predictions)
    .value_counts()
    .sort_index()
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        final_lr_predictions,
        labels=[1, 2, 3]
    )
)

# ==========================================
# STEP 16: SAVE REMAINING FOUR MODEL PIPELINES
# ==========================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Create model directory if necessary
os.makedirs("model", exist_ok=True)

# Four models from our previous experiments
remaining_models = {
    "decision_tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "knn": KNeighborsClassifier(
        n_neighbors=5
    ),

    "naive_bayes": GaussianNB(),

    "svm": SVC(
        kernel="rbf",
        C=1.0,
        probability=True,
        random_state=42
    )
}

print("\nSTEP 16: SAVING REMAINING MODELS")

for model_name, classifier in remaining_models.items():

    # Combine encoder and classifier
    pipeline = Pipeline([
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )),
        ("classifier", classifier)
    ])

    # Train using original training features
    pipeline.fit(X_train, y_train)

    # Save entire trained pipeline
    file_path = f"model/{model_name}.joblib"
    joblib.dump(pipeline, file_path)

    print(f"Saved: {file_path}")

# Verify all five model files
print("\nALL SAVED MODELS:")

for file_name in sorted(os.listdir("model")):
    if file_name.endswith(".joblib"):
        print(file_name)

print("\nSTEP 16 COMPLETED")

# ==========================================
# STEP 17: CREATE TEST DATA CSV
# ==========================================

# 1. Copy original test features
test_data = X_test.copy()

# 2. Add actual target values
test_data["Number of Doctors Visited"] = y_test

# 3. Save test dataset
test_data.to_csv(
    "test_data.csv",
    index=False
)

# 4. Verify saved CSV
saved_test_data = pd.read_csv("test_data.csv")

print("\nSTEP 17: TEST DATA CREATION")

print("Test CSV Created Successfully!")

print("Test Data Shape:", saved_test_data.shape)

print("\nFirst 5 Test Records:")
print(saved_test_data.head())

print("\nTarget Class Distribution:")
print(
    saved_test_data["Number of Doctors Visited"]
    .value_counts()
    .sort_index()
)

print("\nFile Name: test_data.csv")


# ==========================================
# STEP 20: FINAL MODEL COMPARISON
# ==========================================

final_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression (Balanced)",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ],

    "Accuracy": [
        final_lr_accuracy,
        dt_accuracy,
        knn_accuracy,
        nb_accuracy,
        svm_accuracy
    ],

    "AUC": [
        final_lr_auc,
        dt_auc,
        knn_auc,
        nb_auc,
        svm_auc
    ],

    "Precision": [
        final_lr_precision,
        dt_precision,
        knn_precision,
        nb_precision,
        svm_precision
    ],

    "Recall": [
        final_lr_recall,
        dt_recall,
        knn_recall,
        nb_recall,
        svm_recall
    ],

    "F1": [
        final_lr_f1,
        dt_f1,
        knn_f1,
        nb_f1,
        svm_f1
    ],

    "MCC": [
        final_lr_mcc,
        dt_mcc,
        knn_mcc,
        nb_mcc,
        svm_mcc
    ]
})

print("\nSTEP 20: FINAL MODEL COMPARISON")

print(final_comparison.round(4).to_string(index=False))

# Save comparison table
final_comparison.to_csv(
    "final_model_comparison.csv",
    index=False
)

# Select the highest macro F1 model
best_model = final_comparison.loc[
    final_comparison["F1"].idxmax()
]

print("\nBEST MODEL BASED ON MACRO F1:")
print(best_model["Model"])

print("Best F1 Score:", round(best_model["F1"], 4))

print("\nSaved: final_model_comparison.csv")


