
# NPHA Doctor Visits Classification

## 1. Problem Statement

The objective of this project is to predict the
number of doctors visited by older adults using
the National Poll on Healthy Aging (NPHA) dataset.

Five machine learning classification algorithms
are implemented and compared.

## 2. Dataset Description

Dataset: National Poll on Healthy Aging (NPHA)

Source: UCI Machine Learning Repository

Dataset URL:
https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha)

Total Instances: 714

Input Features: 14

Target Variable: Number of Doctors Visited

Classification Type: Multiclass (3 classes)

Class 1: 0-1 doctors visited
Class 2: 2-3 doctors visited
Class 3: 4 or more doctors visited

Training Records: 571
Testing Records: 143

## 3. GitHub Repository

Repository Link: [Add your GitHub link after uploading]

## 4. Models Implemented

1. Logistic Regression (Balanced)
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

All models were trained using the same training
dataset and evaluated on the same test dataset.

## 5. Model Evaluation

The following evaluation metrics were used:

- Accuracy
- AUC (Macro One-vs-Rest)
- Precision (Macro)
- Recall (Macro)
- F1 Score (Macro)
- Matthews Correlation Coefficient (MCC)

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression (Balanced) | 0.4336 | 0.6294 | 0.4549 | 0.4613 | 0.4260 | 0.1857 |
| Decision Tree | 0.5245 | 0.5338 | 0.3550 | 0.3613 | 0.3158 | 0.0808 |
| KNN | 0.4755 | 0.5107 | 0.3947 | 0.3720 | 0.3645 | 0.0604 |
| Naive Bayes | 0.2517 | 0.5745 | 0.2345 | 0.4029 | 0.2323 | 0.1162 |
| SVM | 0.5245 | 0.6086 | 0.1748 | 0.3333 | 0.2294 | 0.0000 |

## 6. Model Performance Observations

### Logistic Regression

Balanced Logistic Regression achieved the highest
macro F1 score (0.4260), indicating the strongest
overall balance across the three classes among
the tested configurations.

### Decision Tree

Decision Tree achieved 52.45% accuracy but had
a relatively low macro F1 score of 0.3158.

### KNN

KNN achieved 47.55% accuracy and a macro F1
score of 0.3645.

### Naive Bayes

Gaussian Naive Bayes achieved the lowest accuracy
(25.17%) but a higher macro recall (0.4029)
than most of the other models.

### Support Vector Machine

SVM achieved 52.45% accuracy but predicted
only the majority class, resulting in an
MCC score of 0.0000.

## 7. Overall Winning Model

Balanced Logistic Regression was selected
based on the highest macro F1 score of 0.4260.

Although Decision Tree and SVM achieved higher
accuracy, Balanced Logistic Regression identified
minority classes more effectively.

The overall predictive performance remains limited.

## 8. Streamlit Application

The deployed application allows users to:

- Upload a test CSV file
- Select one of five trained ML models
- Display six evaluation metrics
- View the confusion matrix
- View the classification report

The application loads previously saved models
without retraining on uploaded test data.

Live App Link: [Add after deployment]

# NPHA Doctor Visits Classification

## 1. Problem Statement

The objective of this project is to predict the
number of doctors visited by older adults using
the National Poll on Healthy Aging (NPHA) dataset.

Five machine learning classification algorithms
are implemented and compared.

## 2. Dataset Description

Dataset: National Poll on Healthy Aging (NPHA)

Source: UCI Machine Learning Repository

Dataset URL:
https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha)

Total Instances: 714

Input Features: 14

Target Variable: Number of Doctors Visited

Classification Type: Multiclass (3 classes)

Class 1: 0-1 doctors visited
Class 2: 2-3 doctors visited
Class 3: 4 or more doctors visited

Training Records: 571
Testing Records: 143

## 3. GitHub Repository

Repository Link: [Add your GitHub link after uploading]

## 4. Models Implemented

1. Logistic Regression (Balanced)
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

All models were trained using the same training
dataset and evaluated on the same test dataset.

## 5. Model Evaluation

The following evaluation metrics were used:

- Accuracy
- AUC (Macro One-vs-Rest)
- Precision (Macro)
- Recall (Macro)
- F1 Score (Macro)
- Matthews Correlation Coefficient (MCC)

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression (Balanced) | 0.4336 | 0.6294 | 0.4549 | 0.4613 | 0.4260 | 0.1857 |
| Decision Tree | 0.5245 | 0.5338 | 0.3550 | 0.3613 | 0.3158 | 0.0808 |
| KNN | 0.4755 | 0.5107 | 0.3947 | 0.3720 | 0.3645 | 0.0604 |
| Naive Bayes | 0.2517 | 0.5745 | 0.2345 | 0.4029 | 0.2323 | 0.1162 |
| SVM | 0.5245 | 0.6086 | 0.1748 | 0.3333 | 0.2294 | 0.0000 |

## 6. Model Performance Observations

### Logistic Regression

Balanced Logistic Regression achieved the highest
macro F1 score (0.4260), indicating the strongest
overall balance across the three classes among
the tested configurations.

### Decision Tree

Decision Tree achieved 52.45% accuracy but had
a relatively low macro F1 score of 0.3158.

### KNN

KNN achieved 47.55% accuracy and a macro F1
score of 0.3645.

### Naive Bayes

Gaussian Naive Bayes achieved the lowest accuracy
(25.17%) but a higher macro recall (0.4029)
than most of the other models.

### Support Vector Machine

SVM achieved 52.45% accuracy but predicted
only the majority class, resulting in an
MCC score of 0.0000.

## 7. Overall Winning Model

Balanced Logistic Regression was selected
based on the highest macro F1 score of 0.4260.

Although Decision Tree and SVM achieved higher
accuracy, Balanced Logistic Regression identified
minority classes more effectively.

The overall predictive performance remains limited.

## 8. Streamlit Application

The deployed application allows users to:

- Upload a test CSV file
- Select one of five trained ML models
- Display six evaluation metrics
- View the confusion matrix
- View the classification report

The application loads previously saved models
without retraining on uploaded test data.

Live App Link: [Add after deployment]
