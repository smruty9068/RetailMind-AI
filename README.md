# RetailMind AI
### AI-Powered Customer Churn Prediction System

RetailMind AI is a machine learning project designed to
predict customer churn using customer demographics,
subscription information, service usage, and billing data.

The project follows an incremental development approach,
implementing different machine learning techniques,
feature engineering methods, ensemble models, and
artificial neural networks.

---

## Project Objectives

- Predict whether a customer is likely to churn.
- Identify important customer churn indicators.
- Apply data preprocessing and feature engineering.
- Experiment with multiple machine learning algorithms.
- Compare model performance using evaluation metrics.
- Develop a reusable customer churn prediction pipeline.

---

## Dataset

**Dataset:** Telco Customer Churn

**Source:** IBM Telco Customer Churn Dataset

**Records:** 7,043 customers

**Features:** 20 original columns after removing customerID,
including the target variable.

**Target Variable:** Churn

- 0 = Customer stays
- 1 = Customer leaves

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

# Incremental Development

## Day 1 – Data Preparation and Baseline Modeling

- Dataset exploration
- Data cleaning
- Missing value handling
- Categorical and numerical feature identification
- Initial preprocessing
- Baseline model development

## Day 2 – Model Comparison

- Multiple classification algorithms
- Model training and evaluation
- Performance comparison
- Accuracy, Precision, Recall and ROC-AUC
- Champion model identification

Notebook: `Day2_Model_Comparison.ipynb`

## Day 3 – Model Tuning

- Hyperparameter tuning
- Model optimization
- Performance evaluation
- Champion model selection

### Day 3 Champion Results

| Metric | Score |
|---|---:|
| Accuracy | 80.12% |
| Precision | 64.59% |
| Recall | 55.46% |
| ROC-AUC | 82.96% |

Notebook: `Day3_Model_Tuning.ipynb`

## Day 4 – Feature Engineering and Ensemble Learning

### Feature Engineering

- Tenure-based features
- Charge-based features
- Contract-based binary features
- Service-based features
- Customer service usage count

### Feature Selection

- SelectKBest
- ANOVA F-test
- Selection of top 20 features

### Ensemble Models

1. Random Forest (Bagging)
2. Histogram Gradient Boosting

### Random Forest Results

| Metric | Score |
|---|---:|
| Accuracy | 79.99% |
| Precision | 65.13% |
| Recall | 52.94% |
| F1 Score | 58.41% |
| ROC-AUC | 83.51% |

### Histogram Gradient Boosting Results

| Metric | Score |
|---|---:|
| Accuracy | 79.84% |
| Precision | 65.41% |
| Recall | 51.07% |
| F1 Score | 57.36% |
| ROC-AUC | 83.57% |

### Key Observation

Histogram Gradient Boosting achieved the highest ROC-AUC
among the two Day 4 models, while Random Forest achieved
better recall and F1-score.

Note: These reported Day 4 metrics were obtained from
the initial model training run. The final pipelines were
also retrained with the engineered features.

Notebook: `Day4_Feature_Engineering_Ensemble.ipynb`

## Day 5 – Artificial Neural Network (ANN)

### Objective

Develop an Artificial Neural Network for customer
churn prediction.

### Implementation

- Dataset preparation
- Train-test splitting
- Numerical scaling
- Neural network architecture
- ReLU activation
- Adam optimizer
- Early stopping
- Validation monitoring

### ANN Architecture

Input Layer
    |
Dense Layer (64 neurons, ReLU)
    |
Dense Layer (32 neurons, ReLU)
    |
Output Layer (1 neuron, Sigmoid)

### Model Configuration

| Parameter | Value |
|---|---|
| Hidden Layers | (64, 32) |
| Activation | ReLU |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Maximum Iterations | 200 |
| Early Stopping | Enabled |
| Validation Fraction | 0.20 |

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Environment Note

The development environment used JupyterLite,
which runs Python through WebAssembly (WASM).

TensorFlow/Keras was not supported in this environment.

Therefore, Scikit-learn's MLPClassifier was used
to implement the Artificial Neural Network.

The implementation includes early stopping and
training loss visualization.

This is an alternative ANN implementation and does
not constitute completion of the TensorFlow/Keras-specific
requirement.

Notebook: `Day5_ANN_Neural_Networks.ipynb`

---

# Model Evaluation Metrics

The following metrics are used throughout the project:

- Accuracy: Overall prediction correctness.
- Precision: Correct positive predictions.
- Recall: Ability to identify actual churn customers.
- F1 Score: Harmonic mean of precision and recall.
- ROC-AUC: Ability to distinguish between churn and
  non-churn customers.

---

# Project Structure

Retail incremental/
|
|-- Data/
|   |-- Telco-Customer-Churn.csv
|
|-- Notebooks/
|   |-- Day2_Model_Comparison.ipynb
|   |-- Day3_Model_Tuning.ipynb
|   |-- Day4_Feature_Engineering_Ensemble.ipynb
|   |-- Day5_ANN_Neural_Networks.ipynb
|
|-- Src/
|   |-- __init__.py
|   |-- preprocessing.py
|
|-- models/
|   |-- champion_churn_pipeline.joblib
|   |-- churn_pipeline.joblib
|   |-- churn_v3.joblib
|   |-- day2_champion_model.joblib
|   |-- final_churn_model.joblib
|   |-- day4_random_forest.joblib
|   |-- day4_hist_gradient_boosting.joblib
|   |-- day4_random_forest_pipeline.joblib
|   |-- day4_boosting_pipeline.joblib
|   |-- day5_ann_model.joblib
|   |-- day5_preprocessor.joblib
|
|-- README.md

---

# Future Enhancements

- TensorFlow/Keras-based ANN implementation.
- Advanced hyperparameter optimization.
- Improved churn recall.
- Explainable AI using SHAP.
- Customer churn prediction interface.
- Deployment using Flask or FastAPI.
- Model monitoring and retraining.

---

# Author

Smrutymayee Rout

B.Tech – Computer Science and Engineering

GitHub: smruti9068

---

### Project: RetailMind AI
Machine Learning | Customer Analytics | Churn Prediction
