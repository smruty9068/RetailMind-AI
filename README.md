# RetailMind-AI

An evolving, hands-on retail intelligence and machine learning project. The repository documents a day-wise learning and implementation journey across classical machine learning, feature engineering, neural networks, computer vision, sequence modelling, and generative deep learning.

> **Project status:** Days 1–11 are documented below. The work uses multiple datasets and focused demonstrations rather than one single end-to-end retail dataset. Some later-day examples use synthetic data for learning and should not be interpreted as validated production models.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Day-wise Progress](#day-wise-progress)
- [Datasets](#datasets)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Current Scope and Limitations](#current-scope-and-limitations)
- [Future Improvements](#future-improvements)

---

## Project Overview

RetailMind-AI explores how machine learning and deep learning techniques can support retail-related intelligence tasks. The project progresses from customer churn modelling and feature engineering to image classification, product localization, demand sequence forecasting, and synthetic data generation.

The work is organized as a sequence of Jupyter notebooks. Each stage introduces a technique, applies it to a suitable dataset or demonstration, evaluates or visualizes the result where applicable, and saves selected models or outputs.

### Main problem areas explored

- Customer churn prediction and customer-retention analysis.
- Data preprocessing, feature engineering, feature selection, and model comparison.
- Neural-network-based classification.
- Fashion product image classification and transfer learning.
- Image explainability, foreground segmentation, and object detection.
- Sequence-to-sequence forecasting with attention.
- Synthetic retail customer profile generation using GANs and VAEs.

## Objectives

1. Build a structured workflow for loading, cleaning, preparing, and analysing data.
2. Compare classical machine learning models using appropriate evaluation metrics.
3. Improve model performance through hyperparameter tuning and feature engineering.
4. Explore neural networks and convolutional neural networks for classification.
5. Apply transfer learning to image classification.
6. Study computer-vision techniques for explaining, segmenting, and detecting objects in images.
7. Understand sequence modelling and attention for multi-step forecasting.
8. Explore generative models for creating synthetic data.
9. Maintain reusable notebooks, trained models, and generated outputs in a GitHub repository.

---

## Day-wise Progress

| Day | Focus | Work covered |
|---|---|---|
| **Day 1** | Dataset understanding and initial exploration | Began the project with retail/customer analytics context and initial inspection of the customer churn data. |
| **Day 2** | Baseline model comparison | Prepared the Telco Customer Churn dataset, handled missing/numeric and categorical data, built preprocessing pipelines, and compared Decision Tree, Random Forest, and SVM classifiers. Selected a champion model using evaluation results. |
| **Day 3** | Model tuning | Applied `GridSearchCV` to tune Decision Tree, Random Forest, and SVM pipelines. Used cross-validation, recall-based model selection, learning curves, and final test metrics. Saved the tuned champion model. |
| **Day 4** | Feature engineering and ensemble learning | Created tenure groups, charge-related features, high-monthly-charge and long-term-contract indicators. Applied preprocessing, feature selection with `SelectKBest`, PCA, and ensemble methods including Bagging and Random Forest. Also evaluated HistGradientBoosting and saved model pipelines. |
| **Day 5** | Artificial Neural Network (ANN) | Built an `MLPClassifier` with hidden layers, ReLU activation, Adam optimization, and early stopping. Evaluated Accuracy, Precision, Recall, F1-score, and ROC-AUC, plotted the training loss curve, and compared it with earlier models. |
| **Day 6** | Convolutional Neural Network (CNN) | Used Fashion-MNIST (10 clothing categories). Performed normalization, stratified train/validation/test splitting, class weighting, and data augmentation. Built and trained a CNN with convolution, max-pooling, dropout, and dense layers. Evaluated classification performance and saved the model and training history. |
| **Day 7** | Transfer learning with VGG16 | Resized Fashion-MNIST images, converted grayscale images to RGB, and used pretrained VGG16 as a feature extractor. Trained a classification head, fine-tuned the last VGG16 layers with a lower learning rate, evaluated the model, and saved `fashion_vgg16.keras`. |
| **Day 8** | Grad-CAM and image segmentation | Explored model explainability using Grad-CAM and generated a heatmap for image regions influencing a prediction. Applied Otsu thresholding and morphological operations to create a foreground mask, then generated transparent and white-background product-style images. |
| **Day 9** | Object detection with YOLO | Ran an object-detection demonstration on an input image and saved detected-object crops, detection results, a summary, and visualization outputs. This demonstrates object localization and detection rather than a complete retail shelf-monitoring system. |
| **Day 10** | Seq2Seq and Attention | Built an LSTM encoder-decoder sequence model for a retail-demand-style forecasting example. Used the previous 30 time steps to predict the next 7, trained with chronological train/validation/test splits, evaluated with MAE and RMSE, and explored attention-weight visualization. |
| **Day 11** | GANs and Variational Autoencoders (VAEs) | Built a GAN with a Generator and Discriminator, and a VAE with an Encoder, latent sampling, and Decoder. Used simulated customer features (average basket value and monthly visits) to explore synthetic profile generation, reconstruction, and comparison of generative approaches. |

### Important notes on progress

- Day 1 is summarized as the initial project exploration; the current repository tree does not show a separately named Day 1 notebook.
- Day 8 uses Fashion-MNIST's simple background for the segmentation demonstration. The generated mask is not a general-purpose segmentation model.
- Day 9's object-detection example is a demonstration on an image, not evidence of a deployed real-time retail camera system.
- Day 10 and Day 11 use locally generated synthetic data. Their results demonstrate model workflows and are not real-world retail performance claims.
- GAN training can be unstable; generated samples need careful quality checks before any downstream use.

---

## Datasets

The project uses different datasets depending on the task.

### 1. Telco Customer Churn

**Repository file:** `Data/Telco-Customer-Churn.csv`

Used in the early machine-learning stages for preprocessing, churn classification, model comparison, hyperparameter tuning, feature engineering, ensemble learning, and ANN classification.

The workflow includes:
- Converting `TotalCharges` to numeric values.
- Removing the customer identifier.
- Encoding the `Churn` target (`Yes` / `No`) as a binary label.
- Imputing missing values.
- Scaling numerical features.
- One-hot encoding categorical features.

### 2. Fashion-MNIST

Loaded through `keras.datasets.fashion_mnist`.

Used for CNN image classification, transfer learning, Grad-CAM exploration, and foreground segmentation.

It contains 70,000 grayscale images across 10 clothing categories, each with a size of 28 × 28 pixels.

### 3. Synthetic retail-demand time series

Generated within the Day 10 notebook for demonstrating sequence-to-sequence forecasting. The series includes trend, weekly seasonality, yearly seasonality, and random noise.

### 4. Synthetic retail customer profiles

Generated within the Day 11 notebook. The two features are:
- `avg_basket_value`
- `monthly_visits`

This data is simulated and is used to demonstrate GAN and VAE training.

---

## Technology Stack

| Area | Tools and libraries |
|---|---|
| Programming | Python |
| Notebooks | Jupyter Notebook |
| Data handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Classical ML | Scikit-learn |
| Model persistence | Joblib |
| Deep learning | TensorFlow, Keras |
| Computer vision | OpenCV, Pillow |
| Object detection | Ultralytics YOLO (Day 9 environment) |
| Version control | Git and GitHub |

---

## Repository Structure

```text
RetailMind-AI/
│
├── Data/
│   ├── Telco-Customer-Churn.csv
│   └── Day-09/
│       └── bus.jpg
│
├── Models/
│   ├── day2_champion_model.joblib
│   ├── churn_v3.joblib
│   ├── final_churn_model.joblib
│   ├── day4_random_forest.joblib
│   ├── day4_hist_gradient_boosting.joblib
│   ├── day5_ann_model.joblib
│   ├── fashion_cnn.keras
│   ├── fashion_vgg16.keras
│   └── retail_seq2seq_attention.keras
│
├── Notebooks/
│   ├── Day2_Model_Comparison.ipynb
│   ├── Day3_Model_Tuning.ipynb
│   ├── Day4_Feature_Engineering.ipynb
│   ├── Day4_Feature_Engineering_Ensemble.ipynb
│   ├── Day5_ANN_Neural_Networks.ipynb
│   ├── Day-06.ipynb
│   ├── Day-07.ipynb
│   ├── Day-08.ipynb
│   ├── Day-09.ipynb
│   ├── Day-10-Seq2Seq-Attention.ipynb
│   └── Day-11-GAN-VAE-RetailMind-AI.ipynb
│
├── Outputs/
│   ├── Day-08/
│   └── Day-09/
│
├── Src/
│   ├── __init__.py
│   └── preprocessing.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

*The tree above highlights key files; additional outputs and model artifacts may be present. Day 1 is not listed because a separately named Day 1 notebook was not visible in the current repository tree.*

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/smruty9068/RetailMind-AI.git
cd RetailMind-AI
```

### 2. Create and activate a virtual environment (Windows)

```bash
python -m venv retailmind_env
retailmind_env\Scripts\activate
```

### 3. Install dependencies

Install the packages needed for the notebook you want to run. A general starting point is:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

For deep-learning and computer-vision notebooks, install compatible versions of TensorFlow and OpenCV in the active environment:

```bash
python -m pip install tensorflow opencv-python pillow
```

For the Day 9 YOLO notebook, install Ultralytics if it is required by that notebook:

```bash
python -m pip install ultralytics
```

> Package availability depends on your Python version and operating system. Use the same Python environment as your Jupyter kernel. On Windows, TensorFlow may run on CPU depending on the installed TensorFlow version and hardware configuration.

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open the relevant notebook from the `Notebooks/` directory.

---

## How to Run

1. Clone the repository and install the dependencies required for the selected notebook.
2. Open Jupyter Notebook using the project environment.
3. Open a notebook from `Notebooks/`.
4. Run the cells from top to bottom.
5. Check that input data paths point to the repository's `Data/` folder.
6. Check that model and output paths point to `Models/` or `Outputs/` as appropriate.

Some notebooks expect the working directory to be `Notebooks/` because they use relative paths such as `../Data/` and `../Models/`. If you launch Jupyter from a different directory, adjust the paths or set the notebook working directory accordingly.

---

## Current Scope and Limitations

- The project is a progressive collection of experiments and learning milestones, not yet a single integrated production application.
- Different stages use different datasets because each technique has a different learning objective.
- The churn models are trained on the Telco Customer Churn dataset; Fashion-MNIST is used for image tasks.
- Day 8's Grad-CAM heatmap indicates influential image regions; it is not a pixel-perfect explanation or a general segmentation solution.
- Day 9's detection outputs are from an image-based demonstration. Real store deployment would require appropriate camera data, validation, tracking, and operational testing.
- Day 10's forecasting data is synthetic. Performance on it does not establish forecasting accuracy on real store sales.
- Day 11's generated customer profiles are synthetic. They are not real customer records and should not be used for business decisions without validation.
- Model files saved in `Models/` are environment- and library-version-dependent. Re-training may be necessary if loading them in a different environment.

---

## Future Improvements

- Add a dedicated Day 1 notebook and make the initial exploration reproducible.
- Consolidate preprocessing and feature-engineering code into reusable modules under `Src/`.
- Add a consistent experiment log containing dataset versions, random seeds, model parameters, and evaluation metrics.
- Evaluate churn models with threshold analysis, calibration, and business-focused retention metrics.
- Extend computer vision from single-image demonstrations to validated product and shelf datasets.
- Replace synthetic Day 10 demand data with real time-stamped retail sales and compare against simple forecasting baselines.
- Validate GAN/VAE generated data with distribution, correlation, utility, and privacy checks.
- Integrate the separate modules into a unified retail analytics workflow or dashboard.

---

## Author

**Smrutymayee Rout**

GitHub: [@smruty9068](https://github.com/smruty9068)

Repository: [RetailMind-AI](https://github.com/smruty9068/RetailMind-AI)

---

*RetailMind-AI — Exploring machine learning and deep learning techniques for retail intelligence, one milestone at a time.*
