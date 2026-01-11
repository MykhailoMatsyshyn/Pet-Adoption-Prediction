**[English](README.md)** | **[Українська](README.ua.md)**

# Pet Adoption Prediction 🐾

A machine learning project that predicts the adoption speed of pets in animal shelters. This project helps animal shelters optimize their approach to finding homes for animals by predicting adoption likelihood based on various animal characteristics.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-orange.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)

## 🎯 Overview

**Business Need:** Best Friends Animal Society aims to increase the probability of pet adoption in shelters, minimizing the time animals spend waiting for their forever homes. This machine learning model predicts adoption speed based on animal characteristics, enabling shelters to tailor care approaches and marketing strategies for maximum adoption success.

The model uses a **RandomForestClassifier** with optimized hyperparameters to predict adoption speed. The AdoptionSpeed categories are:

- **0** - Adopted on the same day (adopted immediately)
- **1** - Adopted within the first week (within 1 week)
- **2** - Adopted within the first month (within 1 month)
- **3** - Adopted within 3 months (within 3 months)
- **4** - Not adopted after 100 days (after 100 days)

## ✨ Features

- **Data preprocessing pipeline** with automatic feature engineering
- **Hyperparameter optimization** for maximum model performance
- **Handles class imbalance** using appropriate sampling techniques
- **Comprehensive evaluation metrics** and visualizations
- **Easy-to-use pipeline** for training and prediction

## 📊 Results

### Model Performance

- **Accuracy:** 35% (macro average: 28%)
- **Best performance** on class 4 (not adopted after 100 days) with 52% precision and 54% recall

### Visualization Results

#### Confusion Matrix

![Confusion Matrix](models/logs/result_confusion_matrix.png)

- The confusion matrix shows how often the model predicts each class for each actual class.
- The diagonal values represent correct predictions, while off-diagonal values indicate misclassifications.
- Overall, the number of correct predictions (diagonal values) is relatively low, indicating room for model improvement.

#### ROC Curve (Multiclass)

![ROC Curve](models/logs/roc_curve_multiclass.png)

- The ROC (Receiver Operating Characteristic) curve evaluates the model's ability to distinguish between different adoption speed classes.
- Each curve represents a different class, and the area under the curve (AUC) indicates the model's performance for that class.
- Curves closer to the top-left corner indicate better classification performance.
- The multiclass ROC curve helps visualize the trade-off between true positive rates and false positive rates across all classes.

#### Feature Importances

![Feature Importances](models/logs/feature_importances.png)

- This visualization shows which features are most important for the model's predictions.
- Features with higher importance scores have a greater impact on the adoption speed predictions.
- Understanding feature importance helps identify which animal characteristics most influence adoption outcomes.
- This information can guide shelters in focusing on the most impactful factors when preparing animals for adoption.

#### Prediction Distribution

![Prediction Distribution](models/logs/prediction_distribution.png)

- This histogram shows the frequency of different classes predicted by the model.
- Class 2 has the highest frequency in predictions, while class 0 has the lowest frequency.
- This imbalance may indicate that the model struggles to predict certain classes, possibly due to insufficient training data in these categories.
- The distribution helps identify potential biases in the model's predictions.

#### Comparison Histogram

![Comparison Histogram](models/logs/comparison_histogram.png)

- This diagram allows us to see how actual AdoptionSpeed values are distributed compared to predicted values from the model.
- The model shows a tendency to shift results for some categories, with certain classes dominating the predictions.
- This may indicate imbalances in the training data or insufficient model generalization.
- Comparing actual vs predicted distributions helps assess the model's overall performance and identify areas for improvement.

## 📦 Requirements

- Python 3.7 or higher
- numpy==1.24.3
- pandas==2.2.2
- scikit-learn==1.5.2
- imbalanced-learn==0.12.4
- seaborn==0.11.1
- matplotlib==3.4.3

## 🚀 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/pet-adoption-prediction.git
   cd pet-adoption-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Training the Model

To train the model, ensure your training data is in `data/train_data.csv` and run:

```bash
python pipeline/train.py
```

After execution, the following files will be created in the `models/` directory:

- `param_dict.pickle`: Data preprocessing parameters needed for prediction
- `finalized_model.sav`: Trained model for making predictions
- `models/logs/`: Directory containing evaluation metrics and visualizations

### Making Predictions

To make predictions on new data, place your data in `data/new_data.csv` and run:

```bash
python pipeline/predict.py
```

Results will be saved in `data/prediction_results.csv` with predicted adoption speeds in the `AdoptionSpeed_pred` column.

## 📁 Project Structure

```
Pract_5/
│
├── configs/              # Configuration files
│   ├── columns.py        # Column definitions
│   └── model_best_hyperparameters.py
│
├── data/                 # Data files
│   ├── train_data.csv
│   ├── new_data.csv
│   └── prediction_results.csv
│
├── models/               # Trained models and results
│   ├── finalized_model.sav
│   ├── param_dict.pickle
│   └── logs/            # Evaluation metrics and visualizations
│
├── pipeline/             # Main scripts
│   ├── train.py
│   └── predict.py
│
├── utils/                # Utility functions
│   └── plotting_functions.py
│
└── requirements.txt
```

## 🔧 Model Details

The model uses **RandomForestClassifier** with optimized hyperparameters based on previous analysis. Predictions are based on animal characteristics such as:

- Age and breed information
- Health status
- Behavioral traits
- Physical attributes
- Location and other significant factors

The model handles class imbalance and provides probability estimates for each adoption speed category, helping shelters prioritize animals that need the most attention.
