# ML Complex Computing Problem — Hotel Booking Cancellation Prediction

A comparative machine learning project that evaluates five classification algorithms on a hotel booking cancellations dataset. The goal is to predict whether a booking will be canceled or not, and to identify the best-performing model for potential deployment.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [License](#license)

---

## Overview

This project implements a complete end-to-end machine learning pipeline — from data loading and preprocessing through exploratory analysis, model training, evaluation, and visualization. Five classifiers are benchmarked on accuracy, precision, recall, F1-score, ROC-AUC, and training time.

**Key deliverables:**
- Trained and evaluated models with reproducible metrics
- Exploratory Data Analysis (EDA) visualizations
- Confusion matrix and ROC curve comparisons
- A final bar-chart comparison of all models

---

## Dataset

**Source:** [Kaggle — Hotel Booking Cancellations Dataset](https://www.kaggle.com/)

**Shape:** 36,285 rows × 17 columns

**Target:** `booking status` — binary classification (Canceled / Not_Canceled)

| Class | Count | Percentage |
|-------|-------|------------|
| Not_Canceled | 24,396 | 67.2% |
| Canceled | 11,889 | 32.8% |

**Features:**

| Feature | Type | Description |
|---------|------|-------------|
| `Booking_ID` | String | Unique reservation identifier (dropped) |
| `number of adults` | Integer | Number of adult guests |
| `number of children` | Integer | Number of children |
| `number of weekend nights` | Integer | Weekend nights booked |
| `number of week nights` | Integer | Weekday nights booked |
| `type of meal` | Categorical | Meal plan selected |
| `car parking space` | Integer | Parking spaces requested |
| `room type` | Categorical | Room category |
| `lead time` | Integer | Days between booking and arrival |
| `market segment type` | Categorical | Booking channel / segment |
| `repeated` | Integer | Whether guest is a repeat customer |
| `P-C` | Integer | Number of previous cancellations |
| `P-not-C` | Integer | Number of previous non-cancellations |
| `average price` | Float | Average room price |
| `special requests` | Integer | Number of special requests made |
| `date of reservation` | String | Reservation date (dropped) |

**Data Quality:** No missing values, no duplicate rows.

---

## Models

All models are trained with `random_state=42` on an 80/20 stratified train-test split.

| # | Model | Key Parameters |
|---|-------|----------------|
| 1 | **Logistic Regression** | `max_iter=1000` |
| 2 | **Decision Tree** | Default scikit-learn parameters |
| 3 | **Support Vector Machine** | RBF kernel, `probability=True` |
| 4 | **Random Forest** | 100 estimators |
| 5 | **Gradient Boosting** | 100 estimators |

Numerical features are scaled with `StandardScaler` (used for Logistic Regression and SVM). Categorical features (`type of meal`, `room type`, `market segment type`) are label-encoded.

---

## Results

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Training Time (s) |
|-------|----------|-----------|--------|----------|-----|-------------------|
| Logistic Regression | 0.8103 | 0.8295 | 0.9035 | 0.8649 | 0.8617 | 0.1983 |
| Decision Tree | 0.8617 | 0.9026 | 0.8903 | 0.8964 | 0.8500 | 0.1391 |
| SVM (RBF) | 0.8289 | 0.8315 | 0.9367 | 0.8810 | 0.8796 | 155.9238 |
| Random Forest | 0.8946 | 0.9085 | 0.9377 | 0.9228 | 0.9483 | 3.2786 |
| Gradient Boosting | 0.8543 | 0.8636 | 0.9303 | 0.8957 | 0.9087 | 3.6416 |

### Visualizations

| Plot | Description |
|------|-------------|
| `eda_plots.png` | Class distribution, correlation heatmap, lead time & price distributions, lead time by status, special requests by status |
| `confusion_matrices.png` | Confusion matrix for each of the 5 models |
| `roc_curves.png` | Overlaid ROC curves for all models with AUC scores |
| `model_comparison.png` | Bar chart comparing Accuracy, Precision, Recall, and F1-Score across models |

---

## Setup & Usage

### Prerequisites

A standard Python data science environment (e.g., Anaconda, Google Colab, or a fresh virtual environment):

- Python ≥ 3.8
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Intelligent-Hotel-Booking-Cancellation-Prediction.git
cd Intelligent-Hotel-Booking-Cancellation-Prediction

# (Optional) Create and activate a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run

```bash
python main.py
```

The script will:
1. Load and preprocess the dataset
2. Perform exploratory data analysis with visualizations
3. Train all 5 models
4. Print evaluation metrics to the console
5. Save 4 PNG plot files to the project root

---

## Project Structure

```
.
├── main.py                          # Main pipeline script
├── Hotel Booking Cancellations.csv  # Dataset (from Kaggle)
├── eda_plots.png                    # EDA visualizations (generated)
├── confusion_matrices.png           # Confusion matrices (generated)
├── roc_curves.png                   # ROC curves (generated)
├── model_comparison.png             # Model comparison chart (generated)
├── AGENTS.md                        # Agent instructions
└── README.md                        # This file
```

---

## Methodology

### Preprocessing
- `Booking_ID` and `date of reservation` are dropped as they are not useful features in their raw form.
- Three categorical columns are label-encoded.
- Numerical features are standardized (zero mean, unit variance) for models sensitive to feature scale.

### Train-Test Split
- Stratified 80/20 split preserves the original class distribution in both training and test sets.
- A fixed `random_state=42` ensures reproducibility.

### Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Training Time.
- **Visual diagnostics:** Confusion matrices, ROC curves, and a grouped bar chart.

---

## License

This project is for educational and demonstration purposes. The dataset is sourced from Kaggle and is subject to its original license terms.

---

