# ML CCP — Hotel Booking Cancellations

## Entrypoint

`main.py` — single-file script that loads the dataset, preprocesses, trains 5 classifiers, and saves plots.

## Run

```bash
python main.py
```

Only dependency is a standard Anaconda/Colab environment: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

## Data

- `Hotel Booking Cancellations.csv` (~36k rows, Kaggle source) — must be in the repo root (same dir as `main.py`).
- Target column: `booking status` (Canceled / Not_Canceled).
- Preprocessing drops `Booking_ID` and `date of reservation` (not useful as-is).
- Categorical columns (`type of meal`, `room type`, `market segment type`) encoded with `LabelEncoder`.
- Numerical features scaled with `StandardScaler` (for LR and SVM).

## Models (5)

1. Logistic Regression (max_iter=1000)
2. Decision Tree (default params)
3. SVM (RBF kernel, probability=True)
4. Random Forest (100 estimators)
5. Gradient Boosting (100 estimators)

All use `random_state=42`.

## Output

Console prints all metrics + 4 PNG files saved to repo root:
- `eda_plots.png`
- `confusion_matrices.png`
- `roc_curves.png`
- `model_comparison.png`

## Key facts

- 80/20 stratified split (`random_state=42`).
- Training time is measured per model.
- `LabelEncoder` maps `Canceled=0`, `Not_Canceled=1` (alphabetical).
- No tests, no CI, no package manager — this is a self-contained notebook-style script.
- Warnings suppressed via `warnings.filterwarnings("ignore")`.
