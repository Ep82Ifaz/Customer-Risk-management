# Customer Risk Classification

An end-to-end machine learning project that predicts a customer's loan **risk category** (Low / Medium / High) from demographic, financial and credit-behaviour data.

## Project Files

| File | Description |
|---|---|
| `Customer_Risk_ML_Project.ipynb` | Main notebook covering the full pipeline (61 cells) |
| `random_customer_risk_dataset.csv` | Input dataset (5,000 customers, 13 columns) |
| `best_risk_model.pkl` | Exported best model (created when the notebook runs) |

## Dataset

| Column | Type | Description |
|---|---|---|
| `Customer_ID` | ID | Unique identifier (dropped before modelling) |
| `Age` | numeric | Customer age |
| `Annual_Income` | numeric | Yearly income |
| `Credit_Score` | numeric | Credit score |
| `Loan_Amount` | numeric | Requested loan amount |
| `Employment_Type` | categorical | Salaried, Self-employed, Business, Unemployed |
| `Education_Level` | categorical | High School, Bachelor, Master, PhD |
| `Marital_Status` | categorical | Married, Single, Divorced |
| `Existing_Loans` | numeric | Number of current loans |
| `Late_Payments` | numeric | Number of late payments |
| `Debt_to_Income` | numeric | Debt-to-income ratio |
| `Years_Employed` | numeric | Years in current employment |
| `Risk_Category` | **target** | Low (1,700), Medium (1,700), High (1,600) |

The data has no missing values and no duplicate rows.

## Pipeline

1. Dataset loading and exploration
2. Data cleaning (missing and duplicate values)
3. Exploratory data analysis
4. Data visualization
5. Outlier detection and handling (IQR capping)
6. Feature engineering (8 new features, e.g. loan-to-income, late-payment rate, credit band)
7. Categorical encoding (ordinal for education, one-hot for the rest)
8. Feature scaling (`StandardScaler`, fit on training data only)
9. Stratified train-test split (80/20)
10. Class imbalance analysis and SMOTE
11. Model building: XGBoost, CatBoost, LightGBM, Random Forest, Gradient Boosting, Logistic Regression
12. Hyperparameter tuning (`RandomizedSearchCV`, scored on macro-F1)
13. Stratified 5-fold cross-validation
14. Evaluation: accuracy, precision, recall, F1
15. Confusion matrices and one-vs-rest ROC-AUC
16. Feature importance and SHAP analysis
17. Model comparison
18. Best model selection (by cross-validated macro-F1)
19. Export to `.pkl`

### Data-leakage safeguards

- The train-test split happens **before** scaling; the scaler is fit on the training set only.
- SMOTE is wrapped in an `imblearn` pipeline, so it only resamples training folds during cross-validation, never validation folds or the test set.

## Results

Results from a test run (tuning is randomized, so exact numbers can vary slightly):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.776 | 0.781 | 0.777 | 0.778 | 0.920 |
| CatBoost | 0.770 | 0.773 | 0.771 | 0.772 | 0.909 |
| XGBoost | 0.759 | 0.764 | 0.760 | 0.761 | 0.906 |
| Gradient Boosting | 0.751 | 0.758 | 0.751 | 0.754 | 0.905 |
| LightGBM | 0.752 | 0.755 | 0.753 | 0.754 | 0.901 |
| Random Forest | 0.738 | 0.743 | 0.739 | 0.741 | 0.896 |

Precision, recall and F1 are macro-averaged. **Logistic Regression** was selected as the best model (cross-validated macro-F1 about 0.77). The tree-based models did not outperform it, which suggests the risk labels depend fairly linearly on the features.

## Setup

Requires Python 3.9+.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost lightgbm imbalanced-learn shap jupyter
```

## Usage

1. Put `random_customer_risk_dataset.csv` in the same folder as the notebook.
2. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Customer_Risk_ML_Project.ipynb
   ```
3. Run all cells (**Kernel → Restart & Run All**).

**Runtime:** hyperparameter tuning is the slow part, roughly 15–20 minutes on a 2-core machine. Lower `N_ITER` in the tuning cell for a faster run, or raise it for a deeper search.

## Using the Exported Model

`best_risk_model.pkl` is a dictionary containing the fitted pipeline, scaler, column order, label names and encoding rules.

The final cell of the notebook defines `predict_risk()`, which applies the same feature engineering and encoding to a raw customer record:

```python
import pickle

with open("best_risk_model.pkl", "rb") as f:
    art = pickle.load(f)

sample = {
    "Age": 35, "Annual_Income": 45000, "Credit_Score": 590, "Loan_Amount": 60000,
    "Employment_Type": "Self-employed", "Education_Level": "Bachelor",
    "Marital_Status": "Single", "Existing_Loans": 3, "Late_Payments": 4,
    "Debt_to_Income": 1.6, "Years_Employed": 2.0,
}
label, probabilities = predict_risk(sample, art)   # defined in the notebook's last cell
print(label, probabilities)
```

> Only load `.pkl` files you created yourself; unpickling untrusted files can execute arbitrary code.

## Tech Stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, CatBoost, LightGBM, SHAP, matplotlib, seaborn.
