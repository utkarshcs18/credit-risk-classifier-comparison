# Credit Risk Classifier Comparison

## Problem Statement

Banks and NBFCs need to decide, **before approving a loan**, whether an applicant is likely to **default** (fail to repay) or repay successfully. Manually reviewing every application is slow and inconsistent.

This project builds a **supervised binary classification** model that predicts loan default risk based on an applicant's financial and personal profile. Three algorithms — **Logistic Regression, Decision Tree, and SVM** — are trained on the same data and compared, so the final model choice is backed by evidence, not guesswork.

## Objectives

- Perform EDA on the loan applicant dataset
- Handle missing values and encode categorical features
- Handle class imbalance (defaults are usually the minority class)
- Train and compare Logistic Regression, Decision Tree, and SVM
- Evaluate using Accuracy, Precision, Recall, F1-score (not accuracy alone — a bank cares more about catching defaulters than raw accuracy)
- Pick a final model with a clear, defensible reason

## Input

A dataset where each row is one loan applicant, with columns such as:
- `Age`, `Income`, `Employment_Type`, `Credit_Score`, `Loan_Amount`, `Loan_Term`, `Existing_Debts`, `Marital_Status`, `Education`, etc.
- Target column: `Loan_Status` / `Default` / `Risk_Flag` → `1` (default/risky) or `0` (repaid/safe)

Dataset: [Loan Prediction Dataset (Analytics Vidhya, via Kaggle)](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

## Output

- A trained model that, given a **new applicant's details**, outputs:
  - Predicted class: `0` (safe) or `1` (risky/default)
  - Probability score (e.g., "78% chance of default")
- A comparison table of all three models' metrics on the same test set
- A short written conclusion on which model is best suited for this problem and why

## Folder Structure

```
credit-risk-classifier-comparison/
│
├── data/
│   ├── raw/
│   │   └── loan_data.csv
│   └── processed/
│       └── loan_data_cleaned.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_comparison.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── predict.py
│
├── results/
│   ├── confusion_matrices/
│   │   ├── logistic_regression_cm.png
│   │   ├── decision_tree_cm.png
│   │   └── svm_cm.png
│   ├── metrics_comparison.csv
│   └── feature_importance.png
│
├── requirements.txt
├── README.md
└── main.py
```

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Decision Tree, SVM, metrics)
- imbalanced-learn (SMOTE, for class imbalance)
- Matplotlib, Seaborn

## How to Run

```bash
git clone https://github.com/utkarshcs18/credit-risk-classifier-comparison.git
cd credit-risk-classifier-comparison
pip install -r requirements.txt
python main.py
```


<!-- 
## Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | -        | -         | -      | -        |
| Decision Tree       | -        | -         | -      | -        |
| SVM                 | -        | -         | -      | -        |
 -->
