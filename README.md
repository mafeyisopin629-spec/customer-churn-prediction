# Customer Churn Prediction Using Gradient Boosting

**Author:** Mafeyisopin Ayeni  
**Course:** Machine Learning  
**Assignment:** Gradient Boosting  
**Year:** 2026

---

## Project Overview

This project builds a machine learning pipeline to predict customer churn 
using the Telco Customer Churn dataset from Kaggle. The goal was to identify 
which customers are likely to cancel their subscription so the business can 
step in and try to keep them.

I trained and compared two gradient boosting models: a standard Gradient 
Boosting Machine (GBM) and a tuned XGBoost. XGBoost was chosen as the final 
model because it caught 81.8% of actual churners (Recall of 0.818), which is 
the most important metric for this type of problem.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Gradient_Boosting.docx` | Original assignment brief provided by the lecturer |
| `Part_C_Practical_Coding_Exercises.ipynb` | Titanic GBM vs Logistic Regression, Random Forest vs GBM comparison, XGBoost early stopping |
| `Part_D_Mini_Project_Customer_Churn.ipynb` | Full churn prediction pipeline including EDA, feature engineering, model training, SHAP analysis |
| `Churn_Prediction_Report_Mafeyisopin_Ayeni.pdf` | Final project report (5-8 pages) |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset used (source: Kaggle IBM Telco) |

---

## Dataset

- **Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Records:** 7,043 customers
- **Features:** 21 (demographics, services, account info)
- **Target:** Churn (Yes/No)

---

## Results Summary

| Metric | GBM | XGBoost (Tuned) |
|--------|-----|-----------------|
| Accuracy | 0.796 | 0.717 |
| Precision | 0.641 | 0.481 |
| Recall | 0.529 | **0.818** |
| F1 Score | 0.580 | 0.606 |
| ROC-AUC | 0.839 | 0.838 |

XGBoost was selected as the final model due to its significantly higher Recall. 
In churn prediction, missing an actual churner is more costly than 
flagging a loyal customer by mistake.

---

## Key Findings from SHAP Analysis

- **Contract type** was the strongest churn driver. Month-to-month customers 
  churn at around 43%
- **Tenure** had a strong protective effect. The longer a customer stays, 
  the less likely they are to leave
- **Monthly charges** increased churn risk, reflecting price sensitivity
- Customers with more services subscribed were less likely to churn

---

## Libraries Used

```python
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
```

---

## How to Run

1. Clone this repository
2. Install the required libraries:
