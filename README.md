# Telco Customer Churn Prediction

A data cleaning, EDA, and machine learning project built on IBM's Telco Customer Churn dataset — predicting which customers are likely to leave a telecom provider, and identifying the key drivers behind churn.

## 📌 Project Overview

Customer churn is one of the most expensive problems in the telecom industry — acquiring a new customer costs significantly more than retaining an existing one. This project walks through a full data science workflow: cleaning messy raw data, exploring patterns in customer behavior, engineering features, and training classification models to predict churn.

## 📊 Dataset

- **Source:** [Telco Customer Churn (IBM Sample Dataset)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) via Kaggle
- **Size:** 7,043 customers, 21 columns
- **Target:** `Churn` (Yes/No)
- **Features:** demographics (gender, senior citizen, partner, dependents), account info (tenure, contract, payment method, charges), and subscribed services (phone, internet, streaming, security add-ons)

## 🧹 Data Cleaning

- Converted `TotalCharges` from string to numeric, handling blank-space entries (new customers with `tenure = 0`) by filling with 0
- Collapsed redundant categories like `"No internet service"` and `"No phone service"` into a simple `"No"` to reduce noise in categorical columns
- Verified no duplicate `customerID` entries
- Dropped `customerID` as a non-predictive identifier

## 🔍 Exploratory Data Analysis

- Confirmed class imbalance: ~27% of customers churned
- Churn rate broken down by contract type — month-to-month customers churn far more than one/two-year contract holders
- Churn vs. tenure distribution — newer customers churn at a much higher rate
- Monthly charges compared across churned vs. retained customers
- Correlation heatmap across all encoded features to spot relationships worth engineering further

## 🛠️ Feature Engineering

- **Tenure buckets:** grouped customers into 0–1, 1–2, 2–3, 3–4, and 4+ year bands to capture non-linear tenure effects
- **Service count:** created a `numService` feature counting how many add-on services (security, backup, streaming, etc.) each customer subscribes to
- One-hot encoded remaining categorical variables (`drop_first=True` to avoid multicollinearity)

## 🤖 Modeling

Three classifiers were trained and compared:

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.57 | 0.69 | 0.62 | 0.842 |
| Random Forest | 0.99 | 0.99 | 0.99 | 0.9999 |
| Gradient Boosting | 0.59 | 0.75 | 0.66 | 0.870 |

Class imbalance was addressed using **SMOTE** oversampling.

> ⚠️ **Known issue:** SMOTE was applied to the full dataset before the train/test split, which allows synthetic samples derived from test-set customers to leak into training. This almost certainly explains the near-perfect Random Forest score above — it is not a trustworthy estimate of real-world performance. The fix (in progress) is to fit SMOTE on the training set only, after splitting. Logistic Regression and Gradient Boosting scores are more representative of realistic performance.

## 📈 Key Findings

- **Contract type** is one of the strongest churn signals — month-to-month customers are far more likely to leave than those on annual contracts
- **Tenure** matters a lot — churn risk is highest in a customer's first year and drops steadily afterward
- **Monthly charges** trend higher among churned customers, suggesting price sensitivity plays a role
- Customers with **fewer subscribed services** tend to churn more, hinting that engagement/bundling could be a retention lever

## 🧰 Tech Stack

- Python, pandas, NumPy
- seaborn, matplotlib
- scikit-learn
- imbalanced-learn (SMOTE)

## 📁 Project Structure

```
├── Telco_Customer_Churn.ipynb   # Main notebook: cleaning, EDA, modeling
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
└── README.md
```

## ▶️ How to Run

```bash
git clone <your-repo-url>
cd telco-customer-churn
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
jupyter notebook Telco_Customer_Churn.ipynb
```

## 🔜 Next Steps

- Fix the SMOTE leakage by resampling only the training split
- Add cross-validation for more robust model comparison
- Tune hyperparameters (GridSearchCV / RandomizedSearchCV) for Random Forest and Gradient Boosting
- Add SHAP values for model explainability and feature importance visualization
- Package the best model behind a simple prediction script or API

## 👤 Author

Omar — Junior Data Scientist

---

*This project was built as part of a portfolio to demonstrate end-to-end data science workflow: cleaning, EDA, feature engineering, and classification modeling.*
