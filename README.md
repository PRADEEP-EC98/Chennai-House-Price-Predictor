# 🏠 Chennai House Price Predictor

A Machine Learning-powered Streamlit application that predicts residential property prices in Chennai based on property characteristics such as area, BHK, bathroom count, location, builder, and construction status.

Web App -> https://chennai-house-price-predictor-trx2yw5ooqpudcncnfeyrm.streamlit.app

---

## 🚀 Project Overview

The objective of this project was to develop a reliable house price prediction model for Chennai properties. The project covers the complete machine learning workflow:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Missing Value Treatment
- Outlier Analysis
- Feature Selection
- Model Building & Evaluation
- Streamlit Deployment

The final model was trained using **Linear Regression** and deployed through a Streamlit web application.

---

## 📂 Project Structure

```text
Chennai_House_Price_Predictor/
│
├── Data/
│   ├── chennai_house_market_data_1.csv
│   ├── preprocessed_data_outliers_not_removed.csv
│   └── preprocessed_data_outliers_removed.csv
│
├── Models/
│   └── house_price_linearmodel.pkl
│   └── house_price_linearmodel_adjusted_to_outliers.pkl
│
├── Notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   └── LinearRegression.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔍 Data Preprocessing

### Missing Value Treatment

Several features contained missing values and were treated using domain-specific techniques.

### Age Feature Evaluation

The age feature contained:

- 891 missing values
- Approximately 34% missing data

Further analysis revealed:

- Correlation between Age and Price: **-0.018**
- Minimal predictive value
- High proportion of imputed values
- Model performance improved slightly after removing the feature

Therefore, **Age was excluded from the final model.**

---

## 📊 Outlier Analysis

Two datasets were created:

1. Dataset with outliers retained
2. Dataset after IQR-based outlier removal

Outliers were identified using **Price Per Square Foot**.

### Model Comparison

| Dataset | RMSE | Test R² | Train R² |
|----------|----------|----------|----------|
| With Outliers | 27.23 | 0.9443 | 0.9465 |
| Without Outliers | 27.23 | 0.9328 | 0.9522 |

### Final Decision

Although the dataset containing outliers achieved a slightly higher Test R² score, extensive validation using real-world property examples and unseen inputs demonstrated that the outlier-removed model produced more stable and realistic price estimates.

The difference in performance metrics was small, while prediction quality improved in practical scenarios.

Therefore, the **outlier-removed dataset** was selected for the final production model.

---

## 🤖 Final Model

### Algorithm

**Linear Regression**

### Features Used

#### Numerical Features

- Area
- BHK
- Bathroom

#### Categorical Features

- Construction Status
- Location
- Builder

### Features Excluded

- Age

---

## 📈 Model Performance

| Metric | Value |
|----------|----------|
| MSE | 140.81 |
| RMSE | 27.23 |
| Test R² | 0.9328 |
| Train R² | 0.9522 |

### Interpretation

- The model explains approximately **93.28%** of the variance in house prices.
- Average prediction error is approximately **27.23 lakhs**.
- The model demonstrates strong predictive capability across different property configurations.
- Model predictions were validated against real market examples and produced realistic estimates.

---

## ✅ Key Findings

- Property area is one of the strongest drivers of house prices