# 🏠 Chennai House Price Predictor

A Streamlit web application that predicts house prices in Chennai based on property attributes such as area, BHK, location, and builder using Machine Learning.

## Features
- Interactive user interface built with Streamlit.
- Dynamic selection for locations and builders.
- Real-time price inference using a trained Scikit-Learn pipeline.

## Local Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
# Final Report

## Project Overview

The objective of this project was to develop a machine learning model capable of accurately predicting residential property prices in Chennai based on key property characteristics. The workflow consisted of data preprocessing, missing value treatment, outlier analysis, feature selection, model training, evaluation, and deployment preparation.

---

## Data Preprocessing

The original dataset contained missing values in several features. Missing values were handled using domain-driven approaches, including locality-based imputation for certain attributes such as bathroom count and property age.

### Age Feature Evaluation

The age feature contained 891 missing values out of 2620 records, representing approximately 34% of the dataset. Missing values were initially imputed using the mean property age within the corresponding locality, followed by a median fallback for locations lacking sufficient information.

Further analysis revealed that the relationship between age and price was extremely weak:

- Correlation between Age and Price: **-0.018**
- Age feature contained a high proportion of imputed values.
- Predictions became less intuitive when age was included.
- Model performance improved slightly after removing age.

As a result, the age feature was excluded from the final model.

---

## Outlier Analysis

To evaluate the impact of outliers, two datasets were prepared:

1. Dataset with outliers retained
2. Dataset after outlier removal using the Interquartile Range (IQR) method based on Price Per Square Foot

A temporary feature, **Price Per Square Foot**, was created to identify extreme observations. Records outside the acceptable IQR range were removed, and the feature was subsequently dropped to prevent data leakage.

### Model Comparison

| Dataset | RMSE | Test R² | Train R² |
|----------|----------|----------|----------|
| With Outliers | 27.23 | 0.9443 | 0.9465 |
| Without Outliers | 27.23 | 0.9328 | 0.9522 |

### Interpretation

Although outlier removal reduced the presence of extreme observations, it did not improve predictive performance. The model trained on the original dataset achieved a higher Test R² score while maintaining an identical RMSE. Additionally, the gap between Training and Testing R² was smaller when outliers were retained, indicating better generalization.

Therefore, the final model was trained using the dataset with outliers retained.

---

## Final Model

A Linear Regression model was selected as the final algorithm due to its strong predictive performance, interpretability, and simplicity.

### Features Used

**Numerical Features**
- Area
- BHK
- Bathroom

**Categorical Features**
- Construction Status
- Location
- Builder

### Features Excluded

- Age

The feature was removed because of its weak predictive power, large percentage of missing values, and lack of measurable improvement in model performance.

---

## Model Performance

The final Linear Regression model achieved the following results on the testing dataset:

| Metric | Value |
|----------|----------|
| MSE | 741.66 |
| RMSE | 27.23 |
| Test R² | 0.9443 |
| Train R² | 0.9465 |

### Interpretation

- The model explains approximately **94.43% of the variance** in Chennai house prices.
- The average prediction error is approximately **27.23 lakhs**.
- Training and testing performance are nearly identical, indicating minimal overfitting.
- The model demonstrates strong generalization capability and stable predictive behavior on unseen data.

---

## Key Findings

- Property area is one of the strongest predictors of house prices.
- Location significantly influences property valuation.
- Builder reputation contributes meaningfully to price variation.
- The age feature provided negligible predictive value and was removed.
- Outlier removal did not improve model performance and reduced generalization quality.
- The final model achieved strong predictive accuracy while maintaining simplicity and interpretability.

---

## Conclusion

A Linear Regression model was successfully developed for Chennai house price prediction. Through systematic preprocessing, feature evaluation, and comparative analysis, the final model was optimized by removing low-value features and retaining original data distributions where appropriate.

The final model achieved an RMSE of **27.23** and a Test R² score of **0.9443**, demonstrating excellent predictive performance and strong generalization capability. These results indicate that the model is reliable for estimating residential property prices and is suitable for deployment within the Chennai House Price Prediction Streamlit application.