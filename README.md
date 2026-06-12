# 💳 Credit Scoring System

## 📌 Objective

Predict an individual's creditworthiness using past financial data and machine learning techniques. The system analyzes customer financial information and estimates the probability of loan default, helping financial institutions assess credit risk.

---

## 📊 Dataset

**Home Credit Default Risk Dataset**

The dataset contains customer demographic, financial, and credit-related information used to predict whether a customer is likely to default on a loan.

---

## 🚀 Live Demo

**Streamlit Application:**

https://codealphacreditscoringmodel-ig5eoywbfxz8ba4atmguk8.streamlit.app

---

## ✨ Features

* Data Cleaning and Preprocessing
* Missing Value Handling
* Categorical Feature Encoding
* Feature Scaling using StandardScaler
* Credit Risk Prediction
* Interactive Streamlit Web Interface
* Model Performance Evaluation
* Feature Importance Analysis
* Real-Time Creditworthiness Assessment

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib
* Joblib
* Git & GitHub

---

## 🤖 Machine Learning Models Evaluated

| Model               | Accuracy | Precision | Recall | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ------- |
| Logistic Regression | 68.7%    | 15.96%    | 67.43% | 0.744   |
| Decision Tree       | 85.99%   | 15.34%    | 16.25% | 0.542   |
| Random Forest       | 22.52%   | 9.18%     | 96.66% | 0.740   |

After evaluation, Logistic Regression was selected for deployment due to its balanced performance and strong ROC-AUC score.

---

## 📈 Model Performance

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 68.7%  |
| Precision | 15.96% |
| Recall    | 67.43% |
| F1 Score  | 25.81% |
| ROC-AUC   | 0.744  |

---

## 🎯 Prediction Categories

### 🟢 Creditworthy Customer

Low probability of loan default.

### 🟡 Medium Risk Customer

Moderate probability of loan default.

### 🔴 High Risk Customer

High probability of loan default.

---

## 📂 Project Structure

```text
CodeAlpha_Credit_Scoring_Model/
│
├── app.py
├── credit_scoring.py
├── credit_scoring_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── feature_medians.pkl
├── requirements.txt
├── README.md
├── feature_importance.csv
├── model_comparison.csv
└── HomeCredit_columns_description.csv
```
---

## 👩‍💻 Author

**Srihitha**

GitHub: https://github.com/Srihitha234

---

## ⭐ Future Enhancements

* Hyperparameter Tuning
* Advanced Feature Engineering
* XGBoost Integration
* Explainable AI (SHAP)
* Enhanced Dashboard Visualizations
* Batch Prediction Support

