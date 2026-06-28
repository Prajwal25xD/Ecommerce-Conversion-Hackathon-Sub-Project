# 🩺 Health Age Group Classification using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether an individual belongs to the **Adult** or **Senior** age group using demographic, physical, and laboratory health indicators.

The project follows a complete Machine Learning workflow including:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Engineering
- Model Building
- Cross Validation
- Model Evaluation
- Prediction Generation

---

# 📂 Dataset

The dataset consists of health-related attributes collected from individuals.

### Features

| Feature | Description |
|----------|-------------|
| SEQN | Unique respondent ID |
| RIAGENDR | Gender |
| PAQ605 | Physical Activity |
| BMXBMI | Body Mass Index |
| LBXGLU | Fasting Glucose |
| DIQ010 | Diabetes Diagnosis |
| LBXGLT | Glucose Tolerance |
| LBXIN | Insulin Level |

**Target Variable**

```
age_group
```

Classes:

- Adult
- Senior

---

# 📊 Dataset Summary

| Metric | Value |
|---------|-------|
| Training Samples | 1,966 |
| Training Samples (after cleaning) | 1,952 |
| Test Samples | 312 |
| Features Used | 7 |
| Numerical Features | 4 |
| Categorical Features | 3 |

---

# 📈 Exploratory Data Analysis

The EDA notebook performs:

- Dataset overview
- Missing value analysis
- Target distribution
- Feature distributions
- Correlation analysis
- Outlier detection (IQR)
- Train vs Test comparison
- Numerical statistics
- Categorical feature analysis

### Key Findings

- Adult class dominates the dataset (~84%)
- Missing values are below 1% for all features.
- BMI and Glucose variables contain moderate outliers.
- Train and Test datasets follow similar distributions.
- Strong positive correlation exists between Glucose and Glucose Tolerance.

---

# ⚙️ Data Preprocessing

The ML pipeline includes:

- Missing value imputation
- Standard Scaling
- One-Hot Encoding
- ColumnTransformer Pipeline
- Stratified Train/Test Split
- Cross Validation

---

# 🤖 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Decision Tree

---

# 📊 Cross Validation Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|----------|-----------|---------|----------|----------|
| Logistic Regression | **72.33%** | 32.34% | **66.16%** | **43.43%** | **75.77%** |
| Random Forest | **84.30%** | 51.78% | 16.71% | 24.65% | 74.10% |
| Gradient Boosting | **83.41%** | 46.50% | 19.55% | 27.14% | 73.27% |

---

# 🏆 Selected Model

Although Random Forest achieved higher accuracy, **Logistic Regression** was selected because it provided the highest recall and F1 score for the minority class.

### Validation Performance

| Metric | Score |
|---------|-------|
| Validation Accuracy | **69%** |
| Validation F1 Score | **35.29%** |
| Validation ROC-AUC | **69.83%** |

---

# 📌 Feature Importance

Top contributing features include:

1. LBXGLT (Glucose Tolerance Test)
2. LBXIN (Insulin Level)
3. PAQ605 (Physical Activity)
4. DIQ010 (Diabetes Diagnosis)
5. LBXGLU (Fasting Glucose)
6. RIAGENDR (Gender)
7. BMXBMI (BMI)

---

# 📤 Predictions

The final pipeline was trained on the complete training dataset and generated predictions for all test samples.

Prediction Summary:

- Total Test Records: **312**
- Adult Predictions: **214**
- Senior Predictions: **98**

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# 📁 Project Structure

```
├── eda_health_project.ipynb
├── ml_pipeline_health_project.ipynb
├── submission.csv
├── README.md
```

---

# 🚀 Future Improvements

- XGBoost
- LightGBM
- CatBoost
- Hyperparameter Optimization
- Ensemble Learning
- SHAP Explainability
- Feature Selection
- Threshold Optimization

---

# 📌 Conclusion

This project demonstrates an end-to-end Machine Learning pipeline for binary health classification. From exploratory data analysis to preprocessing, model evaluation, and prediction generation, the workflow follows industry-standard practices using Scikit-learn pipelines. Logistic Regression achieved the best balance between recall and F1 score, making it the preferred model for this imbalanced classification problem.
