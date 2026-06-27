import nbformat as nbf
import os

os.makedirs('output', exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ============================================================
md("""# Machine Learning Pipeline - Health Project
## Binary Classification: Adult vs Senior

**Goal:** Predict `age_group` (Adult=0, Senior=1) using NHANES health examination data.

---

## 1. Import Libraries""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
import xgboost as xgb

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("All libraries loaded successfully.")""")

# ============================================================
md("""## 2. Load Data""")

code("""train = pd.read_csv('data/Train_dataset.csv')
test = pd.read_csv('data/Test_dataset.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
train.head()""")

# ============================================================
md("""## 3. Exploratory Data Analysis (Summary)

Key findings from EDA:
- **Target imbalance**: ~84% Adult, ~16% Senior
- **Missing values**: <1% per feature, can be imputed
- **Key predictors**: Glucose metrics (LBXGLU, LBXGLT), Insulin (LBXIN), Diabetes status (DIQ010), BMI
- **SEQN**: Unique ID, not predictive — drop it
- **PAQ605**: Has value `7` (Refused) in train but not in test — encode carefully""")

# ============================================================
md("""## 4. Preprocessing""")

md("""### 4.1 Handle Target Variable
- Map 'Adult' → 0, 'Senior' → 1
- Drop rows with NaN target""")

code("""# Encode target
train['age_group'] = train['age_group'].map({'Adult': 0, 'Senior': 1})

# Drop rows where target is NaN
train = train.dropna(subset=['age_group']).reset_index(drop=True)
train['age_group'] = train['age_group'].astype(int)

print(f"Train target distribution:\\n{train['age_group'].value_counts()}")
print(f"Train shape after dropping NaN target: {train.shape}")""")

md("""### 4.2 Separate Features and Target""")

code("""# Drop SEQN (unique ID, not predictive)
X = train.drop(columns=['SEQN', 'age_group'])
y = train['age_group']
X_test = test.drop(columns=['SEQN'])

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Test shape: {X_test.shape}")""")

md("""### 4.3 Define Feature Groups""")

code("""num_features = ['BMXBMI', 'LBXGLU', 'LBXGLT', 'LBXIN']
cat_features = ['RIAGENDR', 'PAQ605', 'DIQ010']

print("Numerical features:", num_features)
print("Categorical features:", cat_features)""")

md("""### 4.4 Build Preprocessing Pipeline

Strategy:
- **Numerical**: Impute with median → RobustScaler (handles outliers well)
- **Categorical**: Impute with most frequent → OneHotEncode (handle_unknown='ignore' for PAQ605=7)""")

code("""from sklearn.preprocessing import RobustScaler, OneHotEncoder

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

print("Preprocessing pipeline defined.")""")

md("""### 4.5 Train/Validation Split

Use **stratified** split to preserve class distribution.""")
 
code("""X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Train target distribution:\\n{y_train.value_counts()}")
print(f"Val target distribution:\\n{y_val.value_counts()}")""")

# ============================================================
md("""## 5. Model Training & Evaluation""")

md("""### 5.1 Define Models

We'll train and compare:
1. **Logistic Regression** — interpretable baseline
2. **Random Forest** — ensemble, handles non-linearity
3. **XGBoost** — gradient boosting, usually performs well
4. **Gradient Boosting** — alternative boosting""")

code("""models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        random_state=42, eval_metric='logloss', scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
    )
}

print(f"Defined {len(models)} models.")""")

md("""### 5.2 Cross-Validation Evaluation

Use Stratified 5-Fold CV for robust evaluation.""")
 
code("""cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    scores = {
        'accuracy': cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy'),
        'precision': cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='precision'),
        'recall': cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='recall'),
        'f1': cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1'),
        'roc_auc': cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    }

    cv_results.append({
        'Model': name,
        'Accuracy': f"{scores['accuracy'].mean():.4f} (+/- {scores['accuracy'].std():.4f})",
        'Precision': f"{scores['precision'].mean():.4f} (+/- {scores['precision'].std():.4f})",
        'Recall': f"{scores['recall'].mean():.4f} (+/- {scores['recall'].std():.4f})",
        'F1': f"{scores['f1'].mean():.4f} (+/- {scores['f1'].std():.4f})",
        'ROC-AUC': f"{scores['roc_auc'].mean():.4f} (+/- {scores['roc_auc'].std():.4f})"
    })

    print(f"\\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f"  {metric:10s}: {scores[metric].mean():.4f} (+/- {scores[metric].std():.4f})")

cv_df = pd.DataFrame(cv_results)
print("\\n\\n=== Cross-Validation Results ===")
print(cv_df.to_string(index=False))""")

md("""### 5.3 Visualize CV Results""")

code("""# Plot CV scores
fig, ax = plt.subplots(figsize=(12, 5))
cv_plot = pd.DataFrame(cv_results)
metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
x = np.arange(len(metrics_plot))
width = 0.2

for i, row in cv_plot.iterrows():
    means = [float(row[m].split()[0]) for m in metrics_plot]
    ax.bar(x + i * width, means, width, label=row['Model'])

ax.set_ylabel('Score')
ax.set_title('Cross-Validation Performance Comparison')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_plot)
ax.legend(loc='lower right')
ax.set_ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig('output/cv_comparison.png', dpi=150, bbox_inches='tight')
plt.show()""")

md("""### 5.4 Evaluate on Validation Set""")

code("""val_results = []
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    val_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1': f1_score(y_val, y_pred),
        'ROC-AUC': roc_auc_score(y_val, y_proba)
    })

    print(f"\\n{'='*50}")
    print(f"{name} - Validation Performance")
    print(f"{'='*50}")
    print(classification_report(y_val, y_pred, target_names=['Adult', 'Senior']))

val_df = pd.DataFrame(val_results)
print("\\n\\n=== Validation Set Results ===")
print(val_df.to_string(index=False))""")

md("""### 5.5 Confusion Matrices""")

code("""fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (name, model) in zip(axes.flatten(), models.items()):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False,
                xticklabels=['Adult', 'Senior'], yticklabels=['Adult', 'Senior'])
    ax.set_title(f'{name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('output/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()""")

md("""### 5.6 ROC Curves""")

code("""plt.figure(figsize=(10, 7))
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc = roc_auc_score(y_val, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Validation Set')
plt.legend(loc='lower right')
plt.savefig('output/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()""")

# ============================================================
md("""## 6. Model Selection""")

md("""### Selection Criteria:
1. **F1-score** (primary) — balances precision and recall for imbalanced classes
2. **ROC-AUC** — overall discriminative ability
3. **Interpretability** — can we explain predictions?
4. **Consistency** — low variance across CV folds

From the results, we'll pick the best model and proceed.""")

code("""# Display consolidated comparison
print("=== Cross-Validation Summary ===")
print(cv_df.to_string(index=False))
print("\\n=== Validation Set Summary ===")
print(val_df.to_string(index=False))""")

md("""### Model Selection Decision

Based on the comparison, we select the **best-performing model** considering:
- Highest F1-score on validation set
- Highest ROC-AUC
- Low variance across CV folds
- Model interpretability""")

code("""# Select the best model based on validation F1 score
best_idx = val_df['F1'].idxmax()
best_model_name = val_df.loc[best_idx, 'Model']
best_model = models[best_model_name]

print(f"Selected Model: {best_model_name}")
print(f"Validation F1: {val_df.loc[best_idx, 'F1']:.4f}")
print(f"Validation ROC-AUC: {val_df.loc[best_idx, 'ROC-AUC']:.4f}")

# Train final pipeline on full training data
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])
final_pipeline.fit(X, y)
print("\\nFinal pipeline trained on full training data.")""")

# ============================================================
md("""## 7. Feature Importance Analysis""")

code("""# Extract feature names after preprocessing
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
feature_names = num_features + list(cat_feature_names)

# Get feature importances
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("Feature Importances:")
print(feat_imp.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, x='importance', y='feature', palette='viridis')
plt.title(f'Feature Importances - {best_model_name}')
plt.tight_layout()
plt.savefig('output/feature_importances.png', dpi=150, bbox_inches='tight')
plt.show()""")

# ============================================================
md("""## 8. Predict on Test Set""")

code("""# Predict on test set
test_preds = final_pipeline.predict(X_test)
test_proba = final_pipeline.predict_proba(X_test)[:, 1]

print(f"Test predictions shape: {test_preds.shape}")
print(f"Predicted distribution:\\n{pd.Series(test_preds).value_counts()}")
print(f"\\nPredicted proportions:\\n{pd.Series(test_preds).value_counts(normalize=True).mul(100).round(2)}")""")

md("""### Create Submission File""")

code("""submission = pd.DataFrame({'age_group': test_preds})
submission.to_csv('submission.csv', index=False)

print(submission.head(10))
print(f"\\nSubmission saved to submission.csv")
print(f"Shape: {submission.shape}")
print(f"Values: 0={sum(submission['age_group']==0)}, 1={sum(submission['age_group']==1)}")""")

md("""## 9. Summary""")

md("""### Pipeline Summary

| Step | Description |
|------|-------------|
| **Preprocessing** | Median imputation for numerical, mode for categorical; RobustScaler; OneHotEncoder |
| **Models tested** | Logistic Regression, Random Forest, XGBoost, Gradient Boosting |
| **Best model** | Selected based on F1-score and ROC-AUC |
| **Validation strategy** | Stratified 5-fold CV + 80/20 holdout |
| **Test predictions** | Saved to `submission.csv` |

### Key Insights
- Glucose metrics (LBXGLU, LBXGLT) are the strongest predictors
- Diabetes status (DIQ010) is the most important categorical feature
- BMI and insulin (LBXIN) provide additional predictive power
- The model handles class imbalance via `class_weight='balanced'` or `scale_pos_weight`
""")

# Build notebook
nb.cells = cells

output_path = 'ml_pipeline_health_project.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook saved to {output_path}")
