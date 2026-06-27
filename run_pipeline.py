"""
ML Pipeline for Health Project - Binary Classification
Predicts age_group (Adult=0, Senior=1) from NHANES health data.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_csv('data/Train_dataset.csv')
test = pd.read_csv('data/Test_dataset.csv')

print("=" * 60)
print("DATA LOADED")
print("=" * 60)
print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. TARGET ENCODING
# ============================================================
train['age_group'] = train['age_group'].map({'Adult': 0, 'Senior': 1})
train = train.dropna(subset=['age_group']).reset_index(drop=True)
train['age_group'] = train['age_group'].astype(int)

print(f"\nTarget distribution:\n{train['age_group'].value_counts()}")

# ============================================================
# 3. SPLIT FEATURES / TARGET
# ============================================================
X = train.drop(columns=['SEQN', 'age_group'])
y = train['age_group']
X_test = test.drop(columns=['SEQN'])

print(f"\nFeatures: {list(X.columns)}")

# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================
num_features = ['BMXBMI', 'LBXGLU', 'LBXGLT', 'LBXIN']
cat_features = ['RIAGENDR', 'PAQ605', 'DIQ010']

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

print("\nPreprocessing pipeline built.")

# ============================================================
# 5. TRAIN / VAL SPLIT
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")
print(f"Train target:\n{y_train.value_counts()}")
print(f"Val target:\n{y_val.value_counts()}")

# ============================================================
# 6. DEFINE MODELS
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        random_state=42, eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
    )
}

# ============================================================
# 7. CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("CROSS-VALIDATION (5-Fold Stratified)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

    print(f"\n{name}:")
    print(f"  Accuracy: {scores['accuracy'].mean():.4f} (+/- {scores['accuracy'].std():.4f})")
    print(f"  Precision: {scores['precision'].mean():.4f} (+/- {scores['precision'].std():.4f})")
    print(f"  Recall: {scores['recall'].mean():.4f} (+/- {scores['recall'].std():.4f})")
    print(f"  F1: {scores['f1'].mean():.4f} (+/- {scores['f1'].std():.4f})")
    print(f"  ROC-AUC: {scores['roc_auc'].mean():.4f} (+/- {scores['roc_auc'].std():.4f})")

cv_df = pd.DataFrame(cv_results)
print("\n\nCross-Validation Summary:")
print(cv_df.to_string(index=False))

# ============================================================
# 8. VALIDATION SET EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("VALIDATION SET EVALUATION")
print("=" * 60)

val_results = []
trained_pipelines = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    trained_pipelines[name] = pipeline

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

    print(f"\n{name}:")
    print(classification_report(y_val, y_pred, target_names=['Adult', 'Senior']))
    print(f"  ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")

val_df = pd.DataFrame(val_results)
print("\n\nValidation Summary:")
print(val_df.to_string(index=False))

# ============================================================
# 9. SELECT BEST MODEL
# ============================================================
best_idx = val_df['F1'].idxmax()
best_model_name = val_df.loc[best_idx, 'Model']
best_model = models[best_model_name]

print(f"\n{'=' * 60}")
print(f"SELECTED MODEL: {best_model_name}")
print(f"{'=' * 60}")
print(f"Validation F1: {val_df.loc[best_idx, 'F1']:.4f}")
print(f"Validation ROC-AUC: {val_df.loc[best_idx, 'ROC-AUC']:.4f}")
print(f"Selection rationale: Highest F1-score (balances precision & recall for imbalanced data)")

# ============================================================
# 10. TRAIN FINAL MODEL ON FULL DATA
# ============================================================
print("\nTraining final model on full training data...")
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])
final_pipeline.fit(X, y)
print("Final pipeline trained.")

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
feature_names = num_features + list(cat_feature_names)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 5 Feature Importances:")
print(feat_imp.head(5).to_string(index=False))

# ============================================================
# 12. PREDICT ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("PREDICTING ON TEST SET")
print("=" * 60)

test_preds = final_pipeline.predict(X_test)
test_proba = final_pipeline.predict_proba(X_test)[:, 1]

print(f"Predictions distribution:")
print(f"  Adult (0): {sum(test_preds == 0)} ({sum(test_preds == 0) / len(test_preds) * 100:.1f}%)")
print(f"  Senior (1): {sum(test_preds == 1)} ({sum(test_preds == 1) / len(test_preds) * 100:.1f}%)")

# ============================================================
# 13. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({'age_group': test_preds})
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission saved to submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Sample (first 10 rows):")
print(submission.head(10).to_string())
print("\nDone!")
