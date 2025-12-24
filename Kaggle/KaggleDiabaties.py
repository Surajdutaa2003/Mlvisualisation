# ==============================
# Diabetes Challenge - Optuna Tuning for XGBoost
# ==============================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import optuna

# Load data
train = pd.read_csv(r'Dataset\trainDataDiabaties.csv')

X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']

# Encoding
categorical_cols = ['gender', 'ethnicity', 'education_level',
                    'income_level', 'smoking_status', 'employment_status']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 1.5),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc'
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    return auc

# Run Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # 50 trials – 20-30 min lagega

print("\nBest parameters:")
print(study.best_params)

print("\nBest AUC:", study.best_value)