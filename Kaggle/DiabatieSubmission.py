# ==============================
# Diabetes Prediction Challenge - Full Ultimate Code with Optuna Tuning
# ==============================

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

# -----------------------------
# 1. Load data
# -----------------------------
train_path = r'Dataset\trainDataDiabaties.csv'  # tera path
train = pd.read_csv(train_path)

print("Data loaded:", train.shape)

X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']

# -----------------------------
# 2. Categorical encoding
# -----------------------------
categorical_cols = ['gender', 'ethnicity', 'education_level',
                    'income_level', 'smoking_status', 'employment_status']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# -----------------------------
# 3. Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. Optuna Objective with StratifiedKFold + Early Stopping (Fixed)
# -----------------------------
def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 1.5),
        'random_state': 42,
        'n_jobs': -1
    }

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in kfold.split(X_scaled, y):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        y_prob = model.predict(dval)
        auc_scores.append(roc_auc_score(y_val, y_prob))

    return np.mean(auc_scores)

# -----------------------------
# 5. Run Optuna
# -----------------------------
print("\nStarting Optuna tuning (50 trials – time lagega, chai pi le!)")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)

print(f"\nBEST CV AUC: {study.best_value:.5f}")
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# -----------------------------
# 6. Train Final Model on Full Data
# -----------------------------
best_params = study.best_params
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'auc'

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_scaled, y)

print("Final model trained on full data!")

# -----------------------------
# 7. Test data load and predict
# -----------------------------
test_path = r'Dataset\testDiabaties.csv'
test = pd.read_csv(test_path)

test_ids = test['id']
X_test = test.drop(['id'], axis=1)

# Encoding
for col in categorical_cols:
    if col in X_test.columns:
        le = LabelEncoder()
        X_test[col] = le.fit_transform(X_test[col].astype(str))

# Scaling
X_test_scaled = scaler.transform(X_test)

# Predict
test_prob = final_model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# 8. Submission
# -----------------------------
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_prob
})

submission.to_csv('submission_ultimate_optuna.csv', index=False)
print("\nsubmission_ultimate_optuna.csv bana diya – Kaggle pe upload kar de!")
print(submission.head())