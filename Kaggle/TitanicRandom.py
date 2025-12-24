# ==============================
# Diabetes Challenge - Final Push for Better AUC
# ==============================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# Load data
train = pd.read_csv(r'Dataset\trainDataDiabaties.csv')

X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']

# Categorical encoding
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

# XGBoost with better parameters
xgb_model = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=10,
    learning_rate=0.01,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    gamma=0.1,
    scale_pos_weight=1,  # pehle zyada tha, ab neutral
    random_state=42,
    n_jobs=-1,
    eval_metric='auc',
    verbosity=0
)

print("Training final model... (15-20 min lagega, chai pi le bhai!)")
xgb_model.fit(X_train, y_train)

# Evaluation
y_prob = xgb_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print(f"\nFINAL AUC-ROC: {auc:.4f}")

# Save model
import joblib
joblib.dump(xgb_model, 'final_diabetes_model.pkl')
joblib.dump(scaler, 'final_scaler.pkl')
print("Model saved!")