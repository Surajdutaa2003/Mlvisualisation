# # ==============================
# # Diabetes Challenge - Final Medal Winning Code (Ensemble)
# # ==============================

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import xgboost as xgb
# from lightgbm import LGBMClassifier

# # -----------------------------
# # 1. Load training data
# # -----------------------------
# train_path = r'Dataset\trainDataDiabaties.csv'
# train = pd.read_csv(train_path)

# X_train_full = train.drop(['id', 'diagnosed_diabetes'], axis=1)
# y_train_full = train['diagnosed_diabetes']

# # Categorical encoding
# categorical_cols = ['gender', 'ethnicity', 'education_level',
#                     'income_level', 'smoking_status', 'employment_status']

# for col in categorical_cols:
#     le = LabelEncoder()
#     X_train_full[col] = le.fit_transform(X_train_full[col].astype(str))

# # Scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_full)

# # -----------------------------
# # 2. Train XGBoost
# # -----------------------------
# xgb_model = xgb.XGBClassifier(
#     n_estimators=1240,
#     max_depth=7,
#     learning_rate=0.03107,
#     subsample=0.93389,
#     colsample_bytree=0.95054,
#     min_child_weight=5,
#     gamma=0.32964,
#     scale_pos_weight=0.71869,
#     random_state=42,
#     n_jobs=-1
# )

# xgb_model.fit(X_train_scaled, y_train_full)

# # -----------------------------
# # 3. Train LightGBM
# # -----------------------------
# lgb_model = LGBMClassifier(
#     n_estimators=1500,
#     max_depth=10,
#     learning_rate=0.02,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     min_child_weight=5,
#     reg_alpha=0.1,
#     reg_lambda=0.1,
#     random_state=42,
#     n_jobs=-1,
#     class_weight='balanced',
#     verbose=-1
# )

# lgb_model.fit(X_train_scaled, y_train_full)

# # -----------------------------
# # 4. Load test data
# # -----------------------------
# test_path = r'Dataset\testDiabaties.csv'
# test = pd.read_csv(test_path)

# test_ids = test['id']
# X_test = test.drop(['id'], axis=1)

# # Encoding
# for col in categorical_cols:
#     if col in X_test.columns:
#         le = LabelEncoder()
#         X_test[col] = le.fit_transform(X_test[col].astype(str))

# # Scaling
# X_test_scaled = scaler.transform(X_test)

# # -----------------------------
# # 5. Ensemble Prediction (Average of XGBoost + LightGBM)
# # -----------------------------
# xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
# lgb_prob = lgb_model.predict_proba(X_test_scaled)[:, 1]

# ensemble_prob = (xgb_prob + lgb_prob) / 2  # Simple average – magic karta hai!

# # -----------------------------
# # 6. Submission
# # -----------------------------
# submission = pd.DataFrame({
#     'id': test_ids,
#     'diagnosed_diabetes': ensemble_prob
# })

# submission.to_csv('submission_ensemble_medal.csv', index=False)
# print("\nsubmission_ensemble_medal.csv bana diya – yeh medal laayega!")
# print(submission.head())





# ==============================
# Diabetes Challenge - Improved Strong Baseline
# ==============================

# ==============================
# Diabetes Prediction + Submission
# Weighted Refit Strategy
# ==============================


# ============================================
# Diabetes Prediction using Gradient Boosting
# LightGBM + CatBoost Ensemble
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from scipy.stats import rankdata

# -----------------------------
# 1. Paths (LOCAL)
# -----------------------------
TRAIN_PATH = r'Dataset\trainDataDiabaties.csv'
TEST_PATH  = r'Dataset\testDiabaties.csv'

SUB_LGB = r'submission_lgb.csv'
SUB_CAT = r'submission_cat.csv'
SUB_ENS = r'submission_ensemble.csv'

# -----------------------------
# 2. Load data
# -----------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

TARGET = 'diagnosed_diabetes'
ID_COL = 'id'

X = train_df.drop([ID_COL, TARGET], axis=1)
y = train_df[TARGET]

X_test = test_df.drop([ID_COL], axis=1)

# -----------------------------
# 3. Identify categorical cols
# -----------------------------
categorical_cols = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status'
]

cat_idx = [X.columns.get_loc(col) for col in categorical_cols]

# =====================================================
# PART A — LIGHTGBM (Label Encode + Scale)
# =====================================================

X_lgb = X.copy()
X_test_lgb = X_test.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_lgb[col] = le.fit_transform(X_lgb[col].astype(str))
    X_test_lgb[col] = le.transform(X_test_lgb[col].astype(str))

scaler = StandardScaler()
X_lgb_scaled = scaler.fit_transform(X_lgb)
X_test_lgb_scaled = scaler.transform(X_test_lgb)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_lgb_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

lgb_model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model.fit(X_tr, y_tr)

lgb_val_preds = lgb_model.predict_proba(X_val)[:, 1]
print("LightGBM AUC:", roc_auc_score(y_val, lgb_val_preds))

lgb_test_preds = lgb_model.predict_proba(X_test_lgb_scaled)[:, 1]

# =====================================================
# PART B — CATBOOST (NO ENCODING NEEDED)
# =====================================================

X_cat = X.copy()
X_test_cat = X_test.copy()

X_tr, X_val, y_tr, y_val = train_test_split(
    X_cat, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

cat_model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

cat_model.fit(
    X_tr, y_tr,
    cat_features=cat_idx,
    eval_set=(X_val, y_val),
    verbose=False
)

cat_val_preds = cat_model.predict_proba(X_val)[:, 1]
print("CatBoost AUC:", roc_auc_score(y_val, cat_val_preds))

cat_test_preds = cat_model.predict_proba(X_test_cat)[:, 1]

# =====================================================
# PART C — ENSEMBLE (AUC-FRIENDLY)
# =====================================================

ensemble_preds = (
    rankdata(lgb_test_preds) +
    rankdata(cat_test_preds)
) / 2

# =====================================================
# PART D — SAVE SUBMISSIONS
# =====================================================

pd.DataFrame({
    'id': test_df[ID_COL],
    'diagnosed_diabetes': lgb_test_preds
}).to_csv(SUB_LGB, index=False)

pd.DataFrame({
    'id': test_df[ID_COL],
    'diagnosed_diabetes': cat_test_preds
}).to_csv(SUB_CAT, index=False)

pd.DataFrame({
    'id': test_df[ID_COL],
    'diagnosed_diabetes': ensemble_preds
}).to_csv(SUB_ENS, index=False)

print("\n✅ Files saved:")
print(SUB_LGB)
print(SUB_CAT)
print(SUB_ENS)
