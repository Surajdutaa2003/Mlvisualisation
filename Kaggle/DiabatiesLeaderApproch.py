# # # ==============================
# # # Diabetes Challenge - FINAL PUSH (0.70+ Attempt)
# # # Weighted Refit + XGB-Heavy Ensemble
# # # ==============================

# # import numpy as np
# # import pandas as pd
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # import xgboost as xgb
# # from lightgbm import LGBMClassifier

# # # -----------------------------
# # # 1. Load Data
# # # -----------------------------
# # train_path = r'Dataset\trainDataDiabaties.csv'
# # test_path  = r'Dataset\testDiabaties.csv'

# # train = pd.read_csv(train_path)
# # test  = pd.read_csv(test_path)

# # TARGET = 'diagnosed_diabetes'
# # ID_COL = 'id'

# # X = train.drop([ID_COL, TARGET], axis=1)
# # y = train[TARGET]

# # # -----------------------------
# # # 2. Encode categoricals
# # # -----------------------------
# # cat_cols = [
# #     'gender','ethnicity','education_level',
# #     'income_level','smoking_status','employment_status'
# # ]

# # for col in cat_cols:
# #     le = LabelEncoder()
# #     X[col] = le.fit_transform(X[col].astype(str))
# #     test[col] = le.transform(test[col].astype(str))

# # # -----------------------------
# # # 3. Scaling
# # # -----------------------------
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)
# # X_test_scaled = scaler.transform(test.drop(ID_COL, axis=1))

# # # -----------------------------
# # # 4. Weighted Refit Setup
# # # -----------------------------
# # TAIL_SIZE   = 21000
# # TAIL_WEIGHT = 12.5

# # weights = np.ones(len(X_scaled))
# # weights[-TAIL_SIZE:] = TAIL_WEIGHT

# # # -----------------------------
# # # 5. Train XGBoost (MAIN)
# # # -----------------------------
# # xgb_model = xgb.XGBClassifier(
# #     n_estimators=1200,
# #     max_depth=7,
# #     learning_rate=0.03,
# #     subsample=0.93,
# #     colsample_bytree=0.95,
# #     min_child_weight=5,
# #     gamma=0.3,
# #     scale_pos_weight=0.72,
# #     random_state=42,
# #     n_jobs=-1
# # )

# # xgb_model.fit(
# #     X_scaled,
# #     y,
# #     sample_weight=weights
# # )

# # # -----------------------------
# # # 6. Train LightGBM (SUPPORT)
# # # -----------------------------
# # lgb_model = LGBMClassifier(
# #     n_estimators=1500,
# #     learning_rate=0.02,
# #     num_leaves=31,
# #     max_depth=10,
# #     subsample=0.8,
# #     colsample_bytree=0.8,
# #     min_child_weight=5,
# #     reg_alpha=0.1,
# #     reg_lambda=0.1,
# #     class_weight='balanced',
# #     random_state=42,
# #     n_jobs=-1,
# #     verbose=-1
# # )

# # lgb_model.fit(
# #     X_scaled,
# #     y,
# #     sample_weight=weights
# # )

# # # -----------------------------
# # # 7. Prediction
# # # -----------------------------
# # xgb_prob = xgb_model.predict_proba(X_test_scaled)[:,1]
# # lgb_prob = lgb_model.predict_proba(X_test_scaled)[:,1]

# # # 🔥 FINAL ENSEMBLE (XGB-heavy)
# # final_prob = 0.6 * xgb_prob + 0.4 * lgb_prob

# # # -----------------------------
# # # 8. Submission
# # # -----------------------------
# # submission = pd.DataFrame({
# #     'id': test[ID_COL],
# #     'diagnosed_diabetes': final_prob
# # })

# # submission.to_csv(
# #     'submission_final_070_attempt.csv',
# #     index=False
# # )

# # print("✅ submission_final_070_attempt.csv ready")
# # print(submission.head())




# # ============================================
# # Diabetes Prediction – FINAL STRONG ENSEMBLE
# # XGBoost + LightGBM + CatBoost
# # Weighted Refit + Distribution-aware training
# # ============================================

# import numpy as np
# import pandas as pd

# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from catboost import CatBoostClassifier
# import xgboost as xgb
# from lightgbm import LGBMClassifier

# # --------------------------------------------
# # 1. LOAD DATA (LOCAL PATH – unchanged)
# # --------------------------------------------
# train_path = r'Dataset\trainDataDiabaties.csv'
# test_path  = r'Dataset\testDiabaties.csv'

# train = pd.read_csv(train_path)
# test  = pd.read_csv(test_path)

# target = 'diagnosed_diabetes'

# X = train.drop(['id', target], axis=1)
# y = train[target]

# X_test = test.drop(['id'], axis=1)

# # --------------------------------------------
# # 2. CATEGORICAL ENCODING
# # --------------------------------------------
# categorical_cols = [
#     'gender', 'ethnicity', 'education_level',
#     'income_level', 'smoking_status', 'employment_status'
# ]

# for col in categorical_cols:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col].astype(str))
#     X_test[col] = le.transform(X_test[col].astype(str))

# # --------------------------------------------
# # 3. SCALING
# # --------------------------------------------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_test_scaled = scaler.transform(X_test)

# # --------------------------------------------
# # 4. WEIGHTED REFIT STRATEGY (KEY)
# # --------------------------------------------
# cutoff = int(len(X_scaled) * 0.97)  # last ~3% = test-like
# weights = np.ones(len(X_scaled))
# weights[cutoff:] = 15.0             # strong emphasis

# # --------------------------------------------
# # 5. XGBOOST MODEL
# # --------------------------------------------
# xgb_model = xgb.XGBClassifier(
#     n_estimators=1200,
#     max_depth=7,
#     learning_rate=0.03,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     min_child_weight=5,
#     gamma=0.3,
#     scale_pos_weight=0.75,
#     random_state=42,
#     n_jobs=-1
# )

# xgb_model.fit(X_scaled, y, sample_weight=weights)
# xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

# # --------------------------------------------
# # 6. LIGHTGBM MODEL
# # --------------------------------------------
# lgb_model = LGBMClassifier(
#     n_estimators=900,
#     learning_rate=0.03,
#     num_leaves=31,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     class_weight='balanced',
#     random_state=42,
#     n_jobs=-1,
#     verbose=-1
# )

# lgb_model.fit(X_scaled, y, sample_weight=weights)
# lgb_prob = lgb_model.predict_proba(X_test_scaled)[:, 1]

# # --------------------------------------------
# # 7. CATBOOST MODEL (DIVERSITY MODEL)
# # --------------------------------------------
# cat_model = CatBoostClassifier(
#     iterations=1200,
#     depth=7,
#     learning_rate=0.03,
#     loss_function='Logloss',
#     eval_metric='AUC',
#     random_seed=42,
#     verbose=200
# )

# cat_model.fit(X_scaled, y, sample_weight=weights)
# cat_prob = cat_model.predict_proba(X_test_scaled)[:, 1]

# # --------------------------------------------
# # 8. FINAL ENSEMBLE (BEST WEIGHTS)
# # --------------------------------------------
# final_prob = (
#     0.45 * xgb_prob +
#     0.35 * lgb_prob +
#     0.20 * cat_prob
# )

# # --------------------------------------------
# # 9. SUBMISSION
# # --------------------------------------------
# submission = pd.DataFrame({
#     'id': test['id'],
#     'diagnosed_diabetes': final_prob
# })

# submission.to_csv(
#     'submission_final_catboost_ensemble.csv',
#     index=False
# )

# print("✅ submission_final_catboost_ensemble.csv created")
# print(submission.head())
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load
train = pd.read_csv(r'Dataset\trainDataDiabaties.csv')
test = pd.read_csv(r'Dataset\testDiabaties.csv')

TARGET = 'diagnosed_diabetes'
ID_COL = 'id'

X = train.drop([ID_COL, TARGET], axis=1).copy()
y = train[TARGET].copy()
X_test = test.drop([ID_COL], axis=1).copy()

# 2. Advanced Feature Engineering (Medical Context)
for df in [X, X_test]:
    if 'bmi' in df.columns and 'age' in df.columns:
        df['metabolic_index'] = df['bmi'] * df['age']
    # If blood pressure exists, this is a top-tier feature
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

# 3. Robust Encoding (Fixes the 'object' error)
# We target encode known high-cardinality columns, 
# then LabelEncode EVERYTHING else that is still a string.
target_enc_cols = ['gender', 'ethnicity', 'education_level', 'income_level']
for col in [c for c in target_enc_cols if c in X.columns]:
    # Use mapping to avoid errors with unseen categories in test
    mapping = train.groupby(col)[TARGET].mean()
    X[col] = X[col].map(mapping)
    X_test[col] = X_test[col].map(mapping).fillna(mapping.mean())

# Catch-all: Convert any remaining 'object' columns (like employment_status) to numbers
obj_cols = X.select_dtypes(include=['object']).columns
for col in obj_cols:
    le = LabelEncoder()
    # Use astype(str) to handle potential NaNs
    X[col] = le.fit_transform(X[col].astype(str))
    # Handle new categories in test set gracefully
    X_test[col] = X_test[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

# 4. Final Models (Focusing on the 0.699+ stability)
# Lower learning rate and more estimators for better generalization
xgb_model = xgb.XGBClassifier(
    n_estimators=1500, 
    max_depth=5, 
    learning_rate=0.015,  # Slowed down to find the global minimum
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42,
    tree_method='hist'
)

lgb_model = LGBMClassifier(
    n_estimators=1500, 
    learning_rate=0.015, 
    num_leaves=31, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42, 
    verbose=-1
)

print("Training XGBoost...")
xgb_model.fit(X, y)
print("Training LightGBM...")
lgb_model.fit(X, y)

# 5. Simple Weighted Blend (0.7 / 0.3 Ratio)
p1 = xgb_model.predict_proba(X_test)[:, 1]
p2 = lgb_model.predict_proba(X_test)[:, 1]

final_prob = (p1 * 0.7) + (p2 * 0.3)

# 6. Submission
submission = pd.DataFrame({'id': test[ID_COL], 'diagnosed_diabetes': final_prob})
submission.to_csv('submission_final_fix.csv', index=False)
print("✅ Created: submission_final_fix.csv - Good luck!")