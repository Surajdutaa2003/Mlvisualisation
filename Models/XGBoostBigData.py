# ==============================
# XGBoost - Apple vs Orange (Bigger Dataset)
# ==============================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

# -----------------------------
# 1. Bigger synthetic dataset
# -----------------------------
np.random.seed(42)

# Apples (lighter & smaller)
apple_weights = np.random.normal(loc=160, scale=15, size=60)
apple_sizes   = np.random.normal(loc=7.0, scale=0.3, size=60)
apples = np.column_stack((apple_weights, apple_sizes))
apple_labels = np.zeros(60)

# Oranges (heavier & bigger)
orange_weights = np.random.normal(loc=300, scale=25, size=60)
orange_sizes   = np.random.normal(loc=8.8, scale=0.4, size=60)
oranges = np.column_stack((orange_weights, orange_sizes))
orange_labels = np.ones(60)

# Combine
X = np.vstack((apples, oranges))
y = np.concatenate((apple_labels, orange_labels))

# -----------------------------
# 2. Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Train-Test Split (BIG test set)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42
)

# -----------------------------
# 4. XGBoost Model
# -----------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# -----------------------------
# 5. Accuracy
# -----------------------------
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# -----------------------------
# 6. Predict new fruit
# -----------------------------
new_fruit = np.array([[280, 8.3]])
new_fruit_scaled = scaler.transform(new_fruit)

prediction = xgb_model.predict(new_fruit_scaled)
probability = xgb_model.predict_proba(new_fruit_scaled)[0, 1]

print(f"Prediction: {prediction[0]} (0=Apple, 1=Orange)")
print(f"Probability of Orange: {probability:.4f}")
