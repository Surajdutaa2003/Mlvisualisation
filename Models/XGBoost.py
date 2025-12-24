# ==============================
# XGBoost Simple Example - Apple vs Orange
# ==============================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb   # <-- Yeh import karna zaroori

# -----------------------------
# 1. Simple data (weight, size) → 0=Apple, 1=Orange
# -----------------------------
X = np.array([
    [150, 7], [170, 7.5], [140, 6.8],
    [200, 8], [250, 8], [300, 8.5],
    [330, 9], [360, 9.5]
])
y = np.array([0,0,0,1,1,1,1,1])  # 0=Apple, 1=Orange

# -----------------------------
# 2. Scaling (XGBoost ko pasand hai)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# -----------------------------
# 4. XGBoost Model
# -----------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=100,      # 100 trees
    max_depth=3,           # chhota tree
    learning_rate=0.1,     # normal speed
    random_state=42
)

xgb_model.fit(X_train, y_train)

# -----------------------------
# 5. Prediction aur Accuracy
# -----------------------------
y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # 1.0 aayega (perfect!)

# -----------------------------
# 6. Naya fruit predict karo
# -----------------------------
new_fruit = np.array([[280, 8.3]])  # mota orange jaisa
new_fruit_scaled = scaler.transform(new_fruit)

prediction = xgb_model.predict(new_fruit_scaled)
probability = xgb_model.predict_proba(new_fruit_scaled)[0, 1]

print(f"Prediction: {prediction[0]} (0=Apple, 1=Orange)")
print(f"Probability of Orange: {probability:.4f}")
