import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# -----------------------------
# 1. Load CSV dataset
# -----------------------------
df = pd.read_csv(r'Dataset\apples_and_oranges_Classification.csv')

print(df.head())
print(df.info())

# -----------------------------
# 2. Encode labels (Apple=0, Orange=1)
# -----------------------------
df['Class'] = df['Class'].map({
    'apple': 0,
    'orange': 1
})

# -----------------------------
# 3. Separate features & target
# -----------------------------
X = df.drop(columns=['Class'])
y = df['Class']

# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# -----------------------------
# 5. Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6. LightGBM Model
# -----------------------------
lgbm_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=7,
    max_depth=3,
    min_child_samples=3,  # Lowered for small dataset (default=20)
    random_state=42,
    verbose=-1  # Suppress warnings
)

lgbm_model.fit(X_train, y_train)  # Use unscaled data - LightGBM doesn't need scaling

# -----------------------------
# 7. Accuracy
# -----------------------------
y_pred = lgbm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# -----------------------------
# 8. Predict new fruit (example)
# -----------------------------
new_fruit = pd.DataFrame([[280, 8.3]], columns=X.columns)
prediction = lgbm_model.predict(new_fruit)
probability = lgbm_model.predict_proba(new_fruit)[0, 1]

print(f"Prediction: {int(prediction[0])} (0=Apple, 1=Orange)")
print(f"Probability of Orange: {probability:.4f}")
