import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------
# Perfect Combined Dataset (Realistic Apples & Oranges)
# -------------------------------------------------------

np.random.seed(42)

# Apples (20 samples)
apple_weights = np.random.normal(loc=150, scale=10, size=20)   # around 150g ±10
apple_sizes   = np.random.normal(loc=7.0, scale=0.3, size=20)  # around 7cm ±0.3
apples = np.column_stack((apple_weights, apple_sizes))
apples_label = np.zeros(20)

# Oranges (20 samples)
orange_weights = np.random.normal(loc=260, scale=20, size=20)  # around 260g ±20
orange_sizes   = np.random.normal(loc=8.8, scale=0.4, size=20) # around 8.8cm ±0.4
oranges = np.column_stack((orange_weights, orange_sizes))
oranges_label = np.ones(20)

# Combine dataset
X = np.vstack((apples, oranges))
y = np.hstack((apples_label, oranges_label))

# -------------------------------------------------------
# Train-test split with class balance (stratify)
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -------------------------------------------------------
# Scale features
# -------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# -------------------------------------------------------
# Logistic Regression model
# -------------------------------------------------------
clf = LogisticRegression()
clf.fit(X_train_s, y_train)

# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------
y_pred = clf.predict(X_test_s)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Cleaner, readable confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Actual vs Predicted):")
print("                Predicted")
print("              |   Apple (0)   |   Orange (1)")
print("------------------------------------------------")
print(f"Actual Apple  |      {cm[0,0]}         |       {cm[0,1]}")
print(f"Actual Orange |      {cm[1,0]}         |       {cm[1,1]}")

# -------------------------------------------------------
# Predict new fruit
# -------------------------------------------------------
sample = np.array([[280, 8.3]])
sample_s = scaler.transform(sample)

print("\nPrediction (0=apple,1=orange):", clf.predict(sample_s)[0])
print("Probability of orange:", clf.predict_proba(sample_s)[0,1])
