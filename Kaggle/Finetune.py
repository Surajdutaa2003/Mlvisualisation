import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Paths
# -----------------------
TRAIN_PATH = "Dataset/trainFinetune.csv"
TEST_PATH  = "Dataset/testfinetune.csv"

# -----------------------
# Load Data
# -----------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# -----------------------
# Text Builder (Bias Safe)
# -----------------------
def build_text(df):
    a = df["prompt"].fillna("") + " " + df["response_a"].fillna("")
    b = df["prompt"].fillna("") + " " + df["response_b"].fillna("")
    return a, b

train_a, train_b = build_text(train_df)
test_a, test_b   = build_text(test_df)

# -----------------------
# Encode Labels
# -----------------------
# The train set has winner_model_a, winner_model_b, winner_tie as binary labels
y = train_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

# -----------------------
# TF-IDF Vectorizer
# -----------------------
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_a = tfidf.fit_transform(train_a)
X_b = tfidf.transform(train_b)

# Difference feature removes position bias
X = X_a - X_b

# -----------------------
# Train / Validation Split
# -----------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42
)

# -----------------------
# Model
# -----------------------
# Use MultiOutputClassifier for multiple binary outputs
base_model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    n_jobs=-1
)

model = MultiOutputClassifier(base_model, n_jobs=-1)
model.fit(X_tr, y_tr)

# -----------------------
# Validation Check
# -----------------------
val_preds = model.predict_proba(X_val)
# Extract probabilities for class 1 from each output
val_preds_array = np.column_stack([pred[:, 1] for pred in val_preds])
val_loss = log_loss(y_val, val_preds_array)
print(f"Validation Log Loss: {val_loss:.4f}")

# -----------------------
# Train on Full Data
# -----------------------
model.fit(X, y)

# -----------------------
# Test Predictions
# -----------------------
X_test = tfidf.transform(test_a) - tfidf.transform(test_b)
test_preds = model.predict_proba(X_test)

# Extract probabilities for class 1 from each output
test_preds_array = np.column_stack([pred[:, 1] for pred in test_preds])

# -----------------------
# Submission File
# -----------------------
submission = pd.DataFrame({
    "id": test_df["id"],
    "winner_model_a": test_preds_array[:, 0],
    "winner_model_b": test_preds_array[:, 1],
    "winner_tie": test_preds_array[:, 2]
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")

