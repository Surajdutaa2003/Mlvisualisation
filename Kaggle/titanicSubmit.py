# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

# # ==============================
# # Load data
# # ==============================
# train_data = pd.read_csv("Dataset/train.csv")
# test_data  = pd.read_csv("Dataset/test.csv")

# # ==============================
# # Combine for common preprocessing
# # ==============================
# full_data = [train_data, test_data]

# # ==============================
# # Handle missing values
# # ==============================
# for df in full_data:
#     df.drop(columns="Cabin", inplace=True)
#     df["Age"].fillna(df["Age"].mean(), inplace=True)
#     df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# # ==============================
# # Feature Engineering
# # ==============================

# # ---- Title from Name
# for df in full_data:
#     df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

# rare_titles = ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"]
# for df in full_data:
#     df["Title"] = df["Title"].replace(rare_titles, "Rare")
#     df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})

# title_map = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
# for df in full_data:
#     df["Title"] = df["Title"].map(title_map).fillna(0)

# # ---- Family features
# for df in full_data:
#     df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
#     df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# # ---- Age & Fare binning
# train_data["AgeBin"] = pd.cut(train_data["Age"], 5, labels=False)
# test_data["AgeBin"]  = pd.cut(test_data["Age"], 5, labels=False)

# train_data["FareBin"] = pd.qcut(train_data["Fare"], 4, labels=False)
# test_data["FareBin"]  = pd.qcut(test_data["Fare"], 4, labels=False)

# # ==============================
# # Encode categorical columns
# # ==============================
# for df in full_data:
#     df.replace({
#         "Sex": {"male": 0, "female": 1},
#         "Embarked": {"S": 0, "C": 1, "Q": 2}
#     }, inplace=True)

# # ==============================
# # Prepare Train / Test
# # ==============================
# X_train = train_data.drop(
#     columns=["PassengerId","Name","Ticket","Survived"]
# )
# Y_train = train_data["Survived"]

# X_test = test_data.drop(
#     columns=["PassengerId","Name","Ticket"]
# )

# passenger_ids = test_data["PassengerId"]

# # ==============================
# # Train Improved Random Forest
# # ==============================
# model = RandomForestClassifier(
#     n_estimators=400,
#     max_depth=10,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     max_features="sqrt",
#     random_state=42
# )

# model.fit(X_train, Y_train)

# # ==============================
# # Predict & Submit
# # ==============================
# predictions = model.predict(X_test)

# submission = pd.DataFrame({
#     "PassengerId": passenger_ids,
#     "Survived": predictions
# })

# submission.to_csv("submission.csv", index=False)

# print("✅ submission.csv created successfully")
# # 2ndtry


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# ==============================
# Load data
# ==============================
train_df = pd.read_csv("Dataset/train.csv")
test_df  = pd.read_csv("Dataset/test.csv")

test_ids = test_df["PassengerId"]

# ==============================
# Combine for feature engineering
# ==============================
full = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# ==============================
# Feature Engineering
# ==============================

# ---- Title
full["Title"] = full["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
full["Title"] = full["Title"].replace(
    ["Lady","Countess","Capt","Col","Don","Dr",
     "Major","Rev","Sir","Jonkheer","Dona"],
    "Rare"
)
full["Title"] = full["Title"].replace(
    {"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"}
)

# ---- Family features
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# ---- Deck from Cabin
full["Deck"] = full["Cabin"].str[0].fillna("Unknown")

# ---- Fare per person
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["FarePerPerson"] = full["Fare"] / full["FamilySize"]

# ---- Age
full["Age"] = full["Age"].fillna(full["Age"].median())

# ---- Embarked
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# ==============================
# Drop unused columns
# ==============================
full.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin"],
    inplace=True
)

# ==============================
# Split back
# ==============================
train = full.iloc[:len(train_df)]
test  = full.iloc[len(train_df):]

X_train = train.drop(columns=["Survived"])
y_train = train["Survived"]

X_test = test.drop(columns=["Survived"])

# ==============================
# Categorical features (CatBoost magic)
# ==============================
cat_features = [
    X_train.columns.get_loc(col)
    for col in ["Sex", "Embarked", "Title", "Deck"]
]

# ==============================
# CatBoost Model (strong config)
# ==============================
model = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=0
)

model.fit(X_train, y_train, cat_features=cat_features)

# ==============================
# Predict & Submit
# ==============================
preds = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": preds.astype(int)
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully")
