# -*- coding: utf-8 -*-
"""
Project 15: Titanic Survival Prediction using Machine Learning
Improved & Industry-Ready Version
"""

# Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==============================
# Data Collection & Processing
# ==============================

titanic_data = pd.read_csv('Dataset/train.csv')

# Basic info
print(titanic_data.info())
print(titanic_data.isnull().sum())

# ==============================
# Handling Missing Values
# ==============================

# Drop Cabin (too many missing values)
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# Fill Age with mean
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())

# Fill Embarked with mode
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(
    titanic_data['Embarked'].mode()[0]
)

# ==============================
# Encoding Categorical Columns
# ==============================

titanic_data.replace(
    {'Sex': {'male': 0, 'female': 1},
     'Embarked': {'S': 0, 'C': 1, 'Q': 2}},
    inplace=True
)

# ==============================
# Feature & Target Separation
# ==============================

X = titanic_data.drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1
)
Y = titanic_data['Survived']

# ==============================
# Train-Test Split
# ==============================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# ==============================
# Feature Scaling (IMPORTANT)
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)   # fit ONLY on train
X_test = scaler.transform(X_test)          # transform test

# ==============================
# Model Training
# ==============================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# ==============================
# Model Evaluation
# ==============================

# Training accuracy
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_pred)
print('Training Accuracy:', train_accuracy)

# Test accuracy
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_pred)
print('Test Accuracy:', test_accuracy)
