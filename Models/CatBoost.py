import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.DataFrame({
    'weight': [150, 170, 140, 200, 250, 300, 330, 360],
    'size':   [7.0, 7.5, 6.8, 8.0, 8.0, 8.5, 9.0, 9.5],
    'color':  ['red', 'red', 'red', 'orange', 'orange', 'orange', 'orange', 'orange'],  # categorical!
    'label':  [0, 0, 0, 1, 1, 1, 1, 1]  # 0=Apple, 1=Orange
})


X=data.drop('label',axis=1)
y=data['label']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

cat_feature=['color']


cat_model=CatBoostClassifier(
    iterations=100,
    depth=4,
    learning_rate=0.1,
    random_seed=42,
    verbose=False
)


cat_model.fit(X_train,y_train,cat_features=cat_feature)


y_pred=cat_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy:.4f}")

new_fruit = pd.DataFrame({
    'weight': [280],
    'size':   [8.3],
    'color':  ['orange']
})

prediction = cat_model.predict(new_fruit)
probability = cat_model.predict_proba(new_fruit)[0, 1]

print(f"Prediction: {int(prediction[0])} (0=Apple, 1=Orange)")
print(f"Probability of Orange: {probability:.4f}")