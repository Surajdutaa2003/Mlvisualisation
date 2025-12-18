import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# Toy dataset:hours studied vs score
X=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y=np.array([35,40,45,50,55,60,65,70,75,80])


# Split the data into training and testing sets

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
    
)
# model
model=LinearRegression()
model.fit(X_train,y_train)

# evaluate
y_pred=model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


# single prediction
hours=np.array([6.5])
pred_score=model.predict(hours.reshape(1, -1))[0]
print(f"Predicted score for 6.5 hours: {pred_score:.2f}")




