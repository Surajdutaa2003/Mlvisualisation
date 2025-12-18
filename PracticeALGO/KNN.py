# knn_example.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# same toy X,y (weight,size) -> apple/orange
X = np.array([
    [150, 7], [170, 7.5], [140, 6.8],
    [200, 8], [250, 8], [300, 8.5],
    [330, 9], [360, 9.5]
])
y = np.array([0,0,0,1,1,1,1,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)  # k=3 commonly
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

print("Accuracy:", accuracy_score(y_test, y_pred))

# single predict
sample = scaler.transform([[280, 8.3]])
print("KNN predicts:", knn.predict(sample)[0])
