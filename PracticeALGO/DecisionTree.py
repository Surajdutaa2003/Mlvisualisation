import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt

X = np.array([
    [150, 7], [170, 7.5], [140, 6.8],
    [200, 8], [250, 8], [300, 8.5],
    [330, 9], [360, 9.5]
])
y = np.array([0,0,0,1,1,1,1,1])


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

clf=DecisionTreeClassifier(max_depth=3,random_state=42)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

plt.figure(figsize=(6,4))
plot_tree(clf,feature_names=["weight","size"],class_names=["apple","orange"],filled=True)
plt.show()
