import numpy as np  
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


#1 data
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])
print(x_data.shape, y_data.shape) #(4, 2) (4,)

#2 model
model = SVC()
# model = Perceptron()

#3 fit
model.fit(x_data, y_data)

#4 evaluate, predict
acc = model.score(x_data, y_data)
print(acc)

y_pred = model.predict(x_data)
acc2 = accuracy_score(y_data, y_pred)
print(acc2)

print("=====================================")
print(y_data)
print(y_pred)
