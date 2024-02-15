import numpy as np  
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


#1 data
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 0, 0, 1])
print(x_data.shape, y_data.shape) #(4, 2) (4,)

#2 model
model = Perceptron()

# A perceptron is a simple machine learning algorithm that works well for binary classification tasks. 
# It's a type of linear classifier, meaning it makes predictions based on a linear predictor function combining a set of weights with the feature vector.
# The perceptron takes in an input, multiplies it by a weight, and then sums the results. 
# If the sum is greater than a certain threshold, the perceptron returns 1; otherwise, it returns 0.
# During training, the perceptron is provided with examples for which the output is known.
# It makes predictions for these examples, and then adjusts its weights based on the errors it made.

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
