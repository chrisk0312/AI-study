import numpy as np  
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.models import Sequential

#1 data
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])
print(x_data.shape, y_data.shape) #(4, 2) (4,)

#2 model
# model = LinearSVC()
# model = Perceptron()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))


#3 fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=100, batch_size=1)

# #4 evaluate, predict
# acc = model.score(x_data, y_data)
# print(acc)
results = model.evaluate(x_data, y_data)    
print('acc : ', results[1])

y_pred = model.predict(x_data)
y_pred = np.round(y_pred).reshape(-1,).astype(int)
acc2 = accuracy_score(y_data, y_pred)
print(acc2)

print("=====================================")
print(y_data)
print(y_pred)
