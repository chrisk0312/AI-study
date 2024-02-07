import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#1
datasets = load_wine()
x = datasets.data
y= datasets['target']
print(x.shape, y.shape) #(178, 13) (178,)
print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))
print(y)

x= x[:-35]
y= y[:-35]
print(y)
print(np.unique(y, return_counts=True)) 
#(array([0, 1, 2]), array([59, 71,  13], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.75, shuffle= True, random_state=123,
    stratify=y, 
)


'''
#2 
model=Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train,y_train, epochs=50, validation_split=0.2)

#4
results = model.evaluate(x_test, y_test)
print("loss:", results[0])
print("acc:", results[1])

y_pred = model.predict(x_test)
print(y_test)
print(y_pred)

y_pred = np.argmax(y_pred, axis=1)
print(y_pred)

acc= accuracy_score(y_test, y_pred)
print('acc',acc)
f1 = f1_score(y_test, y_pred, average='weighted')
print('f1',f1)

# acc 0.8055555555555556
# f1 0.7676470588235293
'''
##################### smote ########################
print("==================smote 적용==========================")
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('sklearn:', sk.__version__) #sklearn: 1.3.0

smote = SMOTE(random_state=123)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.value_counts(y_train))
# 0    53
# 1    53
# 2    53

#2 
model=Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train,y_train, epochs=50, validation_split=0.2)

#4
results = model.evaluate(x_test, y_test)
print("loss:", results[0])
print("acc:", results[1])

y_pred = model.predict(x_test)
print(y_test)
print(y_pred)

y_pred = np.argmax(y_pred, axis=1)
print(y_pred)

acc= accuracy_score(y_test, y_pred)
print('acc',acc)
f1 = f1_score(y_test, y_pred, average='weighted')
print('f1',f1)

# acc 0.8611111111111112
# f1 0.8230574324324325