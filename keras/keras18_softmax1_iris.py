from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing

#data
datasets = load_iris()
# print(datasets.DESCR) #(n,4)
# :Number of Instances: 150 (50 in each of three classes)
# :Number of Attributes: 4 numeric, predictive attributes and the class
# :Attribute Information:
# ============== ==== ==== ======= ===== ====================
#                 Min  Max   Mean    SD   Class Correlation
# ============== ==== ==== ======= ===== ====================
# sepal length:   4.3  7.9   5.84   0.83    0.7826
# sepal width:    2.0  4.4   3.05   0.43   -0.4194
# petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
# petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
# :Class Distribution: 33.3% for each of 3 classes.

# print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data
y = datasets.target
# y = to_categorical(y)                                       #keras를 이용한 one hot encording
# y = pd.get_dummies(y)                                     #pandas를 이용한 one hot encording
# label_binarizer = sklearn.preprocessing.LabelBinarizer()  #sklearn를 이용한 one hot encording(3줄)
# label_binarizer.fit(range(max(y)+1))
# y = label_binarizer.transform(y)
y = y.reshape(-1,1)                                         #sklearn를 이용한 one hot encordeing 주된 방법
ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)     
y = ohe.fit_transform(y)

# print(x)
print(f"{x.shape=}, {y.shape=}")        #x.shape=(150, 4), y.shape=(150,)
# print(np.unique(y, return_counts=True)) #array([0, 1, 2]), array([50, 50, 50])

r = int(np.random.uniform(1,1000))
r=326
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r,stratify=y)

#model
model = Sequential()
model.add(Dense(128,input_dim=4,activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8,  activation='relu'))
model.add(Dense(3,  activation='softmax'))

#compile & fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(monitor='val_acc',mode='max',patience=30,restore_best_weights=True,verbose=1)
hist = model.fit(x_train,y_train,epochs=1024,batch_size=1,validation_split=0.4,verbose=2,callbacks=[es])

#evaluate & predict
loss = model.evaluate(x_test,y_test,verbose=0)
# y_predict = np.around(model.predict(x_test))  이렇게 하면 0.3 0.4 0.3 같은 경우를 처리하지 못함
y_test = np.argmax(y_test,axis=1)                   #둘다 (n,)의 백터형태로 출력되므로 accuracy_score가능
y_predict = np.argmax(model.predict(x_test),axis=1)
# print(y_predict)

#결과값 출력
print(f"{r=}\nLOSS: {loss[0]}\nACC:  {accuracy_score(y_test,y_predict)}({loss[1]}by loss[1])")

#테스트 잘 분리되었는지 확인 및 예측결과값 비교
print(np.unique(y_test,return_counts=True))
print(np.unique(y_predict,return_counts=True))

#그래프 출력
plt.figure(figsize=(12,9))
plt.title("Iris Classification")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(hist.history['acc'],label='acc',color='red')
plt.plot(hist.history['val_acc'],label='val_acc',color='blue')
plt.legend()
plt.show()

# Epoch 312: early stopping
# 1/1 [==============================] - 0s 62ms/step
# r=167
# LOSS: 0.05837009474635124
# ACC:  1.0

#stratify 적용 후

# patience = 30
# Epoch 43: early stopping
# 1/1 [==============================] - 0s 52ms/step
# r=326
# LOSS: 0.0691366046667099
# ACC:  1.0(1.0by loss[1])
# y_test's    contents: 0=10, 1=10, 2=10
# y_predict's contents: 0=10, 1=10, 2=10