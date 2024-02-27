from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import time
from sklearn.preprocessing import OneHotEncoder


#1. 데이터

datasets = load_wine()
print(datasets.DESCR) 
print(datasets.feature_names) #'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'       


x= datasets.data
y= datasets.target
print(x.shape, y.shape) #(178, 13) (178,)

print(np.unique(y, return_counts= True)) #(array([0, 1, 2]), array([59, 71, 48]
print(pd.value_counts(y)) # 1    71 # 0    59 # 2    48

x= x.reshape(178,13,1,1)
#y = pd.get_dummies(y)
#y = to_categorical(y)
y= y.reshape(-1,1)
ohe = OneHotEncoder(sparse= False)
y = ohe.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, shuffle= True, random_state= 11, stratify= y)
print(x_train.shape, x_test.shape) #(142, 13) (36, 13)
print(y_train.shape, y_test.shape) #(142, 3) (36, 3)
print(np.unique(y_test, return_counts= True))

#2. 모델구성

model = Sequential()
model.add(Conv2D(50, (2,1), input_shape=(13,1,1), padding= 'same'))
model.add(Conv2D(20,(2,1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(3, activation= 'softmax'))

#3. 컴파일, 훈련
model.compile (loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
es = EarlyStopping(monitor= 'val_loss', mode= 'min', patience=100, verbose=2, restore_best_weights= True)
start_time = time.time()
his = model.fit(x_train, y_train, epochs=300, batch_size=1, validation_split= 0.2, verbose=2 )
end_time = time.time()

#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print("로스 :", results[0])
print("정확도 :", results[1])

y_predict = model.predict(x_test)
print(y_predict.shape, y_test.shape)

y_test = np.argmax(y_test, axis=1) #아그맥스는 위치값을 빼주는 함수
y_predict = np.argmax(y_predict, axis=1)
result = accuracy_score(y_test, y_predict)

acc = accuracy_score(y_predict, y_test)
print("acc :", acc)

print("걸린 시간 :", round(end_time - start_time, 2), "초")

# 로스 : 0.027650095522403717
# 정확도 : 1.0

#cnn
# acc : 1.0
#걸린 시간 : 45.54 초