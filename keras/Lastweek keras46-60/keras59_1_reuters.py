from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)

print(x_train)
print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)
print(type(x_train)) 
print(y_train) #[ 3  4  3 ... 25  3 25]
print(len(np.unique(y_train))) # 46
print(len(np.unique(y_test))) # 46

print(type(x_train)) # <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'>
print(len(x_train[0]), len(x_train[1])) # 87 56

print ("뉴스기사의 최대길이",max(len(i) for i in x_train)) # 2376 최대길이
print ("뉴스기사의 평균길이",sum(map(len,x_train))/ len(x_train)) # 145.5398574927633 평균길이

#전처리
from keras.utils import pad_sequences

x_train = pad_sequences(x_train, maxlen=100, padding='pre', truncating='pre') # pre: 앞에서부터 0을 채움
x_test = pad_sequences(x_test, maxlen=100, padding='pre', truncating='pre') # pre: 앞에서부터 0을 채움  # truncating: 100개만 남기고 나머지는 자름

print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

# 모델 정의
model = Sequential()
model.add(Embedding(10000, 128, input_length=100))
model.add(LSTM(128))
model.add(Dense(46, activation='softmax'))

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


loss= model.evaluate(x_test, y_test) 
print('loss',loss[0])#  loss: 1.3675 - acc: 0.7013
print('acc',loss[1])#acc: 0.7012885808944702
y_pred = model.predict(x_test)
print(y_pred) # 46개의 카테고리에 대한 확률값이 나옴
result = np.argmax(y_pred,axis=-1)
print('예측값',result) # 46개의 카테고리 중에 하나로 분류됨