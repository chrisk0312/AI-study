import numpy as np
from keras.datasets import mnist
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D




#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)



#print(x_train)
print(x_train[0])
print(y_train[0]) # 5
print(np.unique(y_train, return_counts= True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
print(pd.value_counts(y_test))


# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

# print(x_train.shape[0]) #60000


x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

print(x_train.shape, x_test.shape) #(60000, 784) (10000, 784)


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)



#2.모델
model = Sequential()
model.add(Conv1D(10, kernel_size=2, input_shape=(28,28), activation= 'relu'))
model.add(Flatten())
model.add(Dense(10,activation= 'relu'))
model.add(Dense(10,activation= 'softmax'))









# model.add(Conv2D(8, (2,2), strides=2, padding='same',
#                  input_shape= (10, 10, 1),)) #첫 아웃풋 = filter
# # padding='valid'- 디폴트./
# # 모양유지- padding= 'same' (전사이즈 유지됨.)
# # stride 끝에 남은 데이터는 제거된다.
# #kernel_size 디폴트값-1
# # shape = (batch_size, rows, columns, channels)
# # shape = (batch_size, heights, widths, channels)
# model.add(MaxPooling2D()) #엔빵!
# model.add(Conv2D(filters=10, kernel_size=(2,2)))
# model.add(Conv2D(15, (4,4))) 
# model.add(Flatten()) #(n,22*22*15)의 모양을 펴져있는 모양으로 변형.
# model.add(Dense(units=8))#나가는 출력값
# model.add(Dense(7, input_shape=(8,)))
# # shape = (batch_size(=model.fit의 batch_size와 같음.), input_dim) 
# model.add(Dense(6))
# model.add(Dense(5, activation='swish'))
# model.add(Dense(10, activation= 'softmax'))

model.summary()

#batch_size는 전체 행에서 원하는만큼 행을 나눠 훈련) 레이어별로 다르게 지정 불가. (SHAPE ERROR 뜸.)
# model.summary()
# (아웃풋수 * 커널사이즈 * 커널갯수 )+ bias = Param
# (channels * kernel size + bias) * 9 
#_________________________________________________________________     
#  Layer (type)                Output Shape              Param #        
# =================================================================     
#  conv2d (Conv2D)             (None, 27, 27, 9)         45
#(kernel size(4) * channels(1) + bias(1)) * (filter)(9) = 45
#  conv2d_1 (Conv2D)           (None, 25, 25, 10)        820
#(9 * 9 + 1) * 10 = 820
#  conv2d_2 (Conv2D)           (None, 22, 22, 15)        2415
#(16 * 10 + 1) * 15 = 2415
#  flatten (Flatten)           (None, 7260)              0

#  dense (Dense)               (None, 8)                 58088

#  dense_1 (Dense)             (None, 7)                 63

#  dense_2 (Dense)             (None, 6)                 48

#  dense_3 (Dense)             (None, 5)                 35

#  dense_4 (Dense)             (None, 10)                60

# =================================================================
# Total params: 61,574
# Trainable params: 61,574
# Non-trainable params: 0
# _________________________________________________________________






#3. 컴파일, 훈련
model.compile( loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
model.fit( x_train, y_train, batch_size=256, verbose=1, epochs= 100, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])

# #padding/
# loss: 0.1907665878534317
# acc: 0.9490000009536743
# 걸린시간 : 56.741090059280396 초


# #MaxPooling
# loss: 0.08347584307193756
# acc: 0.9758999943733215
# 걸린시간 : 36.64556956291199 초

# #Dnn
# loss: 0.304256796836853
# acc: 0.916100025177002



# LSTM
# loss: 2.167607307434082
# acc: 0.2402999997138977