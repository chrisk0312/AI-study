
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Conv2D ,Flatten , MaxPooling2D , Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import time


#1 데이터

start_time = time.time()




np_path = "C:\\_data\\_save_npy\\"
# np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])

x_train = np.load(np_path + 'keras39_1_x_train.npy')
y_train = np.load(np_path + 'keras39_1_y_train.npy')
x_test = np.load(np_path + 'keras39_1_x_test.npy')
y_test= np.load(np_path + 'keras39_1_y_test.npy')

print(x_train.shape)


#2 모델구성
model = Sequential()
model.add(Conv2D(64,(2,2),input_shape = (100, 100, 1) , strides=1 , activation='relu' ))
model.add(Dropout(0.1))
model.add(Conv2D(32,(2,2), activation='relu' , padding = 'same'))
model.add(Conv2D(12,(2,2), activation='relu' ))
model.add(Conv2D(12,(3,3), activation= 'relu' ))
model.add(Flatten())
model.add(Dense(40,activation='relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# model.summary()

#3 컴파일, 훈련
filepath = "C:\_data\_save\MCP\_k37\_Brain"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

#es = EarlyStopping(monitor='val_loss' , mode = 'auto' , patience= 200 , restore_best_weights=True , verbose= 1  )
#mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)


model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
#model.fit_generator(xy_train, epochs = 1000 , validation_data= xy_test, verbose= 2 ,callbacks=[es, mcp])
#fit과 동일. directoryInterator 형태를 x,y로 나누지 않아도 통으로 받아줌
#fit_generaton는 batch_size를 받지 않음. 위에서 나눈 batch_size를 받아준다.
#validation_split도 받지않음. validation)data만 받아줌.
#fit = fit_generator와 동일. generator는 곧 없어질예정.
model.fit(x_train, y_train, epochs = 300 , batch_size=16, validation_split= 0.2, verbose= 2 ,)#callbacks=[es, mcp])
#^^^batch_size를 명시해도 먹히지않음.
#validatuon_split 에러.
#steps_per_epochs= 전체데이터 / batch = 160 / 10 = 16 으로 잡아라 (15~1까지는 남는데이터는 손실) 

end_time = time.time()

#4 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , round(end_time - start_time, 2), "초" )



# loss: 0.05903565138578415
# acc: 0.9833333492279053
# 걸린시간 : 13.61 초
