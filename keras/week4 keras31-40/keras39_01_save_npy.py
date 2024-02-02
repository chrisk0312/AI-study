
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

#BATCH_SIZE = int(200)



train_datagen = ImageDataGenerator(rescale=1./255,)
                                #    horizontal_flip= True ,
                                #    vertical_flip = True ,
                                #    width_shift_range = 0.1,
                                #    height_shift_range = 0.1,
                                #    rotation_range = 5 ,
                                #    zoom_range = 1.2,
                                #    shear_range = 0.8 ,
                                #    fill_mode = 'nearest')


test_datagen = ImageDataGenerator(rescale=1./255) #데이터 수치화 도구! #수치가 같아야 훈련이 가능하기에 testdata도 똑같이 수치화는 해줘야함.

path_train = 'C:\\_data\\image\\brain\\train\\' #path 하단 폴더는 라벨(y값)로 지정됨.
path_test = 'C:\\_data\\image\\brain\\test\\' 

start_time2 = time.time()
xy_train = train_datagen.flow_from_directory(path_train, 
                                             target_size = (100,100), #원본데이터보다 작아질수록 성능이많이 떨어짐. 최대한 원본과 사이즈를 맞춰주는게 좋음.
                                             batch_size = 200,
                                             class_mode='binary', 
                                             color_mode= 'grayscale',
                                             shuffle=True)

xy_test = test_datagen.flow_from_directory(path_test, 
                                             target_size = (100,100), #원본데이터보다 작아질수록 성능이많이 떨어짐. 최대한 원본과 사이즈를 맞춰주는게 좋음.
                                             batch_size = 200,
                                             class_mode='binary', 
                                             color_mode= 'grayscale',
                                             shuffle=True)


print(xy_train)

print(xy_train[0][0].shape) # (160, 100, 100, 1)
print(xy_train[0][1].shape) # (160,)

print(xy_test[0][0].shape) # (120, 100, 100, 1)
print(xy_test[0][1].shape) # (120, 100, 100, 1)



end_time2 = time.time()


np_path = "C:\\_data\\_save_npy\\"
np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])



'''
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
model.fit(xy_train, epochs = 10 , steps_per_epoch=16, validation_data= xy_test, verbose= 2 ,)#callbacks=[es, mcp])
#^^^batch_size를 명시해도 먹히지않음.
#validatuon_split 에러.
#steps_per_epochs= 전체데이터 / batch = 160 / 10 = 16 으로 잡아라 (15~1까지는 남는데이터는 손실) 

end_time = time.time()

#4 평가, 예측
results = model.evaluate(xy_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )



print('걸린시간 : ' , round(end_time - start_time,2), "초" )
print('변환시간 : ' , round(end_time2 - start_time2,2), "초" )

'''

