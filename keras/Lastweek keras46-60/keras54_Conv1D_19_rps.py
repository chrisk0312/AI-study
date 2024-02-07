import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils import to_categorical

#1. 데이터

np_path = "C:\\_data\\_save_npy\\"

x = np.load( np_path + 'rps_x.npy')
y = np.load( np_path + 'rps_y.npy')
x = x/255.

train_datagen = ImageDataGenerator(
    #horizontal_flip=True,
    #vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #rotation_range=30,
    zoom_range=1.2,
    shear_range=0.2,
    fill_mode='nearest'
)

augumet_size = 5000

randidx = np.random.randint(x.shape[0],size=augumet_size)

x_augumented = x[randidx].copy()
y_augumented = y[randidx].copy()

x_augumented = x_augumented.reshape(-1,450,150)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augumet_size,
    shuffle= False       
).next()[0]

x = x.reshape(-1, 450, 150)

x = np.concatenate((x, x_augumented))
y = np.concatenate((y, y_augumented))


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, random_state=234, shuffle=True, stratify= y)

print(x_train.shape) #(2016, 150, 150, 3)


#2. 모델구성
model = Sequential()
model.add(Conv1D(12,kernel_size=2, input_shape = (450,150) , activation='relu' ))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(3,activation='softmax'))

#model.summary()


#3 컴파일, 훈련
filepath = "c:\\_data\\_save\\MCP\\_k39\\rps"

es = EarlyStopping(monitor='val_loss', mode= 'auto', patience= 30, restore_best_weights= True, verbose= 1)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only= True, filepath= filepath )

model.compile(loss= 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
hist = model.fit(x_train,y_train, epochs = 100 , batch_size= 128 , validation_split= 0.2, verbose= 2 ,callbacks=[es, mcp])



#4. 평가, 예측

result = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict)

print("로스 :", result[0])
print("정확도 :", result[1])


#로스 : 3.96894829464145e-05
#정확도 : 1.0

