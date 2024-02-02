from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import pandas as pd


#1. 데이터
(x_train, y_train), (x_test, y_test)= fashion_mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.


train_datagen = ImageDataGenerator(
    #rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest', #nearest가 디폴트 #reflet 좌우반전,wrap 감싸줌, 

)


augumet_size = 40000

randidx = np.random.randint(x_train.shape[0],size=augumet_size)
        #np.random.randint(60000, 40000)  앞에서 뒤의 갯수만큼 뽑아내라.
#print(randidx) #[20043 58915  1538 ... 37662 37240 10207]
#print(np.min(randidx), np.max(randidx))   #0 59998



x_augumented = x_train[randidx].copy() #원데이터에 영향을 미치지 않기위해 별도의 메모리 공간을 생성.
y_augumented = y_train[randidx].copy() #원데이터에 영향을 미치지 않기위해 별도의 메모리 공간을 생성.
#print(x_augumented)
#print(x_augumented.shape) #(40000, 28, 28)
#print(y_augumented.shape) #(40000, )

x_augumented = x_augumented.reshape(-1,28,28,1)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augumet_size,
    shuffle= False, #섞이면 데이터가틀어짐.
    save_to_dir='c:/_data/temp/'
).next()[0]

#print(x_augumented)
#print(x_augumented.shape)

#print(x_train.shape)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = np.concatenate((x_train,x_augumented))
y_train = np.concatenate((y_train,y_augumented))
#print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

#2. 모델구성

model = Sequential()
model.add(Conv2D(10, (2,2),
                 input_shape= (28, 28, 1), activation= 'swish')) #첫 아웃풋 = filter
# shape = (batch_size, rows, columns, channels)
# shape = (batch_size, heights, widths, channels)
# 통상적으로 conv2D 레이어는 2단이상으로 쌓음. (1단으로는 성능이 잘 안나와..)
model.add(Conv2D(filters=20, kernel_size=(3,3), activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(10, (2,2), activation= 'relu')) 
model.add(Conv2D(20, (2,2), activation= 'relu'))  
model.add(Conv2D(10, (2,2), activation= 'relu')) 
model.add(MaxPooling2D(2,2))
model.add(Flatten()) #(n,22*22*15)의 모양을 펴져있는 모양으로 변형. (행렬연산임으로 연산 가능!)
model.add(Dense(20, activation= 'relu'))
# shape = (batch_size(=model.fit의 batch_size와 같음.), input_dim) 
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation= 'softmax'))


filepath = "C:\_data\_save\MCP\_k31"

#3. 컴파일, 훈련
model.compile ( loss = 'categorical_crossentropy', optimizer = 'adam', metrics= 'acc')
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 200, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)

start_time = time.time()
model.fit(x_train, y_train, batch_size = 4096, verbose=3, epochs= 100, validation_data= (x_valid, y_valid), callbacks= [es, mcp])
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )