import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten 
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator



#1. 데이터
(x_train,y_train),(x_test,y_test)= mnist.load_data()
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
    fill_mode='nearest', #nearest가 디폴트 #reflect 좌우반전,wrap 감싸줌, 

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
    shuffle= False #섞이면 데이터가틀어짐.
).next()[0]

#print(x_augumented)
#print(x_augumented.shape)

#print(x_train.shape)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = np.concatenate((x_train,x_augumented))
y_train = np.concatenate((y_train,y_augumented))
#print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)



ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

x_train,x_test,y_train,y_test = train_test_split(x_train, y_train,train_size=0.8,random_state=112,
                                                 stratify=y_train)

#2. 모델
model = Sequential()
model.add(Conv2D(9,(2,2), input_shape =(28, 28, 1)))
#               shape =(batch_size, rows, columns, channels)
#               shape =(batch_size, rows, columns, channels)                
model.add(Conv2D(filters=10, kernel_size=(3,3)))
model.add(Conv2D(15,(4,4)))
model.add(Flatten())
model.add(Dense(units=8))
model.add(Dense(7, input_shape= (8,)))
#                   shape=(batch_size,input_dim)
model.add(Dense(6))
model.add(Dense(10, activation='softmax'))

#model.summary()

#(kernel_size*channels + bias) * filters 
# ((shape of width of filter*shape of height filter*number of filters in the previous layer+1(bias))*number of filters


#3. 컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam',
             metrics = ['acc'])
model.fit(x_train, y_train, batch_size=20000, verbose=1, 
          epochs=100, validation_split=0.2
          ) 

#4. 평가,예측
results = model.evaluate(x_test,y_test)
print('loss', results[0])
print('acc', results[1])


