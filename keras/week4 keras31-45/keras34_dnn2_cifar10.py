from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.utils import to_categorical


#acc- 0.77이상


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts= True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],





#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
#x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] ,x_test.shape[2], 3)


x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255



print(x_train.shape)
print(x_test.shape)






x_train = x_train.reshape ( (-1, 32*32*3))
x_test = x_test.reshape ( (-1, 32*32*3))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2.모델

model = Sequential()
model.add(Dense(5000, input_shape =  (32*32*3,), activation= 'relu'))
model.add(Dense(4000, activation = 'relu'))
model.add(Dense(3000, activation = 'relu'))
model.add(Dense(2500, activation = 'relu'))
model.add(Dense(2000, activation = 'relu'))
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(500, activation = 'relu'))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(10, activation='softmax'))


model.summary()


filepath = "C:\_data\_save\MCP\_k31"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

x_train = x_train.reshape ( (-1, 32*32*3))
x_test = x_test.reshape ( (-1, 32*32*3))


#3. 컴파일, 훈련
model.compile( loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc')
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 299, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)

start_time = time.time()
model.fit( x_train, y_train, batch_size=6000, verbose=2, epochs= 500, validation_split=0.3, callbacks= [es,mcp])
end_time =time.time()

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',  results[1])
print('걸린시간 :' , end_time - start_time, "초" )


# basic
# loss: 0.6674924492835999
# acc: 0.7788000106811523
# 걸린시간 : 459.2906882762909 초

# # #stride2,padding same
# loss: 0.9534503817558289
# acc: 0.6705999970436096
# 걸린시간 : 167.60450172424316 초

#MaxPooling
# loss: 0.7770913243293762
# acc: 0.7664999961853027
# 걸린시간 : 169.09292793273926 초

#dnn
#loss: 1.5208626985549927
#acc: 0.48899999260902405
#걸린시간 : 217.5431613922119 초
