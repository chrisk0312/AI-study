import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM

a = np.array(range(1,101)) 
x_predict = np.array(range(96,106))


print(x_predict.shape) #(10,)

size =5 # x 데이터 4개 , y데이터 1개

def split_x(dataset, size): 
    l=[] 
    for i in range(len(dataset) -  size+1): #
        subset = dataset[i : (i + size)] 
       # print(f"{i=}\n{subset=}\n{aaa=}")
        l.append(subset) 
    return np.array(l) 

aaa = split_x(a,size)
bbb = split_x(x_predict,4) 
print(aaa.shape) #(96, 5)
print(bbb.shape) #(6, 5)

#===============================================
print(aaa.shape)  #(96, 5)
x= aaa[:, :-1] #x = bbb[:, :-1]: This extracts all columns except the last one from the array bbb and assigns it to x.
y = aaa[:,-1] #y = bbb[:, -1]: This extracts the last column from the array bbb and assigns it to y.
print(x,y) 
print(x.shape, y.shape)  #(96, 4) (96,)
x = x.reshape(96,4,1)
y = y.reshape(-1,1)
print(x.shape, y.shape) #(96, 4, 1) (96, 1)
print(bbb.shape)#(7, 4)


#2 
model= Sequential()
model.add(LSTM(units=10, input_shape =(4,1)))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)


#4
results= model.evaluate(x,y)
y_pred = model.predict(bbb)
print(y_pred)
print('loss', results)
print('(range(96,106))의 결과', y_pred)

# (range(96,106))의 결과 [[ 98.28417 ]
#  [ 99.038734]
#  [ 99.78014 ]
#  [100.5084  ]
#  [101.223526]
#  [101.92557 ]
#  [102.61458 ]]