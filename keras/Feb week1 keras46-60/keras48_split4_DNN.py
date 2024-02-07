import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

a = np.array(range(1, 101))
x_pred = np.array(range(96, 106)) #96~105
size = 5  #x데이터4개 y데이터 1개



def split_x(dataset, size): #a, timesteps만큼 자름
    aaa = []                #aaa = 
    for i in range(len(dataset) - size +1): # i = data길이 - timesteps +1
        subset = dataset[i : (i + size)]
        aaa.append(subset)                  # 반복
        
    return np.array(aaa)





bbb = split_x(a,size)
b2 = split_x(x_pred,size)
print(bbb)
# print(bbb.shape)


x = bbb[:, :-1]
y = bbb[:, -1]
print(x.shape) #(96, 4)
print(y.shape) #(96,)

print(x)
print(y)



#2. 모델구성
model = Sequential()
model.add(Dense(256, 
               input_shape= (4,1), activation = 'relu'))
#return_sequences로 2개이상의 LSTM을 사용 할 수 있다.
model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1))

model.summary()
#3. 컴파일, 훈련
es = EarlyStopping(monitor= 'loss', mode= 'auto', patience = 200, verbose= 0, restore_best_weights= True )
model.compile(loss= 'mse', optimizer= 'adam', metrics= 'acc')
model.fit(x,y,epochs=20, callbacks= [es], batch_size= 1)

#4. 평가, 훈련
results = model.evaluate(x,y)
print ('loss:', results)
x_pred = np.array(range(96,106)).reshape(-1,1)
x_pred = model.predict(x_pred)
print('예측값 :', x_pred)

    