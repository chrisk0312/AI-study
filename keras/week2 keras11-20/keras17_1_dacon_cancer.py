# https://dacon.io/competitions/open/236068/overview/description

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error
import time
from keras.callbacks import EarlyStopping

#1 데이터
path = "c:\\_data\\dacon\\diabetes\\"
train_csv = pd.read_csv(path + "train.csv",index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path +"sample_submission.csv",)
print(submission_csv)
print(train_csv.shape) 
print(test_csv.shape) 
print(submission_csv.shape) 
print(train_csv.columns)
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

###########x와 y의 값을 분리
x= train_csv.drop(['Outcome'], axis=1) 
print(x)
y = train_csv['Outcome']
print(y)

x_train,x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.3,shuffle=False, random_state=100,)
print(x_train.shape,x_test.shape) #(2177, 8) (8709, 8)
print(y_train.shape, y_test.shape) #(2177,) (8709,)

#2. 모델
model = Sequential()
model.add(Dense(32,input_dim = 8))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer= 'adam',
              metrics =['acc']
              )
start_time = time.time()
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience=100,
                   verbose=2,
                   restore_best_weights = True,
                   )
hist = model.fit(x_train, y_train, epochs=300, batch_size=32,
          validation_split=0.3,
          callbacks= [es],
          verbose=2)
end_time = time.time()


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
print(y_submit)
print(y_submit.shape)

results = model.predict(x)
y_rounded= np.around(y_predict)
from sklearn.metrics import  accuracy_score

def acc(y_test, y_predict):
    return(accuracy_score(y_test,y_predict))
rounded_array = np.around(np.array([y_predict]))
acc =(y_test, y_rounded)
print("ACC :",acc)
print("걸린시간:", round (end_time - start_time,2), "초")

print("===================================")
#########submission.csv 만들기(count 컬럼에 값만 제출)#############
submission_csv['Outcome'] = np.around(y_submit)
print(submission_csv)
print(submission_csv.shape)
submission_csv.to_csv(path+"submission_0110_4.csv", index=False)
print("로스 :",loss)



import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'],c='red',label = 'loss',marker=".")
plt.plot(hist.history['val_loss'], c='blue',label='val_loss',marker ='.' )
plt.legend(loc='upper right')
plt.title('sigmoid loss')
plt.xlabel('에포')
plt.ylabel('로스')
plt.grid()
plt.show()


