import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import time
#1.데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)
#print(datasets.feature_names)

x = datasets.data
y= datasets.target
#print(x.shape) #(569, 30),
#print(y.shape) #(569)

print(np.unique(y, return_counts= True)) #(array([0, 1]), array([212, 357])
print(pd.Series(y).value_counts()) #다 똑가틈
print(pd.DataFrame(y).value_counts()) #다 똑가틈 (dataframe을 쓰면 메모리를 더 씀,행렬데이터는 dataframe만 가능)
print(pd.value_counts(y)) #다 똑가틈


#zero_num = len(y[np.where(y==0)]) #넘
#one_num = len(y[np.where(y==1)]) #파
#print(f"0: {zero_num}, 1: {one_num}") #이
#print(df_y.value_counts()) 0 =  212, 1 = 357 #pandas
#print(, unique)
# print("1", counts)
#sigmoid함수- 모든 예측 값을 0~1로 한정시킴.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.68, shuffle= False, random_state= 334)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
mms = MaxAbsScaler()
#mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)
#2. 모델구성

model = Sequential()
model.add(Conv1D (1024, kernel_size=2, input_shape = (30,1)))
model.add(Flatten())
model.add(Dense (512, activation= 'relu'))
model.add(Dense (256,  ))
model.add(Dropout(0.3))
model.add(Dense (128))
model.add(Dropout(0.3))
model.add(Dense (64))
model.add(Dense (1,  activation= 'sigmoid')) #이진분류에서는 최종레이어 액티베이션은 100% 무조건 시그모이드!


#3. 컴파일, 훈련
import datetime
date= datetime.datetime.now()
# print(date) #2024-01-17 11:00:58.591406
# print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d-%H%M") #m=month, M=minutes
# print(date) #0117_1100
# print(type(date)) #<class 'str'>

path= 'c:/_data/_save/MCP/_k26/' #경로(스트링data (문자))
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #filename= 에포4자리수-발로스는 소숫점4자리까지 표시. 예)1000-0.3333.hdf5
filepath = "".join([path, 'k26_6_', date, "_", filename]) #""공간에 ([])를 합쳐라.



from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor = 'val_acc', mode = 'auto', patience = 500, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_acc', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)


model.compile(loss= 'binary_crossentropy', optimizer= 'adam',metrics=['acc',])
start_time= time.time()
hist = model.fit(x_train, y_train, callbacks=[mcp], epochs= 10000, batch_size = 8, validation_split= 0.25)
end_time= time.time()





#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(x_test)

#y_predict = np.round(y_predict, -1)


def ACC(y_test, y_predict):
    return accuracy_score(y_test, np.around(y_predict))
acc = ACC(y_test, y_predict)
print("ACC : ", acc)
print("걸린 시간:", round(end_time - start_time, 2), "초")






# print("걸린 시간 :", round(end_time - start_time, 2), "초" )

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus']= False
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], color= 'red', label= 'loss', marker= '.')
# plt.plot(hist.history['val_loss'], color= 'blue', label = 'val_loss', marker= '.')
# plt.plot(hist.history['acc'], color= 'violet', label = 'acc', marker= '.')
# plt.legend(loc = 'upper right')
# plt.title("유방암 loss")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.grid()
# plt.show()


#loss : [0.12679751217365265, .acc 0.9707602262496948]

#mms = MinMaxScaler()
#ACC :  0.9890710382513661
#로스 :  [0.047952909022569656

#mms = StandardScaler()
#ACC :  0.9617486338797814
#로스 :  [0.08647095412015915

#mms = MaxAbsScaler()
#ACC :  1.0
#로스 :  [0.05968305468559265

#mms = RobustScaler()
#ACC :  0.9836065573770492
#로스 :  [0.07321718335151672,

#ACC :  1.0

#ACC :  0.9890710382513661

#drop
#ACC :  1.0

#LSTM
#ACC :  0.9617486338797814

#Conv1D
#ACC :  0.9398907103825137