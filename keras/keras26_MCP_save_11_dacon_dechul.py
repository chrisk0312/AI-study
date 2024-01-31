#https://dacon.io/competitions/official/236214/data

#문자 수치화
#값 일부 자르기 (label encoder)
#값 자른거 수치화까지!

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from keras.callbacks import EarlyStopping
import time
import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터
path = "C:\\_data\\dacon\\dechul\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
#print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
#print(test_csv)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
#print(submission_csv)

#print(train_csv.shape) #(96294, 14)
#print(test_csv.shape)  #(64197, 13)
#print(submission_csv.shape) #(64197, 2)

#print(train_csv.columns) #'대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#       '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수', '대출등급'

#대출기간 처리
train_loan_time = train_csv['대출기간']
train_loan_time = train_loan_time.str.split()
for i in range(len(train_loan_time)):
    train_loan_time.iloc[i] = int((train_loan_time)[i][0])
    
#print(train_loan_time)   

test_loan_time = test_csv['대출기간']
test_loan_time = test_loan_time.str.split()
for i in range(len(test_loan_time)):
    test_loan_time.iloc[i] = int((test_loan_time)[i][0])

train_csv['대출기간'] = train_loan_time
test_csv['대출기간'] = test_loan_time

le = LabelEncoder()

#print(test_csv)

le.fit(train_csv['대출기간'])
train_csv['대출기간'] = le.transform(train_csv['대출기간'])
le.fit(test_csv['대출기간'])
test_csv['대출기간'] = le.transform(test_csv['대출기간'])

#근로기간 처리
train_working_time = train_csv['근로기간']
test_working_time = test_csv['근로기간']

for i in range(len(train_working_time)):
    data = train_working_time.iloc[i]
    if data == 'Unknown':
        train_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        train_working_time.iloc[i] = int(30)
    elif data == '< 1 year' or data == '<1 year':
        train_working_time.iloc[i] = int(0)
    else:
        train_working_time.iloc[i] = int(data.split()[0])
    
train_working_time = train_working_time.fillna(train_working_time.mean())

for i in range(len(test_working_time)):
    data = test_working_time.iloc[i]
    if data == 'Unknown':
        test_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        test_working_time.iloc[i] = int(30)
    elif data == '< 1 year' or data == '<1 year':
        test_working_time.iloc[i] = int(0)
    else:
        test_working_time.iloc[i] = int(data.split()[0])
    
test_working_time = test_working_time.fillna(test_working_time.mean())

train_csv['근로기간'] = train_working_time
test_csv['근로기간'] = test_working_time 
#print(test_csv['근로기간'])

le.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le.transform(train_csv['주택소유상태'])
le.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] = le.transform(test_csv['주택소유상태'])

le.fit(train_csv['대출목적'])
train_csv['대출목적'] = le.transform(train_csv['대출목적'])
le.fit(test_csv['대출목적'])
test_csv['대출목적'] = le.transform(test_csv['대출목적'])


x = train_csv.drop(['대출등급'], axis = 1)
mms = MinMaxScaler()
mms.fit(x)
x = mms.transform(x)
test_csv=mms.transform(test_csv)

#print(x)
y = train_csv['대출등급']


y = le.fit_transform(y)

y= y.reshape(-1,1)
ohe = OneHotEncoder(sparse= False)
y = ohe.fit_transform(y)

#print(x.shape, y.shape)  #(96294, 13) (96294,)
#print(np.unique(y, return_counts= True)) #array(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype=object), array([16772, 28817, 27623, 13354,  7354,  1954,   420],

#print(train_csv)

#print(y.shape) #(96294,)
print(np.unique(y, return_counts= True)) #Name: 근로기간, Length: 96294, dtype: float64


#for data in train_loan_time:
#    if type(data) != type(1):
#        print("not int" , data) 




######## 결측치처리
#print(test_csv.isnull().sum())
#train_csv[train_csv.iloc[:,:] == 'unknown'] = np.NaN
#test_csv[test_csv.iloc[:,:] == 'unknown'] = np.NaN

#train_csv = train_csv.dropna(axis=0)
#test_csv = train_csv.dropna(axis=0)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72,  shuffle= True, random_state= 301, stratify= y) 

     
     
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
#mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)


        
#for label in x_test:
#    if label not in le.classes_: # unseen label 데이터인 경우( )
#        le.classes_ = np.append(le.classes_, label) # 미처리 시 ValueError발생


#for label in x_train:
#    if label not in le.classes_: # unseen label 데이터인 경우( )
#        le.classes_ = np.append(le.classes_, label) # 미처리 시 ValueError발생
 
#encoder.fit(y_train)
#y_train = encoder.transform(y_train)
#print(y_train)
#print(y_test)


#2. 모델구성

model = Sequential()
model.add(Dense(13, input_dim = 13,activation='sigmoid'))
model.add(Dense(52))

model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation = 'softmax'))



#3. 컴파일, 훈련

x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)


from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 400, verbose = 0, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath='../_data/_save/MCP/keras26_11_MCP1.hdf5')

model.compile(loss= 'mse', optimizer= 'adam', metrics= 'acc' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
hist = model.fit(x_train, y_train, callbacks=[es,mcp], epochs= 30000, batch_size = 2500, validation_split= 0.2)


#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print("로스 :", results[0])
print("정확도 :" , results[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)

y_submit = np.argmax(model.predict(test_csv,verbose=0),axis=1)
y_test = np.argmax(y_test,axis=1)

f1 = f1_score(y_test,y_predict,average='weighted')
print("=========================\nF1: ",f1)


y_submit = le.inverse_transform(y_submit)

import datetime
dt = datetime.datetime.now()
submission_csv['대출등급'] = y_submit
submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_F1{f1:4}.csv",index=False)


# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus']= False
# plt.figure(figsize= (9,6))
# plt.plot(hist.history['loss'], c = 'red', label = 'loss', marker = '.')
# plt.plot(hist.history['val_acc'], c = 'pink', label = 'val_acc', marker = '.')
# plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')
# plt.legend(loc = 'upper right')
# plt.title("대출등급 LOSS")
# plt.xlabel('epoch')
# plt.grid()
# plt.show()

#minmax
#로스 : 0.45476970076560974
#정확도 : 0.8460853695869446

#mms = StandardScaler()

#로스 : 0.46329641342163086
#정확도 : 0.8380373120307922

#mms = MaxAbsScaler()
#로스 : 0.44124674797058105
#F1:  0.8497932009729917


#mms = RobustScaler()
#로스 : 0.3884388208389282
#F1:  0.865791228309671













