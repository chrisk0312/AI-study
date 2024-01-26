#https://dacon.io/competitions/official/236214/data

from keras.models import Sequential, Model
from keras.layers import Dense,Dropout, Input
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from keras.callbacks import EarlyStopping
import time
import warnings
warnings.filterwarnings(action='ignore')
from imblearn.over_sampling import SMOTE




#1. 데이터
le = LabelEncoder()
path = "C:\\_data\\dacon\\dechul\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')



#대출기간 처리
train_loan_time = train_csv['대출기간']
train_loan_time = train_loan_time.str.split()
for i in range(len(train_loan_time)):
    train_loan_time.iloc[i] = int((train_loan_time)[i][0])
    
test_loan_time = test_csv['대출기간']
test_loan_time = test_loan_time.str.split()
for i in range(len(test_loan_time)):
    test_loan_time.iloc[i] = int((test_loan_time)[i][0])

train_csv['대출기간'] = train_loan_time
test_csv['대출기간'] = test_loan_time

#근로기간 처리
train_working_time = train_csv['근로기간']
test_working_time = test_csv['근로기간']

for i in range(len(train_working_time)):
    data = train_working_time.iloc[i]
    if data == 'Unknown':
        train_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        train_working_time.iloc[i] = int(15)
    elif data == '< 1 year' or data == '<1 year':
        train_working_time.iloc[i] = int(0.5)
    else:
        train_working_time.iloc[i] = int(data.split()[0])
        
    
train_working_time = train_working_time.fillna(train_working_time.mean())


for i in range(len(test_working_time)):
    data = test_working_time.iloc[i]
    if data == 'Unknown':
        test_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        test_working_time.iloc[i] = int(15)
    elif data == '< 1 year' or data == '<1 year':
        test_working_time.iloc[i] = int(0.5)
    else:
        test_working_time.iloc[i] = int(data.split()[0])
    
test_working_time = test_working_time.fillna(test_working_time.mean())

train_csv['근로기간'] = train_working_time
test_csv['근로기간'] = test_working_time 



#대출목적 전처리
test_loan_perpose = test_csv['대출목적']
train_loan_perpose = test_csv['대출목적']


for i in range(len(test_loan_perpose)):
    data = test_loan_perpose.iloc[i]
    if data == '결혼':
        test_loan_perpose.iloc[i] = np.NaN

test_loan_perpose = test_loan_perpose.fillna(method='ffill')
test_csv['대출목적'] = test_loan_perpose
train_csv = train_csv.dropna(axis=0)
test_csv = test_csv.dropna(axis=0)
     



######## 결측치확인
print(test_csv.isnull().sum()) #없음.
print(train_csv.isnull().sum()) #없음.


le.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le.transform(train_csv['주택소유상태'])
le.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] = le.transform(test_csv['주택소유상태'])
#print(train_csv['주택소유상태'])
#print(test_csv['주택소유상태'])


le.fit(train_csv['대출목적'])
train_csv['대출목적'] = le.transform(train_csv['대출목적'])
le.fit(test_csv['대출목적'])
test_csv['대출목적'] = le.transform(test_csv['대출목적'])

le.fit(train_csv['대출기간'])
train_csv['대출기간'] = le.transform(train_csv['대출기간'])
le.fit(test_csv['대출기간'])
test_csv['대출기간'] = le.transform(test_csv['대출기간'])

x = train_csv.drop(['대출등급'], axis = 1)


#print(x)
y = train_csv['대출등급']


y = le.fit_transform(y)

y= y.reshape(-1,1)
ohe = OneHotEncoder(sparse= False)
y = ohe.fit_transform(y)

#print(x.shape, y.shape)  #(96294, 13) (96294, 7)
#print(np.unique(y, return_counts= True)) #array(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype=object), array([16772, 28817, 27623, 13354,  7354,  1954,   420],

#print(train_csv)

#print(y.shape) #(96294,)
#print(np.unique(y, return_counts= True)) #Name: 근로기간, Length: 96294, dtype: float64


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7,  shuffle= True, random_state= 777, stratify= y) 

     
     
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
mms = StandardScaler()
#mms = MaxAbsScaler()
#mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)
test_csv= mms.transform(test_csv)
#1standard swish (0.90), 4standard relu (0.88)완 2robust swish(0.90)완 3robust relu (0.88)(완)
        

 
#encoder.fit(y_train)
#y_train = encoder.transform(y_train)
#print(y_train)
#print(y_test)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=777)
x_train, y_train = smote.fit_resample(x_train, y_train)

# print(pd.value_counts(y_train))



# #2. 모델구성

model = Sequential()  
model.add(Dense(128, input_shape= (13,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(7, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation = 'softmax'))





#3. 컴파일, 훈련

import datetime
date= datetime.datetime.now()
# print(date) #2024-01-17 11:00:58.591406
# print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d-%H%M") #m=month, M=minutes
# print(date) #0117_1100
# print(type(date)) #<class 'str'>

path1= 'C:/_data/_save/MCP/_k28/' #경로(스트링data (문자))
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #filename= 에포4자리수-발로스는 소숫점4자리까지 표시. 예)1000-0.3333.hdf5
filepath = "".join([path1, 'k28_11_', date, "_", filename]) #""공간에 ([])를 합쳐라.


x_train = np.asarray(x_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)


from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50, verbose = 3, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)

model.compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= 'acc' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
hist = model.fit(x_train, y_train, callbacks=[es,mcp], epochs= 1000000, batch_size = 2048, validation_split= 0.2, verbose=2)


#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print("로스 :", results[0])
print("정확도 :" , results[1])

y_predict = model.predict(x_test, verbose=0)
y_predict = np.argmax(y_predict,axis=1)

y_submit = np.argmax(model.predict(test_csv),axis=1)
y_test = np.argmax(y_test,axis=1)


f1 = f1_score(y_test,y_predict,average='macro')
print("F1: ",f1)


y_submit = le.inverse_transform(y_submit)

import datetime
dt = datetime.datetime.now()
submission_csv['대출등급'] = y_submit

submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_F1_{f1:4}.csv",index=False)

