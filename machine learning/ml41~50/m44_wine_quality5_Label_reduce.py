from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler

path = "C:\\_data\\dacon\\wine\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# train_csv = pd.concat([train_csv,train_csv])
# train_csv = pd.concat([train_csv,train_csv])
# train_csv = pd.concat([train_csv,train_csv])
# train_csv = pd.concat([train_csv,train_csv])

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함
# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]
x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']

print(np.unique(y,return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import time
st = time.time()    # pandas로 해보기
y_fix0 = y.copy()
y_fix0.loc[y_fix0 == 3] = 5
y_fix0.loc[y_fix0 == 4] = 5
y_fix0.loc[y_fix0 == 8] = 7
y_fix0.loc[y_fix0 == 9] = 7

# y = y_fix0
et = time.time()
print(et-st,"sec")

st = time.time()    # numpy로 해보기
y_fix = np.asarray(y.copy())
y_fix_3 = np.where(y_fix == 3)
y_fix_4 = np.where(y_fix == 4)
y_fix_8 = np.where(y_fix == 8)
y_fix_9 = np.where(y_fix == 9)

y_fix[y_fix_3] = 4
# y_fix[y_fix_4] = 5
y_fix[y_fix_9] = 8
# y_fix[y_fix_8] = 7

# y = y_fix
et = time.time()
print(et-st,"sec")

st = time.time()    # for문으로 해보기
y_fix2 = np.asarray(y.copy())
for idx, data in enumerate(y_fix2):
    if data == 3:
        y_fix2[idx] = 5
    elif data == 4:
        y_fix2[idx] = 5
    elif data == 8:
        y_fix2[idx] = 7
    elif data == 9:
        y_fix2[idx] = 7

y = y_fix2
et = time.time()
print(et-st,"sec") 
        
print(np.unique(y,return_counts=True))

y = LabelEncoder().fit_transform(y)
print(np.unique(y,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

def remove_outlier(dataset:pd.DataFrame):
    for label in dataset:
        data = dataset[label]
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3-q1
        upbound    = q3 + iqr*1.5
        underbound = q1 - iqr*1.5
        dataset.loc[dataset[label] < underbound, label] = underbound
        dataset.loc[dataset[label] > upbound, label] = upbound
        
    return dataset

# print(train_csv.head(10))
# print(test_csv.head(10))

x = remove_outlier(x)
# print(train_csv.shape,x.shape,sep='\n')
# print(train_csv.max(),train_csv.min())
# print(x.max(),x.min())

x = x.astype(np.float32)
y = y.astype(np.float32)
print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=770,stratify=y)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)



# model
model = RandomForestClassifier()

# fit & pred
model.fit(x_train,y_train,)
pred = model.predict(x_test)
result = model.score(x_test,y_test)

print(result)

from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)
f1 = f1_score(y_test,pred,average='macro')
print("F1 : ",f1)

# 7개 전부
# ACC:  0.6963636363636364
# F1 :  0.4172314374211008

# 5개
# ACC:  0.9345156889495225
# F1 :  0.9258579831112991

# 3개
# ACC:  0.69
# F1 :  0.6783743403213683