#https://dacon.io/competitions/open/236070/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import time
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline


#1. 데이터
path = "C:\\_data\\daicon\\iris\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv") 
print(submission_csv)


print(train_csv.shape) # (120, 5)
print(test_csv.shape) # (30, 4)
print(submission_csv.shape) # (30, 2)

print(train_csv.columns) #'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
#       'petal width (cm)', 'species']

print(train_csv.info())
print(test_csv.info())

x = train_csv.drop(['species'], axis = 1)
#print(x)

y = train_csv['species']

#print(y)

print(x.shape, y.shape) #(120, 4) (120,)
print(np.unique(y, return_counts= True)) #array([0, 1, 2] array([40, 41, 39]





x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, shuffle = True, random_state= 123, stratify= y)


#2. 모델구성

# model= LinearSVC(C=113)
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())


#3. 컴파일, 훈련

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측

results = model.score(x_test, y_test)
print("model.score :", results)
y_predict = model.predict(x_test)




print(y_predict)

print(y_predict.shape, y_test.shape)



y_submit = model.predict(test_csv)


result = accuracy_score(y_test, y_predict)



submission_csv['species'] = y_submit


ltm = time.localtime(time.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}.csv", index=False)
acc = accuracy_score(y_predict, y_test)
print("acc :", acc)


print("걸린 시간 :", round(end_time - start_time, 2), "초" )

#로스 : 0.20624160766601562
#acc : 0.9166666666666666


#Linear
#acc : 0.9722222222222222

#pipeline
# acc : 0.9444444444444444
