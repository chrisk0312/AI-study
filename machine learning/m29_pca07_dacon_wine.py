#https://dacon.io/competitions/open/235610/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import time
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier




#1. 데이터
path = "C:\\_data\\daicon\\wine\\"



train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
print(x)
y = train_csv['quality']



x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
print(x)

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

print(test_csv)



#print(x.shape,y.shape) #(5497, 12) (5497, 7)

#columns = train_csv.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)

for i in range(len(x.columns)):
    scaler = StandardScaler()
    x_1 = scaler.fit_transform(x)
    pca = PCA(n_components=i+1)
    x_1 = pca.fit_transform(x)  
    x_train, x_test, y_train, y_test = train_test_split(x_1, y, train_size=0.8, random_state=777, shuffle= True, stratify=y)

    #2. 모델
    model = RandomForestClassifier(random_state=777)

    #3. 훈련
    model.fit(x_train, y_train)

    #4. 평가, 예측
    results = model.score(x_test, y_test)
    print('===============')
    #print(x.shape)
    print('feature 갯수',i+1,'개', 'model.score :',results)
    evr = pca.explained_variance_ratio_ #설명할수있는 변화율
    #n_component 선 갯수에 변화율

    print(evr)
    print(sum(evr))

    evr_cunsum = np.cumsum(evr)
    print(evr_cunsum)       

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, shuffle= True, random_state=364, stratify= y)

# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
# from sklearn.preprocessing import StandardScaler, RobustScaler

# #mms = MinMaxScaler()
# #mms = StandardScaler()
# #mms = MaxAbsScaler()
# mms = RobustScaler()

# mms.fit(x_train)
# x_train= mms.transform(x_train)
# x_test= mms.transform(x_test)


# #2. 모델구성

# model = LinearSVR()

# x_train = np.asarray(x_train).astype(np.float32)
# x_test = np.asarray(x_test).astype(np.float32)
# test_csv = np.asarray(test_csv).astype(np.float32)


# #3. 컴파일, 훈련

# start_time = time.time()
# model.fit(x_train, y_train)
# end_time = time.time()

# #4. 평가, 예측

# results = model.score(x_test, y_test)
# print("model.score:", results)
# y_predict = model.predict(x_test)


# y_submit = model.predict(test_csv)


# y_submit = (y_submit)+3
# submission_csv['quality'] = y_submit
# #acc = accuracy_score(y_predict, y_test)
# ltm = time.localtime(time.time())
# #print("acc :", acc)
# print("로스 :", results)
# save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
# submission_csv.to_csv(path+f"submission_{save_time}e.csv", index=False)




# print("걸린 시간 :", round(end_time - start_time, 2), "초" )

#minmax
#acc : 0.5581818181818182
#로스 : 1.0659217834472656

#StandardScaler
#acc : 0.5681818181818182
#로스 : 1.0569180250167847

#MaxAbsScaler()
#acc : 0.5709090909090909
#로스 : 1.0683833360671997

#mms = RobustScaler()
#acc : 0.5609090909090909
#로스 : 1.0590828657150269

#LinearSVR()
# model.score: 0.2888460317435524
# 로스 : 0.2888460317435524


# feature 갯수 1 개 model.score : 0.5009090909090909
# ===============
# feature 갯수 2 개 model.score : 0.5381818181818182
# ===============
# feature 갯수 3 개 model.score : 0.5627272727272727
# ===============
# feature 갯수 4 개 model.score : 0.5954545454545455
# ===============
# feature 갯수 5 개 model.score : 0.6581818181818182
# ===============
# feature 갯수 6 개 model.score : 0.6563636363636364
# ===============
# feature 갯수 7 개 model.score : 0.649090909090909
# ===============
# feature 갯수 8 개 model.score : 0.6463636363636364
# ===============
# feature 갯수 9 개 model.score : 0.6609090909090909
# ===============
# feature 갯수 10 개 model.score : 0.6718181818181819
# ===============
# feature 갯수 11 개 model.score : 0.6690909090909091
# ===============
# feature 갯수 12 개 model.score : 0.6572727272727272

# 0.95317628 0.99430958 0.99918288 0.9996546  0.99995729 0.99998275
#  0.99998878 0.99999329 0.99999705 0.9999998  1.         1.          