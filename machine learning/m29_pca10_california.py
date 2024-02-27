import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVR

#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, random_state= 3884 ) #282
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from sklearn.decomposition import PCA


x_train = pd.DataFrame(x_train)
#columns = x_train.feature_names
columns = x_train.columns
x_train = pd.DataFrame(x_train,columns=columns)

for i in range(len(x_train.columns)):
    scaler = StandardScaler()
    x1_train = scaler.fit_transform(x_train)
    x1_test = scaler.fit_transform(x_test)
    pca = PCA(n_components=i+1)
    x1_train = pca.fit_transform(x_train)
    x1_test = pca.transform(x_test)

    #2. 모델
    model = RandomForestRegressor(random_state=777)

    #3. 훈련
    model.fit(x1_train, y_train)

    #4. 평가, 예측
    results = model.score(x1_test, y_test)
    print('===============')
    #print(x.shape)
    print('feature 갯수',i+1,'개', 'model.score :',results)
    evr = pca.explained_variance_ratio_ #설명할수있는 변화율
    #n_component 선 갯수에 변화율
#   print(evr)
 #  print(sum(evr))
    evr_cunsum = np.cumsum(evr)
    print(evr_cunsum)
    
# feature 갯수 1 개 model.score : -0.40394381837368054
# ===============
# feature 갯수 2 개 model.score : 0.01882418916292994
# ===============
# feature 갯수 3 개 model.score : 0.026062163405931038
# ===============
# feature 갯수 4 개 model.score : 0.28526543245521907
# ===============
# feature 갯수 5 개 model.score : 0.5485057675291982
# ===============
# feature 갯수 6 개 model.score : 0.6875565184286214
# ===============
# feature 갯수 7 개 model.score : 0.763780805982108
# ===============
# feature 갯수 8 개 model.score : 0.7633874691341704      
 
#  0.9997716  0.99988722 0.99998563 0.99999214 0.99999742 0.99999978
#  0.99999998 1.             
         
# print(x)
# print(y)
# print(x.shape, y.shape)

# print(datasets.feature_names)
# #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# print(datasets.DESCR)

# #2. 모델구성

# model = LinearSVR(C=123)


# #3. 컴파일, 훈련
# start_time = time.time()
# model.fit(x_train, y_train)
# end_time = time.time()

# #4. 평가, 예측
# loss = model.score(x_test, y_test)
# print("model.score :", loss)
# y_predict = model.predict(x_test) 
# result = model.predict(x)



# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("r2 :", r2)
# print("걸린 시간 :", round(end_time - start_time, 2), "초")



#로스 : 0.5067346692085266
#R2 스코어 : 0.6165943551579782
#epochs 5000, batch_size= 200
#8, 16, 10, 8, 4, 1 랜덤 282


#LinearSVR
# model.score : 0.4207089223795233
# r2 : 0.4207089223795233
# 걸린 시간 : 0.39 초