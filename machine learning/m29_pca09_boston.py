from sklearn.datasets import load_boston

# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 삭제 후 재설치
#pip uninstall scikit-learn
#pip uninstall scikit-image                  
#pip uninstall scikit-learn-intelex
#pip install scikit-learn==1.1.3


datasets = load_boston()
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']

print(datasets.DESCR) #Describe #행 = Tnstances

# [실습]
#train_size 0.7이상, 0.9이하
#R2 0.8 이상

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') #warning 무시. 나타내지않음.
import time
from sklearn.svm import LinearSVR

#1. 데이터

x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20, random_state= 4041 ) #4041
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from sklearn.decomposition import PCA


columns = datasets.feature_names
#columns = datasets.columns
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
'''

0.81002097 0.96730422 0.98995507 0.99724003 0.99846464 0.99921063
 0.99961588 0.99986617 0.99995721 0.99999093 0.9999983  0.99999992

feature 갯수 1 개 model.score : -0.08112705091177852
===============
feature 갯수 2 개 model.score : 0.3028437218149459
===============
feature 갯수 3 개 model.score : 0.3918700616065195
===============
feature 갯수 4 개 model.score : 0.4056842310503598
===============
feature 갯수 5 개 model.score : 0.6641555242070352
===============
feature 갯수 6 개 model.score : 0.702720986395823
===============
feature 갯수 7 개 model.score : 0.7022254145253655
===============
feature 갯수 8 개 model.score : 0.7408360426355776
===============
feature 갯수 9 개 model.score : 0.7385720450522116
===============
feature 갯수 10 개 model.score : 0.7501018948288931
===============
feature 갯수 11 개 model.score : 0.7752075336837628
===============
feature 갯수 12 개 model.score : 0.7899627750387566
===============
feature 갯수 13 개 model.score : 0.7921234587083393
print(x_train)          
print(y_train)
print(x_test)
print(x_test)

#2. 모델구성

model= LinearSVR()

#3. 컴파일, 훈련

start_time = time.time() #현재시간이 들어감
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가, 예측
loss = model.score(x_test, y_test)

print("model.score :", loss)
y_predict = model.predict(x_test) 
result = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)
print("걸린 시간 :", round(end_time - start_time, 2), "초") #def로 정의하지 않은 함수는 파이썬에서 기본으로 제공해주는 함수.

#로스 : 11.972917556762695
#R2 스코어 : 0.8349828030281251
#test_size= 0.20, random_state= 4041
#epochs = 5000, batch_size= 20
# 노드 - 1, 9, 13, 9, 3, 1

#LinearSVR
# model.score : 0.5005368994698873
# R2 스코어 : 0.5005368994698873
# 걸린 시간 : 0.01 초

'''