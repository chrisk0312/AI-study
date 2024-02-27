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
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split,GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore') #warning 무시. 나타내지않음.
import time
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler



#1. 데이터

x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20, random_state= 4041 ) #4041

print(x_train)          
print(y_train)
print(x_test)
print(x_test)

parameters = [
    {"RF__n_estimators": [100, 200], "RF__max_depth": [6, 10, 12], "RF__min_samples_leaf": [3, 10]},
    {"RF__max_depth": [6, 8, 10, 12], "RF__min_samples_leaf": [3, 5, 7, 10]},
    {"RF__min_samples_leaf": [3, 5, 7, 10], "RF__min_samples_split": [2, 3, 5, 10]},
    {"RF__min_samples_split": [2, 3, 5, 10]},
]    
     


#2. 모델구성
     
#model = RandomForestClassifier()


#model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([('mm',MinMaxScaler()),('RF',RandomForestRegressor())])
 
#model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
#model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

#전처리, 모델구성을 한번에 할 수 있음.
#시스템상에서만 바뀌는것, 프린트 했을때는 바뀐값을 출력해주지 않음.




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

#PIPELINE
# model.score : 0.8768154727363252
# R2 스코어 : 0.8768154727363252
# 걸린 시간 : 0.12 초

#pipe2
# model.score : 0.8513699722779766
# R2 스코어 : 0.8513699722779766
# 걸린 시간 : 16.72 초