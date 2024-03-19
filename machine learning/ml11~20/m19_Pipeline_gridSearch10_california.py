import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split,GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
import time
from sklearn.svm import LinearSVR

#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, random_state= 3884 ) #282

print(x)
print(y)
print(x.shape, y.shape)

print(datasets.feature_names)
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)
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
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

#4. 평가, 예측
loss = model.score(x_test, y_test)
print("model.score :", loss)
y_predict = model.predict(x_test) 
result = model.predict(x)



from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)
print("걸린 시간 :", round(end_time - start_time, 2), "초")



#로스 : 0.5067346692085266
#R2 스코어 : 0.6165943551579782
#epochs 5000, batch_size= 200
#8, 16, 10, 8, 4, 1 랜덤 282


#LinearSVR
# model.score : 0.4207089223795233
# r2 : 0.4207089223795233
# 걸린 시간 : 0.39 초

#PIPELINE
# model.score : 0.8131827850825886
# r2 : 0.8131827850825886
# 걸린 시간 : 5.23 초

#pipe2
# model.score : 0.8118655939627433
# r2 : 0.8118655939627433
# 걸린 시간 : 129.21 초

