import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score


#1. 데이터


datasets = fetch_covtype()
print(datasets)
print(datasets.DESCR)

print(datasets.feature_names) #'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 
#'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 
# 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 
# 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 
# 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20',
# 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29',
# 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38',
# 'Soil_Type_39']

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(581012, 54) (581012,)
print(pd.value_counts(y))
print(np.unique(y, return_counts= True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],)

#y = pd.get_dummies(y)

y_ohe1 = to_categorical(datasets.target) #0부터 라벨이 주어짐. 마지막숫자 +1만큼 라벨이 생성됨.
y_ohe1 = y_ohe1[:, 1:]




#ohe = OneHotEncoder(sparse= False)

print(y_ohe1.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe1, train_size= 0.7,  shuffle= True, random_state= 398, stratify= y_ohe1) #y의 라벨값을 비율대로 잡아줌 #회귀모델에서는 ㄴㄴ 분류에서만 가능
#print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
#print(y_train.shape, y_test.shape) #(7620, ) (3266, )
print(np.unique(y_test, return_counts = True ))


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
mms = StandardScaler()
#mms = MaxAbsScaler()
#mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)

#2. 모델구성

model = Sequential()
model.add(Dense(128, input_dim = 54))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(7, activation = 'softmax'))



#3. 컴파일, 훈련
model.compile  (loss = 'categorical_crossentropy', optimizer = 'adam', metrics= 'acc')

es = EarlyStopping(monitor= 'val_loss', mode= 'min',
                   patience=4000, verbose=2, restore_best_weights= True) #es는 verbose2가 es 정보를 보여줌.
start_time = time.time()
his = model.fit(x_train, y_train, epochs= 4000, batch_size=800, validation_split= 0.25, verbose=2, ) #검증모델은 간접적인 영향을 미침.
end_time = time.time()
  

#4. 평가, 예측



results = model.evaluate(x_test, y_test)
print("로스 :", results[0])
print("정확도 :", results[1])

y_predict = model.predict(x_test)





print(y_predict)

print(y_predict.shape, y_test.shape) #(30, 3) (30, 3)

y_test = np.argmax(y_test, axis=1) #원핫을 통과해서 아그맥스를 다시 통과시켜야함
y_predict = np.argmax(y_predict, axis=1 )
print(y_test, y_predict)
result = accuracy_score(y_test, y_predict)

acc = accuracy_score(y_predict, y_test)
print("acc :", acc)


print("걸린 시간 :", round(end_time - start_time, 2), "초" )


#print("mms = StandardScaler")
#로스 : 0.2562239468097687
#정확도 : 0.8990269899368286

#print('#mms = MaxAbsScaler')
#로스 : 0.2306247502565384
#정확도 : 0.910214364528656

#print('#mms = RobustScaler')
#로스 : 0.23458202183246613
#정확도 : 0.9094054102897644

#minmax
# 로스 : 0.34160080552101135
# 정확도 : 0.8604621887207031
