import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#1.데이터
datasets = load_breast_cancer()


x = datasets.data
y= datasets.target

# print(np.unique(y, return_counts= True)) #(array([0, 1]), array([212, 357])
# print(pd.Series(y).value_counts()) #다 똑가틈
# print(pd.DataFrame(y).value_counts()) #다 똑가틈 (dataframe을 쓰면 메모리를 더 씀,행렬데이터는 dataframe만 가능)
# print(pd.value_counts(y)) #다 똑가틈


#zero_num = len(y[np.where(y==0)]) #넘
#one_num = len(y[np.where(y==1)]) #파
#print(f"0: {zero_num}, 1: {one_num}") #이
#print(df_y.value_counts()) 0 =  212, 1 = 357 #pandas
#print(, unique)
# print("1", counts)
#sigmoid함수- 모든 예측 값을 0~1로 한정시킴.
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.68, shuffle= False, random_state= 334)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# #mms = MinMaxScaler()
# #mms = StandardScaler()
# #mms = MaxAbsScaler()
# mms = RobustScaler()

# mms.fit(x_train)
# x_train= mms.transform(x_train)
# x_test= mms.transform(x_test)

scaler = StandardScaler()
x = scaler.fit_transform(x)

lda = LinearDiscriminantAnalysis()
x = lda.fit_transform(x,y)
# n_components cannot be larger than min(n_features, n_classes - 1).

# pca = PCA(n_components=3)   # 컬럼을 몇개로 줄일것인가(숫자가 작을수록 데이터 손실이 많아짐) 
#                             #사용전에 스케일링(주로 standard)을 해서 모양을 어느정도 맞춰준 뒤 사용
# x = pca.fit_transform(x)
#print(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle= True, stratify=y)



#2. 모델
model = RandomForestClassifier(random_state=777)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('===============')
print(x_train.shape)
print('model.score :',results)



 
 
#  0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.        
# ===============
# feature 갯수 1 개 model.score : 0.8947368421052632
# ===============
# feature 갯수 2 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 3 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 4 개 model.score : 0.9736842105263158
# ===============
# feature 갯수 5 개 model.score : 0.9736842105263158
# ===============
# feature 갯수 6 개 model.score : 0.9736842105263158
# ===============
# feature 갯수 7 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 8 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 9 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 10 개 model.score : 0.956140350877193
# ===============
# feature 갯수 11 개 model.score : 0.9736842105263158
# ===============
# feature 갯수 12 개 model.score : 0.956140350877193
# ===============
# feature 갯수 13 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 14 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 15 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 16 개 model.score : 0.956140350877193
# ===============
# feature 갯수 17 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 18 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 19 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 20 개 model.score : 0.956140350877193
# ===============
# feature 갯수 21 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 22 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 23 개 model.score : 0.956140350877193
# ===============
# feature 갯수 24 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 25 개 model.score : 0.956140350877193
# ===============
# feature 갯수 26 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 27 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 28 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 29 개 model.score : 0.9649122807017544
# ===============
# feature 갯수 30 개 model.score : 0.956140350877193



# (455, 1)
# model.score : 0.9912280701754386