import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier

#1.데이터
datasets = load_breast_cancer()


x = datasets.data
y= datasets.target

print(np.unique(y, return_counts= True)) #(array([0, 1]), array([212, 357])
print(pd.Series(y).value_counts()) #다 똑가틈
print(pd.DataFrame(y).value_counts()) #다 똑가틈 (dataframe을 쓰면 메모리를 더 씀,행렬데이터는 dataframe만 가능)
print(pd.value_counts(y)) #다 똑가틈


#zero_num = len(y[np.where(y==0)]) #넘
#one_num = len(y[np.where(y==1)]) #파
#print(f"0: {zero_num}, 1: {one_num}") #이
#print(df_y.value_counts()) 0 =  212, 1 = 357 #pandas
#print(, unique)
# print("1", counts)
#sigmoid함수- 모든 예측 값을 0~1로 한정시킴.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.68, shuffle= False, random_state= 334)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)

#2. 모델구성
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models:
    model.fit(x_train, y_train)
    results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_predict, y_test)
    print(type(model).__name__, "acc :", acc)
    print(type(model).__name__, ':', model.feature_importances_)
        







#loss : [0.12679751217365265, .acc 0.9707602262496948]

#mms = MinMaxScaler()
#ACC :  0.9890710382513661
#로스 :  [0.047952909022569656

#mms = StandardScaler()
#ACC :  0.9617486338797814
#로스 :  [0.08647095412015915

#mms = MaxAbsScaler()
#ACC :  1.0
#로스 :  [0.05968305468559265

#mms = RobustScaler()
#ACC :  0.9836065573770492
#로스 :  [0.07321718335151672,


#LinearSVC
# model.sore: 0.9617486338797814
# acc : 0.9617486338797814

# DecisionTreeClassifier acc : 0.8743169398907104
# DecisionTreeClassifier : [0.         0.03478658 0.         0.         0.01607915 0.02253338
#  0.         0.         0.         0.00845964 0.00324181 0.
#  0.         0.         0.00275314 0.         0.         0.
#  0.         0.         0.02335996 0.00258756 0.71692205 0.00970337
#  0.06644442 0.         0.07280151 0.01018282 0.         0.01014462]
# RandomForestClassifier acc : 0.9672131147540983
# RandomForestClassifier : [0.05254952 0.01330487 0.04105    0.07485901 0.00872065 0.00753846
#  0.05727807 0.13165028 0.00331791 0.0042324  0.00939497 0.0032734
#  0.00759812 0.03644384 0.00560741 0.00365043 0.0076336  0.00208988
#  0.00659568 0.00476697 0.0993854  0.02165213 0.12562537 0.07781659
#  0.01976186 0.01296562 0.05911353 0.08718944 0.01030168 0.0046329 ]
# GradientBoostingClassifier acc : 0.9617486338797814
# GradientBoostingClassifier : [1.11081732e-03 7.20791541e-03 4.27269102e-04 2.27232667e-03
#  6.02198749e-03 1.23091347e-03 1.28730546e-03 2.05181002e-01
#  5.67953548e-03 1.93534010e-04 6.66489180e-03 2.75549096e-04
#  9.74984421e-04 8.52693459e-03 2.56668217e-04 7.14799229e-04
#  5.67885282e-04 8.38909274e-04 5.22238377e-04 4.04831416e-04
#  2.21797779e-02 5.00401217e-02 4.62374839e-01 9.61220311e-02
#  3.25853114e-02 1.17924481e-03 3.89112865e-02 4.29278565e-02
#  3.12495207e-03 1.94280387e-04]
# XGBClassifier acc : 0.9453551912568307
# XGBClassifier : [0.02393833 0.02867567 0.01606421 0.00278238 0.01022092 0.02669617
#  0.00245873 0.16521733 0.00094606 0.0016372  0.01868628 0.00889118
#  0.01493881 0.01243768 0.00181433 0.00563054 0.00128863 0.00072563
#  0.00243493 0.0019352  0.02740289 0.01897488 0.47150862 0.03872457
#  0.01696069 0.         0.02445868 0.04084015 0.00490873 0.0088006 ]