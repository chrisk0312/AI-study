#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier



#1. 데이터
path = "c:\\_data\\daicon\\diabetes\\"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)
test_csv = pd.read_csv(path +"test.csv",index_col=0  ).drop(['Pregnancies', 'DiabetesPedigreeFunction'], axis=1)

test = train_csv['SkinThickness']
for i in range(test.size):
      if test[i] == 0:
         test[i] =test.mean()
        
#train_csv['BloodPressure'] = test


submission_csv = pd.read_csv(path + "sample_submission.csv") 


print(train_csv.columns) #'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
      # 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      

  

x = train_csv.drop(['Outcome','Pregnancies','DiabetesPedigreeFunction'], axis=1)

#print(x)
y = train_csv['Outcome']
#print(y)

print(np.unique(y, return_counts= True)) #(array([0, 1]), array([424, 228])
print(pd.Series(y).value_counts()) #다 똑가틈
print(pd.DataFrame(y).value_counts()) #다 똑가틈 (dataframe을 쓰면 메모리를 더 씀,행렬데이터는 dataframe만 가능)
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.85, shuffle= False, random_state= 293)  

#2. 모델구성
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models:
    model.fit(x_train, y_train)
    results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_predict, y_test)
    print(type(model).__name__, "acc :", acc)
    print(type(model).__name__, ':', model.feature_importances_)
        


submission_csv['Outcome'] =  np.around(y_submit) #(2진분류는 소수점값으로 나옴. 답지에는 0,1로 입력해야하기때문에 반올림 해줘야함)



import time as tm

def ACC(y_test, y_predict):
    return accuracy_score(y_test, np.around(y_predict))
acc = ACC(y_test, y_predict)
print("정확도 : ", acc)


ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{acc:.3f}.csv", index=False)


####### submission.csv 만들기 (count컬럼에 값만 넣어주면 됨) #####






#로스 : 3175.002197265625
#R2 스코어 : 0.5593716340440571
#RMSE :  56.347159447801296



#로스 : [1.0998154878616333, 0.6938775777816772]
#정확도 :  0.6938775510204082


#LinearSVC
# model.score: 0.6224489795918368
# 정확도 :  0.6224489795918368

# DecisionTreeClassifier acc : 0.6326530612244898
# DecisionTreeClassifier : [0.34056487 0.10192279 0.12650353 0.0735802  0.1891239  0.16830472]
# RandomForestClassifier acc : 0.6632653061224489
# RandomForestClassifier : [0.31247469 0.09810343 0.13008101 0.08759674 0.19529702 0.1764471 ]
# GradientBoostingClassifier acc : 0.6938775510204082
# GradientBoostingClassifier : [0.43732727 0.05318482 0.07107199 0.0511532  0.23568403 0.15157869]
# XGBClassifier acc : 0.7040816326530612
# XGBClassifier : [0.32236007 0.10850321 0.10885453 0.1279968  0.17787774 0.15440768]