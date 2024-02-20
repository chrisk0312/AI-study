#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


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


# #columns = train_csv.feature_names
# columns = train_csv.columns
# x = pd.DataFrame(x,columns=columns)

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
#   print(evr)
 #  print(sum(evr))
    evr_cunsum = np.cumsum(evr)
    print(evr_cunsum)      
'''
갯수별
#0.90117744 0.96071785 0.98441032 0.99211935 0.99789987 1.     

y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)
y_submit = model.predict(x_test)


submission_csv['Outcome'] =  np.around(y_submit) #(2진분류는 소수점값으로 나옴. 답지에는 0,1로 입력해야하기때문에 반올림 해줘야함)



import time as tm

def ACC(y_test, y_predict):
    return accuracy_score(y_test, np.around(y_predict))
acc = ACC(y_test, y_predict)
print("정확도 : ", acc)


ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{acc:.3f}.csv", index=False)


'''

# feature 갯수 1 개 model.score : 0.6412213740458015
# ===============
# feature 갯수 2 개 model.score : 0.6870229007633588
# ===============
# feature 갯수 3 개 model.score : 0.7022900763358778
# ===============
# feature 갯수 4 개 model.score : 0.7099236641221374
# ===============
# feature 갯수 5 개 model.score : 0.7251908396946565
# ===============
# feature 갯수 6 개 model.score : 0.7251908396946565