#https://dacon.io/competitions/open/236070/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler


#1. 데이터
path = "C:\\_data\\daicon\\iris\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv") 
print(submission_csv)


print(train_csv.shape) # (120, 5)
print(test_csv.shape) # (30, 4)
print(submission_csv.shape) # (30, 2)

print(train_csv.columns) #'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
#       'petal width (cm)', 'species']

print(train_csv.info())
print(test_csv.info())

x = train_csv.drop(['species'], axis = 1)
#print(x)

y = train_csv['species']

#print(y)

print(x.shape, y.shape) #(120, 4) (120,)
print(np.unique(y, return_counts= True)) #array([0, 1, 2] array([40, 41, 39]


columns = x.columns

#columns = datasets.columns
x = pd.DataFrame(x,columns=columns)
y = pd.DataFrame(y)
print(y.shape)

y_=[1,2,3]
for i in range(len(y_)):
    scaler = StandardScaler()
    x_1 = scaler.fit_transform(x)
    lda = LinearDiscriminantAnalysis(n_components=i+1)
    x_1 = lda.fit_transform(x_1,y)  
    x_train, x_test, y_train, y_test = train_test_split(x_1, y, train_size=0.8, random_state=777, shuffle= True, stratify=y)

    #2. 모델
    model = RandomForestClassifier(random_state=777)

    #3. 훈련
    model.fit(x_train, y_train)

    #4. 평가, 예측
    results = model.score(x_test, y_test)
    print('===============')
    #print(x.shape)
    print(x_1.shape)
    print('lda_feature 갯수',i+1,'개', 'model.score :',results)
        



# feature 갯수 1 개 model.score : 0.9166666666666666
# ===============
# feature 갯수 2 개 model.score : 1.0
# ===============
# feature 갯수 3 개 model.score : 0.9583333333333334
# ===============
# feature 갯수 4 개 model.score : 0.9583333333333334

# 0.91959926 0.05714377 0.01838378 0.00487319]
# 1.0000000000000002      

#lda_feature 갯수 1 개 model.score : 0.9166666666666666
#lda_feature 갯수 2 개 model.score : 1.0