from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np

#1 데이터
path = "c:\\_data\\dacon\\diabetes\\"
train_csv = pd.read_csv(path + "train.csv",index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path +"sample_submission.csv",)
print(submission_csv)
print(train_csv.shape) 
print(test_csv.shape) 
print(submission_csv.shape) 
print(train_csv.columns)
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

###########x와 y의 값을 분리
x= train_csv.drop(['Outcome'], axis=1) 
print(x)
y = train_csv['Outcome']
print(y)

x_train,x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.3,shuffle=False, random_state=100,)
print(x_train.shape,x_test.shape) #(2177, 8) (8709, 8)
print(y_train.shape, y_test.shape) #(2177,) (8709,)


#2. 모델
model = LinearSVC(C=100)

#3. 컴파일,훈련
model.fit(x_train, y_train)

#4. 평가,예측
results = model.score(x_test, y_test)
print("model.score:", results)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
print(y_predict)
accuracy_score = accuracy_score(y_predict,y_test)
print("accuracy_score:", accuracy_score)


print("===================================")
#########submission.csv 만들기(count 컬럼에 값만 제출)#############
submission_csv['Outcome'] = np.around(y_submit)
print(submission_csv)
print(submission_csv.shape)
submission_csv.to_csv(path+"submission_0207.csv", index=False)

