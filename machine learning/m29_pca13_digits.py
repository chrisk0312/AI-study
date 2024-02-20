from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time


datasets = load_digits() #mnist 원판
x = datasets.data
y = datasets.target
print(x)
print(y)

print(x.shape) #(1797, 64)
print(y.shape) #(1797,)
print(pd.value_counts(y, sort= False)) #sort= False 제일 앞 데이터부터 순서대로 나옴
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, stratify=y)

from sklearn.ensemble import RandomForestClassifier
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
    model = RandomForestClassifier(random_state=777)

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
feature 갯수 1 개 model.score : 0.3488888888888889
===============
feature 갯수 2 개 model.score : 0.6355555555555555
===============
feature 갯수 3 개 model.score : 0.7777777777777778
===============
feature 갯수 4 개 model.score : 0.8333333333333334
===============
feature 갯수 5 개 model.score : 0.8911111111111111
===============
feature 갯수 6 개 model.score : 0.9222222222222223
===============
feature 갯수 7 개 model.score : 0.9444444444444444
===============
feature 갯수 8 개 model.score : 0.9533333333333334
===============
feature 갯수 9 개 model.score : 0.9577777777777777
===============
feature 갯수 10 개 model.score : 0.9644444444444444
===============
feature 갯수 11 개 model.score : 0.9711111111111111
===============
feature 갯수 12 개 model.score : 0.9688888888888889
===============
feature 갯수 13 개 model.score : 0.9711111111111111
===============
feature 갯수 14 개 model.score : 0.9688888888888889
===============
feature 갯수 15 개 model.score : 0.9622222222222222
===============
feature 갯수 16 개 model.score : 0.9666666666666667
===============
feature 갯수 17 개 model.score : 0.9688888888888889
===============
feature 갯수 18 개 model.score : 0.9733333333333334
===============
feature 갯수 19 개 model.score : 0.9711111111111111
===============
feature 갯수 20 개 model.score : 0.9733333333333334
===============
feature 갯수 21 개 model.score : 0.9733333333333334
===============
feature 갯수 22 개 model.score : 0.9733333333333334
===============
feature 갯수 23 개 model.score : 0.9688888888888889
===============
feature 갯수 24 개 model.score : 0.9777777777777777
===============
feature 갯수 25 개 model.score : 0.9711111111111111
===============
feature 갯수 26 개 model.score : 0.9733333333333334
===============
feature 갯수 27 개 model.score : 0.9755555555555555
===============
feature 갯수 28 개 model.score : 0.9733333333333334
===============
feature 갯수 29 개 model.score : 0.9777777777777777
===============
feature 갯수 30 개 model.score : 0.9755555555555555
===============
feature 갯수 31 개 model.score : 0.9777777777777777
===============
feature 갯수 32 개 model.score : 0.9755555555555555
===============
feature 갯수 33 개 model.score : 0.9777777777777777
===============
feature 갯수 34 개 model.score : 0.9733333333333334
===============
feature 갯수 35 개 model.score : 0.9777777777777777
===============
feature 갯수 36 개 model.score : 0.9733333333333334
===============
feature 갯수 37 개 model.score : 0.9755555555555555
===============
feature 갯수 38 개 model.score : 0.9644444444444444
===============
feature 갯수 39 개 model.score : 0.98
===============
feature 갯수 40 개 model.score : 0.98
===============
feature 갯수 41 개 model.score : 0.9711111111111111
===============
feature 갯수 42 개 model.score : 0.9666666666666667
===============
feature 갯수 43 개 model.score : 0.9733333333333334
===============
feature 갯수 44 개 model.score : 0.9822222222222222
===============
feature 갯수 45 개 model.score : 0.9688888888888889
===============
feature 갯수 46 개 model.score : 0.9777777777777777
===============
feature 갯수 47 개 model.score : 0.9711111111111111
===============
feature 갯수 48 개 model.score : 0.9666666666666667
===============
feature 갯수 49 개 model.score : 0.9755555555555555
===============
feature 갯수 50 개 model.score : 0.9777777777777777
===============
feature 갯수 51 개 model.score : 0.98
===============
feature 갯수 52 개 model.score : 0.9644444444444444
===============
feature 갯수 53 개 model.score : 0.9688888888888889
===============
feature 갯수 54 개 model.score : 0.9711111111111111
===============
feature 갯수 55 개 model.score : 0.9644444444444444
===============
feature 갯수 56 개 model.score : 0.9644444444444444
===============
feature 갯수 57 개 model.score : 0.9711111111111111
===============
feature 갯수 58 개 model.score : 0.9666666666666667
===============
feature 갯수 59 개 model.score : 0.9711111111111111
===============
feature 갯수 60 개 model.score : 0.9755555555555555
===============
feature 갯수 61 개 model.score : 0.9755555555555555
===============
feature 갯수 62 개 model.score : 0.9711111111111111
===============
feature 갯수 63 개 model.score : 0.9755555555555555
===============
feature 갯수 64 개 model.score : 0.9644444444444444

0.14598593 0.27997621 0.39867748 0.48367699 0.54291732 0.59277794
 0.63560098 0.6721849  0.70700088 0.73793916 0.76261025 0.78585004
 0.80401763 0.82083254 0.83600055 0.84987312 0.8628955  0.87515963
 0.88522159 0.89448827 0.903457   0.91149853 0.91916217 0.92651022
 0.93344951 0.93932563 0.94485324 0.94997775 0.95473147 0.95910851
 0.96280498 0.96629541 0.96975197 0.97316114 0.9762042  0.97911553
 0.98154929 0.98379031 0.9860012  0.98810692 0.9899617  0.99156222
 0.99309251 0.99451571 0.99572712 0.99679455 0.99773953 0.99860956
 0.99916477 0.99952573 0.99974834 0.99983471 0.99988536 0.99993459
 0.99997588 0.999985   0.99999388 0.99999784 0.99999886 0.99999959
 1.         1.         1.         1.       

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_jobs": [-1],"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"n_jobs": [-1],"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"n_jobs": [-1],"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1],"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1], "min_samples_split": [2, 3, 5, 10]},
]    

# model = RandomForestClassifier()

model = GridSearchCV(RandomForestClassifier(), 
                     parameters, 
                     cv=kfold, 
                     verbose=1, 
                     refit= True, #디폴트 트루~
                     )#n_jobs=-1) #CPU 다 쓴다!



# model = RandomizedSearchCV(RandomForestClassifier(), 
#                      parameters, 
#                      cv=kfold, 
#                      verbose=1, 
#                      refit= True, #디폴트 트루~
#                      n_jobs=-1) #CPU 다 쓴다!

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

y_predict = model.predict(x_test)
print("acc_score", accuracy_score(y_test, y_predict))

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")

#randomforest
# acc_score 0.9866666666666667
# 걸린시간 : 0.14 초

#RandomsearchCV
# acc_score 0.98
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3, min_samples_split=5, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 5, 'min_samples_leaf': 3}
# best_score : 0.965857083849649
# score : 0.98
# 최적튠 ACC : 0.98
# 걸린시간 : 1.74 초

#GridSearchCV
# acc_score 0.9866666666666667
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': -1}
# best_score : 0.9725237505163156
# score : 0.9866666666666667
# 최적튠 ACC : 0.9866666666666667
# 걸린시간 : 17.16 초
'''
