from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



#1. 데이터

datasets = load_wine()
print(datasets.DESCR) 
print(datasets.feature_names) #'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'       


x= datasets.data
y= datasets.target
print(x.shape, y.shape) #(178, 13) (178,)

print(np.unique(y, return_counts= True)) #(array([0, 1, 2]), array([59, 71, 48]
print(pd.value_counts(y)) # 1    71 # 0    59 # 2    48


columns = datasets.feature_names
#columns = datasets.columns
x = pd.DataFrame(x,columns=columns)
y = pd.DataFrame(y)



scaler = StandardScaler()
x_1 = scaler.fit_transform(x)
lda = LinearDiscriminantAnalysis(n_components=1)
x_1 = lda.fit_transform(x,y)  
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
print('lda_feature 갯수',x_train[1].shape,'개', 'model.score :',results)

# 갯수별 변화율
# 0.99809123 0.99982715 0.99992211 0.99997232 0.99998469 0.99999315
#  0.99999596 0.99999748 0.99999861 0.99999933 0.99999971 0.99999992
#  1.   

# 로스 : 0.027650095522403717
# 정확도 : 1.0

# linearSVC
# model.score: : 0.9722222222222222
# acc : 0.9722222222222222


# feature 갯수 1 개 model.score : 0.6388888888888888
# ===============
# feature 갯수 2 개 model.score : 0.75
# ===============
# feature 갯수 3 개 model.score : 0.8611111111111112
# ===============
# feature 갯수 4 개 model.score : 0.9444444444444444
# ===============
# feature 갯수 5 개 model.score : 0.9166666666666666
# ===============
# feature 갯수 6 개 model.score : 0.9722222222222222
# ===============
# feature 갯수 7 개 model.score : 0.9722222222222222
# ===============
# feature 갯수 8 개 model.score : 0.9722222222222222
# ===============
# feature 갯수 9 개 model.score : 0.9444444444444444
# ===============
# feature 갯수 10 개 model.score : 0.9444444444444444
# ===============
# feature 갯수 11 개 model.score : 1.0
# ===============
# feature 갯수 12 개 model.score : 1.0
# ===============
# feature 갯수 13 개 model.score : 0.9444444444444444


# lda_feature 갯수 1 개 model.score : 0.9444444444444444