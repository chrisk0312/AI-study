from sklearn.datasets import load_iris,  load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np

datasets = load_digits()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape) #(1797, 64) (1797,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

lda = LinearDiscriminantAnalysis(n_components=2) #n_components= : 몇개로 차원을 축소할 것인지
x = lda.fit_transform(x,y)
print(x)
print(x.shape) #(150, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle=True, stratify=y)


model = RandomForestClassifier()

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result) #0.9666666666666667

# (150, 4) (150,)
# (150, 4)
# 0.9333333333333333

EVR = lda.explained_variance_ratio_
print(EVR)
print(sum(EVR)) #1.0

evr_cumsum=np.cumsum(EVR)
print(evr_cumsum)