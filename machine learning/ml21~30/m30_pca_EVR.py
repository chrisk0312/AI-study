from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)#(442, 10) (442,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=18) #n_components= : 몇개로 차원을 축소할 것인지
x = pca.fit_transform(x)
print(x)
print(x.shape) #(442, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle=True)

model = RandomForestClassifier()

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result) #0.9666666666666667

EVR = pca.explained_variance_ratio_
print(EVR)
print(sum(EVR)) #1.0

evr_cumsum=np.cumsum(EVR)
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()

# (150, 4) (150,)
# (150, 4)
# 0.9333333333333333

# When you perform PCA on a dataset, it calculates the principal components of the data. 
# Each principal component is a linear combination of the features that explains a certain amount of variance in the data. 
# The explained_variance_ratio_ attribute gives you the proportion of the dataset's variance that lies along the axis of each principal component.