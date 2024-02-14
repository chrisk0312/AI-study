from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data
datasets = load_iris()
x = datasets.data
y = datasets['target']

df= pd.DataFrame(x, columns=datasets.feature_names)
print(df)
df['target'] = y    
print(df)

print('============================================')
print(df.corr())


import seaborn as sns
print(sns.__version__) # 0.11.2

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), 
            square=True, 
            annot=True, # 실제 값 화면에 나타내기
            cbar=True) # color bar를 표시할지 여부
plt.show()

'''
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=123,stratify=y)


# Define a list of models
model1 = DecisionTreeClassifier(random_state=666)
model2 = RandomForestClassifier(random_state=666)
model3 = GradientBoostingClassifier(random_state=666)
model4 = customXGBClassifier(random_state=666)
models = [model1, model2, model3, model4]

for model in models:
    model.fit(x_train, y_train)
    print("=========================================")
    print(model)
    print("acc:", model.score(x_test, y_test))
    print("model.feature_importances_:", model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#              align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)
#     plt.title(model)

# plot_feature_importances_dataset(model3)
# plt.show()

from xgboost import plot_importance
plot_importance(model)
plt.show()

'''