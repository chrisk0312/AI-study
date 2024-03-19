from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class customXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'



#data
datasets = load_iris()
x,y = load_iris(return_X_y=True)
print(x.shape, y.shape) #(150, 4) (150,)

datasets = load_iris()
x = datasets.data
y = datasets['target']


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

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    plt.title(model)

plot_feature_importances_dataset(model3)
plt.show()
