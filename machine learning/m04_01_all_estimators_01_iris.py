from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

#data
datasets = load_iris()
x,y = load_iris(return_X_y=True)
print(x.shape, y.shape) #(150, 4) (150,)
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=77,stratify=y)


allAlgorithms = all_estimators(type_filter='classifier')

print("allAlogrithms :", allAlgorithms)
print("모델의 갯수 :", len(allAlgorithms))

for name, algorithm in allAlgorithms:
    try:
        # model
        model = algorithm()
        #fit
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(name, "의 정답률 = ", acc)
    except:
        # print(name, "은 없음!")
        continue
