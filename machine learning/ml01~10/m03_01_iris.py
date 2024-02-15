'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#data
datasets = load_iris()
x,y = load_iris(return_X_y=True)
print(x.shape, y.shape) #(150, 4) (150,)
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=77,stratify=y)

# model=LinearSVC(C=100)
#C가 크면 training point들을 더 잘 분류하려고 노력함.(과적합), C가 작으면 training point들을 덜 잘 분류하려고 노력함.(과소적합 )
model = Perceptron()
model = LogisticRegression()
model = KNeighborsClassifier()
model = DecisionTreeClassifier()
model = RandomForestClassifier()

model.fit(x_train,y_train)

results = model.score(x_test,y_test)
print("model.score:", results)
y_predict = model.predict(x_test)  
print(y_predict) 
acc = accuracy_score(y_predict,y_test)
print("acc:", acc)

'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#data
datasets = load_iris()
x,y = load_iris(return_X_y=True)
print(x.shape, y.shape) #(150, 4) (150,)
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=77,stratify=y)


# Define a list of models
models = [
    LinearSVC(),
    Perceptron(),
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

# Iterate over models
for model in models:
    # Fit the model
    model.fit(x_train, y_train)
    
    # Evaluate the model
    score = model.score(x_test, y_test)
    print(f'{model.__class__.__name__}: {score}')