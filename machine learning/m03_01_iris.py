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

# LinearSVC(): Linear Support Vector Classification. It's similar to SVC with a linear kernel. 
# It's implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions 
# and should scale better to large numbers of samples.

# Perceptron(): The Perceptron is a simple algorithm suitable for large scale learning. It's a type of linear classifier,
# i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

# LogisticRegression(): Logistic Regression (aka logit, MaxEnt) classifier. 
# This model uses the logistic function to squeeze the output of a linear equation between 0 and 1. 
# The logistic regression model is used in statistics to model the probability of a certain class or event existing such as pass/fail, win/lose, etc.

# KNeighborsClassifier(): Classifier implementing the k-nearest neighbors vote. 
# It's a type of instance-based learning, or lazy learning, where the function is only approximated locally
# and all computation is deferred until classification.

# DecisionTreeClassifier(): A decision tree classifier. Given a data of attributes together with its classes, 
# a decision tree produces a sequence of rules that can be used to classify the data.

# RandomForestClassifier(): A random forest is a meta estimator that fits a number of decision tree classifiers 
# on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
# The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).




# Iterate over models
for model in models:
    # Fit the model
    model.fit(x_train, y_train)
    
    # Evaluate the model
    score = model.score(x_test, y_test)
    print(f'{model.__class__.__name__}: {score}')