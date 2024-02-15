from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC


#data
datasets = load_iris()
x = datasets.data
y = datasets.target
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=77,stratify=y)

# Support Vector Machines (SVMs) are a set of supervised learning methods used for classification, regression, and outliers detection. 
# LinearSVC is an implementation of Support Vector Classification for the case of a linear kernel.
# The objective of a Linear Support Vector Machine is to find a hyperplane in an N-dimensional space 
# (where N is the number of features) that distinctly classifies the data points.
# To separate the two classes of data points, there are many possible hyperplanes that could be chosen.
# The objective of an SVM is to find a plane that has the maximum margin, i.e., the maximum distance between data points of both classes.
# LinearSVC in scikit-learn, by default, handles binary classification, but it can also handle multi-class classification.


model=LinearSVC(C=100)
#C가 크면 training point들을 더 잘 분류하려고 노력함.(과적합), C가 작으면 training point들을 덜 잘 분류하려고 노력함.(과소적합 )

model.fit(x_train,y_train)


results = model.score(x_test,y_test)
print("model.score:", results)
y_predict = model.predict(x_test)  
print(y_predict) 
acc = accuracy_score(y_predict,y_test)
print("acc:", acc)