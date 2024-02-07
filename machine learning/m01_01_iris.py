from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC


#data
datasets = load_iris()
x = datasets.data
y = datasets.target
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=77,stratify=y)


model=LinearSVC(C=100)
#C가 크면 training point들을 더 잘 분류하려고 노력함.(과적합), C가 작으면 training point들을 덜 잘 분류하려고 노력함.(과소적합 )

model.fit(x_train,y_train)


results = model.score(x_test,y_test)
print("model.score:", results)
y_predict = model.predict(x_test)  
print(y_predict) 
acc = accuracy_score(y_predict,y_test)
print("acc:", acc)