import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# Importing the Libraries used for creating the MachineLearning Models

def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 

# Using an array to store the test set Accuracy
list = []
ModelNames = ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'K-Nearest-Neighbors']



data = pd.read_csv('dataset_cleaned2.csv')
X = pd.read_csv('dataset_cleaned3.csv')

# Changing the list 'Class' Into a numpy array
Y = data['Class'].to_numpy()

transform = preprocessing.StandardScaler()

transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

# Splitting up the dataset into test and train. These are used for every machine learning model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Assuring that the array has the correct number of recorsd.
print(Y_test.shape)

parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}


# Using a logistic regression to predict the model
LR = LogisticRegression()
logreg_cv = GridSearchCV(estimator = LogisticRegression(), cv=10, param_grid=parameters)
logreg_cv.fit(X_train, Y_train)

# Obtaining the best parametres for the model. The one that yields the highest accuracy


parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}
lr=LogisticRegression()

logreg_cv = GridSearchCV(estimator = lr, cv=10, param_grid=parameters)
# Fit the model.
logreg_cv.fit(X_train, Y_train)

# Obtaining the best parametres for the model. The paramtres that yields the highest accuracy.
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

# Displays the accuracy of the test. Higher the Better.
print("test set accuracy :",logreg_cv.score(X_test, Y_test))
list.append(logreg_cv.score(X_test, Y_test))

yhat=logreg_cv.predict(X_test)
#Plotting a confusion matrix to see a visual representation of the data.
plot_confusion_matrix(Y_test,yhat)

# Using a Support Vector Machine (SVC) to obtain predictions for the model.
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()

svm_cv = GridSearchCV(svm,parameters,cv=10)
svm_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)

print("test set accuracy :",svm_cv.score(X_test, Y_test))
list.append(logreg_cv.score(X_test, Y_test))

yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

# Using a Deceision Tree to obtain predictions for the model.

parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()

tree_cv = GridSearchCV(tree,parameters,cv=10)
tree_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)

print("test set accuracy :",tree_cv.score(X_test, Y_test))
list.append(logreg_cv.score(X_test, Y_test))

yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}


# Using a K-Nearest-Neighbors (KNN) to obtain predictions for the model.

KNN = KNeighborsClassifier()

knn_cv = GridSearchCV(KNN,parameters,cv=10)
knn_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)

print("test set accuracy :",knn_cv.score(X_test, Y_test))
list.append(logreg_cv.score(X_test, Y_test))


yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

listSeries = pd.Series(list)

HighestAccuracy = listSeries.index(listSeries.max())
# Prints out hte model and the highest accuracy.
print('The model with the highest test accuracy was: {}. With an Accuracy of {}'.format(list[HighestAccuracy],listSeries.max()))