import numpy as np
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import (GridSearchCV, cross_val_score, cross_val_predict,
                                     StratifiedKFold, learning_curve)

# define arrays
x = []
y = []

# Open the training data file and load line by line with splitter (,) and append them in x , y
with open ("TrainingData.txt") as r:
    line = r.readlines()
    line = line[:]
    for i, item in enumerate(line):
        line[i] = line[i].strip("\n").split(",")
        line[i] = [float(v) for v in line[i]]
        x.append(line[i][0:-1])
        y.append(line[i][-1])


# spliting the data to  training 80%  & for test and validation 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x = np.array(x)
y = np.array(y)
x_train_full = x
y_train_full = y




#Gaussian Naive Bayes Classifier
gNb = GaussianNB()
gNb.fit(x_train, y_train)
print('\n################           GNB before Tuning          ##############')
print("\nGaussian Naive Bayes Classifier Accuracy in 80% training data :",gNb.score(x_train_full, y_train_full))
print("Gaussian Naive Bayes Classifier Accuracy in 20% test data       :",gNb.score(x_test, y_test))

from sklearn.model_selection import RepeatedStratifiedKFold
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
cv_method = RepeatedStratifiedKFold(n_splits=5,  n_repeats=3, random_state=999)
gs_NB = GridSearchCV(estimator=gNb, param_grid=params_NB, cv=cv_method,verbose=1,scoring='accuracy')
gs_NB.fit(x_train, y_train)
print("best parameterers")
print(gs_NB.best_estimator_)
model = gs_NB.best_estimator_

print('\n################ GNB Results after Tuning hyper-parameters ##############')
print("Gaussian Naive Bayes Classifier Accuracy in 80% training data :",gs_NB.score(x_train, y_train))
print("Gaussian Naive Bayes Classifier Accuracy in 20% test data       :",gs_NB.score(x_test, y_test))

print('\n##############################################################################')

#Linear Discriminant Analysis
lDa= LinearDiscriminantAnalysis()
lDa.fit(x_train, y_train)
print('\n\n\n################           LDA before Tuning          ##############')
print("\nLinear Discriminant Analysis Accuracy in 80% training data :",lDa.score(x_train_full, y_train_full))
print("Linear Discriminant Analysis Accuracy in 20% test data       :",lDa.score(x_test, y_test))

# Linear Discriminant Analysis - Parameter Tuning
LDA = LinearDiscriminantAnalysis()
K_fold = StratifiedKFold(n_splits=10)

## Search grid for optimal parameters
lda_param_grid = {"solver" : ["svd"],
                  "tol" : [0.0001,0.0002,0.0003]}


gsLDA = GridSearchCV(LDA, param_grid = lda_param_grid, cv=K_fold,
                     scoring="accuracy", n_jobs= 4, verbose = 1)

gsLDA.fit(x_train,y_train)
LDA_best = gsLDA.best_estimator_

print('\n################ LDA Results after Tuning hyper-parameters ##############')
print("\nLinear Discriminant Analysis Accuracy in 80% training data  :",lDa.score(x_train, y_train))
print("Linear Discriminant Analysis Accuracy in 20% test data        :",lDa.score(x_test, y_test))

