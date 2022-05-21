import numpy as np
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

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


print("\nComparison between machine learning algorithms with 80% training data  & 20% testing data:- \n")

#Linear Discriminant Analysis
lDa= LinearDiscriminantAnalysis()
lDa.fit(x_train, y_train)
print("\nLinear Discriminant Analysis Accuracy in 80% training data :",lDa.score(x_train_full, y_train_full))
print("Linear Discriminant Analysis Accuracy in 20% test data       :",lDa.score(x_test, y_test))


#KNeighbors Classifier
knn = KNeighborsClassifier(n_neighbors=5) #5 us the default
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
y_pred_train = knn.predict(x_train_full)
print("\nKNeighbors Classifier Accuracy in 80% training data        : ",metrics.accuracy_score(y_train_full, y_pred_train))
print("KNeighbors Classifier Accuracy in 20% test data              :",metrics.accuracy_score(y_test, y_pred))

#Gaussian Naive Bayes Classifier
gNb = GaussianNB()
gNb.fit(x_train, y_train)
print("\nGaussian Naive Bayes Classifier Accuracy in 80% training data :",gNb.score(x_train_full, y_train_full))
print("Gaussian Naive Bayes Classifier Accuracy in 20% test data       :",gNb.score(x_test, y_test))

#Decision Tree Classifier
dt = DecisionTreeClassifier().fit(x_train, y_train)
print("\nDecision Tree classifier Accuracy in 80% training data        :",dt.score(x_train_full, y_train_full))
print("Decision Tree classifierAccuracy in 20% test data               :",dt.score(x_test, y_test))


