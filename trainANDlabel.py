import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

#Reading training data
#convert dataset to pandas DataFrame
trainData = pd.read_csv('TrainingData.txt', header=None)

#Reading the data from TestingData.txt to test 
testData = pd.read_csv('TestingData.txt', header=None)
x_label = testData.values.tolist()

#define predictor and response variables
dataY = trainData[24].tolist()                #predictor independent variable (the data)
trainData = trainData.drop(24, axis=1)  	#delete the labels in training data
dataX = trainData.values.tolist()             #response dependent variable   (the label 0 or 1)

#Storing full training data before splitting
dataX = np.array(dataX)
dataY = np.array(dataY)
x_train_full = dataX
y_train_full = dataY

## spliting the data to  training 80%  & for test and validation 20% 
# training & testing the data 
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=0)


#Applying algorithm Linear Discriminant Analysis to predict the label for test data 
lDa = LDA()
lDa.fit(x_train, y_train)     # train the model 
y_pred = lDa.predict(x_label)   #prediction process

#Copying the data resulting from the test to the file TestingResults.txt
predData = pd.DataFrame({'Prediction': y_pred})
testData = testData.join(predData)
testData.to_csv("TestingResults.txt", header=None, index=None)
print("\n Done !!!!")
print("\n the file TestingResults.txt is completed the the labels")
#print(classification_report(y_classify,y_pred))
