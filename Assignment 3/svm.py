#-------------------------------------------------------------------------
# AUTHOR: Dakota Eckheart
# FILENAME: svm.py
# SPECIFICATION: Use different parameters to find the best SVM classifier
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1 hour for svm.py
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv
import os

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0
best_c = 0
best_k = ''
best_d = 0
best_dfs = ''

#reading the data in a csv file
optdigits_tra = os.path.dirname(os.path.abspath(__file__)) + '/optdigits.tra'
with open(optdigits_tra, 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1])

#reading the data in a csv file
optdigits_tes = os.path.dirname(os.path.abspath(__file__)) + '/optdigits.tes'
with open(optdigits_tes, 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for _c in c: #iterates over c
    for d in degree: #iterates over degree
        for k in kernel: #iterates kernel
           for dfs in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=_c, kernel=k, degree=d, decision_function_shape=dfs)

                #Fit Random Forest to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                accuracy = 0
                for i, row in enumerate(dbTest):
                    test = row[:-1]
                    class_predicted = clf.predict([test])[0]
                    if int(class_predicted) == int(row[-1]):
                        accuracy += 1

                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                accuracy = accuracy/len(dbTest)
                if accuracy > highestAccuracy:
                    highestAccuracy = accuracy
                    best_c = _c
                    best_d = d
                    best_k = k
                    best_dfs = dfs
                    print("Highest SVM accuracy so far: ", highestAccuracy, 
                    " Parameters: c=", _c, 
                    " degree=", d, 
                    " kernel=", k, 
                    " decision_function_shape = ", dfs)

#print the final, highest accuracy found together with the SVM hyperparameters
#Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
print("\nHighest SVM accuracy: ", highestAccuracy, 
" Parameters: c=", best_c, 
" degree=",best_d, 
" kernel= ", best_k, 
" decision_function_shape = ", best_dfs)













