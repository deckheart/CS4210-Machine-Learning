#-------------------------------------------------------------------------
# AUTHOR: Dakota Eckheart
# FILENAME: knn.py
# SPECIFICATION: Find kth nearest neighbor and output LOO-CV error rate
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
X = []
Y = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

errors = 0

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]
    temp_row = []
    temp_row.append(int(instance[0]))
    temp_row.append(int(instance[1]))
    X.append(temp_row)
    #print(X)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]
    class_dict = {
        '+': 1,
        '-': 2
    }
    
    Y.append(class_dict.get(instance[-1]))
    #print(Y)
    
    #store the test sample of this iteration in the vector testSample
    testSample = temp_row

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    class_predicted = clf.predict([testSample])[0]
    print("Predicted: ", class_predicted, "\tActual: ", Y[-1])

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    
    if class_predicted != Y[-1]:
        errors += 1

#print the error rate
print("Error rate: ", errors/len(db))
