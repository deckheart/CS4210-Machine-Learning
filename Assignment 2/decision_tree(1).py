#-------------------------------------------------------------------------
# AUTHOR: Dakota Eckheart
# FILENAME: decision_tree.py
# SPECIFICATION: Create decision trees and prediction model for lenses
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
dataSet_accuracies = []

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    feature_dict = {
        'Young': 1,
        'Presbyopic': 2,
        'Prepresbyopic': 3,
        'Myope': 4,
        'Hypermetrope': 5,
        'No': 6,
        'Yes': 7,
        'Reduced': 8,
        'Normal': 9
    }
    for row in dbTraining:
        temp_row = []
        for i in range(len(row)-1):
            x = feature_dict.get(row[i])
            temp_row.append(x)
        X.append(temp_row)

    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    class_dict = {
        'Yes': 1,
        'No': 2
    }
    for row in dbTraining:
        Y.append(class_dict.get(row[-1]))

    lowest_accuracy = 1

    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        dbTest = []
        Xt = []
        Yt = []

        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)

        for row in dbTest:
            temp_row = []
            for i in range(len(row)-1):
                x = feature_dict.get(row[i])
                temp_row.append(x)
            Xt.append(temp_row)

        for row in dbTest:
            Yt.append(class_dict.get(row[-1]))
        

        temp_accuracies = []
        correct = 0

        for data, training_class in zip(Xt, Yt):
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            class_predicted = clf.predict([data])[0]

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if training_class == class_predicted:
                correct += 1
        
        if lowest_accuracy > (correct/len(Xt)):
            lowest_accuracy = (correct/len(Xt))


    #find the lowest accuracy of this model during the 10 runs (training and test set)
    dataSet_accuracies.append(lowest_accuracy)

#print the lowest accuracy of this model during the 10 runs (training and test set).
print('final accuracy when training on contact_lens_training_1.csv: ', dataSet_accuracies[0])
print('final accuracy when training on contact_lens_training_2.csv: ', dataSet_accuracies[1])
print('final accuracy when training on contact_lens_training_3.csv: ', dataSet_accuracies[2])
