#-------------------------------------------------------------------------
# AUTHOR: Dakota Eckheart
# FILENAME: naive_bayes.py
# SPECIFICATION: Classify instances with high confidence using Naive Bayes algorithm
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

db_train = []
X = []
Y = []

#reading the training data
#reading the data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db_train.append (row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
outlook_dict = {
    'Sunny': 1,
    'Overcast': 2,
    'Rain': 3
}
temperature_dict = {
    'Hot': 1,
    'Mild': 2,
    'Cool': 3
}
humidity_dict = {
    'High': 1,
    'Normal': 2
}
wind_dict = {
    'Strong': 1,
    'Weak': 2
}

for instance in db_train:
    temp_instance = []
    temp_instance.append(outlook_dict.get(instance[1]))
    temp_instance.append(temperature_dict.get(instance[2]))
    temp_instance.append(humidity_dict.get(instance[3]))
    temp_instance.append(wind_dict.get(instance[4]))
    X.append(temp_instance)

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
playTennis_dict = {
    'Yes': 1,
    'No': 2
}
for instance in db_train:
    Y.append(instance[-1])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

db_test = []
Xt = []
Yt = []

#reading the data in a csv file
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db_test.append(row)

for instance in db_test:
    temp_instance = []
    temp_instance.append(outlook_dict.get(instance[1]))
    temp_instance.append(temperature_dict.get(instance[2]))
    temp_instance.append(humidity_dict.get(instance[3]))
    temp_instance.append(wind_dict.get(instance[4]))
    Xt.append(temp_instance)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]
for instance in Xt:
    predicted = clf.predict_proba([instance])[0]
    result = []

    if predicted[0] >= 0.75 or predicted[1] >= 0.75:
        if predicted[0] > predicted[1]:
            result.append('No')
            result.append(predicted[0])
        else:
            result.append('Yes')
            result.append(predicted[1])
    else:
        if predicted[0] > predicted[1]:
            result.append('No?')
            result.append(predicted[0])
        else:
            result.append('Yes?')
            result.append(predicted[1])
    Yt.append(result)

for instance_x, class_y in zip(db_test, Yt):
    print (str(instance_x[0]).ljust(15) + instance_x[1].ljust(15) + 
        instance_x[2].ljust(15) + instance_x[3].ljust(15) + 
        instance_x[4].ljust(15) + class_y[0].ljust(15) + 
        str(round(class_y[1], 2)).ljust(15))
