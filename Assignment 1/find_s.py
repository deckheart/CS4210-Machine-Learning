#-------------------------------------------------------------------------
# AUTHOR: Dakota Eckheart
# FILENAME: find_s.py
# SPECIFICATION: Find maximally specific hypothesis of data set
# FOR: CS 4210- Assignment #1
# TIME SPENT: ~1.5 hours for find_s algorithm
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
import csv

num_attributes = 4
db = []
print("\n The Given Training Data Set \n")

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes #representing the most specific possible hypothesis
print(hypothesis)

#find the first positive training data in db and assign it to the vector hypothesis
# Looks through training data, finds the first 'Yes' class and ignores the following rows
for row in db:
  if row[-1] == 'Yes':
    hypothesis = row
    break


#find the maximally specific hypothesis according to your training data in db and assign it to the vector hypothesis (special characters allowed: "0" and "?")
for row in db:
  if row[-1] == 'Yes':
    i = 0   # index to use for hypothesis list
    for feature in row:
      if feature != hypothesis[i]:
        hypothesis[i] = '?'
      i += 1

hypothesis.pop()  # get rid of assigned class (the final column value)
print("\n The Maximally Specific Hypothesis for the given training examples found by Find-S algorithm:\n")
print(hypothesis) # for contact_lens, should print ['?', 'Myope', '?', '?']