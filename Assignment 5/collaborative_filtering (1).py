# -*- coding: utf-8 -*-
"""collaborative_filtering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vP_5SCoua5__36xi5NwdAdaM_FAxamvi
"""

#-------------------------------------------------------------------------
# AUTHOR: Dakota Eckheart
# FILENAME: collaborative_filtering.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: 2 hours for collaborative_filtering.py
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()
user100 = df.iloc[[-1]]   # get last row
del user100['User ID']
del user100['galleries']
del user100['restaurants']

similarities = []
#print(user100)

#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
for i, row in df.iterrows():
  if i == len(df)-1:
    break   # avoid relating User 100 to themself
  vec1 = np.array([[row['dance clubs'], row['juice bars'], row['museums'], row['resorts'], row['parks/picnic spots'], row['beaches'], row['theaters'], row['religious institutions']]])
  vec2 = np.array([user100['dance clubs'], user100['juice bars'], user100['museums'], user100['resorts'], user100['parks/picnic spots'], user100['beaches'], user100['theaters'], user100['religious institutions']]).reshape(1,8)

  cs = cosine_similarity(vec1, vec2)
  similarities.append((i, cs[0][0]))

#find the top 10 similar users to the active user according to the similarity calculated before
top_ten = sorted(similarities, key=lambda t: t[1], reverse=True)[:10]
print(top_ten)

#Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
galleries_weighted = []
restaurants_weighted = []

for tup in top_ten:
  weighted_gallery = float(df.at[tup[0], 'galleries']) * tup[1]
  weighted_restaurant = float(df.at[tup[0], 'restaurants']) * tup[1]
  galleries_weighted.append(weighted_gallery)
  restaurants_weighted.append(weighted_restaurant)

print("Galleries prediction: ", str(round(sum(galleries_weighted)/len(galleries_weighted), 2)))
print("Restaurant prediction: ", str(round(sum(restaurants_weighted)/len(restaurants_weighted), 2)))