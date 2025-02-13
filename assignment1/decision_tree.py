#-------------------------------------------------------------------------
# AUTHOR: Jessica Pinto
# FILENAME: decision_tree.py
# SPECIFICATION: This program takes in data from contact_lens.csv, converts the data into numeric values, and creates a decision tree based on the given data. 
# FOR: CS 4210- Assignment #1
# TIME SPENT: Total assignment (including Assignment_1.pdf) time was 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
#--> add your Python code here
# X = 
for row in db:
  # age 
  match row[0]: 
    case "Young": 
      row[0] = 1
    case "Presbyopic": 
      row[0] = 2
    case "Prepresbyopic": 
      row[0] = 3
  # spectacle prescrption 
  match row[1]: 
    case "Myope": 
      row[1] = 1
    case "Hypermetrope": 
      row[1] = 2
  # astigmatism
  match row[2]:
    case "Yes": 
      row[2] = 1
    case "No": 
      row[2] = 2
  # tear production rate 
  match row[3]:
    case "Reduced": 
      row[3] = 1
    case "Normal": 
      row[3] = 2
  X.append([row[0], row[1], row[2], row[3]])

    
     
#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
#--> addd your Python code here
# Y =
for row in db: 
  match row[4]:
    case "Yes": 
      row[4] = 1
    case "No": 
      row[4] = 2
  Y.append([row[4]])

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
