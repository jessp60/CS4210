#-------------------------------------------------------------------------
# AUTHOR: Jessica Pinto
# FILENAME: knn.py
# SPECIFICATION: Program that takes in training data as samples and use the KNN classifier to classifier test data. The program outputs the error rate of the classifier at the end. 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Loop your data to allow each instance to be your test set

error = 0 
testIn = 0
train = 9
results = []
for i in db:  
    #print(db[testIn])
    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    Y = []

    for data in db:  
      X.append([float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]), float(data[7]), float(data[8]), float(data[9]), float(data[10]), float(data[11]), float(data[12]), float(data[13]), float(data[14]), float(data[15]), float(data[16]), float(data[17]), float(data[18]), float(data[19])])
    #Remove the instance for testing
    testSample = [X[testIn]]
    X.pop(testIn)
    
    #Transform the original training classes to numbers and add them to the vector Y. 
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    for data in db: 
      if data[20] == "ham": 
        Y.append(1)
      elif data[20] == "spam":
        Y.append(0)

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample.append([Y[testIn]])
    Y.pop(testIn)
    #print("Test:", testSample)
    testIn += 1

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #--> add your Python code here
    class_predicted = clf.predict([testSample[0]])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != testSample[1][0]: 
      error += 1

#Print the error rate
#--> add your Python code here
print("Error Rate for KNN:",round(error/(len(db)-1), 2))
