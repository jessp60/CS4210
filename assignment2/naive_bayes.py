#-------------------------------------------------------------------------
# AUTHOR: Jessica Pinto 
# FILENAME: naive_bayes.py
# SPECIFICATION: Program that takes in training and testing information from a csv file to be used in classification. Once the classificaiton has been made, the program will print the results that match the specified threshold.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

dbTraining = []
#Reading the training data in a csv file
#--> add your Python code here
with open("assignment2/weather_training.csv", 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append([row[1], row[2], row[3], row[4], row[5]])

X = []
Y = []
#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
for row in dbTraining: 
        match row[0]: 
            case "Sunny": 
                row[0] = 1
            case "Overcast": 
                row[0] = 2
            case "Rain": 
                row[0] = 3
        # spectacle prescrption 
        match row[1]: 
            case "Hot": 
                row[1] = 1
            case "Mild": 
                row[1] = 2
            case "Cool": 
                row[1] = 3
        # astigmatism
        match row[2]:
            case "High": 
                row[2] = 1
            case "Normal": 
                row[2] = 2
        # tear production rate 
        match row[3]:
            case "Strong": 
                row[3] = 1
            case "Weak": 
                row[3] = 2
        match row[4]: 
            case "Yes": 
                row[4] = 1
            case "No": 
                row[4] = 0 
        X.append([row[0], row[1], row[2], row[3]])
        Y.append(row[4])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
dbTesting = []
with open("assignment2/weather_test.csv", 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTesting.append([row[1], row[2], row[3], row[4], row[5], row[0]])

#Printing the header as the solution
#--> add your Python code here
for row in dbTesting: 
    head = row[0], row[1], row[2], row[3], row[5]

    match row[0]: 
        case "Sunny": 
            row[0] = 1
        case "Overcast": 
            row[0] = 2
        case "Rain": 
            row[0] = 3
    match row[1]: 
        case "Hot": 
            row[1] = 1
        case "Mild": 
            row[1] = 2
        case "Cool": 
            row[1] = 3
    match row[2]:
        case "High": 
            row[2] = 1
        case "Normal": 
            row[2] = 2
    match row[3]:
        case "Strong": 
            row[3] = 1
        case "Weak": 
            row[3] = 2
    match row[4]: 
        case "Yes": 
            row[4] = 1
        case "No": 
            row[4] = 0 
    
    # Make predictions
    classifier_prediction = clf.predict_proba([[row[0], row[1], row[2], row[3]]])
    if classifier_prediction[0][0] >= 0.75: 
        result = round(classifier_prediction[0][0], 3)
    elif classifier_prediction[0][1] >= 0.75: 
        result = round(classifier_prediction[0][1], 3)
    else: 
        continue
    print(head[4] + ", "+ head[0] + ", " +  head[1] + ", " + head[2] + ", " + head[3], end =": ")
    print(result)
