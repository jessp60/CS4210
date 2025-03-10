#-------------------------------------------------------------------------
# AUTHOR: Jessica Pinto
# FILENAME: decision_tree_2.py
# SPECIFICATION: Reading in training data from three csv files to train a classifier, then using the testing samples from another file to output the error rates of the classifer based on the amount of training data it recieved. 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 hours 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    for row in dbTraining: 
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
        match row[4]:
            case "Yes": 
                row[4] = 1
            case "No": 
                row[4] = 2
        Y.append([row[4]])  

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

    #Loop your training and test tasks 10 times here
    accuracies = []
    
    for i in range (10):
       correct = 0
       #Fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here  
       dbTest = []
       with open("contact_lens_test.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)
       
        for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here

           # Transform features 
             # age 
            match data[0]: 
                case "Young": 
                    data[0] = 1
                case "Presbyopic": 
                    data[0] = 2
                case "Prepresbyopic": 
                    data[0] = 3
            # spectacle prescrption 
            match data[1]: 
                case "Myope": 
                    data[1] = 1
                case "Hypermetrope": 
                    data[1] = 2
            # astigmatism
            match data[2]:
                case "Yes": 
                    data[2] = 1
                case "No": 
                    data[2] = 2
            # tear production rate 
            match data[3]:
                case "Reduced": 
                    data[3] = 1
                case "Normal": 
                    data[3] = 2
            match data[4]:
                case "Yes": 
                    data[4] = 1
                case "No": 
                    data[4] = 2

        # for data in dbTest: 
            # Make predictions 
            class_predicted = clf.predict([[data[0], data[1], data[2], data[3]]])[0]
            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if class_predicted == data[4]:
                correct += 1
        accuracy = correct/len(dbTest)
        accuracies.append(accuracy)

    #Find the average of this model during the 10 runs (training and test set)
    #Print the average accuracy of this model during the 10 runs (training and test set).
    print("Final accuracy training on", ds, round(sum(accuracies)/len(accuracies), 2))

