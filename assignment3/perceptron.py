#-------------------------------------------------------------------------
# AUTHOR: Jessica Pinto
# FILENAME: perceptron.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd
import warnings 


n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('/Users/jessicapinto/Documents/GitHub/CS4210/assignment3/optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('/Users/jessicapinto/Documents/GitHub/CS4210/assignment3/optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

maxPAcc  = -1
maxMLPAcc = -1
maxAccuracy = -1
print()
#iterates over n
for rate in n: 

    for val in r: #iterates over r

        #iterates over both algorithms
        algos = ["perceptron", "multiperceptron"]

        for algo in algos: #iterates over the algorithms

            #Create a Neural Network classifier
            if algo == "perceptron":
               #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
               clf = Perceptron(eta0 = rate, shuffle = val, max_iter=1000)    
               #print(f"Perceptron with parameters {rate}, {val}:", end=" ")
            else:
               #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
               clf = MLPClassifier(activation = "logistic", learning_rate_init = rate, hidden_layer_sizes=25, shuffle = val, max_iter=1000) 
               #print(f"Multi Layer Perceptron with parameters {rate}, {val}:", end = " ")
               maxAccuracy = maxMLPAcc
            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            corrPred = 0
            #hint: to iterate over two collections simultaneously with zip() Example:
            for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: 
                pred = clf.predict([x_testSample])
                if pred[0] == y_testSample:
                    corrPred += 1
            accuracy = corrPred/len(X_test)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            if algo == "perceptron": 
                if accuracy > maxPAcc:
                    print(f"Highest Perceptron accuracy so far: {round(accuracy, 4)}, Parameters: learning rate = {rate}, shuffle = {val}")
                    maxPAcc = accuracy
                    maxPParams = [rate, val]
            else: 
                if accuracy > maxMLPAcc:
                    print(f"Highest Multilayer Perceptron accuracy so far: {round(accuracy, 4)}, Parameters: learning rate = {rate}, shuffle = {val}")
                    maxMLPAcc = accuracy
                    maxMLPParams = [rate, val]
            


print("-------------------------------------------------------------------------------------------------------------")
print(f"Best Perceptron accuracy found with parameters learning rate = {maxPParams[0]}, shuffle = {maxPParams[1]}: {round(maxPAcc, 4)}")
print(f"Best Multi Layer Perceptron accuracy found with parameters learning rate = {maxMLPParams[0]}, shuffle = {maxMLPParams[1]}: {round(maxMLPAcc, 4)}")











