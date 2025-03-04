from sklearn import tree 
import csv 

dataSets = ['/Users/jessicapinto/Documents/GitHub/CS4210/assignment2/contact_lens_training_1.csv', '/Users/jessicapinto/Documents/GitHub/CS4210/assignment2/contact_lens_training_2.csv', '/Users/jessicapinto/Documents/GitHub/CS4210/assignment2/contact_lens_training_3.csv']

for ds in dataSets: 
    dbTraining = []
    X = []
    Y = []

    # Read the training data in csv file 
    with open(ds, 'r') as csvfile: 
        reader = csv.reader(csvfile)
        for i, row in enumerate (reader):
            if i > 0: 
                dbTraining.append(row)
    for row in dbTraining: 
        match row[0]: 
            case "Young": 
                row[0] = 1 
