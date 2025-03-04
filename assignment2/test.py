from re import L
from sklearn import tree 
import csv 

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets: 
  dbTraining = []
  X = []
  Y = []

  # Reading the training data in a csv file 
  with open(ds, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
      if i > 0: # skip header 
        dbTraining.append(row)


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

  for i in range(10): 
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
    clf = clf.fit(X,Y)
    result = []
    dbTest = []

    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)
  
    for data in dbTest: 
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
    
    class_predicted = clf.predict([[data[0], data[1], data[2], data[3]]])[0]
    correct = 0
    result.append([class_predicted, data[4]])
    if class_predicted == data[4]: 
      correct += 1
    print(class_predicted, data[4])

  acccuracy = correct/len(dbTest)
  print("accuracy", ds, ":",acccuracy)
