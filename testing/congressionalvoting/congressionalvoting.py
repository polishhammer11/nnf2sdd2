import numpy as np
import pandas as pd
import numpy

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


data_dir = "house-votes-84.data"
column_names = ['Class Name','handicapped-infants','water-project-cost-sharing',
                'adoption-of-the-budget-resolution','physician-fee-freeze',
                'el-salvador-aid','religious-groups-in-schools','anti-satellite-test-ban',
                'aid-to-nicaraguan-contras','mx-missile','immigration','synfuels-corporation-cutback',
                'education-spending','superfund-right-to-sue','crime','duty-free-exports',
                'export-administration-act-south-africa']


dataset = pd.read_csv(data_dir, names=column_names)
dataset = dataset.replace({'y': 1, 'n': 0, 'democrat': 1, 'republican': 0} )
dataset = dataset.replace({'?':0}) 
train_features = dataset
train_labels = train_features.pop('Class Name')



model = LogisticRegression()
#model = LogisticRegression(penalty='l1',solver='liblinear',C=.001)
classifier = model.fit(train_features,train_labels)
train_accuracy = 100*classifier.score(train_features,train_labels)
print("LogisticRegression: %.8f%% (training accuracy)" % (train_accuracy,))



model.coef_ #list of weight values
model.intercept_ #bias/threshold

A = np.array(train_features)
y = np.array(train_labels)
w = np.array(model.coef_).T 
b = np.array(model.intercept_) 
print(sum((A@w > -b).flatten() == y)/len(y)*100) #check for training accuracy


#creating file
arr = model.coef_[0]
file1 = open("neuron/congressionalvoting.neuron","w")
file1.write("name: example")
file1.write("\nsize: ")
file1.write((str)(len(model.coef_[0])))
file1.write("\nweights: ")
for x in range(len(arr)):
    file1.write(" ")
    file1.write(str(model.coef_[0][x]))
                    
file1.write("\nthreshold: ")
file1.write(str(-model.intercept_[0]))
