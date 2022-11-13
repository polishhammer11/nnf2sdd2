import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

data_dir = "train-0-1.txt"
dataset=pd.read_csv(data_dir)
train_features = dataset

train_labels = train_features.pop("1.48")


model = LogisticRegression(penalty='l1',solver='liblinear',C=.002,random_state=0) #tol=1e-8
classifier = model.fit(train_features,train_labels)
train_accuracy = 100*classifier.score(train_features,train_labels)
print("LogisticRegression: %.8f%% (training accuracy)" % (train_accuracy,))


A = np.array(train_features)
y = np.array(train_labels)
w = np.array(model.coef_).T 
b = np.array(model.intercept_) 
acc = sum((A@w > -b).flatten() == y)/len(y)
print("       my accuracy: %.8f%%" % (100*acc,))

print_integer = False
# ACAC: if using integers, need to re-check accuracy

digits = 4
alpha = 10**digits

def write_parameter(file1,parameter):
    if print_integer:
        parameter = alpha*parameter
        file1.write("%d" % parameter)
    else:
        file1.write("%f" % parameter)

#creating file
file1 = open("neuron/0-1digits.neuron","w")
file1.write("name: example")
file1.write("\nsize: %d" % len(model.coef_[0]))
file1.write("\nweights: ")
for parameter in model.coef_[0]:
    file1.write(" ")
    write_parameter(file1,parameter)
file1.write("\nthreshold: ")
write_parameter(file1,model.intercept_[0])
file1.write("\n ")

non_zero_count = sum(1 for w in model.coef_[0] if w != 0)
print("non-zero: %d" % non_zero_count)
