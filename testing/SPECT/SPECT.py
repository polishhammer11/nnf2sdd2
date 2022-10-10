import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

data_dir = "SPECT.train"
class_names=["OVERALL_DIAGNOSIS", "F1:", "F2:", "F3:", "F4:", "F5:", "F6:", "F7:", "F8:",
             "F9:", "F10:", "F11:", "F12:", "F13:", "F14:", "F15:", "F16:", "F17:", "F18:", 
             "F19:", "F20:", "F21:", "F22:"]

dataset=pd.read_csv(data_dir, names = class_names)
train_features = dataset
train_labels = train_features.pop("OVERALL_DIAGNOSIS")

model = LogisticRegression()
classifier = model.fit(train_features,train_labels)
train_accuracy = 100*classifier.score(train_features,train_labels)
print("LogisticRegression: %.8f%% (training accuracy)" % (train_accuracy,))

A = np.array(train_features)
y = np.array(train_labels)
w = np.array(model.coef_).T 
b = np.array(model.intercept_) 
acc = sum((A@w >= -b).flatten() == y)/len(y)
print("       my accuracy: %.8f%%" % (100*acc,))

#creating file
arr = model.coef_[0]
file1 = open("neuron/SPECT.neuron","w")
file1.write("name: example")
file1.write("\nsize: ")
file1.write((str)(len(model.coef_[0])))
file1.write("\nweights: ")
for x in range(len(arr)):
    file1.write(" ")
    file1.write(str(model.coef_[0][x]))
                    
file1.write("\nthreshold: ")
file1.write(str(-model.intercept_[0]))

