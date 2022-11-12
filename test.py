
#!/usr/bin/env python3

from circuits.linear import *

def print_model(model):
    model_vars = sorted(model.keys())
    print(" ".join("%d:%d" % (var,model[var]) for var in model_vars))

def create_neuron(filedir):
    import numpy as np
    import pandas as pd

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    
    data_dir = filedir
    dataset=pd.read_csv(data_dir)
    train_features = dataset
    dataset.columns = [*dataset.columns[:-1], 'Label']
    train_features = dataset
    train_labels = train_features.pop("Label")
        
    model = LogisticRegression(penalty='l1',solver='liblinear',C=.002) # tol=1e-8,
    classifier = model.fit(train_features,train_labels)
    train_accuracy = 100*classifier.score(train_features,train_labels)
    print("LogisticRegression: %.8f%% (training accuracy)" % (train_accuracy,))


    A = np.array(train_features)
    y = np.array(train_labels)
    w = np.array(model.coef_).T 
    b = np.array(model.intercept_) 
    acc = sum((A@w > -b).flatten() == y)/len(y)
    print("       my accuracy: %.8f%%" % (100*acc,))


    #creating file
    arr = model.coef_[0]
    file1 = open("testing/digitclassification/neuron/digitclassification.neuron","w")
    file1.write("name: example")
    file1.write("\nsize: ")
    file1.write((str)(len(model.coef_[0])))
    file1.write("\nweights: ")
    for x in range(len(arr)):
        file1.write(" ")
        file1.write("%f" % model.coef_[0][x])
        
    file1.write("\nthreshold: ")
    file1.write("%f" % -model.intercept_[0])    
    non_zero_count = 0
    for w in model.coef_[0]:
        if w != 0:
            non_zero_count += 1
    print("non-zero: %d" % non_zero_count)
        

if __name__ == '__main__':

    #filename = 'examples/example.neuron'
    #congressdata = 'testing/congressionalvoting/neuron/congressionalvoting.neuron'
    

    #digits = 'testing/0-1digits/neuron/0-1digits.neuron'
    #digits = 'testing/0-1digits/train-0-1.txt'
    digits = 'testing/digitclassification/csv/train-5-8.txt'
    create_neuron(digits)
    digitsr = 'testing/digitclassification/neuron/digitclassification.neuron'
    
    #spect = 'testing/SPECT/neuron/SPECT.neuron'
    #tictactoe = 'testing/tictactoe/neuron/tictactoe.neuron'
    #banknote = 'testing/banknotedata/neuron/banknote.neuron'
    c = IntClassifier.read(digitsr)
    print("=== INPUT NEURON:")
    print(c)
    
    #assert c.is_integer
    passing = []
    failing = []
    #c = c.with_precision(3)
    #print("== quantized neuron:")
    #print(c)

    
    
    print("")
    #obdd_manager,node = c.compile()
    #for model in node.models():
    #    print_model(model)
    print()
    #print("Passing Tests")


    c.a_star_graph(c.a_star_search(passing), c.a_star_search_f(failing))
    #c.breadth_first_search()
    c.make_image(passing,failing,digits)

    # NEW FASTER SEARCH
    #passing,failing = c.a_star_search_alt()
    #c.a_star_graph_alt(passing,failing)



    #c.a_star_graph(c.breadth_first_search(),c.breadth_first_search_f()) #passing and failing models
    #c.print_all_true_models(passing) #using depth-first search
    #c.print_all_false_models(failing)#using depth-first search
    #c.print_bounds_graph(passing,failing) 
    #passing.sort(key=lambda x: x.size, reverse = True) #best case sorted passing tests
    #failing.sort(key=lambda x: x.size, reverse = True) #best case sorted failing tests
    #c.print_bounds_graph(passing,failing)
    #passing.sort(key=lambda x: x.size)   #worst case sorted passing tests
    #failing.sort(key=lambda x: x.size)   #worst case sorted failing tests
    #c.print_bounds_graph(passing,failing)

    """
    import matplotlib.pyplot as plt
    plt.show()
    
    while True:
        print("=== Lower Bound:")
        c.lowerbound()
        
        print("=== Upper Bound:")
        c.upperbound()
        print()
        c.fastmove()
        c.checktriviality()
        print()
    
        print("Select an Option")
        print("1. Lower Upper Bound")
        print("2. Raise Lower Bound")
        print("3. Lower threshold")
        print("4. Raise threshold")
        choice = input()



        if choice == "2":
            c.lowerupperbound()
            
        if choice == "1":
            c.raiselowerbound()
        
        if choice == "3":
            c.lowerthreshold()

        if choice == "4":
            c.raisethreshold()
        

    
    
        print("")
    
        print("=== NEW NEURON:")
        print(c)
        assert c.is_integer
        #c = c.with_precision(3)
        #print("== quantized neuron:")
        #print(c)
    
        print("")
        obdd_manager,node = c.compile()
        for model in node.models():
            print_model(model)
        
        print("=== Lower Bound:")
        c.lowerbound()
    
        print("=== Upper Bound:")
        c.upperbound()
        if(c.size=="0"):
            break
    """
