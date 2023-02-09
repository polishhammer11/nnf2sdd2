
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
        
    model = LogisticRegression(penalty='l1',solver='liblinear',C=.001,random_state=1) # tol=1e-8b,
    #model = LogisticRegression(penalty='l1',solver='liblinear')
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
        

def voting_neuron(filedir):
    import numpy as np
    import pandas as pd
    import numpy

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    

    data_dir = filedir
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
    #model = LogisticRegression(penalty='l1',solver='liblinear',C=.002,random_state=1)
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
    file1 = open("testing/congressionalvoting/neuron/congressionalvoting.neuron","w")
    file1.write("name: example")
    file1.write("\nsize: ")
    file1.write((str)(len(model.coef_[0])))
    file1.write("\nweights: ")
    for x in range(len(arr)):
        file1.write(" ")
        file1.write(str(model.coef_[0][x]))
                    
    file1.write("\nthreshold: ")
    file1.write(str(-model.intercept_[0]))




def neuron_search_graph(filedir,datatype):
    import matplotlib
    import matplotlib.pyplot as plt

    if(datatype=="v"):
        congressdata = 'testing/congressionalvoting/house-votes-84.data'
        voting_neuron(congressdata)
        neuron = 'testing/congressionalvoting/neuron/congressionalvoting.neuron'
    

    if(datatype=="d" or datatype=="alld"):
        create_neuron(filedir)
        neuron = 'testing/digitclassification/neuron/digitclassification.neuron'
    
    if(datatype=="s"):
        spect = 'testing/SPECT/neuron/SPECT.neuron'
        neuron = spect
    if(datatype=="t"):
        tictactoe = 'testing/tictactoe/neuron/tictactoe.neuron'
        neuron = tictactoe
    if(datatype=="b"):
        banknote = 'testing/banknotedata/neuron/banknote.neuron'
        neuron = banknote
    
    c = IntClassifier.read(neuron)
    print("=== INPUT NEURON:")
    print(c)
    
    #assert c.is_integer
    passing = []
    failing = []
    #c = c.with_precision(3)
    #print("== quantized neuron:")
    #print(c)

    
    
    print()
    #A Star Search
    print("SEARCH ONE:")
    passing,failing,input_map = c.a_star_search_alt()
    print("SEARCH TWO:")
    passingf,failingf,input_mapf = c.a_star_search_alt_f()
    
                   
    some_failing = [ c.set_inputs(input_map,setting) for setting in failingf[:1] ]
    some_passing = [ c.set_inputs(input_map,setting) for setting in passing[:1] ]

        
    #Digits Image Classification
    if(datatype=="d"):
        #c.make_image(some_passing,some_failing,digits)
        print()


    #Voting Records Analysis
    if(datatype=="v"):
        c.voting_analysis(some_passing,some_failing)
        #c.num_of_votes(12,congressdata)
        #c.vote_desc(some_passing,some_failing,congressdata) 


    #A* Graph
    c.a_star_graph_alt(passing,failingf,linestyle="-")


    #Depth First Search
    print("SEARCH THREE:")
    dfs_passing,_ = c.dfs_greedy(find_true=True)
    print("SEARCH FOUR:")
    _,dfs_failing = c.dfs_greedy(find_true=False)
    c.a_star_graph_alt(dfs_passing,dfs_failing,linestyle="-.")


    #Naive Search
    print("SEARCH FIVE:")
    naive_passing,naive_failing = c.dfs_naive()
    c.a_star_graph_alt(naive_passing,naive_failing,linestyle=":")

    
    #Print Graph
    if(datatype!="alld"):
        plt.show()




if __name__ == '__main__':


    datatype = "alld"


    #For all Digit Pairs
    if(datatype=="alld"):
        for i in range(0,9):
            for j in range(1,10):
                if(i!=j and j>i):
                    print(i,"-",j)
                    digits = 'testing/digitclassification/csv/train-%d-%d.txt' % (i,j)
                    neuron_search_graph(digits,datatype)


    #For One Digit Pair
    elif(datatype=="d"):
        digits = 'testing/digitclassification/csv/train-3-8.txt'
        neuron_search_graph(digits,datatype)

    #For Congressional Voting Data
    elif(datatype=="v"):
        congressdata = 'testing/congressionalvoting/house-votes-84.data'
        neuron_search_graph(congressdata,datatype)

    #For Other Data
    else:
        neuron_search_graph("sdf",datatype)
            
