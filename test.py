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
        
    model = LogisticRegression(penalty='l1',solver='liblinear',C=.002,random_state=1) # tol=1e-8b,
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

def test_neuron(c,train_filename):
    import numpy as np
    import pandas as pd

    dataset = pd.read_csv(train_filename)
    train_features = dataset
    dataset.columns = [*dataset.columns[:-1], 'Label']
    train_features = dataset
    train_labels = train_features.pop("Label")

    A = np.array(train_features)
    y = np.array(train_labels)
    w = np.array(c.weights)
    b = np.array(-c.threshold) 
    acc = sum((A@w > -b).flatten() == y)/len(y)
    print("       my accuracy: %.8f%%" % (100*acc,))

    prs = A@w + b
    labels = np.array(train_labels)
    indices = []
    indices.append(np.argmax(prs*labels))
    indices.append(np.argmin(prs*labels))
    labels = 1-labels # flip
    indices.append(np.argmin(prs*labels))
    indices.append(np.argmax(prs*labels))
    print("indices: ", indices)
    labels = [1,0,0,1]
    return indices,labels

def neuron_search_graph(filedir,datatype,i=None,j=None):
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
    
    if(datatype=="d" or datatype=="alld"):
        print("=== TESTING NEURON:")
        indices,labels = test_neuron(c,filedir)
    
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
    print("#passing/failing: %d,%d" % (len(passing),len(failingf)))

        
    #Digits Image Classification
    if(datatype=="d" or datatype == "alld"):
        import numpy as np
        import pandas as pd

        passc = [ c.set_inputs(input_map,s) for s in passing[:10] ]
        failc = [ c.set_inputs(input_mapf,s) for s in failingf[:10] ]


        #with open(filedir, 'r') as f:
        #    images = f.readlines()
        dataset = pd.read_csv(filedir)
        images = np.array(dataset)

        for index,label in zip(indices,labels):
            image = images[index]
            image,image_label = image[:-1],image[-1]
            image_filename = "digits/digit-%d-%d-i%d-l%d" % (i,j,index,label)
            c.make_image(image,label,passc,failc,image_filename)

    #Voting Records Analysis
    if(datatype=="v"):
        some_failing = [ c.set_inputs(input_map,s) for s in failingf[:1] ]
        some_passing = [ c.set_inputs(input_map,s) for s in passing[:1] ]
        c.voting_analysis(some_passing,some_failing)
        #c.num_of_votes(congressdata)
        #c.vote_desc(some_passing,some_failing,congressdata) 


    #A* Graph
    c.a_star_graph_alt(passing,failingf,linestyle="-")


    #Depth First Search
    print("SEARCH THREE:")
    dfs_passing,_ = c.dfs_greedy(find_true=True)
    print("SEARCH FOUR:")
    _,dfs_failing = c.dfs_greedy(find_true=False)
    c.a_star_graph_alt(dfs_passing,dfs_failing,linestyle="-.")
    print("#passing/failing: %d,%d" % (len(dfs_passing),len(dfs_failing)))

    #Naive Search
    print("SEARCH FIVE:")
    naive_passing,naive_failing = c.dfs_naive()
    c.a_star_graph_alt(naive_passing,naive_failing,linestyle=":")
    print("#passing/failing: %d,%d" % (len(naive_passing),len(naive_failing)))
    
    #Print Graph
    if(datatype!="alld"):
        plt.show()
        pass
    if(datatype == "alld"):
        if i is not None and j is not None:
            plt.title('digits (%d,%d)' % (i,j))
            plt.savefig("digits/digits-%d-%d.png" % (i,j))
            plt.savefig("digits/digits-%d-%d.pdf" % (i,j))
    plt.clf()


if __name__ == '__main__':


    datatype = "v"


    #For all Digit Pairs
    if(datatype=="alld"):
        for i in range(0,10):
            for j in range(i+1,10):
                print(i,"-",j)
                digits = 'testing/digitclassification/csv/train-%d-%d.txt' % (i,j)
                neuron_search_graph(digits,datatype,i=i,j=j)


    #For One Digit Pair
    elif(datatype=="d"):
        i,j = 0,1
        digits = 'testing/digitclassification/csv/train-%d-%d.txt' % (i,j)
        neuron_search_graph(digits,datatype,i=i,j=j)

    #For Congressional Voting Data
    elif(datatype=="v"):
        congressdata = 'testing/congressionalvoting/house-votes-84.data'
        neuron_search_graph(congressdata,datatype)

    #For Other Data
    else:
        neuron_search_graph("sdf",datatype)
            
