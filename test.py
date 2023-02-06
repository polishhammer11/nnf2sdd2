
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





if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    #filename = 'examples/example.neuron'
    #congressdata = 'testing/congressionalvoting/house-votes-84.data'
    #voting_neuron(congressdata)
    #digitsr = 'testing/congressionalvoting/neuron/congressionalvoting.neuron'
    


    digits = 'testing/digitclassification/csv/train-0-1.txt'
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


    #c.a_star_graph(c.a_star_search(passing), c.a_star_search_f(failing))
    #c.breadth_first_search()


    #c.make_image(passing,failing,digits)




    # NEW FASTER SEARCH
    print("SEARCH ONE:")
    passing,failing,input_map = c.a_star_search_alt()
    print("SEARCH TWO:")
    passingf,failingf,input_mapf = c.a_star_search_alt_f()
    #import pdb
    #pdb.set_trace()
    
    
                
    some_failing = [ c.set_inputs(input_map,setting) for setting in failingf[:1] ]
    some_passing = [ c.set_inputs(input_map,setting) for setting in passing[:1] ]
        
    #Digits Classification Image
    c.make_image(some_passing,some_failing,digits)
    c.a_star_graph_alt(passing,failingf,linestyle="-")


    #Voting Records Analysis
    #c.voting_analysis(some_passing,some_failing)
    #c.num_of_votes(12,congressdata)
    #c.vote_desc(some_passing,some_failing,congressdata) 



    passingd=[]
    failingd =[]
    passingp=[]
    failingp=[]
    #c.a_star_graph(c.breadth_first_search(),c.breadth_first_search_f()) #passing and failing models

    """
    print("SEARCH THREE:")
    c.print_all_true_models(passingd) #using depth-first search
    print("SEARCH FOUR:")
    c.print_all_false_models(failingd)#using depth-first search
    c.print_bounds_graph(passingd,failingd) 
    """


    print("SEARCH THREE:")
    dfs_passing,_ = c.dfs_greedy(find_true=True)
    print("SEARCH FOUR:")
    _,dfs_failing = c.dfs_greedy(find_true=False)
    c.a_star_graph_alt(dfs_passing,dfs_failing,linestyle="-.")


    #passing.sort(key=lambda x: x.size, reverse = True) #best case sorted passing tests
    #failing.sort(key=lambda x: x.size, reverse = True) #best case sorted failing tests
    #c.print_bounds_graph(passing,failing)
    #passing.sort(key=lambda x: x.size)   #worst case sorted passing tests
    #failing.sort(key=lambda x: x.size)   #worst case sorted failing tests



    
    print("SEARCH FIVE:")
    """
    c.pick_first(passingp,failingp)
    c.pick_first_graph(passingp,failingp)
    """
    naive_passing,naive_failing = c.dfs_naive()
    c.a_star_graph_alt(naive_passing,naive_failing,linestyle=":")

    #c.print_bounds_graph(passing,failing)

    

    plt.show()
