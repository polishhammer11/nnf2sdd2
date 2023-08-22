#!/usr/bin/env python3

from circuits.linear import *
import itertools

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
        
def voting_neural_network(filedir):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import numpy as np
    import pandas as pd
    import numpy
    import tensorflow as tf

    # tf.random.set_seed(30) # MSE
    tf.random.set_seed(2)

    data_dir = filedir
    column_names = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing',
                    'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                    'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
                    'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                    'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports',
                    'export-administration-act-south-africa']

    dataset = pd.read_csv(data_dir, names=column_names)
    dataset = dataset.replace({'y': 1, 'n': 0, 'democrat': 1, 'republican': 0})
    dataset = dataset.replace({'?': 0})
    train_features = dataset
    train_labels = train_features.pop('Class Name')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  # loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    model.fit(dataset, train_labels, epochs=1000)
    # accuracy = model.evaluate(dataset, train_labels)
    # print("accuracy: %.4f%%" % (100 * accuracy[-1],))

    weights = model.get_weights()
    w1, w2 = weights[2]
    b = weights[3]

    # print(w1,w2,b)

    if w1 < -b and w2 < -b and (w1 + w2) > -b:
        print("AND gate")
        print(weights[0])
        print(weights[1])
        print(weights[2])
        print(weights[3])
    elif w1 > -b or w2 > -b and 0 < -b:
        print("OR gate")
        """
        print("=== neuron 1 + neuron 2 weights")
        print(weights[0])
        print("=== neuron 1 + neuron 2 bias")
        print(weights[1])
        print("=== neuron 3 weights")
        print(weights[2])
        print("=== neuron 3 bias")
        print(weights[3])
        """
        n1_w = weights[0][:, 0]
        n2_w = weights[0][:, 1]
        n1_b = weights[1][0]
        n2_b = weights[1][1]

        n1_filename = "testing/congressionalvoting/neuron/voting-n1.neuron"
        print("saving neuron n1 as %s" % n1_filename)
        with open(n1_filename, "w") as f:
            weight_st = " ".join("%.8f" % w for w in n1_w)
            f.write("name: n1\n")
            f.write("size: %d\n" % len(n1_w))
            f.write("weights: %s\n" % weight_st)
            f.write("threshold: %.8f\n" % n1_b)

        n2_filename = "testing/congressionalvoting/neuron/voting-n2.neuron"
        print("saving neuron n2 as %s" % n2_filename)
        with open(n2_filename, "w") as f:
            weight_st = " ".join("%.8f" % w for w in n2_w)
            f.write("name: n2\n")
            f.write("size: %d\n" % len(n2_w))
            f.write("weights: %s\n" % weight_st)
            f.write("threshold: %.8f\n" % n2_b)



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


def neuron_search_graph(filedir, datatype,i=None,j=None):
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
    if(datatype=="e"):
        congressdata = 'testing/congressionalvoting/house-votes-84.data'
        voting_neural_network(congressdata)

        neuron = 'testing/congressionalvoting/neuron/voting-n1.neuron'
        neuron2 = 'testing/congressionalvoting/neuron/voting-n2.neuron'
        d = IntClassifier.read(neuron2)
    
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

    if (datatype == "e"):
        passing2, failing2, input_map2 = d.a_star_search_alt()
        passingf2, failingf2, input_mapf2 = d.a_star_search_alt_f()

        
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
        import pdb;
        pdb.set_trace()
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
        #plt.show()
        pass
    if(datatype == "alld"):
        if i is not None and j is not None:
            plt.title('digits (%d,%d)' % (i,j))
            plt.savefig("digits/digits-%d-%d.png" % (i,j))
            plt.savefig("digits/digits-%d-%d.pdf" % (i,j))

    if(datatype=="e"):


        passe = [ c.set_inputs(input_map,s) for s in passing[:100] ]
        faile = [c.set_inputs(input_mapf, s) for s in failingf[:100]]

        passe2 = [ c.set_inputs(input_map2,s) for s in passing2[:100] ]
        faile2 = [c.set_inputs(input_mapf2, s) for s in failingf2[:100]]

        plt.clf()

        ttc = truth_table(c)
        ttd = truth_table(d)
        candd = tt_and(ttc,ttd)


        c.bounds_graphs(passe, faile, passe2, faile2,c, d, candd)


        #c.latex_truth_table(passe,faile)




    #plt.clf()

def truth_table(c):

    #import pdb; pdb.set_trace()
    num_inputs = len(c.weights)
    inputs = list(itertools.product([0, 1], repeat=num_inputs))

    def function(*args):
        weighted_sum = sum(w * i for w, i in zip(c.weights, args))
        return weighted_sum >= c.threshold

    header = '\t'.join( str(c.weights[i]) for i in range(num_inputs)) + " Output"
    #print(header)

    table = []

    for input_values in inputs:
        output = function(*input_values)
        if output == True:
            input_val = input_values + (1,)
            table.append(list(input_val))
        else:
            input_val = input_values + (0,)
            table.append(list(input_val))

    return table


def tt_and(c,d):
    ttand = []
    for b1, b2 in zip(c, d): ttand.append(b1[-1] and b2[-1])
    return ttand

def tt_notand(c,d):
    for b1, b2 in zip(c, d): print(b1, b2, not b1['Output'] and b2['Output'])

def tt_notandnot(c,d):
    for b1, b2 in zip(c, d): print(b1, b2, not b1['Output'] and not b2['Output'])

def tt_or(c,d):
    for b1, b2 in zip(c, d): print(b1, b2, b1['Output'] or b2['Output'])

def tt_notor(c, d):
    for b1, b2 in zip(c, d): print(b1, b2, not b1['Output'] and b2['Output'])

def tt_notornot(c,d):
    for b1, b2 in zip(c, d): print(b1, b2, not b1['Output'] or not b2['Output'])



if __name__ == '__main__':


    datatype = "e"


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

    elif(datatype=="e"):
        #import pdb
        #pdb.set_trace()
        example = 'examples/example.neuron'
        example2 = 'examples/example2.neuron'
        #c = IntClassifier.read(example)
        #d = IntClassifier.read(example2)

        #ctt = truth_table(c)
        #dtt = truth_table(d)

        #print(ctt)
        #print(dtt)
        #candd = tt_and(ctt,dtt)
        #print(candd)

        neuron_search_graph(example, datatype)





    else:
        neuron_search_graph("sdf",datatype)
            
