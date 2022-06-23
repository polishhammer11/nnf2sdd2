#!/usr/bin/env python3

from circuits.linear import *

def print_model(model):
    model_vars = sorted(model.keys())
    print(" ".join("%d:%d" % (var,model[var]) for var in model_vars))

if __name__ == '__main__':
    filename = 'examples/example.neuron'
    c = Classifier.read(filename)
    print("=== INPUT NEURON:")
    print(c)
    assert c.is_integer
    #c = c.with_precision(3)
    #print("== quantized neuron:")
    #print(c)

    """
    print("===setting index 5 to 1")
    a = c.set_input(5,1)
    print(a)
    print("===setting index 1 to None")
    a = a.set_input(1,None)
    print(a)
    """
    
    print("")
    obdd_manager,node = c.compile()
    #for model in node.models():
    #    print_model(model)
    print()
    print("Fastest")
    c.fasttriviallytrue()
    print()
    
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
        

        
        
    
    
    
    
    
    
    
    
    
   

    
