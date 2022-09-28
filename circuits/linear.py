#!/usr/bin/env python3

import math
from decimal import Decimal
from .obdd import ObddManager, ObddNode
from .timer import Timer

class Inputs:
    """AC: represent this as a priority queue?  This would 
    make finding the max weight faster
    """

    def __init__(self,weights=None):
        if weights is None:
            self.original_weights = []
            self.weights = {}
            self.setting = {}
        else:
            self.original_weights = weights
            self.weights = { index:int(weight) for index,weight \
                             in enumerate(weights) }
            self.setting = {}

    def __repr__(self):
        st = []
        for index in self.weights:
            weight = self.weights[index]
            st.append("  input %d: weight %d" % (index,weight))
        for index in self.setting:
            value,weight = self.setting[index]
            value = "None" if value is None else str(value)
            st.append("  input %d: weight %d (set to %s)" % \
                      (index,weight,value))
        return "\n".join(st)

    def copy(self):
        inputs = Inputs()
        inputs.weights = dict(self.weights)
        inputs.setting = dict(self.setting)
        return inputs

    def set(self,index,value):
        assert index in self.weights
        weight = self.weights[index]
        del self.weights[index]
        self.setting[index] = (value,weight)
        return weight

    def get_biggest_weight(self):
        """return weight with largest absolute value"""
        if len(self.weights) == 0: return None
        biggest_index,biggest_abs_weight = None,0
        for index,weight in self.weights.items():
            abs_weight = abs(weight)
            if abs_weight >= biggest_abs_weight:
                biggest_index = index
                biggest_abs_weight = abs_weight
        biggest_weight = self.weights[biggest_index]
        return (biggest_index,biggest_weight)

    def get_model(self):
        return { index:value for index,(value,w) in self.setting.items() if value is not None }

class Classifier:
    """For representing Andy's linear classifier (neuron) format."""

    def __init__(self,name="none",size="0",weights=[],threshold="0"):
                 #num_values="2",prior="0",offset="0"):
        self.name = name
        self.size = size
        self.weights = weights # AC: this should disappear
        self.inputs = Inputs(weights)
        self.threshold = threshold
        self.is_integer = self.check_integrality()
        # extra stuff from Andy's format
        #self.num_values = num_values
        #self.prior = prior
        #self.offset  = offset

    def __repr__(self):
        st = []
        st.append("name: %s" % self.name)
        st.append("size: %s" % self.size)
        st.append("weights: %s" % " ".join(self.weights))
        st.append("threshold: %s" % self.threshold)
        st.append("inputs:\n%s" % str(self.inputs))
        return "\n".join(st)

    def dict(self):
        return {
            "name": self.name,
            "size": self.size,
            "weights": list(self.weights),
            "threshold": self.threshold
        }


    def lowerupperbound(self):
        intweights = [int(x) for x in self.weights]

        intweights.sort() 

        intweights.pop(0)

        intweight = [str(x) for x in intweights]

        newsize = int(self.size)
        newsize -= 1
        self.size = str(newsize)
        str(self.size)
        self.weights = intweight


    def raiselowerbound(self):
        intweights = [int(x) for x in self.weights]
        intweights.sort()
        intweights.pop()
        intweight = [str(x) for x in intweights]
        newsize = int(self.size)
        newsize -= 1
        self.size = str(newsize)
        str(self.size)
        self.weights = intweight


    def lowerthreshold(self):
        intweights = [int(x) for x in self.weights]
        intweights.sort()
        thresh = intweights.pop()
        intweight = [str(x) for x in intweights]
        newthresh = int(self.threshold)
        newthresh -= thresh
        self.threshold = str(newthresh)
        self.weights = intweight
        newsize = int(self.size)
        newsize -= 1
        self.size = str(newsize)
        str(self.weights)


    def raisethreshold(self):
        intweights = [int(x) for x in self.weights]
        intweights.sort()
        thresh = intweights.pop(0)
        intweight = [str(x) for x in intweights]
        newthresh = int(self.threshold)
        newthresh -= thresh
        self.threshold = str(newthresh)
        self.weights = intweight
        newsize = int(self.size)
        newsize -= 1
        self.size = str(newsize)
        str(self.weights)
    

    def minimizebothbounds(self):
        intweights = [int(x) for x in self.weights]
        intweights.sort()
        intweights.pop()
        intweights.pop(0)
        intweight = [str(x) for x in intweights]
        newsize = int(self.size)
        newsize -= 2
        self.size = str(newsize)
        str(self.size)
        self.weights = intweight


        
        
    def fastmove(self):
        intweights = [int(x) for x in self.weights]
        intweights2 = [int(x) for x in self.weights]
        absintweights = [abs(x) for x in intweights]
        
        twointweights = [int(x) for x in self.weights], [0 for x in self.weights]
        thresh = int(self.threshold)

        lowerbound = sum(i for i in intweights if i<0)


        maxnum = max(absintweights)
        maxindex = absintweights.index(maxnum)
         
   
     
    
        if intweights[maxindex] > 0:
                
            print("Set" , intweights[maxindex] , "to 1") 
            twointweights[1][maxindex] = 1
            thresh -= maxnum
            
        if intweights[maxindex] < 0:
                
            print("Set" , intweights[maxindex] , "to 0")
            twointweights[1][maxindex] = 0

            absintweights.pop(maxindex)
            intweights2.pop(maxindex)
            lowerbound = sum(i for i in intweights2 if i<0)

        for i in twointweights:
            for j in i:
                print(j, end = " ")
            print()
        

        

        

        
        
        
        









#intweights = [int(x) for x in self.weights]
#        inst = {}
#        twointweights = [int(x) for x in self.weights], [0 for x in self.weights]
#        thresh = int(self.threshold)
#        lowerbound = sum(i for i in intweights if i<0) 
#        absweights = [abs(x) for x in intweights]
#        absweights.sort()
#        while lowerbound < thresh:
#            mostneg = min(intweights)
            #print(twointweights[1])
#            if(mostneg < 0):
#                lowerbound -= mostneg
                   

#            inde = intweights.index(mostneg)

#            twointweights[1][inde] = 1

#            inst[inde+1] = 1
            

#            col = len(self.weights)
#            row = 2
#            for j in range(col):
#                for i in range(row):
#                    if(twointweights[col][row] == mostneg):
#                        twointweights[col][row+1] = 1
                    
#        print(twointweights[0])
#        print(twointweights[1])
#        print(inst)



        

        

        
    
    	    

    def format_andy(self):
        st = []
        st.append( "%s\n" % self.name )
        st.append( "%s %s %s %s %s\n" % ( self.size,self.num_values,
                                          self.prior,self.threshold,
                                          self.offset ) )
        st.append( "%s\n" % " ".join(self.weights) )
        return "".join(st)

    @staticmethod
    def read_andy(filename):
        """Read Andy's neuron format (deprecated)"""
        with open(filename,'r') as f:
            lines = f.readlines()
        lines = [ line.strip() for line in lines ]
        name = lines[0]
        size,num_values,prior,threshold,offset = lines[1].split()
        weights = lines[2].split()
        neuron = { "name": name, "size": size,
                   "weights": weights,"threshold": threshold }
        return Classifier(**neuron)

    @staticmethod
    def parse(st):
        """Parse a neuron string format"""
        neuron = {}
        for line in st.split('\n'):
            if not line: continue
            field,value = line.split(':')
            field = field.strip()
            value = value.strip()
            neuron[field] = value
        assert "size" in neuron
        assert "threshold" in neuron # or "bias" in neuron
        assert "weights" in neuron
        neuron["weights"] = neuron["weights"].split()
        return Classifier(**neuron)

    @staticmethod
    def read(filename):
        """Read a neuron from file"""
        with open(filename,'r') as f:
            st = f.read()
        return Classifier.parse(st)

    def save(self,filename=None):
        if filename is None: filename = self.filename
        with open(filename,'w') as f:
            f.write(str(self))

    def _biggest_weight(self):
        biggest = 0
        for weight in self.weights:
            w = abs(float(weight))
            if w > biggest:
                biggest = w
        return biggest

    def check_integrality(self):
        weights = self.weights + [self.threshold]
        check = [ float(w).is_integer() for w in weights ]
        return sum(check) == len(check)

    def with_precision(self,digits):
        biggest = self._biggest_weight()
        scale = Decimal(biggest).adjusted()
        scale = -scale + digits-1
        scale = 10**scale
        new_weights = [ scale*float(weight) for weight in self.weights ]
        new_weights = [ str(int(weight)) for weight in new_weights ]
        new_threshold = str(int(scale*float(self.threshold)))
        neuron = self.dict()
        neuron["weights"] = new_weights
        neuron["threshold"] = new_threshold
        c = Classifier(**neuron)
        assert c.is_integer
        return c

    def _get_bounds(self):
        assert self.is_integer
        lower,upper = 0,0
        for weight in self.weights:
            weight = int(weight)
            if weight < 0:
                lower += weight
            else:
                upper += weight
        return (lower,upper)

    def _to_obdd(self,matrix):
        var_count = int(self.size)
        manager = ObddManager(var_count)
        one,zero = manager.one_sink(),manager.zero_sink()
        last_level = matrix[var_count+1]
        for node in last_level:
            last_level[node] = one if last_level[node] else zero
        for dvar in range(var_count,0,-1):
            level,next_level = matrix[dvar],matrix[dvar+1]
            for node in level:
                hi,lo = level[node] # get indices
                hi,lo = next_level[hi],next_level[lo] # get nodes
                level[node] = manager.new_node(dvar,hi,lo)
        return (manager,matrix[1][0])

    def compile(self):
        assert self.is_integer
        var_count = int(self.size)
        matrix = [ dict() for _ in range(var_count+2) ]
        matrix[1][0] = None # root node
        for i in range(1,var_count+1):
            level,next_level = matrix[i],matrix[i+1]
            weight = int(self.weights[i-1])
            for node in level:
                hi,lo = (node+weight,node)
                level[node] = (hi,lo) # (hi,lo)
                next_level[hi] = None
                next_level[lo] = None
        last_level = matrix[var_count+1]
        threshold = int(self.threshold)
        for node in last_level:
            last_level[node] = node >= threshold
        return self._to_obdd(matrix)


class IntClassifier(Classifier):
    def __init__(self,name="none",size=0,weights=[],threshold=0):
        super().__init__(name=name,size=size,weights=weights,threshold=threshold)
        assert self.is_integer
        self.size = int(size)
        self.weights = [int(x) for x in weights]
        self.threshold = int(threshold)

    def __repr__(self):
        st = []
        st.append("name: %s" % self.name)
        st.append("size: %d" % self.size)
        st.append("weights: %s" % " ".join(str(weight) for weight in self.weights))
        st.append("threshold: %d" % self.threshold)
        st.append("bounds: [%d,%d]" % \
                  (self.lowerbound(),self.upperbound()))
        st.append("inputs:\n%s" % str(self.inputs))
        return "\n".join(st)

    @staticmethod
    def read(filename):
        classifier = Classifier.read(filename)
        return IntClassifier(name=classifier.name,
                             size=classifier.size,
                             weights=classifier.weights,
                             threshold=classifier.threshold)

    def copy(self):
        name = self.name
        size = self.size
        weights = list(self.weights) # AC: remove this eventually
        threshold = self.threshold
        classifier = IntClassifier(name=name,size=size,
                                   weights=weights,
                                   threshold=threshold)
        classifier.inputs = self.inputs.copy()
        return classifier

    def set_input(self,index,value):
        """return a new copy of the classifier where the input index
        has been set to value"""

        new_classifier = self.copy()
        weight = new_classifier.inputs.set(index,value)
        new_classifier.size -= 1
        if value != 0 and value is not None:
            new_classifier.threshold -= value*weight
        return new_classifier


    def lowerbound(self):
        none_weights = [ weight for value,weight in self.inputs.setting.values() \
                         if value is None ]
        weights = none_weights + list(self.inputs.weights.values())
        return sum(w for w in weights if w<0)

    def upperbound(self):
        none_weights = [ weight for value,weight in self.inputs.setting.values() \
                         if value is None ]
        weights = none_weights + list(self.inputs.weights.values())
        return sum(w for w in weights if w>0)

    def is_trivially_true(self):
        return self.threshold <= self.lowerbound()

    def is_trivially_false(self):
        return self.threshold > self.upperbound()

    def fast_trivially_true(self):
        """automize finding the fastest way to make a test trivially
        true and false

        """

        if self.is_trivially_false():
            print("already trivially false")
            return

        c = self
        count = 0
        while not c.is_trivially_true():
            count += 1
            index,weight = c.inputs.get_biggest_weight()

            if weight > 0:
                #print(count, "Set" , weight , "to 1")
                c = c.set_input(index,1)
            else:
                #print(count, "Set" , weight , "to 0")
                c = c.set_input(index,0)

        #print()
        #print("=== trivial classifier:")
        #print(c)
        return c
        

    def print_all_true_models(self,explanationsize):
        
        #import pdb;
        #pdb.set_trace()
        "recursive method that finds all true models in a thresold test"
        c = self
        
        if c.is_trivially_true():
            print()
            #print(c.inputs)
            explanationsize.append(c)
            return
            

        if c.is_trivially_false():
            return 

        index,weight = c.inputs.get_biggest_weight()
        
       
        if(weight<0):
            b = c.set_input(index,0)
            b.print_all_true_models(explanationsize)

            a = c.set_input(index,1)
            a.print_all_true_models(explanationsize)
        else:    
            a = c.set_input(index,1)
            a.print_all_true_models(explanationsize)

            b = c.set_input(index,0)
            b.print_all_true_models(explanationsize)

    def print_all_false_models(self,explanationsize):
        
        #import pdb;
        #pdb.set_trace()
        "recursive method that finds all true models in a thresold test"
        c = self
        
        if c.is_trivially_true():
            return
            

        if c.is_trivially_false():
            explanationsize.append(c)
            return 

        index,weight = c.inputs.get_biggest_weight()

        if(weight<0):
            b = c.set_input(index,1)
            b.print_all_false_models(explanationsize)

            a = c.set_input(index,0)
            a.print_all_false_models(explanationsize)
        else:    
            a = c.set_input(index,0)
            a.print_all_false_models(explanationsize)

            b = c.set_input(index,1)
            b.print_all_false_models(explanationsize)




    def print_bounds_graph(self,passing,failing):
        #import pdb;
        #pdb.set_trace()
        c = self
        import matplotlib.pyplot as plt
        x = [0]
        y = [0]
        for i in range(len(passing)):
            x.append(i+1)
            y.append(y[i] + 2**len(passing[i].inputs.weights))
        f = 0
        f += 2**c.size

        x2 = [0] 
        y2 = [f] 
        for i in range(len(failing)):
            x2.append(i+1)
            y2.append(y2[i] - 2**len(failing[i].inputs.weights))
        
        plt.axhline(y = y2[len(failing)], color = 'red', linestyle = '--')
        plt.plot(x,y,marker='o',markersize=1)
        plt.plot(x2,y2,marker = 'o', markersize=1)
        #plt.show()

    def a_star_search(self):
        #import pdb;
        #pdb.set_trace()
        from queue import PriorityQueue
        c = self
        fq = PriorityQueue()
        q = PriorityQueue()
        count = 0
        q.put((c.size,count,c))
        
        while(not q.empty()):
            current = q.get()
            current2 = current
            if(current[2].is_trivially_false()):
                continue
            if(current[2].is_trivially_true()):
                fq.put(current)
                continue
            else:
                index,weight = current[2].inputs.get_biggest_weight()
                count+=1
                a = current[2].set_input(index,1)
                q.put((len(a.inputs.setting),count,a))

                b = current2[2].set_input(index,0)
                count += 1
                q.put((len(b.inputs.setting),count,b))
                
        return fq

    def a_star_search_f(self):
        #import pdb;
        #pdb.set_trace()
        from queue import PriorityQueue
        c = self
        fq = PriorityQueue()
        q = PriorityQueue()
        count = 0
        q.put((c.size,count,c))
        
        while(not q.empty()):
            current = q.get()
            current2 = current
            if(current[2].is_trivially_false()):
                fq.put(current)
                continue
            if(current[2].is_trivially_true()):
                continue
            else:
                index,weight = current[2].inputs.get_biggest_weight()
                count+=1
                a = current[2].set_input(index,1)
                q.put((len(a.inputs.setting),count,a))

                b = current2[2].set_input(index,0)
                count += 1
                q.put((len(b.inputs.setting),count,b))
                
        return fq

    def a_star_graph(self,pq,fq):
        import matplotlib.pyplot as plt
        #import pdb
        #pdb.set_trace()
        x = [0]
        y = [0]
        x2 = [0]
        y2 = [2**self.size]
        count = 0
        while(not pq.empty()):
            current = pq.get()
            y.append(y[count] + 2**len((current[2].inputs.weights)))
            count += 1
            x.append(count)
        count = 0
        while(not fq.empty()):
            current = fq.get()
            y2.append(y2[count] - 2**len((current[2].inputs.weights)))
            count += 1
            x2.append(count)
            
        plt.plot(x,y,marker='o',markersize=1)
        plt.plot(x2,y2,marker='o',markersize=1)
        #plt.show()

            
        
        
    


            

        
        




'''
        while(len(q)>0):
            current = p.pop()
            if(c == self.fast_trivially_true()):
                return c
    
            

        c.print_all_true_models(passing)
        for i in range(len(passing)):
            q.append((passing[i].size,i, passing[i]))
        q.sort(reverse=True)
        while(not q):
'''         
            
            

        
            

        




if __name__ == '__main__':
    precision = 2
    #filename = 'examples/169_wd2_0'
    #output_filename = 'examples/169_wd2_0-quantized'
    filename = 'examples/9_wd2_0'
    output_filename = 'examples/9_wd2_0-quantized'
    #filename = 'examples/test.nn'
    #output_filename = 'examples/test-quantized.nn'
    c = Classifier.read(filename)
    print(c)
    d = c.with_precision(precision)
    print(d)
    d.save(output_filename)
    with Timer("compiling"):
        obdd_manager,node = d.compile()
    with Timer("size"):
        count_before = len(list(node))
    with Timer("reducing"):
        node = node.reduce()
    with Timer("size"):
        count_after = len(list(node))
    print("node count before reduce: %d" % count_before)
    print(" node count after reduce: %d" % count_after)

    obdd_manager.save_vtree("tmp.vtree")
    obdd_manager.save_sdd("tmp.sdd",node)

    with Timer("to sdd"):
        offset = int(c.offset)
        sdd_manager,sdd_node = obdd_manager.obdd_to_sdd(node,offset=offset)
    with Timer("read sdd"):
        sdd_filename = b'tmp.sdd'
        alpha = sdd_manager.read_sdd_file(sdd_filename)

    print("sdd nodes/size: %d/%d" % (sdd_node.count(),sdd_node.size()))
    print("sdd nodes/size: %d/%d" % (alpha.count(),alpha.size()))
