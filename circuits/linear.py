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
            self.weights = { index:float(weight) for index,weight \
                             in enumerate(weights) }
            #self.remove_zero_weights() # ACACAC
            self.setting = {}

    def __repr__(self):
        st = []
        for index in self.weights:
            weight = self.weights[index]
            st.append("  input %d: weight %.4f" % (index,weight))
        for index in self.setting:
            value,weight = self.setting[index]
            value = "None" if value is None else str(value)
            st.append("  input %d: weight %.4f (set to %s)" % \
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

    def remove_zero_weights(self):
        zero_indices = [ index for index,weight in self.weights.items() if weight == 0 ]
        for index in zero_indices:
            del self.weights[index]

    def get_biggest_weight(self):
        """return weight with largest absolute value"""
        # ACAC: use priority queue
        if len(self.weights) == 0: return None
        biggest_index,biggest_abs_weight = None,0
        for index,weight in self.weights.items():
            abs_weight = abs(weight)
            if abs_weight >= biggest_abs_weight:
                biggest_index = index
                biggest_abs_weight = abs_weight
        biggest_weight = self.weights[biggest_index]
        return (biggest_index,biggest_weight)

    def set_biggest_weight(self,value):
        # ACAC: TODO
        pass

    def settings_needed(self,target):
        """returns the number of inputs that need to be set to achieve
        a target decrease in the gap of the threshold test"""
        sorted_weights = [ abs(self.weights[index]) for index in self.weights ]
        sorted_weights.sort(reverse=True)
        weight_sum = 0.0
        weight_count = 0
        for weight in sorted_weights:
            if weight_sum >= target:
                return weight_count
            weight_sum += weight
            weight_count += 1
        return weight_count

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
        intweights = [float(x) for x in self.weights]

        intweights.sort() 

        intweights.pop(0)

        intweight = [str(x) for x in intweights]

        newsize = float(self.size)
        newsize -= 1
        self.size = str(newsize)
        str(self.size)
        self.weights = intweight


    def raiselowerbound(self):
        intweights = [float(x) for x in self.weights]
        intweights.sort()
        intweights.pop()
        intweight = [str(x) for x in intweights]
        newsize = float(self.size)
        newsize -= 1
        self.size = str(newsize)
        str(self.size)
        self.weights = intweight


    def lowerthreshold(self):
        intweights = [float(x) for x in self.weights]
        intweights.sort()
        thresh = intweights.pop()
        intweight = [str(x) for x in intweights]
        newthresh = float(self.threshold)
        newthresh -= thresh
        self.threshold = str(newthresh)
        self.weights = intweight
        newsize = float(self.size)
        newsize -= 1
        self.size = str(newsize)
        str(self.weights)


    def raisethreshold(self):
        intweights = [float(x) for x in self.weights]
        intweights.sort()
        thresh = intweights.pop(0)
        intweight = [str(x) for x in intweights]
        newthresh = float(self.threshold)
        newthresh -= thresh
        self.threshold = str(newthresh)
        self.weights = intweight
        newsize = float(self.size)
        newsize -= 1
        self.size = str(newsize)
        str(self.weights)
    

    def minimizebothbounds(self):
        intweights = [float(x) for x in self.weights]
        intweights.sort()
        intweights.pop()
        intweights.pop(0)
        intweight = [str(x) for x in intweights]
        newsize = float(self.size)
        newsize -= 2
        self.size = str(newsize)
        str(self.size)
        self.weights = intweight


        
        
    def fastmove(self):
        intweights = [float(x) for x in self.weights]
        intweights2 = [float(x) for x in self.weights]
        absintweights = [abs(x) for x in intweights]
        
        twointweights = [float(x) for x in self.weights], [0 for x in self.weights]
        thresh = float(self.threshold)

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
            line = line.strip()
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
            weight = float(weight)
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
        #assert self.is_integer
        self.size = int(size)
        self.weights = [float(x) for x in weights]
        self.threshold = float(threshold)

    def __repr__(self):
        st = []
        st.append("name: %s" % self.name)
        st.append("size: %d" % self.size)
        #st.append("weights: %s" % " ".join(str(weight) for weight in self.weights))
        st.append("threshold: %.4f" % self.threshold)
        st.append("bounds: [%.4f,%.4f]" % \
                  (self.lowerbound(),self.upperbound()))
        #st.append("inputs:\n%s" % str(self.inputs))
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

    def gap(self):
        return self.threshold - self.lowerbound()

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
            #print()
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

    def breadth_first_search(self):
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

    def breadth_first_search_f(self):
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

    def _search_weights(self):
        from itertools import accumulate

        weights = self.inputs.weights.values()
        weights = [ w for w in weights if w != 0 ]

        keyf = lambda x: abs(x)
        sorted_weights = sorted(weights,key=keyf,reverse=True)
        abs_weights = [ abs(w) for w in sorted_weights ]
        accum_weights = [0.0] + list(accumulate(abs_weights))

        return sorted_weights, accum_weights

    @staticmethod
    def _find(sorted_list,target,lo=0):
        hi = len(sorted_list)
        
        for i,val in enumerate(sorted_list[lo:],start=1):
            if val >= target:
                return i

        # AC: this line should not normally be reached
        return 999999

    @staticmethod
    def _add_to_opened(d,accum_weights,opened):
        IntClassifier.id_count += 1
        depth,t,lb,ub = d
        gap = t-lb
        if gap > 0:
            target = accum_weights[depth] + gap
            h_cost = IntClassifier._find(accum_weights,target,lo=depth+1)
        else: # already at goal
            h_cost = 0
        f_cost = depth + h_cost
        node = (f_cost,-IntClassifier.id_count,d)
        opened.put(node)

    def a_star_search_alt(self):
        from queue import PriorityQueue        

        c = self
        is_true =  lambda x: x[1] <= x[2]
        is_false = lambda x: x[1] > x[3]

        IntClassifier.id_count = 0
        closed_list = []
        opened = PriorityQueue()
        sorted_weights,accum_weights = c._search_weights()

        # initial threshold test
        depth = 0
        t = c.threshold
        lb = sum(w for w in sorted_weights if w < 0)
        ub = sum(w for w in sorted_weights if w > 0)

        d = (depth,t,lb,ub)
        IntClassifier._add_to_opened(d,accum_weights,opened)

        true_count, false_count = 0,0
        lower_bound,upper_bound = 0,2**c.size

        while(not opened.empty()):
            f_cost,_,current = opened.get()
            depth,t,lb,ub = current
            var_count = c.size - depth

            if IntClassifier.id_count % 10000 == 0:
                osize = opened.qsize()
                csize = len(closed_list)
                print("open/closed (cost): %d,%d (%d)" % (osize,csize,f_cost))
                print("true/false: %d,%d" % (true_count,false_count))

            if is_false(current):
                false_count += 1
                upper_bound -= 2**var_count
            elif is_true(current):
                true_count += 1
                closed_list.append(current)
                lower_bound += 2**var_count
            else:
                weight = sorted_weights[depth]

                # update lower/upper bounds
                if weight > 0: new_lb,new_ub = lb,ub-weight
                else:          new_lb,new_ub = lb-weight,ub

                # set value to one
                new_t = t-weight
                child = (depth+1,new_t,new_lb,new_ub)
                if is_false(child):
                    false_count += 1
                    upper_bound -= 2**(var_count-1)
                else:
                    IntClassifier._add_to_opened(child,accum_weights,opened)

                # set value to zero
                new_t = t
                child = (depth+1,new_t,new_lb,new_ub)
                if is_false(child):
                    false_count += 1
                    upper_bound -= 2**(var_count-1)
                else:
                    IntClassifier._add_to_opened(child,accum_weights,opened)

        print("lower bound: ", lower_bound)
        print("upper bound: ", upper_bound)

        closed = PriorityQueue()
        for item in closed_list:
            closed.put(item)
        return closed
        

    def a_star_search(self):
        #import pdb
        #pdb.set_trace()
        from queue import PriorityQueue
        
        c = self

        closed_list = []
        closed = PriorityQueue() # ACAC: make this a list
        opened = PriorityQueue()
        count = 0
        path_count = 0
        true_count, false_count = 0,0
        lower_bound,upper_bound = 0,2**c.size

        g_cost = len(c.inputs.setting)
        h_cost = c.inputs.settings_needed(c.gap())
        opened.put((g_cost+h_cost,-count,c))

        goal = 0

        #import pdb; pdb.set_trace()

        while(not opened.empty()):
            current = opened.get()
            c = current[2]
            #print(c)
            if count % 100 == 0:
                print("open/closed (cost): %d,%d (%d)" % (opened.qsize(),len(closed_list),current[0]))
                print("true/false: %d,%d" % (true_count,false_count))

            if c.is_trivially_false():
                false_count += 1
                upper_bound -= 2**child0.size
                continue
            if c.is_trivially_true():
                true_count += 1
                path_count += 1
                closed_list.append(current)
                lower_bound += 2**c.size
            else:
                path_count += 1
                index,weight = c.inputs.get_biggest_weight()
                child0 = c.set_input(index,0)
                child1 = c.set_input(index,1)

                #if not child0.is_trivially_false():
                count += 1
                g_cost = len(child0.inputs.setting)
                h_cost = child0.inputs.settings_needed(child0.gap())
                opened.put((g_cost+h_cost,-count,child0))

                #if not child1.is_trivially_false():
                count += 1
                g_cost = len(child1.inputs.setting)
                h_cost = child1.inputs.settings_needed(child1.gap())
                opened.put((g_cost+h_cost,-count,child1))

        #goals = []
        #while (not closed.empty()): goals.append(closed.get())
        print("nodes found: ", count+1)
        print("path nodes found: ", path_count)
        print("lower bound: ", lower_bound)

        for item in closed_list:
            closed.put(item)
        return closed
                

    def a_star_search_f(self):
        #import pdb
        #pdb.set_trace()
        from queue import PriorityQueue
        
        c = self

        closed = PriorityQueue()
        opened = PriorityQueue()
        count = 0
        path_count = 0
        opened.put((c.size,count,c))

        goal = 0

        while(not opened.empty()):
            current = opened.get()
            c = current[2]

            if(current[2].is_trivially_false()):
                path_count += 1
                closed.put(current)
                continue
            if(current[2].is_trivially_true()):
                continue
            else:
                path_count += 1
                index,weight = current[2].inputs.get_biggest_weight()

                child0 = current[2].set_input(index,0)
                child1 = current[2].set_input(index,1)
                count += 1
                #opened.put((child0.gap(),count,child0))
                opened.put((len(child0.inputs.setting),count,child0))
                count += 1
                #opened.put((child1.gap(),count,child1))
                opened.put((len(child1.inputs.setting),count,child1))

        #goals = []
        #while (not closed.empty()): goals.append(closed.get())
        #print("nodes found: ", count+1)
        #print("path nodes found: ", path_count)
        #print("goals found: ", len(goals))

        return closed
                
                
    def a_star_graph(self,pq,fq):
        import matplotlib.pyplot as plt
        #import pdb
        #pdb.set_trace()
        x = [0]
        y = [0]
        x2 = [0]
        y2 = [2**self.size]
        count = 0
        size = fq.qsize()

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
            
        plt.plot(x,y,marker='*',markersize=1)
        plt.plot(x2,y2,marker='*',markersize=1)
        plt.axhline(y = y2[size], color = 'red', linestyle = '--')
        #plt.show()

            
        
        
     
            
            

        
            

        




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
