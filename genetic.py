import numpy as np
import copy

mutationSize = 0.05 #reduce over time? 0.07 worked well
mutationsFrac = 0.3

def fittest(population, agent, *args):
    "finds fittest agents from population. agent is the agent class so that they can be created to compared against"
    best = [0]*len(population)
    for i in range(len(population)):
        if len(args)==0:
            best[i] = agent() #need agent to compare against
        else:
            best[i] = agent(args[0]) #[0] is because args is put inside an extra arg list when passed from populationcontrol
            
    for i in range(len(population)):############needs checking############################
        if population[i].score >= best[i].score:
            best.insert(i, population[i])
            break #break so not put in multiple times
    return best[:len(population)]

def reproduce(parent, NChild, agent, *args):
    "produces NChildren number of children from the two parents and mutates"
    children = [0]*NChild
    for c in range(NChild):
        splitPoint = np.random.randint(1, parent[0].brain.Nhidd-1)
        if len(args)==0:
            child = agent()
        else:
            child = agent(args[0]) # [0] because args is put inside an extra list when passed from population control
        #splice together to create initial child
############################reproduce not just for two best?##########################
######################need to sort so that doesnt copy half of parent (e.g put into one list, or choose randomly whether s1, s2, b is copied)##################################
        for i in range(child.brain.Nhidd):
            for j in range(splitPoint): #copy up to splitting point
                child.brain.synapses1[:,j] = parent[0].brain.synapses1[:,j] #copy weights from first parent
                child.brain.synapses2[:,j] = parent[0].brain.synapses2[:,j] #copy weights from first parent
                child.brain.biases[j] = parent[0].brain.biases[j]
            for k in range(splitPoint, child.brain.Nhidd): #copy from splitting point
                child.brain.synapses1[:,k] = parent[1].brain.synapses1[:,k] #copy weights from second parent
                child.brain.synapses2[k,:] = parent[1].brain.synapses2[k,:] #copy weights from second parent
                child.brain.biases[k] = parent[1].brain.biases[k]
        NMutations = int(mutationsFrac*child.brain.Nhidd)+1 #int rounds down, dont want too low so +1
        children[c] = copy.deepcopy(child)#####################
############################need to sort out so that suitable number of mutations happen###################################
        for j in range(NMutations):
            r1 = np.random.randint(child.brain.Nhidd)
            magnitude = 2*np.random.randint(2) - 1
            children[c].brain.synapses1[0,r1] += mutationSize*magnitude

            r2 = np.random.randint(child.brain.Nhidd)
            magnitude = 2*np.random.randint(2) - 1
            children[c].brain.synapses1[1,r2] += mutationSize*magnitude

            r3 = np.random.randint(child.brain.Nhidd)
            magnitude = 2*np.random.randint(2) - 1
            children[c].brain.synapses2[r3,0] += mutationSize*magnitude

            r4 = np.random.randint(child.brain.Nhidd)
            magnitude = 2*np.random.randint(2) - 1
            children[c].brain.synapses2[r4,1] += mutationSize*magnitude
  
            r5 = np.random.randint(child.brain.Nhidd)
            magnitude = 2*np.random.randint(2) - 1
            children[c].brain.biases[r5] += mutationSize*magnitude
    
    return children

def populationControl(population, NChild, agent, *args):
    "creates new generation from parents and their children"
    best = fittest(population, agent, args)
    for i in best:
        i.resurect()
    children = reproduce(best, NChild, agent, args)
    newPop = best[:-NChild]
    for i in range(NChild):
        newPop.append(children[i])
    return newPop
    
                             

