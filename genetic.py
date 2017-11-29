import numpy as np
import copy

mutationSize = 0.05 #reduce over time? 0.07 worked well

def fittest(population, agent):
    "finds fittest five agents from population"
    best = [0]*10
    for i in range(10):
        best[i] = agent() #need agent to compare against (xpos =score=0)
    for i in range(len(population)):
        dist = population[i].xPos
        dx = population[i].brain.wallDist
        dy = population[i].brain.yDisplace
        score = dist - np.sqrt(dx*dx + dy*dy) #+ population[i].wallPass
        population[i].score = score
        for j in range(10):
            if score >= best[j].score:
#                best[j] = population[i]
                best.insert(j, population[i])
                break #break so as to not overwrite both parents
    return best[:10]

def reproduce(parent, NChild, agent):
    "produces NChildren number of children from the two parents and mutates"
    splitPoint = np.random.randint(1, parent[0].brain.Nhidd)
    child = agent()
    #splice together to create initial child
    for i in range(child.brain.Nhidd):
        for j in range(splitPoint): #copy up to splitting point
            child.brain.synapses1[:,j] = parent[0].brain.synapses1[:,j] #copy weights from first parent
            child.brain.synapses2[j] = parent[0].brain.synapses2[j] #copy weights from first parent
            child.brain.biases[j] = parent[0].brain.biases[j]
        for k in range(splitPoint, child.brain.Nhidd): #copy from splitting point
            child.brain.synapses1[:,k] = parent[1].brain.synapses1[:,k] #copy weights from second parent
            child.brain.synapses2[k] = parent[1].brain.synapses2[k] #copy weights from second parent
            child.brain.biases[k] = parent[1].brain.biases[k]
    children = [0]*NChild #create array of length NChild
    NMutations = int(0.3*child.brain.Nhidd)+1 #int rounds down, dont want too low so +1
   
    for i in range(NChild):
        children[i] = copy.deepcopy(child)
        #mutate
        for j in range(NMutations):
            mutationSize=np.random.randn()/8
            r1 = np.random.randint(child.brain.Nhidd)
            magnitude = 2*np.random.randint(2) - 1
            children[i].brain.synapses1[0,r1] += mutationSize*magnitude

            r2 = np.random.randint(child.brain.Nhidd)
            magnitude = 2*np.random.randint(2) - 1
            children[i].brain.synapses1[1,r2] += mutationSize*magnitude

            r3 = np.random.randint(child.brain.Nhidd)
            magnitude = 2*np.random.randint(2) - 1
            children[i].brain.synapses2[r3] += mutationSize*magnitude

            r4 = np.random.randint(child.brain.Nhidd)
            magnitude = 2*np.random.randint(2) - 1
            children[i].brain.biases[r4] += mutationSize*magnitude
    
    return children

def populationControl(population, NChild):
    "creates new generation from parents and their children"
    best = fittest(population)
    for i in best:
        i.resurect()
    children = reproduce(best, NChild)
    newPop = best
    for i in range(NChild):
        newPop.append(children[i])
    return newPop
    
                             

