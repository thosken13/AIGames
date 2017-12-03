import mountainCar
import genetic
import Brain
import numpy as np
import matplotlib.pyplot as plt

class Driver:
    def __init__(self, args):
        Nin = args[0]
        Nhidd = args[1]
        self.brain = Brain.Brain(Nin, Nhidd, 3)
        self.score = 0
        self.vmax = 0

    def action(self, observation):
        self.brain.input(observation)
        output = self.brain.calculate()
        #print("output = ", output)
        return np.argmax(output)


    def update(self, reward, observation):
        #self.score += reward
        #self.score += abs(observation[1])
        if abs(observation[1])>self.vmax:
            self.vmax = abs(observation[1])
            self.score = self.vmax

    def resurect(self):
        self.score = 0
        self.vmax = 0

maxGenerations = 20
NChildren = 6
Nagents = 10
agents = []
for a in range(Nagents):
    agents.append(Driver((2,3)))
scores = np.zeros((maxGenerations, Nagents))
for gen in range(maxGenerations):
    for i, agent in enumerate(agents):
        mountainCar.play(agent)
        print(agent.score)
        scores[gen,i] = agent.score
    agents = genetic.populationControl(agents, NChildren, Driver, 2, 3)
    print(gen+1)
    
x = np.arange(1,maxGenerations+1)
y = np.mean(scores,1)
e = np.std(scores,1)
for i in range(maxGenerations):
    plt.errorbar(x, y, yerr=e)
plt.xlabel("Generation")
plt.ylabel("Fitness Score")
plt.show()

