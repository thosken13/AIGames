import gym
import dqn
import sys
sys.path.append('../')
import runEpisode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

maxIter = 1000
alpha = 1 
minAlpha = 0.05 
alphaRate = 35  
gamma = 0.98     
epsilon = 1     
mineps = 0.01   
epsilonRate = 1000
hiddenNodes = 30
batchSize = 1000
dropOutKeepProb = 0.5

x = []
yscores = []
yeps = []
yalpha = []

environment = gym.make('LunarLander-v2')
initObs = environment.reset()
agent = dqn.dqn(environment, alpha, gamma, epsilon, hiddenNodes, batchSize, dropOutKeepProb, initObs)
streak = 0
for i in range(maxIter):
    t = runEpisode.play(environment, agent, False)
    x.append(i+1)
    yscores.append(agent.score)
    yalpha.append(agent.alpha)
    yeps.append(agent.epsilon)
    if i+1 >= 100:
        if sum(yscores[-100:])/100 >= 195:
            print("Solved after {} episodes!".format(i+1))
            break
    agent.reset()
    #agent.alpha = max(alpha * (0.85 ** (i//alphaRate)), minAlpha)
    agent.epsilon = max(min(1, 1 - math.log10((i+1)/epsilonRate)), mineps)
    agent.alpha = agent.epsilon
    if i%20 == 0:
        print("Episode Number {}".format(i))
    agent.equateWeights()

runEpisode.play(environment, agent, True)

data = {"score": yscores}
df = pd.DataFrame(data)
plt.subplot(3,1,1)
plt.plot(x, df.rolling(window=int(maxIter/500)).mean(), "x")
plt.ylabel("SMA({}) of score".format(maxIter/500))
plt.subplot(3,1,2)
plt.plot(x, yeps)
plt.ylim((0,1))
plt.ylabel("exploration rate")
plt.subplot(3,1,3)
plt.plot(x, yalpha)
plt.ylim((0,1))
plt.plot(x, np.ones_like(x)*minAlpha)
plt.xlabel("episode")
plt.ylabel("learning rate")
plt.show()
