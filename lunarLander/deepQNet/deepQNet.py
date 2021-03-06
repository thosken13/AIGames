import gym
import dqn
import randomPlay
import sys
sys.path.append('../')
import runEpisode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#unset info messages due to being in interactive
import logging as _logging
from logging import WARN
_logger = _logging.getLogger('tensorflow')
_logger.setLevel(WARN)

maxIter = 1000
alpha1 = 0.00002
alpha2 = 0.0005 #constant for ADAM optimizer (decay built in)
lrSplit = 100
targetUpdateFrac = 0.01####
#minAlpha = 0.05 
#alphaRate = 35  
gamma = 0.99     
epsilon = 1     
mineps = 0.1
epsFactor = 0.996
epsilonRate = 50#100
hiddenNodes = 50
dropOutKeepProb = 0.5#############################
#need to sort out these!!
batchSize = 32
minBatches = 50 #number of batches before starting training
trainFreq = 100 #train when totStepNumber%trainFreq == 0
numTrainBatches = 100 #number of batches that are trained (adjust alongside trainFreq) stops needing to load session so many times
setNetFreq = 10
maxExperience = 500 #oldest batch is removed once experience = (maxExperience+1)*batchSize

#meanObs = np.array([0, 0.6, 0, -0.6, 0, 0, 0, 0])
#stdObs = np.array([0.3, 0.3, 0.6, 0.5, 0.5, 0.4, 0.1, 0.1])

x = []
yscores = []
yeps = []
yalpha = []

environment = gym.make('LunarLander-v2')
initObs = environment.reset()
meanObs, stdObs = randomPlay.randomPlay(environment)
agent = dqn.dqn(environment, alpha1, alpha2, gamma, epsilon, hiddenNodes, batchSize, minBatches, dropOutKeepProb, initObs, maxExperience, trainFreq, numTrainBatches, setNetFreq, lrSplit, targetUpdateFrac, meanObs, stdObs)
streak = 0
for i in range(maxIter):
    if i%20 == 0:
        print("Episode Number {}, epsilon {}".format(i, agent.epsilon))
    t = runEpisode.play(environment, agent, False)
    x.append(i+1)
    yscores.append(agent.score)
    #yalpha.append(agent.alpha)
    yeps.append(agent.epsilon)
    if i+1 >= 100:
        if sum(yscores[-100:])/100 >= 195:
            print("Solved after {} episodes!".format(i+1))
            break
    agent.finalScore = agent.score
    agent.scorer(agent.score)
    agent.reset()
    #agent.alpha = max(alpha * (0.85 ** (i//alphaRate)), minAlpha)
    #agent.epsilon = max(min(1, 1 - math.log10((i+1)/epsilonRate)), mineps)
    agent.epsilon=max(agent.epsilon*epsFactor, mineps)
    #agent.alpha = agent.epsilon

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
#plt.subplot(3,1,3)
#plt.plot(x, yalpha)
#plt.ylim((0,1))
#plt.plot(x, np.ones_like(x)*minAlpha)
plt.xlabel("episode")
plt.ylabel("learning rate")
plt.show()
