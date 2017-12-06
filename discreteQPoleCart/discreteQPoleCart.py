import gym
import QLearnTabular
import sys
sys.path.append('../')
import runEpisode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

maxIter = 1000
nStates=8
alpha = 1
minAlpha = 0.1
gamma = 1
epsilon = 1
mineps = 0.01

x = []
yscores = []
yeps = []
yalpha = []

environment = gym.make('CartPole-v0')
agent = QLearnTabular.QLearnTabular(nStates, environment, alpha, gamma, epsilon)
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
    agent.alpha = max(alpha * (0.85 ** (i//80)), minAlpha)
    agent.epsilon = max(min(1, 1.0 - math.log10((i+1)/50)), mineps)

runEpisode.play(environment, agent, True)

data = {"score": yscores}
df = pd.DataFrame(data)
plt.subplot(3,1,1)
plt.plot(x, df.rolling(window=int(maxIter/500)).mean(), "x")
plt.subplot(3,1,2)
plt.plot(x, yeps)
plt.ylim((0,1))
plt.subplot(3,1,3)
plt.plot(x, yalpha)
plt.ylim((0,1))
plt.show()
