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

x = np.arange(maxIter)
y = []
yeps = []
yalpha = []

environment = gym.make('CartPole-v0')
agent = QLearnTabular.QLearnTabular(nStates, environment, alpha, gamma, epsilon)
streak = 0
for i in range(maxIter):
    t = runEpisode.play(environment, agent, False)
    if t >= 195:
        streak += 1
        if streak == 100:
            print("Solved after {} episodes!".format(i+1))
            break
    else:
        streak = 0
    y.append(agent.score)
    agent.reset()
    agent.alpha = max(alpha * (0.85 ** (i//100)), minAlpha)
    yalpha.append(agent.alpha)
    agent.epsilon = max(min(1, 1.0 - math.log10((i+1)/50)), mineps)
    yeps.append(agent.epsilon)

runEpisode.play(environment, agent, True)

data = {"score": y}
df = pd.DataFrame(data)
df.rolling(window=int(maxIter/500)).mean()
plt.subplot(3,1,1)
plt.plot(x, df.rolling(window=10).mean(), "x")
plt.subplot(3,1,2)
plt.plot(x, yeps)
plt.subplot(3,1,3)
plt.plot(x, yalpha)
plt.show()
