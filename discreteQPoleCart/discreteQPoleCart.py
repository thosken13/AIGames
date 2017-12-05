import gym
import QLearnTabular
import sys
sys.path.append('../')
import runEpisode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

maxIter = 10000
nStates=8
alpha = 1
minAlpha = 0.1
gamma = 1
epsilon = 1
deps = -0.0003
mineps = 0.05

x = np.arange(maxIter)
y = []

environment = gym.make('CartPole-v0')
agent = QLearnTabular.QLearnTabular(nStates, environment, alpha, gamma, epsilon)

print("Start")
for i in range(maxIter):
    runEpisode.play(environment, agent, False)
    y.append(agent.score)
    agent.reset()
    alph = max(alpha * (0.85 ** (i//100)), minAlpha)
    agent.alpha = alph
    print("alpha ",alph)
    #agent.epsilon += deps
    agent.epsilon = max(min(1, 1.0 - math.log10((i+1)/25)), mineps)

data = {"score": y}
df = pd.DataFrame(data)
df.rolling(window=int(maxIter/500)).mean()
plt.plot(x, df.rolling(window=10).mean(), "x")
plt.show()
