import gym
import QLearnTabular
import sys
sys.path.append('../../')
import runEpisode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

maxIter = 1000                         #with no cart velocity          with experience replay
nStates=8         #best:8 solved 460          8  188
alpha = 1         #best:1 solved 460  
minAlpha = 0.05   #best:0.05 solved 460                              0.01   523
alphaRate = 40    #best:40 solved 460                                20     523
gamma = 1         #best:1 solved 460
epsilon = 1       #best:1 solved 460
mineps = 0.01     #best:0.01 solved 460       0.01 188
epsilonRate = 12  #best:40 solved 460         12   188                   35     523

x = []
yscores = []
yeps = []
yalpha = []

environment = gym.make('LunarLander-v2')
agent = QLearnTabular.QLearnTabular(nStates, environment, alpha, gamma, epsilon)
streak = 0
for i in range(maxIter):
    t = runEpisode.play(environment, agent, True)
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
    agent.alpha = agent.epsilon   #best: solved 452
    if i%20 == 0:
        print("Episode Number {}".format(i))

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
