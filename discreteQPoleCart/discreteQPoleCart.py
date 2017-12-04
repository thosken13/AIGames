import gym
import QLearnTabular
import sys
sys.path.append('../')
import runEpisode
import matplotlib.pyplot as plt
import numpy as np

maxIter = 10000
nStates=40
alpha = 1
minAlpha = 0.003
gamma = 1
deps = -0.0000

x = np.arange(maxIter)
y = []

environment = gym.make('CartPole-v0')
agent = QLearnTabular.QLearnTabular(nStates, environment, alpha, gamma, deps)

print("Start")
#for i in range(maxIter):
while True:
    runEpisode.play(environment, agent, True)
    y.append(agent.score)
    driver.reset()
    alph = max(alpha * (0.85 ** (i//100)), minAlpha)
    agent.alpha = alph

print(agent.qTable)
print(agent.epsilon)
plt.plot(x,y)
plt.show()
