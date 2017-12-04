import gym
import sys
sys.path.append('../')
import mountainCar
import QLearnTabular
import matplotlib.pyplot as plt
import numpy as np

maxIter = 10000
nStates=40
alpha = 1
minAlpha = 0.003
gamma = 1
deps = -0.00005

x = np.arange(maxIter)
y = []

environment = gym.make('MountainCar-v0')
driver = QLearnTabular.QLearnTabular(nStates, environment, alpha, gamma, deps)

for i in range(maxIter):
    mountainCar.play(environment, driver)
    y.append(driver.score)
    driver.reset()
    alpha = max(alpha * (0.85 ** (i//100)), minAlpha)
    driver.alpha = alpha

print(driver.qTable)
print(driver.epsilon)
plt.plot(x,y)
plt.show()
