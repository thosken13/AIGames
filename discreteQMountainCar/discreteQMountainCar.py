import gym
import sys
sys.path.append('../')
import mountainCar
import QLearnTabular
import matplotlib.pyplot as plt
import numpy as np

maxIter = 1000
nStates=40
alpha = 0.3
gamma = 1
deps = -0.0005

x = np.arange(maxIter)
y = []

environment = gym.make('MountainCar-v0')
driver = QLearnTabular.QLearnTabular(nStates, environment, alpha, gamma, deps)

for _ in range(maxIter):
    mountainCar.play(environment, driver)
    y.append(driver.score)
    driver.reset()

plt.plot(x,y)
plt.show()
