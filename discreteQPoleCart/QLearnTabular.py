import numpy as np
import math


class QLearnTabular:
    def __init__(self, nStates, environment, alpha, gamma, epsilon):
        self.nStates = nStates #number of states per observation per action
        self.env = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.prevState = 0
        self.prevAction = 0

        dim = self.env.observation_space.shape[0]
        shape = []
        for d in range(dim):
            shape.append(nStates)
        shape.append(self.env.action_space.shape[0])
        self.qTable = np.zeros(shape)
        self.visited = np.zeros(shape)################################

        self.score = 0

    def discretiseState(self, observation):
        "discretise the observations so that can use q-learning in tabular form"
        envMin = self.env.observation_space.low
        envMin[1] = -2.5
        envMin[3] = -3.3
        envMax = self.env.observation_space.high
        envMax[1] = 2.5
        envMax[3] = 3.3
        envStep = (envMax - envMin)/self.nStates
        s = []
        for i in range(np.shape(observation)[0]): 
            s_ = int((observation[i] - envMin[i])/envStep[i])
            if s_ <0:
                s.append(0)
                print("underflow at {}, value of {}".format(i, observation[i]))
            elif s_ < self.nStates:
                s.append(s_)
            else:
                s.append(self.nStates-1)
                print("overflow at {}, value of {}".format(i, observation[i]))
        self.prevState = s
        return s

    def action(self, observation):
        "choose an action based off Q-Values"
        s = self.discretiseState(observation)
        if np.random.rand() < self.epsilon:
            self.prevAction = self.env.action_space.sample()
        else:
            self.prevAction = np.argmax(self.qTable[tuple(s)])
        return self.prevAction
            
    def update(self, reward, observation):
        "update the Q value table"
        sPrev = self.prevState
        sPrev.append(self.prevAction)
        s = self.discretiseState(observation)
        self.qTable[tuple(sPrev)] = (1 - self.alpha)*self.qTable[tuple(sPrev)] + self.alpha*(reward + self.gamma*np.max(self.qTable[tuple(s)]))
        self.score+=reward

    def reset(self):
        self.score=0



