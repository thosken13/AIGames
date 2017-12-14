import numpy as np
import math


class SARSALambdaTabular:
    def __init__(self, nStates, environment, alpha, gamma, epsilon, lambd):
        self.nStates = nStates #number of states per observation per action
        self.env = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.prevState = 0
        self.prevAction = 0
        self.prevreward = 0
        self.prevPrevStateAction = 0
        self.score = 0

        dim = self.env.observation_space.shape[0]
        shape = []
        for d in range(3):#dim):
            shape.append(nStates)
        shape.append(self.env.action_space.shape[0])
        self.qTable = np.zeros(shape)
        self.eligibilityTrace = np.zeros(shape)


    def discretiseState(self, observation):
        "discretise the observations so that can use q-learning in tabular form"
        envMin = self.env.observation_space.low
        envMin[1] = -3.4  #-2.5
        envMin[3] = -3.3
        envMax = self.env.observation_space.high
        envMax[1] = 3.4   #2.5
        envMax[3] = 3.3
        envStep = (envMax - envMin)/self.nStates
        s = []
        for i in [0,2,3]:#range(np.shape(observation)[0]): 
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
        self.eligibilityTrace[tuple(sPrev)] += 1
        if self.score != 0: #can't update on first step because don't have prevPrevStateAction
            delta = self.prevReward + self.gamma*self.qTable[tuple(sPrev)] - self.qTable[tuple(self.prevPrevStateAction)]
            self.qTable += self.alpha*delta*self.eligibilityTrace
        self.eligibilityTrace = self.lambd*self.gamma*self.eligibilityTrace
        self.score+=reward
        self.prevPrevStateAction = sPrev
        self.prevReward = reward

    def reset(self):
        self.score=0



