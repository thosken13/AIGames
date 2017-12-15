import numpy as np
import math


class QLinear:
    def __init__(self, environment, alpha, gamma, epsilon):
        self.env = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.prevState = 0
        self.prevAction = 0
        self.score = 0
        self.actions = self.env.action_space.size[0]
        self.space = self.env.observation_space.shape[0]
        self.linApproxWeights = np.random.random(self.space*self.actions)
        print(linApproxWeights, np.shape(linApproxWeights))#############

    def linearValueApproximator(self, obvservation, action):
        featureVec = np.hstack((observation,observation,observation)) #will be value of feature if the corresponding action is taken, else 0   (f1, f2, f1, f2, f1, f2)
        actionMultiplier = np.zeros_like(self.linApproxWeights)
        actionMultiplier[action:action+self.space] = 1 #turn off all state-actions where the action isn't being taken
        featureVec = featureVec*actionMultiplier
        return np.dot(featureVec, self.linApproxWeights)
        

    def action(self, observation):
        "choose an action based off Q-Values"
        if np.random.rand() < self.epsilon:
            self.prevAction = self.env.action_space.sample()
        else:
            actVals = []
            for a in range(self.actions):
                actVals.append(linearValueApproximator(observation, a))
            self.prevAction = np.argmax(actVals)
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



