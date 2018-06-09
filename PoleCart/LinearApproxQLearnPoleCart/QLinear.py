import numpy as np
import math


class QLinear:
    def __init__(self, environment, alpha, gamma, epsilon):
        self.env = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.prevStateAct = 0
        self.prevAction = 0
        self.prevQ = 0
        self.score = 0
        self.actions = self.env.action_space.shape[0]
        self.space = self.env.observation_space.shape[0]
        self.linApproxWeights = np.random.uniform(low=-1, high=1, size=self.space*self.actions)
        print(self.linApproxWeights)#############

    def featureVec(self, observation, action):
        fVec = np.hstack((observation,observation)) #will be value of feature if the corresponding action is taken, else 0   (f1, f2, f1, f2, f1, f2)
        actionMultiplier = np.zeros_like(self.linApproxWeights)
        actionMultiplier[action:action+self.space] = 1
        fVec = fVec*actionMultiplier #turn off all state-actions where the action isn't being taken
        return fVec


    def linearQApproximator(self, observation, action):
        fVec = self.featureVec(observation, action)
        return np.dot(fVec, self.linApproxWeights)
        
    def maxActionVal(self, observation):
        actVals = []
        for a in range(self.actions):
            actVals.append(self.linearQApproximator(observation, a))
        act = np.argmax(actVals)
        return act

    def action(self, observation):
        "choose an action based off Q-Values"
        if np.random.rand() < self.epsilon:
            self.prevAction = self.env.action_space.sample()
        else:
            self.prevAction = self.maxActionVal(observation)
        self.prevStateAct = self.featureVec(observation, self.prevAction)
        self.prevQ = self.linearQApproximator(observation, self.prevAction)
        return self.prevAction        

    def update(self, reward, observation):
        target = reward + self.gamma*self.linearQApproximator(observation, self.maxActionVal(observation))
        dW = self.alpha*(target - self.prevQ)*self.prevStateAct
        self.linApproxWeights += dW
        self.score += reward

    def reset(self):
        self.score=0



