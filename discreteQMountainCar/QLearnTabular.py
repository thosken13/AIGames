import numpy as np

class QLearnTabular:
    def __init__(self, nStates, environment, alpha, gamma, deps):
        self.nStates = nStates #number of states per observation per action
        self.env = environment
        self.epsilon = 0.02
        self.deps = deps
        self.alpha = alpha
        self.gamma = gamma
        self.prevState = 0
        self.prevAction = 0

        dim = self.env.observation_space.shape[0]
        shape = []
        shape.append(self.env.action_space.shape[0])
        for d in range(dim):
            shape.append(nStates)
        self.qTable = np.zeros(shape)
        self.visited = np.zeros(shape)################################

        self.score = 0

    
    def discretiseState(self, observation):
        "discretise the observations so that can use q-learning in tabular form"
        envMin = self.env.observation_space.low
        envMax = self.env.observation_space.high
        envStep = (envMax - envMin)/self.nStates
        i = int((observation[0] - envMin[0])/envStep[0])
        j = int((observation[1] - envMin[1])/envStep[1])
        self.prevState = (i,j)
        return i, j

    def action(self, observation):
        "choose an action based off observation"
        s = self.discretiseState(observation)
        if np.any(self.qTable[:, s[0],s[1]] == 0):
            self.prevAction = np.argmin(self.qTable[:, s[0], s[1]])
        elif np.random.rand() < self.epsilon:
            self.prevAction = self.env.action_space.sample()
        else:
            s = self.discretiseState(observation)
            self.prevAction = np.argmax(self.qTable[:, s[0], s[1]])
        return self.prevAction
            

    def update(self, reward, observation):
        "update the Q value table"
        s = self.discretiseState(observation)
        self.qTable[self.prevAction, self.prevState[0], self.prevState[1]] = (1 - self.alpha)*self.qTable[self.prevAction, self.prevState[0], self.prevState[1]] +\
                                                                             self.alpha*(reward + self.gamma*np.max(self.qTable[:, s[0], s[1]]))

        self.score+=reward
        #self.epsilon += self.deps

    def reset(self):
        self.score=0



