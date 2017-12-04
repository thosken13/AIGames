import numpy as np

class QLearnTabular:
    def __init__(self, nStates, environment):
        self.nStates = nStates
        self.env = environment
        self.epsilon = 1
        self.qTable = np.zeros(nStates, self.env.observation_space.shape[0]

    
    def discretiseState(observation):
        "discretise the observations so that can use q-learning in tabular form"
        envMin = self.env.observation_space.low
        envMax = self.env.observation_space.high
        envStep = (envMax - envMin)/nState
        i = int((observation[0] - envMin[0])/envStep[0])
        j = int((observation[1] - envMin[1])/envStep[1])
        return i, j

    def action(observation):
        "choose an action based off observation"
        if np.random.rand() < epsilon:
            act = self.env.action_space.sample()
        else:
            

    def update(reward, observation):