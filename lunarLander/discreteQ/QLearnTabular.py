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
        self.score = 0

        dim = self.env.observation_space.shape[0]
        shape = []
        for d in range(dim):
            shape.append(nStates)
        shape.append(self.env.action_space.shape[0])
        self.qTable = np.zeros(shape)

        self.experiences = []
        self.visited = np.zeros(shape)
        self.envMin = self.env.observation_space.low
        self.envMin[0] = -1
        self.envMin[1] = -1.1
        self.envMin[2] = -1.5
        self.envMin[3] = -1.51
        self.envMin[4] = -2.5
        self.envMin[5] = -5
        self.envMin[6] = 0.0
        self.envMin[7] = 0.0
        self.envMax = self.env.observation_space.high
        self.envMax[0] = -self.envMin[0]
        self.envMax[1] = -self.envMin[1]
        self.envMax[2] = -self.envMin[2]
        self.envMax[3] = -self.envMin[3]
        self.envMax[4] = -self.envMin[4]
        self.envMax[5] = -self.envMin[5]
        self.envMax[6] = 1.1
        self.envMax[7] = 1.1
        self.envStep = (self.envMax - self.envMin)/self.nStates
#################### could tweak state min max more (find range around good position) ##################################

    def discretiseState(self, observation):
        "discretise the observations so that can use q-learning in tabular form"
        s = []
        for i in range(np.shape(observation)[0]): 
            s_ = int((observation[i] - self.envMin[i])/self.envStep[i])
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
            
    def experienceReplay(self, stateAction, reward, nextState):
        "build memory of experiences to help speed up learning by sampling from the memory"
        e = (stateAction, reward, nextState)
        self.experiences.append(e)
        if len(self.experiences) == 2000:      #CAN CHANGE SIZE OF MEMORY
            del self.experiences[0]
        for i in range(min(100 , len(self.experiences))):           #CAN CHANGE NUMBER OF EXPERIENCES TO LEARN FROM
            e_ = self.experiences[np.random.randint(0, len(self.experiences))]
            self.qTable[tuple(e_[0])] = (1 - self.alpha)*self.qTable[tuple(e_[0])] + self.alpha*(e_[1] + self.gamma*np.max(self.qTable[tuple(e_[2])]))
      #############what learning rate for experience?###################

    def frequencyLearningRate(self, sPrev):
        "have a learning rate proportional to frequency"
        self.visited[tuple(sPrev)] += 1
        self.alpha = 1/self.visited[tuple(sPrev)]
        

    def update(self, reward, observation):
        "update the Q value table"
        sPrev = self.prevState
        sPrev.append(self.prevAction)
        s = self.discretiseState(observation)
        #self.frequencyLearningRate(sPrev)
        self.qTable[tuple(sPrev)] = (1 - self.alpha)*self.qTable[tuple(sPrev)] + self.alpha*(reward + self.gamma*np.max(self.qTable[tuple(s)]))
        self.score+=reward
        #self.experienceReplay(sPrev, reward, s)

    def reset(self):
        self.score=0



