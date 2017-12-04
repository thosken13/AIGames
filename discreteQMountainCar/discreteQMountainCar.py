import gym
import sys

sys.path.append('../')
import mountainCar

nState=40

def discretiseState(environment, observation):
    "discretise the observations so that can use q-learning in tabular form"
    envMin = environment.observation_space.low
    envMax = environment.observation_space.high
    envStep = (envMax - envMin)/nState
    i = int((observation[0] - envMin[0])/envStep[0])
    j = int((observation[1] - envMin[1])/envStep[1])
    return i, j

