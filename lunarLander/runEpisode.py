import sys
sys.path.append('../../../')
import gym
import numpy as np

def play(env, agent, render, printRes=True):
    #print(env.action_space)
    #print(env.observation_space)
    observation = env.reset()
    agent.prevObs = agent.processObs(observation)
    #print(observation)
    done = False
    t=0
    while not done:
        #print("observation = ", observation)
        #action = env.action_space.sample()
        action = agent.action(observation)
        #print(action)
        observation, reward, done, info = env.step(action)
        agent.update(reward, observation, done)
        t+=1
        if printRes:
            if reward >= 100:
                print("Landed!")
            if reward == 10:
                print("Leg-ground contact")
        if render:
           env.render()
    if printRes:
        print("Episode finished after {} timesteps, with a score of {}".format(t, agent.score))
    return t
    
    
    
    
