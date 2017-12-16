import gym
import numpy as np

def play(env, agent, render):
    #print(env.action_space)
    #print(env.observation_space)
    observation = env.reset()
    #print(observation)
    done = False
    t=0
    while not done:
        #print("observation = ", observation)
        #action = env.action_space.sample()
        action = agent.action(observation)
        #print(action)
        observation, reward, done, info = env.step(action)
        agent.update(reward, observation)
        if render:
            env.render()
        t+=1
    print("Episode finished after {} timesteps, with a score of {}".format(t, agent.score))
    return t
