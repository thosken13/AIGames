import gym
import numpy as np

def play(agent):
    env = gym.make('MountainCar-v0')
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
        env.render()
        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            env.render()
            break
        t+=1
