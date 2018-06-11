import gym
import numpy as np
import sys
sys.path.append('../../')
import runEpisode
import agent as nnAgent

def testAndExperiment():
    env = gym.make('CartPole-v0')
    env.seed(0)
    print("obs space", env.observation_space.shape)
    print("act space", env.action_space.shape)
    #test random seed of environment is fixed correctly by env.seed()
    # for i in range(5):
    #     print(env.reset())
    #     env.seed(0)
    # for i in range(5):
    #     print(env.action_space.sample()) #spaces are seeded separately and apparently by a fixed seed
    agent = nnAgent.NNAgent(env)
    #agent.test()
    # agent.testNpSeed()
    print(agent.action([1,2,3,4]))

def playAndTrain(numEpisodes, render=False):
    env = gym.make('CartPole-v0')
    env.seed(0)
    agent = nnAgent.NNAgent(env)
    for e in range(numEpisodes):
        done=False
        score=0
        obs = env.reset()
        agent.prevState = np.array(obs)
        while not done:
            action = agent.action(obs)
            obs, reward, done, info = env.step(action)
            agent.update(obs, action, reward, done)
            if render:
                env.render()
            if not done:
                score+=1
        agent.score=score



#testAndExperiment()
playAndTrain(1000)
