import gym
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
    for i in range(5):
        print(env.reset())
        env.seed(0)
    for i in range(5):
        print(env.action_space.sample()) #spaces are seeded separately and apparently by a fixed seed
    agent = nnAgent.NNAgent(env)
    #agent.test()
    agent.testNpSeed()



testAndExperiment()
