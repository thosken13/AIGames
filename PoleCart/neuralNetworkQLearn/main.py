import gym
import sys
sys.path.append('../../')
import runEpisode
import agent as nnAgent

env = gym.make('CartPole-v0')
env.seed(0)
print("obs space", env.observation_space.shape)
print("act space", env.action_space.shape)
for i in range(5):
    print(env.action_space.sample())

agent = nnAgent.NNAgent()
agent.test()
