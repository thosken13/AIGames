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
    agent.testNpSeed()
    print(agent.action([1,2,3,4]))

def getRefObs():
    env = gym.make('CartPole-v0')
    env.seed(0)
    env.reset()
    d=False
    while not d:
        obs, r, d, i = env.step(env.action_space.sample())
        print(obs)

def runEpisode(environment, agent, train, render):
    done=False
    score=0
    obs = environment.reset()
    agent.prevState = np.array(obs)
    while not done:
        action = agent.action(obs)
        obs, reward, done, info = environment.step(action)
        if train:
            agent.update(obs, action, reward, done)
        if render:
            environment.render()
        if not done:
            score+=1
    return score

def runEpisodes(numEpisodes, environment, agent, train, render, saveFreq):
    for e in range(numEpisodes):
        runEpisode(environment, agent, train, render)
        if (e+1)%validationFreq == 0:
            epsilon = agent.epsilon
            agent.epsilon = 0
            score = runEpisode(environment, agent, False, False)
            agent.epsilon = epsilon
            agent.score=score
        if saveFreq and (e+1)%saveFreq==0: #check saveFreq not None (no saving)
            agent.save()
            agent.restore()
        agent.reset()

def playAndTrain(numEpisodes, saveFreq=10, render=False, nHidd=[10,10,10]):
    env = gym.make('CartPole-v0')
    env.seed(0)
    agent = nnAgent.NNAgent(env, nNeuronsHidLayers=nHidd, alpha=0.005, epsilonDecay=0.99, batchSize=16)
    runEpisodes(numEpisodes, env, agent, True, render, saveFreq)
    agent.kill()

def loopArchitecture():
    for layers in range(1,6):
        for nodesPerLayer in range(2,15):
            playAndTrain(400, saveFreq=None, nHidd=[nodesPerLayer]*layers)


validationFreq=10

#testAndExperiment()
#getRefObs()
#playAndTrain(300)
playAndTrain(300, saveFreq=None)
#loopArchitecture()
