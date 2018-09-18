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

def runEpisodes(numEpisodes, environment, agent, train, render, saveFreq, solveStop=False):
    for e in range(numEpisodes):
        print("Episode No.: {}".format(e), "Episode Length: ", runEpisode(environment, agent, train, render))
        if (e+1)%validationFreq == 0: #do a validation step
            epsilon = agent.epsilon
            agent.epsilon = 0
            score=0
            for i in range(nValEps):
                score += runEpisode(environment, agent, False, False)
            score /= nValEps
            agent.score=score
            agent.epsilon = epsilon
            print("Vaidation score: {}".format(agent.score))
        if saveFreq and (e+1)%saveFreq==0: #check saveFreq not None (no saving)
            agent.save()
            agent.restore()
        if solveStop and agent.score > 190:
            print("Solved in {} episodes!".format(e+1))
            return
        agent.reset()

def playAndTrain(numEpisodes, saveFreq=10, render=False, nHidd=[10,10,10], solveStop=False):
    env = gym.make('CartPole-v0')
    env.seed(0)
    agent = nnAgent.NNAgent(env, nNeuronsHidLayers=nHidd, alpha=0.001, epsilonDecay=0.99, batchSize=16)
    runEpisodes(numEpisodes, env, agent, True, render, saveFreq, solveStop=solveStop)
    agent.kill()

def randomPlay(numEpisodes, render=False):
    env = gym.make('CartPole-v0')
    env.seed(0)
    agent = nnAgent.NNAgent(env)
    agent.minEpsilon = 1
    runEpisodes(numEpisodes, env, agent, True, render, None)
    agent.kill()

def loopArchitecture(numEpisodes):
    randomPlay(numEpisodes)
    for layers in range(2,6):
        for nodesPerLayer in reversed(range(2,15)):
            playAndTrain(numEpisodes, saveFreq=None, nHidd=[nodesPerLayer]*layers, solveStop=True)


validationFreq=10
nValEps=10 #number of validation episodes to calculate score

#testAndExperiment()
#getRefObs()
#playAndTrain(300)
#playAndTrain(300, saveFreq=None)
loopArchitecture(400)
