import gym
import numpy as np
import sys
sys.path.append('../../')
import runEpisode
import agent as nnAgent

def testAndExperiment():
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print("obs space", env.observation_space.shape)
    print("act space", env.action_space.shape)
    #test random seed of environment is fixed correctly by env.seed()
    # for i in range(5):
    #     print(env.reset())
    #     env.seed(0)
    # for i in range(5):
    #     print(env.action_space.sample()) #spaces are seeded separately and apparently by a fixed seed
    agent = nnAgent.NNAgent(8, 4)
    #agent.test()
    #agent.testNpSeed()
    print(agent.action([1,2,3,4,5,6,7,8]))
    print(agent.qNetsDict["session"].run(agent.qNetsDict["outTargetQ"], feed_dict={agent.qNetsDict["inTargetQ"]: np.reshape([1,2,3,4,5,6,7,8], (1,8))}))

def getRefObs():
    #[x, y, xVel, yVel, angle, angular velocity, left leg down, right leg down]
    env = gym.make('LunarLander-v2')
    env.seed(0)
    env.reset()
    d=False
    import matplotlib.pyplot as plt
    x=[]
    y=[]
    ic=0
    while not d:
        obs, r, d, i = env.step(env.action_space.sample())
        env.render()
        print(obs[4])
        x.append(ic)
        y.append(obs[4])
        ic+=1
    plt.plot(x, y)
    plt.show()

def runEpisode(environment, agent, train, render, returnReward=False):
    done=False
    score=0
    rewards=[]
    epLength=0
    obs = environment.reset()
    agent.prevState = np.array(obs)
    while not done and epLength < maxEpLeng:
        epLength+=1
        action = agent.action(obs)
        obs, reward, done, info = environment.step(action)
        if train:
            agent.update(obs, action, reward, done)
        if render:
            environment.render()
        score += reward
        rewards.append(reward)
        #print(reward)
    if returnReward:
        return score, rewards, epLength
    return score, epLength

def runEpisodes(numEpisodes, environment, agent, train, render, validationSteps=True, solveStop=False):
    for e in range(numEpisodes):
        score, epLength = runEpisode(environment, agent, train, render)
        agent.lastEpScore = score
        agent.hundredEpScores.append(score)
        if score >= 200:
            agent.solvesT += 1
        agent.lastEpLength = epLength
        print("Episode No.: {} Episode Score: {}".format(e, score))
        if validationSteps and (e+1)%validationFreq == 0: #do a validation step
            epsilon = agent.epsilon
            agent.epsilon = 0
            score=0
            for i in range(nValEps):
                s, rs, l = runEpisode(environment, agent, False, True, returnReward=True)
                print("score: {}".format(s))
                if s >= 200:
                    agent.solvesV += 1
                #print(rs)
                score += s
            score /= nValEps
            agent.score=score
            agent.epsilon = epsilon
            print("Vaidation score: {}".format(agent.score))
            if solveStop and agent.score > 190:
                print("Solved in {} episodes!".format(e+1))
                return
        agent.reset()

def playAndTrain(numEpisodes, render=False, nHidd=[40,40,40], solveStop=False, **agentParams):
    env = gym.make('LunarLander-v2')
    env.seed(0)
    agent = nnAgent.NNAgent(8, 4, **agentParams)
    runEpisodes(numEpisodes, env, agent, True, render, solveStop=solveStop)
    agent.kill()

def randomPlay(numEpisodes, render=False):
    env = gym.make('LunarLander-v2')
    env.seed(0)
    agent = nnAgent.NNAgent(8, 4, runName="random", minEpsilon=1.1)
    runEpisodes(numEpisodes, env, agent, True, render, validationSteps=False)
    agent.kill()

def loopArchitecture(numEpisodes):
    randomPlay(numEpisodes)
    for layers in range(2,6):
        for nodesPerLayer in reversed(range(2,15)):
            playAndTrain(numEpisodes, nHidd=[nodesPerLayer]*layers, solveStop=True)


validationFreq=50
nValEps=5 #number of validation episodes to calculate score
maxEpLeng=10000

#testAndExperiment()
#getRefObs()
#playAndTrain(300)
#randomPlay(2000)
#playAndTrain(2000)
#may need to add a limit on episode length, had two crashes possibly due to memory
playAndTrain(2000)
playAndTrain(3000, alpha=0.0001)
playAndTrain(2000, alpha=0.001, gamma=0.999)
