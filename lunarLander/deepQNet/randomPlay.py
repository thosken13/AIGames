import sys
sys.path.append("../")
import runEpisode
import numpy as np

class randomAgent:
    
    def __init__(self, environment):
        self.env = environment
        self.obs =[]
    
    def action(self, observation):
        "plays random actions"
        return self.env.action_space.sample()
    
    def update(self, reward, observation, done):
        self.obs.append(observation)
        
    def processObs(self, observation):
        pass
        
    def calcMeanStd(self):
        """calculate mean and standard deviation for each feature
         from the observations"""
        obs = np.array(self.obs)
        mean = np.mean(obs, 0)
        stdev = np.std(obs, 0)
        return mean, stdev
        
        
def randomPlay(environment, nIter=500):
    "play random to produce mean and standard deviation"
    agent = randomAgent(environment)
    for _ in range(nIter):
        runEpisode.play(environment, agent, False, False)
    return agent.calcMeanStd()
        
    
