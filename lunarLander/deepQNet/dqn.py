import numpy as np
import tensorflow as tf
import random

class dqn:
    def __init__(self, environment, alpha, gamma, epsilon, hiddenNodes, batchSize):
        self.env = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.prevAction = 0
        self.prevObs = 0
        self.score = 0
        self.actions = self.env.action_space.shape[0]
        self.features = self.env.observation_space.shape[0]
        self.experience = []
        self.batchSize = batchSize
        self.netDict1 = self.buildModel(hiddenNodes) # 2 nets to alternate between for stability
        self.netDict2 = self.buildModel(hiddenNodes)
        
    def buildModel(self, hiddenNodes):
        "builds the neural network"
        g = tf.Graph()
        with g.as_default():
            #variables
            inpt = tf.placeholder(tf.float32, shape=[None, self.features]) #any number (batchSize for training, or single for evaluation) of vectors of length self.obs
            hidd1Weights = tf.Variable(tf.random_normal(shape=[self.features, hiddenNodes]))
            hidd1Biases = tf.Variable(tf.random_normal(shape=[hiddenNodes])) #broadcasting applies to addition to all rows
            hidd2Weights = tf.Variable(tf.random_normal(shape=[hiddenNodes, hiddenNodes]))
            hidd2Biases = tf.Variable(tf.random_normal(shape=[hiddenNodes]))
            outptWeights = tf.Variable(tf.random_normal(shape=[hiddenNodes, self.actions]))
            outptBiases = tf.Variable(tf.random_normal(shape=[self.actions]))
            #computational graph
            hidd1 = tf.nn.leaky_relu(tf.add(tf.matmul(inpt, hidd1Weights), hidd1Biases))
            keepProb = tf.placeholder(tf.int32)
            hidd1DropOut = tf.nn.dropout(hidd1, keepProb)
            hidd2 = tf.nn.leaky_relu(tf.add(tf.matmul(hidd1DropOut, hidd2Weights), hidd2Biases))
            "unsure about softmax"outpt = tf.nn.softmax(tf.add(tf.matmul(hidd2, outptWeights), outptBiases))
            #Optimization
            target = tf.placeholder(tf.float32, shape=[self.actions])
            cost = tf.losses.mean_squared_error(target, outpt) #check axis done over
            optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(cost) #implement something explicitly?
            
        netDict = {"graph": g, "in": inpt, "out": outpt, "keepProb": keepProb,
                   "target": target, "optimizer": optimizer}
        return netDict
        
    def qApproxNet(self, observation, netDict):
        "calculates approximation for Q values for all actions at state"
        with tf.Session(graph=netDict["graph"]) as sess:
            qVals = sess.run(netDict["out"], feed_dict={netDict["in"]: observation, netDict["keepProb"]: 1})
        return qVals
    
    def action(self, observation, netDict):
        "chooses an action given observation"
        if np.random.rand() < self.epsilon:
            self.prevAction = self.env.action_space.sample()
        else:
            self.prevAction = np.argmax(qApproxNet(observation, netDict))
        return self.prevAction
    
    def train(self, trainDict, evalDict, keepProb):
        "train Q approximator network using batches from experience replay"
        batch = random.sample(self.experience, self.batchSize)
        prevObs = []
        reward = []
        nextObs = []
        for i in range(self.batchSize):
            prevObs.append(batch[i][0])
            reward.append(batch[i][1])
            nextObs.append(batch[i][2])
        with tf.Session(graph=trainDict["graph"]) as sess:
            target = np.array(reward) + self.gamma*np.array(qApproxNet(nextObs, evalDict))
            sess.run(trainDict["optimizer"], feed_dict={trainDict["in"]: prevObs, trainDict["keepProb"]: keepProb, trainDict["target"]: target})
        
    def test(self):
        "test the network"
        
    def update(self, reward, observation):
        "updates the q network approximator given result of action"
        self.experience.append([self.prevObs, reward, observation])
        if len(experience) >= self.batchSize:
            #choose train and eval graph
            self.train()        
        self.prevObs = observation
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
