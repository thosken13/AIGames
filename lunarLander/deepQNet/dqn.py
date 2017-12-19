import numpy as np
import tensorflow as tf

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
            inpt = tf.placeholder(tf.float32, shape=[self.batchSize, self.features]) #any number (batchSize) of vectors of length self.obs
            hidd1Weights = tf.Variable(tf.random_normal(shape=[self.features, hiddenNodes]))
            hidd1Biases = tf.Variable(tf.random_normal(shape=[self.features, hiddenNodes]))
            hidd2Weights = tf.Variable(tf.random_normal(shape=[hiddenNodes, hiddenNodes]))
            hidd2Biases = tf.Variable(tf.random_normal(shape=[hiddenNodes, hiddenNodes]))
            outptWeights = tf.Variable(tf.random_normal(shape=[hiddenNodes, self.actions]))
            putptBiases = tf.Variable(tf.random_normal(shape=[hiddenNodes, self.actions]))
            #computational graph
            hidd1 = tf.nn.leaky_relu(tf.add(tf.matmul(inpt, hidd1Weights), hidd1Biases))
            keepProb = tf.placeholder('float')
            hidd1DropOut = tf.nn.dropout(hidd1, keepProb)
            hidd2 = tf.nn.leaky_relu(tf.add(tf.matmul(hidd1DropOut, hidd2Weights), hidd2Biases))
            "unsure about softmax"outpt = tf.nn.softmax(tf.add(tf.matmul(hidd2, outptWeights), outptBiases))
            ############## add optimizer ################
        netDict = {"graph": g, "in": inpt, "out": outpt, "keepProb": keepProb}
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
        with tf.Session(graph=trainDict["graph"]) as sessT:
            with tf.Session(graph=evalDict["graph"]) as sessE:
                #################
        
    def test(self):
        "test the network"
        
    def update(self, reward, observation):
        "updates the q network approximator given result of action"
        self.experience.append((self.prevObs, self.prevAction, reward, observation))
        if len(experience) >= self.batchSize:
            #train
        
        self.prevObs = observation
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
