import numpy as np
import tensorflow as tf
import random

class dqn:
    def __init__(self, environment, alpha, gamma, epsilon, hiddenNodes, batchSize, keepProb, initObs):
        self.env = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.prevAction = 0
        self.prevObs = initObs
        self.score = 0
        self.actions = self.env.action_space.shape[0]
        self.features = self.env.observation_space.shape[0]
        self.experience = []
        self.batchSize = batchSize
        self.netDict = self.buildModel(hiddenNodes)
        self.keepProb = keepProb
        
    def buildModel(self, hiddenNodes):
        "builds the neural network"
        print("Building network")
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
            keepProb = tf.placeholder(tf.float32)
            hidd1DropOut = tf.nn.dropout(hidd1, keepProb)
            hidd2 = tf.nn.leaky_relu(tf.add(tf.matmul(hidd1DropOut, hidd2Weights), hidd2Biases))
            ########## unsure about softmax ########################
            outpt = tf.nn.softmax(tf.add(tf.matmul(hidd2, outptWeights), outptBiases))
            #Optimization
            target = tf.placeholder(tf.float32, shape=[None, self.actions])
            cost = tf.losses.mean_squared_error(target, outpt) #not completely sure about
            optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(cost) #implement something explicitly?
            saver = tf.train.Saver()
        netDict = {"graph": g, "in": inpt, "out": outpt, "keepProb": keepProb,
                   "target": target, "optimizer": optimizer, "saver": saver}
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, "sessionFiles/savedNetwork1")
            saver.save(sess, "sessionFiles/savedNetwork2")
        print("Built!")
        return netDict
        
    def qApproxNet(self, observation):
        "calculates approximation for Q values for all actions at state"
        with tf.Session(graph=self.netDict["graph"]) as sess:
            self.netDict["saver"].restore(sess, "sessionFiles/savedNetwork1")
            qVals = sess.run(self.netDict["out"], feed_dict={self.netDict["in"]: observation, self.netDict["keepProb"]: 1})
        return qVals
    
    def action(self, observation):
        "chooses an action given observation"
        if np.random.rand() < self.epsilon:
            self.prevAction = self.env.action_space.sample()
        else:
            self.prevAction = np.argmax(qApproxNet(observation))
        return self.prevAction
    
    def train(self):
        "train Q approximator network using batches from experience replay"
        batch = random.sample(self.experience, self.batchSize)
        prevObs = []
        reward = []
        nextObs = []
        for i in range(self.batchSize):
            prevObs.append(batch[i][0])
            reward.append(batch[i][1])
            nextObs.append(batch[i][2])
        with tf.Session(graph=self.netDict["graph"]) as sess:
            self.netDict["saver"].restore(sess, "sessionFiles/savedNetwork2")
            target = np.reshape(np.array(reward), (self.batchSize, 1)) + self.gamma*self.qApproxNet(nextObs)
            sess.run(self.netDict["optimizer"], feed_dict={self.netDict["in"]: prevObs, self.netDict["keepProb"]: self.keepProb, self.netDict["target"]: target})
            self.netDict["saver"].save(sess, "sessionFiles/savedNetwork2")
        
    def test(self):
        "test the network"
        
    def update(self, reward, observation):
        "updates the q network approximator given result of action"
        self.experience.append([self.prevObs, reward, observation])
        if len(self.experience) >= self.batchSize:
            self.train()
        self.prevObs = observation
        self.score += reward
    
    def equateWeights(self):
        "copies the more recently trained weights to the other graph"
        with tf.Session(graph=self.netDict["graph"]) as sess:
            self.netDict["saver"].restore(sess, "sessionFiles/savedNetwork2")
            #maybe need to run session here
            self.netDict["saver"].save(sess, "sessionFiles/savedNetwork1")
    
    def reset(self):
        "resets ready for another episode run"
        self.score=0
    
    
    
    
    
    
    
    
    
    
    
    
