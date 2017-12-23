import numpy as np
import tensorflow as tf
import random

class dqn:
    def __init__(self, environment, alpha, gamma, epsilon, hiddenNodes, batchSize, keepProb, initObs, maxExp):
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
        self.totStepNumber=1
        self.trainSteps=0
        self.summarySteps=0
        self.maxExperience = maxExp
        
    def buildModel(self, hiddenNodes):
        "builds the neural network"
        print("Building network")
        g = tf.Graph()
        with g.as_default():
            inpt = tf.placeholder(tf.float32, shape=[None, self.features], name="inputFeatures") #any number (batchSize for training, or single for evaluation) of vectors of length self.obs
            with tf.name_scope("hiddenLayer1"):
                hidd1Weights = tf.Variable(tf.random_normal(shape=[self.features, hiddenNodes]), name="weights1")
                hidd1Biases = tf.Variable(tf.random_normal(shape=[hiddenNodes]), name="biases1") #broadcasting applies to addition to all rows
                hidd1 = tf.nn.leaky_relu(tf.add(tf.matmul(inpt, hidd1Weights), hidd1Biases), name="1stHiddenLayer")
                keepProb = tf.placeholder(tf.float32, name="keepProbability")
                hidd1DropOut = tf.nn.dropout(hidd1, keepProb)
            with tf.name_scope("hiddenLayer2"):
                hidd2Weights = tf.Variable(tf.random_normal(shape=[hiddenNodes, hiddenNodes]), name="weights2")
                hidd2Biases = tf.Variable(tf.random_normal(shape=[hiddenNodes]), name="biases2")
                hidd2 = tf.nn.leaky_relu(tf.add(tf.matmul(hidd1DropOut, hidd2Weights), hidd2Biases), name="2ndHiddenLayer")
            with tf.name_scope("outputLayer"):
                outptWeights = tf.Variable(tf.random_normal(shape=[hiddenNodes, self.actions]), name="weightsOut")
                outptBiases = tf.Variable(tf.random_normal(shape=[self.actions]), name="biasesOut")
                ########## unsure about softmax ########################
                outpt = tf.nn.softmax(tf.add(tf.matmul(hidd2, outptWeights), outptBiases), name="outputActionValues")
            with tf.name_scope("optimizer"):
                #Optimization
                target = tf.placeholder(tf.float32, shape=[None, self.actions], name="target")
                cost = tf.losses.mean_squared_error(target, outpt) #not completely sure about
                optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(cost) #implement something explicitly?
            with tf.name_scope("summaries"):
                tf.summary.histogram("histogram", hidd1Weights)
                tf.summary.histogram("histogram", hidd1Biases)
                tf.summary.histogram("histogram", hidd2Weights)
                tf.summary.histogram("histogram", hidd2Biases)
                tf.summary.histogram("histogram", outptWeights)
                tf.summary.histogram("histogram", outptBiases)
                tf.summary.scalar("cost", cost)
            saver = tf.train.Saver()
            summary = tf.summary.merge_all()
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, "sessionFiles/savedNetwork1")
            saver.save(sess, "sessionFiles/savedNetwork2")
            writer = tf.summary.FileWriter("tensorBoardFiles", graph=g)
            writer.close()
        netDict = {"graph": g, "in": inpt, "out": outpt, "keepProb": keepProb,
                   "target": target, "optimizer": optimizer, "saver": saver,
                   "summaryWriter": writer, "summary": summary}
        print("Built!")
        return netDict
        
    def writeSummary(self, sess, feedDict):
        summaryString = sess.run(self.netDict["summary"], feed_dict=feedDict)
        self.netDict["summaryWriter"].add_summary(summaryString, self.summarySteps)
        self.netDict["summaryWriter"].flush()
        self.summarySteps+=1
        
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
            self.prevAction = np.argmax(self.qApproxNet(np.reshape(observation, (1,self.features))))
        return self.prevAction
    
    def train(self, summary=False):
        "train Q approximator network using batches from experience replay"
        batch = random.sample(self.experience, self.batchSize)
        prevObs = []
        action = []
        reward = []
        nextObs = []
        for i in range(self.batchSize):
            prevObs.append(batch[i][0])
            action.append(batch[i][1])
            reward.append(batch[i][2])
            nextObs.append(batch[i][3])
        with tf.Session(graph=self.netDict["graph"]) as sess:
            self.netDict["saver"].restore(sess, "sessionFiles/savedNetwork2")
            target = self.qApproxNet(prevObs) #will give no error contribution from qvals where action wasn't taken
            discountFutureReward = self.gamma*np.max(self.qApproxNet(nextObs), 1)# 1 to get max in each row
            for i in range(self.batchSize):
                target[i,action[i]] = reward[i] + discountFutureReward[i]
            feedDict = {self.netDict["in"]: prevObs, self.netDict["keepProb"]: self.keepProb, self.netDict["target"]: target}
            sess.run(self.netDict["optimizer"], feed_dict=feedDict)
            if summary:
                self.writeSummary(sess, feedDict)
            self.netDict["saver"].save(sess, "sessionFiles/savedNetwork2")
        self.equateWeights() #set network weights equal (to trained weights) after training one according to the error provided by evaluating the other
        
    def test(self):
        "test the network"
        
    def update(self, reward, observation):
        "updates the q network approximator given result of action"
        self.experience.append([self.prevObs, self.prevAction, reward, observation])
        ############# need to do something about prevObs for first step in EVERY EPISODE ############################
        if self.totStepNumber%self.batchSize == 0:
            self.trainSteps+=1
            if self.trainSteps%10 == 0:
                self.train(True)
            else:
                self.train()
            if self.totStepNumber%((self.maxExperience+1)*self.batchSize) == 0:#####
                self.experience = self.experience[self.batchSize:-1]           #####
        self.prevObs = observation
        self.score += reward
        self.totStepNumber+=1
    
    def equateWeights(self):
        "copies the more recently trained weights to the other graph"
        with tf.Session(graph=self.netDict["graph"]) as sess:
            self.netDict["saver"].restore(sess, "sessionFiles/savedNetwork2")
            #maybe need to run session here
            self.netDict["saver"].save(sess, "sessionFiles/savedNetwork1")
    
    def reset(self):
        "resets ready for another episode run"
        self.score=0
    
    
    
    
    
    
    
    
    
    
    
    
