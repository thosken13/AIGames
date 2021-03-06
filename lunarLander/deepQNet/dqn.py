import numpy as np
import tensorflow as tf
import random

class dqn:
    def __init__(self, environment, alpha1, alpha2, gamma, epsilon, hiddenNodes, batchSize, minBatches, keepProb, initObs, maxExp, trainFreq, numTrainBatches, setNetFreq, lrSplit, targetUpdateF, meanObs, stdObs):
        self.env = environment
        self.epsilon = epsilon
        self.learnRateTarget = alpha1
        self.learnRateTrain = alpha2
        self.gamma = gamma
        self.actions = self.env.action_space.shape[0]
        self.features = self.env.observation_space.shape[0]
        self.experience = []
        self.trainVariables = []
        self.targetVariables = []
        self.batchSize = batchSize
        self.netDict = self.buildModel(hiddenNodes)
        self.keepProb = keepProb
        self.totStepNumber=1
        self.trainFreq = trainFreq
        self.numTrainBatches = numTrainBatches
        self.setNetFreq = setNetFreq
        self.learnRateSplit = lrSplit
        self.targetUpdateF = targetUpdateF
        self.minBatches = minBatches
        self.summaryFreq = 5
        self.trainSteps=0
        self.summarySteps=0
        self.maxExperience = maxExp
        self.meanObs = meanObs
        self.stdObs = stdObs
        self.prevAction = 0
        self.prevObs = self.processObs(initObs)
        self.score = 0
        self.finalScore = -999
        self.scoreList = []
        
    def buildModel(self, hiddenNodes):
        "builds the neural network"
        print("Building network")
        g = tf.Graph()
        with g.as_default():
            inpt = tf.placeholder(tf.float32, shape=[None, self.features], name="inputFeatures") #any number (batchSize for training, or single for evaluation) of vectors of length self.obs
            with tf.name_scope("hiddenLayer1"):
                hidd1Weights = tf.Variable(tf.random_normal(shape=[self.features, hiddenNodes]), name="weights1")
                hidd1Biases = tf.Variable(tf.random_normal(shape=[hiddenNodes]), name="biases1") #broadcasting applies to addition to all rows
                hidd1 = tf.nn.leaky_relu(tf.add(tf.matmul(inpt, hidd1Weights), hidd1Biases), name="1stHiddenLayer", alpha = 0.01)
                keepProb = tf.placeholder(tf.float32, name="keepProbability")
                hidd1DropOut = tf.nn.dropout(hidd1, keepProb)
            with tf.name_scope("hiddenLayer2"):
                hidd2Weights = tf.Variable(tf.random_normal(shape=[hiddenNodes, hiddenNodes]), name="weights2")
                hidd2Biases = tf.Variable(tf.random_normal(shape=[hiddenNodes]), name="biases2")
                hidd2 = tf.nn.leaky_relu(tf.add(tf.matmul(hidd1DropOut, hidd2Weights), hidd2Biases), name="2ndHiddenLayer", alpha = 0.01)
            with tf.name_scope("outputLayer"):
                outptWeights = tf.Variable(tf.random_normal(shape=[hiddenNodes, self.actions]), name="weightsOut")
                outptBiases = tf.Variable(tf.random_normal(shape=[self.actions]), name="biasesOut")
                ########## unsure about softmax ########################
                #outpt = tf.nn.softmax(tf.add(tf.matmul(hidd2, outptWeights), outptBiases), name="outputActionValues")
                outpt = tf.add(tf.matmul(hidd2, outptWeights), outptBiases, name="outputActionValues")
            with tf.name_scope("optimizer"):
                #Optimization
                target = tf.placeholder(tf.float32, shape=[None, self.actions], name="target")
                #cost = tf.losses.mean_squared_error(target, outpt) #not completely sure about
                cost = tf.losses.huber_loss(target, outpt) #could change delta (https://en.wikipedia.org/wiki/Huber_loss)
                #cost = target - outpt
                #cost = tf.Print(cost, [cost])
                learnRate = tf.placeholder(tf.float32, name="learningRate")
                optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(cost) #implement something explicitly?
            with tf.name_scope("summaries"):
                tf.summary.histogram("hidd1Weights", hidd1Weights)
                tf.summary.histogram("hidd1Biases", hidd1Biases)
                tf.summary.histogram("hidd2Weights", hidd2Weights)
                tf.summary.histogram("hidd2Biases", hidd2Biases)
                tf.summary.histogram("outptWeights", outptWeights)
                tf.summary.histogram("outptBiases", outptBiases)
                tf.summary.scalar("cost", cost)
                score = tf.placeholder(tf.float32, name="score")
                tf.summary.scalar("score", score)
            saver = tf.train.Saver()
            summary = tf.summary.merge_all()
            #tf.get_default_graph().finalize()
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            #saver.save(sess, "sessionFiles/savedNetwork1")
            saver.save(sess, "sessionFiles/savedNetwork2")
            writer = tf.summary.FileWriter("tensorBoardFiles", graph=g)
            variables = tf.trainable_variables()
            for v in variables:
                self.trainVariables.append(v.eval())
                self.targetVariables.append(v.eval())             
        netDict = {"graph": g, "in": inpt, "out": outpt, "keepProb": keepProb, "variables": variables,
                   "target": target, "optimizer": optimizer, "learningRate": learnRate, 
                   "saver": saver, "score": score, "summaryWriter": writer, "summary": summary}
        print("Built!")
        return netDict
        
    def scorer(self, score):
        self.scoreList.append(score)
        if len(self.scoreList) > 7:
            self.scoreList = self.scoreList[1:]
    
    def movingScore(self):
        return sum(self.scoreList)/len(self.scoreList)
        
    def writeSummary(self, sess, feedDict):
        summaryString = sess.run(self.netDict["summary"], feed_dict=feedDict)
        self.netDict["summaryWriter"].add_summary(summaryString, self.summarySteps)
        self.netDict["summaryWriter"].flush()
        self.summarySteps+=1
        
    def saveVars(self, trainNet=True):
        "save variable values"
        variables = tf.trainable_variables()
        for i, v in enumerate(variables):
            if trainNet:
                self.trainVariables[i] = v.eval()
            else:
                self.targetVariables[i] = v.eval()
        
    def processObs(self, observation):
        "normalise observations to aid network performance"
        return (observation - self.meanObs)/self.stdObs
        
    def chooseNet(self, trainNet=True):
        "return the desired net weights list"
        if trainNet:
            netChoice = self.trainVariables
        else:
            netChoice = self.targetVariables
        return netChoice
        
    def qApproxNet(self, observation, trainNet=True): #WAS ALWAYS USING TARGET VALS, SHOULDN'T FOR ACTION CHOICE!!
        "calculates approximation for Q values for all actions at state"
        #
        #
        #
        #SHOULD I BE LOADING A TRAINED NETWORK?!?!?!?!!?!?
        #maybe not, is variables from netChoice enough?
        #
        #
        
        with tf.Session(graph=self.netDict["graph"]) as sess:
            netChoice = self.chooseNet(trainNet)
            qVals = sess.run(self.netDict["out"], feed_dict={self.netDict["in"]: observation, self.netDict["variables"][0]: netChoice[0], self.netDict["variables"][1]: netChoice[1], self.netDict["variables"][2]: netChoice[2], self.netDict["variables"][3]: netChoice[3], self.netDict["variables"][4]: netChoice[4], self.netDict["variables"][5]: netChoice[5], self.netDict["keepProb"]: 1})
        return qVals
    
    def action(self, observation):
        "chooses an action given observation"
        if np.random.rand() < self.epsilon:
            self.prevAction = self.env.action_space.sample()
        else:
            processedObs = self.processObs(observation)
            self.prevAction = np.argmax(self.qApproxNet(np.reshape(processedObs, (1,self.features))))
        return self.prevAction
    
    def train(self, learnRate, savedNet, summary=False):
        "train Q approximator network using batches from experience replay"
        with tf.Session(graph=self.netDict["graph"]) as sess:
            self.netDict["saver"].restore(sess, "sessionFiles/savedNetwork"+str(savedNet))
            for b in range(self.numTrainBatches):
                batch = random.sample(self.experience, self.batchSize)
                prevObs = []
                action = []
                reward = []
                nextObs = []
                done = []
                for i in range(self.batchSize):
                    prevObs.append(batch[i][0])
                    action.append(batch[i][1])
                    reward.append(batch[i][2])
                    nextObs.append(batch[i][3])
                    done.append(batch[i][4])
                    
                target = self.qApproxNet(prevObs) #will give no error contribution from qvals where action wasn't taken (so use trained net)
                discountFutureReward = self.gamma*np.amax(self.qApproxNet(nextObs, trainNet=False), 1)# 1 to get max in each row (use target network because this is computing part of the target value)
                for i in range(self.batchSize):
                    if not done[i]:
                        target[i,action[i]] = reward[i] + discountFutureReward[i]
                    else:
                        target[i,action[i]] = reward[i]
                feedDict = {self.netDict["in"]: prevObs, self.netDict["keepProb"]: self.keepProb, self.netDict["target"]: target, self.netDict["score"]: self.movingScore(), self.netDict["learningRate"]: learnRate}
                #DO DICTIONARY COMPREHENSION ABOVE AND FOR OTHER FEEDdICT
                #OR TRY TUPLE AGAIN
                sess.run(self.netDict["optimizer"], feed_dict=feedDict)
            if summary:
                self.writeSummary(sess, feedDict)
            self.netDict["saver"].save(sess, "sessionFiles/savedNetwork"+str(savedNet))
            self.saveVars()
             
    def targetUpdate(self):
        "update target network from training network, scaled by targetUpdateF"
        for i, v in enumerate(self.targetVariables):
            v += self.targetUpdateF*(self.trainVariables[i] - v)
        
    def update(self, reward, observation, done):
        "updates the q network approximator given result of action"
        processedObs = self.processObs(observation)
        self.experience.append([self.prevObs, self.prevAction, reward, processedObs, done])
        if self.totStepNumber>=self.batchSize*self.minBatches and self.totStepNumber%self.trainFreq == 0:
            self.trainSteps+=1
            if self.trainSteps >= self.learnRateSplit:
                lrTarget = self.learnRateTarget
            else:
                lrTarget = self.learnRateTrain
            if self.trainSteps%self.summaryFreq == 0: #writeSummary
                self.train(self.learnRateTrain, 2, True) 
                self.targetUpdate()
                #self.train(lrTarget, 1)
            else:
                self.train(self.learnRateTrain, 2)
                self.targetUpdate()
                #self.train(lrTarget, 1)
            if self.totStepNumber%((self.maxExperience+1)*self.batchSize) == 0:#####
                self.experience = self.experience[self.batchSize:-1]           #####
        self.prevObs = processedObs
        self.score += reward
        self.totStepNumber+=1
    
    def reset(self):
        "resets ready for another episode run"
        self.score=0
    
    
    
    
    
    
    
    
    
    
    
    
