import tensorflow as tf
import time
import numpy as np
import random
import os

class NNAgent:
    def __init__(self, environment, tensorboardDir="tensorboard/", alpha=0.001, gamma=0.99, epsilonDecay=0.995,
                 nNeuronsHidLayers=[10,10,10], batchSize=3, minExp=100):
        #RL parameters
        self.learnRate=alpha
        self.gamma=gamma
        self.epsilonDecay=epsilonDecay
        #model and training stuff
        self.nNeuronsHidLayers = nNeuronsHidLayers
        self.tensorboardDir=tensorboardDir
        self.netDict=self.buildModel()
        self.batchSize=batchSize
        self.minExp = minExp
        #environment
        self.environment = environment
        #things that vary with time
        self.steps=0
        self.epsilon=1
        self.score=0 #last episode score
        self.episodes=0
        self.experience=[]
        self.prevState=0#need to set manually at start of episode

        np.random.seed(5)
        random.seed(4)

    def buildModel(self):
        tf.set_random_seed(1234)
        with tf.name_scope("nn"):
            inputLayer = tf.placeholder(shape=[None, 4], dtype=tf.float32) #4 observations in polecart
            layers = [inputLayer]
            for i, n in enumerate(self.nNeuronsHidLayers):
                hidLayer = tf.layers.dense(layers[-1], n, activation=tf.nn.relu, name="hiddenlayer"+str(i+1))#try other activations
                layers.append(hidLayer)
            out = tf.layers.dense(layers[-1], 2, activation=None, name="out") #2 actions in polecart
            layers.append(out)
        with tf.name_scope("optimizer"):
            target = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="target")
            cost = tf.losses.mean_squared_error(target, out) #try alternatives?
            learnRate = tf.placeholder(tf.float32, name="learningRate")
            optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(cost)
        with tf.name_scope("summaries"):
            score = tf.placeholder(tf.float32, name="score")
            tf.summary.scalar("score ", score)
            eps = tf.placeholder(tf.float32, name="epsilon")
            tf.summary.scalar("epsilon ", eps)
            episodes = tf.placeholder(tf.float32, name="episodes")
            tf.summary.scalar("episodes ", episodes)
            tf.summary.scalar("cost", cost)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)
        summaryWriter = tf.summary.FileWriter(self.tensorboardDir+self.newTBDir(), graph=tf.get_default_graph())

        netDict = {"in": inputLayer, "out": layers[-1], "target": target,
                   "score": score, "epsilon": eps, "episodes": episodes,
                   "optimizer": optimizer, "learningRate": learnRate,
                   "summaryWriter": summaryWriter, "summary": summary}
        return netDict

    def newTBDir(self):
        "Produce name for new tensorboard run directory (next run number)."
        files = os.listdir(self.tensorboardDir)
        lastRunN=0
        for f in files:
            if int(f[3:]) > lastRunN:
                lastRunN = int(f[3:])
        return "run"+str(int(lastRunN)+1)

    def action(self, observations):
        """
            Choose an action either from a random policy, or using neural net.
        """
        if np.random.random() < self.epsilon:
            action = self.environment.action_space.sample()
        else:
            actionVals = self.session.run(self.netDict["out"],
                             feed_dict={self.netDict["in"]: np.reshape(observations, (1,4))})
            action = np.argmax(actionVals)
            print(actionVals, action)
        self.steps+=1
        return action

    def getBatch(self):
        batchIn = []
        actions = []
        rewards = []
        newStates = []
        dones = []
        sample = random.sample(self.experience, self.batchSize)
        for s in sample:
            batchIn.append(s[0])
            actions.append(s[1])
            rewards.append(s[2])
            newStates.append(s[3])
            dones.append(s[4])
        #calculate targets
        targets = self.session.run(self.netDict["out"], feed_dict={self.netDict["in"]: np.array(batchIn)})
        for i in range(self.batchSize):
            futureReward = self.session.run(self.netDict["out"], feed_dict={self.netDict["in"]: np.reshape(newStates[i], (1,4))})
            targets[i, actions[i]] = rewards[i] + (not dones[i])*self.gamma*np.max(futureReward[0])
        return batchIn, targets

    def update(self, newState, action, reward, done):
        """
            Add experience to replay buffer, and train.
        """
        newState = np.array(newState)
        self.experience.append([self.prevState, action, reward, newState, done])
        self.prevState = newState
        if self.steps > self.minExp :
            batchIn, batchTargets = self.getBatch()
            _, summary = self.session.run([self.netDict["optimizer"], self.netDict["summary"]],
                                feed_dict={self.netDict["in"]: batchIn, self.netDict["target"]: batchTargets,
                                           self.netDict["score"]: self.score, self.netDict["learningRate"]: self.learnRate,
                                           self.netDict["epsilon"]: self.epsilon, self.netDict["episodes"]: self.episodes})
            self.netDict["summaryWriter"].add_summary(summary, self.steps)
            self.netDict["summaryWriter"].flush()

    def reset(self):
        """
            Do things that need doing before the start of a new episode.
        """
        self.epsilon *= self.epsilonDecay
        self.episodes+=1

    def test(self):
        x = np.reshape(np.array([1,2,3,4]), (1,4))
        y = np.reshape(np.array([5,6]), (1,2))
        summaryString, opt = self.session.run([self.netDict["summary"], self.netDict["optimizer"]],
                                                feed_dict={self.netDict["in"]: x, self.netDict["target"]: y,
                                                           self.netDict["score"]: 5, self.netDict["learningRate"]: 0.01})
        print(opt)
        self.netDict["summaryWriter"].add_summary(summaryString, self.steps)
        self.netDict["summaryWriter"].flush()

    def testNpSeed(self):
        for i in range(5):
            print(np.random.random())
