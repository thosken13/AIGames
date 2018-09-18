import tensorflow as tf
import time
import numpy as np
import random
import os
import sys
import pickle

class NNAgent:
    def __init__(self, environment, tensorboardDir="tensorboard/", modelsDir="models/", agentsDir="agents/",
                 alpha=0.001, gamma=0.99, epsilonDecay=0.99, minEpsilon=0.1,
                 nNeuronsHidLayers=[10,10,10], batchSize=6, minExp=100):
        #RL parameters
        self.learnRate=alpha
        self.gamma=gamma
        self.epsilonDecay=epsilonDecay
        self.minEpsilon=minEpsilon
        #model and training stuff
        self.nNeuronsHidLayers = nNeuronsHidLayers
        self.tensorboardDir=tensorboardDir
        self.modelsDir=modelsDir
        self.agentsDir=agentsDir
        self.runNum=self.newRunNum()
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
            inputLayer = tf.placeholder(shape=[None, 4], dtype=tf.float32, name="inputLayer") #4 observations in polecart
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
            optimizer = tf.train.AdamOptimizer(learning_rate=learnRate, name="adamOpt").minimize(cost)
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
        with tf.name_scope("referenceActionValues"):
            referenceObsUpL = tf.placeholder(tf.float32, name="referenceObsUpL")
            tf.summary.scalar("referenceObsUpL", referenceObsUpL)
            referenceObsFallL = tf.placeholder(tf.float32, name="referenceObsFallL")
            tf.summary.scalar("referenceObsFallL", referenceObsFallL)
            referenceObsRiseL = tf.placeholder(tf.float32, name="referenceObsRiseL")
            tf.summary.scalar("referenceObsRiseL", referenceObsRiseL)
            referenceObsUpR = tf.placeholder(tf.float32, name="referenceObsUpR")
            tf.summary.scalar("referenceObsUpR", referenceObsUpR)
            referenceObsFallR = tf.placeholder(tf.float32, name="referenceObsFallR")
            tf.summary.scalar("referenceObsFallR", referenceObsFallR)
            referenceObsRiseR = tf.placeholder(tf.float32, name="referenceObsRiseR")
            tf.summary.scalar("referenceObsRiseR", referenceObsRiseR)
        summary = tf.summary.merge_all()

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)
        summaryWriter = tf.summary.FileWriter(self.tensorboardDir+self.runNum, graph=tf.get_default_graph())

        netDict = {"in": inputLayer, "out": layers[-1], "target": target,
                   "score": score, "epsilon": eps, "episodes": episodes,
                   "refObsUpL": referenceObsUpL, "refObsFallL": referenceObsFallL,
                   "refObsRiseL": referenceObsRiseL, "refObsUpR": referenceObsUpR,
                   "refObsFallR": referenceObsFallR, "refObsRiseR": referenceObsRiseR,
                   "optimizer": optimizer, "learningRate": learnRate,
                   "summaryWriter": summaryWriter, "summary": summary,
                   "saver": saver}
        return netDict

    def newRunNum(self):
        "Produce name for new tensorboard run directory and model directory(next run number)."
        #check tensorboard files directory
        tbFiles = os.listdir(self.tensorboardDir)
        tbLastRunN=0
        for f in tbFiles:
            if int(f[3:]) > tbLastRunN:
                tbLastRunN = int(f[3:])
        #check model files directory
        modelFiles = os.listdir(self.modelsDir)
        modelLastRunN=0
        for f in modelFiles:
            if int(f[3:]) > modelLastRunN:
                modelLastRunN = int(f[3:])
        #check agents files directory
        agentFiles = os.listdir(self.agentsDir)
        agentLastRunN=0
        for f in agentFiles:
            if int(f[3:]) > agentLastRunN:
                agentLastRunN = int(f[3:])
        #check have same run number from tensorboard and models
        if tbLastRunN != modelLastRunN:
            sys.exit("""New run number does not match between model files and tensorbaord files.
                        Model (tensoboard) files may have been deleted while the associated tensorbaord (model) file was not.""")
        if modelLastRunN != agentLastRunN:
            print(modelLastRunN, agentLastRunN)
            sys.exit("""New run number does not match between model files and agent files.
                        Model (agent) files may have been deleted while the associated agent (model) file was not.""")
        return "run"+str(modelLastRunN+1)

    def save(self):
        """
            Save the model and the whole agent. Want to save the agent as well as model because of hyper parameters and scores that are kept in the object.
        """
        #save session then close and remove session and other tf objects from this agent object (issues with pickling session and tf objects)
        self.netDict["saver"].save(self.session, self.modelsDir+self.runNum+"/model", global_step=self.steps)
        self.session.close()
        self.session=None
        self.netDict={}
        pickleOut = open(self.agentsDir+self.runNum ,"wb")
        pickle.dump(self, pickleOut)
        pickleOut.close()
        print("Saved")

    def restore(self):
        """
            Restore the model, the agent must be restored outside of the object, from its pickle (before restoring the model).
        """
        tf.reset_default_graph() #clean up
        self.session = tf.Session()
        latest_checkpoint = tf.train.latest_checkpoint(self.modelsDir+self.runNum)
        # Load latest checkpoint Graph via import_meta_graph:
        #   - construct protocol buffer from file content
        #   - add all nodes to current graph and recreate collections
        #   - return Saver
        saver = tf.train.import_meta_graph(latest_checkpoint + '.meta') #returns the saver object
        #restore model into session
        saver.restore(self.session, latest_checkpoint)
        graph = tf.get_default_graph()
        #restore netDict manually because had to be deleted before pickling the object
        self.netDict = {}
        self.netDict["in"] = graph.get_tensor_by_name("nn/inputLayer:0")
        self.netDict["out"] = graph.get_tensor_by_name("nn/out/BiasAdd:0")
        self.netDict["target"] = graph.get_tensor_by_name("optimizer/target:0")
        self.netDict["score"] = graph.get_tensor_by_name("summaries/score:0")
        self.netDict["refObsUpL"] = graph.get_tensor_by_name("referenceActionValues/referenceObsUpL:0")
        self.netDict["refObsFallL"] = graph.get_tensor_by_name("referenceActionValues/referenceObsFallL:0")
        self.netDict["refObsRiseL"] = graph.get_tensor_by_name("referenceActionValues/referenceObsRiseL:0")
        self.netDict["refObsUpR"] = graph.get_tensor_by_name("referenceActionValues/referenceObsUpR:0")
        self.netDict["refObsFallR"] = graph.get_tensor_by_name("referenceActionValues/referenceObsFallR:0")
        self.netDict["refObsRiseR"] = graph.get_tensor_by_name("referenceActionValues/referenceObsRiseR:0")
        self.netDict["epsilon"] = graph.get_tensor_by_name("summaries/epsilon:0")
        self.netDict["episodes"] = graph.get_tensor_by_name("summaries/episodes:0")
        self.netDict["optimizer"] = graph.get_operation_by_name("optimizer/adamOpt")
        self.netDict["learningRate"] = graph.get_tensor_by_name("optimizer/learningRate:0")
        self.netDict["summaryWriter"] = tf.summary.FileWriter(self.tensorboardDir+self.runNum)
        self.netDict["summary"] = graph.get_tensor_by_name("Merge/MergeSummary:0")#tf.summary.merge_all()
        self.netDict["saver"] = saver
        print("Restored")

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
            #print(actionVals, action)
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
        #pass experience to memory
        newState = np.array(newState)
        self.experience.append([self.prevState, action, reward, newState, done])
        self.prevState = newState
        if self.steps > self.minExp :
            #to follow the results for fixed inputs
            refObs = np.array([[0, 0, 0, 0],
                               [0, 0, 0.05, 1],
                               [0, 0, 0.05, -1]])
            refOut = self.session.run(self.netDict["out"],
                             feed_dict={self.netDict["in"]: refObs})
            #get batch and train
            batchIn, batchTargets = self.getBatch()
            _, summary = self.session.run([self.netDict["optimizer"], self.netDict["summary"]],
                                feed_dict={self.netDict["in"]: batchIn, self.netDict["target"]: batchTargets,
                                           self.netDict["score"]: self.score, self.netDict["learningRate"]: self.learnRate,
                                           self.netDict["refObsUpL"]: refOut[0,0], self.netDict["refObsFallL"]: refOut[1,0], self.netDict["refObsRiseL"]: refOut[2,0],
                                           self.netDict["refObsUpR"]: refOut[0,1], self.netDict["refObsFallR"]: refOut[1,1], self.netDict["refObsRiseR"]: refOut[2,1],
                                           self.netDict["epsilon"]: self.epsilon, self.netDict["episodes"]: self.episodes})
            self.netDict["summaryWriter"].add_summary(summary, self.steps)
            self.netDict["summaryWriter"].flush()

    def reset(self):
        """
            Do things that need doing before the start of a new episode.
        """
        self.epsilon = max([self.epsilon*self.epsilonDecay, self.minEpsilon])
        self.episodes+=1

    def kill(self):
        self.save() #session is closed in save
        tf.reset_default_graph() #clean up
        #self.reset_default_graph()

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
