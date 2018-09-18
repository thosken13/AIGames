import tensorflow as tf
import numpy as np
import random
from collections import deque
import os
import sys

class NNAgent:
    def __init__(self, inSize, outSize, nNeuronsHidLayers=[42, 42],
                 alpha=0.001, lrDecayOverNSteps=1e6, lrMin=1e-10, gamma=0.99, linEpsDecayLen=400, minLinEpsilon=0.2,
                 expEpsDecay=0.999, minEpsilon=0.1, targetBurnIn=2000,
                 runName=None, tensorboardDir="tensorboard/", modelsDir="models/",
                 batchSize=32, minExp=1000, maxExp=int(5e5), updateTargetNet=1000,
                 seed=42, l2Reg=False, costF="hubert", activation="leakyRelu"):
        self.seed=seed
        self.l2Reg=l2Reg
        self.costF=costF
        self.activation=activation
        #RL parameters
        self.initLearnRate=alpha
        self.learnRate=alpha
        self.lrDecay=1/lrDecayOverNSteps #lrDecayOverNSteps is the number of steps over which the learning rate will decay via cosine to its minimum value
        self.lrMin=lrMin
        self.gamma=gamma
        self.linEpsDecayLen=linEpsDecayLen #decay epsilon linearly over number of episodes
        self.minLinEpsilon=minLinEpsilon #value where linear decay turns to exponential
        self.expEpsDecay=expEpsDecay #exponential decay rate after linear decay
        self.minEpsilon=minEpsilon #minimum value of epsilon
        #model and training stuff
        self.inSize = inSize #size of input space
        self.outSize = outSize #size of output space
        self.nNeuronsHidLayers = nNeuronsHidLayers #array containing a list of the number of neurons in each hidden layer
        self.tensorboardDir=tensorboardDir #where to save tensorboard files
        self.modelsDir=modelsDir
        if runName == None:
            self.runName=self.newRunName()
        else:
            self.runName=runName
        self.qNetsDict=self.buildModels()
        self.batchSize=batchSize
        self.minExp = minExp
        self.maxExperience=maxExp
        self.updateTargetNet=updateTargetNet
        self.targetBurnIn=targetBurnIn #number of steps before target network is only updated periodically
        #things that vary with time
        self.steps=0
        self.epsilon=1
        self.score=0 #last validation episode score
        self.lastEpScore=0 #score of last training episode
        self.solvesT=0 #traing episodes where score of 200 achieved
        self.solvesV=0 #validation episodes where score of 200 achieved
        self.lastEpLength=0
        self.predMaxActVal=0 #predicted maxium action value for summary
        self.landings=0
        self.hundredEpScores=deque(maxlen=100) #for calculating 100 episode average to determine when "solved"
        self.episodes=0
        self.experience=deque(maxlen=self.maxExperience)
        self.prevState=0#need to set manually at start of episode

        self.saveHyperParams()

        np.random.seed(self.seed)
        random.seed(self.seed)

    def newRunName(self):
        """Produce name for new tensorboard run file, by incrementing the run
        number from the last one in the directorry."""
        #check tensorboard files directory
        tbFiles = os.listdir(self.tensorboardDir)
        tbLastRunN=0
        for f in tbFiles:
            if f[:3] =="run" and int(f[3:]) > tbLastRunN:
                tbLastRunN = int(f[3:])
        return "run"+str(tbLastRunN+1)

    def saveHyperParams(self):
        "Save the hyper parameters used into a text file."
        directory = self.tensorboardDir+self.runName
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory+"/hyperParams.txt", "w") as textFile:
            string = "Seed: {}\n".format(str(self.seed))
            string += "Learning rate: {}\n".format(str(self.initLearnRate))
            string += "Learning rate decay (cos(steps*pi*lrdecay)): {}\n".format(str(self.lrDecay))
            string += "gamma: {}\n".format(str(self.gamma))
            string += "Number of episodes over which epsilon linearly decays: {}\n".format(str(self.linEpsDecayLen))
            string += "Min. epsilon after linear decay: {}\n".format(str(self.minLinEpsilon))
            string += "Exponential decay rate of epsilon after linear decay: {}\n".format(str(self.expEpsDecay))
            string += "Min. epsilon: {}\n".format(str(self.minEpsilon))
            string += "No. neurons in hidden layers: {}\n".format(str(self.nNeuronsHidLayers))
            string += "Batch size: {}\n".format(str(self.batchSize))
            string += "Min. experience needed before training: {}\n".format(str(self.minExp))
            string += "Max. experience in replay buffer: {}\n".format(str(self.maxExperience))
            string += "Frequency (training steps) of updating target network: {}\n".format(str(self.updateTargetNet))
            string += "Number of training steps before target network is only updated periodically: {}\n".format(str(self.targetBurnIn))
            string += "l2Reg: {}\n".format(self.l2Reg)
            string += "Cost function: {}\n".format(self.costF)
            string += "Activation function: {}\n".format(self.activation)
            textFile.write(string)

    def buildModels(self):
        "Build tensorflow model"
        tf.set_random_seed(self.seed)

        def qNet(trainable):
            "defines the q network architecture, returning the layer operations. Can set variables as trainable or not."

            l2=None
            if self.l2Reg:
                l2 = tf.nn.l2_loss

            if self.activation == "leakyRelu":
                activationF = tf.nn.leaky_relu
            elif self.activation == "relu":
                activationF = tf.nn.relu
            elif self.activation == "relu6":
                activationF = tf.nn.relu6

            inputLayer = tf.placeholder(shape=[None, self.inSize], dtype=tf.float32, name="inputLayer")
            layers = [inputLayer]
            for i, n in enumerate(self.nNeuronsHidLayers):
                hidLayer = tf.layers.dense(layers[-1], n, activation=activationF, kernel_regularizer=l2, trainable=trainable, name="hiddenlayer"+str(i+1))#try other activations
                layers.append(hidLayer)
            out = tf.layers.dense(layers[-1], self.outSize, activation=None, kernel_regularizer=l2, trainable=trainable, name="out")
            layers.append(out)
            return layers

        with tf.variable_scope("main"):
            mainQNet = qNet(True)
        with tf.name_scope("optimizer"):
            if self.costF == "hubert":
                costF = tf.losses.huber_loss
            elif self.costF == "mse":
                costF = tf.losses.mean_squared_error
            target = tf.placeholder(shape=[None, self.outSize], dtype=tf.float32, name="target")
            cost = costF(target, mainQNet[-1])
            learnRate = tf.placeholder(tf.float32, name="learningRate")
            optimizer = tf.train.AdamOptimizer(learning_rate=learnRate, name="adamOpt").minimize(cost)
        with tf.name_scope("summaries"):
            eps = tf.placeholder(tf.float32, name="epsilon")
            tf.summary.scalar("epsilon", eps)
            tf.summary.scalar("learningRate", learnRate)
            episodes = tf.placeholder(tf.float32, name="episodes")
            tf.summary.scalar("episodes", episodes)
            predMaxActVal = tf.placeholder(tf.float32, name="predMaxActVal")
            tf.summary.scalar("predMaxActVal", predMaxActVal)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
        with tf.name_scope("main"):
            score = tf.placeholder(tf.float32, name="validationScore")
            tf.summary.scalar("validationScore", score)
            epScore = tf.placeholder(tf.float32, name="trainingEpsiodeScore")
            tf.summary.scalar("trainingEpsiodeScore", epScore)
            hunderedAverage = tf.placeholder(tf.float32, name="hunderedAverage")
            tf.summary.scalar("100EpisodeAverage", hunderedAverage)
            solvesT = tf.placeholder(tf.float32, name="solvesT")
            tf.summary.scalar("solvesT", solvesT)
            solvesV = tf.placeholder(tf.float32, name="solvesV")
            tf.summary.scalar("solvesV", solvesV)
            tf.summary.scalar("cost", cost)
            reward = tf.placeholder(tf.float32, name="reward")
            tf.summary.scalar("reward", reward)
            epLength = tf.placeholder(tf.float32, name="epLength")
            tf.summary.scalar("episodeLength", epLength)
            landings = tf.placeholder(tf.float32, name="landings")
            tf.summary.scalar("landings", landings)
        with tf.name_scope("referenceActionValues"):
            referenceObsUpL = tf.placeholder(tf.float32, name="referenceObsUpL")
            tf.summary.scalar("referenceObsUpL", referenceObsUpL)
            referenceObsFallL = tf.placeholder(tf.float32, name="referenceObsFallL")
            tf.summary.scalar("referenceObsFallL", referenceObsFallL)

        #make target network and operations to copy variables when needed
        with tf.variable_scope("target"):
            targetQNet = qNet(False)
            copyOps = []
            for var in tf.trainable_variables():
                op = tf.assign(tf.get_default_graph().get_tensor_by_name("target"+var.name[4:]), var)
                copyOps.append(op)
                tf.summary.histogram("target"+var.name[4:], tf.get_default_graph().get_tensor_by_name("target"+var.name[4:]))

            copyOps = tf.group(*copyOps, name="copyOps")

        summary = tf.summary.merge_all()

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        sess= tf.Session()
        sess.run(init)
        summaryWriter = tf.summary.FileWriter(self.tensorboardDir+self.runName, graph=tf.get_default_graph())
        netDict = {"session": sess,
                   "inMainQ": mainQNet[0], "outMainQ": mainQNet[-1], "target": target,
                   "inTargetQ": targetQNet[0], "outTargetQ": targetQNet[-1], "copyOps": copyOps,
                   "score": score, "epsilon": eps, "episodes": episodes, "reward": reward,
                   "epLength": epLength, "landings": landings, "hunderedAverage": hunderedAverage,
                   "solvesT": solvesT, "solvesV": solvesV,
                   "trainScore": epScore, "predMaxActVal": predMaxActVal,
                   "refObsUpL": referenceObsUpL, "refObsFallL": referenceObsFallL,
                   "optimizer": optimizer, "learningRate": learnRate,
                   "summaryWriter": summaryWriter, "summary": summary, "saver":saver}
        return netDict

    def action(self, observations):
        """
            Choose an action either from a random policy, or using neural net.
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.outSize)
        else:
            #use main network to find best action from action values
            actionVals = self.qNetsDict["session"].run(self.qNetsDict["outMainQ"],
                             feed_dict={self.qNetsDict["inMainQ"]: np.reshape(observations, (1,self.inSize))})
            action = np.argmax(actionVals)
            #print(actionVals[0, action])
            #print(actionVals, action)
            self.predMaxActVal = actionVals[0, action]
        return action

    def getBatch(self):
        "get batch of training data, including computing the targets using double Q learning"
        batchIn = []
        actions = []
        rewards = []
        newStates = []
        dones = []
        sample = random.sample(self.experience, self.batchSize-1)
        sample.append(self.experience[-1])
        for s in sample:
            batchIn.append(s[0])
            actions.append(s[1])
            rewards.append(s[2])
            newStates.append(s[3])
            dones.append(s[4])
        #calculate targets for double Q learning (Q(s,a) <- r + g*Q_t(s', armgax(Q(s',a'))))
        #target vector(matrix of shape [batchSize, self.outSize]) so that unchosen actions values are not changed by optimization
        targetVec = self.qNetsDict["session"].run(self.qNetsDict["outMainQ"], feed_dict={self.qNetsDict["inMainQ"]: np.array(batchIn)})
        #Q vals of new state (main network)
        qNext = self.qNetsDict["session"].run(self.qNetsDict["outMainQ"], feed_dict={self.qNetsDict["inMainQ"]: np.array(newStates)})
        #print("qNext", qNext)
        optAct = np.argmax(qNext, axis=1)
        #print("optAct", optAct, "should be batchSize number of indexes for the maximum vals in qNext")
        #Q vals of new state (target network), from which the above chosen action (optAct) will be used to select action value for target
        qNextTargNet = self.qNetsDict["session"].run(self.qNetsDict["outTargetQ"], feed_dict={self.qNetsDict["inTargetQ"]: np.array(newStates)})
        #print("qNextTarg", qNextTargNet)
        #construct target values and insert into target vector for optimizer
        for i in range(self.batchSize):
            #selected q val for new(next) state
            qValNext = qNextTargNet[i, optAct[i]]
            #print("qValNext", qValNext, "not necessarily optimum action-val of target network")
            #target q value for initial state
            targetVal = rewards[i] + (not dones[i])*self.gamma*qValNext
            #print("targetVal", targetVal)
            #put target value into target vector for to be used in optimization
            targetVec[i, actions[i]] = targetVal
        return batchIn, targetVec

    def update(self, newState, action, reward, done):
        """
            Add experience to replay buffer, and train.
        """
        #pass experience to memory
        newState = np.array(newState)
        self.experience.append([self.prevState, action, reward, newState, done])
        self.prevState = newState
        if reward == 100 and done == True:
            self.landings += 1
        #train if have enough experience
        if len(self.experience) > self.minExp:
            self.steps += 1
            #to follow the results for fixed inputs
            refObs = np.array([[0, 0.25, 0, -0.05, 0, 0, 0, 0], #good landing comming
                               [0, 0.25, 0, -1, -0.4, 0, 0, 0]]) #wonky fast
            refOut = self.qNetsDict["session"].run(self.qNetsDict["outMainQ"],
                             feed_dict={self.qNetsDict["inMainQ"]: refObs})
            #compute running average
            hunderedAverage = sum(self.hundredEpScores)/len(self.hundredEpScores)
            #get batch and train
            batchIn, batchTargets = self.getBatch()
            _, summary = self.qNetsDict["session"].run([self.qNetsDict["optimizer"], self.qNetsDict["summary"]],
                                feed_dict={self.qNetsDict["inMainQ"]: batchIn, self.qNetsDict["target"]: batchTargets,
                                           self.qNetsDict["score"]: self.score, self.qNetsDict["learningRate"]: self.learnRate,
                                           self.qNetsDict["refObsUpL"]: refOut[0,0], self.qNetsDict["refObsFallL"]: refOut[1,0],
                                           self.qNetsDict["epsilon"]: self.epsilon, self.qNetsDict["episodes"]: self.episodes,
                                           self.qNetsDict["reward"]: reward, self.qNetsDict["epLength"]: self.lastEpLength,
                                           self.qNetsDict["trainScore"]: self.lastEpScore, self.qNetsDict["predMaxActVal"]: self.predMaxActVal,
                                           self.qNetsDict["landings"]: self.landings, self.qNetsDict["hunderedAverage"]: hunderedAverage,
                                           self.qNetsDict["solvesT"]: self.solvesT, self.qNetsDict["solvesV"]: self.solvesV})
            self.qNetsDict["summaryWriter"].add_summary(summary, self.steps)
            self.qNetsDict["summaryWriter"].flush()
            #if right time, update target network
            if self.steps%self.updateTargetNet == 0 or self.steps < self.targetBurnIn:
                self.qNetsDict["session"].run(self.qNetsDict["copyOps"])

    def reset(self):
        """
            Do things that need doing before the start of a new episode.
        """
        self.learnRate = 0.5*(1+np.cos(self.steps*np.pi*self.lrDecay))*(self.initLearnRate-self.lrMin) + self.lrMin
        if self.epsilon > self.minLinEpsilon:
            newEps = self.epsilon - 1/self.linEpsDecayLen
        else:
            newEps = self.epsilon*self.expEpsDecay
        self.epsilon = max([newEps, self.minEpsilon])
        self.episodes+=1

    def save(self):
        """
            Save the model.
        """
        #save session then close and remove session and other tf objects from this agent object (issues with pickling session and tf objects)
        self.qNetsDict["saver"].save(self.qNetsDict["session"], self.modelsDir+self.runName+"/model", global_step=self.steps)
        print("Saved")

    def kill(self):
        self.save()
        self.qNetsDict["session"].close()
        tf.reset_default_graph() #clean up
