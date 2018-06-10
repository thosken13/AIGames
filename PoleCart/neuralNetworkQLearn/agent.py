import tensorflow as tf
import time
import numpy as np
import os

class NNAgent:
    def __init__(self, alpha=0.01, gamma=0.99, epsilonDecay=0.99, 
                 nNeuronsHidLayers=[50,50,50]):
        self.learnRate=alpha
        self.gamma=gamma
        self.epsilon=1
        self.epsilonDecay=epsilonDecay
        self.nNeuronsHidLayers = nNeuronsHidLayers
        self.netDict=self.buildModel()
        self.steps=0
        
    def buildModel(self):
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
            #for i, l in enumerate(layers[1:]):
            #    tf.summary.histogram("layer "+str(i+1), l)
            tf.summary.scalar("score ", score)
            tf.summary.scalar("cost", cost)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
        summary = tf.summary.merge_all()
            
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        tf.set_random_seed(1234)
        self.session.run(init)
        summaryWriter = tf.summary.FileWriter("tensorboard/"+self.newTBDir(), graph=tf.get_default_graph())
                
        netDict = {"in": inputLayer, "out": layers[-1], "target": target, 
                   "score": score,
                   "optimizer": optimizer, "learningRate": learnRate, 
                   "summaryWriter": summaryWriter, "summary": summary}
        return netDict
        
    def newTBDir(self):
        "Produce name for new tensorboard run directory"
        files = os.listdir("tensorboard/")
        lastRunN=0
        for f in files:
            if int(f[-1]) > lastRunN:
                lastRunN = int(f[-1])
        return "run"+str(int(lastRunN)+1)
        
    def writeSummary(self, feedDict):
        summaryString = self.session.run(self.netDict["summary"], feed_dict=feedDict)
        self.netDict["summaryWriter"].add_summary(summaryString, self.steps)
        
        
    def test(self):
        x = np.reshape(np.array([1,2,3,4]), (1,4))
        y = np.reshape(np.array([5,6]), (1,2))
        summaryString, opt = self.session.run([self.netDict["summary"], self.netDict["optimizer"]], 
                                                feed_dict={self.netDict["in"]: x, self.netDict["target"]: y, 
                                                           self.netDict["score"]: 5, self.netDict["learningRate"]: 0.01})
        print(opt)
        self.netDict["summaryWriter"].add_summary(summaryString, self.steps)
        self.netDict["summaryWriter"].flush()
        
        
        
        
        
        
        
        
        
        
        
