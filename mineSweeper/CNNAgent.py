import tensorflow as tf
import time
import numpy as np
class CNNAgent:
    """
        An RL agent intended to learn how to play minesweeper.
    """
    def __init__(self, boardSize=16, learningRate=0.01, filterSize=[3, 3], gamma=0, epsilonDecay=0.996, minEps=0.05):
        self.lr = learningRate
        self.gamma = gamma
        self.epsilon = 1
        self.epsilonDecay = epsilonDecay
        self.minEps = minEps
        self.boardSize = boardSize
        self.filterSize = filterSize
        self.session = None#tf.Session(graph=self.netDict["graph"])
        self.netDict = self.buildModel()
        self.totSteps = 0
        self.summarySteps = 0
        
    def buildModel(self):
        """
            Build the CNN model in tensorflow.
            Input is the minesweeper board -> convolutional layer -> output the value of clicking each square.
        """
        g = tf.Graph()
        with g.as_default():
            with tf.name_scope("convNet"):
                inputLayer = tf.placeholder(tf.float32, shape=[None, self.boardSize, self.boardSize, 1], name="inputBoard")
                convOut = tf.layers.conv2d(inputs=inputLayer, 
                                         filters=1, 
                                         kernel_size=self.filterSize, 
                                         padding="same", 
                                         activation=tf.nn.sigmoid)
            with tf.name_scope("optimizer"):
                target = tf.placeholder(tf.float32, shape=[None, self.boardSize, self.boardSize, 1], name="target")
                cost = tf.losses.mean_squared_error(target, convOut) #try alternatives?
                learnRate = tf.placeholder(tf.float32, name="learningRate")
                optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(cost)
            with tf.name_scope("summaries"):
                tf.summary.histogram("convOut", convOut)
                tf.summary.scalar("cost", cost)
            summary = tf.summary.merge_all()
            init = tf.global_variables_initializer()
        #with tf.Session(graph=g) as sess:
        #    sess.run(init)
        #    summaryWriter = tf.summary.FileWriter("tensorboardFiles"+str(time.time()), graph=g)
        
        self.session = tf.Session(graph=g)
        self.session.run(init)
        summaryWriter = tf.summary.FileWriter("tensorboardFiles"+str(time.time()), graph=g)
        
        netDict = {"graph": g, "in": inputLayer, "out": convOut, "target": target, 
                   "cost": cost,
                   "optimizer": optimizer, "learningRate": learnRate, 
                   "summaryWriter": summaryWriter, "summary": summary}
        return netDict
    
    
    def writeSummary(self, sess, feedDict):
        summaryString = sess.run(self.netDict["summary"], feed_dict=feedDict)
        self.netDict["summaryWriter"].add_summary(summaryString, self.summarySteps)
        self.netDict["summaryWriter"].flush()
        self.summarySteps+=1    
    
    
    def action(self, board):
        if np.random.random() > self.epsilon:
            feedDict = {self.netDict["in"]: np.reshape(board, (-1, self.boardSize, self.boardSize, 1))}
            outPut = self.session.run(self.netDict["out"], feed_dict=feedDict)
            action = np.unravel_index(np.argmax(outPut), outPut.shape)[1:-1]
        else:
            action = tuple(np.random.randint(self.boardSize, size=2))
        self.totSteps += 1
        self.epsilon = max(self.epsilon*self.epsilonDecay, self.minEps)
        return action
    
    def update(self):
        pass
        """
        feedDict = {self.netDict["convOut"] : , self.netDict["cost"] : }
        self.writeSummary(self.session, feedDict)"""
    
    
    
