import tensorflow as tf
import time
import numpy as np
import random

class CNNAgent:
    """
        An RL agent intended to learn how to play minesweeper.
        Splits the board into layers for each square value (0,...,8), unexplored and edges of map
    """
    def __init__(self, boardSize=16, learningRate=0.1, filterSize=3, nFilters=64, gamma=0.0, epsilonDecay=0.9995, minEps=0.05, minExp=100, batchSize=32):
        self.lr = learningRate
        self.gamma = gamma
        self.epsilon = 1
        self.epsilonDecay = epsilonDecay
        self.minEps = minEps
        self.boardSize = boardSize
        self.filterSize = filterSize
        self.nFilters = nFilters
        self.session = None#tf.Session(graph=self.netDict["graph"])
        self.netDict = self.buildModel()
        self.totSteps = 0
        self.minExp = minExp #minimum amount of experience before training
        self.batchSize = batchSize
        self.summarySteps = 0
        self.experience=[]
        self.prevBoard=self.splitBoardLayers(np.ones((self.boardSize, self.boardSize))*-9)
        self.score=0
        self.wins=0
        
    def buildModel(self):
        """
            Build the CNN model in tensorflow.
            Input is the minesweeper board -> convolutional layer -> output the value of clicking each square.
        """
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(1)
            with tf.name_scope("convNet"):
                inputLayer = tf.placeholder(tf.float32, shape=[None, self.boardSize+1, self.boardSize+1, 11], name="inputBoard")#boardSize+1 because have layer for wall (which is outside board)
                conv1 = tf.layers.conv2d(inputs=inputLayer, 
                                         filters=self.nFilters, 
                                         kernel_size=self.filterSize, 
                                         padding="same", 
                                         activation=tf.nn.relu,
                                         name="conv1")
                conv2 = tf.layers.conv2d(inputs=conv1, 
                                         filters=self.nFilters, 
                                         kernel_size=self.filterSize, 
                                         padding="same", 
                                         activation=tf.nn.relu,
                                         name="conv2")
                conv3 = tf.layers.conv2d(inputs=conv2, 
                                         filters=self.nFilters, 
                                         kernel_size=self.filterSize, 
                                         padding="same", 
                                         activation=tf.nn.relu,
                                         name="conv3")
                conv4 = tf.layers.conv2d(inputs=conv3, 
                                         filters=self.nFilters, 
                                         kernel_size=self.filterSize, 
                                         padding="same", 
                                         activation=tf.nn.relu,
                                         name="conv4")
                conv5 = tf.layers.conv2d(inputs=conv4, 
                                         filters=self.nFilters, 
                                         kernel_size=self.filterSize, 
                                         padding="same", 
                                         activation=tf.nn.relu,
                                         name="conv5")
                out = tf.layers.conv2d(inputs=conv5, 
                                         filters=1, 
                                         kernel_size=1, 
                                         padding="same", 
                                         activation=tf.nn.sigmoid,#negatives matter
                                         name="out")
            with tf.name_scope("optimizer"):
                target = tf.placeholder(tf.float32, shape=[None, self.boardSize+1, self.boardSize+1, 1], name="target")
                cost = tf.losses.mean_squared_error(target, out) #try alternatives?
                learnRate = tf.placeholder(tf.float32, name="learningRate")
                optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(cost)
            with tf.name_scope("summaries"):
                score = tf.placeholder(tf.float32, name="score")
                tf.summary.scalar("% Explored", score)
                nWins = tf.placeholder(tf.float32, name="wins")
                tf.summary.scalar("Wins", nWins)
                eps = tf.placeholder(tf.float32, name="epsilon")
                tf.summary.scalar("epsilon", eps)
                tf.summary.histogram("conv1", conv1)
                tf.summary.scalar("cost", cost)
                #filterWeights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0]
                #tf.summary.image("filter", tf.reshape(filterWeights, [1,self.filterSize,self.filterSize,1]))
            summary = tf.summary.merge_all()
            init = tf.global_variables_initializer()
        
        self.session = tf.Session(graph=g)
        self.session.run(init)
        summaryWriter = tf.summary.FileWriter("tensorboardFiles", graph=g)#+str(time.time()), graph=g)
        
        netDict = {"graph": g, "in": inputLayer, "out": out, "target": target, 
                   "score": score, "wins": nWins, "epsilon": eps,
                   "optimizer": optimizer, "learningRate": learnRate, 
                   "summaryWriter": summaryWriter, "summary": summary}
        return netDict
    
    def writeSummary(self, sess, feedDict):
        summaryString = sess.run(self.netDict["summary"], feed_dict=feedDict)
        self.netDict["summaryWriter"].add_summary(summaryString, self.summarySteps)
        self.netDict["summaryWriter"].flush()
        self.summarySteps+=1    
    
    def splitBoardLayers(self, board):
        """
            Split the board input into a layer for each square value (0,...8), uncovered, and edge of board
        """
        boardLayers = np.zeros((self.boardSize+1, self.boardSize+1, 11))
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if board[i,j] == -9: #set "on" for uncovered squares in the uncovered square layer
                    boardLayers[i,j,-2] = 1
                else:
                    boardLayers[i, j, int(board[i,j])] = 1 #use value of square to select correct layer
        for i in range(self.boardSize+1):
            for j in range(self.boardSize+1):
                if i==0 or j==0 or i==self.boardSize or j==self.boardSize:
                    boardLayers[i,j] = 1
        return boardLayers
        
    def action(self, board):
        """
            Take an action, either via an epsilon-greedy policy, or from the
            learnt policy.
        """
        if np.random.random() > self.epsilon:
            feedDict = {self.netDict["in"]: np.reshape(self.splitBoardLayers(board), (1, self.boardSize+1, self.boardSize+1, 11))}
            outPut = self.session.run(self.netDict["out"], feed_dict=feedDict)
            action = np.unravel_index(np.argmax(outPut[0,1:-1,1:-1,0]), (1,self.boardSize,self.boardSize,1))[1:-1]
           #UNSURE OF SHAPES ABOVE, AND EVERYWHERE REALLY
        else:
            action = tuple(np.random.randint(self.boardSize, size=2))
        self.totSteps += 1
        return action
    
    def getBatchInList(self):
        """
            Get a batch of inputs for training.
            [previous state, action, current state, reward, done]
        """
        batchList = random.sample(self.experience, self.batchSize)
        batchIn = []
        for experience in batchList:
            batchIn.append(experience[0])
        batchIn = np.reshape(np.array(batchIn), (self.batchSize, self.boardSize+1, self.boardSize+1, 11))
        #print("batchIn", batchIn, np.shape(batchIn))
        return batchIn, batchList
    
    def targetCalc(self, batchList):
        """
            Calculate the targets for the batch of inputs.
        """
        prevStates = []
        actions = []
        currentStates = []
        rewards = []
        dones = []
        for i in range(len(batchList)):
            prevStates.append([batchList[i][0]])
            actions.append([batchList[i][1]])
            currentStates.append([batchList[i][2]])
            rewards.append([batchList[i][3]])
            dones.append([batchList[i][4]])
        targets = []
        #get current prediction of action values for previous state (so values for actions not taken will not conribute to error)
        feedDict1 = {self.netDict["in"]: np.reshape(np.array(prevStates), (self.batchSize, self.boardSize+1, self.boardSize+1, 11))}
        targets = self.session.run(self.netDict["out"], feed_dict=feedDict1) 
        #print("target", target, np.shape(target))
        for i in range(len(batchList)):
            #print("currenstate dict2", np.shape(np.reshape(currentStates[i], (1, self.boardSize+1, self.boardSize+1, 11))))
            if not dones[i]:
                #set action value for action taken to the reward gained + the discounted future reward
                feedDict2 = {self.netDict["in"]: np.reshape(currentStates[i], (1, self.boardSize+1, self.boardSize+1, 11))}
                valueCurrent = self.session.run(self.netDict["out"], feed_dict=feedDict2) #get prediction of action values for new state
                futureReward = np.max(valueCurrent) #max predicted future reward
                targets[i][actions[i]] = rewards[i] + self.gamma*futureReward
            else:
                #set action value for action taken to the reward gained
                targets[i][actions[i]] = rewards[i]
        return targets
        
    def update(self, action, board, reward, done):
        """
            Train the agent.
        """
        board = self.splitBoardLayers(board)
        self.experience.append([self.prevBoard, action, board, reward, done]) #save new experience
        self.prevBoard = board
        if self.totSteps > self.minExp and self.totSteps%10 == 0: #train after building a buffer of experience
            batchIn, batchList = self.getBatchInList() #get a batch of inputs and (the same) batch list of [previous state, action, current state, reward, done]
            batchTarget = self.targetCalc(batchList)
            feedDict = {self.netDict["in"] : batchIn, self.netDict["target"] : batchTarget, self.netDict["learningRate"]: self.lr}
            self.session.run(self.netDict["optimizer"], feed_dict=feedDict)
            #summaries
            feedDict = {self.netDict["in"] : batchIn, self.netDict["target"] : batchTarget, 
                        self.netDict["score"]: self.score, self.netDict["wins"]: self.wins,
                        self.netDict["epsilon"]: self.epsilon}
            self.writeSummary(self.session, feedDict)
    
    def reset(self):
        "reset for new game"
        self.prevBoard=self.splitBoardLayers(np.ones((self.boardSize, self.boardSize))*-9)
        self.epsilon = max(self.epsilon*self.epsilonDecay, self.minEps)
        self.score=0
    
    
