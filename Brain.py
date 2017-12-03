import numpy as np


class Brain:
    def __init__(self, Nin, Nhidd, Nout):
        self.inputL = np.zeros(Nin)
        self.Nhidd = Nhidd
        self.hiddenLayer = np.zeros(Nhidd) #hidden layer neuron values
        self.biases = np.random.randn(Nhidd)
        self.synapses1 = np.random.randn(Nin, Nhidd)#weights for synapses between input and hidden layer
                                          #first row is from first input, second from second, ...
                                          #first collumn links to first neuron in hidden layer
                                
        self.synapses2 = np.abs(np.transpose(np.random.randn(Nout, Nhidd))) #weights for synapses from hidden to output


    def activationSig(self, vals):
        "sigmoid neuron activation function"
        return 1 / (1 + np.exp(-vals))
    
    def activationRELU(self, vals):
        "rectified linear unit activation function"
        out = []
        for val in vals:
            if val >0:
                out.append(val)
            else:
                out.append(0)
        return out
    
    def calculate(self):
        "calculates output of neural net"
        self.hiddenLayer = np.dot(self.inputL, self.synapses1)+self.biases #initial values of hidden layer before activation
        self.hiddenLayer = self.activationRELU(self.hiddenLayer) #hidden layer values after applying activation function
        ######################add softmax for output#################################
        return np.dot(self.hiddenLayer, self.synapses2)/self.Nhidd 

    def input(self, obs):
        self.inputL = obs
        return
