import numpy as np


class Brain:
    def __init__(self, Nin, Nhidd):
        self.inputL = np.zeros(Nin)
        self.Nhidd = Nhidd
        self.hiddenLayer = np.zeros(Nhidd) #hidden layer neuron values
        self.biases = np.random.randn(Nhidd)
        self.synapses1 = np.random.randn(2, Nhidd)#weights for synapses between input and hidden layer
                                          #first row is from first input, second from second
                                          #first collumn links to first neuron in hidden layer
                                
        self.synapses2 = np.abs(np.transpose(np.random.randn(Nhidd))) #weights for synapses from hidden to output


    def activationSig(self, vals):
        "sigmoid neuron activation function"
        return 1 / (1 + np.exp(-vals))
    
    def activationRELU(self, val):
        "rectified linear unit activation function"
        if val >0:
            return val
        else:
            return 0
    
    def calculate(self):
        "calculates output of neural net"
        self.hiddenLayer = np.dot(self.inputL, self.synapses1)+self.biases #initial values of hidden layer before activation
        self.hiddenLayer = self.activation(self.hiddenLayer) #hidden layer values after applying activation function
        return np.dot(self.hiddenLayer, self.synapses2)/Nhidd

    def input(self, obs):
        self.inputL = obs
        return
