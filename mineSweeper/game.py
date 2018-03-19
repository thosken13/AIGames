import numpy as np

class mineSweeper:
    """
        Mine Sweeper game.
        
    """
    def __init__(self, nMines=10):
        self.mines = np.zeros((15,15)) #will hold the positions of the mines
        self.board = np.ones((15,15))*-9 #-9 denotes covered square
        
        minePos = np.random.randint(0, 1, size=nMines,)
        

    def calculatePositionValue(self, position):
        """
            Calculates the number to be shown to the player
            on the uncovered square.
        """

    def gameStep(self, action):
        """
            Take an action in the game.
        """
    
