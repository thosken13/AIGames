import numpy as np
import random

np.set_printoptions(linewidth=132)

class mineSweeper:
    """
        Mine Sweeper game.
        mines: 0 = no mine
               1 = mine
        board: -9 = covered square
               non-negative number = uncovered square with that number of mines adjacent to it
               
    """
    def __init__(self, boardSize=16, nMines=40):
        self.boardSize = boardSize
        self.nMines = nMines
        self.mines = np.zeros((boardSize,boardSize)) #will hold the positions of the mines
        self.board = np.ones((boardSize,boardSize))*-9 #is what the player sees 
        self.oldBoard = np.ones((boardSize,boardSize))*-9 #is the old state of the board
        #set out mines
        posList = []
        for i in range(boardSize):
            for j in range(boardSize):
                posList.append((i,j))
        minePos = random.sample(posList, nMines)
        for i in minePos:
            self.mines[i] = 1
        #game stats
        self.nMoves = 0 #moves taken
        self.explored = 0 #fraction of (non-mine) squares uncovered
        

    def calculatePositionValue(self, position):
        """
            Calculates the number to be shown to the player
            on the uncovered square.
        """
        val = 0
        if position[0]-1 >= 0 and position[1]-1 >= 0:
            val += self.mines[position[0]-1, position[1]-1]
        if position[0]-1 >= 0:
            val += self.mines[position[0]-1, position[1]]
        if position[0]-1 >= 0 and position[1]+1 < self.boardSize:
            val += self.mines[position[0]-1, position[1]+1]
        if position[1]+1 < self.boardSize:
            val += self.mines[position[0], position[1]+1]
        if position[1]-1 < self.boardSize:
            val += self.mines[position[0], position[1]-1]
        if position[0]+1 < self.boardSize and position[1]-1 >=0:
            val += self.mines[position[0]+1, position[1]-1]
        if position[0]+1 < self.boardSize:
            val += self.mines[position[0]+1, position[1]]
        if position[0]+1 < self.boardSize and position[1]+1 < self.boardSize:
            val += self.mines[position[0]+1, position[1]+1]
        return val

    def uncoverSquare(self, position):
        """
            Add value of square at position to the board and return the value.
        """
        if self.board[position] == -9:
            self.explored += 1
        val = self.calculatePositionValue(position)
        self.board[position] = val
        return val
        
    def uncoverMultiSquare(self, position):
        """
            Uncover squares surrounding position because there are no mines.
        """
        if position[0]-1 >= 0 and position[1]-1 >= 0 and self.board[(position[0]-1, position[1]-1)] == -9:
            val = self.uncoverSquare((position[0]-1, position[1]-1))
            if val == 0:
                self.uncoverMultiSquare((position[0]-1, position[1]-1))
        if position[0]-1 >= 0 and self.board[(position[0]-1, position[1])] == -9:
            val = self.uncoverSquare((position[0]-1, position[1]))
            if val == 0:
                self.uncoverMultiSquare((position[0]-1, position[1]))
        if position[0]-1 >= 0 and position[1]+1 < self.boardSize and self.board[(position[0]-1, position[1]+1)] == -9:
            val = self.uncoverSquare((position[0]-1, position[1]+1))
            if val == 0:
                self.uncoverMultiSquare((position[0]-1, position[1]+1))
        if position[1]+1 < self.boardSize and self.board[(position[0], position[1]+1)] == -9:
            val = self.uncoverSquare((position[0], position[1]+1))
            if val == 0:
                self.uncoverMultiSquare((position[0], position[1]+1))
        if position[1]-1 < self.boardSize and self.board[(position[0], position[1]-1)] == -9:
            val = self.uncoverSquare((position[0], position[1]-1))
            if val == 0:
                self.uncoverMultiSquare((position[0], position[1]-1))
        if position[0]+1 < self.boardSize and position[1]-1 >=0 and self.board[(position[0]+1, position[1]-1)] == -9:
            val = self.uncoverSquare((position[0]+1, position[1]-1))
            if val == 0:
                self.uncoverMultiSquare((position[0]+1, position[1]-1))
        if position[0]+1 < self.boardSize and self.board[(position[0]+1, position[1])] == -9:
            val = self.uncoverSquare((position[0]+1, position[1]))
            if val == 0:
                self.uncoverMultiSquare((position[0]+1, position[1]))
        if position[0]+1 < self.boardSize and position[1]+1 < self.boardSize and self.board[(position[0]+1, position[1]+1)] == -9:
            val = self.uncoverSquare((position[0]+1, position[1]+1))
            if val == 0:
                self.uncoverMultiSquare((position[0]+1, position[1]+1))

    def numUncovered(self):
        """
            Calculate the number of squares that were uncovered from that action.
        """
        uncovered = 0
        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.oldBoard[i,j] != self.board[i,j]:
                    uncovered += 1
        self.oldBoard[:] = self.board[:]
        return uncovered

    def gameStep(self, action):
        """
            Takes an action in the game and returns the reward and whether the game has ended.
            The reward is the number of uncovered square, or -100 for a loss.
            action is the position that has been chosen.
        """
        if self.mines[action]:
            return -100, True
        
        else:
            val = self.uncoverSquare(action)
            if val == 0:
                self.uncoverMultiSquare(action)
            reward = self.numUncovered() - 1 #so that shouldn't choose uncovered square (would have 0 reward)
            return reward, False
        
        
        
        
        
    
