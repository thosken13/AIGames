import numpy as np
import game 
import CNNAgent

def play(agent, train=True):
    """
        Play game of minesweeper using AI
    """
    
    gamePlay = game.MineSweeper()
    
    done = False
    while not done:
        print("-")
        boardState = gamePlay.board
        action = agent.action(boardState)
        reward, done = gamePlay.gameStep(action)
        if train:
            agent.update()
            
print("running")
agent = CNNAgent.CNNAgent()
play(agent)
