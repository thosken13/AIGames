import numpy as np
import game 
import CNNAgent

def play(agent, train=True):
    """
        Play game of minesweeper using AI
    """
    
    gamePlay = game.MineSweeper()
    print(gamePlay.board)
    
    done = False
    while not done:
        boardState = gamePlay.board
        action = agent.action(boardState)
        print(action)
        reward, done = gamePlay.gameStep(action)
        print(gamePlay.board)
        if train:
            agent.update()
            
print("running")
agent = CNNAgent.CNNAgent()
play(agent)
