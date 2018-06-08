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
        boardState = gamePlay.board #current board state
        action = agent.action(boardState)
        print(action)
        reward, done = gamePlay.gameStep(action)
        print(gamePlay.board) #new board state
        if train:
            agent.update(action, gamePlay.board, reward, done)
            
print("running")
agent = CNNAgent.CNNAgent()
for i in range(50): #play 50 games
    play(agent)
    agent.reset()
