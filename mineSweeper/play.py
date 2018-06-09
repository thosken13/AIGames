import numpy as np
import game 
import CNNAgentLayers as CNNAgent

np.random.seed(0)

def play(agent, train=True):
    """
        Play game of minesweeper using AI
    """
    
    gamePlay = game.MineSweeper()
    #print(gamePlay.board)
    
    done = False
    while not done:
        boardState = gamePlay.board #current board state
        action = agent.action(boardState)
        print(action)
        reward, done = gamePlay.gameStep(action)
        #print(gamePlay.board) #new board state
        agent.score = gamePlay.explored
        if train:
            agent.update(action, gamePlay.board, reward-1, done) #reward -1 so that incentivise short game
            
print("running")
agent = CNNAgent.CNNAgent()
for i in range(2000): #play N games
    play(agent)
    agent.reset()
    print("epsilon", agent.epsilon)
