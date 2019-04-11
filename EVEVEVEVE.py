from arena import Arena
from mcts import MCTS
from game import OthelloGame, display
from utils import dotdict
from player import *
from wrapper import DensnetWrapper
import numpy as np

if __name__ == "__main__":
    pth = ['overTraining_0.pth', 'overTraining_1.pth', 'overTraining.limit_0.pth',
           'overTraining.limit_1.pth', 'overTraining.limit_2.pth']
    score = [0, 0, 0, 0, 0]

    for i in range(4):
        for j in range(1, 5):
            if i < j:
                print(str(i)+' vs '+str(j))
                g = OthelloGame()
                net0 = DensnetWrapper(g)
                net0.load_checkpoint(pth[i])
                mcts0 = MCTS(g, net0)
                net1 = DensnetWrapper(g)
                net1.load_checkpoint(pth[j])
                mcts1 = MCTS(g, net1)

                def foo0(x):
                    return np.argmax(mcts0.getAction(x, temp=0))

                def foo1(x):
                    return np.argmax(mcts1.getAction(x, temp=0))

                arena = Arena(foo0, foo1, g)
                win0, win1, draw = arena.playGames(20, verbose=False)
                score[i] += (win0*2+draw)
                score[j] += (win1*2+draw)
                print(score)

    # the score is [50, 78, 102, 80, 90]
