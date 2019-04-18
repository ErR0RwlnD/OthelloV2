from arena import Arena
from mcts import MCTS
from game import OthelloGame, display
from utils import dotdict
from player import *
from wrapper import DensenetWrapper
from resnet import GeneratorNet
import numpy as np

if __name__ == "__main__":
    pth = ['best.pth', 'day1-best.pth']
    length = len(pth)
    score = list(0 for _ in range(length))

    for i in range(length-1):
        for j in range(1, length):
            if i < j:
                print(str(pth[i])+'  vs  '+str(pth[j]))
                g = OthelloGame()
                net0 = DensenetWrapper(g)
                net0.load_checkpoint(pth[i])
                mcts0 = MCTS(g, net0)
                net1 = DensenetWrapper(g)
                net1.load_checkpoint(pth[j])
                mcts1 = MCTS(g, net1)

                def foo0(x):
                    return np.argmax(mcts0.getAction(x, temp=0))

                def foo1(x):
                    return np.argmax(mcts1.getAction(x, temp=0))

                arena = Arena(foo0, foo1, g)
                win0, win1, draw = arena.playGames(
                    200, verbose=False, bots=True)
                score[i] += (win0*2+draw)
                score[j] += (win1*2+draw)
                print(score)

    # the score of 5-overtraining-pth is [50, 78, 102, 80, 90]
