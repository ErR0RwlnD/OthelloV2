from arena import Arena
from mcts import MCTS
from game import OthelloGame, display
from utils import dotdict
from player import *
from wrapper import DensnetWrapper
import numpy as np

if __name__ == "__main__":
    g = OthelloGame()
    hp = HumanOthelloPlayer(g).play

    net = DensnetWrapper(g)
    net.load_checkpoint('overTraining.limit_2.pth')
    args = dotdict({'numMCTSSims': 500, 'cpuct': 1.0})
    mcts = MCTS(g, net)

    def netp(x):
        return np.argmax(mcts.getAction(x, temp=0))

    arena = Arena(netp, hp, g, display=display)
    print(arena.playGames(2, verbose=True))
