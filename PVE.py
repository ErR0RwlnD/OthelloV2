from arena import Arena
from mcts import MCTS
from game import OthelloGame, display
from player import HumanOthelloPlayer
from wrapper import DensenetWrapper
import numpy as np

if __name__ == "__main__":
    g = OthelloGame()
    hp = HumanOthelloPlayer(g).play

    net = DensenetWrapper(g)
    net.load_checkpoint('bestc4.pth')
    mcts = MCTS(g, net)

    def netp(x):
        return np.argmax(mcts.getAction(x, temp=0))

    arena = Arena(netp, hp, g, display=display)
    print(arena.playGames(2, verbose=True))
