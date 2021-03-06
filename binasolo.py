from arena import Arena
from mcts import MCTS
from game import OthelloGame, display
from wrapper import DensenetWrapper
import numpy as np

if __name__ == "__main__":
    g = OthelloGame()

    netLocal = DensenetWrapper(g)
    netLocal.load_checkpoint('bestc4.pth')
    mctsLocal = MCTS(g, netLocal)

    netColab = DensenetWrapper(g)
    netColab.load_checkpoint('bestc4.pth')
    mctsColab = MCTS(g, netColab)

    def netL(x):
        return np.argmax(mctsLocal.getAction(x, temp=0))

    def netC(x):
        return np.argmax(mctsColab.getAction(x, temp=0))

    arena = Arena(netL, netC, g, display=display)
    print(arena.playGames(50, verbose=True))
