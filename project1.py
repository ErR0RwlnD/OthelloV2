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

    choise = input('该AI先先手再后手请输入1，先后手再先手请输入2')
    if int(choise) == 1:
        arena = Arena(netp, hp, g, display=display)
        print(arena.playGames(2, verbose=True))
    elif int(choise) == 2:
        arena = Arena(np, netp, g, display=display)
        print(arena.playGames(2, verbose=True))
