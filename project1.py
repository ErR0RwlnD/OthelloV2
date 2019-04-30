from arena import Arena
from mcts import MCTS
from game import OthelloGame, display
from player import HumanOthelloPlayer
from wrapper import DensenetWrapper
import numpy as np
import sys

if __name__ == "__main__":
    g = OthelloGame()
    hp = HumanOthelloPlayer(g).play

    net = DensenetWrapper(g)
    net.load_checkpoint('best.pth')
    mcts = MCTS(g, net)

    def netp(x):
        return np.argmax(mcts.getAction(x, temp=0))

    choise = input('该AI先手请输入1，后手请输入2\n')
    if str(choise) == '1':
        arena = Arena(netp, hp, g, display=display,
                      show_time=True, AI_first=True)
        print(arena.playGames(1, verbose=True))
    elif str(choise) == '2':
        arena = Arena(hp, netp, g, display=display,
                      show_time=True, AI_first=False)
        print(arena.playGames(1, verbose=True))
    else:
        print('指令无法接受')
        sys.exit(1)
