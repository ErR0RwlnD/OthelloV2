from OthelloV2.arena import Arena
from OthelloV2.mcts import MCTS
from OthelloV2.game import OthelloGame, display
from OthelloV2.player import HumanOthelloPlayer
from OthelloV2.wrapper import DensenetWrapper
import numpy as np
import sys

if __name__ == "__main__":
    g = OthelloGame()
    hp = HumanOthelloPlayer(g).play

    net = DensenetWrapper(g)
    net.load_checkpoint('bestc5.pth')
    mcts = MCTS(g, net)

    def netp(x):
        return np.argmax(mcts.getAction(x, temp=0))

    choise = input('该AI先手请输入1，后手请输入2\n')
    if str(choise) == '1':
        arena = Arena(netp, hp, g, display=display,
                      show_time=True, AI_first=True)
        print(arena.playGames(1, verbose=True, bo1=True))
    elif str(choise) == '2':
        arena = Arena(hp, netp, g, display=display,
                      show_time=True, AI_first=False)
        print(arena.playGames(1, verbose=True, bo1=True))
    else:
        print('指令无法接受')
        sys.exit(1)
