import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        if valid[-1] == 1:
            print('no valid action!')
        else:
            for i in range(len(valid)):
                if valid[i]:
                    print(int(i/8), int(i % 8))
            while True:
                a = input()
                x, y = [int(x) for x in a.split(' ')]
                a = 8*x+y
                if valid[a]:
                    break
                else:
                    print('Invalid')
            return a
        return 64


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScoer(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
