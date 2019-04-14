from colab.board import Board
import numpy as np
from colab.hyper import Hyper


class OthelloGame():
    def __init__(self):
        self.n = 8

    def getInitBoard(self):
        b = Board()
        return np.array(b.pieces)

    def getBoardSize(self):
        return (8, 8)

    def getActionSize(self):
        return 65

    def getNextState(self, borad, player, action):
        if action == 64:
            return (borad, -player)
        b = Board()
        b.pieces = np.copy(borad)
        move = (int(action/8), action % 8)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        valids = [0]*self.getActionSize()
        b = Board()
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[8*x+y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        b = Board()
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        return player*board

    def getSymmetries(self, board, pi):
        assert(len(pi) == 65)
        pi_board = np.reshape(pi[:-1], (8, 8))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel())+[pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def getScore(self, board, player):
        b = Board()
        b.pieces = np.copy(board)
        return b.countDiff(player)


def display(board):
    n = board.shape[0]

    print(" |0|", end="")
    for y in range(1, n):
        print(str(y)+"|", end="")
    print("")
    print("------------------")
    for y in range(n):
        print(str(y)+"|", end="")
        for x in range(n):
            piece = board[y][x]
            if piece == Hyper.black:
                print("B", end="")
            elif piece == Hyper.white:
                print("W", end="")
            else:
                print("-", end="")
            if x != n-1:
                print(" ", end="")
        print("|")
    print("------------------")
