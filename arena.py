import numpy as np
import time


class Arena():
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert(self.display)
                print('Turn:', str(it), 'Player', str(curPlayer))
                self.display(board)
            action = players[curPlayer+1](
                self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(
                self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                assert valids[action] > 0
                print(action)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            print('Game over: Ture ', str(it), 'Result',
                  str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num=2, verbose=False, bo1=False, bots=False):
        if bo1:
            return self.playGame(verbose=verbose)
        else:
            maxeps = int(num)

            num = int(num/2)
            oneWon = 0
            twoWon = 0
            draws = 0
            for _ in range(num):
                gameResult = self.playGame(verbose=verbose)
                if gameResult == 1:
                    oneWon += 1
                elif gameResult == -1:
                    twoWon += 1
                else:
                    draws += 1
                if bots:
                    print(str(oneWon)+'  '+str(twoWon))

            self.player1, self.player2 = self.player2, self.player1
            for _ in range(num):
                gameResult = self.playGame(verbose=verbose)
                if gameResult == -1:
                    oneWon += 1
                elif gameResult == 1:
                    twoWon += 1
                else:
                    draws += 1
                if bots:
                    print(str(oneWon)+'  '+str(twoWon))

            return oneWon, twoWon, draws
