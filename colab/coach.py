from collections import deque
from arena import Arena
from mcts import MCTS
import numpy as np
from utils import AverageMeter, dotdict
import time
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle
from game import OthelloGame
from wrapper import DensenetWrapper
import torch
from hyper import Hyper
import gc


class Coach():
    def __init__(self, game, net, drive, args):
        self.game = game
        self.net = net
        self.preNet = net.__class__(self.game)
        self.drive = drive
        self.args = args
        self.mcts = MCTS(self.game, self.net)
        self.skipFirstSelfPlay = False
        if args.load_examples == True:
            with open('examples_log', 'rb') as f:
                content = Unpickler(f).load()
                self.index = content
            assert(f.closed)
            self.skipFirstSelfPlay = True
        else:
            self.index = 0

    def executeEp(self):
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        epStep = 0

        while True:
            epStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(epStep < self.args.tempThreshold)

            pi = self.mcts.getAction(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(
                board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)
            if r != 0:
                return [(x[0], x[2], r*((-1)**(x[1] != self.curPlayer)))for x in trainExamples]

    def learn(self):
        eps_time = AverageMeter()
        end = time.time()

        for i in range(1, self.args.numIters+1):
            print('------ ITER ' + str(i) + ' ------')
            if not self.skipFirstSelfPlay or i > 1:
                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.net)
                    iterationTrainExamples = self.executeEp()
                    for example in iterationTrainExamples:
                        self.saveTrainExamples(example)
                        self.index = (self.index+1) % self.args.maxExamples
                gc.collect()
            self.writeLog()

            eps_time.update(time.time()-end)
            end = time.time()
            print('    Examples generated finished in ' + str(eps_time.val))
            self.net.save_checkpoint('temp.pth')
            self.preNet.load_checkpoint('temp.pth')
            self.net.train()
            eps_time.update(time.time()-end)
            end = time.time()
            print('    training finished in '+str(eps_time.val))

            gc.collect()
            torch.cuda.empty_cache()
            print('    NEW VERSION VS PREVIOUS VERSION')
            preMCTS = MCTS(self.game, self.preNet)
            newMCTS = MCTS(self.game, self.net)
            arena = Arena(lambda x: np.argmax(newMCTS.getAction(x, temp=0)),
                          lambda x: np.argmax(
                preMCTS.getAction(x, temp=0)),
                self.game)
            newWins, preWins, draws = arena.playGames(
                self.args.arenaCompare)

            print(
                '     NEW/PREV WIN RATE : {}/{} ; DRAWs : {}'.format(newWins, preWins, draws))
            if preWins+newWins == 0 or float(newWins)/(preWins+newWins) < self.args.updateThreshold:
                print('    REJECTING new model')
                self.net.load_checkpoint('temp.pth')
            else:
                print('    ACCEPING new model')
                self.net.save_checkpoint(self.getCheckpointFile(i))
                self.net.save_checkpoint(
                    'best-'+str(i)+'.pth', upload=True)
                self.net.save_checkpoint('best.pth', upload=True)
            eps_time.update(time.time()-end)
            end = time.time()
            print('Arena finished in '+str(eps_time.val))
            print('Until iter '+str(i)+' totally cost '+str(eps_time.sum))

            if eps_time.sum > 41400:
                self.net.save_checkpoint('Day-2-colab.pth', upload=True)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_'+str(iteration)+'.pth'

    def saveTrainExamples(self, content):
        folder = Hyper.examples
        filename = os.path.join(folder, 'single_'+str(self.index)+'.examples')
        with open(filename, 'wb+') as f:
            Pickler(f).dump(content)
        assert(f.closed)

    def writeLog(self):
        with open('examples_log', 'wb+') as f:
            Pickler(f).dump(self.index)
        assert(f.closed)
