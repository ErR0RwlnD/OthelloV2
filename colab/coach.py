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
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False
        self.arena = False

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
                trainExamples.append([b, self.curPlayer, p, None])

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
                iterationTrainExamples = deque(
                    [], maxlen=self.args.maxlenOfQueue)
                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.net)
                    iterationTrainExamples += self.executeEp()
                self.trainExamplesHistory.append(iterationTrainExamples)
                gc.collect()

            eps_time.update(time.time()-end)
            end = time.time()
            print('    Examples generated finished in ' + str(eps_time.val))
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)
            self.saveTrainExamples(i-1)
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            self.net.save_checkpoint('temp.pth')
            self.preNet.load_checkpoint('temp.pth')
            self.net.train(trainExamples)
            eps_time.update(time.time()-end)
            end = time.time()
            print('    training finished in '+str(eps_time.val))

            gc.collect()
            if self.arena:
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
                    self.saveTrainExamples(i-1, upload=True)
                eps_time.update(time.time()-end)
                end = time.time()
                print('Arena finished in '+str(eps_time.val))
                print('Until iter '+str(i)+' totally cost '+str(eps_time.sum))
                gc.collect()
                torch.cuda.empty_cache()
                self.arena = False
            else:
                self.arena = True

            if eps_time.sum > 41400:
                self.net.save_checkpoint('Day-2-colab.pth', upload=True)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_'+str(iteration)+'.pth'

    def saveTrainExamples(self, iteration, upload=False):
        folder = Hyper.examples
        filename = os.path.join(
            folder, self.getCheckpointFile(iteration)+'.examples')
        with open(filename, 'wb+') as f:
            Pickler(f).dump(self.trainExamplesHistory)
        assert(f.closed)
        if upload:
            try:
                self.drive.uploadFile(filename)
            except Exception as e:
                print(e)

    def loadTrainExamples(self):
        folder = Hyper.examples
        examplesFile = os.path.join(folder, self.args.examples_file)

        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]\n")
            if r != "y":
                sys.exit()
        else:
            with open(examplesFile, 'rb') as f:
                self.trainExamplesHistory = Unpickler(f).load()
            assert(f.closed)
            self.skipFirstSelfPlay = True
