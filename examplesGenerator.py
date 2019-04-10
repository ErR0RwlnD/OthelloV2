from collections import deque
from arena import Arena
from mcts import MCTS
import numpy as np
from utils import AverageMeter, dotdict
import time
import os
import sys
from pickle import Pickler, Unpickler
import torch
from hyper import Hyper


class Generator():
    def __init__(self, game, net, args, part, verbose=False, limit=False):
        self.game = game
        self.net = net
        self.args = args
        self.part = part
        self.verbose = verbose
        self.limit = limit
        self.mcts = MCTS(self.game, self.net)
        self.trainExamplesHistory = []

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

    def generatePart(self):
        if self.verbose:
            eps_time = AverageMeter()
            end = time.time()
        for i in range(1, self.args.numIters+1):
            self.trainExamplesHistory = []
            for eps in range(self.args.numEps):
                self.mcts = MCTS(self.game, self.net)
                self.trainExamplesHistory.append(self.executeEp())
            if self.verbose:
                eps_time.update(time.time()-end)
                end = time.time()
                print('Iter '+str(i)+' finished in '+str(eps_time.val) +
                      '. total cost: '+str(eps_time.sum))
            self.saveTrainExamples(i-1)

    def saveTrainExamples(self, iteration):
        folder = Hyper.examples
        if not self.limit:
            filename = os.path.join(
                folder, 'overTrain.examples.part-'+str(self.part)+'.iter-'+str(iteration))
        else:
            filename = os.path.join(
                folder, 'overTrain.examples.limit.part-'+str(self.part)+'.iter-'+str(iteration))
        with open(filename, 'wb+') as f:
            Pickler(f).dump(self.trainExamplesHistory)
        assert(f.closed)

    def loadTrainExamples(self):
        folder = Hyper.examples
        modelFile = os.path.join(folder, self.args.examples_file)
        examplesFile = modelFile+'.examples'
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            with open(examplesFile, 'rb') as f:
                self.trainExamplesHistory = Unpickler(f).load()
            assert(f.closed)
            self.skipFirstSelfPlay = True
