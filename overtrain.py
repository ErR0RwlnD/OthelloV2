import numpy as np
from utils import AverageMeter, dotdict
import time
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle
from game import OthelloGame
from wrapper import DensnetWrapper
import torch
from hyper import Hyper
import time


class overTrain():
    def __init__(self, game, net, args):
        self.game = game
        self.net = net
        self.args = args
        self.names = overTrain.getNames()

    def train(self):
        for iteration in range(self.args.iterations):
            iterTime = time.time()
            print('overtraining iter :: '+str(iteration))
            for name in self.names:
                nameTime = time.time()
                print('    '+str(name)+'begin training')
                trainExamples = []
                with open(name, 'rb') as f:
                    examplesDeque = Unpickler(f).load()
                    for e in examplesDeque:
                        trainExamples.append(e)
                assert(f.closed)
                shuffle(trainExamples)
                self.net.train(trainExamples)
                print('    finished '+str((time.time()-nameTime)))
            self.net.save_checkpoint(overTrain.getCheckpointFile(iteration))
            print('iter '+str(iteration)+' finished in ' +
                  str((time.time()-iterTime)))

    @staticmethod
    def getCheckpointFile(iteration):
        return 'overtraining_'+str(iteration)+'.pth'

    @staticmethod
    def getNames():
        names = []
        for iteration in range(30):
            filePath = os.path.join(
                Hyper.examples, 'overtraining.examples.iter-'+str(iteration))
            names.append(filePath)
        return names


if __name__ == "__main__":
    args = dotdict({'iterations': 2})
    game = OthelloGame()
    net = DensnetWrapper(game)
    coach = overTrain(game, net, args)
    coach.train()
