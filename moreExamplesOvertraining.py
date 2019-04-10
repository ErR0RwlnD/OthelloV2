from utils import *
from examplesGenerator import Generator
from resnet import GeneratorNet
from game import OthelloGame
from multiprocessing import Pool
from torch.cuda import empty_cache
from hyper import Hyper
from pickle import Pickler, Unpickler
from wrapper import DensnetWrapper
import time
from random import shuffle


def generateExamples(part):
    args = dotdict({
        'numIters': 10,
        'numEps': 100,
        'tempThreshold': 15,
        'examples_file': 'unkonwn'
    })
    g = OthelloGame()
    net = GeneratorNet(g)
    net.load_checkpoint('overTraining.pth')
    if part % Hyper.processing == 0:
        gen = Generator(g, net, args, part, verbose=True, limit=True)
    else:
        gen = Generator(g, net, args, part, limit=True)
    gen.generatePart()


def getNames():
    names = []
    for part in range(Hyper.processing):
        subnames = []
        for iteration in range(10):
            filePath = os.path.join(
                Hyper.examples, 'overTrain.examples.limit.part-' + str(part)+'.iter-'+str(iteration))
            subnames.append(filePath)
        names.append(subnames)
    return names


def saveOvertrainingExamples(subnames, part):
    overtrainingExamples = []
    for name in subnames:
        with open(name, 'rb') as f:
            examples = Unpickler(f).load()
            for e in examples:
                overtrainingExamples.extend(e)
            f.close()

    filename = os.path.join(
        Hyper.examples, 'overtraining.examples.limit.part-'+str(part))
    with open(filename, 'wb+') as f:
        Pickler(f).dump(overtrainingExamples)
    f.close()


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
        return 'overtraining.limit_'+str(iteration)+'.pth'

    @staticmethod
    def getNames():
        names = []
        for iteration in range(10):
            filePath = os.path.join(
                Hyper.examples, 'overtraining.examples.limit.iter-'+str(iteration))
            names.append(filePath)
        return names


if __name__ == "__main__":
    empty_cache()
    pool = Pool(Hyper.processing)
    for i in range(Hyper.processing):
        pool.apply_async(generateExamples, (i,))
    pool.close()
    pool.join()

    names = getNames()
    for part in range(Hyper.processing):
        saveOvertrainingExamples(names[part], part)

    empty_cache()
    args = dotdict({'iterations': 3})
    game = OthelloGame()
    net = DensnetWrapper(game)
    net.load_checkpoint('overtraining_1.pth')
    coach = overTrain(game, net, args)
    coach.train()
