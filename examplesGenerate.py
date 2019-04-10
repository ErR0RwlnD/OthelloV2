from utils import *
from examplesGenerator import Generator
from resnet import GeneratorNet
from game import OthelloGame
from multiprocessing import Pool
from torch.cuda import empty_cache
from hyper import Hyper


def generateExamples(part):
    args = dotdict({
        'numIters': 30,
        'numEps': 100,
        'tempThreshold': 15,
        'examples_file': 'unkonwn'
    })
    g = OthelloGame()
    net = GeneratorNet(g)
    net.load_checkpoint('overTraining.pth')
    if part % Hyper.processing == 0:
        gen = Generator(g, net, args, part, verbose=True)
    else:
        gen = Generator(g, net, args, part)
    gen.generatePart()


if __name__ == "__main__":
    empty_cache()
    pool = Pool(Hyper.processing)
    for i in range(Hyper.processing):
        pool.apply_async(generateExamples, (i,))
    pool.close()
    pool.join()
