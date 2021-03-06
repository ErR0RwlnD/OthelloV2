from coach import Coach
from wrapper import DensenetWrapper
from game import OthelloGame
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 150000,
    'arenaCompare': 20,  # 4*20 indeed

    'load_model': True,
    'checkpoint_file': 'Day-5.pth',
    'load_examples': True,
    'examples_file': 'checkpoint_4.pth.examples',
    'numItersForTrainExamplesHistory': 20,
})

if __name__ == "__main__":
    g = OthelloGame()
    net = DensenetWrapper(g)
    if args.load_model:
        net.load_checkpoint(args.checkpoint_file)

    c = Coach(g, net, args)
    if args.load_examples:
        c.loadTrainExamples()
    c.learn()
