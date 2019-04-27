from coach import Coach
from wrapper import DensenetWrapper
from game import OthelloGame
from utils import dotdict

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxExamples': 200000,
    'arenaCompare': 50,

    'load_model': True,
    'checkpoint_file': 'best.pth',
    'load_examples': False
})

if __name__ == "__main__":
    g = OthelloGame()
    net = DensenetWrapper(g)

    if args.load_model:
        net.load_checkpoint(args.checkpoint_file)

    c = Coach(g, net, args)
    c.learn()
