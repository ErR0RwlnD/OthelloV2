from coach import Coach
from wrapper import DensenetWrapper
from game import OthelloGame
from utils import dotdict, googleDrive

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.62,
    'maxExamples': 150000,
    'arenaCompare': 50,

    'load_model': True,
    'checkpoint_file': 'best-2.pth',
    'load_examples': True
})

if __name__ == "__main__":
    g = OthelloGame()
    drive = googleDrive()
    net = DensenetWrapper(g, drive)

    if args.load_model:
        net.load_checkpoint(args.checkpoint_file)

    c = Coach(g, net, drive, args)
    c.learn()
