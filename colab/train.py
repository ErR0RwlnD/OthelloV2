from coach import Coach
from wrapper import DensenetWrapper
from game import OthelloGame
from utils import dotdict, googleDrive

args = dotdict({
    'numIters': 1000,
    'numEps': 80,
    'tempThreshold': 15,
    'updateThreshold': 0.64,
    'maxlenOfQueue': 50000,
    'arenaCompare': 50,

    'load_model': True,
    'checkpoint_file': 'best.pth',
    'load_exmaples': False,
    'examples_file': 'checkpoint_8.pth.examples',
    'numItersForTrainExamplesHistory': 20,
})

if __name__ == "__main__":
    g = OthelloGame()
    drive = googleDrive()
    net = DensenetWrapper(g, drive)

    if args.load_model:
        net.load_checkpoint(args.checkpoint_file)

    c = Coach(g, net, drive, args)
    if args.load_exmaples:
        c.loadTrainExamples()
    c.learn()
