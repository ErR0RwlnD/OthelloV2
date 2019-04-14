from colab.coach import Coach
from colab.wrapper import DensnetWrapper
from colab.game import OthelloGame
from colab.utils import dotdict, googleDrive

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'arenaCompare': 20,  # 4*20 indeed

    'load_model': True,
    'checkpoint_file': 'best.pth',
    'load_exmaples': False,
    'examples_file': 'checkpoint_4.pth.examples',
    'numItersForTrainExamplesHistory': 20,
})

if __name__ == "__main__":
    g = OthelloGame()
    drive = googleDrive()
    net = DensnetWrapper(g, drive)

    if args.load_model:
        net.load_checkpoint(args.checkpoint_file)

    c = Coach(g, net, drive, args)
    if args.load_exmaples:
        c.loadTrainExamples()
    c.learn()
