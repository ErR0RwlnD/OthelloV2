import os
import torch


class Hyper():

    # board
    black = 1
    white = -1

    # desnet wrapper
    dropout = 0.3
    epochs = 20
    batch_size = 1600
    cuda = torch.cuda.is_available()

    # MCTS
    sims = 40
    cpuct = 1

    # training
    checkpoints = './checkpoints'
    examples = './trainingExamples'

    # examples generator
    processing = 4
