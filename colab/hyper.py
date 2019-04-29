import os
import torch


class Hyper():

    # board
    black = 1
    white = -1

    # desnet wrapper
    dropout = 0.3
    epochs = 10
    batch_size = 2500
    cuda = torch.cuda.is_available()
    num_cpu = 2

    # MCTS
    sims = 40
    cpuct = 1

    # training
    checkpoints = './drive/checkpoints'
    examples = './examples'
