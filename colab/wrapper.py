import os
import shutil
import time
import random
import numpy as np
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from densenet import OthelloDensenet as densenet
from utils import AverageMeter, dotdict
from hyper import Hyper


class DensenetWrapper():
    def __init__(self, game, drive=None, verbose=False):
        self.net = densenet(game, dropout=0.3)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.drive = drive
        self.verbose = verbose
        if Hyper.cuda:
            self.net.cuda()

    def train(self, examples):
        optimizer = optim.Adam(self.net.parameters())
        for epoch in range(Hyper.epochs):
            if self.verbose:
                print('EPOCH :: '+str(epoch+1))
            self.net.train()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            batch_idx = 0
            while batch_idx < int(len(examples)/Hyper.batch_size):
                sample_idx = np.random.randint(
                    len(examples), size=Hyper.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_idx]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if Hyper.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(
                    ), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                out_pi, out_v = self.net(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi+l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_time.update(time.time()-end)
                if self.verbose:
                    print('    ONE batch finished in : '+str(batch_time.val))
                end = time.time()
                batch_idx += 1

            print('    EPOCH '+str(epoch) +
                  ' finished, tatolly cost '+str(batch_time.sum))
            print('    pi_loss: '+str(pi_losses.val) +
                  '; v_loss: '+str(v_losses.val))

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64))
        if Hyper.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.net.eval()
        with torch.no_grad():
            pi, v = self.net(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, filename, folder=Hyper.checkpoints, upload=False):
        filepath = os.path.join(folder, filename)
        torch.save(self.net.state_dict(), filepath)
        if upload:
            try:
                self.drive.uploadFile(filepath)
            except Exception as e:
                print(e)

    def load_checkpoint(self, filename, folder=Hyper.checkpoints, strict=True):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise('No such checkpoint')
        checkpoint = torch.load(filepath)
        self.net.load_state_dict(checkpoint, strict=strict)
