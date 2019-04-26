import os
import time
import numpy as np
import gc
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from densenet import OthelloDensenet as densenet
from utils import AverageMeter, dotdict
from hyper import Hyper
from pickle import Unpickler


class _ExamplesDataset(Dataset):
    def __init__(self):
        self.examples = os.listdir(Hyper.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        examplePath = os.path.join(Hyper.examples, self.examples[idx])
        with open(examplePath, 'rb') as f:
            example = Unpickler(f).load()
        assert(f.closed)
        return [torch.FloatTensor(np.array(example[0]).astype(np.float64)),
                torch.FloatTensor(np.array(example[1]).astype(np.float64)),
                torch.FloatTensor(np.array(example[2]).astype(np.float64))]


class DensenetWrapper():
    def __init__(self, game, drive=None, verbose=False):
        self.net = densenet(game, dropout=0.3)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.drive = drive
        self.verbose = verbose
        if Hyper.cuda:
            self.net.cuda()

    def train(self):
        optimizer = optim.Adam(self.net.parameters())
        dataset = _ExamplesDataset()
        dataloader = DataLoader(dataset, batch_size=Hyper.batch_size,
                                shuffle=True, num_workers=Hyper.num_cpu, drop_last=True)
        for epoch in range(Hyper.epochs):
            if self.verbose:
                print('EPOCH :: '+str(epoch+1))
            self.net.train()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            for idx, examples in enumerate(dataloader):
                boards, target_pis, target_vs = examples
                if Hyper.cuda:
                    boards = boards.cuda()
                    target_pis = target_pis.cuda()
                    target_vs = target_vs.cuda()

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
        gc.collect()
        filepath = os.path.join(folder, filename)
        torch.save(self.net.state_dict(), filepath)
        if upload:
            try:
                self.drive.uploadFile(filepath)
            except Exception as e:
                print(e)
        gc.collect()

    def load_checkpoint(self, filename, folder=Hyper.checkpoints, strict=True):
        gc.collect()
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise('No such checkpoint')
        checkpoint = torch.load(filepath)
        self.net.load_state_dict(checkpoint, strict=strict)
        gc.collect()
